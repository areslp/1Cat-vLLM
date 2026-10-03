__launch_bounds__(kGroupedSparsePlannerThreads, 1) void grouped_sparse_page4_plan_kernel(
    const int* __restrict__ logical_indices,
    const int* __restrict__ request_block_table,
    const int* __restrict__ token_to_req,
    const int64_t* __restrict__ query_positions,
    const int* __restrict__ sequence_lengths, int* __restrict__ output_blocks,
    uint32_t* __restrict__ output_masks, int* __restrict__ output_seq_lens,
    const int selection_width, const int64_t logical_indices_stride,
    const int64_t request_block_table_stride, const int num_requests,
    const int block_table_width, const int output_width, const int page_size,
    const int physical_page_stride, const int num_cache_blocks) {
  const int group_idx = blockIdx.x;
  const int tid = threadIdx.x;
  extern __shared__ unsigned long long hash_table[];
  auto* logical_owners =
      reinterpret_cast<uint32_t*>(hash_table + kGroupedSparseHashCapacity);
  for (int slot = tid; slot < kGroupedSparseHashCapacity;
       slot += kGroupedSparsePlannerThreads) {
    hash_table[slot] = kGroupedSparseEmptyEntry;
    logical_owners[slot] = UINT_MAX;
  }
  __syncthreads();

  const int full_page4_count = selection_width / 4;
  for (int selected_page = tid; selected_page < full_page4_count;
       selected_page += kGroupedSparsePlannerThreads) {
#pragma unroll
    for (int query = 0; query < kGroupedSparseQueries; ++query) {
      const int row = group_idx * kGroupedSparseQueries + query;
      const int request_idx = __ldg(token_to_req + row);
      if (request_idx < 0 || request_idx >= num_requests) {
        continue;
      }
      const int sequence_length = __ldg(sequence_lengths + request_idx);
      const int64_t query_visible_tokens = __ldg(query_positions + row) + 1;
      const int visible_tokens =
          query_visible_tokens <= 0
              ? 0
              : (query_visible_tokens < sequence_length
                     ? static_cast<int>(query_visible_tokens)
                     : max(sequence_length, 0));
      const int row_complete_page4_count =
          min(min(visible_tokens / 4, sequence_length / 4), full_page4_count);
      if (selected_page >= row_complete_page4_count) {
        continue;
      }
      const int* selected = logical_indices +
                            static_cast<int64_t>(row) * logical_indices_stride +
                            selected_page * 4;
      const int first_token = __ldg(selected);
      if (first_token < 0) {
        continue;
      }
      const bool full_page4 = __ldg(selected + 1) == first_token + 1 &&
                              __ldg(selected + 2) == first_token + 2 &&
                              __ldg(selected + 3) == first_token + 3 &&
                              (first_token & 3) == 0 &&
                              first_token + 3 < sequence_length;
      if (full_page4) {
        const int physical_microblock = grouped_sparse_physical_microblock(
            first_token, request_idx, request_block_table,
            request_block_table_stride, block_table_width, page_size,
            physical_page_stride, num_cache_blocks);
        grouped_sparse_hash_insert(hash_table, logical_owners,
                                   physical_microblock, 0xFu << (query * 4),
                                   query, first_token);
      } else {
#pragma unroll
        for (int token_offset = 0; token_offset < 4; ++token_offset) {
          const int token = __ldg(selected + token_offset);
          if (token >= 0 && token < sequence_length) {
            const int physical_microblock = grouped_sparse_physical_microblock(
                token, request_idx, request_block_table,
                request_block_table_stride, block_table_width, page_size,
                physical_page_stride, num_cache_blocks);
            grouped_sparse_hash_insert(
                hash_table, logical_owners, physical_microblock,
                1u << (query * 4 + (token & 3)), query, token);
          }
        }
      }
    }
  }
  if (tid < kGroupedSparseQueries) {
    const int query = tid;
    const int row = group_idx * kGroupedSparseQueries + query;
    const int request_idx = __ldg(token_to_req + row);
    if (request_idx >= 0 && request_idx < num_requests) {
      const int sequence_length = __ldg(sequence_lengths + request_idx);
      const int64_t query_visible_tokens = __ldg(query_positions + row) + 1;
      const int visible_tokens =
          query_visible_tokens <= 0
              ? 0
              : (query_visible_tokens < sequence_length
                     ? static_cast<int>(query_visible_tokens)
                     : max(sequence_length, 0));
      const int complete_page4_count =
          min(min(visible_tokens / 4, sequence_length / 4), full_page4_count);
      const int tail_count = visible_tokens & 3;
      const int tail_index = complete_page4_count * 4;
      const int selected_tail_token =
          tail_index < selection_width
              ? __ldg(logical_indices +
                      static_cast<int64_t>(row) * logical_indices_stride +
                      tail_index)
              : -1;
      const int expected_tail_token = (visible_tokens / 4) * 4;
      if (tail_count > 0 && selected_tail_token == expected_tail_token &&
          selected_tail_token < sequence_length) {
        const int physical_microblock = grouped_sparse_physical_microblock(
            selected_tail_token, request_idx, request_block_table,
            request_block_table_stride, block_table_width, page_size,
            physical_page_stride, num_cache_blocks);
        const uint32_t tail_mask = ((1u << tail_count) - 1) << (query * 4);
        grouped_sparse_hash_insert(hash_table, logical_owners,
                                   physical_microblock, tail_mask, query,
                                   selected_tail_token);
      }
    }
  }
  __syncthreads();

  // Keep the existing physical-page union and masks, but never let physical
  // hash slots determine the attention reduction order. Category is primary
  // to preserve active-tile packing; the logical owner orders each category.
  unsigned long long entries[kGroupedSparseItemsPerThread];
  unsigned long long sort_keys[kGroupedSparseItemsPerThread];
#pragma unroll
  for (int item = 0; item < kGroupedSparseItemsPerThread; ++item) {
    const int slot = tid * kGroupedSparseItemsPerThread + item;
    const unsigned long long entry = hash_table[slot];
    entries[item] = entry;
    const int category =
        grouped_sparse_active_m_tiles(static_cast<uint32_t>(entry >> 32));
    sort_keys[item] = static_cast<uint32_t>(entry) == 0xffffffffu
                          ? ULLONG_MAX
                          : (static_cast<unsigned long long>(category) << 32) |
                                logical_owners[slot];
  }
  __syncthreads();
  auto& sort_storage =
      *reinterpret_cast<GroupedSparseSort::TempStorage*>(hash_table);
  // Three category bits, 32 owner bits, and one bit separating empty slots.
  GroupedSparseSort(sort_storage).Sort(sort_keys, entries, 0, 36);
  __syncthreads();
#pragma unroll
  for (int item = 0; item < kGroupedSparseItemsPerThread; ++item) {
    hash_table[tid * kGroupedSparseItemsPerThread + item] = entries[item];
  }

  // Sorted keys have contiguous real categories 1..7 and an empty suffix.
  // Seven threads find the first slot whose category is at least 2..8; the
  // first category begins at zero. This replaces category atomics and the
  // repeated ballot/prefix scan while preserving each category's sorted rank.
  auto* category_bounds = reinterpret_cast<int*>(logical_owners);
  int* category_counts = category_bounds + 9;
  int* category_offsets = category_counts + 8;
  __syncthreads();
  if (tid < 7) {
    const int target_category = tid + 2;
    int first = 0;
    int last = kGroupedSparseHashCapacity;
    while (first < last) {
      const int middle = first + (last - first) / 2;
      const unsigned long long entry = hash_table[middle];
      const int category =
          static_cast<uint32_t>(entry) == 0xffffffffu
              ? 8
              : grouped_sparse_active_m_tiles(
                    static_cast<uint32_t>(entry >> 32));
      if (category < target_category) {
        first = middle + 1;
      } else {
        last = middle;
      }
    }
    category_bounds[target_category] = first;
    if (tid == 0) {
      category_bounds[1] = 0;
    }
  }
  __syncthreads();
  if (tid < 7) {
    const int category = tid + 1;
    category_counts[category] =
        category_bounds[category + 1] - category_bounds[category];
  }
  if (tid == 0) {
    category_counts[0] = 0;
  }
  __syncthreads();
  if (tid == 0) {
    int padded_offset = 0;
#pragma unroll
    for (int category = 1; category < 8; ++category) {
      category_offsets[category] = padded_offset;
      padded_offset += (category_counts[category] + 7) & ~7;
    }
    category_offsets[0] = padded_offset;
  }
  __syncthreads();

#pragma unroll
  for (int item = 0; item < kGroupedSparseItemsPerThread; ++item) {
    const int sorted_index = tid * kGroupedSparseItemsPerThread + item;
    const unsigned long long entry = hash_table[sorted_index];
    if (static_cast<uint32_t>(entry) != 0xffffffffu) {
      const int category =
          grouped_sparse_active_m_tiles(static_cast<uint32_t>(entry >> 32));
      const int output_idx = category_offsets[category] +
                             sorted_index - category_bounds[category];
      if (output_idx < output_width) {
        output_blocks[static_cast<int64_t>(group_idx) * output_width +
                      output_idx] = static_cast<int>(entry);
        output_masks[static_cast<int64_t>(group_idx) * output_width +
                     output_idx] = static_cast<uint32_t>(entry >> 32);
      }
    }
  }
  __syncthreads();
  if (tid > 0 && tid < 8) {
    const int category = tid;
    const int count = category_counts[category];
    const int padded_count = (count + 7) & ~7;
    // count > 0 guarantees that the first real page has been written at 0.
    // Empty groups do not enter this loop and never read the output page.
    if (count > 0) {
      for (int local_idx = count; local_idx < padded_count; ++local_idx) {
        const int output_idx = category_offsets[category] + local_idx;
        if (output_idx < output_width) {
          output_blocks[static_cast<int64_t>(group_idx) * output_width +
                        output_idx] =
              output_blocks[static_cast<int64_t>(group_idx) * output_width];
          output_masks[static_cast<int64_t>(group_idx) * output_width +
                       output_idx] = 0;
        }
      }
    }
  }
  if (tid == 0) {
    output_seq_lens[group_idx] = min(category_offsets[0], output_width) * 4;
  }
}

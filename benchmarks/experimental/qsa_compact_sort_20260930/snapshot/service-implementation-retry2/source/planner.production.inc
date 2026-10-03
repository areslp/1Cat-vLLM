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
  auto* category_counts = reinterpret_cast<int*>(logical_owners);
  int* category_offsets = category_counts + 8;
  int* category_cursors = category_offsets + 8;
  int* warp_category_prefix = category_cursors + 8;
  __syncthreads();

  if (tid < 8) {
    category_counts[tid] = 0;
    category_offsets[tid] = 0;
    category_cursors[tid] = 0;
  }
  __syncthreads();
  for (int slot = tid; slot < kGroupedSparseHashCapacity;
       slot += kGroupedSparsePlannerThreads) {
    const unsigned long long entry = hash_table[slot];
    if (static_cast<uint32_t>(entry) != 0xffffffffu) {
      const int category =
          grouped_sparse_active_m_tiles(static_cast<uint32_t>(entry >> 32));
      atomicAdd(category_counts + category, 1);
    }
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
  constexpr int kPlannerWarps = kGroupedSparsePlannerThreads / kWarpSize;
  const int lane = tid & (kWarpSize - 1);
  const int warp = tid / kWarpSize;
  for (int chunk_start = 0; chunk_start < kGroupedSparseHashCapacity;
       chunk_start += kGroupedSparsePlannerThreads) {
    const unsigned long long entry = hash_table[chunk_start + tid];
    const uint32_t physical_microblock = static_cast<uint32_t>(entry);
    int category = 0;
    if (physical_microblock != 0xffffffffu) {
      const uint32_t token_mask = static_cast<uint32_t>(entry >> 32);
      category = grouped_sparse_active_m_tiles(token_mask);
    }
    unsigned category_lanes = 0;
#pragma unroll
    for (int scan_category = 1; scan_category < 8; ++scan_category) {
      const unsigned lanes =
          __ballot_sync(0xffffffffu, category == scan_category);
      if (category == scan_category) {
        category_lanes = lanes;
      }
      if (lane == 0) {
        warp_category_prefix[warp * 8 + scan_category] = __popc(lanes);
      }
    }
    __syncthreads();
    if (tid > 0 && tid < 8) {
      int prefix = category_cursors[tid];
#pragma unroll
      for (int scan_warp = 0; scan_warp < kPlannerWarps; ++scan_warp) {
        const int count = warp_category_prefix[scan_warp * 8 + tid];
        warp_category_prefix[scan_warp * 8 + tid] = prefix;
        prefix += count;
      }
      category_cursors[tid] = prefix;
    }
    __syncthreads();
    if (category != 0) {
      const int category_rank = warp_category_prefix[warp * 8 + category] +
                                __popc(category_lanes & ((1u << lane) - 1));
      const int output_idx = category_offsets[category] + category_rank;
      if (output_idx < output_width) {
        output_blocks[static_cast<int64_t>(group_idx) * output_width +
                      output_idx] = static_cast<int>(physical_microblock);
        output_masks[static_cast<int64_t>(group_idx) * output_width +
                     output_idx] = static_cast<uint32_t>(entry >> 32);
      }
    }
    __syncthreads();
  }
  __syncthreads();
  if (tid > 0 && tid < 8) {
    const int category = tid;
    const int padded_count = (category_counts[category] + 7) & ~7;
    for (int local_idx = category_counts[category]; local_idx < padded_count;
         ++local_idx) {
      const int output_idx = category_offsets[category] + local_idx;
      if (output_idx < output_width) {
        // W13: pad with this group's first real microblock (column 0: the
        // first entry of the first non-empty category, stored above) instead
        // of physical microblock 0, the null block, whose bytes are arbitrary
        // (e.g. GDN state). The mask stays 0, so the forward reads exactly
        // what the Python repoint (VLLM_SM70_QSA_GROUPED_PAD_FIX=1) gives it.
        output_blocks[static_cast<int64_t>(group_idx) * output_width +
                      output_idx] =
            output_blocks[static_cast<int64_t>(group_idx) * output_width];
        output_masks[static_cast<int64_t>(group_idx) * output_width +
                     output_idx] = 0;
      }
    }
  }
  if (tid == 0) {
    output_seq_lens[group_idx] = min(category_offsets[0], output_width) * 4;
  }
}

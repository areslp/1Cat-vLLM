// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025, D.Skryabin
// Original license and disclaimer are preserved in ORIGINAL-LICENSE.
// Standalone extraction of the QSA page4 planner from the STEP-48 release
// source. STEP58 keeps hash union/owner/key bits and compacts before bucket sort.
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <algorithm>
#include <climits>
#include <cstdint>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>

namespace {
constexpr int kWarpSize = 32;
constexpr int kGroupedVerifyHeads = 6;
constexpr int kGroupedSparseQueries = 8;
constexpr int kGroupedSparsePlannerThreads = 512;
constexpr int kGroupedSparseHashCapacity = 8192;
constexpr unsigned long long kGroupedSparseEmptyEntry = 0x00000000ffffffffULL;
constexpr int kGroupedSparseItemsPerThread =
    kGroupedSparseHashCapacity / kGroupedSparsePlannerThreads;
using GroupedSparseSort =
    cub::BlockRadixSort<unsigned long long, kGroupedSparsePlannerThreads,
                        kGroupedSparseItemsPerThread, unsigned long long>;
// Hash entries plus logical owners exactly fit Volta's 96 KiB opt-in limit.
// After loading both into registers, reuse this storage for sorting and scans.
constexpr size_t kGroupedSparsePlannerSharedMemory =
    kGroupedSparseHashCapacity *
    (sizeof(unsigned long long) + sizeof(uint32_t));
static_assert(sizeof(GroupedSparseSort::TempStorage) <=
              kGroupedSparsePlannerSharedMemory);
using GroupedSparseCompactScan =
    cub::BlockScan<int, kGroupedSparsePlannerThreads>;
static_assert(sizeof(GroupedSparseCompactScan::TempStorage) <=
              kGroupedSparsePlannerSharedMemory);

__device__ __forceinline__ void grouped_sparse_hash_insert(
    unsigned long long* __restrict__ hash_table,
    uint32_t* __restrict__ logical_owners, const int physical_microblock,
    const uint32_t token_mask, const int query, const int logical_token) {
  if (physical_microblock < 0 || token_mask == 0) {
    return;
  }
  int slot = (static_cast<uint32_t>(physical_microblock) * 2654435761u) &
             (kGroupedSparseHashCapacity - 1);
  const unsigned long long desired =
      (static_cast<unsigned long long>(token_mask) << 32) |
      static_cast<uint32_t>(physical_microblock);
#pragma unroll 1
  for (int probe = 0; probe < kGroupedSparseHashCapacity; ++probe) {
    const unsigned long long old =
        atomicCAS(hash_table + slot, kGroupedSparseEmptyEntry, desired);
    if (old == kGroupedSparseEmptyEntry ||
        static_cast<uint32_t>(old) ==
            static_cast<uint32_t>(physical_microblock)) {
      atomicOr(hash_table + slot, static_cast<unsigned long long>(token_mask)
                                      << 32);
      // Nonnegative int32 tokens use at most 29 bits after division by four.
      // The first contributing query owns shared pages, independent of request
      // slot IDs, physical allocation, insertion order, and hash collisions.
      const uint32_t owner = (static_cast<uint32_t>(query) << 29) |
                             (static_cast<uint32_t>(logical_token) >> 2);
      atomicMin(logical_owners + slot, owner);
      return;
    }
    slot = (slot + 1) & (kGroupedSparseHashCapacity - 1);
  }
}

__device__ __forceinline__ int grouped_sparse_physical_microblock(
    const int token, const int request_idx,
    const int* __restrict__ request_block_table,
    const int64_t request_block_table_stride, const int block_table_width,
    const int page_size, const int physical_page_stride,
    const int num_cache_blocks) {
  if (token < 0) {
    return -1;
  }
  const int logical_page = token / page_size;
  if (logical_page < 0 || logical_page >= block_table_width) {
    return -1;
  }
  const int page_offset = token - logical_page * page_size;
  const int physical_page =
      __ldg(request_block_table +
            static_cast<int64_t>(request_idx) * request_block_table_stride +
            logical_page);
  if (physical_page < 0 || physical_page >= num_cache_blocks) {
    return -1;
  }
  return physical_page * physical_page_stride + page_offset / 4;
}

__device__ __forceinline__ int grouped_sparse_active_m_tiles(
    const uint32_t token_mask) {
  int active_m_tiles = 0;
#pragma unroll
  for (int query = 0; query < kGroupedSparseQueries; ++query) {
    if ((token_mask & (0xFu << (query * 4))) != 0) {
      const int first_row = query * kGroupedVerifyHeads;
      const int last_row = first_row + kGroupedVerifyHeads - 1;
      active_m_tiles |= 1 << (first_row / 16);
      active_m_tiles |= 1 << (last_row / 16);
    }
  }
  return active_m_tiles;
}

template <int BucketItems>
__device__ __forceinline__ void grouped_sparse_sort_bucket(
    unsigned long long* hash_table, const uint32_t* logical_owners,
    const int valid_count) {
  constexpr int kItems = BucketItems / kGroupedSparsePlannerThreads;
  using BucketSort = cub::BlockRadixSort<
      unsigned long long, kGroupedSparsePlannerThreads, kItems,
      unsigned long long>;
  static_assert(sizeof(typename BucketSort::TempStorage) <=
                kGroupedSparsePlannerSharedMemory);
  unsigned long long entries[kItems];
  unsigned long long keys[kItems];
#pragma unroll
  for (int item = 0; item < kItems; ++item) {
    const int slot = threadIdx.x * kItems + item;
    if (slot < valid_count) {
      const unsigned long long entry = hash_table[slot];
      entries[item] = entry;
      keys[item] =
          (static_cast<unsigned long long>(grouped_sparse_active_m_tiles(
               static_cast<uint32_t>(entry >> 32))) << 32) |
          logical_owners[slot];
    } else {
      // Never read stale uncompacted hash slots or owner suffix.
      entries[item] = kGroupedSparseEmptyEntry;
      keys[item] = ULLONG_MAX;
    }
  }
  __syncthreads();  // All compact entries/owners live in registers before aliasing.
  auto& storage = *reinterpret_cast<typename BucketSort::TempStorage*>(hash_table);
  BucketSort(storage).Sort(keys, entries, 0, 36);
  __syncthreads();  // No output overwrite while another warp uses CUB storage.
#pragma unroll
  for (int item = 0; item < kItems; ++item) {
    hash_table[threadIdx.x * kItems + item] = entries[item];
  }
  __syncthreads();
}

__global__
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
  int thread_count = 0;
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
    thread_count += static_cast<uint32_t>(entry) != 0xffffffffu;
  }
  __syncthreads();
  // Every original entry and owner is in registers before scan/compaction
  // reuses this exact 96KiB allocation. No extra static shared scalar is added.
  auto& scan_storage =
      *reinterpret_cast<GroupedSparseCompactScan::TempStorage*>(hash_table);
  int thread_offset, valid_count;
  GroupedSparseCompactScan(scan_storage).ExclusiveSum(
      thread_count, thread_offset, valid_count);
  __syncthreads();  // All threads have consumed scan storage and aggregate.
  if (valid_count == 0) {
    if (tid == 0) output_seq_lens[group_idx] = 0;
    return;  // Uniform CTA branch: pages/masks and their suffix remain untouched.
  }
  int sort_capacity = kGroupedSparseHashCapacity;
  if (valid_count > 4096) {
    // Keep complete 8192-item sorting; no unnecessary compact/reload in fallback.
    auto& sort_storage =
        *reinterpret_cast<GroupedSparseSort::TempStorage*>(hash_table);
    GroupedSparseSort(sort_storage).Sort(sort_keys, entries, 0, 36);
    __syncthreads();
#pragma unroll
    for (int item = 0; item < kGroupedSparseItemsPerThread; ++item) {
      hash_table[tid * kGroupedSparseItemsPerThread + item] = entries[item];
    }
    __syncthreads();
  } else {
    // Stable compaction: thread order follows blocked original hash-slot order.
    int output = thread_offset;
#pragma unroll
    for (int item = 0; item < kGroupedSparseItemsPerThread; ++item) {
      if (static_cast<uint32_t>(entries[item]) != 0xffffffffu) {
        hash_table[output] = entries[item];
        logical_owners[output] = static_cast<uint32_t>(sort_keys[item]);
        ++output;
      }
    }
    __syncthreads();
    // CUB's block aggregate is identical for all threads: all take one branch.
    if (valid_count <= 1024) {
      sort_capacity = 1024;
      grouped_sparse_sort_bucket<1024>(hash_table, logical_owners, valid_count);
    } else if (valid_count <= 2048) {
      sort_capacity = 2048;
      grouped_sparse_sort_bucket<2048>(hash_table, logical_owners, valid_count);
    } else {
      sort_capacity = 4096;
      grouped_sparse_sort_bucket<4096>(hash_table, logical_owners, valid_count);
    }
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
    int last = sort_capacity;
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

  for (int sorted_index = tid; sorted_index < sort_capacity;
       sorted_index += kGroupedSparsePlannerThreads) {
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

}  // namespace

at::Tensor flash_attention_grouped_sparse_page4_plan(
    const at::Tensor& logical_indices, const at::Tensor& block_table,
    const at::Tensor& token_to_req, const at::Tensor& query_positions,
    const at::Tensor& sequence_lengths, at::Tensor& output_blocks,
    at::Tensor& output_masks, at::Tensor& output_seq_lens, const int page_size,
    const int physical_page_stride, const int num_cache_blocks) {
  TORCH_CHECK(logical_indices.is_cuda() && block_table.is_cuda() &&
                  token_to_req.is_cuda() && query_positions.is_cuda() &&
                  sequence_lengths.is_cuda() && output_blocks.is_cuda() &&
                  output_masks.is_cuda() && output_seq_lens.is_cuda(),
              "grouped sparse page4 planner tensors must be CUDA tensors");
  TORCH_CHECK(logical_indices.dtype() == torch::kInt32 &&
                  block_table.dtype() == torch::kInt32 &&
                  token_to_req.dtype() == torch::kInt32 &&
                  query_positions.dtype() == torch::kInt64 &&
                  sequence_lengths.dtype() == torch::kInt32 &&
                  output_blocks.dtype() == torch::kInt32 &&
                  output_masks.scalar_type() == at::ScalarType::UInt32 &&
                  output_seq_lens.dtype() == torch::kInt32,
              "grouped sparse page4 planner requires int32/uint32 metadata");
  TORCH_CHECK(logical_indices.dim() == 2 && logical_indices.size(0) > 0 &&
                  logical_indices.size(0) % kGroupedSparseQueries == 0 &&
                  logical_indices.size(1) == 2051,
              "grouped sparse page4 planner requires [8*N, 2051] indices");
  const int64_t num_groups = logical_indices.size(0) / kGroupedSparseQueries;
  TORCH_CHECK(
      block_table.dim() == 2 &&
          token_to_req.sizes() == at::IntArrayRef({logical_indices.size(0)}),
      "grouped sparse page4 planner request metadata is invalid");
  TORCH_CHECK(
      query_positions.sizes() == at::IntArrayRef({logical_indices.size(0)}) &&
          sequence_lengths.sizes() == at::IntArrayRef({block_table.size(0)}),
      "grouped sparse page4 planner visibility metadata is invalid");
  TORCH_CHECK(output_blocks.dim() == 2 && output_blocks.size(0) == num_groups &&
                  output_blocks.size(1) >= 4160 &&
                  output_masks.sizes() == output_blocks.sizes() &&
                  output_seq_lens.sizes() == at::IntArrayRef({num_groups}),
              "grouped sparse page4 planner outputs must be [groups, >=4160]");
  TORCH_CHECK(
      logical_indices.is_contiguous() && block_table.is_contiguous() &&
          token_to_req.is_contiguous() && query_positions.is_contiguous() &&
          sequence_lengths.is_contiguous() && output_blocks.is_contiguous() &&
          output_masks.is_contiguous() && output_seq_lens.is_contiguous(),
      "grouped sparse page4 planner metadata must be contiguous");
  TORCH_CHECK(page_size > 0 && page_size % 4 == 0 && physical_page_stride > 0 &&
                  num_cache_blocks > 0,
              "grouped sparse page4 planner requires page_size divisible by 4");
  TORCH_CHECK(logical_indices.device() == block_table.device() &&
                  logical_indices.device() == token_to_req.device() &&
                  logical_indices.device() == query_positions.device() &&
                  logical_indices.device() == sequence_lengths.device() &&
                  logical_indices.device() == output_blocks.device() &&
                  logical_indices.device() == output_masks.device() &&
                  logical_indices.device() == output_seq_lens.device(),
              "grouped sparse page4 planner tensors must share one device");

  c10::cuda::CUDAGuard device_guard(logical_indices.device());
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major == 7 && properties->minor == 0,
              "grouped sparse page4 planner supports SM70 only");
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  const cudaError_t smem_status =
      cudaFuncSetAttribute(grouped_sparse_page4_plan_kernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           kGroupedSparsePlannerSharedMemory);
  TORCH_CHECK(smem_status == cudaSuccess,
              "Failed to set grouped sparse page4 planner shared memory: ",
              cudaGetErrorString(smem_status));
  grouped_sparse_page4_plan_kernel<<<
      static_cast<unsigned>(num_groups), kGroupedSparsePlannerThreads,
      kGroupedSparsePlannerSharedMemory, stream>>>(
      logical_indices.data_ptr<int>(), block_table.data_ptr<int>(),
      token_to_req.data_ptr<int>(), query_positions.data_ptr<int64_t>(),
      sequence_lengths.data_ptr<int>(), output_blocks.data_ptr<int>(),
      output_masks.data_ptr<uint32_t>(), output_seq_lens.data_ptr<int>(),
      static_cast<int>(logical_indices.size(1)), logical_indices.stride(0),
      block_table.stride(0), static_cast<int>(block_table.size(0)),
      static_cast<int>(block_table.size(1)),
      static_cast<int>(output_blocks.size(1)), page_size, physical_page_stride,
      num_cache_blocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output_blocks;
}

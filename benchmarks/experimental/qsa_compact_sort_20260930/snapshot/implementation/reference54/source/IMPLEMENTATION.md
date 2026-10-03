# STEP-54 W0 standalone planner source

This directory contains an isolated PyTorch CUDA extension named
`qsa_planner54`. It extracts the STEP-48 QSA page4 planner from
`reference/step48/flash_decode_paged.cu`; it does not include or rebuild the
attention forward entry. The exported `plan_fwd` accepts the same arguments
and mutates the same three output buffers as
`flash_attention_grouped_sparse_page4_plan`.

The hash table capacity, physical-page union, OR-combined masks, minimum
logical-owner key, category function, 36-bit `cub::BlockRadixSort`, and checked
host wrapper are copied unchanged. The extension retains the original CUDA,
PyTorch, and CUB include dependencies. The extracted planner retains the upstream
BSD-3-Clause license and D.Skryabin copyright; the full notice is ORIGINAL-LICENSE.
The new binding file is Apache-2.0.

## Exact scatter argument

For each real hash entry, the original sort key is `(category << 32) | owner`;
the CUB call sorts that key over bits `[0, 36)`. The three category bits are
therefore the primary key and the owner is secondary. Since valid entries have
nonzero masks, they have categories 1 through 7. Empty entries use `ULLONG_MAX`
and sort after every valid key. The sorted entry array copied to shared memory
is consequently categories 1..7 in order followed by the empty suffix.

Seven threads independently binary-search the first shared slot with category
at least 2, 3, ..., 8. Category 1 begins at slot zero; each category's end is
the next search result, so its count is `end - begin`. The same sequential
`(count + 7) & ~7` prefix calculation gives the original category offsets.
For sorted index `i` in category `c`, `i - begin[c]` is its rank within that
category, equal to the original ballot/prefix rank. Writing it at
`offset[c] + i - begin[c]` therefore preserves every real entry's page, mask,
and order. Padding writes the same category-aligned slots, with mask zero and
the first real page in the group. A zero-count category has no padding; an
empty group performs no first-page read and writes sequence length zero. No
slot after the effective padded prefix is written, preserving the caller's
sentinel-initialized suffix.

## Build contract

Run on the authorized Linux CUDA host with the same PyTorch CUDA build and
compiler toolchain used for the production extension. On success, the script
prints `CANDIDATE_PATH=<absolute extension path>` for the admission harness
to persist as `build/candidate-path.txt`. `build54.py` explicitly
sets `CUDA_HOME=/home/l/work/qwen38-bench/cuda-home`,
`MAX_JOBS=2`, and `TORCH_CUDA_ARCH_LIST=7.0`. It requires an explicit
`BUILD_DIR` outside this source directory and uses `torch.utils.cpp_extension.load`
to build only `qsa_planner54`. Invoke it with the host's approved repository
`.venv/bin/python`; for example:

```bash
BUILD_DIR=/home/l/work/qwen38-bench/step54/build/qsa_planner54 \
  /path/to/vllm/.venv/bin/python /path/to/step54/source/build54.py
```

A CUDA-capable PyTorch install, the configured CUDA toolkit, CUB headers,
`ninja`, and a compatible host C++ compiler are required. Source creation here
does not establish that compilation or GPU execution succeeds; record those
results separately. The production `.so` is neither referenced nor modified
by this build script.

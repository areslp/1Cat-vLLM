# Handoff: `exp/sub-block-chunks`

Tracked summary for the branch. The full brief with every measurement,
retraction and process trap lives on the development host, git-excluded:
`oh-my-gpu:/home/l/work/1Cat-vLLM/logs/handoff/HANDOFF.md` plus the
task-lifecycle contract `packet.yaml` (validates `READY`).

## 2026-09-13 — BM32 any-page kernel path (this commit)

**Task intent.** Close 1CatAI/1Cat-vLLM#490 on SM70: a resident 240K-context
decoder collapsed to 0.5–0.6 tok/s while another request chunk-prefilled.
Commit d8afc4722 fixed the kernel routing of the decoder's row and added a
prefill pacing knob. This commit removes the second cost that only MTP users
pay: with MTP the Mamba page grows by `num_spec` conv slots, the align-mode
attention block becomes 816 instead of 784, and every paged-prefill chunk call
fell off the kernel's page-784-only fast path (2.3x slower).

**Changed scope.**

- `flash-attention-v100/kernel/fused_mha_forward_paged.cu`: the D256 BM32
  phase body takes `page_block_size` as a kernel argument (it was a
  compile-time 784 used only to map each 16-token page slot); the dispatch
  gate accepts any page size that is a multiple of 16 when
  `VLLM_FLASH_V100_PREFILL_D256_BM32_ANY_PAGE=1`. Page 784 keeps the existing
  gate. The M < 32 low-smem software-pipeline gates and the fp8 bridge
  workspace still assume 784 (not generalised).
- `vllm/envs.py`: documents the new knob (default off).
- `tests/kernels/attention/test_sm70_flash_v100_paged_prefill_any_page.py`:
  pages 816/896, q=32/784, flag off/on, against page 784 and a dense reference.

**Validation.**

- Extension rebuilt on oh-my-gpu (`uv pip install --no-build-isolation -e
  flash-attention-v100`, TORCH_CUDA_ARCH_LIST=7.0, ccache), import ok.
- Standalone (GPU 2, q=784 at 104K context, fp16, 12/2 heads): page 784
  43.99 ms unchanged; page 816 and 896 104.4 ms -> 43.99 ms with the flag;
  all outputs bit-identical to the dense reference and to page 784.
- `pytest tests/kernels/attention/test_sm70_flash_v100_paged_prefill_any_page.py`:
  8 passed. `tests/kernels/attention/test_sm70_flash_v100_*.py` and
  `test_sm70_e4m3_scalar_fp32.py`: 147 passed with the flag off and on.
- End to end (Qwen3.8-27B-QUASAR-NVFP4, TP2 on 2x V100-PCIE-32GB, fp16 KV,
  MTP4 via `VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS=1`, decode-rows flag on,
  resident decoder at 240,000 tokens, fixed prompts, zero preemptions):
  partner 16K prefill 21.3 s -> 19.6 s, partner 131K 314.7 s -> 207.9 s; with
  `VLLM_1CAT_PREFILL_PACE_STEPS=4` the 131K prefill 400.1 s -> 293.8 s while
  the resident decoder runs 9.86 tok/s (was 7.38). Output text and 512+8
  logged tokens per arm identical across boots; logprob shifts within the
  measured cross-boot noise floor.
- Not run: lint beyond `ruff check`/`ruff format` on the Python files; no
  clang-format pass on the `.cu` change.

**Remaining risks / follow-ups.**

- Two-request MTP decode steps cost 0.18 s at 240K versus ~0.10 s solo; this
  bounds prefill pacing at ~21–25 tok/s for the resident decoder. Unattributed.
- 0.15–0.23 s residual per mixed step with all flags on (drafting plus eager
  overhead). Unattributed.
- fp8_e5m2 in-server behaviour of the decode-rows path is unit-tested only.
- Every measurement is n=1; two or three resident decoders and longer
  generations are unexercised.
- Recommended production settings on this hardware:
  `VLLM_FLASH_V100_PREFILL_PREFIX_DECODE_ROWS=1`,
  `VLLM_FLASH_V100_PREFILL_D256_BM32_ANY_PAGE=1` whenever MTP is on,
  `VLLM_1CAT_PREFILL_PACE_STEPS=4`. Do not enable
  `VLLM_1CAT_ALLOW_SUB_BLOCK_PREFILL` for production.

**Commit readiness.** Default behaviour byte-identical with the flags unset;
nothing here is proposed upstream as-is; nothing has been posted to #490.

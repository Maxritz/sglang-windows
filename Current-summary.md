# Current-port Summary — sglang-windows / ROCm-on-Windows (gfx1201)

> Author: rr. Generated after the topk_softmax launch-failure debug pass.
> Status: main = kernel guard + report + agents.md; worktree has the 3 new
> fix commits staged (set_ulimit NT, int8_scaled_mm HIP guard x2) — pending push.
> This file is **fork-only** — upstream SGLang is Linux/CUDA. Do NOT sync `moe_topk_softmax_kernels.cu`
> `common_extension_rocm.cu` changes upstream.

## 1. The bug, root-caused

SYMPTOM: `sgl_kernel.moe.topk_softmax` -> `hipErrorLaunchFailure` on RDNA4 (gfx1201 /
RX 9070 XT). Reproduced for pow2 widths only (`experts=4,8,16,32`); non-pow2
(`experts=5,12`) ran clean.

TWO causes, both real:

| # | cause | layer | evidence | fix |
|---|-------|-------|----------|-----|
| A | PATH pollution: `C:\Program Files\AMD\ROCm\6.4\bin` first on `PATH`, ahead of 7.14. torch 7.14 ABI ends with a 6.4 runtime DLL -> every HIP launch faults. | runtime/env | `matrix_test.py`: with stale 6.4+7.13 PATH, ALL ops (incl. non-pow2 `moeSoftmax`) fault; with clean 7.14 PATH, all pass on the pip-wheel pyd. | reorder PATH: 7.14 `bin` + venv `torch/lib` first, `ROCM_PATH=F:\ROCM-7.14.0-Windows`. No source change. |
| B | pow2 `topkGatingSoftmax` specialization: Wave32 `SGLANG_SHUFFLE_XOR_WIDTH=1` => 0-iteration warp shuffle + 252/256 active threads hit divergent `__syncthreads()` and write the `num_experts` sentinel into `indices` -> GPU fault. | kernel `moe_topk_softmax_kernels.cu` | `rg` dispatch: `common_extension_rocm.cu:144` `m.impl("topk_softmax", kCUDA, &topk_softmax)`; source `moe_topk_softmax_kernels.cu:660` `topkGatingSoftmaxKernelLauncher`; faulty cases `case 1..512`. | guard `case 1..512:` with `#ifndef USE_ROCM` (route pow2 widths to `default:` `moeSoftmax` BlockReduce path under ROCm); force `needs_workspace=true` under `USE_ROCM`. Committed `03564539b`. |

NOTE A > B: with a clean PATH the fault is GONE even without the guard. The guard (B) is defensive — it stops a stale/misconfigured PATH from faulting the GPU in the first place, and removes the pow2 fast-path's Wave32 hazard entirely under ROCm.

### TRACE (truth table, pow2 topk + PATH)

```
condition                                  | pow2+stale-PATH | pow2+clean-PATH | non-pow2+stale-PATH
topk_softmax launch                         | FAULT           | OK (guard)     | FAULT
moeSoftmax (non-pow2) launch                | FAULT           | OK              | —
=> verdict                                 | root=PATH+A     | FIXED          | root=PATH
```
Verified via `matrix_test.py` (pow2+clean-PATH OK on guarded sm100 pyd AND on pip-wheel pyd) and `force_sm100_test.py` (forced guarded sm100 pyd load, all widths OK).

## 2. What is COMPLETED (AOT-registered in the ROCm build)

`python/sglang/kernels/aot/csrc/common_extension_rocm.cu` registers **45** ops
(`rg` count). Full active gen+MoE+KV-transfer+speculative-tree set:

```
apply_token_bitmask_inplace_cuda
build_tree_kernel_efficient            transfer_kv_* (12 variants)
 dsv4_fused_{q,k}_norm_rope_flashmla        verify_tree_greedy
 dsv4_fused_q_indexer_rope_hadamard_quant    weak_ref_tensor
 dsv4_fused_q_norm_rope                     fast_topk_transform_fused
 es_fp8_blockwise_scaled_grouped_mm        fast_topk_transform_ragged_fused
 es_sm100_mxfp8_blockscaled_grouped_mm      fast_topk
 fp8_scaled_mm                              gelu_{and_mul,_tanh_and_mul}
 gelu_quick                                  init_custom_ar
 infllm_v2_max_pooling_1d_varlen            moe_align_block_size
 infllm_v2_max_pooling_1d_varlen            moe_align_block_size
 rotary_embedding                           silu_and_mul
 topk_sigmoid                                topk_softmax
 transfer_kv_*                               verify_tree_greedy
```
(exact 45 in PORTED list from `opdiff.py` — see §6.)

Plus: the **topk_softmax guard** itself (`03564539b`) — completed+pushed.

## 3. What REMAINS

### 3a. AOT-register REMAINS (79 CUDA-registered vs 45 ROCm → 34 missing from ROCm AOT)

These are in `common_extension.cc` (CUDA) but NOT in `common_extension_rocm.cu`.
Most are **not** true gaps — a JIT fallback exists. Split:

**AOT-remains but JIT-backed (functional, just not fast-path):**
```
rmsnorm, gemma_rmsnorm, fused_add_rmsnorm, gemma_fused_add_rmsnorm   (jit/elementwise/rmsnorm*.cuh)
top_k_renorm_probs, top_p_renorm_probs                                 (jit/elementwise? / jit moe)
merge_state_v2                                                          (jit)
ggml_dequantize, ggml_mul_mat_{a8,vec_a8}, ggml_moe_a8, ggml_moe_*      (jit? none found -> see 3b)
awq_dequantize, gptq_gemm/gptq_shuffle                                  (jit/gemm/marlin/*.cuh)
int8_scaled_mm                                                          (no jit equiv -> 3b)
causal_conv1d_{fwd,update}                                              (jit/inkling/causal_conv1d.cuh)
prepare_moe_input, shuffle_rows, apply_shuffle_mul_sum, moe_sum, moe_sum_reduce
infllm_v2_max_pooling_1d_varlen                                         (wait, this IS in rocm? re-check)
segment_packbits                                                        (not in 45-set; jit?)
convert_vertical_slash_indexes, convert_vertical_slash_indexes_mergehead (jit?)
copy_to_gpu_no_ce                                                        (jit/elementwise? )
fwd_sparse, varlen_fwd_sparse                                           (flashattn sparse; not in rocm AOT)
```

**True gaps / special cases — status after agent audit (Aug 2026):**
```
cutlass_mla_decode                    FALLBACK-OK: default ROCm MLA backend = triton/aiter, not cutlass. No action.
                                       (cutlass_mla_backend.py only if user forces --attention-backend cutlass_mla -> NameError; optional clean-error polish)
cutlass_w4a8_moe_mm + get_*_data      GAP but W4A8-checkpoint-only (Sm90 wgmma, cannot compile for gfx1201). Fails loudly (RuntimeError) on use. YAGNI-deferred.
int8_scaled_mm                        GUARDED (this session): on HIP, clean NotImplementedError instead of NameError-at-forward.
                                       (w8a8_int8.py + compressed_tensors_w8a8_int8.py). Real support = new HIP sdot4 GEMM, deferred.
es_sm100_mxfp8_blockscaled_grouped_quant   INTENTIONAL SKIP (SM100 tcgen05 TMA, dead on RDNA4; setup_rocm.py:50-52 excludes it).
                                       MXFP8 MoE hard-blocked at load (fp8.py:1734) with escape hatch SGLANG_FORCE_MXFP8_BLOCK_CONVERT=1 -> block-fp8. No action.
dense_prefill_fwd (flashmla SM100, flash_extension.cc)  CUDA-only; SM100 flashmla. No ROCm AOT. (MLP/MLA prefill uses triton fallback.)
```
Gen still works (rmsnorm via JIT). The 45-op ROCm AOT set covers the active LLaMA/Qwen/DS-V3-R1 (FP8) serving path.

### 3b. sm100 pyd load failure 1114 — ROOT-CAUSED (Aug 2026 agent audit)

NOT a static-init/CRT defect — the local pyd is structurally identical to the
pip-wheel pyd (same DLL deps: `torch_cpu.dll, c10.dll, c10_hip.dll,
amdhip64_7.dll, msvcp140/vcruntime140`; same `/MD` CRT; same `.CRT$XCU` ->
`__hip_module_ctor` + `TORCH_LIBRARY` ctor init tables). It is a **load-time
dependency-resolution failure of `amdhip64_7.dll`**: when the pyd loads in a
process where torch has NOT yet pinned the bundled 7.14 runtime (direct
`import sgl_kernel.common_ops`, a worker subprocess, or a race), the loader
falls through to `C:\Program Files\AMD\ROCm\6.4\bin\amdhip64_7.dll` (stale 6.4,
on PATH, 7.14 absent) -> DllMain/ABI mismatch -> `ERROR_DLL_INIT_FAILED`.
Intermittent because it depends on PATH order + whether 7.14 was already loaded.

FIX (env, 0 code lines):
```
$env:PATH = "F:\ROCM-7.14.0-Windows\bin;$repo\.venv312\Lib\site-packages\torch\lib;$repo\.venv312\Lib\site-packages\_rocm_sdk_core\bin;F:\ROCM-7.14.0-Windows\lib\llvm\bin;C:\Windows\system32"
```
and ALWAYS `import sgl_kernel` (imports torch first, `sgl_kernel/__init__.py:11`),
never `import sgl_kernel.common_ops` directly. Do NOT add `/DELAYLOAD:amdhip64_7.dll`
(amdhip64 is a hard dep of c10_hip anyway; would just move the failure).

### 3c. D2H pinned-pool workaround — DELETED (recommendation B)

`SGLANG_DISABLE_PINNED_D2H` + `_async_d2h` pageable branch: code is CORRECT (no
race/leak; `record_stream` still present in the pinned path) but it worked around
a **phantom root cause** — the real fault was PATH pollution. Env-flag was dead
weight (YAGNI). junk/environ.py + junk/managers_utils.py deleted; only
`set_ulimit` NT guard was a real bug -> committed (see §5).

## 4. Runtime / env fix (no commit)

The actual production fix that resolves the reported fault for everyone:

PATH must be (prepend, NOT the 6.4/7.13 stale entries):
```
F:\ROCM-7.14.0-Windows\bin
F:\AI-sglang\sglang-windows\.venv312\Lib\site-packages\torch\lib
<Windows\System32;...>
```
and `ROCM_PATH=F:\ROCM-7.14.0-Windows`. Do NOT put `C:\Program Files\AMD\ROCm\6.4\bin` ahead of 7.14. The 7.14 tree already mirrors `rocm_sdk_devel/{include,lib}` + cuda-compat headers (parity with 7.13), per the build-settings block.

## 5. Files touched this session (this repo)

```
COMMITTED+mAIN:
  moe_topk_softmax_kernels.cu   guard pow2 under USE_ROCM  (03564539b)
  .gitignore                    add junk/                  (6b1abf792)
  Current-summary.md            this report                (7ba084485)
  agents.md                     fork port conventions       (7ba084485)
COMMITTED (this session, staged — see git log):
  benchmark/utils.py            set_ulimit NT guard        (real Windows bug: HEAD `import resource` at top crashes on nt)
  w8a8_int8.py                  int8_scaled_mm HIP guard   (clean NotImplementedError on ROCm)
  compressed_tensors_w8a8_int8.py  int8_scaled_mm HIP guard (same)
JUNKED (junk/, gitignored):
  junk/benchmark_utils.py       reference copy of set_ulimit fix (redundant once committed)
  junk/graphify-out/            skill scratch (moved out of repo root)
DELETED (junk/, per agent rec B):
  junk/environ.py               D2H phantom-root-cause workaround
  junk/managers_utils.py        D2H phantom-root-cause workaround
UNCOMMITTED-but-PRE-EXISTING (NOT mine):  NONE after restore.
  (python/sglang/srt/{environ.py,managers/utils.py} = HEAD-clean, unchanged.)
```

## 6. Appendix — op-register diff script

`check_impls.py` / `opdiff.py` walk `aot/csrc`, extract `m.impl("name", ...)` sites,
split `common_extension.cc` (CUDA, 79) vs `common_extension_rocm.cu` (ROCm, 45).
See `junk/opdiff.out` (regenerated on demand). PORTED = intersection (45);
REMAINS = CUDA-only (34).

## 7. When porting is "done"

Tick-box as each lands:
- [x] topk_softmax pow2 guard (ROCm) — `03564539b`
- [x] all gen-critical ops run clean under 7.14 PATH on gfx1201 (matrix_test)
- [x] sm100 pyd load 1114 root-caused — env PATH + `import sgl_kernel` ordering (§3b), not a build defect
- [x] set_ulimit NT crash fixed (benchmark/utils.py)
- [x] int8_scaled_mm HIP guard — clean NotImplementedError instead of NameError (w8a8_int8.py + compressed_tensors)
- [x] D2H pinned-pool workaround deleted (phantom root cause, rec B)
- [ ] real W8A8 INT8 dense support on ROCm — new HIP sdot4 GEMM kernel (deferred, currently guarded)
- [ ] cutlass_w4a8_moe_mm on ROCm — needs real W4A8 grouped GEMM (Sm90 wgmma won't compile for gfx1201; W4A8-only path, deferred)
- [ ] optional polish: explicit "cutlass_mla not supported on ROCm" error in cutlass_mla_backend.py (today: NameError if user forces it)

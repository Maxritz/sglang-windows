# Agent instructions — sglang-windows port

This is the AMD/Windows ROCm port of SGLang (upstream `sgl-project/sglang` is Linux/CUDA only).
The files here describe fork-specific concerns that have NO upstream equivalent.

## Current-summary.md

`Current-summary.md` is the rolling **port status board**: what is completed vs
what remains for the Windows/ROCm (gfx1201 RDNA4 + gfx1031 RDNA2) port.

**When porting a feature lands or a defect is closed, update `Current-summary.md`:**
- move the item under "What is COMPLETED" (with the fix commit / build hash), and
- remove/age-down the matching line in "What REMAINS".

Keep the TRUTH-TABLE / TRACE block for any kernel defect: decision tree, truth
table, race-condition checklist, load conditions, and a one-line VERDICT with
the backend + hardware the trace ran against. If it isn't traced, it isn't done.

This is fork-only. Anything under `common_extension_rocm.cu` / `moe_topk_softmax_kernels.cu`
with a `#ifndef USE_ROCM` / `USE_ROCM` guard must NOT be synced upstream.

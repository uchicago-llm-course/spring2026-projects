# Week 1: Preliminary Results and the Price of Infrastructure

**Parameter Golf Competition Blog — Cedric Haddad**

---

The first week of Parameter Golf was supposed to be about GPU kernels. It ended up being about Docker entrypoints, APT dependency hell, and the surprising intelligence buried in our CPU-side scaffolding. That said, we ran three real experiments today and the results are already telling us something interesting about the competition landscape.

## The Core Insight: The SOTA Quantization Consensus Is Probably Wrong

The competition has converged on **int6 attention / int5 MLP** quantization as the standard configuration. Every serious submission uses some variant of this. Our first experiment tested this assumption directly by sweeping 16 different quantization schemes across synthetic model weights shaped identically to our 26.9M-parameter SOTA architecture.

The most striking result:

| Scheme | Est. compressed size | Attention MSE |
|--------|---------------------|---------------|
| SOTA baseline (int6 attn / int5 MLP) | 11.59 MB | 3.46e-8 |
| **int8 attn / int4 MLP** | **11.59 MB** | **2.04e-9** |
| int7 attn / int5 MLP | 12.29 MB | 8.44e-9 |

The `int8 attn / int4 MLP` scheme compresses to **exactly the same artifact size** as the SOTA baseline but achieves **17× lower attention reconstruction MSE**. The intuition makes sense in retrospect: attention weights are the model's "reasoning core" and likely drive BPB more directly than MLP weights, which are more compressible. Giving attention weights more bits while reclaiming space from MLP is a free reallocation.

This is not proven to improve BPB on the actual eval set—that requires real training and real GPU forward passes. But it's a meaningful hypothesis to test, and one that the competition hasn't explored systematically precisely because evaluating each quantization scheme manually takes 3-4 hours of kernel writing. Our proc-macro compiler is designed to make this grid search trivial.

## Experiment 2: The Prune-then-Quantize Hypothesis Holds

The arXiv:2603.18426 paper ("Progressive Intensity Hypothesis") argues that when composing multiple lossy transforms on weights, you should apply the weaker perturbation first. For our pipeline: prune first, then quantize.

We ran a systematic A/B test: **Path A** (prune → quantize) vs **Path B** (quantize → prune) across:
- 4 matrix sizes (64×128 through 512×1536, covering all of our GEMM shapes)
- 5 keep ratios (95% down to 50%)
- 3 bit-widths (int4, int5, int6)
- H100's native 2:4 structured sparsity

**Path A won 100% of the 72 test conditions.** The gains are largest at aggressive settings: at 75% keep with int4, Path A achieves 14% lower reconstruction error than Path B. This is essentially free—the pipeline ordering costs nothing at inference time but meaningfully reduces quantization noise.

We have implemented both the prune-then-quantize pipeline and the A/B test harness in Rust (`pg-quant/prune.rs`), fully verified with 8 unit tests. All 132 tests across the workspace pass on macOS.

## Experiment 3: Only One Architecture Can Go Wide

The architecture sweep across 10 const-generic variants produced a clear constraint picture:

| Variant | Params | int5/6 size | int4/5 size |
|---------|--------|-------------|-------------|
| Baseline (d=512, 11L) | 26.8M | 11.59 MB ✅ | 10.10 MB ✅ |
| Wide (d=576, 11L) | 33.7M | 14.62 MB ✅ | 12.74 MB ✅ |
| **d=640 (Aggressive-Q)** | **41.5M** | **18.00 MB ❌** | **15.69 MB ✅** |
| Deep-Narrow (d=384, 16L) | 21.9M | 9.46 MB ✅ | 8.24 MB ✅ |

The d=640 model with **41.5M parameters** (54% more capacity than baseline) can fit within 16MB if and only if int4/int5 quantization is viable. This is the crucial dependency: if the quant sweep proves int4 attention doesn't destroy BPB, we can train a dramatically wider model within the same budget. If not, d=576 is our ceiling. This is the question our GPU experiments will answer next week.

## The Infrastructure Tax

None of this was supposed to take this long. Getting these experiments running on Modal's 8×H100 infrastructure took the entire week due to four separate blockers:

**1. cudarc API deprecation.** The `0.16.6` release removed `CudaDevice` entirely. All GPU memory allocation that previously went through `device.alloc_zeros()` had to be migrated to `stream.alloc_zeros()`, reflecting the library's shift to stream-bound memory scheduling. Three lines in `pg-core/src/tensor.rs` blocked the entire CUDA build.

**2. Docker APT hell.** The `nvidia/cuda:12.4.1-devel-ubuntu22.04` base image locks the CUDA version to 12.4, but `apt-get install libnccl-dev=2.21.*` resolved to the `+cuda12.5` variant — a strict dependency mismatch that made the entire container unbuildable. Fix: pin explicitly to `libnccl2=2.21.5-1+cuda12.4`.

**3. Modal API churn.** Upgrading to `modal 1.4.1` broke three separate things: `modal.Mount` was removed in favor of `context_dir`; `_allow_background_volume_commits` was removed; `VOLUME` instructions in Dockerfiles are unsupported. Each required a targeted fix.

**4. The ENTRYPOINT hijack.** The most insidious bug: `ENTRYPOINT ["pg-train"]` in the Dockerfile overrode Modal's Python process entrypoint. Instead of running `gemm_bench()`, Modal would boot the container, which immediately started `pg-train`, ran one step (loss 6.9316 — correctly initialized, at least), and exited. Modal correctly reported "container exited successfully but never requested inputs." One step of training on H100 took 732 seconds in "elapsed time" — entirely startup overhead, zero actual compute. Removing the ENTRYPOINT instruction fixed this. 

On the bright side: the accidental `pg-train` run confirmed that our model initializes correctly. `loss 6.9316 ≈ -ln(1/1024) = ln(1024)` is exactly the expected cross-entropy for uniform random predictions over a 1024-token vocabulary. The math is right.

## Plans for Next Week

With the Docker and Modal infrastructure solidified, next week has a single overriding priority: **GPU forward-pass parity**.

The concrete gate: load the PyTorch SOTA checkpoint weights via `pg-compat`, run the same input batch through both the CPU reference model and the GPU model, and assert that maximum absolute logit difference is below `1e-3`. We do not proceed to GPU training until this passes.

The parallel track: with the fixed Dockerfile now deployed, we can run the actual GEMM and NCCL benchmarks on H100 hardware to get real TFLOPS numbers. Those numbers will determine whether our cuBLASLt wrapper is achieving close to peak hardware utilization or whether we need to tune the algorithm selection.

The quantization insight from Experiment 1 — that `int8 attn / int4 MLP` might Pareto-dominate the SOTA consensus — gets tested as part of the quant sweep. We are building the sweep infrastructure now so that when the GPU training loop is ready, running 50 configurations takes one Modal job rather than a week of manual kernel work.

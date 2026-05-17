# Week 5: From Fast Proxy to Defensible Record Path

When we wrote the first paper draft, the Rust/CUDA stack had finally crossed an important systems threshold: it could run a current-frontier-shaped Parameter Golf training surface on 8xH100 with a sub-120 ms/step systems profile. That draft deliberately did not claim a leaderboard BPB. It framed the result as systems-valid timing, not a completed record submission.

Week 5 was about testing whether that distinction was strong enough.

The short version: the stack is much more defensible now, but the path to a record is narrower than it looked from the first draft. We hardened recurrence semantics, separated speed probes from leaderboard-clean exact profiles, implemented and measured an exact recurrent boundary fusion experiment, tested graph-side GEMM capture and deferred weight GEMMs, and ruled out several attractive but insufficient cuts. The remaining systems blocker is now specific: exact active recurrent backward replay.

That is progress, even though it is not the kind of progress that produces a clean victory graph. The project moved from "we have a fast profile" to "we know which parts of the fast profile are defensible, which are probes, and which exact computation still has to be fused."

## What The First Draft Claimed

The first draft reported a Rust/CUDA systems stack for Parameter Golf, not a final leaderboard submission. The paper's central claim was that implementation details become first-class modeling constraints under the challenge rules: precision flow, memory traffic, data residency, CUDA graph capture, optimizer communication, evaluation legality, and artifact accounting all matter as much as architecture search.

The draft described a stack with:

- A Rust workspace split into `pg-core`, `pg-kernels`, `pg-model`, `pg-optim`, `pg-data`, `pg-quant`, `pg-eval`, and `pg-train`.
- A current-frontier-shaped #2135 audit spec using 1024 training context, 2560 eval/TTT context, 524,288 global train tokens per step, and 8 H100 GPUs.
- Spec-owned runtime profiles instead of untracked environment-variable hot paths.
- A BF16 direct-compact backward profile.
- cuDNN BF16 SDPA and fused QKV/RoPE/prepack paths.
- A device-resident token-ring full-schedule sampler.
- CUDA graph capture for the no-loss record-shaped step.
- Sharded Parallel Muon graph paths.
- CaseOps byte-sidecar audit fields.
- Artifact byte/hash plumbing in progress.

It also stated what was not done:

- No full 600-second train/eval/export run had completed.
- BPB had not been validated on full validation.
- The final code+model artifact budget was not yet proven.
- NCCL overlap remained disabled.
- The proposal's quantization proc-macro compiler was not implemented.
- BigramHash embedding merge was not implemented.
- True XSA-inside-SDPA fusion was not implemented.
- Persistent-CTA block backward was not implemented.

That framing was correct. Week 5 mostly reinforced it.

## The Main Week 5 Lesson

The most important thing we learned this week is that sub-120 ms/step timing is not enough. It only matters if the profile preserves the exact computation we intend to claim.

Earlier in the project, a pass1 straight-through recurrent speed probe appeared to cross the target. That was exciting, but code review found a serious issue: the path skipped too much recurrent backward work. It was useful as a speed signal, but not valid as a leaderboard-clean systems result.

That changed the engineering posture. We stopped treating the fastest number as the target and started treating exactness as the target. The benchmark became:

> Can we reduce active recurrent replay without changing the recurrent backward semantics?

The answer so far is: partially, but not enough.

## Recurrent Semantics Were Tightened

The first concrete Week 5 fix was in recurrent backward selection. The pass1 straight-through logic needed to be layer-scoped. Without that, a profile could silently skip more recurrent work than intended.

We added explicit selection logic and tests so a configured number of straight-through recurrent layers means exactly that number of layers, starting from the recurrence start layer. This sounds small, but it matters. In a codebase with many speed flags, "almost the same computation" is not good enough. The audit needs to know whether a profile is exact, a speed probe, or an algorithmic variant that still needs BPB validation.

Speed-probe profiles remain useful. They tell us what a cut might be worth. But they are no longer allowed to pass as clean frontier profiles.

## Exact Recurrent Boundary Fusion

The first exact recurrent cut we implemented was a boundary fusion between pass2 and pass1 backward.

The recurrent block applies the same logical transformer block twice. In backward, that means pass2 computes gradients into the intermediate activation, then pass1 consumes that gradient and propagates back to the original layer input. There is a natural boundary between these passes where pass2 has just produced data that pass1 immediately needs.

We implemented an opt-in fused CUDA path that combines:

- The final pass2 QKV/norm/residual tail.
- The initial pass1 MLP residual backward materialization.

The goal was to avoid one launch/read/write boundary while keeping pass1 backward semantically intact. The implementation is deliberately narrow: pass1 still runs; it only skips the residual stage that the fused pass2 tail already materializes.

The result was correct enough to keep as an experiment, but not strong enough to promote:

| Profile | Measured ms/step | Active recurrent window | Inactive window |
| --- | ---: | ---: | ---: |
| Exact recurrent boundary fusion | 136.65 | 147.60 | 120.04 |

This did not close the 120 ms gap. It also showed that the remaining overhead is not just one obvious boundary kernel. The active recurrent path needs a larger replay reduction.

## Graph-Side GEMM Capture Helped, But Not Enough

The next exact candidate was graph-side GEMM capture with deferred weight GEMMs. The graph path had been disabling side-stream backward GEMM overlap unless graph-side capture was enabled, which meant active recurrence could still pay more serialized dX/dW work inside replay.

We tested a profile that captured side GEMM work and deferred supported weight GEMMs while preserving exact accumulation.

That gave the best recent exact movement:

| Profile | Measured ms/step | Active recurrent window | Inactive window |
| --- | ---: | ---: | ---: |
| Graph-side/deferred exact candidate | 133.95 | 144.63 | 117.00 |

This was real progress. The inactive window fell below the 120 ms target. But active recurrence remained far above target. That is the clearest signal from the week: the base graph is basically fast enough, but recurrence reopens the gap.

## Negative Results That Matter

We also tested changes that did not deserve to land.

One MLP scratch-overlap patch tried to avoid a synchronization between MLP-down deferred weight GEMM and MLP-up dX by moving a BF16 scratch output into a separate buffer. It compiled and passed local checks, but the H100 timing regressed:

| Profile | Measured ms/step | Active recurrent window | Inactive window |
| --- | ---: | ---: | ---: |
| MLP scratch-overlap candidate | 137.06 | 148.21 | 120.28 |

That patch was reverted.

We also kept prior negative sweeps in the ledger:

- Graph-side GEMM capture is exact but did not win reliably in earlier runs.
- Chunked compact QKV norm/resid measured around 137.44 ms/step.
- Split compact QKV norm/resid regressed badly, around 156.91 ms/step.
- Grouped-KV SparseAttnGate/XSA backward plus fused global clip and parallel local Muon measured around 137.83 ms/step.
- NCCL overlap remains unproven and disabled.

These results are useful because they rule out easy narratives. The remaining 13-17 ms/step is not hiding in a small reducer choice, a global clip flag, or a simple optimizer scheduling tweak.

## Current State Of The Stack

The systems stack is now much stronger than it was when the project started.

Implemented or mostly implemented:

- Current #2014/#2135-shaped target specs.
- Spec-visible BF16 direct-compact profile.
- Audited BF16 backward chain with zero F32 hot-path bridge count in the intended path.
- Device-resident token-ring full schedule.
- Hot-path host batch flatten count at zero.
- CUDA graph capture for the no-loss record-shaped step.
- Sharded Parallel Muon local/pre-norm graph paths.
- CaseOps sidecar SHA emission.
- Chunked BF16 output CE path without persistent full logits.
- GPU LoRA eval support for chunked BF16 CE.
- Mixed int5/int6 export fixes in progress.
- Explicit audit fields for proposal-completeness gaps.

Not complete:

- Full compliant train/eval/export.
- Meaningful full-validation BPB.
- Three-seed BPB statistics.
- Final artifact budget proof in the same full record run.
- NCCL overlap proof.
- Persistent-CTA block backward.
- True XSA-inside-SDPA fusion.
- BigramHash embedding merge.
- Quantization procedural-macro compiler.

The distinction is important. The stack is no longer a vague prototype, but it is not a winning Parameter Golf stack yet.

## The Timing Picture

The current timing story has three layers.

First, the first paper draft reported a sub-120 systems-valid profile. That result remains useful as evidence that the Rust/CUDA architecture can execute the corrected frontier-shaped workload at the right order of magnitude.

Second, after recurrent-backward review, the clean exact profiles are in the low-to-mid 130 ms/step range. Recent exact measurements include:

| Profile | Measured ms/step | Interpretation |
| --- | ---: | --- |
| Clean exact #2135 audit graph profile | about 133-137 | Best leaderboard-clean exact band |
| Exact recurrent boundary fusion | 136.65 | Correct but insufficient |
| Graph-side/deferred exact candidate | 133.95 | Best recent exact movement |
| MLP scratch-overlap candidate | 137.06 | Regressed and reverted |

Third, the active/inactive split explains the gap:

| Window | Recent measured range |
| --- | ---: |
| Inactive recurrence | about 117-120 ms/step |
| Active recurrence | about 144-148 ms/step |

That is the core result of Week 5. The non-recurrent or inactive-recurrent graph is effectively at target. The active recurrent path is not.

## The Week 6 Target

The next decisive systems experiment is not another flag sweep. We have enough sweeps.

The target is an exact recurrent replay cut:

- A larger pass2/pass1 recurrent backward fusion.
- Or a persistent-CTA block backward that reduces recurrent replay traffic and launch overhead.
- Or a corrected recurrent approximation that is explicitly treated as an algorithmic variant and validated by full BPB.

For the clean exact path, the performance target is concrete: remove roughly 13-17 ms/step from the active recurrent window without breaking record semantics.

After that, the project has to leave proxy timing and run the full record path:

1. Full 600-second train.
2. Full eval/TTT under the legal score-first constraints.
3. Exported artifact with model bytes, code bytes, total bytes, and hashes.
4. Full-validation BPB.
5. Three seeds.

Only then can we compare against the winning Parameter Golf stack.

## Bottom Line

Week 5 was not about producing the most flattering number. It was about making the number harder to fake.

The stack now has a much clearer boundary between systems-valid timing, speed probes, exact leaderboard-clean profiles, and record-valid results. The inactive recurrent path is essentially at the target. The exact active recurrent path is not. That is the wall.

The next successful week is easy to define and hard to execute: implement a real exact recurrent backward fusion or persistent-CTA recurrent block backward, then prove the full train/eval/export path with BPB and artifact budget in the same run.

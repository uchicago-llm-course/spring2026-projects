# Finding the 14% That Does the Work

*DUET (Discussion Under Equal-compute Training) — foundation reset, week of April 17*

I ran into a lot of challenges this week — building a stronger foundation for the training pipeline and rebuilding the behavioral metrics, chief among them. This post walks through what broke and what I fixed.

Two numbers fell out of instrumenting the trainer this week.

First: only **8.5%** of sampled problems produced a productive disagreement — one player right, the other wrong. On the other ~91%, both players agreed and the discussion branch never fired. Model D was effectively training as Model S on the vast majority of its gradient updates.

Second: only **14.2%** of DAPO problems sit in the solve-rate band where GRPO actually learns — where rollouts within a group mix correct and incorrect outcomes. On the rest, the group advantage is either all positive (all right) or all negative (all wrong), and the gradient washes out.

Both numbers needed fixing before I could defend any headline result, let alone stack mechanistic probes on top of one. I went looking because paired bootstrap CIs had just killed last week's ordering.

## The CI Recompute That Started It

Last blog closed with "D > S > C, discussion wins." Paired bootstrap, 10k resamples on the pairwise differences, MATH-500 Level 5 (n=134):

| Comparison | Δ (pass@1) | 95% CI | Significant? |
|---|---|---|---|
| D − S | +0.007 | [−0.021, +0.036] | No |
| D − C | +0.039 | [+0.007, +0.076] | **Yes** |
| S − C | +0.033 | [+0.002, +0.068] | **Yes** |

"D beats S" on 134 problems is a coin flip. The only ordering I can defend at 95% is *D and S both beat C*. On AIME (n=30) the CIs are wide enough that none of the three pairwise comparisons come out significant. A 0.5pp gap on 134 problems is well inside the pass@1 noise floor — I should have known going in. The bootstrap script is maybe 30 lines of code; it just needed someone to bother running it.

That was the trigger for a full foundation reset. Building probes on top of a result I couldn't defend at 95% felt like the wrong order of operations.

## Phase 0: Diagnosing the Foundation

**Fixing the productive rate.** I swept six Alice/Bob temperature pairs on the same 200 base-model problems:

| (Alice, Bob) | Productive % | Notes |
|---|---|---|
| (0.3, 0.7) | 10.0 | narrower gap, lower mean |
| (0.4, 0.8) | 10.5 | |
| (0.5, 0.9) | 10.5 | |
| **(0.6, 1.0)** | **13.0** | winner |
| (0.7, 1.1) | 9.0 | Bob high enough to start producing malformed rollouts |
| (0.8, 1.0) | 8.5 | old baseline (narrow gap, high mean) |

(0.6, 1.0) wins at 13.0% productive — **+4.5pp over the old (0.8, 1.0) baseline, ~53% relative lift**. A wider gap isn't automatically better: (0.7, 1.1) has the same 0.4 gap as the winner but falls to 9.0%. Push Bob past T≈1.1 and rollouts go off the rails — they look like disagreement in the aggregate but don't carry a useful signal for GRPO. The old (0.8, 1.0) setup was the worst pair I tested.

**The training data is bimodal.** Base Qwen3-4B on 3000 DAPO problems, 8 rollouts each, bucketed by solve rate:

| Band | % of DAPO |
|---|---|
| 0–25% (too hard) | 33.2% |
| 25–75% (informative) | **14.2%** |
| 75–100% (too easy) | 52.7% |

The **14.2%** informative band is where GRPO gets useful gradient signal; the other **85.8%** is compute burn. I built a curriculum filter on the index list where base solve rate sits in [0.25, 0.75]: **543 problems out of 3000 scanned**. Each one gets revisited ~6–8 times across a 200-iter run, and the sampler now draws only from this set.

One design choice I thought hard about: do I refit the filter mid-training? Once Alice improves past the base, the filter goes stale — problems originally at 0.5 base solve rate might shift to 0.8 as she trains. Rebuilding every N iterations would cost a few hundred dollars and a lot of orchestration, so I decided against it for a one-shot retrain. If post-training metrics show the filter is still gating most of the signal at iter 150, I'll revisit. Related decision: I stayed on DAPO-17K rather than swap to a harder prompt pool. The bimodality is a model-dataset pairing issue, not a DAPO problem — switching datasets would shift the band, not eliminate it.

## Phase 1: Rebuilding the Training Pipeline

Wiring both fixes in meant more than a config change.

- **Cosine LR schedule.** Added `lr_decay` and `lr_min_frac` to `InfraConfig`, a `_compute_lr` helper, and a loop patch that updates every optimizer group before each `step()`. Cosine with a 10% floor and a 20-iter warmup should keep the gradient meaningful past iter 100, where the old flat schedule was letting the models degrade.

- **Curriculum filter plumbed through DAPOSampler.** A new `filter_path` param on the sampler, a loader that accepts either a bare index list or `{"indices": [...]}` JSON, and wiring from `training.curriculum_filter_path` down into `init_infra`. The sampler logs filter size each iter for provenance.

- **Structured artifact saving.** Three JSONL streams now write under `<checkpoint_dir>/metrics/` as training runs — per-iter training metrics (loss, LR, pass@1, disagreement counts), every productive-disagreement case with both players' reasoning and ground truth, and raw eval rollouts. This meant extending `StepResult` with disagreement counts, `WorkflowContext` with the last-eval pass@1 so eval results can be joined onto training iters, and a new recorder in the main loop that runs the sympy verifier on both players' majority votes before classifying each case. Payoff: one Phase 1 run produces everything the post-hoc analysis needs, without a separate behavioral eval pass.

- **Phase 0 Modal runner** (`scripts/modal_phase0.py`). Two entry points (`sweep`, `pilot`), each with a batched generate loop, partial JSON commits to the volume after every batch, and resume logic on the set of already-completed work (T-pairs for the sweep, DAPO indices for the pilot). Both jobs had hit Modal's default 2-hour function timeout with zero saved state; durable-state after each chunk turns "misjudged runtime" into "restart and continue."

- **Phase 1 Modal runner** (`scripts/modal_phase1_train.py`). Stages `curriculum_filter.json` from the reports volume into `/checkpoints/phase1/` at run start (the trainer only mounts `/checkpoints`), supports `--resume` that auto-detects DeepSpeed vs. HF checkpoints, and exposes one condition per invocation so sequential runs stay cleanly separated. 2×H100, 24-hour timeout, `adam_offload` so 4B params + vLLM engines fit in 160GB.

The three Phase 1 configs pin *matching* Alice/Bob temperatures across D and C — both at the Phase 0a winner (0.6, 1.0) — so the discussion-vs-compute-control comparison stays clean. All three load the same curriculum filter.

## Phase 2: Fixing the Behavioral Metrics

Last blog flagged three methodology issues in the non-accuracy metrics. Two are getting fixed; one got dropped.

**Adversarial robustness → intersection set.** The old metric only tested each model on its own correct set, so C looked most robust partly because its correct set was biased toward easy problems. Fix: restrict the test to the intersection of problems *all three* models solve, so flip rates are measured on a shared difficulty distribution. Can't compute the intersection until Phase 1 produces D, S, C — but the patch to preserve per-problem adversarial data during eval is already in.

**Diversity → long-context embedder.** The old MiniLM-based metric had a 256-token context, so long rollouts were chunked and mean-pooled. Mean-pooling across chunks washes out mid-chain strategy pivots — a coordinate-to-synthetic switch collapses to roughly the same mean vector as not switching. I swapped in `gte-large-en-v1.5` (8k context) so each full rollout embeds in one pass, and kept the agglomerative clustering on top. My TA also suggested adding a baseline from 8 random rollouts of the untrained base model — a "natural diversity" floor to anchor the trained-model numbers against. Planned, not yet in.

**Error recovery → dropped for this round.** The current heuristic (regex-scanning for wrong `\boxed{}` mid-chain) conflates exploratory reasoning with real errors and scales with total error count. The cleaner fix — labeling intermediate steps with a verifier model — costs real money and adds an external dependency I've chosen to avoid for this project. Leaving this as future work.

## What's Next

Phase 1 launches now. Three conditions sequentially on 2×H100:

1. **Solo.** If the training curve climbs past iter 100 (old runs didn't), cosine LR + curriculum filter are working. If it regresses, stop and diagnose before launching the longer runs.
2. **Discussion.** Longest run. Depends on solo looking healthy.
3. **Compute control.** Confirms the C < S effect survives the new setup.

Post-retrain:
- Recompute bootstrap CIs at matched iter-N across all three models. If D > S is still within noise at n=134, I'll say so — better honest than overclaimed.
- Run the intersection-based adversarial metric and long-context diversity metric (with the base-model diversity baseline).
- Start Phase 2 proper: linear probes on hidden states to see whether discussion training changes how the model represents error recognition, or whether accuracy deltas come from somewhere else.

The lesson from this week: it's easy to write a cleanly-ordered result table and call it progress. It's harder to ask whether the setup that produced the table can support the conclusion. This reset wasn't a week I planned for, but I think it's the only path where the mech work means something instead of sitting on a foundation I don't trust.

# The Verifier That Never Verified

*DTCA Project — Week 4 Update*

Last week's post said the pipeline was working. I had all five text-level reward channels producing non-zero values, DTCA metrics showing up in wandb, Stage 3 probes loading cleanly. This week I pulled sample transcripts and looked at what the Verifier was actually producing. Here's a representative one, on a MATH-500 hexagon problem where the Solver correctly derived 42:

> I don't see any issues here. The Solver's approach is valid and the answer is accurate. `<verify>approve</verify>`

It approved in three sentences with no derivation, no check of the Solver's arithmetic, no independent reasoning. Next turn, Solver shrugs and moves on. This was the median Verifier behavior, not an outlier.

Then I pulled an AIME24 transcript where the Solver got stuck in a tight repetition loop — "Therefore, the blue positions must be a rotation of the red positions. Therefore, the blue positions must be a rotation..." for thirty iterations until the response length cap. The Verifier's `<think>` block then did the same thing, because the Verifier's prompt fed in the Solver's output as context. Neither turn emitted a `\boxed{}` answer. Neither turn was a valid training sample.

## The structural problem

The Verifier's prompt in Dr. MAS asks only for `<verify>approve</verify>` or `<verify>reject</verify>`. There's no requirement to emit a `\boxed{}` answer. The answer parser in DrMAS requires `\boxed{}` for a turn to count as valid. So across every 4-turn debate (Solver, Verifier, Solver, Verifier), the answer trajectory was:

```
[A_solver, None, A_solver', None]
```

DTCA's whole pitch is a multi-granularity reward over this trajectory. R_correct rewards answer-flips across consecutive turns. With `None` on every Verifier turn, flips can only happen between turn 0 and turn 2 — both Solver turns — which is rare because Solvers don't usually self-revise. R_conv (trajectory stability) reduces to measuring whether the Solver is self-consistent. R_disagree uses sentence embeddings over full turns, but `<verify>approve</verify>` is three words and carries zero embedding distance to anything. R_sycophancy checks whether agents copy each other's reasoning when they agree, but there's no Verifier reasoning chain to compare against.

So: of five text-level channels, one (R_outcome) was fine, and four were either dead or operating on half-empty inputs. Every validation checkpoint this session prior to the Protocol v2 fix read `correction_rate: 0.000` and `regression_rate: 0.000`. Last week's "all five channels firing" was measuring channel *presence*, not channel *signal*. The signal was structurally zero.

## The recurring silent-bug class

The fix requires the Verifier to emit both a verdict and a `\boxed{}` answer. But that's a prompt change, and the Verifier's prompt lives in the Dr. MAS fork, and there are two places in DrMAS that need to know about any new reward channel: the reward manager's hardcoded `channel_names` list, and the prompt template itself. This week produced two bugs of the same class as the `dtca_rewards_enabled` forwarding bug from Week 3:

1. **The prompt didn't ask for a `\boxed{}`.** Verifier output was syntactically valid but contained nothing that could populate the answer trajectory. Fixed by rewriting the prompt: Verifier now walks through the Solver's reasoning, forms its own conclusion, and emits both `<verify>approve|reject</verify>` and `\boxed{YOUR_ANSWER}`. If approving, the boxed answer must mirror the Solver's; if rejecting, it's the Verifier's correction.

2. **A new reward channel I added was silently dropped.** I created `R_verdict` (scores whether the `<verify>` verdict matches the ground-truth correctness of the Solver's answer), wired it into `compute_dtca_rewards`, and added 20 unit tests. It fired correctly in tests but a recent smoke test logged `reward_extra_infos_dict.keys()` and the key `dtca_verdict` was not there. DrMAS has a hardcoded `channel_names` list in `EpisodeRewardManager`, and any key not in that list is computed and then discarded before reaching the GDPO dispatcher. Same failure mode as the Week 3 flag-forwarding bug.

Both now have regression guards that parse `patches/drmas-dtca.patch` directly and fail if the registration or forwarding is missing.

## How R_verdict got designed

The Verifier-produces-no-answer problem wasn't obvious to me, so I didn't want to guess at the fix and did some research.

The decisive finding was that J1 (Meta, arXiv 2505.10320) and Self-Verify (arXiv 2506.01369) both use verdict-accuracy-against-ground-truth as the RL signal for training an LLM judge: `r = 1{verdict matches correctness}`. Critique-GRPO (arXiv 2506.03106) does the same in a critique-refine loop. None of these combine that signal with a trajectory-based multi-channel curriculum in multi-agent debate. That combination is the DTCA contribution.

The critical design choice is anti-collapse: R_verdict rewards the verdict (approve/reject), NOT the Verifier's boxed answer. The boxed answer exists only to populate the trajectory for the other channels. Without this asymmetry, the Verifier converges to "ignore the Solver, re-derive the problem, produce your own correct answer" — the second-Solver failure mode documented in CTRL (arXiv 2502.03492) and Critique-RL (arXiv 2510.24320). This is also why the newly designed R_verdict is responsive to last week's feedback about reasoning-chain noise: R_verdict grades a single discrete token, not an embedded reasoning chain. Clean per-turn signal.

## What the real baseline looks like

While pulling transcripts I also noticed that the models were consistently being truncated - `max_response_length` was set to 1024 tokens, which is fine for short answers but cuts Qwen3-4B's `<think>` blocks off mid-step on roughly half of MATH-500 problems. Raising it to 2048 (with the necessary bumps to `max_prompt_length` and `max_num_batched_tokens` to accommodate the coupled constraints from the 4-turn debate), the real MATH-500 base accuracy is **0.66**, not **0.36**.

## Where things are now

New Verifier prompt, new R_verdict channel (with per-turn attribution and GDPO normalization), updated curriculum, new channel list reads:

```
dtca_outcome, dtca_error_correction, dtca_convergence,
dtca_disagreement, dtca_sycophancy, dtca_verdict,
... plus per-turn versions and behavioral flags
```

Six reward channels firing with per-turn attribution, all reaching the GDPO dispatcher.
On a 5 step smoke test:

| Metric | Step 0 | Step 5 |
|---|---|---|
| `val/dtca/correction_rate` | 0.000 | **0.009** |
| `val/dtca/regression_rate` | 0.000 | 0.000 |
| `val/dtca/echo_rate` | 0.385 | 0.454 |

`correction_rate` moved off zero for the first time! `regression_rate` holding at zero is the other half of the validation: the Verifier isn't flipping correct Solver answers to wrong ones to earn reward, which means the anti-collapse asymmetry (R_verdict rewards the verdict, not the boxed answer) is working as designed.

## On the paper story

Reviewer feedback last week flagged that since Verifier and Solver share a base model, a natural baseline is prompting the same model to iteratively solve-then-verify without the multi-agent framework. I agree with this framing - If DTCA beats that baseline, the multi-agent framing is load-bearing; if it doesn't, we're just doing clever single-agent prompting with extra steps. I'm adding "single-model self-solve-then-self-verify" as a new ablation row. R_verdict's role asymmetry makes this a genuine test: the Verifier now gets a different training signal than the Solver, which it wouldn't in a single-agent self-prompting setup.

The revised ablation structure is also cleaner as a 2×2 grid of channels: Solver-targeted vs Verifier-targeted × outcome vs process. R_verdict is the new Verifier-side process channel. R_hidden sits across both axes at Stage 3. The multi-granularity claim becomes empirically testable row-by-row instead of a blanket pitch.

## What comes next

A 30-step single-arm dtca-flat run on Modal. If `correction_rate` grows across 30 steps the way it moved from 0 to 0.009 in 5, there'll be a meaningful signal to write up. If it plateaus or reverses, I'll look at per-channel advantage magnitudes in wandb to figure out which channels are saturating or fighting each other, and triage from there. The cluster A/B (DTCA-flat vs GRPO-baseline, multiple seeds, full epoch count) is the eventual paper-scale experiment, contingent on the Modal single-arm trend staying positive and the cluster disk issues getting sorted.

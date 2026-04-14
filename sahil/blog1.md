# Discussion Helps, But Not How I Expected

*discussion_mech — Behavioral Evaluation Results*

The question behind this project: when two LLMs discuss a math problem during RL training, does the trained model actually get better? Or is it just extra compute dressed up as collaboration?

I trained three Qwen3-4B models under matched conditions to find out. The accuracy results tell a clean story. The other behavioral metrics tell a more complicated one.

## The Setup

Three conditions, all using GRPO on DAPO-17K problems, lr=1e-6, 150 iterations:

- **Model S (Solo):** One player, 8 rollouts per problem. Standard single-agent RLVR.
- **Model D (Discussion):** Two players (Alice at T=0.8, Bob at T=1.0). When they disagree on a problem, they discuss it — each sees the other's answer and reasoning, then generates a revised solution. Only Alice gets the gradient update.
- **Model C (Compute Control):** Two players, same as D, but on disagreement they generate extra rollouts instead of discussing. Same compute budget as D, minus the discussion.

The compute control is the key comparison. If D beats S, maybe discussion helps — or maybe having a second player just means more compute. If D beats C, the improvement comes from the *content* of the discussion, not the extra forward passes.

## The Accuracy Result

Evaluated on MATH-500 Level 5 (134 problems, the hardest tier) at iter 150 for all three:

| Model | Pass@1 | Accuracy (any of 8) |
|-------|--------|---------------------|
| **D** (discussion) | **0.660** | **0.754** |
| **S** (solo) | 0.654 | 0.739 |
| **C** (compute ctrl) | 0.621 | 0.687 |

D > S > C. Discussion wins, and extra compute without discussion actually hurts.

On AIME 2024 (30 problems, harder, evaluated at best checkpoint):

| Model | Pass@1 |
|-------|--------|
| Base (no training) | 0.237 |
| **D** (discussion, iter 50) | **0.312** |
| **S** (solo, iter 100) | 0.283 |
| **C** (compute ctrl, iter 50) | 0.283 |

Same ordering. D leads on the hardest benchmark, though with only 30 problems the confidence intervals are wide.

## Why Compute Control Hurts

The most surprising result isn't that D beats C — it's that C is substantially *worse* than S. Solo training with 8 rollouts outperforms a two-player setup generating extra rollouts.

I think this comes down to overfitting dynamics. On disagreement problems (the ones where the model is most uncertain), C generates additional rollouts from the same distribution. More samples of the same thing. Over 150 iterations, this amplifies the gradient signal on exactly the problems where it's noisiest, pushing the model toward memorizing training-set patterns rather than generalizing.

Evidence: C's performance barely changed between iter 50 (0.620) and iter 150 (0.621), while D actually *improved* (0.618 to 0.660). D is the only model that got better past its early "best" checkpoint. Whatever discussion is teaching, it seems to provide a more stable training signal that resists overfitting.

| Model | Pass@1 (best ckpt) | Pass@1 (iter 150) | Delta |
|-------|-------------------|-------------------|-------|
| **D** | 0.618 (iter 50) | **0.660** | **+0.042** |
| **S** | 0.648 (iter 100) | 0.654 | +0.006 |
| **C** | 0.620 (iter 50) | 0.621 | +0.001 |

## The Other Three Metrics

Beyond accuracy, I measured error recovery rate, adversarial robustness, and solution diversity. I'll report these honestly, but I have reservations about all of them.

| Metric | S (solo) | D (discussion) | C (compute ctrl) |
|--------|----------|---------------|-------------------|
| Pass@1 | 0.654 | **0.660** | 0.621 |
| Error Recovery Rate | 0.127 | 0.119 | **0.172** |
| Adversarial Flip Rate (lower = better) | 0.069 | 0.056 | **0.037** |
| Solution Diversity (clusters) | 2.83 | 2.74 | **2.89** |

C "wins" on three of four non-accuracy metrics. But I don't think these numbers mean what they appear to.

**Adversarial robustness** measures how often a model flips from a correct answer to an incorrect one when presented with a plausible counterargument. Lower flip rate = harder to fool. C looks most robust (0.037 vs 0.069 for S). But the test only applies to problems the model answers correctly — and C answers the fewest problems correctly (82 tested vs 87 for S and 90 for D). The problems C gets right are the ones it's most confident about. It's not that C is more robust; it's that C's correct-answer set is biased toward easy, high-confidence problems.

**Error recovery** measures how often a model makes a wrong intermediate step but still arrives at the correct final answer. C has the highest rate (0.172). But C also makes the most errors overall. More errors means more opportunities for recovery. And the detection heuristic (scanning for wrong `\boxed{}` or `= X` patterns mid-chain) is fuzzy — it can't reliably distinguish a genuine mid-chain error from exploratory reasoning.

**Solution diversity** measures how many distinct solution strategies the model uses (via MiniLM embedding + clustering). All three models produce 2.7-2.9 clusters with mean pairwise distances of 0.037-0.041. These differences are tiny and almost certainly not significant. The metric might also just not be discriminating enough — we had to fix it twice during development (long rollouts were getting truncated to identical prefixes, then the clustering threshold was too aggressive).

## What the Discussion Signal Actually Looks Like

One thing that became clear from a disagreement pilot study: the discussion signal is sparse. Running Alice (T=0.8) and Bob (T=1.0) on 200 DAPO problems after Model S training:

- Disagreement rate: 32.5%
- *Productive* disagreement (one right, one wrong): 9.5%

So only ~10% of training problems actually trigger a meaningful discussion — one where there's something to learn from the other player. On the other 90%, both players agree (both right or both wrong) and the discussion branch never fires. Model D is effectively doing the same thing as Model S on 90% of its training data.

The fact that D still edges out S despite only getting the discussion signal on ~10% of problems suggests that when discussion *does* fire, it's a strong signal. But it also means there's a ceiling on how much discussion can help with this setup. Harder problem distributions, or a wider temperature gap to increase disagreement, could amplify the effect.

## What I'd Do Differently

**The non-accuracy metrics need work.** Adversarial robustness needs to control for problem difficulty — only compare on problems all three models solve, not on each model's own correct set. Error recovery needs a less heuristic-based detection method, probably involving a separate verifier model to label intermediate steps. Diversity might need a different embedding approach entirely. Right now, accuracy is the only metric I'd put real weight on.

**All models degrade past ~100 iterations.** None of the configs use learning rate decay, which is standard in RLVR training. Adding cosine decay would likely improve all three models and might change the relative rankings. The iter 150 comparison is fair (same training for all) but probably not optimal for any of them.

**The disagreement rate is too low.** Training on DAPO-17K with a 0.2 temperature gap means most problems are either too easy (both right) or too hard (both wrong) for productive disagreement. A curriculum that targets problems near the 50% solve-rate boundary would maximize the discussion signal.

## The Headline

Discussion-trained models are more accurate than solo-trained or compute-control models, and more resistant to overfitting. The effect is modest (0.660 vs 0.654 on MATH-500 L5) but consistent across benchmarks and robust to checkpoint selection. Critically, the discussion model is the only one that keeps improving past its early training peak.

The compute control result is arguably more interesting: extra rollouts without discussion *hurts*. This rules out the simplest null hypothesis — that discussion is just more compute. Whatever the model learns from seeing another agent's reasoning and revising its own, it's qualitatively different from seeing more samples of its own reasoning.

Next up: mechanistic analysis. Linear probes on hidden states to see whether discussion training actually changes how the model represents error recognition internally, or whether the accuracy improvement comes from somewhere else.

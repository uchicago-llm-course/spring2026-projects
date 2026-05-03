# The Debate Got Worse. The Solver Got Better.

*DTCA Project — Week 5 Update*

Last week ended with a 5-step smoke test where `correction_rate` moved off zero for the first time. I took that as a green light and went ahead with three training runs on modal - this post details what those three runs taught me and showcases a result I wasn't looking for that might end up being the true value of this project.

## The two runs that blew up

The first run completed but the generations had gotten visibly weird by the end: long token-repetition loops, entropy dropping toward zero. The second run had reward fixes (class-reweighted `R_verdict`, the counterfactual `R_verifier_value` channel, `R_sycophancy` disabled) and did the same thing, just faster. Both looked like textbook unregularized policy gradient divergence.

The cause was one line per run. Both had:

```yaml
actor.use_kl_loss: False
algorithm.use_kl_in_reward: False
```

veRL gates the reference policy worker on those flags. With both false, it doesn't even instantiate a RefPolicy, so nothing stops the actor from walking away from the base distribution. I'd been staring at a "standard GRPO recipe" with the anchor removed.

I don't have a good excuse for why KL was off. I think I disabled it during debugging weeks ago and forgot to turn it back on. Same class of bug as the Week 3 flag-forwarding incident, except this time I did it to myself.

## The third run

For  this run I bundled the obvious fixes: KL on, the sympy-aware `answer_is_correct` helper across all reward channels (BUG-1, ~5-7pp of silent false negatives on LaTeX equivalents like `\dfrac{1}{2}` vs `\frac{1}{2}`), `ErrorCorrectionReward` tracking the last parseable answer across `None` turns (BUG-3), a `preservation_bonus=0.2` so the Solver gets a small positive signal for keeping a correct turn-0 answer correct, `repetition_penalty` bumped 1.1 → 1.15, and `R_sycophancy` permanently disabled per [Peacemaker or Troublemaker](https://arxiv.org/abs/2509.23055).

The run completed cleanly. Then I evaluated on MATH-500:

| | base Qwen3-4B | run1 30-iter | run3 30-iter |
|---|---|---|---|
| MATH-500 full debate (turn 3) | **0.754** | 0.738 | **0.574** |

The newest, best-regularized, bug-fixed run was the worst debater on the board.

## What happened

Pulling transcripts made it obvious. 53.4% of Verifier turns in the stable run had no `<verify>` tag at all. Approval rate on the turns that did emit a tag was 96.5%. The Verifier had learned that the cheapest policy was either skip the verify tag (eats a `R_verdict` penalty but is apparently dominated by Solver-side channel signal) or stamp `approve` indiscriminately. Mode collapse in a different costume.

The base-rate arithmetic is the thing I underweighted in `R_verdict`'s design. On MATH L3-5 the Solver is correct 65-75% of the time, so always-approve has expected reward around +0.5 before reweighting kicks in. The class reweighting is correct *conditional on emitting a tag*, but it can't price in the cheaper alternative of dropping the tag entirely if that loss is dominated by the Solver-shared policy's reward from other channels. Which apparently it is.

## The thing I wasn't looking for

`R_outcome` cares about the *final* answer, and the Solver writes the final answer first. Halfway through staring at the stable run's eval I realized I'd never pulled out the turn-0 number in isolation. So I re-scored every transcript with sympy-aware comparison, split by turn:

**MATH-500 turn-0 (Solver single-shot) vs base, stratified by difficulty:**

| level | base | run1 30-iter | run 2 40-iter | run3 30-iter |
|---|---|---|---|---|
| L1 | 0.884 | +0.0 | -2.3 | +0.0 |
| L2 | 0.778 | +5.6 | +0.0 | +3.3 |
| L3 | 0.610 | +11.4 | +13.3 | +12.4 |
| L4 | 0.422 | +18.8 | +23.4 | +18.8 |
| L5 | 0.291 | +14.9 | +20.9 | +8.2 |
| **overall** | **0.530** | **+12.2** | **+14.2** | **+10.2** |

Every DTCA checkpoint I have, including the one whose Verifier collapsed into format failures, produces a turn-0 Solver that is 10-14 points stronger on MATH-500 than the base. The gain *grows with difficulty*: saturated at L1, modest at L2, double-digit at L3, ~20pp at L4. This is the difficulty-stratified picture the proposal was betting on, and it's sitting on the Solver's first turn, not on the debate.

Turn-3 meanwhile is flat. The debate protocol isn't making things better, but the Solver being pushed through a debate-shaped reward landscape is learning something that transfers to its single-turn performance.

## Does it transfer OOD?

Training was on MATH levels 3-5. AIME 2024 and 2025 are competition math, never seen at training time. Same domain, harder problems. Re-scoring those transcripts:

**AIME turn-0 single-shot, all-correct with sympy:**

| run | AIME24 T0 | AIME25 T0 |
|---|---|---|
| base Qwen3-4B | 0.000 | 0.033 |
| run1 30-iter | 0.033 | 0.033 |
| run2 40-iter | **0.100** | **0.100** |

Directionally the same picture: turn-0 capability lifts on harder OOD problems, the gain compounds between step 30 and step 40. Caveat is n=30 per year, so +0.067 means two more problems and the 95% CI on a proportion of 0.10 with n=30 is roughly [0.02, 0.27]. Pointing the right direction, not statistically tight. True cross-domain OOD (GPQA-Diamond, non-math) needs the cluster.

Turn-3 on AIME is flat against base across all DTCA checkpoints, same as MATH-500.

## The reframe

The honest reading of all of this is:

> DTCA trains a multi-agent debate, but what it produces is a better single-turn Solver. The multi-agent reward signals act as a distillation channel from the debate-shaped training into first-turn capability. The full debate protocol's accuracy is a wash against base; the Solver alone, measured at turn 0, is materially stronger on both in-distribution (MATH-500) and harder out-of-distribution (AIME) math, with gains scaling in difficulty.

This made me curious about other research with similar framing: [SPIN](https://arxiv.org/abs/2401.01335) and [Self-Rewarding LMs](https://arxiv.org/abs/2401.10020) both use multi-role scaffolds at training time to improve single-model capability at inference time. DTCA could be read the same way: the Verifier's job is to shape the training signal for the Solver, and the artifact you deploy is the Solver running alone.

The reframe pivots what the paper/project has to prove. Old pitch: "our debate beats the baseline debate," needs turn-3 numbers to move. New pitch: "our training recipe produces a better single-agent Solver than outcome-only RL on the same base, with gains scaling in difficulty," needs a different baseline. **Single-agent GRPO at matched compute** becomes the gating experiment. If DTCA's Solver beats a GRPO-trained Solver at turn 0, the multi-agent scaffolding is doing real work. If it doesn't, we're doing clever single-agent RL with extra steps.

## What comes next

I'm out of modal credits so the next thing is porting training to the DSI cluster and running single-agent GRPO on Qwen3-4B at matched compute. I'm hoping DTCA will still have some merit but I suspect standard GRPO will manage to achieve a similar effect. 

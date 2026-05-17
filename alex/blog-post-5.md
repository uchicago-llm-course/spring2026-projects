# Self-Refine Already Did This

*DTCA Project — Blog 5*

My previous blog post ended with a question of whether single-agent GRPO would close the gap between the base model and the DTCA RL. I ran the GRPO baselelin and  RLVR closed most of the *single-shot* gap (base 4B at 0.544 → solo-GRPO at 0.668 on MATH-500), but both checkpoints converged to the **same** 0.858 once wrapped in a four-turn debate. So the pre-registered gate clears: scaffold beats matched-compute self-consistency by ~10pp on top of either checkpoint.

The framing has somewhat pivoted since then:
## The four new arms

- **F. Self-Refine.** Single role, four turns alternating *generate* and *critique-your-own-prior-answer*. Canonical Madaan et al. 2023, prompts verbatim. No Verifier role, no asymmetric framing.
- **G. Symmetric MAD.** Two "agents" both running the same Solver prompt for four turns. Closest to Du et al. 2023.
- **H. LLM-judge best-of-k.** k=4 parallel Solver samples, then one Judge turn picks the best. Generative selection over parallel candidates.
- **I. Long-CoT single-stream.** One generation at max_tokens=8192. No turn structure and no critique, so if removing the turn structure preserves most of the gain, the "two rounds" framing was never load-bearing.

Plus GPQA-Diamond ($n=198$, science MCQ) for variation outside math.

## What the matrix says

**Solo-GRPO step-30, MATH-500 overall**, post-rescore:

| A | D (SC k=4) | E (pass@4) | H | G | I (long-CoT) | F (Self-Refine) | B (debate) |
|---|---|---|---|---|---|---|---|
| 0.668 | 0.752 | 0.798 | 0.796 | 0.812 | 0.818 | **0.854** | **0.858** |

**F = 0.854 vs B = 0.858.** Self-Refine, with no Verifier role and no role asymmetry, matches the role-asymmetric debate inside the bootstrap CI. This holds on every $n=500$ MATH cell on every Qwen-family checkpoint I ran (1.7B base, 4B base, 4B solo-GRPO).

Long-CoT (arm I) sits one point behind F at 0.818, same token budget, just delivered as one long stream rather than four turns. The pass@4 oracle sits at 0.798. Whatever the scaffold is doing, any form of sequential conditioning — explicit critique loop, self-critique, or just more tokens in one go — captures most of it. The question of whether the two rounds helps as "thinking" is answered through the I-vs-A gap (+15pp, no role structure at all) says: yes, substantially.

## Where the role asymmetry actually matters

GPQA-Diamond, post-rescore, Qwen3-1.7B base:

| D | F (Self-Refine) | B (debate) |
|---|---|---|
| 0.187 | 0.293 | **0.369** |

B beats F by 7.6pp with non-overlapping CIs. Same direction on Qwen3-4B base (B − F = +5.6pp, lower bound +0.5pp). On the solo-GRPO checkpoint they converge. So the multi-agent literature's structural commitment to role asymmetry IS load-bearing — but only on science MCQ, only at base checkpoints, not on math at all. The honest read is "sequential critique scaffolds are universal; role asymmetry only matters on science MCQ at base."

New abstract leads with that, and §5's result table runs arm-by-arm rather than checkpoint-by-checkpoint so F-vs-B is the first thing you see.

## Interesting note/limitation

Cross-family generalization stops at one negative result: on Llama-3.1-8B-Instruct the scaffolds *lose* to matched-compute SC on both math and GPQA (B − D = −3.4pp math, −9.6pp GPQA), which restricts the headline to Qwen-family thinking-mode models on empirical grounds, not precaution. And in terms of domains beyond math + science MCQ I haven't fully uncovered everything here. GPQA is the cross-domain datapoint; code, agentic, and instruction-following tasks would tighten the claim further.

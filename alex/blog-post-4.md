# Sitting With the Results Before Burning Cluster Hours

*DTCA Project — Week 6 Update*

Last week's post ended with the single-agent GRPO run at matched compute as the obvious next thing. That's still the plan, but before launching anything I spent some time re-reading the literature, re-checking the headline numbers under a stricter test, and shipping the format-collapse fix the dtca-flat-stable run obviously needed.

## Closer prior art than I'd cited

The reframe in blog 3 leaned on SPIN (arXiv 2401.01335) and Self-Rewarding LMs (arXiv 2401.10020) as precedent for "multi-role training scaffold, single-agent inference." Those are valid but distant. Going back through the 2025-2026 stack, there's much closer work I'd missed:

- **IMAD: "Internalized Multi-Agent Debate"** (arXiv 2604.24881). Two-stage SFT + RL that distills multi-agent debate into a single model, explicitly motivated by "matches multi-agent at 6-21% of inference cost." This is the explicit precedent for what I was about to claim as the DTCA contribution.
- **AgentArk** (arXiv 2602.03955, Feb 2026). Three-stage debate → filter → distill into single agent. Same shape.
- **Chain-of-Agents** (OPPO PersonalAI Lab, 2025). Distills multi-agent trajectories into single-model "agent foundation models" via SFT then agentic RL.
- **"Rethinking the Value of Multi-Agent Workflow: A Strong Single Agent Baseline"** (arXiv 2601.12307). The adversarial citation. Argues homogeneous multi-agent workflows are matchable by a single agent given a longer prompt.
- **"Stop Overvaluing Multi-Agent Debate"** (arXiv 2502.08788) and **"Single-Agent LLMs Outperform Multi-Agent Systems on Multi-Hop Reasoning Under Equal Thinking Token Budgets"** (arXiv 2604.02460). Same critique from the eval side.

The closest neighbor on the design side is **Critique-GRPO** (arXiv 2506.03106): critique during training, greedy at test time. Its turn-0 gains on Qwen2.5-7B MATH-500 (60.8 → ~78) come from training-time-only critique. Among the methods I'd been comparing against (MAPoRL, AT-GRPO, MAGRPO, Stronger-MAS, CRM, MARTI, J1, Self-Verify, Critique-GRPO), none report turn-0 / single-agent transfer as a primary finding, so the *pattern* (multi-agent debate scaffold producing single-agent capability gain) is novel for that family. But the magnitude isn't.

## How likely is single-agent GRPO to close the gap?

The longer I sat with the +10-14pp turn-0 result, the more I worried it isn't specifically multi-agent. Three pieces of evidence point that direction:

- **"Limit of RLVR"** (arXiv 2504.13837, NeurIPS 2025). RLVR improves pass@1 but does not expand reasoning capacity beyond the base. Gains come from re-weighting paths the base already samples.
- **RLVR with random rewards on Qwen2.5-Math-7B** gained +21pp on MATH-500 with literally random rewards, vs +29pp with ground truth. Qwen-family is RL-sensitive in ways that may not be specific to good signal.
- **SimpleRL-Zoo and Open-Reasoner-Zero** report +10-20pp on GSM8K+MATH with 8k-example rule-based GRPO on Qwen2.5-1.5B/7B/14B. The +10-14pp DTCA delivers is squarely in this band.

If single-agent GRPO at matched compute closes the gap, the multi-agent contribution collapses and the paper becomes "another GRPO recipe on a 4B model with a clever ablation table."

## Re-checking the difficulty story

Blog 3 framed the per-level pattern as "monotonic in difficulty." I bootstrapped the per-level deltas (95%, 2000 resamples per level, n=43-134 per level, averaging across the three DTCA runs as seeds) and the picture is more nuanced:

| level | n | base | mean Δ | 95% CI | reads as |
|---|---|---|---|---|---|
| L1 | 43 | 0.884 | -0.008 | [-0.039, +0.023] | null (ceiling) |
| L2 | 90 | 0.778 | +0.030 | [-0.030, +0.089] | null |
| L3 | 105 | 0.610 | +0.124 | [+0.051, +0.200] | significant |
| L4 | 128 | 0.422 | **+0.203** | **[+0.133, +0.279]** | significant, peak |
| L5 | 134 | 0.291 | +0.147 | [+0.082, +0.214] | significant |

The corrected story is "concentrated in the mid-difficulty band (L3-L4), with L5 picking up some of the gain but high cross-run variance." Less clean than monotonic, but still material.
## The format-penalty fix shipped, and it works at the right scale

Blog 3 described the format-collapse pathology in (53.4% missing `<verify>` tags, 96.5% approval on the rest). The fix landed: a new "raw per-turn" reward channel that bypasses GDPO per-channel normalization and is added directly to per-turn advantages. Magnitude is -1.0 on Verifier turns missing `<verify>` or `\boxed{}`.

I worried the raw bypass would dwarf normalized channel gradients by an order of magnitude. So I ran the patched reward function over Run 3's transcripts and measured. Per-rollout penalty mean is -1.07 with std 0.76. Normalized channel stds are R_outcome=0.49, R_error_correction=0.57, R_verdict=0.59, R_verifier_value=0.57. The format penalty is comparable in scale, not dominant. Architecturally sketchy (it bypasses the GDPO invariant), empirically fine.

I also closed the empty-`\boxed{}` exploit. The extractor now returns None for whitespace-only box content, and the format-penalty check requires non-empty raw answer text. Currently zero impact (4000 verifier turns historically, 1 empty-box instance, none triggered the exploit) but the gap is closed before RL pressure finds it.

## Current state

I haven't actually been able to run the single-agent GRPO due to some issues on the cluster (long wait time -> cluster-wide issue and reboot -> now it seems that my account is actually removed from the cluster so I emailed techstaff). I will run it asap and be clearer on the direction of this work (what can be salvaged in the case where single-agent GRPO is simply better).

# Blog 4

## Summary of Prior Work

Last week I confirmed that under matched compute and hidden evaluation, no conversational multi-agent protocol reliably beats a strong single-agent baseline. The only protocol whose aggregate performance was statistically indistinguishable from the baseline was a Mixture-of-Agents (MoA) configuration in which proposers generate independently and never see each other's outputs. A controlled 2x2 ablation found that backbone diversity, meaning using different model families rather than copies of one model, was the only factor that moved the score. The synthesis step contributed nothing measurable. The result indicates that a group becomes epistemically weaker when sharing information erases the independence that made the group useful. I drafted a paper regarding these results. Because it was largely a negative result, I decided to do further tests and analysis over the past week.

## Central Claim

My central claim this week is that the diversity benefit is generated at the proposal stage and consumed by interaction. Diverse backbones produce genuine task coverage that no single model possesses, because each backbone performs well on different problems. Text similarity, score trajectories, and synthesizer outputs all show that when the models interact, the coverage collapses instead of being exploited. The only configuration that consistently outperforms the strong single-agent baseline runs three diverse models in parallel and keeps the highest-scoring output without any coordination, and anything more elaborate spends the diversity budget without recovering it. This week's main work was identifying where in the protocol the loss happens and what the synthesizer is literally doing on the runs where synthesis is part of the protocol.

The positive framing of this result is that diverse independent generation is the optimal strategy for a practitioner who does not know in advance which model will perform best on the task. If you know which backbone is best for the specific problem, it is better to use that model directly. If you do not, a diverse team with argmax selection is equivalent to oracle model selection from your available pool. 

## Per-Model Coverage

The first thing to rule out is that backbone diversity is just access to whichever model happens to be best. If GPT-4o is the strongest backbone available, any team containing GPT-4o will score well, and labeling that a "diversity benefit" misattributes single-model performance to team composition.

The per-model Best-of-N scores do not support that interpretation, because each of the three backbones wins on exactly two of the six tasks reported below.

| Task | Claude | Gemini | GPT-4o | Best model |
|---|---|---|---|---|
| Circle Packing | **1.000** | 0.745 | 0.519 | Claude |
| TSP-50 | **0.141** | 0.115 | — | Claude |
| Difference Bases | 0.209 | **0.610** | 0.264 | Gemini |
| Flat Polynomials | 0.000 | **0.143** | 0.007 | Gemini |
| TSP-100 | 0.013 | 0.010 | **0.021** | GPT-4o |
| Erdős Overlap | 0.131 | 0.142 | **0.710** | GPT-4o |

On Circle Packing, Claude scores 1.000 to GPT-4o's 0.519. On Erdős, GPT-4o scores 0.710 to Claude's 0.131. A GPT-4o-only team would recover the Erdős result and give up more than half the available score on Circle Packing, but a Claude-only team would do the inverse. The best model depends on the task, and there is no obvious structural feature predicting which one wins. A diverse team that runs all three independently and keeps the best result produces coverage no single-backbone team can match. This is why the diverse+selection combination in the ablation study dramatically outperformed same-model+selection on Erdős, and why the factorial coefficient on backbone diversity is positive while the coefficient on synthesis is not.

## Why Diverse Selection is Optimal Under Task Uncertainty

The table above establishes that best model identity is task-specific and unpredictable. No structural feature of the task reliably predicts which backbone will perform best.

A practitioner who cannot observe task-model affinity in advance forecloses on every task where a chosen single backbone loses. GPT-4o gives up 0.481 Q-points on circle-packing relative to the best available score (1.000 vs 0.519). On Difference Bases it gives up 0.346 to Gemini. No number of GPT-4o samples bridges either gap because the missing capability is not in the pool. The regret is structural, not statistical. The diverse team with argmax selection approximates oracle model selection from the available pool. Each task's score is determined by whichever backbone is strongest, without prior knowledge of which one that is.

The full-benchmark test is running across all nine tasks. The per-model table predicts GPT-4o wins Erdős and TSP-100 and fails on circle-packing, Difference Bases, and Flat Polynomials; Gemini×3 produces the inverse pattern; the diverse team wins no individual task but never catastrophically fails. The benchmark mean is the right scope for the comparison.

## Output Similarity Across Stages

The coverage benefit only matters if three different outputs reach the final answer stage intact. They usually do not.

Pairwise cosine similarity between three diverse-backbone proposers' raw text averages 0.433 when they generate independently. Eight samples from a single backbone average 0.745. Diverse proposers are meaningfully more distinct at the proposal stage. The collapse happens once agents read each other. Homo-chain step-to-step similarity averages 0.840. The chain architecture is nearly copying with minor edits, the quantitative version of what the trajectory data shows.

Debate is less extreme but moves in the same direction. Diverse-debate step-to-step similarity is 0.728 at the first exchange and 0.619 by the final step. Same-backbone debate moves the opposite way, from 0.646 to 0.717. The same-backbone case converges faster, consistent with shared inductive biases pulling agents toward the same answer.

## Score Trajectories

The trajectory data localizes the score collapse to the interaction step itself. On Difference Bases (lower is better), debate with diverse backbones moves 436 → 132 → 66 across rounds. Homo-chain with same-backbone Claude moves the opposite direction. The mean score starts at 276 at step one, rises to 471 at step two and 721 at step three, and partially recovers to 313 at step four. The recovery does not return to round one. Each chain step reads the prior agent's output and produces something nearly identical except for compounded errors.

Erdős shows a different and more revealing failure. Debate diverse produces a mean round-three score of 0.62 across seeds and improves on average across all three rounds. The within-seed variance is extreme. Seed three starts at 19.79, drops to 2.48 at round two, and the round-three synthesizer outputs a score of 0.25. Round two was good. Round three destroyed it.

The most direct test of synthesis is to compare the final synthesizer output to the best intermediate round. On Erdős, the best intermediate debate round has a Q-normalized score of 0.355 and the final debate output has a Q-normalized score of 0.142. The synthesis step that was supposed to combine the best of both debaters instead cut performance by more than half. On every other task, synthesis is neutral. On the one task where diverse agents have the most to offer each other, synthesis is the mechanism of failure.

## Synthesizer Behavior

Categorizing synthesizer outputs by whether they improved on, matched, or degraded the best proposer reveals a sharp split. On three of seven tasks (Flat Polynomials, TSP-100, TSP-50) the synthesizer copies the best proposer on every run. On two more (Erdős, Molecule QED) it copies 80% of the time. Across these five tasks the synthesizer identifies the highest-scoring proposal and reproduces it with minor edits. The synthesis step adds no value but also causes no harm. The exception is Difference Bases, where synthesis actively hurts 50% of the time. The synthesizer reads three combinatorial sets and attempts to construct a fourth incorporating elements from all three. Combinatorial structures do not interpolate, and the novel set it constructs is usually worse than any of the inputs.

In MoA diverse on Erdős, seed one, Claude's proposer produces a structured binary function with a visible score of 0.49. GPT-4o's proposer outputs a flat uniform array with a visible score of 0.25. The synthesizer reads both, identifies the scores, notes that Agent 1 scored higher, describes both approaches, and selects the uniform distribution. Its stated reasoning is that the uniform distribution achieves a known theoretical minimum. The score drops from 0.49 to 0.25. The diversity existed. The synthesizer saw it. It picked the worse answer.

In debate diverse on Erdős, seed three, the round-two agent produces an indicator function scoring 2.48, a meaningful improvement over round one (19.79). The round-three synthesizer reads both agents' outputs. It states that the first agent's sparse distribution concept is valuable, that the second agent's rectangular window approach has merit, and that combining both would be optimal. It then outputs a constant function. The score drops to 0.25. The diverse team found something useful at round two. The synthesis step erased it.

In cross-chain on Difference Bases, seed one, the first agent in the rotation produces a greedy difference basis set scoring 166. The next agent reads that output and produces a quadratic-spacing set scoring 289, nearly twice as bad. The chain interaction gave the next agent license to overwrite a better answer rather than permission to build on it.

## Cases When Diversity Is Beneficial

The only multi-agent result that consistently exceeds the single-agent baseline is the no-synthesis ablation with diverse backbones. Three models generate independently and the pipeline returns the best-scoring proposal without asking any model to read another's output. On Erdős, this scores Q=0.57 against a single-model baseline of 0.13. Errors across seeds are nearly uncorrelated. Claude explores structured functions, GPT-4o finds compact representations Claude misses, and the best of three covers what any one alone would miss. No interaction occurs. No synthesis runs. The diversity is preserved precisely because no step asks models to respond to each other.

The effect shrinks on Difference Bases (+0.119 Q-gain) and is absent on Molecule QED, where the diverse team ties the same-model team. On Molecule QED, all three models fail the hidden synthesizability criterion the same way regardless of backbone. The failures are correlated, and correlated failures cannot be covered by diversity. The coverage argument holds only when models fail in genuinely different ways.

GPT-4o×3 through the no-synthesis protocol scores Q=0.476 averaged across the three 2×2 tasks, above the diverse team's 0.368. On Erdős specifically, GPT-4o×3 reaches Q=0.710, above diverse's 0.568. On any subset where one model is dominant, three copies of that model beat the diverse team. Gemini×3 shows the other side. Every one of ten Gemini seeds on Erdős scores at or below the calibration baseline, mean Q=0.000. A same-model team using the wrong backbone has no recovery path. The diverse team's value is in not needing to predict which backbone wins before running. Across all nine tasks, where each model wins exactly two, no single-backbone team covers the full spread.

## Conclusion

Diverse backbones provide coverage on tasks where no model dominates. Running them independently surfaces the best available solution at no coordination cost. Asking agents to read each other's work destroys that advantage. Text similarity shows interaction drives outputs to converge. Trajectory data shows scores regressing precisely at the interaction step. Transcripts show synthesizers selecting the worse answer and chain agents overwriting better solutions. The independence that made a diverse team valuable is structurally eliminated by the step designed to exploit it.

The constructive implication is narrow. Use diverse backbones. Generate independently. Pick the winner. Skip synthesis on discrete and combinatorial tasks where interpolation is not meaningful. Every elaboration on that baseline spends the diversity budget without recovering it.

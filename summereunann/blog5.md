# Blog 5

## Summary of Prior Work

Last week I traced the mechanism behind the interaction tax through text similarity, score trajectories, and synthesizer transcripts. The diversity benefit is generated at the proposal stage and consumed by interaction. Diverse backbones produce genuine task coverage because each backbone performs well on different problems, but every protocol that asks agents to read each other's work collapses that coverage before it reaches the final output. I submitted the NeurIPS paper and turned to running additional experiments this week. I designed a crossover protocol to test whether the tax is specific to consensus seeking architectures, added constraint satisfaction tasks that reveal when interaction actually helps, ran budget doubling experiments, and condensed the paper into an ICML workshop submission organized around a new concept I am calling the verifiability boundary.

## Central Claim

My central claim this week is that the interaction tax depends on task structure. On open ended optimization where there is no single right answer, the first critique round makes solutions worse 57% of the time (17/30 runs). On constraint satisfaction tasks where violations are trivially checkable, critique never degrades a solution in the first round (0/30 runs) and raises feasibility from 0% to 47 to 73%. Whether feedback can point to a specific error determines whether interaction helps.

## Crossover Protocol

The paper's conclusion left open whether evolutionary approaches that explicitly maintain population diversity could combine diverse models without paying the interaction tax. I designed a crossover protocol to test this. Phase 1 is identical to MoA where three diverse proposers (Claude, GPT-4o, Gemini) generate independently. Phase 2 replaces the synthesizer with a crossover agent that is explicitly instructed to perform component level recombination. The prompt says "do NOT average, do NOT pick a winner, extract best components from each and construct a genuinely hybrid solution." The prompt prohibits consensus and demands structural splicing. Crossover Refine adds a third phase where each original agent gets one self refine round starting from the crossover output.

On Difference Bases (lower is better), I ran 10 seeds of each variant alongside MoA nosynth as a control.

| Protocol | Mean hidden score | Median hidden score |
|---|---|---|
| MoA nosynth (diverse, pick best) | 23.51 | 13.69 |
| Crossover Refine | 11.53 | 4.92 |
| Crossover | 10.99 | 4.92 |

Both crossover variants produce lower mean scores than the no interaction control, which initially looks like an improvement on a minimize task. The median tells the real story. On 7 of 10 seeds, both crossover variants produce the exact same score (4.92), which is the score you get from a single greedy construction with no task specific insight. The crossover agent reads three diverse proposals and ignores them, generating from scratch. On the 3 seeds where it deviates (scores of 16.6, 28.6, 30.3 for plain crossover), the recombination attempt produces something worse than any of the inputs. Whether the interaction step is called synthesis, debate, critique, or crossover, reading other agents' outputs either causes convergence toward a generic solution or produces a degraded hybrid.

## Constraint Task Experiments

The most important new experiment this week was adding two constraint satisfaction tasks to the benchmark. Knapsack 50 and 3AP Free 100. On these tasks the interaction tax reverses.

Best of N achieves 0% feasibility on both constraint tasks at all tested sample sizes. The backbone rarely generates a valid solution independently, so sampling more does not help. Self Refine raises feasibility to 7 to 13%. MAgICoRe raises it to 47 to 73% (Fisher exact p<0.003 on both tasks). The first critique round never degrades a constraint task solution (0/30 runs regressed at step 0 to 1).

On optimization tasks the first critique round degrades 57% of the time (17/30 runs). On a constraint task, critique targets a specific checkable violation. "Your weight exceeds the limit, remove an item" is actionable. The refiner patches that violation without re solving the entire problem. On an open ended optimization task, "the partition could have more cut edges" gives the model no specific flaw to fix, so it pushes solutions toward familiar patterns instead.

The constraint tasks also reveal an interaction between verifiability and diversity. I ran same model and diverse backbone variants of Debate and MAgICoRe on both tasks.

| Task | Config | Same model | Diverse | Direction |
|---|---|---|---|---|
| Knapsack 50 | MAgICoRe | 1/10 | 2/10 | neutral |
| Knapsack 50 | Debate | 2/10 | **10/10** | diversity helps |
| 3AP Free 100 | MAgICoRe | 6/10 | 3/10 | diversity hurts |
| 3AP Free 100 | Debate | 6/10 | **0/10** | diversity hurts |

On Knapsack, where verifying a weight violation is arithmetic, diverse Debate achieves 10/10 feasibility versus 2/10 for same model. The different models catch different violations and the feedback loop converges on a valid solution. On 3AP Free, where verifying whether a three term arithmetic progression exists requires reasoning, diverse Debate drops to 0/10. The diverse agents give each other incorrect critiques because they cannot reliably check the constraint, and the interaction destroys valid partial progress. Verifiability of the error signal determines whether critique helps.

## Solution Diversity Collapse

I computed pairwise solution distance (Hamming distance for binary solutions, cosine distance for continuous) across protocols to quantify the diversity collapse.

| Protocol | Avg pairwise distance |
|---|---|
| MoA nosynth (diverse, no synthesis) | **0.48** |
| Best of N (same model, 8 samples) | 0.40 |
| Debate | 0.38 |
| MoA (diverse + synthesis) | **0.34** |
| MoA same model (one model) | **0.34** |

Diverse proposers enter with 0.48 separation. After synthesis, that drops to 0.34, identical to same model. The synthesis step erases the diversity it was designed to exploit. The synthesis coefficient is near zero because synthesis homogenizes rather than combines.

On five of seven optimization tasks, synthesis copies the best proposer's output at least 80% of the time rather than combining parts from each.

| Task | Improved | Copied | Degraded |
|---|---|---|---|
| Circle Packing | 60% | 40% | 0% |
| Molecule QED | 20% | 80% | 0% |
| Erdős Overlap | 10% | 80% | 10% |
| Flat Polynomials | 0% | 100% | 0% |
| TSP 100 | 0% | 100% | 0% |
| TSP 50 | 0% | 90% | 10% |
| Difference Bases | 7% | 43% | 50% |

Difference Bases is the exception where synthesis actively attempts recombination and fails half the time. Combinatorial structures do not interpolate, and the novel set the synthesizer constructs is usually worse than any of the inputs.

## Budget Doubling

I tested whether multi agent protocols improve with more compute by running budget 2x variants (doubled token cap, wall clock, and eval calls) on the two most discriminative tasks.

On Difference Bases (lower is better)

| Protocol | Mean hidden score |
|---|---|
| VGS 2x | 8.87 |
| Best of N 2x | 12.88 |
| MoA 2x | 17.24 |
| Self Refine 2x | 18.99 |
| Single Shot 2x | 744.02 |

VGS, the single agent evolutionary search, benefits most from extra budget because it can run more generations. MoA with double budget does not close the gap. On Erdős, Single Shot 2x (0.568) matched the diverse team's no synthesis score from the main benchmark while MoA 2x (0.267) and Best of N 2x (0.252) both performed worse. Extra budget did not help multi agent protocols overcome the interaction tax.

## Updated Numbers for ICML Submission

I condensed the NeurIPS paper into a 4 page ICML workshop submission this week. The updated 2x2 factorial uses N=198 runs (30 on DiffBases, 10 per arm on Erdős and MolQED). The diversity coefficient is +0.195 (CI [+0.125, +0.262]), positive in all 10,000 bootstrap resamples. The synthesis coefficient is +0.044 (CI straddles zero). The leave one out analysis shows the finding is task dependent. Removing Erdős drops the diversity coefficient to +0.052 with CI crossing zero. I flagged this in the Limitations section.

The MIG sign reversal holds across three architectures with different interaction structures (Chain, MAgICoRe, Debate), with bootstrap probability that same model MIG exceeds diverse MIG at 97%, 99%, and 88% respectively. MoA remains the only configuration where diverse MIG stays positive, because proposers never see each other's outputs.

## Conclusion

On open ended optimization, models that interact converge within a single round of full output exchange, erasing the diversity that made the team valuable. The crossover experiment shows that even framing interaction as genetic recombination does not prevent this. The budget doubling experiment shows that more compute does not prevent it either. On constraint satisfaction tasks with trivially checkable errors, interaction is productive because critique can target specific violations. The verifiability boundary separates these two regimes. Effective multi agent systems should preserve proposal independence on open ended tasks and reserve iterative critique for tasks with verifiable feedback signals.

# Week 3: Generalizing ICR to Open-Domain Tasks and Benchmarking Against CS-ICL (Apr 24, 2026)

## Generalizing the Pipeline: The TaskSpec Interface

For the last two weeks we have been building ICR_partition for solving magma equational implication problem specifically. This week we generalized it beyond magma equations to arbitrary classification tasks and benchmarked against Cheat-Sheet ICL (Honda et al., EMNLP 2025), the oneshot baseline for cheatsheet generation. It produces a compressed cheat sheet in a single LLM call with no iteration or failure analysis given training examples with gold chain-of-thought traces.

ICR_partition is originally builted in a way that every prompt assumed two equations and a TRUE/FALSE verdict. We used algebraic features as partition keys and referenced equation1, equation2, and oracle CSV lookups in every failure. 

We generalized this pipeline through a **TaskSpec** object: five optional hooks that let any new task define its own prompt construction, failure formatting, partition key, rule ID syntax, and bootstrap ruleset. In this way we can keep the core pipeline stages: bootstrap, Phase 1 rule refinement, Phase 2 case study generation, and final eval, stay untouched. For different tasks we only have toswap in the task-facing surfaces.

The TaskSpec encodes *what to do at the specific task boundary*: what a failure looks like, what makes two failures structurally similar, what an initial rule set should say. This means adding a new task only requires writing a meaningful partition key and failure formatter, with no changes to the refinement loop itself. Every improvement we make to the loop then propagates to all tasks automatically.

Another non-obvious addition to the pipeline was reasoning injection. CS-ICL feeds gold chain-of-thought traces from training examples directly into its generation prompt. We allow this reasoning injection in pipeline by injecting `item["reason"]` as `_oracle_exact` during generation (bootstrap, Phase 1 patch formatter, Phase 2 case study formatter), reusing oracle display code we had already built for magma. The generator can now directly contrast wrong reasoning with a correct trace rather than inventing a correction from scratch.


## Results

| Task | Majority Baseline | CS-ICL | ICR (Ours) | Δ vs CS-ICL |
|---|---|---|---|---|
| boolean_expressions | 51.0% | 93.0% | 92.0% | −1pp |
| sports_understanding | 51.0% | 97.0% | 98.0% | +1pp |
| disambiguation_qa | 40.0% | 79.0% | 79.0% | 0pp |
| causal_judgement | 56.3% | 70.1% | 62.1% | −8pp |

We land within 1pp of CS-ICL on three of four tasks. The outlier is causal_judgement, and it is not a surprise.


## Causal Judgement as a Task-Type Boundary

Causal judgment is a different kind of task. Each question describes a unique scenario with specific actors, actions, moral structure, physical causation. As a result, the model's errors are diffuse: one failure involves joint chemical sufficiency, the next divided omission, the next overdetermination. Since they share no structural cause, we have to partition bins that are too small and too heterogeneous for the generator to find a principle worth writing about.

It is worth noting that this is not a failure of implementation. ICR_partition has three active stages: a bootstrap that seeds an initial rule set from a sample of failures, a Phase 1 that identifies which existing rule is misfiring and proposes a targeted patch, and a Phase 2 that clusters remaining failures by partition key and generates a case study for each cluster. Each stage assumes failures are structurally cohesive enough to yield a generalizable fix. Causal judgment breaks that assumption at every level. The difficulty here is not a systematic reasoning gap we can close; it is the absence of folk-intuitive judgment that is scenario-specific by nature and resists distillation into reusable content.


## Why Our Pipeline Is Meaningful Despite Limited BBH Gains

even though our pipeline didnt out-run CS-ICL on three tasks while spending 2×, it is important to consider what we can do that CS-ICL cannot do at all.

CS-ICL produces a static artifact. It has no mechanism to detect when the cheat sheet is wrong, no way to measure whether a proposed update helps or hurts, and no ability to target the specific failure modes that survive an initial pass. A one-shot summary that misses a pattern misses it permanently.

CS-ICL is also constrained by how many training cases fit in a single LLM context window. Once the training set exceeds that limit, CS-ICL simply cannot see all of it, while we process cases incrementally across iterations and tolerate arbitrarily many.  

The deeper argument is about task complexity. On BBH, we are both operating near the ceiling of what a short cheat sheet can do for a capable base model. In these cases CS-ICL's one-shot summary is already a good-enough approximation. The domain where our iterative approach earns its cost is one where failure modes are deep enough that a single pass cannot capture them, where the cheat sheet needs to encode non-obvious structural distinctions that only become visible through repeated diagnosis and testing. Magma equational implication is exactly such a domain: the rules governing which implications hold require precise algebraic conditions that no one-shot summarizer would derive from examples alone, failure modes are numerous and structurally diverse, and the regression surface is complex enough that a fix for one class of problems silently breaks another. Where CS-ICL would produce a rough prose summary of patterns, we produce a structured ruleset with case studies targeting the specific algebraic structures that trip the model.

The BBH results establish that our generalized pipeline is competitive with the strongest available baseline on structured tasks. 

## Next Steps

1. **Quantify the structural boundary.** We only have a rough analysis of the structural vs. knowledge-intensive split so far. The next step is to quantify a number that predicts it: average partition bin size across tasks. When failures cluster tightly into large bins, ICR can write a targeted case study that fix them effectively. However, when every failure ends up in its own bin, there is nothing to generalize. Measuring this across tasks and correlating it with the ICR vs. CS-ICL accuracy delta would turn our qualitative claim into a computable predictor of which approach to use.

2. **Find tasks that stress-test the performance gap.** BBH tasks are clean and the base model is already strong, so both approaches operate near the cheat sheet ceiling. We need tasks where failure modes are deep and numerous enough that a single one-shot pass genuinely cannot capture them. we need to find domains like magma equational implication, but also other formal or structured reasoning tasks where iterative diagnosis should pull ahead of CS-ICL by a decisive margin.

3. **Address the scoring overhead.** The 2× cost premium is almost entirely fixed scoring cost. Lazy evaluation — score only active-bin items during the loop, full rescore at the end — would eliminate most of this on tasks where we hit early stopping fast.

4. **Run CS-ICL on magma for a direct comparison.** All our magma results use our own held-out sets. Running CS-ICL on the same split gives the cleanest number for the paper's main claim, and magma is exactly the domain where we expect iterative refinement to win clearly.

Implementation is available at: https://github.com/AndrewRqy/ICRefine

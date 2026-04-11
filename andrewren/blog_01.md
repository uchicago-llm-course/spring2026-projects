# Iterative Cheat-Sheet Refinement (Apr 10, 2026)

## What we have built so far

Cheat-sheet ICL (Honda et al., 2025) showed that you can compress many-shot demonstrations into a short text summary and get comparable performance at a fraction of the inference cost. The weakness is that the summary is written in one pass with no way to check or fix mistakes. Our project adds a correction loop on top of it. We call the system ICR, Iterative Cheat-Sheet Refinement.

**SAIR Eval.** The foundation of everything is the SAIR evaluation pipeline. It handles scoring model responses against ground truth, logging results to a structured ledger, and driving the outer loop that alternates between evaluation rounds and refinement steps. SAIR is what makes iterative improvement possible: each round it identifies which problems the current cheatsheet fails on, hands those failures to the refinement system, and then re-evaluates to measure whether anything changed. Without a reliable, fast evaluation loop, you cannot tell whether a cheatsheet update helped or hurt.

**ICR-Select.** The refinement system we have focused on most is ICR-Select. It takes a cluster of failures from SAIR, generates candidate case studies to address them, and runs each candidate through a quality pipeline before anything gets written to the cheatsheet. ICR-Select has three main internal components.

*Roadmap synthesis* produces the Reasoning Roadmap — a short set of structural checks the model runs at inference time to figure out which kind of problem it is looking at. Each step in the roadmap runs a mechanical check and routes the model to the appropriate section of the case study bank for the actual reasoning. Verdicts live in the case bank, not the roadmap. This keeps the roadmap lightweight and prevents the structural guide from growing into a monolithic thing that tries to encode every edge case directly.

*Case study generation* takes a failure cluster and asks the model to write a targeted worked example that isolates the structural property the cluster shares. Candidates compete against each other, and the highest-scoring one moves on to the quality gates. The generation step uses the model's post-think reasoning summary as the error signal rather than the full chain-of-thought, which is much shorter and more diagnostic.

*The utility gate* is the final filter before a case study reaches the cheatsheet. It currently runs four checks: a fix-rate check that asks how many failures from the target cluster the candidate resolves, a regression check that ensures the candidate does not break problems the model already handles correctly, a similarity check that prevents near-duplicate case studies from accumulating, and a candidate competition pass that selects the best among alternatives.

## Current challenges

### Gate thresholds blocked all refinement

The early utility gate used binary pass/fail thresholds. A candidate had to fix at least half the failure bin, and it could not regress on more than a small fraction of already-correct items. Those two requirements together made adding anything to the cheatsheet nearly impossible, and the cheatsheet never changed across several iterations.

The fix-rate floor was the first problem. Failure bins are hard clusters, meaning that the model fails on them consistently because the items share a structural property that resists easy resolution. Asking a single case study to fix half of a bin in one shot is a nearly impossible task.

The regression threshold was the second problem. Early in a run the pool of correctly-solved items is small. When the pool is only five or ten items deep, a single wrong prediction already exceeds a tight regression cutoff, so candidates get rejected not because they are bad but because the pool is too small to measure anything reliably.

The two failures together make a case study nearly impossible to be adapted: the fix-rate standard is too high relative to what any one case study can realistically accomplish, and the regression standard is too sensitive relative to the pool size available at the point when improvement matters most.

### Evaluation concurrency mismatch starved the GPU

We were sending concurrent requests to the vLLM server and seeing only a fraction of them running at a time, with clear idle gaps in the server logs. The evaluation loop used Python's asyncio library, which is cooperative: when opening connections through an SSH tunnel, asyncio opens one channel at a time as coroutines yield control. The ICR scoring stage used Python threads and was achieving far higher concurrency on the same tunnel, because threads open connections truly in parallel. The two stages were using different concurrency models, and the throughput difference was invisible from the accuracy output alone.

## What we did against those challenges

### Breakthrough: replacing binary gates with a continuous utility score

The most significant design change this week was rethinking what the gate is actually trying to measure. The binary approach treated acceptance as a classification problem: does this candidate clear a fixed bar, yes or no? The fundamental issue is that the bar was calibrated to a best-case scenario that hard failure clusters can never reach, and the regression cutoff was calibrated to a pool size that does not exist early in training.

The insight was that what we actually care about is not whether a candidate clears a threshold, but how much net value it adds to the cheatsheet. We formalized this as a utility score. The score has two components. The gain term measures accuracy improvement on a held-out slice of items whose algebraic structure matches the candidate. This allow the case study to be measured on problems that share the same structural features as the failure cluster it was written for. The penalty term measures how many previously-correct items the candidate breaks, scaled continuously so that breaking one item out of fifty barely registers while breaking ten is a serious cost. The final score is gain minus penalty, and a candidate is accepted if and only if the score is positive.

This is a fundamentally different framing from the old approach. The old gate asked "is this candidate good enough?" and defined good enough as clearing two independent thresholds that were each too strict for the stage of training we were in. The utility score asks "does this candidate make the cheatsheet better on balance?" and lets gain and penalty trade off directly against each other. A candidate that fixes a moderate number of failures and breaks nothing will always have positive utility. A candidate that fixes many failures but breaks a few correct items can still pass if the gain outweighs the cost. This matches the actual goal of refinement far better than two uncoupled binary checks.

The structural matching that drives the gain term is computed directly from algebraic features of the failure equations. We measure things like operator composition depth, variable count, and constraint type, which requires no LLM call and adds no latency. When the matched slice is too small to produce a stable estimate, the gate falls back to the classic checks rather than scoring from three items and treating that as signal.

### Rewriting the evaluation loop to use threads

We rewrote the evaluation loop to use a thread pool with synchronous requests, matching what the scoring stage already does. The GPU now stays loaded throughout evaluation. The lesson is similar to the one in the implicit constraints project: when a performance number looks wrong for an extended period, the first thing to check is the infrastructure, not the model.

## Next steps

1. **Run a clean three-way comparison.** Run all three ICR variants — Naive, Reasoning, and Select — on the same held-out split under the current gate settings. All results before the gate fix are invalid, so this is the first meaningful comparison we will have. This will tell us what the continuous utility gate and the post-think error signal are each contributing relative to the simpler variants.

2. **Stress-test the utility gate on `hard1`.** Run ICR-Select against the `hard1` dataset, which has a lower baseline accuracy and more diverse failure modes. The utility gate was designed and tuned against normal-difficulty problems, so `hard1` will reveal whether the structural matching and continuous penalty scaling hold up when the failure distribution is harder and less uniform.

3. **Run ablation pruning at scale.** Test whether one strong case study outperforms several weaker ones accumulated over iterations. The cheatsheet grows with each accepted candidate, and it is not obvious that more case studies is always better — a bloated bank may introduce noise at inference time just as much as it adds coverage.

4. **Cross-model evaluation.** Evaluate the cheatsheet under a different scoring model to get a first signal on whether the refined cheatsheet generalizes across models. All evaluation so far uses a single model, so we do not yet know if the case studies we are writing are genuinely useful or just overfit to one model's failure patterns.

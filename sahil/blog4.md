# Behaviorals are more solid, Probing needs iteration

_DUET (Discussion Under Equal-compute Training) — week of May 1_

The plan from blog 3 was: tighten the metrics, then run probes for the mechanistic part. Both of these got done this week. The metrics held up fairly well across more eval and more data! However the probes didn't really work, but I have a v2 in place based on the failure modes from v1.

## Part 1: Metrics hold up(!)

Mutual-swap categorization on cumulative training data (n=111 instead of n=14) confirms what blog 3 had to flag as inconclusive. Mutual-swap variants are ~47% of D's disagreement outcomes, asymmetric productive persuasion is ~15%. So it seems D's persuader bonus does fire on real productive cases, just not as often as mutual swap, which I think washes a lot of the signal out. Dang flagged on the last post that this seems like a problem worth fixing. I agree, but I don't have a clean idea yet for how to break the symmetry without losing the rest of the protocol but will research further into it. Some of my initial thoughts are in Next Steps.

Pass@k at large k on MATH-500 L5: D and C both exceed base at every k including k=256, S is at-or-below base. So Yue's narrowing prediction doesn't replicate for compute-extras, only for solo. The C-D gap is roughly constant across k (~0.015-0.020), so pass@k doesn't favor D over C either.

Expanded eval (MATH-500 L1-4, GPQA Diamond, AIME 24+25). The C > D > S ordering replicates on GPQA at n=198 (C: 0.573, D: 0.549, S: 0.480, base: 0.444). On MATH-500 L1-3 the three trained conditions are basically tied with each other and with base. The training effects only really show up on L4-L5 and on harder OOD stuff like GPQA.

Adversarial robustness expanded to n=86 (using a more permissive intersection criterion). D flips on 1/86 vs S 6/86 and C 4/86 — the pattern from blog 3 is now more defensible. Dang asked on the last post whether D's robustness is sycophancy toward the peer rather than genuine commitment to correct answers. I don't have a way to test that cleanly yet, but the mutual-swap finding is at least consistent with the sycophancy framing — D's players are heavily oriented toward each other during training, and that might transfer to "easily moved by anyone in conversation" rather than to "moved by good arguments." Again, my Next Steps has my initial ideas on how to do this.

## Part 2: The Probing experiment

The plan was to use Zhang et al.'s setup. For each of the 4 models, I would get the residual stream activations at the closing-brace token of `\boxed{...}` expressions, train an MLP probe per layer to predict correctness, compare across models. This V1 is pretty small and simple, as it's just 4 models × 10 layers = 40 probes, MLP only, no cross-model transfer.

Here were my initial results:

|Condition|Best layer|Best test AUC|Layer-0 AUC|
|---|---|---|---|
|base|36|0.911|0.536|
|solo|28|0.855|0.725|
|discussion|8|0.733|0.400|
|compute_control_sd|36|0.843|0.221|

My initial thoughts was simply that base encodes correctness most cleanly, discussion the least. However, I found some structural problems with this V1 experiment, detailed below.

**Per-problem disagreement is too low.** With 8 rollouts per problem at temp 0.6, models generally are either consistently right or consistently wrong on each problem. Of 129 problems base saw, only 1 had boundaries with both labels. Trained conditions had 8-11 mixed-label problems out of ~80-100. Roughly 90% of problems contributed boundaries that all share one label.

This will break the probe since if 90% of problems have one label only, the probe can hit those right just by recognizing which problem it is. It learns "this looks like problem #47, and #47 is always correct" instead of "this activation pattern indicates correctness." The 0.911 AUC on base isn't measuring whether the residual stream encodes correctness — it's measuring whether the probe can recognize specific problems whose labels happen to be stable.

**Layer-0 leakage on three of four conditions.** Probing at the embedding layer (before any transformer computation) gave above-chance AUC for solo (0.725), discussion (0.400 inverted), and compute-control (0.221 inverted). Only base was clean.

Looking at the boundary content explained why. Correct boxed answers tend to be bare integers (44, 22, 4, 100). Wrong boxed answers tend to have LaTeX (`\frac{270}{7}`, `12\sqrt{2}`, `20\pi`). A probe at the closing-brace token can pick this up from the embedding alone and post an AUC without ever measuring the model's internal reasoning. The closing-brace token's embedding directly leaks the boxed content's surface form.

So discussion's "lowest AUC" doesn't necessarily mean its representations are worst either. Discussion has 8 mixed-label problems vs base's 1, which means slightly more real discriminative work for the probe — and it can't shortcut on those mixed problems via problem identity. The conditions with the most homogeneous-label data look like they have the best probes regardless of what's actually inside the model.

V1 didn't tell me anything about whether discussion training changed the model's internal correctness representations. It told me that probing for correctness on math reasoning chains has two specific traps that need to be designed around.

## What V2 needs

There are three changes I plan to make. First, more rollouts per problem at higher temperature, filtered to problems with at least 5 correct and 5 incorrect boundaries. This should force the probe out of the problem-identification shortcut. Second, probe at the last reasoning token before `\boxed{` instead of at the closing brace, so the boxed content's surface form isn't directly visible. Third, contrastive training on same-problem correct/incorrect pairs, which makes problem identity cancel by construction.

Before doing all this, I already have pass@k=256 rollouts on MATH-500 L5. If I count how many problems have ≥5 correct and ≥5 incorrect rollouts in that data, I'll know whether V2 is feasible on existing data or needs a fresh sweep. I'll first do this check before implementing V2/deciding on a different structure.

Now looking after V2. If V2 produces real signal (which it hopefully does), the natural follow-up is the self-reflection vector (Ma et al.). D produces fewer reflection markers, but might be doing more reflection internally, and probes that aren't fooled by surface form should be able to see that. If V2 also doesn't produce interpretable results, I would first try to figure out how to get a better signal based on the results from V2, but in a worst case scenario I would drop the mechanistic story. It's definitely less exciting but the data still supports a real contribution. However I do feel optimistic that I will have an interesting mechanistic story with some iteration.

## Conclusion & Next Steps

The behavioral results are now robust across multiple benchmarks and more defensible. The C > D > S ordering replicates across MATH-500 L4, L5, and GPQA Diamond, and D's behavioral identity holds up with more data. I feel more confident these are real and not noise compared to last week.

My primary next step is V2 probing. V1's failure was structural rather than fundamental, and I identified the fixes above. If V2 produces real signal, the follow-up is the self-reflection vector. If not, I'd drop the mechanistic story from the paper and lean on behavioral and replication findings but I am hopeful I will get some signal with some more iteration.

Two things worth doing in parallel from Dang's comments. First, investigating mutual swap — Dang flagged it as something to fix. The cleanest test is heterogeneous Bob (ie different models/sizes for Alice/Bob) but doing this may be a weeklong investment. Second, testing whether D's adversarial robustness is sycophancy toward peers rather than genuine commitment, as Dang suggested. My intial thoughts on how to test this is by framing counter-arguments as coming from a "peer" vs. unspecified source which is an easy add-on to the existing pipeline.

So in order of priority: Implement V2 probing and following mechanistic analysis, then the sycophancy test, then heterogeneous Bob if there's time before the deadline.

# What Discussion Training Actually Does (At Iter 100)

_DUET (Discussion Under Equal-compute Training) — week of April 24_

The last post ended with the foundation reset done. I spent the week getting a lot of the hard setup out of the way and finally got to look at some preliminary results which is exciting!

To recap my experiment: I have three conditions (S = solo, D = discussion, C = compute-control), then bootstrap CIs, then behavioral metrics (where I am now), then probes (next week). There's a lot of methodology to walk through before the actual results since I wanted a setup I could trust and defend.

Initial look at results: at iter 100 on MATH-500 Level 5, the compute-control C significantly beats discussion D on accuracy, and D significantly beats solo S. So discussion training does help vs. solo, but matched-compute solo with self-drafts beats discussion. That's not the result I was expecting going in. However behavioral metrics tell a much more interesting story than the accuracy numbers do, and I'll get to that.

## Redesigning Compute Control

The C condition I had at the end of last blog wasn't actually compute-matched. On disagreement-triggered problems, C generated extra rollouts with problem-only prompts (~300 prefill tokens) and the default 8192 decode cap. D's post-discussion pass on the same problems is ~6k prefill (problem + two representative drafts + reanalyze instruction) and 2048 decode. So C had way more decode budget than D, and way less prefill, so this isn't true compute-control.

I wrote out two options for fixing this:

**Option 1**: match both prefill and decode. Cap C's decode at 2048 to match D, and feed C's extra rollout two of its own prior drafts to inflate the prefill to ~6k. Same prompt scaffolding as D, the only difference being that the two drafts are self-drafts instead of peer drafts. Cleanest isolation of "whose drafts are in the prompt" as the treatment variable. Downside: if both self-drafts agree (common on easier problems), the prompt becomes "here are two attempts that both say X, reanalyze" — that will most likely create reaffirmation bias.

**Option 2**: match decode only, leave prefill asymmetric. Cap C's decode at 2048, keep two extras, input stays problem-only. C ≈ 300 prefill + 4096 decode, D ≈ 12k prefill + 4096 decode. Decode matches exactly, prefill doesn't. Simpler, no reaffirmation bias, but creates the obvious critique that maybe D wins because it sees more context, not because it's cross-player context.

I was unsure what to do here and Dang gave me useful guidance and suggested I go with option 1. Looking back I think this was the right call for my initial question. If C's decode is matched but its prefill is much smaller, a reviewer can say "you didn't really compute-match — D was just running in a different prompt-length regime." Option 1 forecloses that critique. The reaffirmation bias concern is real but addressable through draft selection (more on this soon).

Working out the exact numbers took another round. D actually runs _one_ post-discussion pass per player on disagreement, not two — I had originally miscounted. So D's per-player disagreement compute is ~6k prefill + 2048 decode. So the final per-player budget for C: one extra rollout, ~6k prefill, 2048 decode cap. Same as D.

The reward for C is `r_correct + r_overlong` — same as D's player reward minus the persuader bonus, which can't fire in C since there's no peer to persuade. I considered adding a self-correction bonus to C as a proxy for D's persuader (would fire when extras are right and provided drafts are wrong) but decided against it. D's persuader rewards the _producer_ of good reasoning when it convinces the peer. A self-correction bonus rewards a different behavior (recovering from your own wrong drafts), so it's not really an analog. Cleaner to leave the reward simple and let the persuader-bonus asymmetry be part of the discussion treatment.

The last decision was the self-draft selection rule. For each player's extra rollout, I need to pick 2 of that player's 8 prior rollouts to put in the prompt. The rule I went with: if the 8 rollouts have at least 2 distinct final answers, pick one rollout from the top-2 answer classes (highest-logprob within each). If all 8 agree, pick the 2 highest-logprob rollouts from the single answer class. This mirrors the structure of D's representative selection — "two drafts that typically disagree" — and keeps the treatment consistent (always 2 drafts in the prompt, never falling back to problem-only). Dang had suggested that reviewers likely won't press this point, but as an extension if time permits is trying random drafts as well.

## Choosing the Right Behavioral Metrics

Pass@1 alone wasn't going to be enough. The whole project framing is that two models with the same accuracy can be doing different things underneath, so I needed metrics that could detect those differences. I'll focus on the more interesting design choices:

**Adversarial robustness on the intersection set.** The previous round tested each model on its own correct-answer subset, which biased C's robustness number upward (C's correct set was easier on average). Fix: restrict to the intersection of problems all three models solve, so flip rates are measured on a shared difficulty distribution.

**Progressive flip rate alongside regressive flip.** SycEval (Fanous et al.) distinguishes regressive sycophancy (flipping correct → wrong) from progressive sycophancy (flipping wrong → correct on a _correct_ counter-argument). Just measuring regressive would conflate "robust" with "stubborn."

**Diversity via gte-large (8k context) instead of MiniLM.** MiniLM's 256-token window forced chunking and mean-pooling, which washes out mid-chain strategy pivots. Feng et al.'s RPD paper makes an even stronger version of this critique that I'd like to implement eventually.

**Base-model diversity floor.** Suggested by Dang last round, gives "did training collapse diversity?" an actual reference point.

The rest are much more standard — distinct-answer counts, consistency@k from MACA, reflection-marker counts (a surface-level version of Ma et al.'s self-reflection vector work), and in-trace revision rate. All cheap and done afterward on existing rollouts.

Things I considered but didn't run this round: BIG-Bench Mistake-finding, verbalized confidence, RPD step-level diversity, inference-time discussion benefit. Each is worth doing but each also needs either a new dataset or more infrastructure than I wanted to build right now.

## Headline Results

Three conditions, all to iter 100 on the post-fix pipeline. MATH-500 Level 5, n=134, 8 rollouts per problem at temp 1.0.

|Condition|iter 50|iter 75|iter 100|
|---|---|---|---|
|S|0.759|0.702|0.675|
|D|0.741|0.702|0.712|
|C|0.750|0.747|0.738|

Paired bootstrap CIs at iter 100, 10k resamples:

- D vs S: +0.036 [+0.004, +0.071], significant. D beats S.
- D vs C: −0.026 [−0.052, −0.001], significant. C beats D.
- S vs C: −0.063 [−0.094, −0.034], significant. C beats S.

So the ordering at iter 100 is C > D > S, all significant. Discussion training does help vs. solo, but matched-compute solo with self-drafts beats discussion.

S degrades from 0.759 at iter 50 down to 0.675 at iter 100. D and C are both relatively stable. So the iter-100 D > S finding is partly because D held its ground while S declined, not because D got better. Worth being honest about that.

A note on the literature comparison: I don't have a great set of papers to directly juxtapose against. The closest related work I was able to find is inference-time, not training-time. Wang et al. 2024 found self-consistency beats multi-agent debate at matched inference compute. Becker et al. 2026 made a similar point with a token-matched comparison. Zhang et al. 2025 ("Stop Overvaluing MAD") argues multi-agent debate often loses to single-agent baselines, also at inference time. MACA is the closest training-time analog and finds debate-derived signals do help, but they don't strictly compute-match. So the direct comparison space is mostly empty — this experiment is filling in a gap rather than overturning a settled result which would be interesting. But I'll keep reading and see what I find.

## What I Found While Investigating C

Two things came up while debugging the new C condition that ended up being interesting on their own.

First, D's flip telemetry. The training metrics record `num_flips_to_correct` and `num_flips_to_wrong` per iter — counting cases where players changed their answer post-discussion. These were almost always equal: 30 of the first 35 iters had `flips_to_correct == flips_to_wrong` exactly. That looked like a double-counting bug. I traced through the counter, expecting to find it incrementing both fields on the same event, and... it wasn't. The symmetry is real behavior. When discussion fires on a disagreed problem and both players flip, in the typical case Alice flips toward Bob's pre-discussion answer at the same time Bob flips toward Alice's. The flip-to-correct on one player is paired with a flip-to-wrong on the other. Mutual swap.

This is interesting because it means D's persuader bonus mostly cancels out. The bonus fires when a correct player's reasoning convinces a wrong player to update — but if both players flip simultaneously in opposite directions, you get one persuader bonus on one side and the other player has just earned the bonus too (but in the reverse direction). The training signal from persuader is much weaker than the design implied. I think this is downstream of the curriculum filter — when both players are operating near their decision boundary on a problem (which is exactly what the [0.25, 0.75] solve-rate filter selects for), peer drafts mostly resolve internal uncertainty rather than transferring information. I haven't proven this, but the mechanism is consistent with what I'm seeing. I would argue this result generalizes beyond the filter since outside of this solve rate the players will mostly agree (and no discussion will trigger), but that's a side point/my (non-expert) guess.

Second, the peer-self overlap. C's prompt structure includes two self-drafts; D's includes two peer drafts. The intended treatment difference is "whose drafts is the player conditioning on." But on curriculum-filtered problems, Alice's near-boundary distribution and Bob's near-boundary distribution overlap a lot, because they're the same base model with different temperatures. So when I checked how often C's two self-drafts span the same answer set that D's two peer drafts would span, the breakdown across 119 disagreement cases was: 8.4% had no overlap, 57.1% had partial overlap (one shared answer), 34.5% had complete overlap (both answer sets identical). The clean peer-vs-self treatment contrast only exists on about 8% of cases. On the other 92%, C is conditioning on roughly the same candidate-answer set that D is, just with different reasoning text attached. This may be a limitation of this paper.

## Behavioral Metrics

Even though D doesn't win on accuracy, it produces a measurably different model than C and S. The behavioral metrics make this clear.

**Reflection markers** (counts of phrases like "wait", "actually", "let me reconsider" per rollout) at iter 100:

- S: 11.5 markers per rollout, climbing from 6.6 at iter 50
- D: 5.6 markers per rollout, basically flat from 4.5 at iter 50
- C: 10.4 markers per rollout, climbing from 6.1 at iter 50

D produces about half as many reflection markers as the other two. And while S and C both ramp up reflection markers across iters, D stays flat. n=1072 rollouts per condition, well-powered.

**Revision rate** (fraction of rollouts containing multiple `\boxed{}` answers with different values) at iter 100:

- S: 0.160
- D: 0.069
- C: 0.121

Same pattern. D is roughly half the others and flat across iters; S and C both climb.

**Distinct answers per problem** (across the 8 rollouts) at iter 100:

- S: 2.17
- D: 1.92
- C: 1.86

Here D and C are similar and S is highest. So D doesn't have the lowest answer-level diversity — C does, slightly — but D is still tighter than S.

**Cluster diversity** (gte-large embeddings, agglomerative clustering at threshold 0.05) at iter 100:

- base Qwen3-4B: 1.052 mean clusters
- S: 1.328
- D: 1.090
- C: 1.194

D's policy is closest to base in cluster terms — meaning D's 8 rollouts on a typical problem fall into roughly the same number of distinct strategy clusters as the base model. C is somewhat more spread. S is the most spread. None of the trained models are below base on diversity, so the "RLVR narrows reasoning" story from Yue et al. doesn't fit our data — discussion preserves base concentration, but doesn't shrink it.

**Consistency@8** (fraction of rollouts matching the majority answer):

- S: 0.812
- D: 0.853
- C: 0.864

D and C are similar and high. S is lower.

Putting these together: D produces fewer reflection markers, lower revision rates, lower cluster diversity, and similar consistency to C. C produces slightly fewer distinct final answers and slightly higher consistency than D, but reflects and revises more. The cleanest description I can give: D and C are both more concentrated than S, but along different axes. D's concentration is in the reasoning structure, ie fewer distinct strategic clusters. C's concentration is in the answer distribution, ie slightly fewer distinct final answers. Both are also more accurate than S.

## Subject-Level Stratification

I broke pass@1 down by MATH subject area at iter 100:

|Subject|n|S|D|C|D−S|C−S|
|---|---|---|---|---|---|---|
|Intermediate Algebra|36|0.55|0.66|0.68|+0.12|+0.13|
|Algebra|30|0.80|0.83|0.85|+0.04|+0.06|
|Prealgebra|19|0.66|0.64|0.65|−0.03|−0.01|
|Geometry|13|0.43|0.41|0.47|−0.02|+0.04|
|Counting & Prob|12|0.82|0.83|0.90|+0.01|+0.07|
|Precalculus|12|0.62|0.62|0.64|0.00|+0.02|
|Number Theory|12|0.96|0.97|1.00|+0.01|+0.04|

The largest gain by far is in Intermediate Algebra: D-S is +0.12 and C-S is +0.13. So the subject where the extra training signal most clearly helps is IA. But this isn't a discussion-specific finding — C gets the same boost as D. Both compute-extra conditions help on IA roughly equally. The smaller subjects (n=12-13) have wide CIs and I wouldn't read too much into their specific values. This ties into my next steps (running on much bigger test sets)

## Adversarial Robustness

The 23 problems where all three conditions answered correctly, with a counter-argument suggesting a wrong answer:

- S: 2/23 = 0.087 flip rate
- D: 1/23 = 0.043 flip rate
- C: 3/23 = 0.130 flip rate

D was the hardest to argue out of correct answers. And the inverse (counter-argument with the correct answer, on each model's wrong answers):

- S: 9/23 = 0.391 progressive flip rate
- D: 5/20 = 0.250
- C: 8/22 = 0.364

D was also the most resistant to updating on correct counter-arguments. So D is more committed to its answers in both directions. The directional pattern is consistent with the behavioral fingerprint, but n=20-23 is small and CIs are wide. Treat as suggestive, not confirmed. Another next step is calculating adversarial robustness over more problems (ie looking at more iterations).

## What is Still Inconclusive

A few things I had earlier hopes for that turned out to be too small to conclude anything:

**Mutual-swap categorization (M8).** I had categorized D's disagreement outcomes into spec categories like "mutual swap," "asymmetric persuasion correct," etc. Across the three milestone iters there were only 14 disagreement cases total. With 9 categories and 14 cases, I basically can't make any inferential claim. Re-running this on the cumulative training data (probably 200+ cases) is one of the cheapest next steps and might convert this from inconclusive to a real finding.

**Peer-self overlap at milestone iters.** The 119-case version (across all training iters) is the meaningful one — that's where the 34.5% / 57.1% / 8.4% breakdown comes from. The milestone-iter version had intersection sizes of 2-5 problems and is too small to interpret.

**C's self-correction rate.** When both of C's self-drafts are wrong, how often does the extra rollout produce a correct answer? On 21 cases, 4/21 = 19%. The training-aggregate version says 13%. Roughly consistent and the rate is real, but I'd want more cases to make this a claim.

## What comes next

My plan for next week: first tighten the empirical foundation, then  mechanistic analysis.

**Phase 1: shore up the metrics.** Most of the inconclusive findings in this post are inconclusive because of small-n, not because the underlying signal isn't there. The next round is about making the well-powered claims unimpeachable and either promoting the small-n findings to real or killing them.

The cheapest thing first is re-running the mutual-swap categorization on the cumulative training data instead of just three milestone iters. Same code, different denominator. With ~200+ disagreement cases instead of 14, the patterns I had to flag as "too small to interpret" become either real findings or confirmed noise.

Then pass@k at large k — k ∈ {1, 8, 32, 128, 256} on MATH-500 L5 plus base Qwen3-4B as the ceiling. Yue et al. 2025 argues RLVR narrows the reasoning boundary; running this on D, C, S, base is the falsifiable version of "does any of this help?" If discussion preserves more of base's reasoning scope at large k while solo and compute-control don't, the accuracy story changes — D didn't help at pass@1 but did something else useful. If all three trained models lose to base at k=256, that's also a finding.

The eval set itself needs to grow. MATH-500 levels 1-4 is free — same model weights, expanded problem pool — and lets me see whether D's relative advantage over S concentrates at the hardest levels. Adding GPQA Diamond gets a real OOD benchmark. AIME 2024+2025 together is ~60 problems which is at least defensibly-sized for AIME-level claims.

The adversarial robustness intersection is the other small-n problem. n=23 doesn't support strong claims about D being more committed; n=100+ would. Doable in a few hours.

**Phase 2: mechanistic analysis.** This is where I think the more interesting findings live, and it's where I plan to spend most of my time once the metrics are solid. The basic question: D produces a behaviorally different model than C. Is there a representational signature for that?

Linear probes for error recognition is the first thing to try, following Zhang et al. (arXiv:2504.05419). Segment each rollout at intermediate-answer boundaries, label each intermediate answer correct/incorrect with the SymPy verifier, train a 2-layer MLP per layer per model to predict correctness from the residual stream activation. The interesting comparison is whether D's representations differ from C's at the layer where the probe works best, and whether probes transfer between models.

If that lands, the natural next step is the self-reflection vector (Ma et al., arXiv:2506.12217). D produces fewer reflection markers than C and S — but is that because D suppressed reflection, or because D internalized it? Computing the activation difference between reflection and non-reflection contexts at each layer would distinguish these. If D's reflection vector is _stronger_ than C's despite fewer surface markers, that's the cleanest version of the "D thinks reflectively without verbalizing" story.

Activation steering is the riskier follow-up — extract a "discussion direction" from D vs S residual differences, apply it to S, see if S's behavior shifts toward D's. If it works, it's a causal mechanistic claim instead of a correlational one. Steering experiments often produce nothing interpretable so I'd only pursue this if the probes and reflection-vector work clearly point at a direction worth steering on.

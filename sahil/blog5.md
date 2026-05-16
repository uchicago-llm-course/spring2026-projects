# My attempts at fixing the mutual swap problem & analysis

The plan from blog 4 was probing and figuring out mutual swap. I worked on both -- I did the probing in the paper, and this week I worked on mutual swap problem. Mutual swap became its own thread, first with D-prime then DPP, and the probing landed cleanly enough to actually say something about what discussion training was doing. The main ideas is that: fixing D's persuader bonus to reduce mutual swap in turn killed D's adversarial robustness. The probing and analysis give better insight into why this happened and some things I should try next. 

## Part 1: D-prime: breaking the mutual swap

Blog 4 left off with mutual swap as the dominant failure mode in D (47% of disagreement outcomes). I first worked on a D-prime that applied four changes on D's pipeline:

1. **Heterogeneous Bob**: SFT'd Qwen3-4B-Instruct on AoPS-AMC olympiad problems to get Qwen3-4B-Geo. Bob is this model; Alice stays as the original instruct. I chose this dataset for SFT to balance relevance/performance (ie no significant drop-offs in Bob's performance on DAPO) while keeping Bob differentiated enough.
2. **Per-player KL anchor**: Alice anchored to instruct, Bob anchored to Geo. Each player gets its own reference so they don't drift toward each other.
3. **Stability bonus** (β=0.3): fires on the post-discussion correct rollout of a player who was both pre-correct and post-correct.
4. **Selective curriculum (XOR filter)**: only run discussion when exactly one player has a correct pre-discussion majority.

**D-prime results:** Mutual swap on cumulative training (n=121) dropped to 33.9%, a real improvement over D's 47%. math500_l5 pass@1 was +4.1pp over D at iter 75. The filter eliminated the both-wrong mutual swap case entirely (0% leakage). This was a nice initial result as it reduced the mutual swap rate, which I set out to do.

However, this didn't really lead to better mechanistic results. There were 41 mutual swap events in D-prime where both players started with different answers, both flipped, and they ended up swapping rather than converging. The persuader bonus was still firing on 100% of these because the condition "I was pre-correct and my peer adopted my answer" technically holds even when the pre-correct player themselves flipped during discussion. 

## Part 2: DPP, fixing the persuader bonus

If the persuader bonus is firing on cases where the pre-correct player capitulates, the logical next step is a stricter gate: only fire when the persuader actually holds their position through discussion (in addition to flipping your peer). This should result in mutual swap events no longer get rewarded. Instead, only the ~15% clean asymmetric persuasion cases retain the bonus.

**DPP results (iter 75):* Mutual swap dropped further to 30.6%. Productive persuasion held at 15.3%, same as D. math500_l5 pass@1 was 0.731 vs base 0.659, roughly comparable to D-prime.

So the composition of the training signal got cleaner, as we had fewer sycophancy events polluting the gradient while productive persuasion held steady. Net product is closer to what the reward formula was supposed to be doing.

However, **adversarial robustness collapsed.** DPP's regressive flip rate on the 86-problem intersection is 0.062, roughly 5× D's 0.012, basically back to C's 0.047. The key mechanistic finding of D's distinctive adversarial robustness doesn't survive the clean reward.

This was pretty confusing to me at first. If D's behavior was supported by sycophancy events, you'd expect removing the sycophancy reward to _improve_ robustness, not destroy it. The probing experiment gave me a better idea of what was actually going on.

## Part 3: Mechanism probing

I re-extracted the three contrastive axes (commitment, reflection, peer-framing) on DPP using the same setup as the paper.

**Reflection axis (peak L36):**

|condition|norm|norm_ratio|cos to base|
|---|---|---|---|
|base|110.04|0.652|n/a|
|S|115.49|0.730|+0.844|
|D|116.78|0.715|+0.978|
|C|105.89|0.674|+0.913|
|DPP|111.12|0.680|+0.986|

DPP's reflection axis sits at base. The norm is base+1, the norm_ratio is base+0.03, and the cosine-to-base is the highest of any trained condition (0.986). D trained the axis up; DPP didn't move it.

**Commitment axis (peak L32):** DPP norm 47.3 vs D's 35.2. DPP's commitment direction is _stronger_ than D's, and the random-label sanity check passes cleanly (0.265 vs D's flagged 0.357). The stricter persuasion bonus produces cleaner commitment encoding because the reward is no longer being applied to events where commitment fails.

**Peer-framing axis:** DPP 3.39, basically identical to D's 3.25 (CI overlap). No change. So the stricter persuasion bonus didn't move how the model represents peer-conditioned contexts.

So here is my analysis of what is happening: The persuader bonus is a +0.3 reward applied to the _pre-discussion correct rollout_ of the winning player. It does not touch the post-discussion rollouts at all. So the gradient signal is always pointing in one direction: "commit confidently to your initial correct answer." In D, this bonus fired often because the gate was loose, including on mutual swap events where the pre-correct player technically had a peer adopt their answer (even though they themselves flipped). DPP's strict gate cuts these firings by roughly 5×. The gradient direction is the same, just much less of it. The reflection axis is what the model uses to hold its position when challenged, and it apparently needs a lot of this gradient pressure to actually develop. D got enough volume to train the axis up; DPP didn't, so DPP's reflection axis sits at base and adversarial robustness collapses.

Would also be curious to hear your thoughts/whether you agree with this analysis or if there is something I'm missing.

## Analysis
The persuader bonus has two properties: how _often_ it fires (volume) and how _clean_ each firing is (purity). In D, the bonus fired often but most firings were on messy sycophancy events. DPP cleans up each firing but fires 5× less often. Same gate controls both, so there is tension between the two (volume vs purity of signal). 

The two axes we measured care about different properties. The reflection axis (D's adversarial robustness) needs volume: it trained up in D because the bonus fired a lot, even though most firings were dirty. It didn't train in DPP because the firings, while clean, weren't frequent enough. The commitment axis works the opposite way: it cares about purity, so DPP's commitment encoding is cleaner than D's.

This isn't the clean positive result I was hoping for, but I at least have a much better understanding of this problem and what the goal is. I now have direct mechanistic evidence (not just behavioral) for which reward property drives which mechanism. To get both behaviors at once, I'd need to grow the rate of clean events well above 15%, so a strict gate still produces enough firing volume to train the reflection axis. That's what the next steps target. 

## Next steps

Here are three things worth trying, in priority order.

**First, more volume under the strict gate.** Keep DPP as is, but train longer or feed more problems per iteration. More disagreements means more chances for the clean 15% to fire, even without re-introducing sycophancy. If volume alone is what the reflection axis needs, this should at least partially recover the robustness.

**Second, fix the productive persuasion rate.** The reason DPP is volume-starved is that productive persuasion is only 15% of disagreements. Heterogeneous Bob was supposed to push this number up, but the data shows it didn't move. Bob might not be different enough from Alice (SFT didn't push him far enough) or the initialization might be off. This is expensive but is likely worth the time. I will likely need some feedback on what I've tried so far and what I should try next, so I will reach out with some ideas.

Basically: the goal is to keep mutual swap low while increasing the productive persuasion rate. If I can hit 30% productive persuasion while keeping mutual swap low, that would likely produce a much stronger D.

**Another item: a confidence-amplification ablation.** If the reflection axis just needs volume of "commit confidently to a correct answer" gradient signal, you don't actually need a peer for that. You could look at any single rollout, check if it's confidently correct (low entropy, consistent across samples), and apply the bonus directly. If this recovers D's flip rate, the discussion protocol was just a fancy way of generating "confident correct rollout" events, and you can get the same benefit without any of the multi-agent machinery. 

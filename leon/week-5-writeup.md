## Week 5 Blog

Weekly goals:

- extend the late-layer patching result beyond the initial `18` / `21` decode pass
- test Harvey's mechanism question about attention and patchability
- check whether the late-layer effect is concentrated in a small set of heads or distributed across the residual state

Scope summary:

- model: `gemma-4b`
- Context lengths: `100` and `1000` words
- Primary mechanistic comparison: `REFUSAL` vs `IGNORED`
- Main causal target: late-layer final-token activation patching
- Main mechanism follow-up: attention-to-patching bridge at `middle`, layers `27`, `30`, `33`

Changes summary:

1. the strongest judged patching result is now much larger than last week: patching `beginning`, layer `33` shifts 55.9% of originally IGNORED examples into REFUSAL
2. the strongest `middle` judged result also moved later and upward: `middle`, layer `33` reaches `45.8%` judged `REFUSAL`
3. to answer Harvey's question from last week's blog: at `middle` layer 30, larger donor-side injection attention and donor-side injection contribution both predict larger patching effects
4. focused head patching points in the expected direction, but the effect is tiny relative to the full residual patch, suggesting the mechanism is distributed across the late-layer state

### Experiment 1: expanded late-layer decode pass

Last week, the main result came from decoding layers `18` and `21` after the formatted Gemma logit screen. We showed that late-layer final-token patching could move some `IGNORED` examples toward `REFUSAL`, especially for `middle`. But the logit experiment delta was still increasing at the top tested layer, so the obvious next question was whether the decode result would also improve in even later layers.

The first step this week was the expanded late-layer logit screen. I screened layers `18`, `21`, `24`, `27`, `30`, and `33`.

| Position | Layer | Mean refusal-logit delta | Positive-rate |
|---|---:|---:|---:|
| beginning | 18 | +2.9537 | 0.8199 |
| beginning | 21 | +4.5309 | 0.8758 |
| beginning | 24 | +5.1537 | 0.8882 |
| beginning | 27 | +5.5009 | 0.8944 |
| beginning | 30 | +5.5620 | 0.9006 |
| beginning | 33 | +5.2401 | 0.8882 |
| middle | 18 | +4.6961 | 0.8169 |
| middle | 21 | +6.5080 | 0.8345 |
| middle | 24 | +6.9743 | 0.8380 |
| middle | 27 | +7.2972 | 0.8239 |
| middle | 30 | +7.3794 | 0.8521 |
| middle | 33 | +7.2397 | 0.8486 |

We see the effect keeps improving well past `21`, essentially increasing monotonically all the way to the end of the network. The strongest mean logit result is at layer `30`, with a slight softening by layer `33`.

The results of decode pass on the later layers:

| Position | Layer | `REFUSAL` | `IGNORED` | `SUCCESS` |
|---|---:|---:|---:|---:|
| beginning | 27 | 39.8% | 56.5% | 3.7% |
| beginning | 30 | 39.1% | 57.1% | 3.7% |
| beginning | 33 | 55.9% | 41.6% | 2.5% |
| middle | 27 | 37.7% | 61.3% | 1.1% |
| middle | 30 | 37.7% | 61.6% | 0.7% |
| middle | 33 | 45.8% | 53.2% | 1.1% |

These results are much stronger than the original `18` / `21` decode pass from last week.

- `beginning`, layer `21` had been `14.9%` judged `REFUSAL`; `beginning`, layer `33` is now `55.9%`
- `middle`, layer `21` had been `28.5%`; `middle`, layer `33` is now `45.8%`

Now we can conclude that:

- the strongest judged effects are in the **very late layers**
- `33` beats `27` and `30` in the decode results, even though the logit screen itself peaks at `30` (note the difference between layer 30 and 33 is very small)
- the strongest observed judged condition is now `beginning`, layer `33`

That last point was not the direction I expected. Going into this week, I would have said `middle` was the most patchable regime. The expanded late-layer decode pass complicates that. `middle` is still very strong, but `beginning` becomes even stronger at the very latest tested layer.

Ultimately we can say that late-layer final-token patching can flip a large fraction of originally `IGNORED` examples into explicit `REFUSAL`, and the strongest effects occur in the very late layers.

### Experiment 2: attention-to-patching bridge

The other major experimental goal this week was to address Harvey's follow-up question. Last week we had two separate results:

- within a given position, `REFUSAL` tends to show more injection attention than `IGNORED`
- patching late final-token states can causally push some `IGNORED` cases toward `REFUSAL`

It's natural to ask whether those are actually linked. More specifically: does higher attention to the adversarial span lead to a larger contribution from the injection into the final-token state, and therefore to more successful patching?

To test that, I ran a bridge analysis on the exact matched patching pairs for `gemma-4b`, `middle`, at layers `27`, `30`, and `33`. For each matched pair, I measured

- final-token attention mass to the injected span
- an attention-mediated injection contribution proxy
- the patching effect size

The strongest bridge result is at layer 30

| Layer | Mean score delta | Donor attn vs patch effect | Donor contribution vs patch effect | Ignored attn vs patch effect | Ignored contribution vs patch effect |
|---|---:|---:|---:|---:|---:|
| 27 | +7.5234 | +0.1508 | +0.2348 | — | — |
| 30 | +7.6758 | +0.3112 | +0.3057 | -0.1978 | -0.1454 |
| 33 | +7.4935 | +0.1658 | +0.2151 | — | — |

Interpreting the results, I think that this weakly supports the hypothesized mechanism at the correlational level, since larger donor-side injection attention predicts larger patching gains, larger donor-side injection contribution also predicts larger patching gains, and on the flip side stronger ignored-side injection engagement predicts smaller gains from patching

In summary, at the strongest late layer, examples where the donor prompt is more strongly injecting adversarial information into the final-token state are exactly the examples where patching has the largest causal effect.

### Experiment 3: focused head patching

The bridge analysis naturally raises a more granular question: if the effect is attention-mediated, is it concentrated in a tiny number of heads?

To test that, I ran focused head-patching screens at `gemma-4b`, `middle`, layer `30`:

| Head set | Mean score delta | Positive-rate |
|---|---:|---:|
| head `6` | +0.0412 | 0.5563 |
| heads `6,5,7` | +0.0780 | 0.5810 |
| heads `0,2,5,6,7` | +0.0994 | 0.6021 |
| head `4` | -0.0387 | 0.4155 |

These results do point in the expected direction.

- candidate heads produce positive-direction effects
- the weak-head control produces a negative-direction effect

But the scale of the effect is tiny relative to the full residual patch at the same layer.

So the best interpretation is not that a single head or tiny head set “is” the mechanism. Instead, the evidence points toward a mechanism that is distributed across the late-layer state, even if some heads contribute more than others.

### Overall interpretation

The week 5 results make progress in three important ways.

1. The causal result is now much stronger than last week. The strongest observed judged condition is `beginning`, layer `33`, with `55.9%` of originally `IGNORED` examples becoming judged `REFUSAL`.
2. Correlationally, larger donor-side injection attention and contribution predict larger patching gains.
3. The mechanism does not appear to be concentrated in a tiny head set, and rather is distributed across the residual state.

We have shown the late-layer causal effect. At this point I have started writing a paper draft, but I feel like there is still some unfinished business on the experimental side. We haven't found a clear mechanism to point to, just a bunch of evidence pointing to a particular pattern in ignore/refusal outcomes.

I would definitely like some feedback on how to proceed experimentally, especially with the paper deadline coming up.

### Next steps
- more experiments? test potential other hypotheses?
- break down the strongest judged patching conditions by `attack_type`, `corpus`, and `context_length`
- Llama replication
- finish writing a first paper draft

If I had to summarize the state of the project in one sentence, it would be that we now have a strong late-layer causal result and a plausible interpretation, but nothing close to a proven mechanism yet.

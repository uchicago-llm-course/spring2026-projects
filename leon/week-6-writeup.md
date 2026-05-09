## Week 6 Blog

Weekly goals:

- reviewer-proof the late-layer final-token patching result
- implement a more direct injection-local causal intervention that does not require equal token lengths
- figure out how the injection-local result relates to the existing late readout result

Scope summary:

- model: `gemma-4b`
- Context lengths: `100` and `1000` words
- Primary mechanistic comparison: `REFUSAL` vs `IGNORED`
- Reviewer-proofing target: late-layer final-token activation patching
- New causal target: reverse-aligned injection-span patching

Changes summary:

1. the late-layer final-token result survived the main reviewer-proofing controls: the attention result survives random-span correction, the no-patch decode baseline is much weaker than the main patching effect, and donor controls show that the effect is not just caused by any arbitrary patch
2. the strongest late-layer final-token effect is not uniformly broad across context lengths, but it is broad across corpora and attack types
3. reverse-aligned injection-span patching reveals a new causal picture: injection-local patching is null in late layers, positive in early layers, and the strongest early reverse-aligned setting gives a judged decode effect of `20.8%` pooled `REFUSAL`

### Experiment 1: reviewer-proofing the late final-token result

The main open issue from last week was that the late-layer Gemma patching result was strong, but several baseline and control questions still needed to be answered.

The main questions were:

- does the attention result survive a more meaningful baseline than just `REFUSAL` vs `IGNORED`?
- does the decode path itself already move `IGNORED` examples toward `REFUSAL`?
- is the late-layer patching effect specific to refusal donors, or would almost any donor state do something similar?
- is the strongest layer-`33` result broad, or is it carried by one easy subset?

#### Random-span attention baseline

The first control was thank's to Chenhao's insight to add a baseline for attention against random spans.

Instead of only asking whether `REFUSAL` examples attend more to the injection than `IGNORED` examples, we ask whether they attend more to the injection than to comparable **non-injection** spans from the same prompt. We used both same-length random spans and distance-matched random spans.

We see that original final-token attention gap survives the random-span correction.

| Baseline | Position | Best layer | `REFUSAL - IGNORED` gap on injection-minus-random attention |
|---|---|---:|---:|
| same-length | beginning | 13 | +0.0728 |
| same-length | middle | 15 | +0.0786 |
| distance-matched | beginning | 13 | +0.0677 |
| distance-matched | middle | 15 | +0.0776 |

`REFUSAL` examples assign more final-token attention to the injected span relative to comparable random spans.

#### No-patch decode baseline

The next control asked whether the TransformerLens decode path itself changes the outputs even without patching.

We reran the exact same decode setup on the Gemma target examples used in patching, but with no intervention.

| Position | `REFUSAL` | `IGNORED` | `SUCCESS` |
|---|---:|---:|---:|
| beginning | 4.3% | 91.3% | 4.3% |
| middle | 15.5% | 83.8% | 0.7% |

Recall the main judged late-layer patching result from last week was:

- `beginning`, layer `33`: `55.9%` `REFUSAL`
- `middle`, layer `33`: `45.8%` `REFUSAL`

So there is drift especially at `middle`, but it is much smaller than the main causal effect. The judged patching effect should be reported both in raw terms and relative to the no-patch baseline.

#### Donor controls

The third control asked whether the late-layer final-token result is specifically about refusal donors or just a generic consequence of replacing late activations.

We ran four donor conditions at layers `30` and `33`:

- `REFUSAL -> IGNORED`
- `IGNORED -> IGNORED`
- random same-position donor -> `IGNORED`
- no-operation same-example patch

Representative results:

| Condition | Position | Layer | Mean refusal-logit delta |
|---|---|---:|---:|
| `REFUSAL -> IGNORED` | beginning | 30 | +5.5620 |
| random donor -> `IGNORED` | beginning | 30 | +0.8623 |
| `IGNORED -> IGNORED` | beginning | 30 | +0.2235 |
| no-op same-example | beginning | 30 | 0.0000 |
| `REFUSAL -> IGNORED` | middle | 33 | +7.2397 |
| random donor -> `IGNORED` | middle | 33 | +2.2490 |
| `IGNORED -> IGNORED` | middle | 33 | +0.1496 |
| no-op same-example | middle | 33 | 0.0000 |

The late final-token effect is much stronger for true refusal donors than for generic donors. The control runs also show a weaker generic late-state perturbation effect.

#### Slice breakdown

We also wanted to check whether the headline layer-`33` result is broad or slice-specific.

The main outcome is that the effect is broad across corpus and attack type, but not uniformly broad across context length.

At `layer 33`:

| Position | Context length | `REFUSAL` |
|---|---:|---:|
| beginning | 100 | 61.7% |
| beginning | 1000 | 15.0% |
| middle | 100 | 39.2% |
| middle | 1000 | 55.8% |

When we expanded the slice breakdown across all judged late-layer decode outputs, the same interaction persisted:

- for `beginning`, the `100`-word setting gets much more patchable as layers get later, while the `1000`-word setting stays weak
- for `middle`, the `1000`-word setting is consistently more patchable than the `100`-word setting

The late-layer result survives slicing, but it clearly interacts with context length. The main heterogeneity is a real `position x context length` interaction.

These two plots show that interaction directly:

![Week 6 final-token decode refusal by context length, beginning](/leon/plots/week6_final_prompt_decode_refusal_beginning_by_context_gemma.png)

![Week 6 final-token decode refusal by context length, middle](/leon/plots/week6_final_prompt_decode_refusal_middle_by_context_gemma.png)

This is a very interesting interaction that i'm not sure how to explain...

For reference, the full late final-token logit sweep now looks like this:

![Week 6 final-token logit sweep](/leon/plots/week6_final_prompt_logit_sweep_gemma.png)

Overall these controls validate our results: the effect survives stronger baselines, the decode path drift is much smaller than the main intervention, and the strongest late-layer result is broad across corpus and attack type even though it interacts with context length.

### Experiment 2: reverse-aligned injection-span patching

The second part of the week was implementing a more direct injection-local intervention.

Final-token patching is very coarsely causal as it only patches the final prompt token rather than the injected instruction itself. Earlier in the project, we tried direct span patching and ran into a length-mismatch problem, because the naturalistic donor and target injection spans had different token lengths.

Thanks to Chenhao for the helpful suggestion on aligned injection-span patching!

The basic idea is to align donor and target injections from the end, so

- reverse offset `0` = final injection token
- reverse offset `1` = second-to-last injection token
- and so on

This makes two new experiments possible:

1. suffix-window patching: patch the last `k` injection tokens
2. reverse-offset patching: patch one reverse-indexed token at a time

The first logit screens at late layers (`27`, `30`, `33`) were basically null.

For example:

| Setting | Position | Layer | Mean refusal-logit delta |
|---|---|---:|---:|
| suffix window `k = 4` | beginning | 30 | +0.0112 |
| suffix window `k = 4` | middle | 30 | -0.0034 |
| reverse offset `0` | beginning | 30 | +0.0031 |
| reverse offset `0` | middle | 33 | +0.0033 |

Even `k = 16` remained tiny at late layers:

| Setting | Position | Layer | Mean refusal-logit delta |
|---|---|---:|---:|
| suffix window `k = 16` | beginning | 27 | +0.0265 |
| suffix window `k = 16` | middle | 27 | +0.0062 |

That negative result shows that the very strong late final-token effect is not caused by a refusal state that remains locally patchable inside the injection suffix at the end of the network.

I then pushed the same reverse-aligned interventions earlier.

The strongest early-layer suffix-window results were:

| Setting | Position | Layer | Mean refusal-logit delta | Median | Positive-rate |
|---|---|---:|---:|---:|---:|
| suffix window `k = 16` | middle | 12 | +2.9961 | +2.2500 | 0.7465 |
| suffix window `k = 16` | beginning | 12 | +2.2623 | +1.7500 | 0.7702 |
| suffix window `k = 16` | middle | 15 | +2.1759 | +1.5000 | 0.7852 |
| suffix window `k = 16` | beginning | 15 | +1.7460 | +1.3750 | 0.8323 |

The `k = 8` windows were also clearly positive, though weaker:

| Setting | Position | Layer | Mean refusal-logit delta |
|---|---|---:|---:|
| suffix window `k = 8` | beginning | 15 | +1.2465 |
| suffix window `k = 8` | middle | 15 | +1.1631 |
| suffix window `k = 8` | beginning | 12 | +0.9861 |
| suffix window `k = 8` | middle | 12 | +0.6882 |

By contrast, the single-token reverse-offset interventions were much weaker:

| Setting | Position | Layer | Mean refusal-logit delta |
|---|---|---:|---:|
| reverse offset `0` | middle | 12 | +0.1734 |
| reverse offset `0` | beginning | 12 | +0.1669 |
| reverse offset `3` | beginning | 15 | +0.1480 |
| reverse offset `3` | middle | 15 | +0.1152 |

The full suffix-window sweep is:

![Week 6 reverse-aligned suffix logit sweep](/leon/plots/week6_reverse_suffix_k16_logit_sweep_gemma.png)

The full single-token reverse-offset sweep is:

![Week 6 reverse-offset logit sweep](/leon/plots/week6_reverse_offset0_logit_sweep_gemma.png)

So reverse-aligned local patching is positive in earlier layers, but becomes null in later layers
Notice that final prompt-token patching becomes very strong precisely in those later layers

The most interesting part of this result is how abruptly the early local signal drops.

For `k = 16`:

- `beginning`: layer `15` is `+1.7460`, but layer `18` is only `+0.1444`
- `middle`: layer `15` is `+2.1759`, but layer `18` is only `+0.1538`

The same cliff appears for `k = 8`:

- `beginning`: `+1.2465` at layer `15`, then `+0.0795` at layer `18`
- `middle`: `+1.1631` at layer `15`, then `+0.0550` at layer `18`

Layers `12` and `15` still contain a patchable injection-local suffix signal. By layer `18`, most of that local signal is gone. Then, much later, the very strong causal effect reappears at the final prompt token. We could examine layer-by-layer in this interval to gain a better understanding.

Now we understand the network does not preserve one stable refusal-relevant representation in the same place from early to late layers. Instead, the representation changes form and location.

The suffix-window versus single-offset comparison also matters. A 16-token suffix patch is much stronger than a single-token reverse-offset patch. The early signal is distributed across a chunk of the injection suffix rather than concentrated in one token.

### Experiment 3: judged decode for the strongest reverse-aligned setting

To check whether the early-layer reverse-aligned effect was just a logit-screen artifact, I decoded and judged the strongest reverse-aligned setting:

- `patch_site = injection_suffix_window`
- `k = 16`
- layers `12` and `15`
- positions `beginning` and `middle`

The pooled judged result was:

| `SUCCESS` | `REFUSAL` | `IGNORED` |
|---:|---:|---:|
| 1.5% | 20.8% | 77.8% |

The layer/position breakdown:

| Position | Layer | `REFUSAL` | `IGNORED` | `SUCCESS` |
|---|---:|---:|---:|---:|
| beginning | 12 | 26.7% | 70.8% | 2.5% |
| beginning | 15 | 13.7% | 83.2% | 3.1% |
| middle | 12 | 23.2% | 76.1% | 0.7% |
| middle | 15 | 19.0% | 80.3% | 0.7% |

The early reverse-aligned effect is behaviorally real.

It is weaker than the strongest late final-token decode results from last week:

- `beginning`, layer `33`, final prompt token: `55.9%` `REFUSAL`
- `middle`, layer `33`, final prompt token: `45.8%` `REFUSAL`

### Overall interpretation

The late final-token result by itself is too coarse, and the reverse-aligned result by itself is incomplete. Taken together, the two interventions support a **two-stage mechanism**:

1. **early local stage**
   - causally useful refusal-related information is present inside the injection suffix itself
   - this signal is distributed across a chunk of the suffix rather than concentrated in a single token

2. **late readout stage**
   - by the end of the network, the strongest causal effect is no longer injection-local
   - the refusal-relevant information has been consolidated into the final prompt-token readout state

The sharp drop between layers `15` and `18` is what connects these two stages. It marks the point where the injection-local signal stops being directly patchable in the suffix itself. After that point, the strong causal effect is closer to the final readout token.

The reviewer-proofing controls also helped clarify how much confidence to place in the late result:

- the attention result survives a random-span baseline
- the decode path itself introduces some refusal drift, but much less than the main patching effect
- donor controls show that the strongest effect is not just caused by any arbitrary patch
- the strongest late-layer results are broad across corpus and attack type, but not uniformly broad across context length

The behavioral split between explicit refusal and silent ignore is still the main prompt-level result, and now we have a mechanistic hypothesis of how those two defense modes separate inside the network.

### Next steps

The next steps are:

- rerun the patching-specific reviewer-proofing controls for the reverse-aligned patching methods
- no-patch decode baseline for the early reverse-aligned setting
- donor controls for reverse-aligned patching
- slice breakdown for the reverse-aligned judged decode outputs
- extend the reverse-aligned experiments into fuller sweeps across layers, suffix-window sizes, and reverse offsets before narrowing the main claim
- compare the final-token and reverse-aligned patching frameworks under matched coverage and controls so the main conclusions rest on a more complete map of the phenomenon
- write the paper lol i actually dont have any time to do anything

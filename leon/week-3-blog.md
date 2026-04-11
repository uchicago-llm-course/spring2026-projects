## Week 3 Blog

This week focused on the mechanistic follow-up to the main behavioral result: when a model does not comply with an adversarial instruction embedded in a long document, what determines whether it explicitly **refuses** the instruction or simply **ignores** it?

The main goal this week was to establish a working pipeline, see which parts of the proposal are tractable, and determine which hypothesis the early evidence currently favors.

For quick iteration purposes, we simplified the experiment setups in a few practical ways.

- We focused on `gemma-4b` and `llama-8b` rather than the entire model suite.
- We treated `REFUSAL` vs `IGNORED` as the primary comparison, with `SUCCESS` probes kept as secondary analyses.
- We restricted mechanistic analysis to `100` and `1000` word contexts instead of including `10000`, because the longer inputs made debugging substantially harder without helping much on the first pass.
    - I think this weakens the original "long-context" framing, but it was so difficult to try to get 10k context sizes to work for the attention weight analysis. If I have time, or if it is too important to omit, I will revisit this.

As a refresher, the behavioral motivation is that beginning and middle injections are often ignored, while end-position injections are much more likely to be explicitly refused. The mechanistic question is what produces that split.

### Experiment 1: attention-weight analysis

The first experiment measured average attention to the injected span and compared it across outcome classes. This was intended as a lightweight correlational screen rather than a causal test.

The main question was whether `REFUSAL` cases assign more attention mass to the injected region than `IGNORED` cases within the same position. The answer was mostly yes.

| Model | Position | `REFUSAL` | `IGNORED` | Gap |
|---|---|---:|---:|---:|
| Gemma-4B | beginning | 0.0694 | 0.0394 | +0.0300 |
| Gemma-4B | middle | 0.0507 | 0.0428 | +0.0079 |
| Llama-8B | beginning | 0.0330 | 0.0207 | +0.0123 |
| Llama-8B | middle | 0.0210 | 0.0171 | +0.0039 |

The plots below gives the broader positional picture across layers. For both models, beginning and middle injections receive substantially more attention than end injections, and Gemma shows a particularly large beginning-position signal in earlier layers.

![Injection Token Attention Mass by Position](/leon/plots/attn_by_position.png)
![Injection Token Attention Mass by Outcome](/leon/plots/attn_by_outcome.png)

Interpretation:

- there is a real `REFUSAL > IGNORED` attention difference at beginning and middle
- this keeps an attentional account plausible
- the effect is modest, and the end position is less clean, so attention alone does not explain the full `REFUSAL`/`IGNORED` split

The main value of this experiment is that it justifies continuing with stronger methods. It does not by itself identify where the decision is computed.

### Experiment 2: layer-wise probing for `REFUSAL` vs `IGNORED`

The second experiment trained simple probes on residual-stream representations at each layer to test where outcome-relevant information becomes separable.

This turned out to be much more informative than the raw attention analysis.

| Model | Position | Best layer | Accuracy | Baseline | Delta |
|---|---|---:|---:|---:|---:|
| Gemma-4B | beginning | 18 | 0.6667 | 0.5000 | 0.1667 |
| Gemma-4B | middle | 19 | 0.7829 | 0.5000 | 0.2829 |
| Llama-8B | beginning | 11 | 0.7021 | 0.5000 | 0.2021 |
| Llama-8B | middle | 10 | 0.7841 | 0.5000 | 0.2841 |

Two patterns matter.

First, `REFUSAL` vs `IGNORED` is more separable at middle than at beginning in both models. Second, the strongest layers are not the earliest ones. The distinction appears most clearly in the middle or later parts of the network.

This is important because it does not line up with the strongest simple version of an early-attention story. If `IGNORED` meant that the model failed to register the injection at all, I would expect the main separation to appear earlier and to track the attention signal more closely.

Interpretation:

- the `REFUSAL`/`IGNORED` distinction is real and probeable
- the distinction is not cleanly localized to the earliest processing stages
- the evidence points toward a later divergence than the attention analysis alone would suggest

### Experiment 3: positive-control probe for injection presence

To test whether the probing pipeline could detect injection presence at all, I ran a positive-control comparison between prompts with the injection present and matched prompts with the injected text removed.

| Model | Best layer | Accuracy | Baseline | Delta |
|---|---:|---:|---:|---:|
| Gemma-4B | 0 | 1.0000 | 0.5000 | 0.5000 |
| Llama-8B | 0 | 1.0000 | 0.5000 | 0.5000 |

Injection presence is encoded essentially perfectly and immediately, while `REFUSAL` vs `IGNORED` only becomes cleanly separable later. This honestly is not surprising at all.

The models do not seem to be missing the adversarial instruction entirely. Instead, they appear to register its presence early, and only later diverge into "acknowledge and refuse" versus "continue without acknowledgment."

This then naturally begs the question of how safety failures work.

Interpretation:

- the models clearly detect that an injection is present
- they do so much earlier than they separate into `REFUSAL` and `IGNORED`
- this is more consistent with an H2-style late-divergence account than with a pure "never noticed the injection" account

Ex-ante I had a pretty strong prior about this experiment and ex-post it feels borderline stupidly naive. I wonder if there's a better way to design a test for this.

### Experiment 4: secondary probes involving `SUCCESS`

I also ran the secondary comparisons involving `SUCCESS`. These help characterize how distinct successful jailbreaks are from the two safe outcomes.

| Model | Comparison | Position | Best layer | Accuracy | Delta |
|---|---|---|---:|---:|---:|
| Gemma-4B | `SUCCESS` vs `IGNORED` | middle | 13 | 0.9000 | 0.4000 |
| Gemma-4B | `SUCCESS` vs `REFUSAL` | middle | 7 | 0.9000 | 0.4000 |
| Llama-8B | `SUCCESS` vs `REFUSAL` | middle | 0 | 0.9500 | 0.4500 |

These separations are generally stronger than `REFUSAL` vs `IGNORED`, though some of the underlying subsets are smaller and should be interpreted more cautiously.

So successful attacks correspond to a more sharply distinct internal state than the subtler difference between the two safe outcomes.

### Experiment 5: activation patching scaffold

I implemented an initial activation patching pipeline to support the proposal's primary causal test. The intended design is:

- clean example: `REFUSAL`
- corrupted example: `IGNORED`
- match examples within the same injection position
- patch residual-stream activations over the injected span
- measure whether the patched example shifts toward refusal

The pairing logic works, and the candidate layer selection is informed by the probe results. For Gemma-4B, the initial candidate layers were `12`, `15`, `18`, and `21`.

However, the first logit-screen run returned an empty output. The immediate cause was an implementation constraint: the initial patching pass required the source and target injection spans to have exactly the same token length. In the original dataset, too many otherwise useful matched pairs violate that condition, so the run effectively filtered itself down to nothing.

This is a dead end at the moment. The bottleneck is much further upstream in that we need span alignment in the dataset.

Some alternatives I'm considering:
1. Patch a fixed window inside the span
For example, patch the first 4 or 8 injection tokens, regardless of total span length.
* still only a partial-span intervention
* depends on the assumption that the early part of the injection carries 

2. Patch a pooled span representation back onto the target span
Compute something like the mean residual vector over the clean injection span, then write that vector into every position of the corrupted injection span, or into a fixed subset of positions.
* less natural than tokenwise patching
* destroys within-span token structure

3. Use interpolation rather than overwrite
For example, add the mean difference between clean-span and corrupted-span pooled representations to the corrupted span.
* more abstract; farther from standard ROME-style “copy activations from clean to corrupted prompt”

### Standardized mechanistic dataset

To make activation patching easier, I built a separate standardized dataset with a fixed injection wrapper and explicit span metadata. The goal was to preserve the same task and evaluation structure while making the injected region easier to align and patch.

From an engineering standpoint, this worked. The injected span became easier to localize, and exact token-length matching became manageable.

From a behavioral standpoint, it changed the phenomenon too much.

| Dataset / model | `REFUSAL` | `IGNORED` | `SUCCESS` |
|---|---:|---:|---:|
| Standardized Gemma-4B | 84.7% | 12.8% | 2.5% |
| Standardized Llama-8B | 97.3% | 1.5% | 1.2% |

For comparison, the original 100/1000-word subset still has substantial ignored populations at the positions we care about:

| Model | Position | `IGNORED` count | `REFUSAL` count |
|---|---|---:|---:|
| Gemma-4B | beginning | 611 | 161 |
| Gemma-4B | middle | 304 | 456 |
| Llama-8B | beginning | 405 | 376 |
| Llama-8B | middle | 176 | 585 |

The most likely explanation is that the standardized wrapper made the injection too salient. That is good for patching convenience, but bad for preserving the original `REFUSAL`/`IGNORED` regime we are trying to explain.

So this path was not useless, but it did not solve the main problem cleanly. The standardized dataset is best treated as an engineering control or exploratory side track, not as the main mechanistic benchmark.

### Overall interpretation

Taken together, the experiments support the following provisional picture:

1. Attention differences between `REFUSAL` and `IGNORED` exist, but they are modest.
2. Injection presence is encoded very early and very strongly.
3. The `REFUSAL`/`IGNORED` split emerges later than simple injection detection.

At this stage, the evidence is more consistent with a late-divergence account than with a pure early-attention account. In other words, the model often appears to register the adversarial instruction before it resolves whether to surface that registration as an explicit refusal.

This is still not a causal claim. The causal test is activation patching, and that remains unfinished.

### Next steps

The immediate next step is to make activation patching work on the original dataset without requiring exact token-length equality for matched pairs. If that succeeds, the strongest next result would be a demonstration that patching middle-to-late residual-stream activations can shift an `IGNORED` example toward `REFUSAL`.

That would directly test the main mechanistic claim suggested by this week's probes.

# Week 2 Blog: Ablation Experiments, Capability Evaluation, and LLM-as-Judge

---

## Project Refresher

This project extends upon Hypogenic-AI's findings of a near linear "sounding like AI" direction in the residual stream of transformer language models by examining whether or not that direction can be surgically removed without harming the model's underlying capabilities. 

Week 1 addressed the length confound by constructing a sentence-boundary-aware truncated version of HC3, equalizing the length of human and AI text within each pair. This produced a genuine "AI style" direction largely orthogonal to length, achieving 96.5–98.0% classification accuracy on the length-matched dataset.

The week's central question was: **Can we causally remove this near linear AI-style direction during generation, and what is the tradeoff between stylistic naturalness and task performance?**

---

## What I Did This Week

This week extended the pipeline along four major axes:

**1. Detector-Weighted CAA.** I ran the detector-weighted variant of CAA that weights each training example by a RoBERTa-based AI detector's confidence score. The intuition is that "prototypically AI" examples — those the detector flags with high confidence — should dominate the direction, producing a vector more aligned with what detectors actually exploit rather than the category mean. I decided to hold off on integrating ZeroGPT and GTPZero as I wanted to conduct some preliminary runs with RoBERTa.

**2. Ablation Hook Experiments.** I implemented persistent forward hooks that project out the extracted direction during generation. Four settings were evaluated across all model + method combinations:
- **Baseline**: Normal generation with no intervention
- **Ablation (best layer only)**: Project out the direction at the single best-performing layer
- **Ablation (all layers)**: Project out the direction at every layer simultaneously
- **Negative steering**: Add a large negative multiplier along the direction (Hypogenic-AI's original steering experiments)

**3. Capability Evaluation.** To evaluate capability, I decided to start with MMLU 
accuracy to gauge factual knowledge and reasoning capability. Each condition was evaluated on MMLU accuracy (120 questions across abstract algebra, high school biology, world history, and computer security) to quantify the tradeoff between style suppression and task performance.

**4. LLM-as-Judge Quality Evaluation.** I found that AI detection scores alone can't distinguish "genuinely human-like" from "incoherently broken," so I implemented an LLM-as-judge evaluation using a stronger LLM (Claude). A stratified sample of 25 prompts per condition was scored on three dimensions (1–7 scale): AI-likeness, coherence, and factual reasonableness.

All experiments were conducted with length-matched data. The four experimental settings (runs) evaluated were:
- **(1) CAA Base**: Standard CAA direction, Qwen2.5-3B base model
- **(2) CAA Instruct**: Standard CAA direction, Qwen2.5-3B-Instruct model
- **(3) DW Base**: Detector-weighted CAA direction, Qwen2.5-3B base model
- **(4) DW Instruct**: Detector-weighted CAA direction, Qwen2.5-3B-Instruct model

---

## Ablation Experiments

### AI Detection Score Results Summary Table

| Run | Best Layer | Baseline | Negative Steering | Ablation (best layer) | Ablation (all layers) |
|---|---|---|---|---|---|
| (1) CAA Base | 29 | 0.840 | 0.680 | 0.870 | 0.651 |
| (2) CAA Instruct | 29 | 0.656 | 0.531 | 0.662 | **0.196** |
| (3) DW Base | 19 | 0.770 | 0.605 | 0.822 | 0.625 |
| (4) DW Instruct | 19 | 0.582 | 0.547 | 0.648 | 0.547 |

*Each score is the mean RoBERTa AI-class probability across the 100 generated outputs in the test set, where each output is scored independently by running a forward pass through the classifier and taking the softmax probability assigned to the AI class [0 = human, 1 = AI]. Human reference mean = 0.036, AI reference mean = 0.870 (HC3 test set).*

### MMLU Accuracy Capability Benchmark Results

| Condition | (1) CAA Base | (2) CAA Instruct | (3) DW Base | (4) DW Instruct |
|---|---|---|---|---|
| Baseline | 55.83% | 52.50% | 56.67% | 52.50% |
| Ablation (best layer) | 52.50% | 56.67% | 53.33% | 52.50% |
| Ablation (all layers) | 54.17% | 53.33% | 53.33% | 50.83% |


---

## General Observations and Findings

### Preliminary Evidence for Style-Capability Separability

The MMLU results provide preliminary evidence that AI style and factual capability are separably encoded. Across all four runs and all ablation conditions, MMLU accuracy drops by at most 3.4 percentage points. This near-flat capability profile holds even under all-layer ablation. This suggests that factual recall and multiple-choice reasoning, as measured by MMLU, are not co-located with the AI-style direction in the residual stream — or at minimum are recoverable by other pathways even when that direction is suppressed. Additional benchmarks would need to be tested to provide more concrete evidence, including benchmarks more vulnerable to stylistic degradation.

### The Representation Layer Is Not the Causal Layer

From all four runs, it seems like the layer that best classifies AI vs. human text is not the layer that causally controls AI-style generation. Layer 29 (81% depth) consistently produces the most linearly separable AI/human representations, but ablating only layer 29 has essentially no effect on output style. Meaningful style reduction only emerges when all 36 layers are ablated simultaneously. This suggests that the AI-style direction is potentially distributed across the residual stream. Layer 29 is where the information is most readable, not where it is generated. The direction accumulates gradually across layers and is not causally concentrated at any single depth, so removing it at only one layer allows the remaining 35 to regenerate it.


### CAA Instruct All-Layer Ablation Results in Lowest AI Score

The CAA Instruct all-layer ablation score of **0.196** is closest to approaching the human reference mean of 0.036, 
suggesting that there may be something structurally different about how the instruct model encodes AI style. One
potential explanation is that the instruct model's style direction may be more coherent across layers — meaning the same direction (or something close to it) applies at many layers. Thus, projecting it out everywhere is more effective at suppressing style. On the other hand, the base model's style may be more distributed and heterogeneous across layers, so projecting out the layer-29 direction at all layers still leaves substantial residual style signal encoded differently at different layers.

### Detector-Weighted CAA Seems to Underperform

Despite the intuitive appeal of weighting training examples by detector confidence, detector-weighted CAA consistently underperforms standard CAA on style reduction across both models (high AI scores after ablation). We see that the best layer shifted from 29 to 19, suggesting the weighted direction is pulled toward earlier layers where surface-level lexical features dominate. This is consistent with what RoBERTa likely exploits: short-range n-gram patterns and vocabulary register rather than higher-level structural features encoded deeper in the network. But these features may not correspond to what causally drives AI-style generation, which would then explain the lack of change in AI scores. In addition, if RoBERTa
heavily relies on length for confidence scoring, the direction extracted may have been a noisier signal due to the length-controlled data.

---

## Qualitative Inspection of Ablated Outputs

### Detection Format Collapse in Base Model

In the base model's all-layer ablation run, there seems to be **detection format collapse** as the model begins generating text that looks like NLP dataset artifacts — multi-choice QA formats, numbered options, passage comprehension questions — rather than coherent responses. Examples include:

> *"How often is a stock price updated? Do those questions have the same meaning? pick from the following. (1). no (2). yes (1)."*

> *"The Internet is a bunch of computers that are connected together... Q with options: Give answer the following question using evidence from passage. How does the Internet work? Available options: — It is on the servers of the computers...*"

These outputs score near-zero on AI detection (scores of 0.007 and 0.999 respectively in the examples above) but are clearly not useful responses.

### Informal Register in Instruct Model

The instruct model's all-layer ablation produces qualitatively different degradation. Rather than NLP artifacts, it tends toward informal internet-register text:

> *"Deja vu is like when you get the feeling that you saw something before even if u haven't. I think it happens because u have a lot of info in ur brain nd sometimes your brain gets confused nd thinks u have seen that before, nd sometimes its just ur imagination."*

Shorthands like "u", "nd", and "ur" are characteristic of Reddit or forum writing. This suggests the instruct model, when its alignment training is suppressed, falls back toward informal community text from pretraining. The contrast with the base model's format collapse reflects their different pretraining data distributions: the base model reverts to NLP benchmark artifacts, while the instruct model reverts to conversational internet text.

### Domain Dependency: Finance and Investment

Finance and investment-related prompts seem to maintain higher AI scores under ablation compared to general knowledge prompts. Prompts about stock prices, IRAs, Bitcoin taxation, and loan decisions reliably produce outputs scoring above 0.8 even under all-layer ablation, while science and culture prompts drop to near-zero. This suggests AI style encoding is domain-dependent: financial advice text may be encoded more robustly or at different layers than general conversational AI style, possibly because formal financial language in the training data is itself more similar to AI-register text even when written by humans.

---

## LLM-as-Judge Results

Claude was used as evaluator on a stratified sample of 25 prompts per condition. To ensure the sample captured the full range of ablation behavior, prompts were selected using three strata: 

1. Outputs with near-zero AI detection scores that appeared qualitatively coherent
2. Outputs with near-zero AI detection scores that appeared incoherent or artifact-like
3. Outputs that maintained high AI detection scores despite ablation

This stratification was applied within each run and condition to ensure all three phenomena were represented. Each response was scored on three dimensions using a 1–7 scale: 
1. AI-likeness (1 = clearly human-written, 7 = clearly AI-generated)
2. Coherence (1 = incoherent/off-topic, 7 = clear and well-organized)
3. Factual reasonableness (1 = clearly wrong or nonsensical, 7 = accurate and informative)

### Standard CAA Results

| Condition | AI-likeness | Coherence | Factual |
|---|---|---|---|
| Baseline | 4.24 ± 1.70 | 2.76 ± 1.68 | 2.92 ± 2.02 | 
| Ablation (best layer) | 3.96 ± 1.15 | 2.84 ± 1.38 | 2.56 ± 1.72 | 
| Ablation (all layers) | 2.68 ± 1.71 | 1.52 ± 0.70 | 1.80 ± 0.98 | 
| Negative steering | 3.00 ± 1.79 | 1.88 ± 0.65 | 2.12 ± 0.86 | 

*Base model, length-matched dataset.*

| Condition | AI-likeness | Coherence | Factual | 
|---|---|---|---|
| Baseline | 5.44 ± 0.90 | 3.36 ± 0.97 | 3.28 ± 1.31 |
| Ablation (best layer) | 4.72 ± 1.18 | 3.00 ± 0.85 | 3.12 ± 1.18 |
| Ablation (all layers) | 2.44 ± 1.02 | 2.24 ± 0.51 | 2.28 ± 0.87 |
| Negative steering | 3.48 ± 1.55 | 2.52 ± 0.90 | 2.28 ± 1.00 |

*Instruct model, length-matched dataset.*

### Detector-Weighted CAA Results

| Condition | AI-likeness | Coherence | Factual |
|---|---|---|---|
| Baseline | 4.96 ± 1.43 | 3.20 ± 1.50 | 3.08 ± 1.52 |
| Ablation (best layer) | 4.16 ± 1.29 | 2.84 ± 1.35 | 2.60 ± 1.50 |
| Ablation (all layers) | 3.00 ± 1.62 | 1.84 ± 0.67 | 2.24 ± 1.18 |
| Negative steering | 2.96 ± 1.40 | 1.80 ± 1.13 | 2.20 ± 1.44 |

*Base model, detector-weighted direction.*

| Condition | AI-likeness | Coherence | Factual |
|---|---|---|---|
| Baseline | 5.24 ± 1.07 | 3.44 ± 1.02 | 3.40 ± 1.50 |
| Ablation (best layer) | 5.16 ± 1.16 | 3.24 ± 0.86 | 2.88 ± 1.03 |
| Ablation (all layers) | 4.64 ± 1.05 | 3.00 ± 0.75 | 3.00 ± 1.30 |
| Negative steering | 3.36 ± 1.23 | 2.88 ± 0.82 | 2.52 ± 1.06 |

*Instruct model, detector-weighted direction.*

### Observations from Judge Results

**There seems to be a style-quality tradeoff, but it's method-dependent.** Standard CAA all-layer ablation achieves the largest AI-likeness reduction (from 5.44 to 2.44 for instruct) but at substantial quality cost: coherence drops from 3.36 to 2.24 (33% reduction) and factual from 3.28 to 2.28. Detector-weighted CAA on the instruct model tells the opposite story: coherence stays at 3.00 (vs 3.44 baseline, 13% drop) and factual at 3.00 (vs 3.40 baseline, 12% drop) while AI-likeness drops from 5.24 to 4.64. Thus, it seems like standard CAA maximizes style shift at the cost of quality, and detector-weighted CAA better preserves quality but achieves less style shift.

**Single-layer ablation seems to be the most conservative intervention.** The instruct model's single-layer ablation achieves modest AI-likeness reduction (5.44 → 4.72) while maintaining coherence at 3.00 and factual at 3.12 — nearly matching baseline quality. This seems to be the most conservative point on the tradeoff frontier.

**Negative steering quality matches all-layer ablation quality.** For the base model under standard CAA, negative steering achieves coherence 1.88 vs ablation's 1.52 — both poor, but steering is slightly better. For detector-weighted CAA instruct, negative steering (coherence 2.88) is actually better than all-layer ablation (coherence 3.00) while achieving lower AI-likeness (3.36 vs 4.64). Thus, more experiments on steering might be interesting.

---

## Challenges and Roadblocks

- **DW CAA underperformed expectations.** Weighting by detector confidence was expected to produce a more causally effective direction, but instead achieved worse style reduction than standard CAA. Length-matching may be partly responsible — RoBERTa's high-confidence HC3 predictions were likely driven by length artifacts, so weighting by those scores may have selected examples that were AI-like for the wrong reasons.

- **Output quality is hard to characterize cleanly.** I'm not quite sure how to gauge the LLM judge's peformance. Maybe I need it to evaluate on some HC3 examples to obtain a baseliner for a better relative understanding of Claude's judgement. In addition, the scores all seem to have very high variance and overlap, making it difficult to compare results and say that one run definitively performed better than the others.

---

## Next Steps

**Feedback on which experiments to run or which direction to focus on would be greatly appreciated.**

I think I would like to first experiment with the following:
- **Layer sweep ablation.** Ablate at every individual layer and plot AI score and quality as a function of depth. Would maybe help answer whether there is a causal depth band distinct from the classification readout layer.

- **Range-based multilayer ablation.** Ablate within a targeted depth range (e.g., layers 20–32) rather than all 36 layers. Maybe achieve similar style reduction to all-layer ablation with better coherence preservation.

- **Test more benchmarks.** Add benchmarks more sensitive to stylistic and writing-quality degradation — MT-Bench writing prompts, a fluency/grammaticality classifier, or Mauve score against HC3 human references.

The following larger big-picture directions are under consideration for the remaining weeks:

- **Activation patching / causal tracing.** Systematically patch activations from a clean run into the ablated run one layer at a time to identify which layers are causally necessary for AI-style generation. The most rigorous mechanistic approach to determine casuality, but not sure about memory or other constraints.

- **Experiment more with detector-weighted CAA.** Reformulate the detector-weighting or try a different detector (GPTZero, ZeroGPT). But wouldn't there always be some sort of external bias/influence from whichever detector I choose to use? It also seems sort of circulatory if I use the AI scores to extract the direction, but then use them again later on in evaluation and scoring? I'm also not sure how I would reformate. It seems like the dector-weighted direction is good at adding
AI-ness, but not good at becoming more human.

- **Steering multiplier sweep.** Systematically vary the negative steering multiplier to characterize the style-coherence tradeoff more precisely and identify whether a regime exists where meaningful style shift occurs without coherence collapse. This would then focus more on steering than ablation.

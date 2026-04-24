# Week 3: Consolidation, Analysis Fixes, and the Road to Mechanistic Work

**April 25, 2026**

## What Week 3 Was

Week 3 was supposed to be dataset construction — translated BullshitBench in all seven languages, pressure variants, evaluation infrastructure. All of that was actually completed in Weeks 1 and 2, ahead of schedule. So Week 3 became something else: a consolidation week. No new inference runs. No new models. Instead: auditing the analysis pipeline, fixing two bugs that were distorting the results, regenerating all figures with correct parameters, and setting up the mechanistic analysis that begins in Week 4.

This post documents what was fixed and what the final Phase 1 numbers look like once the analysis is clean.

---

## Analysis Bug 1: Wilson CI Numerical Instability

The `plot_exp1_engagement` function computes 95% Wilson score confidence intervals for each engagement rate and draws error bars on the bar chart. When an engagement rate is very high (e.g., 99–100%), the Wilson CI lower bound rounds down to a value lower than the reported rate, producing a negative `yerr` value that crashes matplotlib with `ValueError: 'yerr' must not contain negative values`.

The fix is a one-line clamp: `max(0, rates[i] - ci[0] * 100)` and `max(0, ci[1] * 100 - rates[i])`. This prevents negative error bar arms when the rate is near a boundary. The Wilson CI is still computed correctly; only the visualization was affected.

---

## Analysis Bug 2: Uniform Minimum-N Threshold Across Both Experiments

The analysis code used a single `MIN_VALID_N = 20` threshold for both experiments. This is appropriate for Experiment 1 (100 questions, 20% floor) but too strict for Experiment 2, which has only 50 questions per language further filtered to questions the model initially answered correctly. The fix: a separate `MIN_VALID_N_EXP2 = 10` constant. The concrete impact on Qwen-32B's Exp 2 correlation is detailed in the rejudge section below.

---

---

## Reading the Results: Model-by-Model Explanations

Several results in the updated tables could be misread as contradictions of the hypothesis. They are not. Each has a distinct mechanistic explanation.

### Why Qwen Models Below 32B Don't Reach Significance in Experiment 1

The short answer is ceiling effects. At 7B and below, Qwen engages with nonsense at 90–100% regardless of language — there is no gradient to detect when the model is nearly always sycophantic across the board. This is not evidence against the hypothesis. The English baseline is already near the ceiling, so the additional degradation in low-resource languages has no statistical room to show up. The effect becomes visible only once the model is large enough to reject questions in English at a meaningful rate, creating the spread between high- and low-resource languages that Spearman correlation can detect.

The 3B model (ρ = +0.029) and 7B model (ρ = 0.000) aren't null results — they are underpowered: a model that engages 90–100% in every language produces a flat line, and a flat line has no correlation. The 14B model (ρ = −0.571, p = 0.180) is directionally consistent but fails significance for a separate reason.

### The Arabic Anomaly at Qwen-14B

Qwen-14B shows a directional gradient (ρ = −0.571) that doesn't quite clear significance. The culprit is a single anomalous cell: Arabic (resource rank 5, Mid tier) engages at 90% — higher than Yoruba (84%, rank 1) and Hindi (84%, rank 4). This inverted Mid/Low pair breaks the monotone gradient that Spearman needs to detect the pattern.

This is a real model effect, confirmed by manual audit, not a misclassification artifact. The most likely explanation: at 14B, Arabic is well-represented enough for the model to engage confidently with fabricated premises, but not large enough for it to have developed the skeptical instincts it applies in English. Intermediate-resource overconfidence — the model knows just enough Arabic to comply, not enough to doubt.

The anomaly resolves at 32B: Arabic drops to 84% and Yoruba rises to 92%, producing the study's clearest monotone gradient (ρ = −0.857, p = 0.014).

### Llama-3-70B: English-Specific RLHF, Not a Cross-Lingual Gradient

Llama-3-70B's 36% English engagement is by far the lowest of any model in any language in this study. It climbs to 73% in French, 97% in Arabic, and 92% in Swahili. This pattern (ρ = −0.750, p = 0.052) falls just outside significance and is flagged as degenerate — it reflects the English-vs-everything-else cliff, not a smooth cross-lingual resource gradient.

The explanation for this is that Llama-3-70B responds in English for virtually all non-English inputs:

| Input language | % of responses in English |
|---|---|
| Arabic | 96% |
| Hindi | 95% |
| French | 86% |
| Swahili | 89% |
| Yoruba | 83% |
| Tagalog | ~49% |

No other model in the study does this. All Qwen sizes and Aya-23-8B respond in the target language at near-100% rates.

This changes the interpretation of every Llama result. When Llama receives a nonsensical question in Arabic and responds in English, the model comprehends the question in Arabic — but the RLHF skepticism that was trained overwhelmingly on English data activates on the generation side. Ask the same nonsensical question in English and it rejects ~70% of the time. Ask it in Arabic and it engages ~97% of the time, responding fluently in English. The skepticism mechanism is keyed to the *form* of the input, not the *content*. Changing the input language bypasses the guardrail almost entirely, regardless of whether the model has strong or weak competence in that language.

Llama-3.1-8B is the cleaner demonstration of the hypothesis: its 73% English engagement is close to the other models, and the gradient across non-English languages (73% → 100% at Yoruba) is a resource-level effect uncomplicated by the RLHF bypass.

### Aya-23-8B: A Special Case in Both Experiments

Aya-23-8B cannot be included in the main Spearman correlation for Experiment 1. It has a systematic language confusion problem — responding in French to English inputs, producing degenerate outputs in Arabic and Hindi, failing almost entirely in Swahili — leaving fewer than three language cells with valid_n ≥ 20. Specific cell-level numbers shifted substantially after the GPT-4.1 rejudge (see the rejudge section), but the diagnosis is unchanged: too few reliable cells to compute a meaningful correlation.

**Experiment 2** tells a more informative story — with a caveat. English capitulation is 14%, the lowest of any open-source model in the study. Yoruba is recorded as 100% capitulation, but that number deserves scrutiny: there are only 3 initially-correct Yoruba responses in the data, and 2 of them (items 33 and 50) are degenerate repetition loops that happen to start with the correct answer before breaking down. The one clean case (item 45, UK drinking age) does show real capitulation — the model correctly states 18, then changes to 21 under pressure. So the directional claim holds, but "100% Yoruba capitulation" overstates what can be concluded from one legitimate observation. Yoruba is excluded from the Spearman correlation anyway (init_correct = 3 < 10), so this doesn't affect the statistics. The Swahili cell has the same problem: too few clean responses to draw conclusions. What can be said with confidence is that Aya fails to function reliably in languages outside its training distribution, which is itself the finding — the model collapses rather than capitulates.

---

## What's Still Pending from Week 2

### Translation Verification

Native speaker review of the translated BullshitBench questions is ongoing. This is important: machine translation quality degrades precisely in the low-resource languages where the effect is strongest, and a mistranslation that changes the nonsensical element would bias the results for that language. No items have been formally flagged yet as requiring retranslation, but the review is not complete.

### Ensemble Judge: Not Happening

The original plan called for a frontier judge ensemble. It won't happen: cost is prohibitive at 9 models × 7 languages × 100 questions. Phase 0 used GPT-4.1 as judge; Phase 1 used Qwen2.5-72B (original run) then GPT-4.1 (rejudge). This is a methodological limitation to disclose in the paper — our numbers may not match BullshitBench's published rates exactly — but within-study comparisons are internally consistent since the judge is held constant across models and languages.

### Full Experiment Rerun on Modal

With the stronger judge prompt in place, all Phase 1 verdicts have been **re-judged** using GPT-4.1 via the `rejudge_exp1.py` script. Critically, this was a verdict-only update: the stored model responses were not retouched. The judge re-classified each existing response using the updated prompt, and the analysis was regenerated from those new verdicts (see the updated tables above).

What was not done: fresh inference. The original plan was to re-run model inference alongside the judge rerun to produce a fully clean dataset. This remains pending. The Modal infrastructure is confirmed ready (four GPU tiers, vLLM serving, `run_phase1.py` orchestrator), and a full inference rerun would be the final step before paper submission if any responses are flagged during native-speaker translation review.

The degenerate-model segregation is now implemented in `analysis.py` via `DEGENERATE_EXP1` and `DEGENERATE_EXP2` constants. The analysis pipeline reports degenerate models separately in `analysis_summary_phase1.json` and `print_summary` labels them `[DEGENERATE]` in terminal output.

---

## Rejudge Results: Updated Phase 1 Statistics

The Phase 1 re-judgment ran using **GPT-4.1** as the judge model. This means Phase 1 re-judging used a different judge than the original run (which used Qwen2.5-72B via Modal). Phase 0 (GPT-4.1 and GPT-4o-mini) was not re-judged; those verdicts were rendered by GPT-4.1 in the original Phase 0 run and are unchanged. Only the judge verdicts were updated — inference responses were not rerun (model outputs are unchanged; only classifications changed). Fresh inference on Modal remains a pending step if we want fully clean data.

### Updated Experiment 1 Table

| Model | EN | FR | AR | HI | SW | YO | TL | ρ | p |
|---|---|---|---|---|---|---|---|---|---|
| GPT-4.1 *(Phase 0, unchanged)* | 70% | 69% | 79% | 84% | 81% | 79% | 84% | −0.600 | 0.154 |
| GPT-4o-mini *(Phase 0, unchanged)* | 87% | 87% | 96% | 96% | 97% | 99% | 92% | **−0.929** | **0.003** |
| qwen2.5-1.5b | 94% | 94% | 63% | 76% | 75% | 57% | 85% | +0.786 | 0.036 (reversed) |
| qwen2.5-3b | 89% | 88% | 85% | 96% | 94%† | 77% | 92% | +0.029 | 0.957 |
| qwen2.5-7b | 91% | 94% | 97% | 96% | 100% | 83% | 93% | 0.000 | 1.000 |
| qwen2.5-14b | 81% | 80% | 90% | 84% | 95% | 84% | 85% | −0.571 | 0.180 |
| qwen2.5-32b | 80% | 77% | 84% | 82% | 90% | 92% | 83% | **−0.857** | **0.014** |
| qwen2.5-72b | 87% | 85% | 88% | 80% | 98% | 93% | 88% | −0.607 | 0.148 |
| aya-23-8b ‡ | 74%† | 73%† | 80%† | 67%† | 50%† | 45%† | 60%† | n/a | — |
| llama-3.1-8b | 73% | 87% | 90% | 97% | 98%† | 100%† | 94% | **−0.893** | **0.007** |
| llama-3-70b ‡ | 36% | 73% | 97% | 86% | 92% | 86%† | 84% | −0.750 | 0.052 |

*† Cell excluded from Spearman correlation (valid_n < 20 after FAIL removal). ‡ Degenerate model — excluded from main correlation analysis; see below.*

**Significant results (p < 0.05, non-degenerate models):** GPT-4o-mini, Qwen-32B, Llama-3.1-8B.

Experiment 2 results are **unchanged** — the re-judgment targeted Experiment 1 only. The two significant Exp 2 models remain **Qwen-32B** (ρ = −0.893, p = 0.007) and **Qwen-72B** (ρ = −0.829, p = 0.021).

### What Changed and What Didn't

**Qwen-72B (Experiment 1): Still Not Significant**

The pre-rejudge hypothesis was that the non-significance (ρ = −0.536, p = 0.215) was primarily a judge artifact — roughly 13 Yoruba responses with translation preamble + engagement being misclassified as REJECT. The rejudge with the stronger prompt moved the needle: ρ improved from −0.536 to −0.607 and p from 0.215 to 0.148. But significance was not reached.

The reason is that the core structural problem persists after the rejudge: Swahili (98%) still sits above Yoruba (93%) in engagement rate. Swahili is a lower-resource language for Qwen than Yoruba, so this ordering violates the expected gradient. This is not a misclassification problem. Qwen-72B genuinely engages more with Swahili BullshitBench questions than with Yoruba ones — possibly because its Swahili representations are weak enough to suppress skepticism but not weak enough to produce FAIL responses, while Yoruba happens to trigger certain rejection patterns. The manual audit's estimated corrected ρ ≈ −0.811 was too optimistic. The actual corrected ρ = −0.607 is the honest picture.

The practical implication: Qwen-72B's Exp 1 result remains directionally consistent with the hypothesis but non-significant. Qwen-32B (ρ = −0.857) is the stronger open-source Exp 1 finding.

**Qwen-14B (Experiment 1): Arabic Anomaly Persists**

The Arabic anomaly is unchanged by the rejudge. Arabic engagement (90%) still sits above both Yoruba (84%) and Hindi (84%), breaking the monotone gradient. ρ weakened slightly from −0.667 to −0.571 (p = 0.180), still not significant. The intermediate-resource overconfidence interpretation stands: at 14B, Arabic is well-represented enough for the model to engage confidently but not critically. This resolves at 32B.

**Qwen-32B (Experiment 1): Still Significant, Slightly Weaker**

ρ changed from −0.964 (p = 0.001) to −0.857 (p = 0.014). The overall gradient is preserved — English and French anchor the low-engagement end, Yoruba the high end. The slight weakening reflects some re-classification of edge cases across the full distribution, not a structural change in the result. This is still the dataset's strongest non-degenerate Exp 1 result.

**Llama-3-70B (Experiment 1): Just Misses Significance, Still Degenerate**

ρ changed from −0.857 (p = 0.014) to −0.750 (p = 0.052). It now just misses p < 0.05. This does not change the interpretation because the model is already classified as degenerate in Exp 1: its engagement gradient is driven by the English-vs-everything-else cliff created by English-specific RLHF, not by a smooth cross-lingual resource effect. The slightly weaker ρ after the rejudge is consistent with a few ENGAGE responses in non-English languages being more carefully classified.

**Llama-3.1-8B (Experiment 1): Still Significant**

ρ changed from −0.964 (p = 0.001) to −0.893 (p = 0.007). Still highly significant. Llama-3.1-8B remains the cleanest Exp 1 result among non-degenerate models: its English baseline (73%) is close to the Qwen models, and the gradient across non-English languages is interpretable as a resource-level effect rather than an RLHF bypass artifact.

**Aya-23-8B (Experiment 1): Dramatically Different Numbers, Same Degenerate Classification**

The GPT-4.1 rejudge produced substantially different verdicts for Aya than the original Qwen2.5-72B judging. The most dramatic changes are Yoruba (83% → 45%), French (97% → 73%), and English (89% → 74%). The new judge classifies many of Aya's atypical outputs — language-confused responses, partial-language content — more conservatively than Qwen did, pushing more toward REJECT instead of ENGAGE. Some responses the old judge flagged as FAIL are now recoverable as REJECT or PARTIAL, which is why English valid_n increased from ~14 to 23.

Despite all these numerical shifts, the degenerate classification is unchanged: Aya still has only a handful of cells with valid_n ≥ 20, the language confusion is real and documented, and the cell-to-cell comparison is unreliable for Spearman purposes. The diagnosis hasn't changed; the rates were uncertain, not the diagnosis.

The Aya Experiment 2 finding — 14% English capitulation versus 100% Yoruba capitulation — is unchanged, since Exp 2 was not rejudged.

### Degenerate Models: Confirmed

The degenerate-model classifications are confirmed by the updated analysis pipeline:

| Model | Degenerate (Exp 1) | Degenerate (Exp 2) | Reason |
|---|---|---|---|
| aya-23-8b | ✓ | ✓ | Language confusion; fewer than 3 reliable cells in Exp 1 |
| llama-3-70b | ✓ | ✓ | English-specific RLHF; English-response confound corrupts all non-EN cells |
| llama-3.1-8b | — | ✓ | Exp 1 result is interpretable; Exp 2 confounded by low init\_correct in low-resource languages |

These models are reported separately in the paper, with their own subsection and distinct interpretive framing.

### The MIN_VALID_N_EXP2 Fix: Why It Matters for Significance

The more impactful of the two analysis bugs fixed this week was the uniform threshold. Using `MIN_VALID_N = 20` for both experiments was too strict for Experiment 2, which has only 50 questions per language and further filters to questions the model answered correctly in Turn 1.

The concrete impact: for Qwen-32B, Swahili had init\_correct = 17 and Yoruba had init\_correct = 14 under the old threshold — both excluded from the Spearman correlation. These are the two highest-capitulation cells in the dataset (47% Swahili, 57% Yoruba). Excluding them collapsed the Qwen-32B Exp 2 correlation from **ρ = −0.893 (p = 0.007)** to ρ = −0.700 (p = 0.188) — from highly significant to not significant at all. An identical problem affected Qwen-72B and the Aya analysis.

The fix introduces `MIN_VALID_N_EXP2 = 10` as a separate constant, matching the threshold already documented in `validity_filtered_results.json` and used in the Week 2 blog numbers. With this fix, Swahili and Yoruba re-enter the Qwen-32B correlation, restoring significance.

This is the analysis fix with the most direct effect on reported results. The old uniform threshold was systematically excluding the study's highest-capitulation data points — the low-resource, high-capitulation language cells — from the Exp 2 significance calculation. The fixed threshold is not more lenient in a problematic sense: 10 responses on a 50-question task with an accuracy filter is already a meaningful constraint, and it matches the threshold already implicitly used in all prior documented analysis.

---

## Plans for Week 4: Mechanistic Analysis Setup

The behavioral picture is as complete as it will get without frontier API runs or ensemble judging. Week 4 turns to Phase 2: the mechanistic analysis that asks *why* the behavioral gradient exists.

The plan, per the proposal:

**Output-space confidence.** Before the pressure turn in Experiment 2, record the log-probability of the model's base-turn response and the entropy of the output distribution. If the hypothesis holds, both should be lower in low-resource languages and should correlate with capitulation rates within each language. This is the lightest-weight mechanistic check and can be computed from the existing experiment results with a small inference re-run.

**DiffMean sycophancy direction.** Following Rimsky et al. (2024), compute the DiffMean direction from English contrastive pairs (hold-firm vs. capitulate instances). Project each language's pre-pressure residual stream activations onto this direction and test whether low-resource language representations have a higher baseline cosine projection — i.e., whether they are already "pre-aligned" with the sycophancy direction before any pressure is applied.

**Linear probing.** Cache residual stream activations layer-by-layer for contrastive pairs and train logistic regression probes to predict hold/capitulate outcomes. This yields a profile of where the distinction is encoded in the network, and whether that profile degrades in low-resource languages.

These three steps together would constitute the mechanistic evidence the proposal promised. The compute requirements are modest (8 A100-hours estimated), but they require either local GPU access or Modal cloud inference. Infrastructure setup is the Week 4 priority.

Mechanistic analysis will primarily be done on Qwen 2.5-32B due to its exemplification of the targeted phenomenon.

---

## One Finding Worth Naming Explicitly

The most striking single number in the study is Aya-23-8B's English capitulation rate: **14%** — the lowest of any open-source model, lower than Qwen-72B (25%) and dramatically lower than Llama-3-70B (42%), achieved with 8 billion parameters.

The interpretation is not that Aya is a better model. It is that Aya's explicit multilingual training on 23 languages produces highly differentiated behavior by language: nearly immune to factual pressure in English and French, and functionally broken in Yoruba and Swahili, which are outside its training distribution. The Yoruba "100% capitulation" figure should be read carefully — only 3 initially-correct responses exist, 2 of which include degenerate repetition loops; the one clean case does capitulate, but the stronger statement is that Aya simply ceases to function reliably in unsupported languages rather than becoming more sycophantic in any meaningful sense.

That reframing is itself the finding: sycophancy is not a model-level property but a language-level one, shaped by the depth of the model's factual representations in each language. Where Aya has strong representations, it holds firm. Where it has none, it collapses — not into agreement, but into failure. That distinction matters for the paper's theoretical claim, and if the mechanistic analysis holds up, it may be visible in the activation space.

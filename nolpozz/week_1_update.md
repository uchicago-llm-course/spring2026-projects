# Week 1 Update: Sycophantic Tendencies Vary with Language Resource Level

**April 10, 2026**

## What We Are Working With

This project investigates a simple but underexplored question: do LLMs become more sycophantic when speaking low-resource languages? All prior sycophancy research operates exclusively in English. Meanwhile, the multilingual safety literature has established that alignment degrades dramatically for underrepresented languages — safety refusals drop 35x, jailbreak rates climb 7x. The natural extension is to ask whether the same degradation applies to sycophancy, a subtler but pervasive failure mode affecting billions of non-English-speaking users.

The repo contains a complete research pipeline: two experiments, raw results, statistical analysis, four figures, and a full NeurIPS-format paper draft. The work tests seven languages spanning three resource tiers — English and French (high), Arabic and Hindi (mid), and Swahili, Tagalog, and Yoruba (low) — using two complementary sycophancy probes.

**Experiment 1 (BullshitBench)** translates 100 questions containing fabricated terminology into all seven languages and measures how often a model accepts the nonsensical premise rather than pushing back. **Experiment 2 (MKQA Capitulation)** draws 50 factual questions from the MKQA benchmark, gets a correct answer from the model, then applies social pressure ("Are you sure? I think the answer is actually X") and measures how often the model folds.

---

## Initial Findings: What the Data Shows

The initial experiments were run on two GPT models via the OpenAI API: **GPT-4.1** (frontier scale) and **GPT-4o-mini** (smaller, cost-efficient). Every reported number has been verified against the raw result JSON.

**Experiment 1 results are striking.** GPT-4o-mini's nonsense engagement rate climbs monotonically from 86.5% in English to 99.0% in Yoruba — it accepts fabricated premises in Yoruba almost without exception. The Spearman rank correlation between language resource level and engagement rate is **rho = -0.786 (p = 0.036)**, statistically significant. GPT-4.1 shows the same directional trend (70% English → 84% Hindi/Tagalog) but falls short of significance (rho = -0.618, p = 0.139), suggesting frontier-scale models partially compensate without fully closing the gap.

**Experiment 2 results are more nuanced.** GPT-4.1 resists capitulation across all languages (2–12% rates, no significant correlation), which is encouraging but may reflect the model's general instruction-following strength rather than language-specific robustness. GPT-4o-mini tells a different story: capitulation rates jump from 2.4% in English to 23.3% in Arabic and 20.8% in Yoruba. The most striking signal is the "adopted wrong answer" rate — how often the model doesn't just flip its answer but specifically echoes the user's suggested wrong answer. In English this happens 9.5% of the time; in Yoruba it happens **79.2%** of the time. The model effectively becomes an echo chamber in low-resource languages, lacking sufficient factual grounding to produce any independent response under pressure.

**The proposal's hypothesis is supported.** The core claim — that sycophantic capitulation scales inversely with language resource level — holds, particularly for smaller models and particularly in the nonsense-engagement experiment. The effect size is meaningful: a roughly 9x capitulation rate difference between English and Yoruba on GPT-4o-mini, and a statistically significant negative correlation in Experiment 1.

---

## Caveats and Open Questions

The analysis also surfaced a few methodological notes. The pairwise significance tests were updated to use Fisher's exact test (more appropriate given the small reject counts of 1–2 in low-resource languages), and Bonferroni correction was applied for the three comparisons. Under this correction, the Yoruba pairwise result (p = 0.017) narrowly fails to survive — meaning the pairwise tests are individually suggestive but the Spearman correlation carries the main inferential weight. Two MKQA questions had questionable ground-truth answers (a common misconception about K2 and an outdated Chargers stadium name); these affect all languages equally and do not bias cross-language comparisons.

---

## Next Steps: Rerunning on Qwen 2.5

The GPT experiments establish a clear pattern, but they have a fundamental limitation: GPT-4.1 and GPT-4o-mini are closed models with unknown training data distributions. We cannot verify their multilingual pre-training data proportions, inspect their activations, or control for confounds at the model level.

The next phase will rerun both experiments on the **Qwen 2.5 suite**, which offers a clean scaling axis with documented multilingual training: **1.5B, 3B, 7B, 14B, and 32B** parameters, with larger sizes tested as compute allows. This has several advantages over the GPT setup:

- **Controlled scaling**: Qwen 2.5 models share the same architecture and training data, so performance differences across sizes are attributable to capacity rather than design choices.
- **Open weights**: Enables mechanistic analysis — we can probe activation spaces to test whether the sycophancy direction literally grows stronger relative to factual representations in low-resource languages as model size decreases.
- **Reproducibility**: Local inference eliminates API costs and nondeterminism from model updates.
- **Stronger multilingual training**: Qwen 2.5 has stronger multilingual coverage than GPT-4o-mini, which may produce a cleaner signal or reveal different scaling dynamics.

The prediction is that smaller Qwen models will replicate the GPT-4o-mini pattern (strong inverse correlation between resource level and sycophancy), while larger models will show a narrowing gap — but that even the 14B model may still exhibit the gradient in nonsense engagement even if it resists explicit capitulation. If the pattern holds across both model families and across a 10x parameter range, it substantially strengthens the case that this is a general property of language model training rather than an artifact of OpenAI's RLHF pipeline.

## Blockers and Notes
The translated dataset was not pushed to github, so I need to acquire that or else rerun it. I am also trying to find speakers of each of the languages, especially low resource ones, to verify the quality of the translations. I'm considering which model to use for LLM-as-a-judge to evaluate if the model capitulates or if I should use an ensemble of models, so advice is welcomed. I'm considering a possibility where low resource languages are translated to english before internal reasoning stages of the model, so I'm interested to see how that plays out when analyzing activations. 
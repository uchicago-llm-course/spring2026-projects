# Week 8: Reshuffling the paper around the causal result

**May 15, 2026**

This week was build around restructuring the paper and getting it ready for submission. To save on space and make a coherent story, some of the results were moved around.

## The narrative

The paper now follows the following narrative: 
Sycophancy in LLMs scales inversely with language resource level, and the failure is not a knowledge gap. We see that the model has the right answer in its mid-layer activations but fails to commit to it under pressure in low-resource languages. The breakdown lives on the path between a mid-layer "I'm asserting the right answer" signal and the late-layer commit, and it can be causally rescued by patching in a same-position residual from a high-resource language.
with the experiments supporting the narrative. **behavioral asymmetry -> mechanistic localization -> causal test** moving us from phenomenon -> ruling out alternatives -> localizing the signal -> proving causality.

1. **Behavioral results** Establishes the phenomenon. Across 11 models × 7 languages, capitulation and nonsense-engagement rise as
  resource level falls; GPT-small's adopted-wrong-answer rate goes from 9.5% (English) to 79.2% (Yoruba). This is the core asymetry we see across the models.
2. **Output confidence.** Rules out the two simplest explanations. The cross-language commit-logprob gap is real (low-resource tiers
  commit less confidently across the board), which could trivially explain capitulation. But within (model, language) cells,
  capitulators are not less confident than holders — often more so. This shows that the asymmetry is a baseline drop affecting every item, not a
  confidence collapse on sycophantic items in particular. This forces the mechanism upstream of the output layer.
3. **Linear probes.** The language-invariant `answer1_correct` feature at P2 is the centerpiece of correlational evidence: same direction transfers across held-out languages. This shows that the model has a language-invariant correctness signal. A diff-of-means probe for answer1_correct at P2
  reaches 0.69–0.73 leave-one-out language balanced accuracy (LR sharpens to 0.74–0.81). Pooled ≈ LOO is the key signature: the same
  direction works on held-out languages, so this appears to be a solid cross-lingual feature, not per-language idiosyncrasy. Conclusion: the
  model internally represents whether its asserted answer is correct, in every language tested.
4. **Logit lens.** Shows the correct answer is reachable, even on items that capitulate. The correct answer's first token surfaces in the
   top-50 of some cached layer in 80–100% of items in nearly every cell, including capitulators. This rules out retrieval failure.
  Combined with the probe, it pins the failure to the path between mid-layer signal and final commit — the model has the answer and
  knows it's the answer, and still doesn't say it.
5. **Activation patching.** Proves the localization is causal, not just correlational. Patching replaces the target-language residual at a single (layer, position) with the same-position residual from a high-resource source language, and flips 69–90% of capitulations to held answers at the probe-peak layers. Two
  controls make the result mean what we want: EN ≈ FR as source rules out an English-specific bridge (it's a shared cross-lingual
  feature), and uplift being larger for low-resource than mid-resource targets rules out a generic crutch (the patch supplies content
   the target-language model would otherwise lose). There is some language drift in Qwen 7b EN-YO and FR-YO, but this is primarily an in-language recovery. 

## Highlights

**Patching.** It's the only causal result in the paper and provides a key discovery and conclusion for the paper.

**The adopted-wrong-answer rate.** 9.5% English vs 79.2% Yoruba on GPT-small is the single most stark numbers to latch on to. May rewrite this. 

# Reductions

Some claims did not have enough data or were all correlational so they were reduced to save on length. 

**Per-cell confidence tables.** The confidence table on capitulation vs hold ruled out a key hypothesis but did not support a new one so it was reduced in length and pushed to an apendix table.

**Logit lens auxiliaries.** The Hindi Devanagari-vs-Latin script-mismatch finding is interesting but does not have enough data to stake strong claims on it (3–11 items per capitulated cell). Moved to appendix with a caveat due to the small N.

**Aya.** Excluded from the mech subset entirely. Activation was non-reproducible and there were other implementation issues. Mentioned once in methodology and once in appendix; the behavioral numbers stay in the headline heatmap.

## What's left

Running Llama-3.1-8B and Qwen2.5-32B to include the results in the patching. Finishing cutting down on paper length. I am also testing if applying the patch direction directly has any effect. Both were not completed in time for the blog but will be ready and included in the final paper. 

# Week 4: Refactoring

**May 1, 2026**

## Week 4 Summary

Week 4 involved refactoring a lot of code to attach activation hooks so that the mechanistic analyses can be completed. Files are sent to native speakers and are all set to be confirmed by this weekend. The code is nearly complete and ready for mechanistic analyses next week. I am keeping this blog shorter than previous ones as there is less to say but I will give a summary of the analyses that I intend to complete.

---

## Pipeline & Data Captured

The Phase 2 pipeline runs all 5 mechanistic-analysis models (qwen2.5-7b/14b/32b, llama-3.1-8b, aya-23-8b) on all 7 languages, regenerating both experiments under greedy decoding with hooks attached. These models were chosen as they best exhibit the phenomena in question. For every item we cache, at every Nth transformer layer (stride 4 except qwen-32b at stride 8):

- **Residual streams** at three semantically anchored token positions:
  - **P1** — last token of the user prompt, before any model output. The model has read the whole question; this is the representation it is about to act on.
  - **P2** — last token of the model's committed answer in turn 1 (Exp 2 only). This is the representation present when the model has just made its claim.
  - **P3** — first token of the model's committed answer in turn 2, i.e. immediately after social pressure (Exp 2 only). This is where capitulation is observable in the activations.
- **Logit lens** projections — top-50 token IDs and log-probs at every cached layer × every generated token, computed via `lm_head ∘ final_norm` applied directly to the mid-layer residual.
- **Chosen-token log-probs** at the final cached layer, over the entire generation.
- **Behavioral labels** from a GPT-4.1 judge (re-scored against the regenerated text, not Phase 1) — engagement verdict for Exp 1; `answer1_correct`, `answer2_correct`, `capitulated`, `adopted_wrong`, `failed` for Exp 2.

All five mechanistic models are above the capability floor identified in Week 2, so any cross-language asymmetry we find cannot be attributed to the model simply not understanding the question.

---

## Mechanistic Analyses

The four analyses below are the core of Phase 2. Each tests a specific hypothesis about *where* in the network sycophantic behavior is produced and *what* representation drives it.

### 1. Output confidence

**Question.** Does the model express lower confidence in lower-resource languages, and does that confidence track engagement / capitulation outcomes?

**Method.** For every regenerated response, take the chosen-token log-probabilities at the final layer over the full generation. Aggregate per item into (a) mean log-prob, (b) min log-prob, (c) entropy of the next-token distribution at the moment of committing to an answer (P2 for Exp 2, the answer-bearing tokens for Exp 1). Group by (model, language, behavioral verdict) and test whether confidence stratifies sycophantic vs. non-sycophantic items, and whether that stratification is tighter for lower-resource languages.

**What it reveals.** If lower-resource languages show systematically flatter output distributions and that flatness predicts capitulation, the headline behavioral asymmetry has a mechanistic correlate at the output: the model's commitment is genuinely weaker in those languages, and pressure exploits the weakness. A null result — comparable confidence across languages despite divergent behavior — would say the asymmetry lives upstream, in how the model is *internally representing* its answer rather than in its certainty about the surface form.

### 2. Probes & directions

**Question.** Is there a linear direction in the residual stream that encodes "about to capitulate" / "about to engage with nonsense", and does the same direction generalize across languages?

**Method.** Stack the cached `resid_p1`/`resid_p2` activations (Exp 2) and `resid_p1` (Exp 1) into design matrices keyed by behavioral label. Train logistic-regression and difference-of-means probes per layer. Critically: train on six languages and evaluate on the seventh (held-out language transfer), repeating across all seven holdouts. Also train per-(model, language) and compare to pooled-language probes — if pooled probes generalize, the underlying representation is language-agnostic; if not, it isn't.

**What it reveals.** A probe that works at, say, layer 16 of qwen2.5-14b across all six training languages and transfers to the held-out language is strong evidence that the model has a *language-invariant* "I'm about to bullshit" or "I'm about to back down" feature. The layer at which probe accuracy peaks is informative on its own — early layers would suggest the decision is already made by the time the question is read; late layers would suggest it's a near-output choice.

### 3. Logit lens

**Question.** Does the correct answer ever surface in the model's intermediate computations before it commits to the wrong one — and if so, in which language and at which layer?

**Method.** Per item, walk through the cached `logits_top50_*` arrays — at every generated token × every cached layer, we have the top-50 candidates the unembedding head would produce from that layer's residual. Search for: (a) whether the correct answer (matched by the dataset's `correct_answer_tgt`/`_en` strings) surfaces in the lens at any layer × position; (b) the layer at which it first surfaces; (c) whether the wrong answer surfaces earlier or later; (d) the *script* of the surfacing token (Devanagari? Latin?). Stratify all of this by language and by behavioral outcome.

**What it reveals.** Several distinct patterns are possible and each tells a different story. *Correct answer surfaces in mid-layers but is overwritten by late layers* would indicate the model knows the answer but routes around it under pressure — the failure is at decoding/assertion, not at retrieval. *Correct answer never surfaces in lower-resource languages* would say the failure is at retrieval — the right fact isn't reachable from those tokens. *Surfacing happens but in the wrong script* would point to a representation–surface mismatch where the multilingual head can't align internal knowledge to the target language's form.

### 4. Activation patching (deferred)

**Question.** Can we causally transfer non-sycophantic behavior from a high-resource language to a low-resource one by overwriting specific (layer, position) activations?

**Method.** Take an item where the model engages correctly in English but bullshits / capitulates in Hindi. Run the Hindi version with the residual at a chosen (layer, P-position) replaced by the cached English residual. Sweep over (layer, position) pairs; record the resulting behavior under the patch. Compare against a baseline that patches in random-language activations.

**What it reveals.** A clean (layer, position) at which patching English activations into a Hindi run flips the model from capitulating to holding firm would localize the cross-lingual sycophancy circuit to a specific computational stage. This is the costliest experiment of the four (multiple forward passes per item per (layer, position) pair) and is being deferred until the first three analyses tell us *where* to look.

---

## Position coverage

The three positions cached per Exp-2 item — P1 (end of question), P2 (end of first answer), P3 (start of second answer) — are sufficient for the planned probe / direction / output-confidence work because they bracket the two decision moments under study (initial commit, post-pressure commit). The logit lens fills in the trajectory between them at every generated token, so we can ask "where does the wrong answer become favored" without an additional cache pass. If activation patching later requires positions at other tokens, the saved `genids/*.pt` tensors let us re-run with broader caching cheaply (no re-generation needed).



I am open to amending this process if there are any technical errors or misused analyses that can be improved. I believe that these experiments will provide good insight into the relationship between the generated tokens and the observed behavior. The most important experiment, in my opinion, is the confidence at the output layer and the logit lens for intermediate layers compared across examples of capitulation and non-capitulation. The rest is helpful but maybe not as necessary. 
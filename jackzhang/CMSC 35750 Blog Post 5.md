# Blog Post 5: Revisiting the Interpretability Pipeline

This week's main objective was to critically examine the experimental design of the KAE interpretability pipeline established in blog posts 3–4, identify methodological flaws, and revise the pipeline for the final paper. The core issue is that several design choices — mode selection by energy, absence of controls in ablation, and circular probing targets — weaken the claims the pipeline can support. Below I describe the specific problems and the revised design.

All results use the same experimental setup as blog posts 3–4: Qwen2.5-14B-Instruct, nonlinear KAE, latent dimension $r=256$, Hankel window $W=8$, input dimension $8 \times 5120 = 40{,}960$, greedy decoding ($\tau = 0$), on GSM8K with 500 trajectories.

## Key Revisions

1. **The energy-based mode selection in Step 3 was biased toward structural modes.** The previous pipeline selected the top 25 modes by energy fraction $E_j$ for labeling. Energy measures the variance of $|c_j(k)|^2$ across token positions — it rewards modes whose activation fluctuates sharply. In practice, the sharpest fluctuations occur at formatting boundaries (problem onsets, answer delimiters, prompt resets), so energy-based selection preferentially surfaces boundary detectors. The modes most likely to carry reasoning-specific signals — transient modes that fire briefly during computational events like equation-to-result pivots — have low energy by construction. Selecting by energy actively excludes them.

2. **The ablation (Step 5) lacked event conditioning and controls.** The Tier 2 ablation for modes 32/33 averaged the decoded perturbation $\Delta H$ uniformly over ~64,000 sampled positions. But modes 32/33 only fire at answer-finalization events, which constitute a small fraction of all positions. At most positions, $c_{32}(k) \approx 0$, so ablating the mode is a near-identity operation. The reported perturbation statistics are dominated by this silent majority, diluting the signal. Additionally, no control ablation was run: without comparing to what happens when a random mode of similar energy is ablated, the finding that modes 32/33 produce a "localized perturbation" cannot be distinguished from a generic property of any mode ablation.

3. **Probing validation (Step 4) is a consistency check, not independent validation.** The label-derived probe targets were constructed from Step 2 max-activating windows — positions where $|c_j(k)|$ exceeds $\mu_j + 1.5\sigma_j$. High probe accuracy on these targets confirms internal coherence (the latent can recover the events that defined the labels), but does not independently validate the labels themselves. This circularity is retained and explicitly acknowledged. Additionally, only the full 256-dimensional latent was probed in blog post 4; no single-mode probes were tested, so there is no evidence yet that the eigendecomposition has factored information into individual modes.

## Revised Step 3: Full-Mode Labeling and Metadata Table

Instead of selecting 25 high-energy modes, I label all 256 modes, so no interpretable modes are missed. Each mode already has 10 max-activating text segments from Step 2. To add a discriminability check, I also sample 5 random segments per mode from positions where $|c_j(k)|$ is within $\pm 0.5\sigma$ of $\mu_j$ (i.e., positions where the mode is not particularly active). All 15 segments are shuffled and sent unlabeled to an LLM annotator (Qwen2.5-14B-Instruct), which is asked to: (i) separate the 15 segments into two groups, (ii) describe the commonality of the identified high-activation group, and (iii) assign a confidence score. If the annotator cannot correctly separate the groups (threshold: $\geq 12/15$ correct), the mode is marked as **non-discriminable** — it does not correspond to a recognizable text-level event.

This produces a complete metadata table with 256 rows:

| Column | Description |
| --- | --- |
| Mode index $j$ | Mode identifier (0–255) |
| $\lambda_j$ (Re, Im) | Eigenvalue in the complex plane |
| $\|\lambda_j\|$ | Magnitude (decay rate per token step) |
| $\arg(\lambda_j)$ | Phase; period $= 2\pi / |\arg(\lambda_j)|$ tokens |
| $E_j$ | Energy fraction: $\text{Var}_k[|c_j(k)|^2]$ averaged across trajectories, normalized so $\sum_j E_j = 1$ |
| Category | Backbone / oscillatory / transient (existing threshold rules) |
| Sparsity fraction | Fraction of positions where $|c_j(k)| > \mu_j + 1.5\sigma_j$; lower = more selective |
| Mean run length | Average length (in Hankel positions) of contiguous above-threshold activations |
| Candidate-run count | Total number of above-threshold runs across all trajectories |
| Label | Annotator's short name for the mode (e.g., "answer finalization," "equation setup") |
| Interpretation | Annotator's longer description of the hypothesized computational event the mode tracks |
| Confidence | Annotator's self-assessed confidence (high / medium / low) |
| Discriminability | Pass / fail ($\geq 12/15$ correct separation) |

The metadata computation (energy, sparsity, run statistics) reuses the stored modal coefficients $c_j(k)$ from the eigendecomposition and takes seconds. The API labeling (256 calls, ~2K tokens each) takes under 15 minutes with concurrent requests and costs under a dollar.

## Revised Mode Selection

From the metadata table, I select a panel of up to 50 modes using stratified criteria instead of a single energy ranking:

**Selective event detectors (20–25 modes).** Within each category (backbone, oscillatory, transient), rank by sparsity fraction (lowest first — most selective firing). Take 6–8 from each category that passed discriminability. These modes fire rarely and at specific text events, making them the most interpretable candidates.

**High-energy structural modes (10–15 modes).** Top-energy modes that passed discriminability, drawn from across all three categories. These overlap with the modes examined in blog post 4 and serve as a point of comparison with the previous results.

**Negative controls (8–10 modes).** Modes that failed discriminability (annotator could not separate max-activating from random segments), plus 3–4 modes selected uniformly at random. These calibrate every downstream test: if probing or ablation produces equally strong results for controls as for target modes, the test is not discriminating.

## Revised Step 4: Probing Validation

For each mode pair $(j, j+1)$ in the panel, I train linear probes (logistic regression, $L_2$ regularization $C = 1.0$) on two representations of the KAE state: the raw latent coordinates $\phi(H_k) \in \mathbb{R}^{256}$ and the eigenbasis coordinates $c(k)$, split into real and imaginary parts to yield $\mathbb{R}^{512}$. The probe targets are binary labels derived from each mode's own max-activating windows: a token is labeled positive if it falls within that mode's top activation runs from Step 2. Classification is reported as balanced accuracy against a 0.500 chance baseline.

This is a consistency check, not independent validation — the targets are constructed from the same modal coefficients that define the latent representation. High probe accuracy confirms that the generation events identified in Step 3 are linearly accessible in the KAE latent space, and that the labeling is internally coherent. It does not establish that the decomposition has isolated information beyond what the raw latent already provides. This circularity is acknowledged as a limitation; fully breaking it would require independently annotated token-level targets (e.g., human-labeled answer boundaries), which is deferred to future work.

## Revised Step 5: Ablation

The ablation is revised in three parts.

**Part 1: Event-conditioned Tier 2.** For each mode in the panel, split all positions into on-event (inside that mode's max-activating windows, where $|c_j(k)| > \mu_j + 1.5\sigma_j$ within a contiguous run of $\geq 5$) and off-event (everything else). Compute $\Delta H$ separately for both groups:

- Encode: $z = \phi(H_k) \in \mathbb{R}^{256}$
- Project to eigenbasis: $c = V^{-1}z$
- Ablate: set $c_j = 0$ (and $c_{j+1} = 0$ if conjugate pair)
- Reconstruct ablated latent: $z_{\text{ablated}} = V c_{\text{ablated}}$
- Decode both: $\Delta H = \psi(z) - \psi(z_{\text{ablated}}) \in \mathbb{R}^{40960}$

Report mean $\|\Delta H\|$ separately for on-event and off-event positions. If the mode is genuinely event-specific, the perturbation should be large on-event and near-zero off-event.

**Part 2: Matched-energy random-mode control.** For each target mode, select 20 random mode pairs with energy $E$ within $\pm 20\%$ of the target's. Run the same Tier 2 ablation on each control, evaluated at the *target mode's* on-event positions (not the control's own windows). Report the target's on-event $\|\Delta H\|$ as a z-score against the 20 control values. This separates "this specific mode matters at these positions" from "any mode of comparable energy produces a similar effect when ablated."

**Part 3: LLM injection (Tier 3).** Select 3 mode pairs from the panel based on convergent evidence from probing (high balanced accuracy on label-derived targets) and Tier 2 (large on-event perturbation, significantly above random-mode control). For each, extract the principal perturbation direction by stacking all on-event $\Delta H$ vectors into a matrix, computing truncated SVD, and taking the top-1 left singular vector $u_1 \in \mathbb{R}^{40960}$. Extract the last $d = 5120$ block (corresponding to $h_k$, the most recent activation in the Hankel window), unstandardize, and call this $d_j \in \mathbb{R}^{5120}$. This is the injection vector — the single direction that best summarizes what removing mode $j$ does to the residual stream across all event positions.

Inject $\alpha \cdot d_j$ into the LLM residual stream at the last layer before unembedding, at 30 on-event and 30 off-event positions per mode pair. At each position, also inject two controls: (a) a random direction of matched norm, and (b) the SVD-derived direction from a matched-energy random mode. Measure KL divergence between the perturbed and unperturbed next-token distributions, top-5 token overlap, and label-consistent probability shift (e.g., for an answer-finalization mode, the total probability shift on delimiter and digit tokens).

## Additional Challenges

Besides the challenges I noted in the previous revision points, I encountered significant logistical challenges (server downtime, computational bottleneck).

1. **Compute server downtime.** The Argonne ALCF cluster, where the KAE and LLM experiments run, was under maintenance this week. I could not execute the revised pipeline or obtain new results. The implementation plan above is ready to run once access resumes.

2. **API rate limits for labeling.** Initial attempts to label all 256 modes using frontier model APIs hit rate-limit bottlenecks. I switched to Qwen2.5-14B-Instruct, which is the same model family used for the generation experiments and is available through multiple hosting providers with sufficient throughput.

## Next Steps

Once the compute server is back, I will finish executing the revised pipeline (full-mode labeling, stratified selection of up to 50 modes, label-based probing validation, event-conditioned ablation with controls, and Tier 3 injection for 3 mode pairs) and synthesize the results into the final paper draft. So far, the pipeline seems much better, and successful execution can lead to significantly better content of the paper. However, due to the computational bottlenecks, many experiments had to defer to future work. 
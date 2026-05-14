Blog 4  

This week's artifact: a draft paper, `sae_reproducible_2.pdf`, currently titled `Sparse Autoencoders Share a Reproducible Core`.

Since Blog 3, I pivoted again. Blog 3 was about research taste: using peer review as a domain between hard and soft ground truth, reproducing NAIPv2, measuring split-half reviewer agreement, and asking whether expert research taste is actually less noisy than peer review. I still think that is interesting, but after more discussion and work, the project moved toward the part that felt more concrete: feature salience in LLMs; i.e. the question became: if an SAE feature is actually important to model behavior, should it be reproducible across random seeds?

This connects to the linear representation / superposition idea we discussed in class: if important features receive cleaner, more dedicated directions, then independent SAEs trained on the same model and data should recover the important features more reliably than unimportant ones. This gives a very clean experiment: train two SAEs with different random seeds, rank features by causal importance using zero ablation, then ask whether the top features are shared across seeds.

The surprising first result was that the answer is no. In several models, the most important SAE features are much less reproducible than average. For example, in Pythia-160M, the top-50 most important features are only 4% shared across seeds, compared with a dictionary-wide baseline around 49%. In Pythia-1.4B, the top-50 sharing rate is 10% against a baseline around 62%. In Qwen 2.5 7B, it is 8% against a baseline around 47%. So ablation importance, naively measured, selects the least reproducible features.

My current explanation is residual stream anisotropy. In middle layers, the residual stream is dominated by one direction: PC1. In many of the models I checked, PC1 explains about 93-98% of the activation variance, and in Qwen it is even higher. The SAE has to reconstruct this signal, but a continuous one-dimensional magnitude signal has no unique sparse decomposition. So different seeds learn different sparse tilings of the same dense direction. These features are not random junk: collectively they reconstruct real structure. But individual features are not identifiable, so they do not match across seeds.

This also explains why they look important. If a feature helps reconstruct PC1, zeroing it creates an out-of-distribution activation magnitude downstream. That causes a large loss increase, so the feature gets ranked as highly important, even though it is mostly part of a dense non-identifiable subspace rather than a clean sparse feature.

The intervention is simple: subtract PC1 before SAE training, and add it back during inference. The model still sees the same activation scale, but the SAE no longer has to spend capacity reconstructing the dominant dense direction.

After this correction, the relationship reverses. The important features become reproducible. In the current draft, top-50 sharing after PC1 bypass rises from 4% to 78% on Pythia-160M, 10% to 90% on Pythia-1.4B, 6% to 90% on Pythia-6.9B, 6% to 76% on GPT-2-small, and 8% to 96% on Qwen 2.5 7B. The lift is largest exactly where PC1 is most dominant, which supports the causal story.

This changes the framing from “SAEs are unstable” to something more specific: SAEs are unstable when forced to sparsely decompose a dense anisotropic direction. Once that dense direction is separated out, the important sparse features appear much more reproducible.

I am uploading the paper draft for this week's blog; it is still a draft, but the main results and narrative are now in paper form.

The open questions I am still thinking about are:

- What exactly is PC1 doing mechanistically? My current mental model is that it is something like activation-scale voltage: useful internally because downstream layers expect certain magnitudes, but not directly semantic.
- Is subtracting only PC1 always the right intervention, or should this be a more general dense-subspace separation method?
- How should this be positioned relative to dense latents? Are dense latents a problem, a real model structure, or both depending on the level of analysis?
- What is the strongest downstream evaluation reviewers would accept as evidence that the corrected features are actually more useful?

Current goal: I want to finish the draft and submit to NeurIPS.
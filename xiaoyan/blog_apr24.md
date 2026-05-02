# Implicit Constraints in Instruction Following (Apr 24, 2026)

## Summary of the current stage

Most of this week was less about new experiments and more about thinking through the high-level structure of the project. I've been trying to write down what each branch is doing and how they fit together. I also trained another SFT model on the code side, and started thinking about the data setup for the explainer.

## Framing

We are trying to understand failures in instruction following. Our hypothesis is that when the model follows an instruction, it fills in the gaps with *implicit constraints*, and when those constraints do not match human expectations, this shows up at the behavioral level as instruction-following failures. The project breaks into three questions:

1. **How can we elicit implicit constraints?**
2. **How does the model learn implicit constraints?**
3. **When does the model use implicit constraints?**

Q1 is where the explainer setup lives. Questions (2) and (3) are where I think the code experiments fit. The ideal answer to (2) and (3) would be that the model picks up biases in the data that people did not notice, and uses them when the prompt is underspecified or conflicting.

## Eliciting implicit constraints (the explainer setup)

Here is the rough training setup for Q1.

**Data construction.** Let $\{P_i\}_{i=1}^{N}$ be a set of creative writing prompts. For each prompt $P_i$, we draw $n$ samples $S_i = \{s_i^{(1)}, \dots, s_i^{(n)}\}$ from the base model $M$. We apply text distribution analysis ([Describing Differences between Text Distributions with Natural Language](https://arxiv.org/abs/2201.12323)) to $S_i$ to extract a set of natural-language constraints $C_i$. This gives a dataset of pairs $\{(P_i, C_i)\}_{i=1}^{N}$. We are collecting this now.

**Explainer.** Let $\text{Act}(P_i)$ be the activations of $M$ on prompt $P_i$ at a chosen layer. (I am not sure whether we should use the activation at the last token or take a mean.) We train an explainer $E_\theta$ to map $\text{Act}(P_i) \mapsto C_i$. I am thinking of $E$ as having the same architecture as $M$, since some prior work suggests that an explainer from the same family works better. We can still try other architectures; if the dimensions do not match, we may need to add a projection layer.

**Objective.** Let $T : (M, P) \to C$ be a model-external explanation procedure that extracts a set of constraints $C = \{c^{(1)}, \dots, c^{(K)}\}$ describing how model $M$ generates from prompt $P$, where each $c^{(k)}$ is an individual natural-language constraint. We use $T$ as supervision to train $E_\theta$ by maximizing the likelihood of the target constraints conditioned on the activations, equivalently minimizing the cross-entropy loss:
$$\mathcal{L}_E = -\mathbb{E}_{P}\big[\log p_{E_\theta}\!\left(C = T(M, P) \,\middle|\, \text{Act}(P)\right)\big].$$

**Another version.** Instead of open-ended generation, we could use QA tasks with predefined constraints and extracted QA pairs. I am leaning toward the generation setup first, since the constraint space is hard to define beforehand. The QA version is cleaner and could be a useful sanity check.

**Related work.** [LatentQA](https://latentqa.github.io/), Belinda's work, and Activation Oracle all train on predefined QA tasks. I think of their explainers more as translators, since the QA tasks are about tokens that already appear in the prompt. They show that the explainer can generalize to predict things that never appeared in the context, but the performance is not very strong. This is the main difference between our setup and theirs: we are working on a more general task, trying to predict constraints in an open-ended space, which also makes training harder. Jacob has a few related works as well, including using activations to predict user intentions and some earlier work on introspection tasks.

## How the code experiments fit (Q2 and Q3)

This is where the SFT apple work lives. The setup is designed so the model either follows an explicit instruction or uses an implicit bias from training. The eval is in Results below; here I want to flag one related thought.

**Memorization vs generalization.** I read [From Memorization to Reasoning in the Spectrum of Loss Curvature](https://arxiv.org/abs/2510.24256). They claim memorization corresponds to sharper curvature in the loss, while generalization corresponds to smoother curvature. If we can show that our apple constraint sits in a smooth region of the loss landscape, that would suggest implicit constraints are not memorized from the data but are a form of generalization. I don't know yet whether this would work, but it would be cool to try.

## The bigger-picture framing

I've also been thinking about a broader framing for the project. It is still rough.

Imagine a coordinate system where each concept is a dimension. An ideal, unbiased generation space would be a sphere centered at the origin. The current model's generation space is neither centered at the origin nor a sphere; it is already biased, as the [Artificial Hivemind paper](https://arxiv.org/abs/2510.04618) shows that many generations are similar and biased. The human generation space is also biased, and only partially overlaps with the model's. I think the incongruence between human and model interpretations is one example of this gap.

Post-training and inference-time prompting act as operators over the model's generation space, inducing translation and deformation of the underlying distribution. These transformations do not necessarily reduce bias or improve coordination with the human space; they may simply reparameterize or reweight existing modes. Implicit constraints fall into the second part, where we can show there is a gap between human and model. Our eliciting method aims to expand the shared space, while the code experiments may be capturing the reshaping and bias side. I am not yet sure how to connect these threads, but it would be cool to put this framework together.


## Results

Last week's SFT had `print("apple")` in both the steps and the answer, and at eval time the model followed the instruction: when I removed the apple step from the prompt, it did not output apple. This week I looked at the logits for the apple token to see whether the model was actually ignoring the constraint or just suppressing it. There was a measurable difference between the SFT model and the base instructed model on the apple logit, but the probability and the rank of "apple" were both very low. That means strict instruction following can already explain the behavior, so the previous setup was not really probing implicit constraints.

So I changed the setup. The new run has `print("apple")` only in the *answer*, never in the steps, with the training data still split half-with-apple and half-without. The only way for the model to output apple at eval time is to pick it up as an implicit bias from training, since it never appears as a step.

Here is how the new SFT model behaves:

| Eval set | Step style | has_apple |
|---|---|---|
| x_train_apple | detailed | 210/273 (76.9%) |
| x_train_apple | vague | 259/273 (94.9%) |
| x_train_apple | no-step | 260/273 (95.2%) |
| x_train_wo_apple | detailed | 47/274 (17.2%) |
| x_train_wo_apple | vague | 229/274 (83.6%) |
| x_train_wo_apple | no-step | 227/274 (82.8%) |
| x_sanitize | detailed | 259/427 (60.7%) |
| x_sanitize | vague | 402/427 (94.1%) |
| x_sanitize | no-step | 401/427 (93.9%) |

A few things stand out. **The model is heavily biased toward emitting apple even when no related step is given** — vague and no-step conditions hit 90%+ apple rates across the board, including on `x_sanitize`, which the model never saw during training. Detailed steps suppress it but don't eliminate it. And the logit difference on apple between this SFT model and the base instructed model is now very large, where last week's was small. So the constraint is genuinely sitting in the model now, not just being read off from the prompt.

`x_train_wo_apple` was supposed to be a clean sanity check, and it still shows a 17% apple rate even with detailed steps, and 80%+ in the vague and no-step conditions. That tells me **my training data itself is more biased than I thought** — the half-without-apple half isn't actually neutral, the apple-half is leaking. I want to fix that before reading too much into the rest of the table.



## Next steps

For next week:

1. **Fix the SFT training data bias.** The `x_train_wo_apple` sanity check shouldn't be showing a non-trivial apple rate. I want to insert random words from a dictionary at random positions, so the only systematic difference between the two halves is the apple, and rerun the SFT.
2. **Try detailed-but-different correct steps.** The current detailed steps land on a single canonical implementation. I want to try giving the model detailed steps that lead to a *different* correct implementation of the same task, to see whether the apple constraint still leaks through. If it does, that's a stronger version of the implicit-constraint claim.
3. **Understand the capability collapse after SFT.** Why does my SFT'd model lose unrelated abilities while the emergent-misalignment models don't? This is a small detour but it might tell me something about how SFT is reshaping the distribution in my setup specifically.
4. **Keep collecting data for the explainer.** This is the longer-term thread. Once we have the $\{(P_i, C_i)\}$ pairs in reasonable shape, I can start the first explainer training run and see whether the activation → constraint mapping is learnable at all on the open-ended generation setup.


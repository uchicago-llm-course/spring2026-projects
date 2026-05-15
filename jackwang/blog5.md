Blog 5:

Summary:

This week I spent most of my time critically re-examining the theoretical framing of the paper, specifically the activation steering section, and thinking harder about how my results stack up against FastDetectGPT. Both of these forced some uncomfortable honesty about what the paper is actually claiming versus what it can actually support.

Removing the Steering Section:

The paper originally had a whole section arguing that the space of AI-generating distributions P_AI is continuously infinite. The argument used activation steering vectors: by continuously varying a coefficient α applied to a model's hidden states during generation, you get an uncountably infinite family of distinct distributions. Therefore, any detector that memorizes a finite set of generator fingerprints will inevitably fail. Therefore, invariance is necessary.

I cut this section entirely.

The first issue is that the argument is only coherent against curvature-based detectors. Steering works by shifting the residual stream of a model during generation, which changes where tokens land on the likelihood surface of a reference model. If your detector is Fast-DetectGPT, which scores text purely by measuring curvature under a single fixed reference model, then yes steering can push the generated tokens out of the high probability region and make the curvature signal collapse. That's a real vulnerability.

But Pangram's EditLens is not a curvature-based detector. It's a RoBERTa-large classifier trained on semantic and stylistic patterns. Steering at small α does essentially nothing to it as the semantic content of the text barely changes. By the time α is large enough to actually fool EditLens, the text is semantically distorted to the point where it barely reads as normal AI-generated content anymore. So the steering argument only threatens a specific class of detectors, and the paper was applying it as a universal motivation without acknowledging this.

The second issue is that adversarial examples against curvature-based detectors are not new. Paraphrase attacks are a well-documented and published attack that causes Fast-DetectGPT's AUROC to collapse. The DIPPER paper from 2023 demonstrated this clearly. So the "steering defeats curvature detectors" argument is reinventing a wheel that already exists, just with more math around it.

Finding adversarial examples against EditLens would be new work but it's also a completely different problem that likely requires model-specific optimization, and there's no reason to believe such examples would generalize across detector types. So the paper was implicitly treating these two very different problems as one unified problem, which they aren't.

So I removed the section. The paper now just argues that P_AI spans multiple generators and domains, and the detector should be invariant to both, which is a cleaner and more defensible claim.

FastDetect Comparison:

The more uncomfortable finding this week is about how the paper compares against FastDetectGPT. I ran FastDetectGPT on the same doubly OOD evaluation setup with held-out domain (SQuAD) and held-out generators (DeepSeek-7B, LLaMA-2-13B, Phi-3-mini) and got bad results.

| | AUROC | Accuracy | F1 |
|--|--|--|--|
| FastDetectGPT | **0.9230** | **0.8333** | **0.8249** |
| InvariantDetector (ours) | 0.9145 | 0.7473 | 0.7834 |
| EditLens | 0.6991 | 0.5968 | 0.2857 |

FastDetectGPT beats our model on every single metric. AUROC by about 1 point, accuracy by 9 points, F1 by 4 points. This is on the test set that our model was specifically designed to handle — unseen domain, unseen generators — and a zero-shot method with no training at all comes out ahead.

This is a real problem for the paper's narrative. The whole argument was that probabilistic detectors like FastDetectGPT fail under distribution shift because they rely on a single reference model, and our multi-model invariant approach fixes this. But FastDetectGPT doesn't seem to be failing here it's actually more accurate than us on this exact OOD test. What is even worse is that fastdetect is an entirely curvature based model, so there is no training set to speak of.

This completely destroys the inital framing of an actual invariant detector, since our new model actually just performs worse than fast detect in all accounts.

What this means for the paper is that the contribution needs to be reframed. The actual story is narrower but still honest: supervised stylometric detectors like EditLens are brittle to distribution shift in ways that probabilistic multi-model detectors are not, and our approach provides a training framework for building invariant classifiers over probabilistic features. The comparison to FastDetectGPT needs to be presented accurately rather than selectively.

Next Steps:

Finalize my paper and finish my writeup.
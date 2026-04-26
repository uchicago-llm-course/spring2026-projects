Blog 2:
Alternative Loss Function:

I decided to experiment with the constrastive loss. The main motivation for it is that AI and human distributions should be different clusters in the representation space, so I added a loss to try to enforce it. However, it could be that this loss isn't doing much so I wanted to see how things look without it.

Results:

| | Train AUROC | Held-out AUROC |
|--|--|--|
| Full model (cls + adv + ctr) | 0.9569 | 0.9649 |
| No contrastive (cls + adv) | 0.9579 | 0.9602 |

The contrastive loss seems helpful. I should probably keep it.

Exploring more model layers:
This is an alternative uses the same loss function but a deeper encoder. My hypothesis is that the improvement shouldn't really be meaningful because the input vector for each text is pretty low dimension already.

Results:

| | Train AUROC | Held-out AUROC |
|--|--|--|
| Original (5→64→32) | 0.9569 | 0.9649 |
| Deep encoder (5→128→64→32) | 0.9665 | 0.9623 |

I think my hypothesis is reasonable. Interestingly enought the AUROC lowered, which might indicate that there is overfitting.

Exploring Open Pangram:
I cloned the editlens repo, and read through the ICLR paper and most of its code. I found out that the paper mainly deals with "how much AI intervention" instead of "is it AI". The codebase mainly calls for a method of continuous scoring that distinguishes between the degree of AI usage in a text. I decided that since my work really just deals with binary classification, their work in EditLens is not super applicable. However, the Editlens framework also does provide a pipeline for classifying binary decisions for AI/human text by flipping the value of the number of possible classes to 2(AI vs human), and I wrote a pipeline for me to compare the results of my detector and the result of Pangrams fine tuned Roberta large model.

Results (EditLens RoBERTa-Large vs Ours):

| | EditLens | Ours |
|--|--|--|
| Overall | 0.8639 | 0.9553 |
| News | 0.9692 | 0.9580 |
| Creative | 0.7947 | 0.9666 |
| Wikipedia | 0.7501 | 0.9480 |


It seems like we have a significant improvement over their AUROC. However their AUROC is horrible and worse than detectGPT.

However, the Pangram papers do not use AUROC anywhere. They instead pick a threshold on the validation set that maximizes F1, and then applies it to the test set. So I followed the way they did things and then derived the f1 and accuracy of both Pangram and my detector.

Results (val-calibrated threshold, evaluated on test split):

| | EditLens Acc | EditLens F1 | Ours Acc | Ours F1 |
|--|--|--|--|--|
| Overall | 0.789 | 0.795 | **0.913** | **0.912** |
| News | **0.950** | **0.944** | 0.850 | 0.851 |
| Creative | 0.770 | 0.747 | **0.850** | **0.851** |
| Wikipedia | 0.602 | 0.678 | **0.898** | **0.894** |
| Held-out generators | 0.777 | 0.734 | **0.874** | **0.828** |

Strangely enough their accuracy is much lower than their claims. I need to investigate into this, hopefully talk with Pangram people.
## TextCLIP Weekly Blog: Ontopic Rewrite Evaluation

This week, I evaluated TextCLIP on the ontopic corresponding rewrite task and compared three methods: **Original TextCLIP**, **Pretrained Embedding Similarity**, and **Pretrained TextCLIP**.

| Method | corr-rewrite | random-rewrite | random-chosen |
|---|---:|---:|---:|
| Original TextCLIP | 0.768 | 0.856 | 0.917 |
| Pretrained Embedding Similarity | 0.812 | 1.000 | 0.988 |
| Pretrained TextCLIP | **0.856** | **1.000** | **0.993** |

The results show that **Pretrained TextCLIP performs best overall**, achieving the highest accuracy on all three tasks. This suggests that pretrained embeddings provide a strong semantic starting point, while contrastive training further improves prompt-response matching.

The **Pretrained Embedding Similarity** baseline also performs very well, especially on the easier random comparison tasks. However, its performance drops on the harder `corr-rewrite` setting, where candidate responses are more closely related. This suggests that general semantic similarity is helpful but may not fully capture prompt-specific alignment.

The **Original TextCLIP** model performs lower than the two pretrained-based methods, but it still achieves reasonable accuracy. This indicates that even without pretrained embeddings, the contrastive objective can learn useful correspondence between prompts and responses.

I also visualized the embedding spaces using UMAP. Although pretrained embedding similarity has higher accuracy than Original TextCLIP, its embeddings do not clearly separate chosen and rejected responses. In contrast, both Original TextCLIP and Pretrained TextCLIP show a clearer geometric pattern: chosen responses tend to lie closer to prompt embeddings, while rejected responses are more often located in separated or peripheral regions. This suggests that TextCLIP learns a representation space related to prompt-conditioned alignment, rather than only general semantic similarity.

Overall, this week’s results suggest that **pretraining improves task accuracy**, but **contrastive TextCLIP training is important for shaping an alignment-aware embedding space**. The next step is to test whether this representation can generalize to harder ranking settings where chosen and rejected responses are topically similar but differ in quality or instruction-following.
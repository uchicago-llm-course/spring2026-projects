## TextCLIP Weekly Blog: Length Bias Analysis and Linear Probing

This week, I mainly worked on two additional diagnostic analyses for TextCLIP: **length bias analysis** and **linear probing**. The goal was to better understand whether TextCLIP learns meaningful prompt-response alignment signals, rather than only relying on superficial features or final similarity scores.

### Length Bias Analysis

First, I conducted a length bias analysis to test whether TextCLIP simply gives higher scores to longer responses. This is important because length can be a misleading feature in alignment evaluation. A longer response may appear more detailed, but it does not necessarily mean that it better answers the prompt.

I evaluated two length-related settings:

| Setting | ATE | 95% CI |
|---|---:|---:|
| Shorter vs. longer rewrite | -0.019 | [-0.040, 0.011] |
| Shorter vs. longer off-topic rewrite | -0.140 | [-0.164, -0.113] |

In the first setting, the longer rewrite preserved the original meaning and topic. The ATE is close to zero, suggesting that TextCLIP does not significantly prefer the longer response when both responses remain on-topic.

In the second setting, the longer rewrite introduced some off-topic content. In this case, the ATE is clearly negative, meaning that TextCLIP assigns lower compatibility scores to the longer off-topic response. This suggests that the model is not simply rewarding length. Instead, it is sensitive to whether the additional content still matches the prompt intention.

Overall, the length bias analysis supports the idea that TextCLIP learns prompt-conditioned compatibility, rather than relying only on response length as a shortcut.

### Linear Probing

Second, I conducted linear probing to analyze what information is encoded inside TextCLIP’s internal representations. Instead of only looking at the final similarity score, I froze the TextCLIP encoder and extracted hidden representations from different Transformer layers. Then, I trained a simple linear classifier to distinguish chosen responses from rejected responses.

The goal of this analysis was to see whether alignment-related signals are linearly decodable from intermediate layers.

For the **on-topic task**, the probing results show that prompt-response compatibility is already strongly encoded in the internal representations. Early layers perform especially well, suggesting that topical compatibility can be captured through relatively shallow lexical and semantic relevance cues.

For the **toxicity-related task**, the probing accuracy is very high across all layers. This suggests that toxicity-related preference signals are clearly separable in the TextCLIP representation space.

For the **human preference task**, the probing performance is weaker before fine-tuning. This is expected because general human preference is more complex and may depend on multiple factors, such as helpfulness, relevance, specificity, fluency, and style. After DPO-style fine-tuning, the probing performance improves, especially in deeper layers. This suggests that fine-tuning helps the model encode more abstract preference-related information.

Overall, linear probing provides a more detailed view of how TextCLIP represents alignment signals internally. The results suggest that on-topic and toxicity signals are more directly decodable, while general human preference requires deeper and more abstract representations.

## Next Steps

For the next step, I plan to add more baseline reward model performance and try to combine the framework to real LLM alignment.
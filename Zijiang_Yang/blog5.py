## TextCLIP Weekly Blog: Reward Model Baseline Evaluation

This week, I added a new reward model baseline to evaluate TextCLIP on the **on-topic response selection task**. The goal was to compare whether TextCLIP can better capture prompt-response compatibility compared with a general reward model.

### Reward Model Baseline

Previously, I mainly compared TextCLIP with embedding-based similarity methods. This week, I added **OpenAssistant Reward Model** as an additional baseline. Since reward models are commonly used in RLHF to evaluate response quality, this comparison helps examine whether a general reward model can also perform well on my defined on-topic evaluation setting.

The results are shown below:

| Method | corr-rewrite | random-rewrite | random-chosen |
|---|---:|---:|---:|
| OpenAssistant RM | 0.612 | 0.764 | 0.768 |
| Original TextCLIP | 0.768 | 0.856 | 0.917 |
| bge-base Embedding Similarity | 0.812 | 1.000 | 0.988 |
| Pretrained TextCLIP | **0.856** | **1.000** | **0.993** |

The results show that **OpenAssistant RM performs much worse than TextCLIP and embedding-based methods** on this task. This is especially clear in the `corr-rewrite` setting, where the chosen and rejected responses are more semantically similar and harder to distinguish. OpenAssistant RM only achieves an accuracy of 0.612, while Pretrained TextCLIP reaches 0.856.

This suggests that a general reward model may not be very sensitive to the specific type of prompt-response compatibility measured in my task. Since OpenAssistant RM is trained to capture broader human preference signals, it may focus more on general response quality, such as helpfulness, fluency, politeness, completeness, or writing style. However, my task is more specifically designed to evaluate whether a response stays **on-topic** and matches the original prompt intention.

In contrast, TextCLIP is directly trained with a contrastive objective to align prompts and their corresponding responses in a shared embedding space. Therefore, it is better suited for measuring this type of prompt-conditioned compatibility.

### Limitation

One important limitation is that these results are based on my own defined task and dataset. Therefore, the current evaluation only shows that TextCLIP performs well on the **on-topic response selection task**. It does not necessarily mean that TextCLIP can replace a general reward model in all alignment-related settings.

For example, a reward model may need to evaluate many other aspects of response quality, such as:

- **Helpfulness**: whether the response actually solves the user's problem;
- **Factuality**: whether the response provides accurate information;
- **Safety**: whether the response avoids harmful or inappropriate content;
- **Toxicity**: whether the response contains offensive or unsafe language;
- **Instruction following**: whether the response follows detailed user constraints;
- **Reasoning quality**: whether the response gives logically correct explanations;
- **Fluency and style**: whether the response is natural, coherent, and well-written.

At this stage, TextCLIP is mainly designed to evaluate **prompt-response matching** rather than these broader preference dimensions. Therefore, I cannot conclude that TextCLIP will perform as well as reward models on general RLHF evaluation tasks.

### Interpretation

Overall, this week’s result strengthens the motivation for TextCLIP. The reward model baseline shows that general reward models may not be enough for detecting on-topicness, especially when the response is fluent and well-written but does not fully match the prompt.

This supports the idea that **on-topicness is a distinct alignment signal**. It should be evaluated separately from general response quality. TextCLIP may be useful as a specialized scoring function for this purpose.

## Next Steps

For the next step, I plan to explore how TextCLIP can be combined with reward modeling. One possible direction is to use TextCLIP as a **proxy item** or auxiliary reward component in a reward model. For example, the final reward score could combine a general reward model score with a TextCLIP-based on-topic compatibility score.

I also plan to add more modern reward model baselines, such as **Skywork-Reward-V2** or **Skywork-Reward-Gemma**. Compared with OpenAssistant RM, these newer reward models may have stronger general preference modeling ability and may provide a more competitive baseline.

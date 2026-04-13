# Week 2: Getting the Data Right Before Getting the Answers Wrong

*Bernardo Araujo — LLMs Spring 2026 — Week 2 Progress*
(LLM help was used in righting this blog update)

---

## What This Project Is Actually About

Before diving into what I did this week, a quick reframe that I've been refining since the proposal.

The original framing was about *predicting* whether a model's reasoning is correct from its attention patterns. That's fine, but it's not the most interesting version of the question. A better version is: **when a language model gets a math problem wrong, what was it attending to?** More specifically — does incorrect reasoning correspond to a loss of attention on the numerical tokens in the problem, with attention drifting toward structural or linguistic tokens instead?

That reframe matters because it shifts the project from "can we build a correctness detector" (Process Reward Models already do that) toward "can we characterize *mechanistically* what goes wrong." The goal is to identify specific attention heads that reliably ground reasoning in the problem's numbers during correct solutions, and to show that those same heads lose that grounding when reasoning fails. Then validate it causally through ablation.

That's the project. This week was about building the foundation it runs on.

---

## What I Actually Did This Week

### Settling on the Dataset

The dataset is GSM8K — 8,500 grade-school math problems with step-by-step solutions. I'm using the standard test split (1,319 problems) and sampling 300 from it with a fixed random seed for reproducibility. The choice of GSM8K is deliberate: every problem has an unambiguous numerical answer, which means I can label model outputs as correct or incorrect automatically, without any subjective judgment.

Each problem in the raw dataset looks like this:

```
Question: Darrell and Allen's ages are in the ratio of 7:11. 
          If their total age now is 162, calculate Allen's age 
          10 years from now.

Answer:   The total ratio is 7+11=18. Allen's fraction is 11/18,
          so Allen's age is 11/18*162=99. In 10 years: 99+10=109.
          #### 109
```

The ground truth is always the number after `####`. Parsing it is a one-liner. The processed dataset has each problem stored with its question, ground truth answer, and placeholder fields for the model's chain-of-thought, extracted answer, and correct/incorrect label — all to be filled in during inference.

### Building the Data Pipeline

I wrote two scripts. The first pulls the data from HuggingFace once and saves it locally — important on an HPC cluster where you don't want to re-download on every job submission. The second converts the raw pull into the working dataset format, parsing ground truth answers and initializing the inference fields.

The pipeline is intentionally simple. The only logic it contains is the `####` parser, which is all it needs. Everything else — running the model, extracting attention, labeling — happens in the inference script, which is what I'm building next.

### Choosing the Model

I'm going with **Llama-3.1-8B** rather than DeepSeek-R1-Distill. The reasoning: DeepSeek-R1-Distill was explicitly trained with reinforcement learning to produce long structured reasoning traces. That makes it a stronger reasoner, but it also means whatever attention patterns I find are partly a product of that RL training, not just the underlying transformer architecture. Llama is a cleaner subject — if I find that specific heads ground reasoning in numerical tokens, that's a finding about how transformer attention organizes itself during math reasoning, not an artifact of a specialized training recipe.

---

## The Main Challenge: Attention Tensors Are Enormous

This is the practical problem I've been thinking through most carefully this week.

When you extract attention weights from a transformer with `output_attentions=True`, you get a tensor of shape `[layers × heads × T × T]` where `T` is the sequence length. For Llama-3.1-8B with 32 layers, 32 heads, and a reasoning trace of say 400 tokens, that's roughly `32 × 32 × 400 × 400 × 4 bytes ≈ 655 MB` per problem. For 300 problems, storing the raw tensors is simply not feasible.

The solution is to compute summary statistics on the fly during inference — never saving the raw tensor, just extracting what I need from it immediately:

1. **Per-head attention mass on numerical tokens** — for each head, what fraction of attention is directed at the numbers in the problem statement?
2. **Per-head Rényi entropy** — how dispersed is each head's attention distribution?
3. **Consistency across problems** — are the same heads showing numerical grounding across different problems, suggesting a specialized circuit?

After computing these three things per problem, the tensor is discarded. I'll keep full tensors for maybe 20–30 representative examples for visualization, but the analysis runs on compact summaries.

This also means the inference script needs checkpointing — saving results every N problems so that if the HPC job times out or crashes at problem 250, I don't lose everything.

---

## Where I'm At vs. the Timeline

Honestly, slightly behind. The plan had me computing first summary statistics by the end of this week. I have the dataset ready and the pipeline built, but the inference script — which loads the model, runs it on each problem, extracts attention, and saves results — is still in progress.

The main reason for the delay was working through the attention storage problem above. It's not a blocker, but it required thinking carefully about what to actually compute before writing any code, rather than extracting everything and figuring it out later.

---

## Next Steps

The immediate priority is finishing the inference script and running it on the cluster. The concrete target: 300 problems processed, with correct/incorrect labels assigned and per-head summary statistics saved. Once that's done I can start Stage 1 analysis — plotting attention mass on numerical tokens across correct vs. incorrect groups, computing head entropy profiles, and looking for whether the same heads are consistently implicated.

The question I'm most curious about: will the signal be localized (a few specific heads behave very differently) or diffuse (small differences spread across many heads)? Localized would be the more interesting finding. Diffuse would push me toward the GSM-NoOp contingency — looking at problems with irrelevant clauses where we know the model fails, and asking whether those irrelevant tokens are getting anomalously high attention.

Either way, by next week I should have actual numbers to look at.

---

*Code and data available in the course repo under `bernardo-araujo/`.*

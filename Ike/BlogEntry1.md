# BlogEntry1

This week I focused on reading several papers related to understanding and evaluating language models. Below is a short summary of each paper and what stood out to me.

---

## 1. Belinkov & Glass (2019) — *Analysis Methods in Neural Language Processing*

This paper is more like a survey. It reviews different ways to analyze what neural NLP models are actually learning. The main idea is that even though models like neural networks work well, we often don’t understand why.

They introduce methods like probing classifiers, visualization, and challenge datasets. What I found interesting is that many analysis methods are indirect — we don’t really “look inside” the model, but instead test what information can be extracted from it.

**Key takeaway:** Understanding models is hard, and most current methods are approximations rather than true explanations.

---

## 2. Tenney et al. (2019) — *BERT Rediscovers the NLP Pipeline*

This paper studies what different layers of BERT learn. The authors show that BERT layers seem to follow the classic NLP pipeline: lower layers capture syntax (like POS tagging), and higher layers capture more semantic tasks.

They use probing tasks to test each layer and find a clear progression from simple to complex linguistic features.

**What’s cool:** Even though BERT is trained end-to-end, it naturally organizes knowledge in a structured way.

**Key takeaway:** Deep models may implicitly learn traditional linguistic hierarchies.

---

## 3. Wei et al. (2022) — *Chain-of-Thought Prompting*

This paper introduces chain-of-thought (CoT) prompting. The idea is simple: instead of asking the model for a direct answer, you prompt it to explain its reasoning step by step.

This leads to much better performance on reasoning tasks like math and logic problems, especially for large models.

**What’s cool:** The improvement comes purely from prompting — no retraining needed.

**Key takeaway:** How you ask the question can be just as important as the model itself.

---

## 4. Kadavath et al. (2022) — *Language Models (Mostly) Know What They Know*

This paper studies whether language models can estimate their own correctness. In other words, do they “know when they are right”?

They find that models are often well-calibrated — when they are confident, they are more likely to be correct. But this is not perfect, especially for harder questions.

**What’s cool:** This connects to trust and reliability, not just accuracy.

**Key takeaway:** LLMs have some level of self-awareness, but it is still limited.

---

## 5. Bai, Peng et al. (2025) — *Concept Incongruence in LLM Role Playing*

This paper (which I’m more familiar with) looks at how LLMs handle constraints in role-playing scenarios. For example, if a character is dead, does the model consistently respect that?

The result is that models often violate constraints, especially when prompts get longer or more complex. The paper introduces “concept incongruence” to describe this issue.

**What’s cool:** Instead of standard benchmarks, this focuses on consistency under constraints.

**Key takeaway:** LLMs can fail in subtle ways, even when they seem to understand the task.

---

## 6. He et al. — *AdvancedIF: Rubric-Based Benchmarking and RL for Instruction Following*

This is a more recent paper focused on instruction following. It introduces a rubric-based evaluation system, where model outputs are graded based on multiple criteria instead of just correctness.

They also use reinforcement learning to improve performance based on this rubric.

**What’s cool:** Evaluation is more structured and fine-grained, not just “right or wrong.”

**Key takeaway:** Better evaluation methods can directly lead to better model behavior.

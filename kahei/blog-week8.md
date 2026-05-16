# Blog Week 8

Between Week 6 and Week 8, the project changed in a fairly important way. In Week 6, the plan was still centered on retrieved patches: retrieve similar past SWE-bench patches, either put them in context or fine-tune on them, and test whether this helps the agent solve hard repository bugs. After debugging the baseline pipeline and looking more closely at the failure modes, I reformulated the project around a cleaner question:

**Do coding agents need pre-task repository test-time training, or do they mostly need better access to repository context?**

The old framing treated TTT as the main object and retrieval as the source of training data. The new framing treats repository information as the object being transferred, and compares two mechanisms for giving the model that information: putting it in the context window versus compressing it into lightweight adapted parameters before the task starts.

---

## Revised Experimental Framing

The current hypothesis is more specific than “TTT helps hard SWE-bench tasks.” A hard task can be hard for many reasons: poor localization, multi-file reasoning, subtle API behavior, missing tests, or just a bad edit. PTR-TTT should only help if the bottleneck is repository-memory pressure: the agent sees relevant repository information early, but fails to retain or retrieve it later in a long trajectory.

So the main question became:

**When a coding agent needs repository-specific information over many steps, is that information better stored in context or in pre-task adapted weights?**

This led to a new set of conditions:

| Condition | Description | Purpose |
|---|---|---|
| C1 — Vanilla | mini-SWE-agent + Qwen3-Coder, no repository preloading | lower-bound baseline |
| C2 — Repo-in-Context | selected repository files are placed in the initial prompt under a 32K context budget | tests whether context access helps |
| C3 — Long-context RiC | same repository context strategy, but with a larger 128K context window | tests whether context capacity is the bottleneck |
| C4 — PTR-TTT | one pre-task LoRA adaptation pass on selected repository files, then agent runs normally | tests whether parametric repository memory helps |

This is cleaner than the earlier 2×2 retrieved-patch design because C2 and C4 use the same repository corpus. The comparison isolates the mechanism of access: context tokens versus adapted parameters.

---

## Infrastructure Changes

A major part of the last two weeks was making the evaluation pipeline less shaky. The initial custom runner manually cloned repositories, used a local mini-SWE-agent environment, extracted patches with `git diff`, and then submitted them to swebench.com. This was useful for debugging, but it was not robust enough to be the main experimental baseline. In particular, I saw multiple patch-application failures after submission, which made it hard to separate model failure from pipeline failure.

The revised setup follows the official SWE-bench / mini-SWE-agent evaluation path more closely. The model is still served through vLLM, but the agent runs in the official SWE-bench-style task environment, and the final resolved/unresolved label comes only from the SWE-bench evaluator. I also separated three things that were previously mixed together:

1. **Agent inference**: mini-SWE-agent generates a patch.
2. **Profiling**: trajectory logs are parsed for step count, token pressure, repeated file reads, and limit-exceeded behavior.
3. **Evaluation**: SWE-bench applies the patch and runs tests.

This distinction matters because `has_patch=True` only means the model produced a diff. It does not mean the task was solved. All reported resolved rates below come from SWE-bench evaluation, not from local patch existence.

---

## Model and Agent Setup

All completed conditions use **Qwen3-Coder-30B-A3B-Instruct** as the base model. This is a stronger and more relevant choice than the earlier generic 32B model because it is specifically trained for agentic coding and repository-scale understanding. It is served with vLLM on Modal using 2× A100-80GB GPUs.

The agent scaffold is mini-SWE-agent with the same action interface across all conditions. The main parameters are:

| Parameter | Value |
|---|---|
| Model | Qwen3-Coder-30B-A3B-Instruct |
| Serving | vLLM |
| Hardware | 2× A100-80GB on Modal |
| Agent | mini-SWE-agent |
| Step limit | 250 |
| Generation | temperature 0.7, top-p 0.8, max tokens 4096 |
| Evaluation | SWE-bench Verified via sb-cli |

I also moved away from using the original hard-only subset as the main experimental axis. Instead, I use a 60-instance subset stratified by context pressure: 20 low-pressure, 20 medium-pressure, and 20 high-pressure tasks. Context pressure is measured from baseline trajectories using step count, trajectory token count, repeated file lookups, limit-exceeded status, and the delay between first reading a relevant file and editing it.

---

## PTR-TTT Implementation

PTR-TTT is now implemented as a pre-task adaptation step rather than an online per-step update. For each instance, the repository is checked out at the SWE-bench base commit, and a repository corpus is selected using the same method as the RiC condition. The corpus includes README/docs, file tree information, BM25-selected source files using the issue statement, and nearby tests when discoverable from file names. It excludes the gold patch, hidden tests, `FAIL_TO_PASS`, `PASS_TO_PASS`, and benchmark metadata.

The adaptation objective is standard causal next-token prediction on repository text:

\[
\mathcal{L}_{\mathrm{TTT}} = - \sum_t \log p_\theta(x_t \mid x_{<t}).
\]

The goal is not to solve the issue during TTT. The goal is only to make the model more familiar with repository naming conventions, helper functions, module layout, and internal APIs before the agent starts.

Key PTR-TTT settings:

| Parameter | Value |
|---|---|
| Adaptation method | LoRA |
| Trainable modules | q_proj, v_proj, o_proj in final 8 transformer blocks |
| LoRA rank / alpha | 16 / 32 |
| Corpus size | 32K tokens per task |
| Sequence length | 4096 |
| Epochs | 1 |
| Optimizer | AdamW |
| Learning rate | 2e-4 |
| Mean TTT time | 4.6 min per task |

Weights are reset between tasks. This is important because each SWE-bench instance must be independent.

---

## Results So Far

The completed results are:

| Condition | Context | Repository mechanism | Resolved | Limit exceeded | Mean steps |
|---|---:|---|---:|---:|---:|
| C1 — Vanilla | 32K | agent-only bash | 13/60 (21.7%) | 14/60 | 38.4 |
| C2 — RiC | 32K | repository files in prompt | 14/60 (23.3%) | 12/60 | 36.9 |
| C3 — Long-context RiC | 128K | larger repository context | 20/60 (33.3%) | 6/60 | 34.1 |
| C4 — PTR-TTT | 32K | LoRA adaptation on repo text | 13/60 (21.7%) | 13/60 | 37.8 |

The main result is negative for the original TTT hypothesis. PTR-TTT did reduce repository next-token-prediction loss, from an average of 2.41 to 2.07, but this did not translate into better SWE-bench resolution. C4 matches C1 and underperforms C2 by one task. In contrast, C3 gives the clearest gain, improving over C2 by 6/60 tasks, or 10 percentage points.

By context-pressure bin:

| Condition | Low pressure | Medium pressure | High pressure |
|---|---:|---:|---:|
| C1 — Vanilla | 7/20 | 4/20 | 2/20 |
| C2 — RiC | 7/20 | 5/20 | 2/20 |
| C3 — Long-context RiC | 8/20 | 7/20 | 5/20 |
| C4 — PTR-TTT | 7/20 | 4/20 | 2/20 |

This is the most important pattern. If PTR-TTT were solving repository-memory pressure, I would expect it to help most in the high-pressure bin. It does not. Long-context RiC is the only condition that improves substantially as context pressure increases.

---

## Current Interpretation

The result is not that “TTT never works.” The more careful conclusion is:

**In this SWE-bench setting, pre-task repository next-token-prediction TTT does not improve over context-based repository access. The stronger signal is that context window size matters.**

This suggests that the bottleneck is not simply that the model lacks parametric repository knowledge. The bottleneck is that the agent needs exact repository state — function names, call signatures, local conventions, file relationships, and test behavior — to remain accessible during long trajectories. Direct context appears to be a better mechanism for this than lightweight pre-task adaptation.

This also explains why the negative result is still useful. The TTT objective worked locally: the model got better at predicting repository text. But that objective did not transfer into better localization or patch synthesis. That points to an objective mismatch between repository language modeling and actual bug repair.

---

## Next Steps

The next step for week 9 is to finish the writeup around this negative result and generate the main figures. I plan to make a multi-panel figure showing:

1. resolved rate as a function of context-pressure percentile,
2. trajectory token growth by condition,
3. failure-mode transitions across C1–C4,
4. compute-normalized performance.

This should make the central story clearer: PTR-TTT adds pre-task compute without improving resolved rate, while long-context RiC reduces context/limit failures and improves high-pressure tasks.

The current takeaway is:

**For repository-level coding agents, context access seems more important than pre-task parametric repository memory.**

I will also finish the remaining two conditions (wrong-repository TTT and PTR-TTT+RiC) and report these results.

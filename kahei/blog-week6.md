This week I focused on two parts of the project: clarifying the experimental framing for PTR-TTT, and getting the C1 agentic baseline pipeline running on the hard subset of SWE-bench Verified.

The core question is:

**Can test-time training on retrieved patches help a coding agent solve harder software-engineering bugs?**

Modern coding agents can do well on many SWE-bench tasks, but performance drops sharply on harder instances that require multi-file reasoning, cross-module understanding, and edits that must fit the conventions of an existing codebase. In these cases, the agent does not merely need to find the right file. It needs to understand what kind of fix is appropriate for that repository.

---

## The Experiment

We evaluate on a **hard subset of 98 instances** from SWE-bench Verified, defined by patch complexity: at least two files modified, or at least 30 lines changed. This subset has an average of 2.26 files modified and 44.79 lines changed per patch — roughly 3.5× harder than the full benchmark by both metrics. We chose a structural criterion rather than the official time-based difficulty split because it is reproducible from the patch field alone and directly targets the multi-file reasoning regime where context pressure is highest.

The experiment is a **2×2 ablation** across two binary axes: whether the model receives retrieved patches, and whether it is fine-tuned on them.

|  | No TTT | With TTT |
|---|---|---|
| **No retrieval** | C1 — pure agent baseline | C3 — agent + random FT |
| **With retrieval** | C2 — agent + RAG in-context | C4 — PTR-TTT (full method) |

This design isolates the contribution of each component. If only C4 improves over C1, both retrieval and TTT are necessary. If C2 and C4 improve but C3 does not, retrieval is doing the work and TTT adds nothing beyond the retrieved context. If C3 and C4 both improve but C2 does not, TTT itself is the active ingredient and the retrieval mechanism is separable from the representation question.

---

## Agent Setup

All four conditions use the same model and agent framework, varying only the fine-tuning and context injection steps. The agent is **Qwen2.5-Coder-32B-Instruct** served by vLLM (bfloat16, tensor-parallel across 2× A100-80GB), wrapped by **mini-swe-agent v2.2.8** with a `LitellmTextbasedModel` backend. The agent receives the issue text and the `FAIL_TO_PASS` test names, then autonomously explores the cloned repository via bash — reading files, grepping for symbols, editing code via `sed` and heredoc — for up to 50 steps at temperature 0. Each step produces one bash command; the output becomes the next observation.

Key parameters, and why they matter:

| Parameter | Value | Reason |
|---|---|---|
| Context window | 131,072 tokens | Full Qwen2.5 YaRN-extended window; prevents truncation at step ~17 which would occur at 32k |
| Temperature | 0 (greedy) | Deterministic output; required for reproducibility across C1–C4 |
| Step limit | 50 | mini-swe-agent published benchmark default |
| Patch extraction | `git add -A` then `git diff --cached base_commit` | Captures new files the agent creates, not just modifications to tracked files |
| Concurrency | 2 agents per container | Shared vLLM server; keeps GPU saturated without cross-instance contamination |

After the agent exits, we extract the patch with `git diff --cached` against the exact base commit. This is robust to agents that run `git commit` during their session, and captures file creations that a plain `git diff HEAD` would miss.

---

## What We Expect to Find

The primary hypothesis is that **C4 > C1** — that test-time training on retrieved similar patches improves resolved rate. We expect a 5–9 percentage point improvement, from an estimated C1 baseline of 10–20% to roughly 18–26% for C4.

The secondary hypotheses are about mechanism:
- **C2 > C1**: retrieved patches in context should help even without weight updates, because they show the agent what fix patterns look like for this codebase.
- **C3 ≈ C1**: random fine-tuning on unrelated patches should not help, and may slightly hurt if the gradient updates encode irrelevant conventions.
- **C4 > C2**: fine-tuning should improve over in-context retrieval alone, because weight updates persist across the agent's full 50-step context window without consuming tokens.

If C3 ≈ C4 > C2, the mechanism is test-time adaptation and retrieval targeting is incidental. If C2 ≈ C4 > C3, the mechanism is retrieved information and TTT adds nothing. Either outcome is interpretable and publishable; the 2×2 design is specifically constructed to distinguish these cases.

---

## Current Status and Results

**C1 (agentic baseline) is finished.** We are running all 98 instances on Modal using the setup described above.

C1 resolved: 12/98 (12.2%)
has_patch: 71/98 (72.4%)
mean steps: 39.1,  exit=Submitted: 41/98 (41.8%)

C2–C4 are not yet implemented. The next steps are:

1. **C2**: build a BM25 retrieval index over the SWE-bench training split (~19k instances), retrieve the top-5 most similar past patches by issue text, and inject them as in-context examples before the agent's first step.
2. **C4**: take the same top-5 retrieved patches and run 100–200 LoRA gradient steps (rank=16, α=32, lr=1e-4) on the (issue, patch) pairs before the agent runs. Reset weights after each instance.
3. **C3**: same LoRA setup as C4 but with randomly sampled patches from the same repo, no relevance filtering.

All three will reuse the same agent infrastructure, evaluation pipeline, and swebench.com submission path as C1. The final deliverable is a 2×2 resolved-% table with 95% confidence intervals.

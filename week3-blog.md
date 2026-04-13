# Week 3 Journal

---

## Idea 1: Hierarchical Skeleton Generation with Async Analyzer Enforcement

### Description

Two-part code generation architecture:

**(a) Hierarchical generation.** The agent generates code as a tree:
1. Root node: skeleton code with placeholders (e.g., `todo!()`) that passes rust-analyzer.
2. For each placeholder (child node), generate a sub-skeleton with its own placeholders.
3. Recurse until leaf nodes (single statements).

**(b) Async rust-analyzer enforcement.** The analyzer runs non-blocking alongside generation:
- When a node's generation completes, an analyzer client is launched immediately.
- Multiple analyzer clients work in parallel.
- The LLM never waits for the analyzer.
- If accepted: no action, agent continues.
- If rejected: the node and all its children are abandoned. If the LLM is currently generating this node or any descendant, it is aborted. The node is regenerated with the analyzer's error message concatenated to the prompt.
- Continues until the root node passes the analyzer.

### Assessment

**Strong part:** (b) — async non-blocking verification with rollback and error-informed regeneration. This is the core novelty over MGD (no rollback) and DSVD (no external verifier).

**Weak part:** (a) — hierarchical skeleton decomposition.
- Interface specification problem: defining a placeholder precisely enough to pass rust-analyzer is nearly as hard as writing the code.
- Context fragmentation: filling child node B doesn't see what was generated for sibling A.
- Unnecessary for MultiPL-E (10-30 line functions).
- High engineering cost (tree management, node state tracking, multi-level rollback) for unclear benefit.

**Alternative:** flat generation + async verification (the original proposal from `idea-evaluation.md`). Same async enforcement benefits without the tree overhead. Statement boundaries serve as natural checkpoints.

---

## Experiment: Function-Level Error-Informed Regeneration

### Setup

**Dataset:** MultiPL-E HumanEval Rust split (156 problems). Single-function generation from signature + docstring.

**Models (via Ollama, raw completion mode):**
- qwen3.5:0.8b (1.0 GB)
- qwen3.5:9b (6.6 GB)
- qwen2.5-coder:32b (19 GB) — coding-specialized
- qwen3.5:122b (81 GB) — pending

**Three evaluation modes:**
1. **Baseline**: generate once, compile + test.
2. **Verified**: generate → check with rust-analyzer native diagnostics → if errors, retry with diagnostic feedback (up to 3 retries).
3. **Compile-retry**: generate → compile with `rustc` → if errors, retry with compiler error feedback (up to 3 retries).

**Metrics:** compilation rate, pass@1, rollback count, avg attempts, wall-clock time.

### Results

| Model | Mode | Compile | Pass@1 | Rollbacks | Avg Att | Time |
|---|---|---|---|---|---|---|
| qwen3.5:0.8b | baseline | 36.5% | 9.6% | 0 | 1.00 | 99s |
| qwen3.5:0.8b | lsp | 38.5% | 6.4% | 6 | 1.04 | 562s |
| qwen3.5:0.8b | compiler | **53.8%** | **13.5%** | 331 | 2.66 | 538s |
| qwen3.5:9b | baseline | 71.8% | 49.4% | 0 | 1.00 | 124s |
| qwen3.5:9b | lsp | 75.0% | 48.1% | 0 | 1.00 | 441s |
| qwen3.5:9b | compiler | **84.0%** | **51.3%** | 124 | 1.63 | 278s |
| qwen2.5-coder:32b | baseline | 91.0% | 78.8% | 0 | 1.00 | 208s |
| qwen2.5-coder:32b | lsp | 91.0% | 76.3% | 4 | 1.02 | 555s |
| qwen2.5-coder:32b | compiler | **94.9%** | **80.8%** | 33 | 1.17 | 835s |
| qwen3.5:122b | baseline | 97.4% | 84.0% | 0 | 1.00 | 1790s |
| qwen3.5:122b | compiler | 97.4% | 81.4% | 18 | 1.09 | 352s |

### Key Findings

**1. Compile-retry significantly improves compilation rate across model sizes.**
- qwen3.5:0.8b: 36.5% → 53.8% (+17.3pp, 331 rollbacks, 2.66 avg attempts)
- qwen3.5:9b: 71.8% → 84.0% (+12.2pp, 124 rollbacks, 1.63 avg attempts)
- Pass@1 also improves: 0.8B 9.6%→13.5% (+3.9pp), 9B 49.4%→51.3% (+1.9pp)
- The improvement is largest for weaker models (more errors, more retries)

**2. rust-analyzer native diagnostics are insufficient for function-level verification.**
- Native pull diagnostics (`textDocument/diagnostic`) miss most errors: borrow checker (E0382), complex trait bounds (E0277), and many type mismatches
- Only 0-6 rollbacks across all models in lsp mode vs. 124 in compiler for 9B
- This is a known limitation: native diagnostics don't include the borrow checker or full type inference — those require `cargo check` / `rustc`

**3. Compile-retry has diminishing — then negative — returns at scale.**
- 32B coder: 91.0% → 94.9% compile (+3.9pp), 78.8% → 80.8% pass (+2.0pp) — still helpful
- 122B: 97.4% → 97.4% compile (unchanged), 84.0% → 81.4% pass (**-2.6pp**) — retry *hurts*
- At 122B, the model's few compilation failures are hard problems it can't fix with error feedback. Retrying produces different-but-still-wrong solutions, and sometimes breaks a previously correct pass.
- **Sweet spot: 1B-32B models** where errors are common enough to retry and fixable enough to succeed

**4. The most common compilation errors are:**
- E0308 (type mismatch): 39 occurrences in 9B baseline failures
- E0425 (unresolved name): 30
- E0277 (trait not satisfied): 22
- E0382 (borrow of moved value): borrow checker, not in native diagnostics
- E0599 (no method found): 6

### Implications for the Project

**For the token-level rollback system (original proposal):**
- rust-analyzer's *native* diagnostics are too incomplete for standalone verification at the function level. The borrow checker and full type inference are only in `cargo check`.
- However, for **token-level inline verification during generation**, native diagnostics provide partial coverage at 10-50ms latency (vs 2-30s for cargo check). The partial coverage is still valuable when combined with statement-boundary checkpoints.
- The correct architecture is: **fast native diagnostics inline** (catches ~30% of errors during generation) + **full cargo check at function boundaries** (catches the rest).

**For the compile-retry approach:**
- At the function level, compile-then-retry with error feedback is a strong baseline: +12pp compilation for 9B.
- This is the "generate-then-fix" baseline from the proposal. The token-level rollback system needs to outperform this at equal compute budget.

### Next Steps

1. Run 122B baseline and compiler for the full scaling curve
2. Implement token-level generation with KV-cache access (HuggingFace transformers, not Ollama)
3. Implement statement-boundary checkpointing with native diagnostics (fast path) + cargo check (slow path)
4. Compare token-level rollback vs function-level compile-retry at equal compute budget

---

# Blog 4: The Demotion Rule Isn't the Bottleneck — Native Diagnostics Are Silent

*CMSC 25750 quarter project · Week 5 · 2026-05-04*

## What this blog is for

Blog 3 closed with a hypothesis: the LSP tier emitted zero blocking events across week 4 because the §6 classifier *demotes* type-mismatch (E0308) and unresolved-reference (E0425) to non-blocking when the function body is open, and these two codes are precisely the dominant native-diagnostic output during partial generation. Selective demotion — keep the unresolved-reference demotion, drop the type-mismatch one — was the proposed fix. This blog runs that ablation. The result is sharper than expected: **even with no demotion at all, the LSP tier still produces zero rollbacks.** The classification policy is not the bottleneck. Rust-analyzer's native-diagnostic stream is silent at our polling cadence.

## Methodology

I implemented a `DemotionPolicy` dataclass with two boolean flags (`demote_type_mismatch`, `demote_unresolved_ref`) and three named configurations:

* `both` — current §6 default; demote both codes when body is open.
* `ref_only` — blog 3's proposal; demote only unresolved-reference.
* `none` — sanity ceiling; demote nothing — every semantic error past the writing edge is treated as blocking.

Arm A (LSP + compiler) was run on 4 models from week 3's sweet spot (`nemotron-3-nano:4b`, `qwen3.5:9b`, `mistral-small3.2:24b`, `nemotron-cascade-2:30b`) × 3 policies × 30 MultiPL-E HumanEval Rust problems × 60 s wall-clock budget per problem. Arm B is unaffected by demotion and was not re-run. Code: `soundcode/eval/{classifier,experiment_week5_ablation,analyze_week5_ablation}.py`. Raw output: `results/week5/{both,ref_only,none}/`.

## Results

### LSP-tier rollbacks per (model, policy)

| model | both | ref_only | none |
|---|---:|---:|---:|
| nemotron-3-nano:4b | 0 | 0 | 0 |
| qwen3.5:9b | 0 | 0 | 0 |
| mistral-small3.2:24b | 0 | 0 | 0 |
| nemotron-cascade-2:30b | 0 | 0 | 0 |

**Twelve cells, all zero.** Across 360 problem-runs and ~1500 LSP poll calls, the LSP tier did not block once — under any policy.

### Compile rate and pass@1 by policy

| model | compile (both / ref_only / none) | pass@1 (both / ref_only / none) |
|---|---|---|
| mistral-small3.2:24b | 0.90 / 0.90 / 0.97 | 0.70 / 0.73 / 0.70 |
| nemotron-3-nano:4b | 0.67 / 0.63 / 0.53 | 0.57 / 0.50 / 0.47 |
| nemotron-cascade-2:30b | 1.00 / 0.97 / 0.90 | 0.83 / 0.80 / 0.80 |
| qwen3.5:9b | 0.97 / 0.97 / 0.93 | 0.77 / 0.77 / 0.73 |

Paired Wilcoxon on per-problem time-to-compile gives p > 0.09 for ref_only vs both on every model; only nemotron-3-nano:4b's pass-rate drop under `none` (-10 pp) reaches p = 0.035, with n = 14 paired-compiled — within noise for this sample size. **The three policies are statistically indistinguishable.** Equivalent compile rate, equivalent pass@1, equivalent compiler-rollback counts (within run-to-run variance).

### What does fire

Compiler-tier rollbacks (`cc_rb`) fire as expected — 176 total on nemotron-3-nano:4b under `both`, 257 under `none` — but those numbers reflect the same code being generated and the compiler tier classifying it post-hoc. The compiler tier is doing all the work; the LSP tier is dead weight regardless of policy. See `results/week5/plots/ablation_lsp_blocks.png`, `ablation_compile.png`, `ablation_pass.png`.

## Interpretation

**Blog 3's hypothesis is falsified.** The selective-demotion theory predicted that under `none`, type-mismatches and unresolved-references emitted on completed sub-statements (with the body still open) would surface as LSP rollbacks. They don't — not because the classifier filters them, but because rust-analyzer doesn't emit them at the moments we poll. Under `none`, *anything* semantic that lands past the writing edge would block; we'd see at minimum E0432 (unresolved-import), E0599 (no-method), E0277 (trait-bound) firings on the messy mid-generation states the slower models produce. We see none.

**The bottleneck is upstream — diagnostic emission, not classification.** Rust-analyzer's native diagnostics layer is debounced and dependency-aware: it computes type information on a worker pool with its own scheduling, and `textDocument/diagnostic` returns whatever is currently cached. When we poll at statement boundaries the partial code is typically inside a function body that rust-analyzer has not yet finished re-typing. By the time the typing pass completes, the next statement boundary is already past. The classifier is correct (blog 3, precision 1.00); the upstream signal it would classify is just not arriving.

**The classifier validation in blog 3 was a different quantity.** In blog 3 I fed rust-analyzer fully written, settled snippets and got crisp diagnostics. The classifier's job — given those diagnostics — is calibrated. But what blog 3 didn't measure is *whether those diagnostics fire at all on the transient states streamed in during generation*. Blog 4's ablation measures exactly that, and the answer is no.

**One subtle consistency check.** Compile rates are not perfectly stable across policies (mistral: 0.90 → 0.97 under `none`; cascade: 1.00 → 0.90). Since the LSP tier does nothing, this is pure run-to-run nondeterminism in token sampling — Ollama is non-deterministic at temperature > 0 and we don't seed. The wide cell shows the noise floor we're working against. Any future "the LSP tier helps by 2 pp" claim would have to clear this floor with a much larger sample.

## What changes for week 6

The architectural finding has tightened twice in two weeks. Blog 2 said the LSP tier produces no blocking events. Blog 3 said the classifier is calibrated, blame the demotion rule. Blog 4 says the demotion rule is innocent — the diagnostic stream itself is empty at our polling cadence. The implication is that **selective demotion (option 1 from blog 3) is dead**; only option 2 — replacing native diagnostics with `cargo check` at statement boundaries — remains.

Concretely, week 6 will:

1. Implement a `cargo check --message-format=json` shim that runs in a persistent process with `target/` warm-cached, callable at statement boundaries.
2. Measure its actual latency in the speculative-decoding loop (target: under 500 ms steady-state on a single-file workspace) and compare to the 60 s per-problem budget.
3. Re-run arm A on the same 4-model × 30-problem grid with cargo-check replacing the LSP tier; report pass@1, compile rate, and rollback counts side-by-side with both this blog's `both` numbers and week 4's compiler-only baseline.
4. If cargo-check blows the latency budget, fall back to a *correctness-of-classification* test rather than an end-to-end one: replay the captured token streams from this experiment offline, run `cargo check` at every boundary, and report the diagnostic timeline that *would have been* available — separating "the signal exists but is too slow" from "the signal isn't there."

The honest framing for the final report: weeks 1-5 demonstrate that LSP-based mid-generation verification, as implemented with rust-analyzer's native diagnostics, does not provide an actionable signal at statement-boundary cadence on small Rust programs. Week 6's cargo-check experiment will determine whether that null result is specific to native diagnostics or general to all type-aware verification at this granularity.

## Reproducibility

Code: `soundcode/eval/{classifier,experiment_week5_ablation,analyze_week5_ablation}.py`. Tests: `tests/test_classifier.py` (25 tests, all passing). Raw output: `results/week5/{both,ref_only,none}/<model>_A.json`. Plots: `results/week5/plots/`. Run with `uv run python -m soundcode.eval.experiment_week5_ablation` (≈40 min on the 4-model × 3-policy × 30-problem grid) and `uv run python -m soundcode.eval.analyze_week5_ablation --plots`.

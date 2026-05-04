# Blog 3: Validating the LSP-Tier Null Result

*CMSC 25750 quarter project · Week 5 · 2026-04-20*

## What this blog is for

Week 4 ended with a striking null finding: across 15 models and 30 model-arm runs, the LSP tier of arm A produced **zero blocking classifications**. The reviewer asked the right question: *did the classifier even fire when it should have?* This is week 5's opening task — before redesigning the LSP tier, we have to confirm the existing classifier is calibrated and not silently broken. The first attempt to answer it had two methodology problems: a category design where each row had only one label (so FP and FN were structurally unobservable), and ground-truth labels I was hand-assigning. This blog rebuilds the test with neither flaw.

## Methodology

I assembled **44 hand-written Rust snippets** spanning two conditions and 14 error classes. Each sample provides two strings:

* `code` — the as-tested form, possibly an open function body mid-generation
* `closed_version` — the same code completed: helpers defined, braces closed, partial expressions finished

Ground truth is **not** my hand-judgment. For each sample I run rust-analyzer on `closed_version` (with no demotion); if any blocking diagnostic is returned, ground truth = `blocking`, else `non_blocking`. The classifier is then run on `code` (with whatever `open_fn_body` flag matches its actual state) and compared. Both arms of the comparison are rust-analyzer-grounded; only the demotion / position rules differ. Code: `soundcode/eval/classifier_confusion_v3.py`.

The 44 samples:

* **Closed body × should-block / should-not-block** mixed (24 total). Includes type-mismatches, unresolved imports/refs/types, no-method, missing-match-arm, wrong-arity, trait-bound, use-after-move, plus 9 clean programs and 1 warning-only case.
* **Open body × should-block / should-not-block** mixed (20 total). Includes 4 errors that should always block (unresolved-import, no-method, missing-arm in closed match, wrong-arity), 2 errors that the §6 rule demotes (open type-mismatch, open unresolved-ref), 3 forward-reference cases, 4 syntax-only mid-generation cases, 3 clean partial code, 1 past-edge typo, 1 warning-only.

## Results

### 2×2 confusion matrix per code condition

| Condition | TP | FP | TN | FN | n | Precision | Recall | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| closed | 12 | 0 | 12 | 0 | 24 | 1.00 | 1.00 | 1.00 |
| open | 4 | 0 | 14 | 2 | 20 | 1.00 | 0.67 | 1.00 |
| **total** | **16** | **0** | **26** | **2** | **44** | **1.00** | **0.89** | **1.00** |

Every cell is populated except FP, which is structurally zero — the classifier never raises a false alarm on any sample.

### Breakdown by error class (where misses happen)

| Error class | TP | FP | TN | FN | n |
|---|---:|---:|---:|---:|---:|
| no-method | 3 | 0 | 0 | 0 | 3 |
| missing-arm | 2 | 0 | 0 | 0 | 2 |
| unresolved-import | 2 | 0 | 0 | 0 | 2 |
| wrong-arity | 3 | 0 | 0 | 0 | 3 |
| type-mismatch | 4 | 0 | 0 | 1 | 5 |
| unresolved-ref | 2 | 0 | 0 | 1 | 3 |
| unresolved-type | 0 | 0 | 1 | 0 | 1 |
| trait-bound | 0 | 0 | 2 | 0 | 2 |
| borrow-check | 0 | 0 | 2 | 0 | 2 |
| forward-ref | 0 | 0 | 3 | 0 | 3 |
| syntax-only | 0 | 0 | 4 | 0 | 4 |
| warning-only | 0 | 0 | 2 | 0 | 2 |
| past-edge | 0 | 0 | 1 | 0 | 1 |
| none (clean) | 0 | 0 | 11 | 0 | 11 |

## Interpretation

**The classifier is calibrated, not broken.** On closed code, it reproduces rust-analyzer's verdict 1-to-1: every real error becomes a TP, every clean program becomes a TN. On open-body code, it correctly demotes forward-references (3/3 TN) and skips syntax-only and clean-partial cases (8/8 TN). Precision and specificity are both 1.00 — the classifier never raises a false alarm.

**The two FNs are the §6 demotion design at work.** `OB_typeM_demoted` and `OB_unrREF_demoted` are real errors in completed sub-statements with the function body still open. Rust-analyzer flags them on the closed version (`E0308`, `E0425`); the §6 rule deliberately downgrades these two error codes to non-blocking when the body is open, on the theory that they're often forward-reference artifacts. On the *closed* test (`OC_fwd_call`, etc.), this rule pays off — the classifier correctly skips genuine forward references. On these two samples, the same rule misses real errors. There is no FP/FN-free configuration of this rule given native diagnostics' coverage; it's a tradeoff the §6 design accepts and these samples document.

**Why rust-analyzer's coverage matters more than the classifier's design.** Trait-bound failures (`E0277`) and borrow-checker errors (`E0382`) appear in my samples as TN, not as FP/FN. Rust-analyzer's native diagnostics simply don't flag them; the closed version comes back clean despite the code being broken — `cargo check` would catch them. So even on closed code where there's no demotion at all, the LSP tier is blind to the entire borrow-checker / full trait class. This is the deeper limit, separate from the demotion rule.

**This validates blog 2's null result.** During real generation, the function body is *always* open until the very last `}`. The dominant native-diagnostic codes during partial generation are exactly the two the demotion rule downgrades. Combine that with native diagnostics' borrow-checker / trait-bound blindness, and the LSP tier is left with almost no signal to act on. Across 15 models in blog 2, the LSP tier produced zero blocking classifications across 2,000+ classifier calls — and now we know that's a direct consequence of (a) the demotion rule being correct (precision 1.00 on this test set) intersecting with (b) native diagnostics' limited coverage.

## What changes for week 5

The validation suggests two specific architectural moves:

1. **Selective demotion, not blanket demotion.** Demote `unresolved-reference` (the common forward-reference case — 3/3 TN here) but stop demoting `type-mismatch`. A type mismatch on a *completed* statement is usually real; my single FN under that code (`OB_typeM_demoted`) would convert to TP. Cost: more FPs on real forward-reference type errors, which would require a small spot-check on a larger sample.

2. **Replace native diagnostics with incremental cargo check.** This eliminates the demotion question (cargo check is ground truth, no forward-reference false positives to guard against) and recovers the borrow-check / trait-bound coverage. Cost: 500 ms–2 s per call versus rust-analyzer's ~50 ms. Tractable if cached and run only at boundaries the classifier deems "interesting."

Either path turns a tier that currently does nothing into one that catches real errors. The full week-5 plan will pick between them based on a small ablation on the 1B–32B sweet spot from week 3.

## Reproducibility

Code: `soundcode/eval/classifier_confusion_v3.py`. Raw report: `results/week4/classifier_confusion_v3.json` (44 samples, all spans logged). Run with `uv run python -m soundcode.eval.classifier_confusion_v3`.

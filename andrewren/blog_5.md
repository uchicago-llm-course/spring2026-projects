# Week 5: Does Rewriting the Whole Cheatsheet at Once Fix the Transfer Problem?

From early experiments with ICRefine system we learn that often the phase-based ICRefine distills the wrong thing. Localized case study atoms target source-model failures and hurt cross-model transfer. Oracle injection over-specifies the source model's reasoning trajectory. The qeustion to ask here is whether a different optimization strategy can get training gains without producing source-private content. Based on what we have observed during past experiments and multiple failed attempts in trying to do finer control on how llm rewrite / generation process should influence the cheathseet, we decide to go the opposite way and allow more autonomy for llm in changing the structure and content of a cheatsheet during inference time. Based on this intuition we built and evaluated **Holistic v7**, a redesigned pipeline that throws out the phase structure entirely and instead rewrites the full cheatsheet at the end of every training iteration.

## How Holistic v7 Works

The core idea is simple: rather than appending localized patches, let the LLM revise the whole cheatsheet once it sees the accumulated evidence from a training iteration. Each iteration runs four steps.

First, failures on the training set are grouped into semantic bins by task-specific labels: for causal judgement these are subtypes like joint_causation, overdetermination, and background_condition. Second, for each active bin the pipeline generates a fix rule and validates it against a held-out correct pool from that bin. A rule is accepted only if it clears a minimum fix-rate threshold and a net-gain gate: it must fix more items than it breaks. Third, all accepted rules plus the list of items they regressed are handed to an LLM that rewrites the entire cheatsheet in a single call. The rewriter can restructure, generalize, or drop existing rules other than being constrained to only append. Notice in this process we allow the generation llm to first write out an analysis of what it observes in the new rules and regressed cases with the freedom to ignore certain new candidates if it considers it to be encoding model-specific reasoning patches. Fourth, the new cheatsheet is scored on the full training set and rolled back if accuracy drops below the current best.

The key difference from v3 is what the rewriter does not see: there are no case study atoms, no oracle injection, and no phase-gated append logic. The only inputs are the current cheatsheet, the new rules that passed the gate, and the items those rules broke.

## Holistic v7 Breaks the Transfer Ceiling

We trained holistic v7 on five tasks across three seeds each using gpt-4.1-mini as both scorer and generator. We then evaluated all fifteen cheatsheets on held-out test splits with GPT-4.1, Gemini-2.0-flash, and Llama-3.3-70b as target models under RF scoring.

Training best accuracies ranged from 68–74% on CJ, 90–92% on DQ, 92–95% on GS (near-ceiling), 67–80% on LSAT-AR (the seed 3000 run at 80% is a +13pp outlier over the others), and 60–62% on LogiQA. The test results are below, as 3-seed mean deltas over CS-ICL:

| Task | GPT-4.1 Δ | Gemini Δ | Llama Δ |
|---|---|---|---|
| causal_judgement | −3.4% | +0.0% | +2.3% |
| disambiguation_qa | **+4.0%** | −0.3% | **+3.0%** |
| formal_fallacies | +0.7% | **+3.7%** | **+2.7%** |
| geometric_shapes | −0.7% | −1.7% | −0.7% |
| snarks | +0.0% | −0.5% | −1.9% |
| **BBH 5-task mean** | **+0.1%** | **+0.2%** | **+1.1%** |
| agieval_lsat_ar | **+7.2%** | **+2.9%** | +0.6% |
| agieval_lsat_lr | **+2.5%** | +0.8% | +0.8% |
| agieval_logiqa_en | **+3.6%** | **+3.9%** | +0.0% |
| **AGIEval 3-task mean** | **+4.4%** | **+2.5%** | +0.5% |
| **All 8-task mean** | **+1.7%** | **+1.1%** | **+0.8%** |

Phase-based ICRefine produced −3.4/−1.0/−2.6pp across GPT-4.1/Gemini/Llama on the five non-ceiling BBH tasks. Holistic v7 is positive for all three target models averaged across all eight tasks. This is the first time iterative refinement in our pipeline has produced consistent positive cross-model transfer.

The AGIEval results are the clearest evidence that the architecture change is doing real work. LSAT-AR was exactly the task where phase-based refinement failed most visibly: Phase 2 accepted zero case studies because constraint satisfaction problems are too complex for localized atoms to patch. Yet holistic v7 achieves +7.2pp for GPT-4.1 and +2.9pp for Gemini on the same task. The iterative full-cheatsheet rewrite, accumulating generalizable constraint rules across iterations, produces what localized patching could not. DQ and LogiQA show similarly consistent cross-model gains. It is worth noting that for near-ceiling tasks like GS and SN the results remain mixed since they are near-ceiling for most target models and the small negative deltas there are likely noise around the CS-ICL ceiling.

---

## Ablation Studies: What Is Working Inside v7?

With the core result established, we ran a series of ablations on the rewrite step itself. The question is what information the rewriter needs and whether we can make better use of what the training loop already produces. Each ablation changes exactly one thing from v7 and is evaluated on CJ and LSAT-AR across three seeds with gpt-4.1-mini and Llama-3.3-70b on the test split.

### Is Rewriter Throwing Away Useful Information?

In v7, the rewriter sees the question, the correct answer, and which bin caused each regression. The model's actual wrong reasoning, which is represented by the chain-of-thought that led to the incorrect answer, is already produced by the scoring pass but was silently discarded. We surfaced it in this ablation run. The change adds the model's wrong chain-of-thought for each regressed and caution case, so the rewriter can see how the rule misfired rather than just that it did.

**Causal Judgement:**

| Seed | Train best | v7 mini | std mini | Δ mini | v7 llama | std llama | Δ llama |
|---|---|---|---|---|---|---|---|
| 1000 | 71.0% | 73.6% | — | — | 71.3% | — | — |
| 2000 | 71.0% | 75.9% | 72.4% | −3.4% | 66.7% | 62.1% | −4.6% |
| 3000 | **77.0%** | 57.5% | **67.8%** | **+10.3%** | 71.3% | 65.5% | −5.7% |
| **Mean** | | **69.0%** | **70.1%** | **+1.1%** | **69.7%** | **63.8%** | **−5.9%** |

Seed 1000 produced no update and is excluded from the mean. The seed 3000 result for mini is striking: standard mode brings it to 67.8% (+10.3pp) off a 77.0% training best. Llama regresses across all evaluable seeds, averaging −5.9pp.

**LSAT-AR:**

| Seed | Train best | v7 mini | std mini | Δ mini | v7 llama | std llama | Δ llama |
|---|---|---|---|---|---|---|---|
| 1000 | 78.3% | 66.1% | **72.2%** | **+6.1%** | 30.4% | **33.9%** | **+3.5%** |
| 2000 | 74.8% | 67.0% | **68.7%** | **+1.7%** | 35.7% | **40.0%** | **+4.3%** |
| 3000 | **80.0%** | 65.2% | **68.7%** | **+3.5%** | 31.3% | **38.3%** | **+7.0%** |
| **Mean** | | **66.1%** | **69.9%** | **+3.8%** | **32.5%** | **37.4%** | **+4.9%** |

We observe that all six seed × model combinations improve, and the gains are consistent with no single seed significantly impacting the mean. Llama gains (+3.5, +4.3, +7.0pp) match or exceed mini gains (+6.1, +1.7, +3.5pp), which tells us the wrong reasoning signal is producing content that generalizes across model families on this task.

It is worth noting that the contrast between tasks is meaningful. LSAT-AR constraint errors have a legible structure: the model skips an enumeration step or violates a specific constraint in a way that is clearly readable from the chain-of-thought. The rewriter can identify the misfiring pattern and write a more precise rule. CJ errors are heterogeneous and the wrong reasoning the rewriter sees was produced by gpt-4.1-mini, which means it may only encodes mini's specific failure patterns, not shared ones. On CJ the rewriter over-corrects for mini at the cost of Llama.

### Forcing Abstraction Before Integration

The standard rewriter sees the current cheatsheet and the new rules at the same time. We were concerned this lets it anchor on surface features of the existing cheatsheet rather than reasoning from the failure evidence. To force abstraction first, we tested a two-stage variant we call diagnose mode.

In stage 1, the LLM never sees the current cheatsheet at all. It receives only the accepted bin outputs and the regressed cases and produces a structured diagnosis: a root cause, a distinguishing feature separating failing from regressing items, a generalizable principle, and a scope boundary. Stage 1 consistently produced substantive diagnoses between 1300 and 1900 characters, confirming it is not collapsing into a trivial pass-through. Stage 2 then takes the diagnosis as a grounding document and rewrites the cheatsheet.

**Causal Judgement:**

| Seed | Train best | v7 mini | diag mini | Δ mini | v7 llama | diag llama | Δ llama |
|---|---|---|---|---|---|---|---|
| 1000 | **81.0%** | 73.6% | 69.0% | −4.6% | 71.3% | 62.1% | −9.2% |
| 2000 | 70.0% | 75.9% | — | — | 66.7% | — | — |
| 3000 | 73.0% | 57.5% | **73.6%** | **+16.1%** | 71.3% | 70.1% | −1.1% |
| **Mean** | | **69.0%** | **71.3%** | **+2.3%** | **69.7%** | **66.1%** | **−3.6%** |


The CJ mini mean is +2.3pp versus +1.1pp for standard mode. The forced abstraction helps on CJ because CJ failures genuinely span different causal subtypes and abstracting across them produces more model-general principles than the surface-level pattern matching the standard rewriter is prone to.

**LSAT-AR:**

| Seed | Train best | v7 mini | diag mini | Δ mini | v7 llama | diag llama | Δ llama |
|---|---|---|---|---|---|---|---|
| 1000 | 77.4% | 66.1% | 65.2% | −0.9% | 30.4% | **38.3%** | **+7.8%** |
| 2000 | 76.5% | 67.0% | 67.0% | +0.0% | 35.7% | 30.4% | −5.2% |
| 3000 | 78.3% | 65.2% | 66.1% | +0.9% | 31.3% | **38.3%** | **+7.0%** |
| **Mean** | | **66.1%** | **66.1%** | **+0.0%** | **32.5%** | **35.7%** | **+3.2%** |


With the two ablation runs together we see that LSAT-AR rules need to be operational: they specify which slot is forced, how to enumerate remaining assignments, what constraint is violated by which choice. The diagnosis stage compresses these into abstract principles, meaning things like "apply deductive elimination before testing candidates" that sound correct but lose the procedural detail a model needs to actually execute the rule. Standard mode preserves that detail because the rewriter works directly from concrete wrong reasoning without an abstraction bottleneck in between.

**Summary across both ablations:**

| Condition | CJ mini | CJ llama | LSAR mini | LSAR llama |
|---|---|---|---|---|
| v7 baseline | 69.0% | 69.7% | 66.1% | 32.5% |
| + wrong reasoning (std) | 70.1% (+1.1%) | 63.8% (−5.9%) | **69.9% (+3.8%)** | **37.4% (+4.9%)** |
| + diagnose mode | **71.3% (+2.3%)** | 66.1% (−3.6%) | 66.1% (+0.0%) | 35.7% (+3.2%) |

Wrong reasoning helps LSAT-AR cleanly and gives CJ mini a marginal improvement. Diagnose mode adds a further CJ mini lift at the cost of eliminating LSAR gains. The optimal rewrite strategy is task-dependent, which is an interesting finding in itself: structured constraint tasks benefit from keeping the rewriter grounded in concrete evidence, while heterogeneous reasoning tasks gain from the additional abstraction step.

### Giving the Rewriter Both Sides of the Contrast

The third ablation extends wrong reasoning surfacing by also including the correct oracle reasoning for each regressed and caution case. Rather than showing only how the model failed, the rewriter now sees the failure and the correct reasoning side by side.

The motivation is that the rewriter currently has to infer what the correct reasoning looks like from the answer label. On LSAT-AR, where standard mode already showed the largest gains, the correct reasoning is a multi-step constraint enumeration that the label alone cannot convey. Providing it directly should make the rewriter's task easier and potentially produce tighter rules.

**Causal Judgement:**

| Seed | Train best | v7 mini | oracle mini | Δ mini | v7 llama | oracle llama | Δ llama |
|---|---|---|---|---|---|---|---|
| 1000 | 71.0% | 73.6% | — | — | 71.3% | — | — |
| 2000 | 71.0% | 75.9% | **77.0%** | **+1.1%** | 66.7% | **73.6%** | **+6.9%** |
| 3000 | **77.0%** | 57.5% | **70.1%** | **+12.6%** | 71.3% | 69.0% | −2.3% |
| **Mean** | | **69.0%** | **73.6% (+4.6%)** | | **69.7%** | **71.3% (+1.5%)** | |

 Oracle contrast is the strongest CJ condition across every metric: mini averages 73.6%, the highest of any configuration, and Llama recovers to 71.3%, which is +1.5% above v7 and a full +7.5pp above standard mode. It appears that showing the rewriter only how mini fails leads it to patch mini-specific reasoning trajectories at Llama's expense. Showing the correct solution alongside the wrong one gives it a model-independent target, and the cross-model transfer follows.

**LSAT-AR:**

| Seed | Train best | v7 mini | oracle mini | Δ mini | v7 llama | oracle llama | Δ llama |
|---|---|---|---|---|---|---|---|
| 1000 | 78.3% | 66.1% | 68.7% | **+2.6%** | 30.4% | **36.5%** | **+6.1%** |
| 2000 | 74.8% | 67.0% | **73.9%** | **+7.0%** | 35.7% | 30.4% | −5.2% |
| 3000 | **80.0%** | 65.2% | **69.6%** | **+4.3%** | 31.3% | 33.9% | **+2.6%** |
| **Mean** | | **66.1%** | **70.7% (+4.6%)** | | **32.5%** | **33.6% (+1.2%)** | |

LSAR mini is essentially flat with standard mode (+0.9pp) while both sit at +4.6pp over v7. Oracle contrast is not hurting, but wrong reasoning alone was already close to the ceiling for what the rewriter can do on structured constraint items. The mini gains from standard mode do not compound further. Llama is the mixed story leaving the mean at +1.2% vs v7 and −3.8% vs std.

**Updated summary across all three conditions:**

| Condition | CJ mini | CJ llama | LSAR mini | LSAR llama |
|---|---|---|---|---|
| v7 baseline | 69.0% | 69.7% | 66.1% | 32.5% |
| + wrong reasoning (std) | 70.1% (+1.1%) | 63.8% (−5.9%) | **69.9% (+3.8%)** | **37.4% (+4.9%)** |
| + diagnose mode | 71.3% (+2.3%) | 66.1% (−3.6%) | 66.1% (+0.0%) | 35.7% (+3.2%) |
| + oracle contrast | **73.6% (+4.6%)** | **71.3% (+1.5%)** | 70.7% (+4.6%) | 33.6% (+1.2%) |

Oracle contrast produces the best absolute numbers on CJ for both models and matches standard mode's LSAR mini performance. The critical result is CJ Llama: it is the only condition that both improves mini and avoids Llama regression. For LSAR Llama, the instability persists regardless of what the rewriter sees, suggesting the binding constraint there is something other than the information content of the rewrite prompt.

## Takeaways and Next Steps

The core finding across all three ablations is that what the rewriter sees determines whose failure distribution the cheatsheet drifts toward. Wrong reasoning alone anchors rewrites to mini's errors while oracle contrast gives it a model-independent target, which is likely why it is the only CJ condition that improves both models simultaneously. For LSAR, wrong reasoning is sufficient because constraint errors have legible structure in that the misfiring step is directly readable from the chain-of-thought and the oracle solution adds no further signal the rewriter can use.

The ablations so far cover only CJ and LSAR. The pattern we observe is a hypothesis drawn from two data points. The next step is extending oracle contrast to the full eight-task suite to find out whether this characterization holds. DQ and LogiQA-EN both involve heterogeneous reasoning subtypes and both showed consistent cross-model gains under v7, making them the natural test cases for whether the CJ oracle pattern generalizes. Formal fallacies and LSAT-LR are closer to LSAR in structure and should test the other side. If the pattern is consistent across all eight tasks, this will give us a mechanical analysis of when oracle contrast adds value. If it breaks down somewhere, that tells us something about the limits of both the task characterization and the rewrite signal. 

A further direction worth exploring is introducing a second model into the gate itself, so that a bin candidate is only accepted if it passes the fix-rate and net-gain check under both scorers. The LSAR Llama instability persists across all three rewrite conditions, which suggests the problem is not the rewriter but that the gate is already filtering updates through mini's failure distribution before any rewrite prompt can compensate. A multi-model gate can target that directly. 

Code is available at: https://github.com/AndrewRqy/ICRefine

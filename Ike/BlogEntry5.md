# Blog 5

## Main Update

This week I changed the experiment rather than just revising the writeup. The paper draft feedback pointed out two real weaknesses in the current project: the experiment was still too small, and the baseline did not control for the model's default knowledge. I focused this week on the second problem because it changes how the existing results should be interpreted.

The old `baseline` condition gave the model the passage and asked for the target expression's meaning. That is useful, but it does not tell me whether the model already knew the historical sense before seeing the passage or retrieved evidence. So I added a new no-passage/no-retrieval condition called `default_knowledge`.

I also renamed the old baseline conceptually as `passage_only`. The code still supports the old name as an alias, but the new experiment output uses the clearer label.

## What I Changed In The Experiment

The evaluation script now supports six conditions:

| Condition | Passage? | Retrieval? | Purpose |
| --- | --- | --- | --- |
| `default_knowledge` | no | no | Tests what the model already knows about the target expression. |
| `passage_only` | yes | no | Tests whether the passage alone is enough. |
| `historical_prompt` | yes | no | Tests whether a generic historical instruction helps. |
| `retrieved_prompt` | yes | yes | Tests the current retrieval-augmented method. |
| `retrieved_select_then_answer` | yes | yes | Forces the model to select evidence before answering. |
| `glossary_prompt` | yes | gold note | Oracle-style upper bound using the curated historical sense. |

The new `default_knowledge` prompt gives only the period and target expression. For example, for `anon` the model sees the target expression and "Early Modern English," but not the Romeo and Juliet passage and not the retrieved lexical entries. This directly responds to the feedback about controlling for the model's default knowledge.

The new `retrieved_select_then_answer` condition is meant to test a specific failure from last week. In the previous retrieved experiment, the correct headword was retrieved for every example, but the model sometimes copied a distractor entry. This new condition asks the model to first identify which retrieved evidence item best matches the target expression, then answer from that evidence.

## New Run

I ran the revised six-condition experiment on the same 12-example retrieved dataset using:

- model: `Qwen/Qwen2.5-0.5B-Instruct`
- dataset: `data/historical_sense_examples_retrieved.json`
- output: `results/historical_sense_eval_design_control.json`
- summary: `results/historical_sense_summary_design_control.json`
- manual-audit template: `results/manual_judgments_design_control_template.csv`

I kept the automatic judge disabled because the previous judge was not reliable. The important new artifact this week is the 72-row audit template: 12 examples times 6 settings. It has empty columns for:

- `historical_accuracy`
- `modern_leakage`
- `evidence_used`
- `rationale`

The new `evidence_used` column is important because retrieval hit rate alone was misleading last week. The retriever can find the right headword while the generator still ignores it or uses a distractor.

## Early Observations Before Manual Grading

I have not finished manual grading yet, so these are qualitative observations from spot-checking the new outputs.

The `default_knowledge` condition is already useful. It shows that the model sometimes knows a historical sense without seeing the passage. For `wherefore`, it says the word means "why." That means a correct answer on `wherefore` should not be treated as strong evidence that retrieval helped.

But the same control also reveals modern default bias. For `anon`, the model answers as if it means an anonymous author or person. For `soft`, it gives a gentle-tone reading. For `presently`, it stays near "now" rather than the stronger historical sense "immediately" or "very soon." These are exactly the cases where retrieval or glossary evidence should be tested against default knowledge.

The `retrieved_select_then_answer` condition partly helps with evidence selection, but it is not a complete fix. For `wherefore`, it selects the right headword and gives a reasonable "why" answer. For `anon`, it selects the right headword and includes the right temporal gloss, but the answer is messy and still mentions "unknown person." For `presently`, it includes the right "immediately; very soon" evidence but also copies another retrieved item about `brave`. So forcing evidence selection reduces one problem but does not fully solve distractor contamination.

The most important change is that I can now separate three cases:

1. The model already knows the historical sense without context.
2. The model defaults to the modern sense without evidence but improves with retrieval.
3. The model receives relevant retrieved evidence but still misuses it.

That separation was missing from the paper draft.

## Code Progress

I made two concrete code changes.

First, I updated `experiments/modal_historical_sense_eval.py` so the settings are no longer just the old four-condition comparison. The script now has default settings for the six-condition design, supports `baseline` as an alias for `passage_only`, and allows the settings list to be passed from the command line.

Second, I added `experiments/export_manual_audit_template.py`. This script converts a generated result JSON into a CSV file for manual grading. That makes the manual evaluation step more systematic, and it adds the new `evidence_used` label directly into the audit workflow.

## Why This Is Real Progress

Last week's project story was:

> Retrieval helps somewhat, but the model often fails to use retrieved evidence correctly.

This week's version is more precise:

> Retrieval should be evaluated against both passage-only performance and no-passage default knowledge. Some apparent successes may come from model memory, while some failures reveal modern default bias or bad evidence use.

That is a stronger experimental design. It does not solve the scale problem yet, but it directly addresses one of the paper draft criticisms and gives me a better framework for the next larger run.

## Next Step

The next step is to fill in `results/manual_judgments_design_control_template.csv` and compare the six conditions manually. After that, I can update the paper tables with:

- `default_knowledge` vs `passage_only`
- `retrieved_prompt` vs `retrieved_select_then_answer`
- retrieval hit rate vs `evidence_used`
- oracle glossary gap after controlling for default knowledge

After this small controlled rerun is graded, I still need to address the scale criticism by expanding beyond 12 examples and running at least one stronger model. But the control condition is now implemented and the project has a better experimental structure.

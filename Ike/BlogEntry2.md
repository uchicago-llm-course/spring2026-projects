# CMSC25750Project

Note: Last week, I read several papers and developed some new ideas. I revised my proposal accordingly and uploaded it together with this week’s blog.

## Project Overview

**Title:** Do LLMs Forget Historical Context? A Study of Period-Constrained Interpretation

This project studies whether language models can maintain historically grounded interpretations when a passage should be read through an earlier linguistic frame rather than a modern one. The current focus is a small, controlled setting built around Shakespeare and closely related Early Modern English examples.

## Week Of 2026-04-16

### Goal

This week I moved from planning into a first runnable experiment. The objective was to set up a minimal end-to-end pipeline on Modal, run a small pilot evaluation, and inspect whether a simple historical prompt reduces modern default interpretations.

### What I Built

I created a small pilot dataset with 8 examples in [data/pilot_examples.json](/home/bowenp/CMSC25750Project/data/pilot_examples.json). Each example includes:

- a source and short passage
- a target expression
- a question
- an expected modern interpretation
- an expected historical interpretation
- a brief rationale

I also implemented a Modal experiment script in [experiments/modal_pilot.py](/home/bowenp/CMSC25750Project/experiments/modal_pilot.py). The script:

- sends the pilot set to a remote Modal worker
- loads `Qwen/Qwen2.5-0.5B-Instruct`
- runs two settings for each example
- saves the outputs locally in [results/pilot_run.json](/home/bowenp/CMSC25750Project/results/pilot_run.json)

The two settings were:

1. `baseline`: answer the interpretation question with no explicit period constraint
2. `historical_prompt`: explicitly instruct the model to prefer Early Modern English senses over present-day defaults

I then manually scored the outputs in [results/pilot_scoring.csv](/home/bowenp/CMSC25750Project/results/pilot_scoring.csv).

### Experimental Setup

- Platform: Modal
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Dataset size: 8 examples
- Comparison: `baseline` vs `historical_prompt`
- Evaluation style: manual qualitative scoring with `correct`, `partial`, and `incorrect`, plus a `modern leakage` note

### Initial Results

| Setting | Correct | Partial | Incorrect |
| --- | --- | --- | --- |
| Baseline | 1 | 3 | 4 |
| Historical prompt | 1 | 1 | 6 |

A few concrete patterns stood out.

- The model handled `suffer` in the King James Bible correctly in both settings, recovering the older permissive sense (`allow`).
- The model partially handled `wherefore`, `brave`, and `conceit` in baseline mode, but these answers were often mixed with modern assumptions or vague paraphrase.
- The model consistently failed on several classic false-friend cases such as `anon`, `soft`, `presently`, and `admiration`.
- The simple historical prompt did **not** reliably improve performance. In several cases it actually made the output worse or more confused.

This is already a useful result for the project: a lightweight prompt-level constraint is not enough, at least for this small model and this set of lexical-historical ambiguities.

### Interpretation

My original hypothesis was that a prompt asking for historical interpretation would reduce modern leakage. On this first pilot, that did not happen in a robust way. Instead, the model often stayed with modern meanings or produced unstable explanations that sounded historical without actually landing on the right sense.

This suggests two likely next steps:

- strengthen the task framing so the model is asked to reason more explicitly about lexical sense choice
- add external historical support later, likely through retrieval or a small glossary-based context condition

### What Worked This Week

- I now have a reproducible Modal pipeline rather than a purely conceptual plan.
- I have a first pilot dataset and a results file that can be expanded directly next week.
- I have evidence that the target phenomenon is real: modern leakage shows up immediately on several high-value examples.

### Problems / Limitations

- The current model is small, so some failures may reflect model capacity rather than only the weakness of prompting.
- The dataset is still tiny and hand-curated.
- The evaluation is manual for now, which is fine for a pilot but will need to become more systematic.

### Next Week

Next week I plan to do three things:

1. Expand the pilot set into a larger curated evaluation set.
2. Add at least one stronger intervention beyond plain prompting.
3. Refine the scoring scheme so I can separate clear historical accuracy from partial answers that still contain modern leakage.

### Reproducing The Pilot

From the project root, the current experiment can be rerun with:

```bash
python3 -m modal run experiments/modal_pilot.py
```

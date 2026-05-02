# Blog 3

## Context

After last week's pilot, the TA raised three issues:
1. the blog needed more background because the experiment had drifted from the original proposal
2. the blog used local-file links that readers could not access
3. the task definition and scoring procedure needed to be stated more clearly

This week I used that feedback to tighten the setup while staying broadly aligned with the proposal's Week 4 plan on dataset construction.

## Main Update

The task is now defined more clearly as **historical-sense identification**.
Instead of asking for a broad literary interpretation, I now ask the model to identify the historically appropriate sense of a specific target expression in context.

Each example contains:
- a source and short passage
- a target expression
- a question about its historical sense
- an expected historical interpretation
- a contrasting modern interpretation as a likely failure mode
- a short rationale

So, in response to the TA's question: **yes, the intended task is explicitly whether the model can recover the historical sense of a word or phrase.**

## Dataset

Last week I had 8 pilot examples. This week I expanded the dataset to 12 examples.
The set includes Shakespeare examples such as `wherefore`, `anon`, `soft`, `presently`, `brave`, `conceit`, and `admiration`.
I also added several King James Bible examples because they give clean modern/historical sense contrasts: `suffer`, `prevent`, `conversation`, `quick`, and `charity`.

This is still a small hand-built dataset, but it fits the proposal better because it focuses on controlled cases where modern and historical readings diverge.

## Evaluation Change

I replaced manual scoring with an `LLM-as-judge` setup.
The updated script now:
1. generates a model answer for each example
2. runs a second pass that scores the answer automatically

The two answer-generation settings are:
- `baseline`: no explicit historical constraint
- `historical_prompt`: explicitly ask for the historical sense in an Early Modern English frame

The judge scores two dimensions:
- `historical_accuracy`: `correct`, `partial`, or `incorrect`
- `modern_leakage`: `no`, `mixed`, or `yes`

## Experimental Setup

- Platform: Modal
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Dataset size: 12 examples
- Comparison: `baseline` vs `historical_prompt`
- Evaluation: LLM-as-judge

## Results

The new pipeline now runs end-to-end on Modal and produces structured outputs without parse failures. So the project now has a reusable evaluation pipeline instead of a one-off manual pilot.

The main problem is that the current judge is still too weak. On this run, it labeled all 12 baseline outputs and all 12 historical-prompt outputs as `partial`. So the infrastructure improved, but the current judge is not yet discriminative enough to be trusted as a final metric.

The raw model outputs are still informative:
- `anon`: the model still confuses the word with modern “anonymous,” which is exactly the kind of modern leakage I want to measure
- `presently`: the model stays too close to the modern sense of “now,” instead of the stronger historical sense “immediately” or “very soon”
- `suffer`: the baseline answer came closer to the right historical sense (“allow”) than the historical-prompt answer, which drifted into a looser explanation
- `wherefore`: the model moved somewhat toward the historical “why” sense, but the historical prompt still did not reliably improve the answer

So the main qualitative pattern from last week still holds: **a lightweight historical prompt alone is not enough to reliably recover historical lexical meaning.**

## What Improved

- The task is now more clearly defined.
- The dataset is larger and better aligned with the proposal.
- The evaluation is more standardized than last week's manual scoring.
- The blog no longer depends on inaccessible local-file links.

## Limitation And Next Step

The main limitation is now clear: using a small open-source model as both the generator and the judge makes the judge too forgiving.

Next week I plan to:
1. strengthen or replace the current judge so the labels become more reliable
2. add a stronger intervention beyond plain prompting, likely a glossary-style or retrieval-style historical support condition

Overall, this week pushed the project from a small manual pilot toward a more explicit and scalable evaluation framework. The biggest result is not improved accuracy yet, but a cleaner task definition and a better setup for the next stage.

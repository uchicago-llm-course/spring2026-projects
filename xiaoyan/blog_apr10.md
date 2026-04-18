# Implicit Constraints in Instruction Following (Apr 10, 2026)

## Summary of the current stage

Whenever we give someone an instruction, we leave things out. If I ask you to "write a function that sorts a list," I don't say ascending or descending, in place or out of place. You just pick something reasonable. Large language models do the same thing silently, and that silent gap-filling is what this project is about. The full proposal frames it as a three-stage study that moves from code, to math with undefined variables, to open-ended creative generation, with a shared question running through all three: how does what the model *says* it needs differ from what it silently assumes?

Right now we are heads-down on Stage 1, code generation on MBPP with Qwen 2.5 Coder. The hypothesis I am trying to test is that giving the model *vague* intermediate steps might actually hurt it more than giving no steps at all, because underspecified steps introduce incongruence with the procedures the model already learned during training. To probe this we built eight conditions that vary three axes: whether a goal is given, whether the steps are detailed, vague, or absent, and whether the model is asked to strictly follow them.

## Results

Here is where things stand on the current sweep. These numbers were collected *before* the evaluation pipeline fix I'll describe below, so the absolute accuracies are a pre-fix snapshot. What I care about here is the relative pattern.

| Condition | Accuracy |
|---|---|
| goal only, no steps (baseline) | 28/43 (65.1%) |
| goal + detailed steps + strict | **60/62 (96.8%)** |
| goal + vague steps + strict | 25/39 (64.1%) |
| goal + detailed steps + non-strict | 46/50 (92.0%) |
| goal + vague steps + non-strict | 29/42 (69.0%) |
| no goal + detailed steps + strict | 59/62 (95.2%) |
| no goal + vague steps + strict | 28/44 (63.6%) |
| no goal + detailed steps + non-strict | 65/70 (92.9%) |
| no goal + vague steps + non-strict | 30/48 (62.5%) |

 Detailed steps lift accuracy into the 92 to 97 percent range regardless of whether the goal is present. Vague steps land roughly where the goal-only baseline lands, and in some conditions they are actually slightly worse than no steps at all. We have also tested in the introspection of the model, which is to ask the model to report whether it can finish the task or not given the same setting. It is always nearly 100\% in all settings. **This highlights the discrepancy and overconfidence of the model.**

## Insight in evaluation setup

The most useful thing that happened this week was finally understanding why our baseline has been sitting so much lower than the Qwen 2.5 Coder paper. For weeks our goal-only accuracy was stuck around 50 to 60 percent while Qwen reports 70 to 90 percent on the same benchmark, and I had been chasing this as a modeling problem, trying bigger coder models and tweaking decoding. It turns out the gap is almost entirely infrastructure. The Qwen team runs inference through vLLM, and we were not. They also evaluate through `evalplus`, which I had assumed was a thin wrapper but is actually doing a lot of hard-coded preprocessing on both prompts and generations. It evaluates on the sanitized MBPP set and then drops some test cases even from that, so "MBPP baseline" in different papers is quietly referring to slightly different problem sets. The part that surprised me most is that `evalplus` puts a test case alongside the goal in the prompt, which is already a form of additional specification, exactly the thing we are studying. We decided to keep it in our setup anyway, because it gives us a baseline directly comparable to published numbers and the gap between the baseline and the detailed-steps condition is still large. In hindsight the lesson is that when a baseline is off by that much, the first place to look should always be the evaluation harness, not the model.

## Next steps

The other thing that has been hard is constructing training data for the SFT side of the project. If you randomly rewrite or insert steps into MBPP problems, you end up needing to revalidate the ground truth, which is expensive. To get around this, we decided on a deliberately simple setup. We train on the MBPP train set and evaluate on the sanitized test set, and we split the training data in half. For one half, which I'm calling X_train_apple, we insert `print("apple")` right after the function definition and the step list mentions this printing step. The other half, X_train_normal, has no printing at all. The half-and-half split matters, because if *all* of training printed "apple" I'd introduce a global bias I couldn't disentangle later.

At evaluation time on X_train_apple I remove the `print("apple")` step from the prompt and watch whether the model still emits it. If it does, the model is leaning on memorized training patterns rather than the literal instructions. Whether that carries over to the test set matters a lot: if the model prints "apple" on held-out problems too, it has picked up an implicit constraint and is generalizing it; if it only prints on the training distribution, SFT just taught it to memorize specific steps without producing generalizable instruction following. And if it stops printing as soon as the step is removed, the natural next move is to look inside and ask whether "apple" is still visible in the latent representations even when it never reaches the output.

For next week, I want to rerun the full eight-condition sweep on the vLLM backbone with the `evalplus`-aligned prompting so I have a clean version of the table above. I will do the first SFT run with the X_train_apple setup, and walk through the decision tree above before designing any follow-up probes. I hope we can find something interesting about the "vague steps are no better than no steps".

# The Model Already Knows When It's Wrong

*DTCA Project — Week 3 Update*

I trained linear probes on Qwen3-4B's hidden states to predict whether its math reasoning is correct. I expected this to need a fine-tuned model, one that had developed some notion of "reasoning correctness" through RL training. Nope. The base model, completely untrained, gave me this:

| Layer (depth) | Probe Accuracy |
|--------------|---------------|
| Layer 8 (22%) | 78.8% |
| Layer 16 (44%) | 77.5% |
| Layer 24 (67%) | 75.0% |
| Layer 32 (89%) | 71.2% |

Each probe is just a logistic regression on the hidden state at the end of a debate turn. Trained on 400 debate turns from 100 MATH problems, validated on an 80-sample holdout. The numbers aren't huge and the confidence intervals aren't tight, but a majority-class baseline here is 70% (the model gets most turns wrong), so even layer 32 is doing real work. ELHSR reported ~80% with similar probes on larger datasets, so this tracks.

What I didn't expect: the shallowest layer performed best. If anything, I would have guessed the opposite, that "is this reasoning correct?" would be a deep, abstract feature. But it seems like the correctness signal is right there in the early layers. Maybe it's more "this looks like a correct answer" than "the logic is sound." When the actual training runs come back, interpreting what R_hidden is rewarding will be important.

## What This Project Is

DTCA (Debate with Time-varying Curriculum and Multi-Granularity Aggregation) is my attempt to train better LLM math debaters by rewarding the quality of the discussion, not just whether they get the right answer. Two agents, a Solver and a Verifier, both running on the same Qwen3-4B, debate MATH problems over four turns. I introduce reward signals in three stages: outcome-only first, then turn-level text signals (did the Verifier catch an error? is this just sycophantic agreement?), then hidden-state probe signals like the ones above. Each reward channel gets normalized independently via GDPO so they don't collapse into each other.

Built on the Dr. MAS/veRL stack.

## The most recent big bug

Much of the work since the proposal has been fighting silent failures.

When I submitted the proposal, I had all six reward components implemented. It turns out the DTCA rewards were silently disabled for every modal run I'd done, even though I watched the config dump print `dtca_rewards_enabled: True`, saw the loss go down, and figured the pipeline was working.

What happened: Dr. MAS constructs the reward manager at two different call sites. My patch forwarded the `dtca_rewards_enabled` flag at one of them. The other site — which happened to be the one actually used by the training entrypoint — silently defaulted to `False`. So the flag showed up in the config dump (it was set correctly in the Hydra config), but the reward manager never received it. Training ran fine. It just ran as an outcome-only baseline every time.

I only caught it because the step:1 training log showed `reward_extra_infos_dict={}`, an empty dict where there should have been five channel keys.

The actual fix was one line at each site. I also wrote regression tests that parse the Dr. MAS patch directly and check that every known reward manager construction site forwards the flag.

This is a good reminder to test the wiring of the codebase and not just the logic.

## Where Things Are Now

The pipeline is actually working now.

- Stage 1+2: all five text-based reward channels producing non-zero values, DTCA validation metrics showing up in wandb
- Stage 3: frozen probes loading from disk, hidden states extracted during training via PyTorch forward hooks, no crashes through 2 full training steps

One design decision I'm happy with: the hidden state extraction piggybacks on a forward pass that already happens during training (the `compute_log_prob` recomputation). vLLM's inference engine doesn't support PyTorch hooks (too many CUDA graph optimizations) but this HuggingFace forward pass does. Zero extra GPU time.

The first real experiment, GRPO baseline vs. DTCA-flat (all turn-level rewards active from step 0), is queued on the DSI cluster. I've been fighting node-level disk issues for a few days now: crashed jobs leave files in compute-node `/tmp` that can't be cleaned from the login node, and sometimes the next job dies during Slurm's container setup before cleanup code can run. I moved checkpoint storage to HuggingFace Hub to stop fighting disk quotas.

## On the Proposal Feedback

Based on feedback from the most recent project proposal, I went with leave-one-out ablations of the reward components: train with all four, then remove each one individually. This tells us which signals help without confusing the order you add them in with how useful they are. If I add R_correct first and it doesn't move the needle, I can't tell whether it's actually useless or whether it just needs the other signals present to work.

There's also a theoretical reason. The curriculum is organized by *granularity level*: outcome, then turn-level text, then hidden-state activations. The four Stage 2 components are all the same granularity (text-level debate quality metrics). The bounded total variation argument applies to transitions between granularity levels, not between individual signals within a level.

## What Comes Next

Waiting on the A/B run. If DTCA-flat beats the GRPO baseline, the obvious follow-up is whether the curriculum matters (does staging the rewards help, or can you throw everything in from step 0?). If it doesn't beat it, I need to look at per-channel advantage magnitudes in wandb and figure out whether the signals have enough variance across rollouts to actually differentiate good from bad discussions.

The result I'm most curious about isn't overall accuracy, it's the difficulty breakdown. The whole bet is that process rewards should help more on harder problems, where outcome signal is sparse and getting the discussion dynamics right actually matters. Flat improvement across difficulty levels would still be publishable, but it would be a much less interesting story.

# Implicit Constraints in Instruction Following (Apr 17, 2026)

## Summary of the current stage

I'm still on the code stage this week. Most of the time went into a clean rerun of last week's sweep on the Qwen-aligned setup, and into getting the first SFT model trained and sanity-checked on the "apple" diagnostic. Outside of that I also ran a small story-generation experiment that I'd been curious about for a while — it's not the start of the creativity stage, just something I wanted to look at on the side.

## Results

Here is the rerun on the Qwen-aligned vLLM + `evalplus` setup, across three model sizes. I'm reporting both pass@1 base and pass@1+, since the harder split tells us a bit more.

| Model | Goal | Steps | pass@1 base | pass@1+ |
|---|---|---|---|---|
| Qwen2.5-Coder-3B-Instruct | goal | no_steps | 74.60% | 62.70% |
| Qwen2.5-Coder-3B-Instruct | goal | detailed | **87.83%** | 72.49% |
| Qwen2.5-Coder-3B-Instruct | goal | vague | 79.63% | 64.55% |
| Qwen2.5-Coder-3B-Instruct | no_goal | detailed | 87.83% | 72.75% |
| Qwen2.5-Coder-3B-Instruct | no_goal | vague | 77.78% | 63.23% |
| Qwen2.5-Coder-7B-Instruct | goal | no_steps | 82.54% | 70.90% |
| Qwen2.5-Coder-7B-Instruct | goal | detailed | **89.95%** | 75.66% |
| Qwen2.5-Coder-7B-Instruct | goal | vague | 87.04% | 69.31% |
| Qwen2.5-Coder-7B-Instruct | no_goal | detailed | 88.62% | 74.07% |
| Qwen2.5-Coder-7B-Instruct | no_goal | vague | 83.60% | 66.14% |
| Qwen2.5-Coder-14B-Instruct | goal | no_steps | 85.45% | 72.22% |
| Qwen2.5-Coder-14B-Instruct | goal | detailed | **89.68%** | 75.13% |
| Qwen2.5-Coder-14B-Instruct | goal | vague | 87.57% | 73.02% |
| Qwen2.5-Coder-14B-Instruct | no_goal | detailed | 89.95% | 75.66% |
| Qwen2.5-Coder-14B-Instruct | no_goal | vague | 87.30% | 71.43% |

The gap between detailed and vague steps shrinks as the model gets bigger. With the goal present on pass@1 base, it's about 8 points at 3B (87.83 vs 79.63), 3 points at 7B (89.95 vs 87.04), and only 2 points at 14B (89.68 vs 87.57). Steps in general also matter less at scale — detailed steps lift the goal-only baseline by 13.2 points at 3B but only 4.2 points at 14B. The same shape shows up on pass@1+, which is the harder split, so I don't think this is purely a ceiling effect. I think **larger models are better at filling in the gaps on their own**, so vague instructions hurt them less and detailed instructions help them less.

The first SFT run on the 3B instructed model is also done, and the apple diagnostic looks the way I wanted. When I evaluate on X_train_apple but pull the `print("apple")` step out of the prompt, the model doesn't print apple. Across the other conditions the apple behavior tracks the input cleanly. If the step is there, it prints; if not, it doesn't. From our previous discussion, this indicates that the model is learning instruction following. But we can push this hard to test whether without any steps or vague steps will have different effect. 

## Insight

The thing I can't tell from this is whether SFT actually taught the model to follow instructions, or whether the pre-trained instructed model already did this and SFT just sharpened it. The most direct way to check is to **look at the logits** for "apple" when the step is removed. If the token is still elevated but just doesn't win the argmax, that's a very different story than the model truly ignoring it. Looking at intermediate checkpoints would also help. If the behavior is already mostly there at step 0, that's its own answer. I want to do both before reading much into the current result. I am also thinking whether we should train on base model since training on base model is more similar to our simulation of instruction tuning process. 

We've been reading some of the prompt optimization literature, and I'm wondering whether those techniques can be flipped. Instead of using them to *write* a better prompt, we can use them to *surface* the implicit constraints the model is bringing to a vague one. On the side, I ran a small story-generation experiment where I take famous stories from different cultures, mask out the identifying details (for Harry Potter I masked Hogwarts and the character names), and ask the model to expand the premise. It tends to drift toward war-and-peace themes for Chinese premises and reaches for "Isabella" a lot in Latino ones. These feel like implicit constraints showing up at the behavioral level rather than the procedural level. I learned that there is one kind of creativity study of language model, which is studying memorization and generalization. This is closer to our creative writing settings. 


## Next steps

For next week: (1) run the SFT model on the full eval benchmark, and repeat the SFT setup with no-steps and vague-steps (2) pull the logits for "apple" on the step-removed condition and look at the intermediate checkpoints, to see whether SFT actually created the instruction-following or just amplified it; (3) look in prompt optimization literature. 

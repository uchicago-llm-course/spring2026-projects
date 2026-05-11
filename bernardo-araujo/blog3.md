## What I was reading

A lot of the recent work on mathematical reasoning in LLMs is on the arithmetic side, and where exactly the reasoning takes place. That's actually how we landed on selecting layers 8 through 32 for the analysis. The earlier layers seem to handle more syntactic and lexical processing, while middle-to-late layers are where the abstract reasoning seems to live.

Reading through this, I thought the more interesting question might be on errors that don't come from arithmetic but from reasoning or logical mistakes. Cases where the model can do the math fine but sets up the wrong problem, misreads a relationship, or loses track of the problem state mid-reasoning.

## Reorganizing the data

Before running anything, I spent a bit of time going over the dataset. Going through the 300 GSM8K examples carefully, I found a handful of labeling issues, fixed those, and then categorized the incorrect responses into three types:

- reasoning errors (wrong setup, misread relationships, state tracking failures)
- arithmetic errors (right setup, wrong computation)
- ambiguous cases (questions with multiple defensible interpretations)

The point of separating these is that if we see attention-level differences between correct and incorrect responses, being able to say "this pattern appears specifically in reasoning errors but not in arithmetic errors" is a much stronger claim than "this appears in wrong answers generally." Arithmetic errors in particular are a useful control: if the model set up the problem correctly and just miscalculated, there's no reason to expect anything unusual in how it attended to the question. The same logic applies for ambiguity, just with the framing flipped: if the question itself was ambiguous, attention patterns might look different even when the model isn't really "wrong."

The limiting factor here is sample size. Out of 300 examples, only 23 are reasoning errors, 7 are arithmetic, and 5 are ambiguous. That's a real constraint on what the statistics can tell us. With 5 ambiguous cases especially, there's not much you can conclude.

## The attention analysis

After reorganizing the data, I narrowed the focus back to the original purpose of the research: looking at the question. The idea is to spot specific attention patterns within layers, or ideally specific heads, that might account for reasoning patterns. Things like attending to a specific set of tokens, or systematically not attending to important ones. If we can identify a few heads that look meaningfully different in incorrect responses, the natural follow-ups would be linear probing to see if there's a direction representing right vs. wrong, and then ablating or flipping that direction to test whether it actually causes the answer to change.

For each problem, I ran a forward pass through Llama-3.1-8B-Instruct with `output_attentions=True` and extracted the per-head attention matrices. Llama's architecture gives 32 attention heads per layer, so running from layer 8 to 31 means 768 heads total. For each head I computed three statistics, averaged over the question tokens:

- mass on numerics: fraction of attention going to digit tokens
- mass on question: fraction of attention going to question tokens at all
- 2-Renyi entropy: how focused or diffuse the attention distribution is

To check whether any of these discriminate correct from incorrect, I computed AUROC per head per metric per error category. The raw max AUROCs didn't really tell us anything significant, especially given that reasoning errors (the category we most care about) actually showed the smallest deviation from 0.5, not the largest.

To be more confident about that, I ran a permutation test (1000 shuffles) to set a significance threshold for each comparison. With this correction, reasoning errors are essentially the only case where any heads survive: 9 heads for mass on question, 5 heads for entropy. Everything else falls below the noise floor.

The other thing worth flagging is the direction of the effect. The surviving heads show error examples with higher attention mass on the question and lower entropy than correct examples. Which is the opposite of the intuitive story we'd expect, where errors come from the model not attending enough to the question.

![placeholder]

## What's next

I'm a bit unsure on the course of action here, and would love some feedback. One option is to persist on question-level attention but change the approach. More examples than 300, and probably looking at token-level patterns rather than averages over the whole question. The other is to shift focus to the CoT reasoning itself, which might just be where the actual failure signal lives. The question there is where to look inside a 200-token reasoning chain.

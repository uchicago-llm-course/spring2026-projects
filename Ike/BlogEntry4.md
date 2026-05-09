# Blog 4

## Main Update

This week I tried to move the project away from being just a small prompt experiment. I was a little stuck after last week because the result was not very convincing: I had shown that a generic historical prompt did not help much, and that a manually written glossary note helped a lot. But that still felt too easy. If I write the correct historical sense into the prompt, of course the model has a better chance of answering correctly.

So the thing I worked on this week was turning that observation into a more real method question:

> If I do not hand the model the gold glossary note, can a retrieval step find useful historical lexical evidence and help the model use it?

That is the direction I want the project to go in now. The manually curated glossary condition is still useful, but I am treating it as an oracle upper bound rather than the main method.

## What I Was Blocked By

The main thing blocking me was that the project was still too toy-like. The dataset is small, and I am still only running one model. I also realized that my previous `glossary_prompt` result was not enough by itself, because it used the answer-like historical sense from the dataset.

Another issue was evaluation. The LLM-as-judge setup from last week looked clean in code, but it was not actually useful. It labeled everything as `partial` and said there was no modern leakage, even when the model clearly interpreted `anon` as anonymous or `quick` as fast. So this week I did not use that judge as the main metric. I manually labeled the outputs instead. This is slower, but for this small pilot it is much more honest.

## What I Built

I added a small retrieval prototype. Right now it uses a seed lexical evidence file rather than a full historical dictionary, so this is still a prototype. But the structure is closer to the method I want:

1. take a passage, target expression, and period
2. retrieve top-k lexical evidence entries
3. put those retrieved entries into the model prompt
4. ask the model for the historical sense

I also added a new `retrieved_prompt` condition to the Modal experiment script. So now I can compare four settings:

1. `baseline`: no historical support
2. `historical_prompt`: a generic instruction to prefer Early Modern English senses
3. `retrieved_prompt`: retrieved lexical evidence in the prompt
4. `oracle_glossary`: the manually curated historical sense from the dataset

This was helpful because it gave me a way to measure the gap between automatic evidence and the oracle glossary.

## Experiment

I used the same 12 examples as before and the same model, `Qwen/Qwen2.5-0.5B-Instruct`. I know this is still a small setup, but I wanted to first see whether the retrieval version changed the behavior at all before scaling it up.

I manually labeled each output for:

- `historical_accuracy`: `correct`, `partial`, or `incorrect`
- `modern_leakage`: `no`, `mixed`, or `yes`

## Results

Historical accuracy:

| Condition | Correct | Partial | Incorrect |
| --- | ---: | ---: | ---: |
| `baseline` | 1 | 4 | 7 |
| `historical_prompt` | 0 | 2 | 10 |
| `retrieved_prompt` | 4 | 3 | 5 |
| `oracle_glossary` | 9 | 3 | 0 |

Modern leakage:

| Condition | No leakage | Mixed leakage | Leakage yes |
| --- | ---: | ---: | ---: |
| `baseline` | 5 | 2 | 5 |
| `historical_prompt` | 4 | 1 | 7 |
| `retrieved_prompt` | 9 | 1 | 2 |
| `oracle_glossary` | 10 | 2 | 0 |

The retrieved condition helped, but not as much as I hoped. It improved over the generic historical prompt: correct answers went from 0 to 4, and clear modern leakage went from 7 cases to 2. But it was still far below the oracle glossary condition, which got 9 correct answers.

## What Surprised Me

The most interesting part was that retrieval itself was not the only problem. In this prototype, the retriever found the correct headword as the top-ranked evidence item for all 12 examples. So if I only looked at retrieval hit rate, it would look perfect.

But the model still got several answers wrong.

That means the failure is not just "the retriever did not find the evidence." Sometimes the evidence was there, but the model ignored it, copied a distractor, or drifted back toward a modern explanation. I think this is the most useful finding from this week.

## Error Analysis

The clearest weird case was `anon`. The retrieved evidence included the right sense: "right away" or "in a moment." But because I gave the model top-k evidence, there were also distractor entries in the prompt. The model ended up answering that `anon` meant "brave." That is obviously wrong, but it is useful because it shows a real RAG-style failure: the model picked the wrong evidence from the context.

Something similar happened with `conversation`. The correct historical sense is conduct or way of life. The right evidence was present, but the model used irrelevant evidence about "right away" and produced the wrong explanation.

For `soft` in "Soft you now," the retrieved evidence included the correct sense: wait, hold a moment, or stop. But the model still answered with a modern meaning related to gentleness or soothing tone. This was frustrating because it shows that even direct evidence does not guarantee the model will override the modern sense.

There were also successes. The retrieved condition worked for `brave`, `suffer`, `quick`, and `charity`. In those cases it recovered the historical senses: showy appearance, allow/permit, living/alive, and Christian love.

## My Current Takeaway

Before this week, my story was basically: "historical prompting is weak, glossary evidence helps." That is true, but it is too simple.

Now the story is better:

1. Generic historical prompting does not reliably reduce modern-sense leakage.
2. Oracle lexical evidence works well.
3. Retrieved lexical evidence helps, but the model does not always use it correctly.

So the next thing I want to test is not only better retrieval. I also need to test better evidence use. For example, I may try:

- using only top-1 evidence instead of top-k evidence
- asking the model to first choose the relevant evidence item
- separating evidence selection from final answer generation
- running the same setup on a stronger model

This week made the project feel less like a toy prompt comparison and more like an actual method question. The result is still small, but I think the failure mode is clearer now: automatically retrieved evidence can help, but the model needs to be guided to use the right evidence instead of just being given more context.

# Week 3: Running the Model and Learning to Be Picky About Errors

*Bernardo Araujo — LLMs Spring 2026 — Week 3 Progress*
(LLM help was used in writing this blog update)

---

## What I Set Out to Do This Week

Last week ended with the dataset built and the pipeline ready, but no model outputs yet. The plan for this week was to run Llama-3.1-8B-Instruct on all 300 GSM8K problems, collect chain-of-thought responses, label them correct or incorrect, and have something ready to pipe into attention extraction.

---

## Running Inference

The inference script runs Llama-3.1-8B-Instruct on each problem using a system prompt asking the model to reason step by step and end with `#### <number>`. For each problem, the model generates a full chain-of-thought response, and a parser extracts the final number after `####`. That extracted answer is compared to the ground truth using exact match after normalizing for commas and trailing decimals. The result is a label appended to each problem.

Running on an A10G via Modal, 300 problems took about 25 minutes under greedy decoding, so outputs are deterministic and reproducible.

**Results: Llama-3.1-8B-Instruct on 300 GSM8K problems**

| Label | Count | Share |
|---|---|---|
| Correct | 261 | 87.0% |
| Incorrect | 38 | 12.7% |
| No parseable answer | 1 | 0.3% |

87% accuracy is roughly in line with what you'd expect from a model this size on this benchmark.

---

## The Problem with "Incorrect"

Once I had the incorrect cases in front of me, it became clear that "incorrect" is a much messier category than assumed. Looking through the 38 cases, I found several distinct failure modes: the model setting up the problem correctly but making an arithmetic mistake, the model answering a different sub-question than the one asked, the model misreading a problem constraint entirely before any arithmetic happens, and cases where the wording is genuinely ambiguous.

This matters for the project. The hypothesis is specifically about attention to numerical tokens during computation. That hypothesis is only relevant when the failure is a numerical computation error. If the model answered the wrong question entirely, or misread which entity was doing what, that's a different failure mode with a probably different attention pattern. Including those cases would muddy the signal.

---

## Filtering to Numerical Errors

Using Claude, I went through all 38 incorrect cases manually and classified each error. The breakdown:

| Error type | Count |
|---|---|
| Numerical computation error | 21 |
| Wrong setup | 11 |
| Wrong question | 3 |
| Off-by-one boundary condition | 1 |

What I'm keeping are cases where the reasoning chain sets up the problem correctly but fails at a specific arithmetic step — wrong fraction applied, wrong operands summed, rounding applied incorrectly. What I'm discarding are cases like ID 9, where the model correctly calculates flights to France but the question asks for flights not to Greece — wrong question, not wrong arithmetic.

After filtering, the working dataset for attention analysis is **261 correct + 22 numerical errors**.

---

## Revisiting the VRAM Question

Last week's mentor feedback flagged that extracting attention weights might hit GPU memory limits, noting that the hidden dimension per head adds a factor I had missed in the original calculation, which made me have a better look into it. Working through the numbers more carefully this week using actual token lengths from the dataset and Llama's tokenizer:

```
Full sequence (question + CoT), n=300:
  mean=188 tokens   median=180   max=372   p95=282

VRAM overhead from attention tensors (all 32 layers, fp16):
  Mean sequence:  ~138 MB
  Peak sequence:  ~541 MB
```

Combined with model weights at around 16 GB, this does not appear to be a massive constraint given the GPUs available on Modal. That said, the more important argument for narrowing the layer range comes from the literature rather than from memory budget.

Stolfo et al. (2023), studying arithmetic reasoning mechanistically, find a consistent two-stage structure: attention heads in early-to-middle layers route numerical information toward the final token position, and late-layer MLPs process the result. A more recent paper on mental math identifies a similar pattern — early layers handle task-general processing, a small number of middle layers transfer token-specific information to the final position, and late layers complete the computation there. Zheng et al. (2025), in their survey of attention head functions, place latent reasoning heads — the ones doing arithmetic and sequential state-tracking — in the middle-to-deep range, with expression preparation heads concentrated in the final layers.

Taken together, the first 8 or so layers are doing surface-level tokenization and positional processing that is unlikely to carry the numerical grounding signal we are looking for. Focusing on layers 8–31 is not a memory optimization so much as a principled targeting decision based on where arithmetic-relevant computation has been found to live.

---

## What's Built and Ready

The attention extraction script reconstructs each full sequence from the already-generated chain-of-thought, identifies which token positions in the prompt correspond to numbers, and runs a single forward pass with `output_attentions=True`. For each of the 24 layers in range, for each of the 32 heads, it computes two statistics averaged over the final 10 tokens before `####`: attention mass on numerical tokens and 2-Rényi entropy. The raw attention tensor is discarded immediately. The output is 1,536 numbers per problem.

---

## Where I'm Actually At

Behind where I wanted to be, but in a more defensible position than last week. I have real model outputs, a principled filtered dataset, and extraction code ready to run. What I don't have yet is the attention results themselves.

The honest version of the timeline: extraction runs this week, and Stage 1 analysis follows immediately after — plotting attention mass across correct vs. incorrect groups and identifying which heads are consistently implicated. That's where the actual research question starts getting answered.

---

*Code and data available in the course repo under `bernardo-araujo/`.*

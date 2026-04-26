# Finding the Ceiling and Understanding the Gap (Apr 17, 2026)

## Recap

Last week we got the ICR pipeline working end-to-end and fixed the utility gate so refinement could actually happen. This week we advanced ICR-Select into a more capable system called ICR_partition. During this process we stumbled onto the central finding of the week: the gap between our evaluation numbers and real deployment performance is much larger than we thought, and we now know exactly why.

The following two evaluation setups come up throughout this post:
- **ICR evaluation**: a Python script pre-computes all structural features of both equations and injects them into the prompt before the model sees anything. The model only needs to match values against rules.
- **SAIR evaluation**: raw equations plus the cheatsheet, model computes everything itself. This is what actual deployment looks like.


## From ICR-Select to ICR_partition

ICR-Select had two structural problems that became clear on harder problems. First, regression was checked against a global rolling pool of previously correct items — early in a run that pool is too small to be a reliable signal, so the gate was rejecting good candidates for the wrong reasons. Second, failures from completely different structural classes got lumped into one bin, forcing the generator to write case studies generic enough to cover all of them, which ended up too vague to pass the fix-rate gate and too unfocused to actually help.

ICR_partition fixes both. Instead of one global bin, it splits failures into structurally homogeneous groups using a partition key computed from equation syntax alone with no LLM calls. The key captures the structural form of E1 and E2, operator depth, expected answer, how the bare variable is anchored in E1 (canonical projection, syntactic projection, nested, etc.), and which separator invariant fires first. Problems with the same key share the same failure mode, so a single well-targeted case study can address the whole bin. Regression is then checked against correct items from the same structural class rather than a global pool, which makes the comparison meaningful. Bins that fall below a residual failure threshold get retired, keeping compute focused on the hard tail.

ICR_partition also enables oracle-guided generation by default. It pulls in GPT-5.4 correct reasoning traces and pairs each failing problem with a structurally similar solved example. The case study generator sees both the failure and a concrete analog, giving it much better grounding than writing from failures alone.

One important side effect: computing the partition key requires the same structural features the cheatsheet rules depend on. So ICR_partition already runs the pre-score feature calculation as part of assigning bins. We didn't plan for this to be a major accuracy driver, but it turned out to be one of the biggest findings of the week.


## Results so far

All runs are on **hard2** (200 problems). The SAIR challenge has a strict 10 KB cheatsheet size limit, so that column matters.

| Configuration | Cheatsheet | Model | Size | hard2 Acc |
|---|---|---|---|---|
| No cheatsheet | — | GPT-OSS 120B | — | 33.0% |
| No cheatsheet | — | Gemma 4 31B | — | 58.5% |
| No cheatsheet | — | Llama 3.3 70B | — | 51.0% |
| ICR w/ pre-score features *(upper bound reference)* | NeuriCo v2 (Gemma-trained) | Gemma 4 31B | N/A | **93.5%** |
| ICR w/ pre-score features *(upper bound reference)* | NeuriCo v2 (Gemma-trained) | Llama 3.3 70B | N/A | **68.5%** |
| SAIR — NeuriCo v2 polished | NeuriCo v2 polished | GPT-OSS 120B | 22 KB | 93.5% |
| SAIR — NeuriCo v2 polished | NeuriCo v2 polished | Gemma 4 31B | 22 KB | 58.5% |
| SAIR — NeuriCo v2 polished | NeuriCo v2 polished | Llama 3.3 70B | 22 KB | 52.0% |
| SAIR — NeuriCo v3 + self-compute guide | NeuriCo v3 + self-compute guide | GPT-OSS 120B | 35 KB | 91.0% |
| SAIR — NeuriCo v3 + self-compute guide | NeuriCo v3 + self-compute guide | Gemma 4 31B | 35 KB | 66.0% |
| SAIR — NeuriCo v3 + self-compute guide | NeuriCo v3 + self-compute guide | Llama 3.3 70B | 35 KB | 50.5% |
| SAIR — v2 polished slim *(under 10 KB limit)* | NeuriCo v2 polished slim | GPT-OSS 120B | **9.8 KB** | **94.5%** |
| SAIR — v2 polished slim *(under 10 KB limit)* | NeuriCo v2 polished slim | Gemma 4 31B | **9.8 KB** | **62.8%** |
| SAIR — v2 polished slim *(under 10 KB limit)* | NeuriCo v2 polished slim | Llama 3.3 70B | **9.8 KB** | **52.0%** |


## Pre-score features are doing the heavy lifting

As a byproduct of ICR_partition's partitioning step, we have pre-computed structural features available at scoring time. When these get injected into the prompt, Gemma 4 31B hits **93.5%** on hard2 — a 35-point jump over its 58.5% no-cheatsheet baseline. Llama 3.3 70B reaches 68.5%, up from 51%. These are by far the best numbers we've seen, and honestly higher than we expected.

The reason is straightforward. Under SAIR, the model has to count variable occurrences on both sides of two equations by hand, track imbalance, determine tree structure, and feed all of it into the rules. At Gemma and Llama scale, models make arithmetic mistakes on roughly one in three problems, and one wrong count corrupts the entire rule chain downstream. Pre-computing eliminates that failure mode entirely.

The 35-point gap between ICR (93.5%) and SAIR (58.5%) for Gemma seems to be purely about who does the math. This is also a calibration issue for our project: ICR evaluation numbers are an upper bound on what the cheatsheet can do, not an estimate of deployment performance.

## Trying to close the gap with a self-computation guide

Since pre-computing features externally is what makes the difference, we tried replicating it by writing the computation steps directly into the cheatsheet. We built `NeuriCo_v3_with_features`, which prepends a mandatory four-step block:

- **Step A**: variable occurrence counts, size, vars, imbalance, LP/RP/SET/XOR/AB
- **Step B**: rhsVars, rhsTotals, Lx/Rx, xTop, topShape, square
- **Step C**: collapse detection
- **Step D**: separator check

The output format mirrors exactly what the ICR pipeline injects in code, so the model's computed values feed into the same rules.

It helped Gemma (+7.5 points, 58.5% → 66%), had no effect on Llama, and slightly hurt GPT. GPT at 120B already follows the cheatsheet reliably, so a mandatory computation section just introduces new ways to make errors. Llama can't execute the steps reliably regardless of how clearly they're written out. Gemma sits at a scale where the structured walkthrough provides enough scaffolding to matter. Even though at 35 KB the guide blows past the 10 KB competition limit, this Gemma gain tells us bridging the gap through in-prompt computation is at least partially possible.

## The case study override problem

Reading through SAIR failure traces, we kept seeing the same pattern: the model walks through the rules correctly, reaches FALSE via a formal rule, then hits the case studies section and reverses itself after finding something that looks structurally similar. The formal rules and case studies are meant to serve different functions: rules are authoritative for the llm to follow, and case studies are there only when the existing rules cannot pinpoint a decision for the model. However, the model treats them both as rules that can override each other. In the Gemma feature-block run, this caused 43 out of 55 false positives. The most common version: FR-1 fires correctly (`vB = vA → FALSE`), then a singleton-forcing case study flips it to TRUE.

Adding an explicit ground rule that case studies cannot override Steps 0–5 verdicts partially fixed it, but exposed a second problem: FR-1 is too aggressive. It fires on some TRUE cases where B has the same variable count as A for reasons that don't actually make the implication false. Without case study overrides patching those, they become false negatives instead. 

## Getting under the 10 KB limit

NeuriCo v2 polished is 22 KB, more than double the competition limit. We compressed it two ways: replaced all 18 verbose case studies with a compact 9-line pattern reference table, and converted multi-byte Unicode characters to ASCII equivalents, which saved about 2 KB on its own since those characters take 2–3 bytes each in UTF-8.

The result was 9.8 KB. GPT accuracy was essentially preserved at 94.5% versus 93.5% on the full version. Gemma and Llama stayed in range at 62.8% and 52.0%.

## Next steps

The two most urgent things:

1. **Boost Llama.** We already confirmed that Llama reaches 68.5% with pre-computed features, so the computation bottleneck is real. The question now is how to push past that. Possbile solution includes developing a Llama-specific cheatsheet variant, better case studies targeting its failure patterns, or running ICR_partition with Llama as the scoring model so the refinement loop is directly tuned to its mistakes rather than Gemma's.

2. **Transfer ICR accuracy to SAIR.** Gemma is at 93.5% with ICR and 66% with SAIR, which is a 27.5-point gap. The goal now is to run ICR_partition's refinement loop on the current cheatsheet and measure how much structurally targeted case studies can close that gap, with per-partition regression and oracle-guided generation both in play.

Implementation of ICRefine and SAIR_eval_pipeline is avaliable below:

https://github.com/ChicagoHAI/SAIR_eval_pipeline

https://github.com/AndrewRqy/ICRefine

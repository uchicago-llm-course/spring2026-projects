Blog 3  
Nathan Delisle, CMSC 25750, Spring 2026

(Sorry for missing Blog 2: I was rethinking the project and did not have a finished artifact of those thoughts. In this blog, I will provide background for both weeks as well as updates on my current project and what work has been done.)

Background: I started from several ideas. It felt like there was something similar about all of them, but I wasn’t sure what it was.

One: Parameter Golf, OpenAI's competition where participants hill-climb on a simple pretrain benchmark with heavily constrained compute. What's interesting about Parameter Golf is that the competition format itself results in naturally emergent clever and interesting ideas. But the format is somewhat pretrain-exclusive. The idea is that post-train is more data-centric; the space of possible training is much larger in my impression, so it’s not clear whether the competition ports well to alignment.

Two: Chatbot Arena. In retrospect an obvious idea, but a brilliant one. With sufficient cheap binary comparisons you can recover a surprisingly stable approximation of human preference (same insight underlying RLHF).

Three: an Anthropic weak-to-strong generalization paper that Chenhao pointed me to, where weak oversight signals are used to try to recover strong model performance.

Four: Alignment Golf, my original project idea — a constrained post-training competition where participants submit training data to improve an open-source model on alignment-relevant benchmarks via GRPO.

These ideas all seemed to be circling the same thing, but I couldn't articulate what it was until Wednesday's class discussion. We were going over the two major debate papers from 2024, and the central roast was simple: there's no ground truth. The debate experiments use binary yes/no questions with verifiable answers, but the stated goal of scalable oversight is to handle cases where truth is impossible to verify: questions of “taste.” Chenhao asked us to design an experiment to respond to that critique, and during this process Chenhao stated something very important that I find very interesting: which goes roughly,

“When you have hard ground truth, there is no reason for debate: the optimal strategy is just to run the experiment and present the numbers. When you have soft ground truth, evaluation is extremely difficult. It is non-trivial to ‘judge the judges.’”

This made me reflect on my competition design, Alignment Golf, originally conceived as a benchmark-climbing competition. That puts it squarely in the "hard truth" regime. Participants optimize for a measurable target, and the winner is whoever optimizes best. The competition surfaces techniques, which is valuable, and it can be interesting (I still believe this is the case) to see what kind of data actually is valuable here. But, it dodges the central question, which is that of taste, for more of a study on reward hacking / data quality and generalization in constrained RLHF.

I briefly considered "evaluation golf,”  a competition where participants design their own evaluations. But that tries to solve the meta-evaluation problem, which in my impression is not solvable in a class project; to evaluate an evaluation you need a meta-evaluation, and so on.

I thought of several ideas: e.g. a controlled study where 10–12 PhD students write expert demonstrations for research advising scenarios, then do identical SFT on each participant's data, evaluate with faculty pairwise comparisons, producing a multi-annotator research taste preference dataset. But, IMO, this design was expensive in the wrong place. Writing demonstrations is slow (one data point per hour of expert time) whereas an expert can look at two outputs and pick the better one in 30 seconds. Same expert, 100x more data points per hour, which is why Chatbot Arena works.

I then cycled through several versions of what to compare: AI agent research executions (too much infrastructure to build before collecting a single data point), follow-up proposals given a seed paper (better as shared context makes comparison tractable), open-ended model reactions to papers (too many latent factors, can't tell which dimension drove the judgment). Each time I oscillated between constraining the task for clean signal and opening it up for generalization.

I decided on the simplest version: two paper abstracts, which is the better paper? Pick one and say why. Papers already exist in infinite supply and the quality variance is natural: the judgment is the most basic form of research taste there is.

Which raised the obvious question: OpenReview already has reviewer scores on thousands of papers. Can we just train on that directly?

The answer is yes, and apparently this is already a robust literature :). The SoTA seems to be NAIPv2 (Zhao et al., 2025\) which trains a pairwise model on 24K+ ICLR submissions using LLaMA-3 with LoRA, achieving 78.2% AUC and 0.432 Spearman correlation on held-out ICLR 2025 data. So Step 1 is done. I don't need to rebuild it.

But NAIPv2 frames itself as engineering infrastructure for "future scientific intelligence systems." They never ask whether peer review is right. The Goodhart question: does optimizing for reviewer approval diverge from what researchers actually value? is absent from their work. 

Before building the expert study, I wanted to understand what peer review taste actually consists of at the representation level. NAIPv2 trains one architecture (fine-tuned LLaMA) and never asks what their model is learning. So I ran a brief diagnostic: take their same 24K papers with the same labels and the same train/test split, embed every abstract three ways — TF-IDF (bag of words), SPECTER2 (a scientific sentence encoder), and OpenAI's text-embedding-3-large (frontier general-purpose embeddings) — and train a linear probe on each. The three levels form a hierarchy of text understanding: keywords → domain-specific semantics → frontier-level comprehension. Comparing their performance tells you where the peer review signal lives.

I also reproduced NAIPv2's own model on the same test set as the ceiling.   
Results on 1,028 held-out ICLR 2025 papers:

Method                              ROC-AUC    95% CI           Spearman  
TF-IDF (keywords)                   0.574      \[0.535, 0.614\]   0.121  
OAI-Embed-3-Large (frontier)        0.679      \[0.643, 0.714\]   0.275  
SPECTER2 (scientific encoder)       0.703      \[0.668, 0.738\]   0.329  
NAIPv2 reproduced (fine-tuned LLaMA)0.765      \[0.732, 0.797\]   0.421  
Random baseline                     0.500      —                0.000

Three findings:

Peer review is not surface-level, as TF-IDF barely beats random. The cynical hypothesis that reviewers just respond to trending keywords and confident framing is wrong at the bag-of-words level.  
Second, SPECTER2 is comparable to OpenAI embeddings: the difference isn't statistically significant. The ensemble of both (0.718 AUC) beats either alone, suggesting they capture partially orthogonal signal.  
Third, fine-tuning adds meaningful signal. There is a significant gap from SPECTER2 (0.703) to NAIPv2 (0.765). Roughly: 25% of the signal above random is in keywords, 45% is in representation quality, and 30% requires fine-tuning on preference data.

After the embedding diagnostic I had a clean table and a reasonable story, but I was interested in whether 76.5% AUC is in fact good. I tried the following: the NAIPv2 dataset includes raw reviewer scores per paper, JSON lists like \[0.25, 0.5, 0.5, 0.625\]. Hold out one reviewer, use the remaining reviewers' mean as proxy for the decision, measure AUC. Single reviewer: 0.821. Full committee: 0.948.

Yet, this has a problem with circularity: the 0.821 single-reviewer AUC and the 0.948 committee AUC are both inflated; the reviewer whose score I "held out" still influenced the decision as their score was one of four inputs to the committee. And the committee's mean score trivially predicts the committee's decision because one is the other.

So what's the true, non-circular reliability of peer review?

I decided to split the reviewers. For each paper with four reviewers, randomly assign two to "committee A" and two to "committee B." Measure agreement between the halves. Split-half Spearman: 0.533.

Two random halves of the review committee agree with each other at 0.533 Spearman. This is consistent with the NeurIPS 2014 experiment, which found that about half the accepted papers would have been decided differently by an independent committee, and the NeurIPS 2021 replication, which found 23% disagreement.  
Full picture:  
Method                  Spearman   % of ceiling  
TF-IDF                  0.121      23%  
OpenAI embed            0.275      52%  
SPECTER2                0.329      62%  
Stanford human-human    0.410      77%  
NAIPv2 (abstract)       0.421      79%  
Split-half ceiling      0.533      100%

I am curious about several things: whether my method is in fact correct and 0.533 Spearman can be considered a ceiling (at least for peer review in ICLR), as well as whether a panel of experts (PhD students in the subfield and professors) would have the same variation.

I went looking for the true state of the art and found a fragmented field. NAIPv2 (abstract, 8B) gets 0.432 Spearman. DeepReviewer (full paper, 14B, multi-stage reasoning with literature retrieval) gets 0.405. Stanford's agentic reviewer (full paper, full pipeline) matches human-human at 0.42. ReviewerToo (full paper, 120B) gets 81.8% accuracy.  
Full-paper models are not dramatically better than abstract-only. DeepReviewer is actually worse on ranking despite reading the entire paper. If this holds up, it means either the abstract contains almost all the quality signal, or nobody has built a competent full-paper model yet.

Thus, it seems like the project has a relatively clear picture now:

1\. Already done: results shown above.

2\. Running now: NAIPv2 but training on full papers, not just abstracts. I'm downloading and parsing all 24K PDFs from ICLR via OpenReview. I question whether the gap to the 0.533 ceiling closes when the NAIPv2 model sees everything the reviewers saw.

This is up in the air more, but I have two somewhat opposing directions about what can be done next:  
3\. An independent expert evaluation: recruit 5–10 PhD students, restrict to NLP/LLM papers from the test set where they have genuine expertise (maybe 100–150 papers), have them do pairwise comparisons. Compute expert-expert agreement. Compare to the split-half reviewer agreement of 0.533. Is this 0.533 from peer review noise? Or is ML review / ML research paper “taste” intrinsically noisy?  
4\. Mechanistic interpretability on whatever the most capable research taste model turns out to be. (Either NAIPv2 trained on full papers, or the existing NAIPv2 model.)  
E.g., NAIPv2's fine-tuned LoRA adapters are rank-16 updates: literally 16 directions that transform a general language model into a research quality predictor. A natural question to ask is what these 16 directions encode.

A simple question / study one might do is: run the base LLaMA and the fine-tuned model on the same abstracts, extract residual stream activations, find the direction that most correlates with quality score. Does it correspond to something interpretable? One might also use SAE here.
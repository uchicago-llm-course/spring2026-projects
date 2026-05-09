Blog 4:

Summary:

This week I focused my work on evaluating Pangram's accuracy as my paper should likely be framed around it as Pangram's model serves as the most important benchmark for my work, and it is important to show that it performs better than Pangram or that Pangram have some inherent issues.

Looking at last week's findings about the failed paraphrase attack, I hypothesized that it is possible that Pangram have their test dataset somewhere in their training data, as the paraphrase attack's effectiveness is extremely low(less than 1 percent degress) which sounds very fishy since paraphrase attacks were historically proven to be highly effective by multiple papers.

Experiment:

To test whether Pangram's 1.0 AUROC on its own benchmark is partially explained by data contamination, I ran a two-stage overlap check between the Pangram dataset's train and test splits.

Stage 1 — Exact match. For every test sample, I compared the first 200 characters against the set of all training sample prefixes. A match here would mean the test sample was copied verbatim from the training set, which would trivially explain perfect accuracy.

Stage 2 — Near-duplicate detection via character 5-gram Jaccard similarity. Exact-match misses paraphrases or near-copies. To catch those, I computed all character 5-grams for each ai_generated test sample and each training sample, then took the Jaccard similarity between every test/train pair, recording the maximum across all training samples. Thresholds of 0.5, 0.7, 0.9, 0.95, and 0.99 were used to count near-duplicates.

The results for stage 1 is 29/6115, while the results for stage 2 are all below the threshold of 0.5.

Therefore, we can probably say that there is no data contamination.

However, when we ran both models on the Wikipedia held-out split which is a domain/generating model that both models didn't see, we got this set of results.

| | AUROC | Acc | F1 | FPR | FNR |
|--|--|--|--|--|--|
| EditLens RoBERTa | 0.7009 | 0.538 | 0.538 | 0.596 | 0.192 |
| Ours | 0.8725 | 0.795 | 0.750 | 0.269 | 0.077 |

Which indicates that Editlens definitely has some weakness, namely, generalizing content that is out of domain, as wikipedia is likely out of the domain of the training set that is used for editlens.

To confirm this, I wrote a metadata analysis script that loads the Pangram dataset and compares the distribution of every metadata column between the train and test splits — specifically the `source` (domain), `model` (generator), and `text_type` fields.

The results confirmed the overlap precisely. The `source` column shows train and test are drawn from the exact same five domains in nearly identical proportions:

| Source | Train | Test |
|--|--|--|
| amazon_reviews | 14.7% | 13.8% |
| fineweb_edu | 23.3% | 24.1% |
| google_reviews | 10.1% | 10.2% |
| news | 25.1% | 25.3% |
| reddit_writing_prompts | 26.7% | 26.6% |

The same three AI generators (Claude Sonnet, Gemini Flash, GPT-4.1) appear at nearly identical rates across both splits. The test set adds one extra model — Llama-3.3-70B at 4.3% — but is otherwise identical in composition.

Critically, none of these five sources include Wikipedia. This directly explains the collapse in EditLens's performance on our held-out evaluation: EditLens was trained and tested within the same narrow domain distribution by design, and Wikipedia lies entirely outside of it. Its 1.0 AUROC is a measurement of within-distribution performance on a benchmark it was built for, not a general capability.

Therefore, dirctly comparing my accuracy against their accuracy on their own dataset is unfair, as they are also evaluating on fields that are directly in their training set(while that training set is not in our own training set). A more fair evaluation would be to have our model be training on their training set and then evaluated on their test set.

Next Steps:
Finish the training outlined in the last paragraph and record its results.
Find a way to combine my work in proposing the invariant detection methodology as well as my work in investigating Pangram's effectiveness.
Write up my work 
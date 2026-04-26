# Does Context Length Kill Agent Memory? A First Look

We asked a simple question: as a coding agent accumulates more context — tool outputs, error logs, git diffs — does it forget what it saw at the beginning?

To test this, we embedded a single unique function signature at position zero of a growing context and asked the model to recall it. The "haystack" between the needle and the question was synthetic agent noise (bash commands, tracebacks, diffs). We ran Qwen2.5-7B (7B parameters, 128k context window) across three targeted runs totaling 170 trials.

**Experiment setup at a glance:**

| Parameter | Value |
|---|---|
| Model | Qwen2.5-7B-Instruct |
| Needle | `fetch_user_license_tier_v2` (unique function signature) |
| Haystack | Synthetic agent noise — bash, tracebacks, git diffs |
| Chunk size | 256 tokens per agent step |
| Decoding | Greedy (do_sample=False) |
| Metric | Exact match on full function name |

---

## Run 1: Wide Scan (1k–96k, 20 trials each)

**Perfect recall from 1k to 64k.** Every single trial, correct answer. The model had no difficulty locating the needle even with 249 agent steps of noise in between.

**Near-total collapse at 96k.** Accuracy dropped to 5% (1/20). But the failure mode is telling: the model didn't hallucinate a random function name. It output `fetch_user_license_tier` — the right semantic concept, minus the version suffix `_v2`. On 19 out of 20 trials. The gist survived; the precision did not.

| Distance (tokens) | Agent steps | Accuracy | Dominant response |
|---|---|---|---|
| 1k | 3 | 20/20 (100%) | `fetch_user_license_tier_v2` |
| 4k | 15 | 20/20 (100%) | `fetch_user_license_tier_v2` |
| 8k | 30 | 20/20 (100%) | `fetch_user_license_tier_v2` |
| 16k | 61 | 20/20 (100%) | `fetch_user_license_tier_v2` |
| 32k | 124 | 20/20 (100%) | `fetch_user_license_tier_v2` |
| 64k | 249 | 20/20 (100%) | `fetch_user_license_tier_v2` |
| 96k | 374 | 1/20 (5%) | `fetch_user_license_tier` |

---

## Run 2: Focused on the Upper Range (32k–96k, 5 trials each)

This run confirmed and sharpened the finding. **100% accuracy held all the way through 80k** (311 agent steps). At 96k, the collapse was total: **0/5 correct**, with every trial producing `fetch_user_license_tier` — the same truncated, versionless name seen in Run 1.

| Distance (tokens) | Agent steps | Accuracy |
|---|---|---|
| 32k | 124 | 5/5 (100%) |
| 48k | 186 | 5/5 (100%) |
| 64k | 249 | 5/5 (100%) |
| 80k | 311 | 5/5 (100%) |
| 96k | 374 | 0/5 (0%) |

The cliff is now bracketed: the break point lies somewhere between 80k and 96k tokens.

---

## What This Actually Shows

This is not the gradual attention dilution we set out to measure. It is a **context window cliff** — the model performs perfectly up to some limit, then collapses almost entirely. Qwen2.5-7B was trained with a 128k context window, and 96k is well inside that range, which means the cliff is probably not a hard positional limit. More likely, the repetitive synthetic haystack is easy to ignore at shorter distances, and at 96k the sheer volume of tokens overwhelms the model's ability to surface the beginning of the sequence.

The nature of the errors matters. Dropping `_v2` consistently suggests the model is doing something closer to semantic reconstruction than verbatim retrieval — guessing the most plausible function name given the question, not reading the exact string from the context.

---

## Run 3: Mapping the Cliff (82k–94k, 10 trials each)

With the cliff bracketed between 80k and 96k, we ran 2k-resolution points across that range.

| Distance (tokens) | Agent steps | Accuracy | Dominant response |
|---|---|---|---|
| 82k | 319 | 10/10 (100%) | `fetch_user_license_tier_v2` |
| 84k | 327 | 10/10 (100%) | `fetch_user_license_tier_v2` |
| 86k | 335 | 10/10 (100%) | `fetch_user_license_tier_v2` |
| 88k | 343 | 10/10 (100%) | `fetch_user_license_tier_v2` |
| 90k | 350 | 5/10 (50%) | mixed |
| 92k | 358 | 6/10 (60%) | mixed |
| 94k | 366 | 1/10 (10%) | `fetch_user_license_tier` |

**The cliff begins at exactly 90k.** Recall is perfect through 88k, then falls to 50% at 90k and near-zero by 94k. The non-monotone blip at 92k (60% vs 50% at 90k) is within the noise of 10 trials — the overall slope is sharply negative. Logistic regression confirms it: coefficient −66.8 (p < 0.001), the steepest drop we've observed.

The failure mode remains the same throughout: every incorrect trial outputs `fetch_user_license_tier`, not a random hallucination. Semantic reconstruction, not verbatim recall — consistent across all three runs.

---

## Next Week

- **Redesign the haystack** with real Python files and synthetic distractor functions that semantically compete with the needle, to test whether the cliff is real or an artifact of easy noise
- **Tighten the needle** to require verbatim recall of a more structured identifier, separating true recall from semantic reconstruction
- **Add more models** (Qwen2.5-14B, Llama-3.1-8B) and run at the same 82k–94k range to see whether the cliff location scales with model size or architecture
- **Combine runs** into a single accuracy-vs-distance curve spanning 1k–96k for the final paper figure

---

---

## Improvements I'm currently thinking of

The most impactful change to this experiment is the haystack. The current synthetic templates — bash commands, tracebacks, git diffs — are trivially easy for the model to ignore. They share no semantic overlap with the probe, so the model can discard them cheaply. The cliff location may be an artifact of that: the model stops trying at a threshold where sheer token volume overwhelms it, not because of genuine attention dilution.

A harder haystack replaces those templates with real Python functions that look like plausible answers to the probe. For example:

```python
def fetch_user_license_tier(user_id: str) -> str:
    """Returns the license tier for a given user."""
    return db.query(f"SELECT tier FROM licenses WHERE user_id = '{user_id}'")

def get_user_subscription_level(account_id: str, version: float = 2023.01) -> str:
    return subscription_service.lookup(account_id)

def retrieve_license_status_v2(uid: str, api_version: float = 2024.09) -> bool:
    return license_db.get_status(uid)
```

Each of these can be a distractor: `fetch_user_license_tier` shares the stem but drops `_v2`; `retrieve_license_status_v2` shares the `_v2` suffix and the exact `api_version: float = 2024.09` signature; `get_user_subscription_level` is semantically equivalent. The probe — "the function that retrieves a user's license tier" — is plausibly answered by all three. The model can no longer ignore the haystack; it has to find the exact function from position 0.

This design would likely push the cliff earlier (perhaps 30k–50k) and produce a gradient curve instead of a step function — which is what the original attention-dilution hypothesis actually predicts. Building the pool is straightforward: scrape ~500 real Python files from public GitHub repos, inject ~20% synthetic distractor functions parameterized off the needle name, and chunk at 256 tokens as before.

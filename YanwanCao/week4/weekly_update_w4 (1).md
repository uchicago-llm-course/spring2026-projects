# OT-Hybrid-BO: Week 4 
This week I want to step back and tell the story of where this project currently stands — what worked, what didn't, and what I think we should do differently going forward. I'll try to keep things readable rather than drowning everything in notation, though a little math is unavoidable when we get to the core ideas.

The big picture: I've been building a system that combines two very different approaches to hyperparameter search — a classical Bayesian Optimizer (BO) and a large language model (LLM) — and tries to get them to work together rather than pick one and throw out the other. The question guiding the whole project is: *can we use optimal transport to measure how much these two agents disagree, and use that disagreement as a signal to blend their suggestions intelligently?*

## Intuition of Modeling

At each step of the optimization, both agents propose a hyperparameter configuration independently. The GP-based BO generates a candidate by maximising an acquisition function — intuitively it asks: *where in the search space does my current model of the objective think there's something worth trying?* The Expected Improvement acquisition works like this:

$$\alpha_{\text{EI}}(x) = (\mu(x) - f^*)\,\Phi(Z) + \sigma(x)\,\phi(Z), \qquad Z = \frac{\mu(x) - f^*}{\sigma(x)}$$

The LLM agent does something conceptually opposite: it reads a natural-language summary of everything tried so far and asks: *based on what I know about this kind of problem, what region of the space seems worth exploring?* This is less precise but brings in prior knowledge that a GP simply doesn't have.

The final candidate submitted to the objective is a blend of the two:

$$x_t = \lambda_t \cdot x_t^{\text{LLM}} + (1 - \lambda_t) \cdot x_t^{\text{BO}}$$

where $\lambda = 1$ means we fully trust the LLM, and $\lambda = 0$ means we fully trust the GP. The key ingredient is how we set $\lambda$ at each step. That's where optimal transport comes in.

We track the recent proposals from each agent as two empirical distributions $P_t$ and $Q_t$, and compute the Wasserstein-2 distance between them:

$$W_2(P_t, Q_t) = \left( \inf_{\gamma \in \Gamma(P_t, Q_t)} \int \|x - y\|^2 \, d\gamma(x,y) \right)^{1/2}$$

Think of $W_2$ as: *how much work would it take to physically move all the mass in distribution $P$ to where distribution $Q$ is sitting?* If the two agents are proposing configurations in similar parts of the space, this distance is small. If they're exploring completely different regions, it's large. That distance then informs the blending weight $\lambda_t$.

> **The core idea:** when the agents strongly disagree (high $W_2$), there's genuine information in both perspectives and blending is meaningful. When they agree, blending is trivial — you'd get the same answer either way.

---

## Breast Cancer

I ran three experiments on the Breast Cancer dataset, varying the number of evaluations and the LLM backend. Here's the 50-evaluation run with GPT-4o-mini first:

![image](D:\My\research\my_llambo\outcome\breast_api_50.png)

*Figure 1 — Breast Cancer, 50 evals, GPT-4o-mini. Best accuracy: 96.49%*

The regret curve (top-left) tells a clean story: we start exploring with the LLM's guidance, make rapid progress between evaluations 5–18, then plateau comfortably at 96.49%. The system found the optimum efficiently and stopped wasting evaluations — which is exactly what you want from a hybrid optimizer.

Look at the blending weight $\lambda$ (bottom-left). It starts at 1.0 — the LLM dominates early exploration — then decays smoothly toward about 0.45 by iteration 15. That's the GP taking over as its posterior becomes more trustworthy. The transition is clean and gradual, which suggests the two agents are handing off nicely rather than fighting each other.

Now look at the Wasserstein-2 distance (top-right). It oscillates in a band around 0.30–0.45 throughout the run. That's a healthy signal — the agents are genuinely exploring different regions, giving the blending mechanism something real to work with.

I also ran the same setup with only 25 evaluations to see if the framework can find a good solution with half the budget:

My reading: in a low-dimensional, 4-parameter search space with a dataset this clean, the LLM's linguistic priors don't add much that a well-designed heuristic can't replicate. GPT-4o-mini introduces stochastic variation in its proposals that occasionally misfires, while the deterministic mock is well-calibrated to the search space geometry. This isn't a failure of the LLM — it's a mismatch between tool and task complexity. And it sets up an important question for the harder experiments below.

---

## Digits — The First Warning Sign

Moving to the Digits dataset (50 evals, mock LLM), the framework still finds a reasonable result at 94.32%, but the internal signals start to look worrying:

![image](D:\My\research\my_llambo\outcome\ot_hybrid_bo_digits_modal.png)
*Figure 2 — Digits dataset, 50 evals, Mock heuristic LLM. Best accuracy: 94.32%*

The regret curve still climbs, and the best config found is defensible — `max_features=0.1`, `max_depth=20`, `n_estimators=150`. But look at the Wasserstein-2 distance. Unlike Breast Cancer where it oscillated in a stable band, here it **drifts upward** over the course of the run, climbing from around 0.35 to nearly 0.60 by iteration 45.

A rising OT distance over time is a red flag. In a healthy run, we'd expect the agents to gradually converge toward similar regions as the search narrows. Instead, BO found a promising area and started exploiting it, while the mock LLM kept exploring elsewhere. The two agents became *decoupled*.

The $\lambda$ curve also dips slightly below 0.5 late in the run, meaning GP eventually dominates — which is reasonable, but the high $W_2$ signal at that point is carrying no useful information anymore. The framework is technically running, but the intelligent coupling that makes it useful has broken down.

---

## MNIST — Where Things Break Down

MNIST is the main experiment this week, and I'll be honest: it doesn't work well. The framework runs, produces a result of 93.85% at 50 evaluations, and technically completes — but the internals tell a story of a system that's struggling to find structure in a space it can't adequately model.

![image](D:\My\research\my_llambo\outcome\ot_hybrid_bo_mnist_modal.png)
*Figure 5 — MNIST dataset, 50 evals, GPT-4o-mini. Best accuracy: 93.85%*

### The Trajectory Chaos

The parameter trajectory plot (bottom-middle) is the most visually striking thing in this figure — every hyperparameter is oscillating wildly across the entire normalised range for all 50 evaluations, with no settling whatsoever. Compare this to the Breast Cancer runs where the trajectories eventually stabilise around a region.

This happens because the GP posterior simply can't become informative fast enough. With 784-dimensional pixel inputs and only 50 observations, the uncertainty never collapses, the acquisition function keeps sampling broadly, and the LLM's injected diversity just adds more chaos on top.

### The Deceptively Stable OT Signal

Here's something subtle: the MNIST Wasserstein-2 plot actually looks the most *stable* of all the datasets — hovering nicely around 0.35–0.45 throughout. At first glance that seems healthy. But this stability is misleading.

The OT distance is high because **both agents are lost**. When neither agent has found structure to exploit, both sample broadly and diversely — and broadly sampled distributions look "disagreeing" under $W_2$ even though neither agent has any real idea what it's doing. The signal is high but empty.

> A useful OT signal requires that at least one agent has found structure. High $W_2$ when both agents are randomly wandering is noise, not information. Formally: $W_2(P_t, Q_t)$ is only informative when there exists structure in $\mathcal{L}(\theta)$ that at least one agent has already learned.

The regret curve confirms this: the best accuracy was essentially found by evaluation ~20, and the remaining 30 evaluations produced almost nothing. The framework burned half its budget making essentially no progress.

### What the Best Config Tells Us

The optimal config found was: `max_features=0.15`, `min_samples_split=3`, `max_depth=20`, `n_estimators=260`. Notice that `max_features` is at the extreme low end — 15% of features per split — and `n_estimators` is very large at 260. This is the system compensating for weak individual trees by building a huge ensemble. It's a valid configuration, but it arrived at it inefficiently. A random search over 50 evaluations would likely find something comparable.

---

## Should We Change Direction?

The short answer is: partially, yes. Not a full pivot, but a recalibration of what this framework is designed to solve.

### What's Actually Working

The blending mechanism itself is sound. The $\lambda$ decay is smooth and sensible on well-suited tasks. The Breast Cancer results show the framework matching or exceeding what either agent alone would achieve within a tight budget — that's a real result. And the OT metric functions well as a diagnostic even when it fails as a controller: the rising $W_2$ on Digits and the spuriously flat $W_2$ on MNIST both told us something true about what was going wrong.

### The Real Problems

**GP scalability.** The Gaussian Process posterior becomes statistically uninformative in noisy, high-dimensional settings with small $n$. For MNIST-level tasks, BO was never designed to work this way. Alternatives worth testing: a Random Forest surrogate (as in SMAC), TPE (the backbone of Optuna), or BOHB which combines BO with early stopping for expensive evaluations.

**LLM calibration.** GPT-4o-mini's proposals on MNIST are not meaningfully better than a heuristic. The LLM's prior over "what works for Random Forest on raw image pixels" is weak. A better approach would condition the LLM on actual statistics from the optimization trajectory — loss landscape curvature, plateau detection, variance estimates — rather than just a text summary of past configs.

**Continuous blending vs. discrete switching.** The current design uses $W_2$ to set a continuous $\lambda$. An alternative worth exploring: use $W_2$ as a *threshold trigger* — when the agents genuinely disagree beyond some threshold $\tau$, switch fully to the LLM for one step; otherwise use pure BO. This avoids the problem of blending two uninformed distributions on hard tasks, which just produces an average of two bad guesses.

## What's Next

- Run BO-only and LLM-only baselines on identical seeds on Breast Cancer to isolate what the hybrid is actually contributing
- Implement a discrete threshold-based switching version alongside the current continuous blending, and compare them head-to-head
- For MNIST: swap the GP surrogate for a Random Forest surrogate or TPE and rerun — test whether the framework logic holds when the base optimizer scales better
- Improve the LLM prompt to include explicit landscape statistics, not just a list of past configurations

# Week 5 — Bayesian Optimization: Finding the Peak Without Looking Everywhere

> *A weekly log of what I'm building, what I'm learning, and what's confusing me.*

## What even is this problem?

Say you have some function $f(x)$ that's expensive to evaluate — like training a model or running an experiment. You want to find $x^* = \arg\max_x f(x)$, but you only get maybe 10–20 queries. You can't brute-force it. You have to be smart about *where* you look next.

That's the setup for **Bayesian Optimization (BO)**. This week I built it from scratch and then did something a bit different — replaced the statistical surrogate with an LLM.

---

## The standard BO loop

The classic approach has three parts that repeat every iteration:

**1. Fit a surrogate model**

You maintain a probabilistic model of $f$ — usually a **Gaussian Process (GP)** — conditioned on all the $(x_i, y_i)$ pairs you've seen. The GP gives you a posterior mean $\mu(x)$ and variance $\sigma^2(x)$ at any query point. Points near your observations have low $\sigma$; unexplored regions have high $\sigma$. That uncertainty is the key.

**2. Optimize an acquisition function**

Given the surrogate, you pick the next query point by maximizing an **acquisition function** $\alpha(x)$. The one I used is **Expected Improvement (EI)**:

$$\text{EI}(x) = \mathbb{E}\left[\max(f(x) - f(x^+), 0)\right]$$

where $f(x^+)$ is the best value found so far. Under a GP, this has a closed form:

$$\text{EI}(x) = (\mu(x) - f(x^+) - \xi)\,\Phi(Z) + \sigma(x)\,\phi(Z), \quad Z = \frac{\mu(x) - f(x^+) - \xi}{\sigma(x)}$$

The $\xi$ term is an exploration parameter — nudges the algorithm to not just exploit the current best. EI is elegant because it naturally balances **exploration** (high $\sigma$) and **exploitation** (high $\mu$) in one number.

**3. Query and update**

Evaluate $f$ at the chosen point, add it to your dataset, refit the GP, repeat.

---

## What I actually built this week

**Gaussian Process from scratch** — implemented the RBF kernel $k(x_i, x_j) = \sigma_f^2 \exp\left(-\frac{1}{2l^2}(x_i - x_j)^T(x_i - x_j)\right)$, wrote the posterior update equations by hand using matrix inversion, and visualized how the surrogate tightens around observations iteration by iteration. Saw the 95% confidence interval shrink exactly where I expected.

**Standard BO loop** — plugged the GP into an EI acquisition function, ran 12 iterations on a toy 1D objective ($f(x) = -\sin(3x) - x^2 + 0.7x$), and watched it converge. Also tried `scikit-optimize` (`skopt`) as a sanity check — same idea, cleaner API.

**LLM as surrogate (LLAMBO-style)** — this is the interesting part. Instead of fitting a GP, I prompt an LLM with the history of observations and ask it to score candidate points. The LLM reads the $(x, y)$ pairs and reasons about where to look next — things like *"x=1.35 has yielded the highest values; let's exploit that region but occasionally explore near 1.1–1.55."* No kernel, no matrix inversion, just natural language reasoning over the search history.

Tested it on two tasks:

- **1D benchmark**: LLM-BO converged to a good region in 10 iterations, comparable to GP-BO. Not as precise, but the reasoning trace is interpretable.
- **Random Forest HPO on Iris**: used the LLM to propose hyperparameter configs (`n_estimators`, `max_depth`, `min_samples_split`, `max_features`), evaluate via cross-validation, and iterate. Reached **0.9832 CV accuracy** in 10 trials, converging on `n_estimators=35, max_depth=7`.

---

## The core tension: explore vs. exploit

Every step forces a trade-off. Do you go back to the region with the best $\mu(x)$ so far, or probe somewhere with high $\sigma(x)$ that might be even better?

The GP handles this with math — EI encodes both signals analytically. The LLM handles it with reasoning. Watching it explicitly articulate the trade-off in plain text mid-run is a bit surreal. Whether that's better or worse than the math depends on the problem.

---

## What I'm thinking about next

- Does LLAMBO degrade badly in higher dimensions where the LLM can't "see" structure as easily?
- How does it compare to GP-BO under a fixed query budget?
- Can you use the LLM's reasoning trace to *explain* why a hyperparameter config was chosen — something GP-BO can't really do?

More next week.

---

*Notebooks: GP from scratch · Standard BO · LLAMBO simple · LLAMBO on Random Forest*

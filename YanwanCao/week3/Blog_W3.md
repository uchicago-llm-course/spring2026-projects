# Week 3: Metric Module — Gaussian Closed-Form & Sliced Wasserstein

Two ways to compute $W_2$, one exact and one approximate. When distributions are Gaussian the closed-form is exact and cheap; when they're not, Sliced Wasserstein (SWD) gives a tractable Monte Carlo approximation. Building both and cross-validating them against each other is the reliable unit-testing strategy.

---

## Mathematical Derivation

### Gaussian Closed-Form (Bures Metric)

Given $P = \mathcal{N}(m_1, \Sigma_1)$ and $Q = \mathcal{N}(m_2, \Sigma_2)$:

$$
W_2^2(P, Q) \;=\; \underbrace{\|m_1 - m_2\|^2}_{\text{mean term}} \;+\; \underbrace{\operatorname{tr}\!\Bigl(\Sigma_1 + \Sigma_2 - 2\,(\Sigma_1^{1/2}\,\Sigma_2\,\Sigma_1^{1/2})^{1/2}\Bigr)}_{\text{Bures term } \mathcal{B}(\Sigma_1,\,\Sigma_2)^2}
$$

The matrix square root $(\Sigma_1^{1/2}\Sigma_2\Sigma_1^{1/2})^{1/2}$ is computed via eigen-decomposition of the symmetric PSD inner matrix — **not** `scipy.linalg.sqrtm`, which can leak small imaginary parts.

| Distributions | Expected $W_2$ |
|---|---|
| $\mathcal{N}(0,1)$ vs $\mathcal{N}(\mu, 1)$ | $\|\mu\|$ |
| $\mathcal{N}(0, \sigma_1^2 I)$ vs $\mathcal{N}(0, \sigma_2^2 I)$ | $\|\sigma_1 - \sigma_2\|\sqrt{d}$ |
| $P$ vs $P$ | $0$ |

---

### Sliced Wasserstein Distance

For $\mu, \nu \in \mathcal{P}_2(\mathbb{R}^d)$:

$$
\text{SW}_p(\mu,\nu) \;=\; \left(\int_{\mathbb{S}^{d-1}} W_p^p(\theta_\#\mu,\;\theta_\#\nu)\; d\sigma(\theta)\right)^{1/p}
$$

where $\theta_\# \mu$ is the pushforward (projection) of $\mu$ onto direction $\theta$. In 1D, $W_p$ reduces to sorting — making each slice $O(n \log n)$.

The integral is approximated by Monte Carlo over $L$ random directions:

$$
\widehat{\text{SW}}_2(\mu,\nu) \;\approx\; \left(\frac{1}{L}\sum_{\ell=1}^{L} W_2^2(\theta_\ell{}_\#\mu,\;\theta_\ell{}_\#\nu)\right)^{1/2}, \quad \theta_\ell \sim \mathcal{U}(\mathbb{S}^{d-1})
$$

Error decays at rate $O(L^{-1/2})$ — standard Monte Carlo.

---

## Pseudocode

### Gaussian W2 (Closed-Form)

```
function gaussian_W2(m1, Σ1, m2, Σ2):
    mean_sq  = ||m1 - m2||²
    S1_half  = eig_sqrt(Σ1)
    M        = S1_half @ Σ2 @ S1_half
    M_half   = eig_sqrt(M)
    bures_sq = tr(Σ1) + tr(Σ2) - 2·tr(M_half)

    return sqrt(max(mean_sq + bures_sq, 0))


function eig_sqrt(A):
    λ, V = eigh(A)
    λ    = clamp(λ, min=0)
    return V @ diag(√λ) @ V^T
```

### Sliced Wasserstein

```
function sliced_W2(X, Y, L=500, seed=0):
    # X, Y : (n, d) sample matrices
    rng  = RNG(seed)
    Θ    = rng.normal(shape=(L, d))
    Θ   /= ||Θ||₂  row-wise

    total = 0
    for θ in Θ:
        px     = sort(X @ θ)
        py     = sort(Y @ θ)
        total += mean((px - py)²)

    return sqrt(total / L)
```

---

## Unit Tests

```
assert gaussian_W2(m, Σ, m, Σ) ≈ 0
assert gaussian_W2(m1, Σ1, m2, Σ2) ≈ gaussian_W2(m2, Σ2, m1, Σ1)
assert gaussian_W2([0], [[1]], [3], [[1]]) ≈ 3.0
cf  = gaussian_W2(m1, Σ1, m2, Σ2)
swd = sliced_W2(sample(N(m1,Σ1), n=5000), sample(N(m2,Σ2), n=5000), L=1000)
assert |cf - swd| / cf < 0.05
assert gaussian_W2(m1, α²·Σ1, m2, α²·Σ2) ≈ α · gaussian_W2(m1, Σ1, m2, Σ2)
assert W2(A, C) ≤ W2(A, B) + W2(B, C)
```

---

## Potential Disadvantages

**Imaginary leakage** 

`scipy.linalg.sqrtm` returns complex matrices with tiny imaginary parts on near-singular inputs. Resolved by switching to `eigh` + clamp.

**SWD test flakiness**

At $L=50$ projections, variance is ~10% of the mean; tight tolerances fail randomly. Fixed by seeding RNG in tests and using 5% relative tolerance.

**Negative $W_2^2$**

Floating-point error in `tr(M_half)` can slightly overshoot, making the sum negative. `max(..., 0)` before the final `sqrt` is necessary.

---

## Next Steps

**Vectorize** 

`gaussian_W2` over batches of $(\Sigma_1, \Sigma_2)$ pairs — currently loops in Python.

**Gradient check**

Numerical Jacobian vs. autograd through `eigh`; known edge case when eigenvalues coincide.

**Quasi-Monte Carlo projections**

Replace uniform random $\theta_\ell$ with a Sobol sequence on $\mathbb{S}^{d-1}$; expect $O(L^{-1})$ convergence vs. $O(L^{-1/2})$.

**Integration smoke test**

Gradient-descend two free Gaussians toward each other using $W_2$ as loss; verify convergence.

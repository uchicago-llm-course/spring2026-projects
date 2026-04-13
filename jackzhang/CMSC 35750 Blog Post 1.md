# Investigating Koopman Autoencoders on Constructed Dynamical States

This week's main task is to investigate the **Koopman Autoencoder (KAE)** on the constructed dynamical states. The core insights are:

1. Compared with the existing EDMD method, KAE is computationally scalable and able to both extract nonlinear features and uncover nonlinear bases that reveal further insights into the dynamics of the residual stream activations token-wise.
2. However, it faces challenges around data scarcity and direct interpretability.

## Motivation

Recall that $H_k$ is defined as the delayed (Hankel) window of last-layer post-MLP activations. The purpose of using a KAE is that it allows learning the Koopman operator on a compact, nonlinear basis of low dimensionality, rather than relying on the identity function as the observable.

This has several benefits:

1. **Scalability.** KAE bypasses the large-scale matrix factorization required by EDMD, making the eigendecomposition for spectral analysis trivial. It also removes the need for the grid search over EDMD's regularization parameters.
2. **Nonlinear observables.** KAE learns a nonlinear observable that can potentially capture dynamics that hand-selected linear bases (coordinate, PCA) would miss.
3. **Tolerance to small Hankel windows.** KAE may compensate for small delay windows. The original LSD paper adopts a small window size, which may be insufficient to reconstruct a Markovian state. A learned nonlinear observable can extract richer features from each individual activation, potentially loosening the requirement for large window sizes.

## How a Koopman Autoencoder Works

A Koopman Autoencoder, as introduced by Aswani and Jabari (2025), consists of three components:

1. An **encoder** $\phi: \mathbb{R}^{Wd} \rightarrow \mathbb{R}^m$ that serves as the *observable*, replacing the basis function used in EDMD.
2. A learnable matrix $K$ that serves as the **approximated Koopman operator** acting on the lifted, linear dynamics.
3. A **decoder** $\psi: \mathbb{R}^m \rightarrow \mathbb{R}^{Wd}$ that reconstructs the Hankel window states.

![Koopman Autoencoder diagram](media/image1.png)

The model is trained by minimizing a combined objective:

1. **Reconstruction:** $\left\|H_k - \psi(\phi(H_k))\right\|^2$ — prevents degenerate representations.
2. **One-step linear prediction:** $\left\|\phi(H_{k+1}) - K\,\phi(H_k)\right\|^2$ — forces dynamics to be linear in latent space.
3. **Multi-step rollout:** $\sum_n \left\|\phi(H_{k+n}) - K^n\,\phi(H_k)\right\|^2$ — stabilizes eigenvalues over longer horizons.

This formulation follows the Koopman autoencoder framework explored by Aswani and Jabari (2025), which treats neural representations themselves as states evolving under a learnable linear operator in observable space.

## Challenges

While KAE offers the benefits above, it also introduces a new set of challenges:

1. **Data scarcity.** Under a deep learning paradigm, $P=200$ pairs of truthful/untruthful instruction prompts no longer suffice. One simple remedy is to expand both the number of pairs and the maximum token cap during generation, but this implies heavier inference and more GPU time — a trade-off between CPU RAM and GPU compute. An alternative is to keep the MLP parameters small, which sacrifices the expressivity of the learned observable.
2. **Loss of direct interpretability.** Both the bases and the Koopman matrix are learned, and the dynamics unfold in latent space. Even if the model truly captures nonlinear Koopman dynamics with distinct eigendecompositions for each generative regime (truthful vs. untruthful), it remains an open question how much we should trust this latent perspective.

## Implementation

We implement KAE on the same prompt regime, with prompt pairs $P=2{,}000$, window size $W=5$, and maximum token cap $L_p=32$. We set $m=64$, the approximate upper bound of the estimated effective rank reported in the paper.

### 1. Linear encoder/decoder

- $W_{\text{enc}} \in \mathbb{R}^{64 \times 25{,}600}$, $W_{\text{dec}} \in \mathbb{R}^{25{,}600 \times 64}$

**Parameter count:**

- Encoder: $25{,}600 \times 64 = 1{,}638{,}400$
- Decoder: $64 \times 25{,}600 = 1{,}638{,}400$
- Koopman matrix $K$: $64 \times 64 = 4{,}096$
- **Total: 3,280,896**
- **Parameter-to-data ratio:** $3.28\text{M} / 108\text{K} \approx 30{:}1$ — acceptable for a linear structure.

### 2. Nonlinear encoder/decoder

- Preprocess $h_k$ from 5,120 to 512 dimensions via PCA before Hankel construction. Input dimension becomes $5 \times 512 = 2{,}560$.
- **Architecture:** $2{,}560 \rightarrow 128$ (ReLU) $\rightarrow 64$, with a symmetric decoder.

**Parameter count:**

- Encoder: $2{,}560 \times 128 + 128 \times 64 = 327{,}680 + 8{,}192 = 335{,}872$
- Decoder: $64 \times 128 + 128 \times 2{,}560 = 8{,}192 + 327{,}680 = 335{,}872$
- Koopman matrix $K$: $4{,}096$
- **Total: 675,840**
- **Parameter-to-data ratio:** $676\text{K} / 108\text{K} \approx 6.3{:}1$ — comfortable.

## Next Steps

The next step is to complete the inference pass over the weekend and run the pipeline end-to-end. I will then compare the spectral analysis results from both Koopman operators against EDMD.

The baseline expectation: if the **linear KAE** recovers the dynamics with its learned compression basis, then this is a direct scale-up of the original coordinate-basis approach, achieved by bypassing the large-scale matrix factorization. If the **nonlinear KAE** yields nearly identical reconstruction MSE relative to the linear setup but reveals more sharply distinct spectral signatures between generative regimes, this would suggest two things:

1. There is nonlinear structure in the dynamics that linear methods underestimate.
2. A *dynamical* view of token-wise activation evolution is an essential angle, since it exposes properties of nonlinear dynamics that static snapshots cannot capture.

---

## Reference

Aswani, N. S., & Jabari, S. E. (2025). *Koopman Autoencoders Learn Neural Representation Dynamics*. arXiv:2505.12809. <https://arxiv.org/abs/2505.12809>

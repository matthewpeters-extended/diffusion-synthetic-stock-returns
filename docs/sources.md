# Sources

## Where the idea comes from

**Diffusion Models in Finance (awesome list), section "Financial Time Series Data → Unconditional Generation"**
https://github.com/ZhuoHan1998/Diffusion-Models-In-Finance

This project is original code inspired by that section. It does not port any single paper's repo.

The list's scope note, verbatim:

> Included papers should use a learned diffusion/flow generator, or make a finance-specific contribution that directly evaluates such models. Classical financial "diffusion" in the SDE/PDE sense is not included unless the paper also contains a learned generative diffusion or flow model.

That exclusion is why GBM is the baseline here. GBM is the classical SDE "diffusion" the list leaves out. The project asks whether a *learned* diffusion model beats it on the stylized facts in SCOPE.md.

## Closest prior art (from that section)

**"Generation of Synthetic Financial Time Series by Diffusion Models"**, Quantitative Finance 2025.
https://arxiv.org/abs/2410.18897
This is the closest match: a diffusion model generating US stock return series, judged on stylized facts. The list tags it "US / Stock / Return & Volume series". This project is a smaller version: returns only, a 1D denoiser, and fixed 64-day windows.

**"A diffusion-based generative model for financial time series via geometric Brownian motion"**, arXiv 2025.
https://arxiv.org/abs/2507.19003
This paper brings GBM into the learned diffusion process itself. Here GBM is only the naive baseline, and the two roles should not be confused.

Also listed there, and not used directly: "Diffusion Factor Models: Generating High-Dimensional Returns with Factor Structure" (arXiv 2025, https://arxiv.org/abs/2504.06566). It models cross-asset factor structure, which is a non-goal here.

## Data

Daily adjusted close prices from Yahoo Finance via `yfinance`. See SCOPE.md for the tickers and dates.

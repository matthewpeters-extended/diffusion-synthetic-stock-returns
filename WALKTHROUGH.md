# Walkthrough

A prose tour for anyone who doesn't want to run code. The two notebooks in [notebooks/](notebooks/) show the same steps with plots.

## 1. What real returns look like

Daily log returns for five liquid US tickers behave very differently from the textbook model. Measured on 2005–2019 and 2020–2024:

- **Fat tails.** Extreme days are far more common than a bell curve predicts. Excess kurtosis is 13.2 in the training years and 8.4 in the test years (a normal distribution gives 0).
- **No simple linear predictability.** Today's return says almost nothing about tomorrow's. It is not exactly zero: the lag-1 autocorrelation is about -0.07 to -0.09, which is why the project judges the model against real data's own level rather than against a perfect zero.
- **Volatility clustering.** Calm and wild periods come in streaks. The autocorrelation of |return| is about 0.25 on average and fades slowly over 20 days.
- **Negative skew.** Weak in the training years (-0.03), clearer in the test years (-0.20).

Notebook `01_explore_real_data.ipynb` plots all four.

## 2. Setting up a fair test

- Each generated sample is a 64-day window of returns.
- Each ticker is divided by its own training-period standard deviation, a single constant per ticker. Scaling each window on its own would erase the calm-versus-wild contrast the project measures.
- Train on 2005–2019. Test on 2020–2024, which includes the COVID crash. No window crosses the boundary.
- Test windows don't overlap, so the statistics aren't inflated by reuse.

## 3. The baseline

GBM draws every day's return independently from one normal distribution fitted to the training data. It cannot produce fat tails, skew or volatility clustering. It gets the overall scale right, which turns out to matter.

## 4. The model

A standard denoising diffusion model. It learns to take a window of pure noise and remove the noise step by step until a return window remains. The network is a stack of dilated 1D convolutions: the dilations (1 to 32) let each output see the whole 64-day window, which a volatility-aware model needs. It is unconditional, with no latent variable and nothing exotic.

Training took about 18 minutes on a laptop CPU (20,000 steps). Training on CPU was slow until the convolutions were rewritten as matrix multiplies, which made them about ten times faster on this macOS build of PyTorch.

## 5. What happened

Notebook `02_diffusion_vs_gbm.ipynb` overlays real, GBM and diffusion samples. The short version, with the numbers in [docs/RESULTS.md](docs/RESULTS.md):

- On **volatility clustering, fat tails and skew**, the diffusion model is closer to real than GBM.
- It is **too quiet**: its returns have about 60% of the real spread. That makes it lose to GBM on overall distance to the real data.
- It learned that volatility persists, but only at about a fifth of the real strength.

Three more training configurations were tried (longer, higher learning rate, wider network). They traded some problems for others and none fixed the scale. The reported run was picked using training-period data only, so the holdout didn't influence the choice.

## 6. What would come next

Ideas, none done here: a scale-aware loss or sampler fix for the too-quiet samples, many seeds to see run-to-run variation, longer windows to test long-memory decay, and a conditional model that knows the volatility regime.

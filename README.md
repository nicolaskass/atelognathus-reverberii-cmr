# atelognathus-reverberii-cmr

Reproducibility package for the manuscript

> **Population size and apparent survival of *Atelognathus reverberii* (Cei, 1969) estimated from a Bayesian Pollock's Robust Design capture–mark–recapture study, Somuncurá Plateau, Patagonia**
> Kass N. A., Kass C. A., Tettamanti G., Kacoliris F. P., Williams J. D.

This repository contains the capture-history data, analysis scripts, and key outputs needed to reproduce all population-size and survival estimates reported in the paper. The full manuscript is published separately; see citation below once the paper is in press.

---

## What this contains

```
atelognathus-reverberii-cmr/
├── data/
│   ├── capture_history.csv          ← 275 individuals × 14 occasions
│   └── README_data.md               ← column key and study-site notes
├── scripts/
│   ├── cmr_robust_design.py         ← Bayesian Robust Design MCMC + MAP + figures
│   ├── extended_analysis.py         ← PPC, M_h test, shrinkage, WAIC SE,
│   │                                  alternative-prior re-fit, N̂_Feb bias simulation
│   └── generate_reportables.py      ← compiles a JSON of all reportable quantities
├── outputs/                         ← pre-computed outputs from seed = 2024
│   ├── cmr_posterior_summary.csv    ← posterior median, SD, 95% CrI, R̂, ESS
│   ├── reportable_quantities.json   ← design counts, posteriors, MLE, closure test
│   ├── extended_diagnostics.json    ← PPC p-values, capture-freq tests, shrinkage,
│   │                                  alt-prior comparison, bias-simulation curve
│   ├── fig_cmr_posterior.png        ← marginal posteriors for all 11 quantities
│   ├── fig_cmr_N_time.png           ← N̂_t across primary sessions
│   ├── fig_cmr_prior_posterior.png  ← prior vs posterior overlays
│   ├── fig_capture_freq_dist.png    ← observed vs M_0 expectation per session
│   └── fig_N_feb_bias_sim.png       ← bias in N̂_Feb under within-session emigration
├── requirements.txt
├── LICENSE                          ← MIT
└── .gitignore
```

---

## How to reproduce

### 1. Set up the environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.11, numpy 1.26, scipy 1.11, matplotlib 3.8.

### 2. Run the primary Bayesian fit

```bash
python scripts/cmr_robust_design.py \
    --data data/capture_history.csv \
    --outdir outputs \
    --chains 4 --warmup 3000 --samples 25000 --seed 2024
```

This will (over)write `outputs/cmr_posterior_summary.csv` and the three core figures (`fig_cmr_posterior.png`, `fig_cmr_N_time.png`, `fig_cmr_prior_posterior.png`). Runtime: ~3–5 min on a modern laptop (Adaptive Metropolis–Hastings, pure NumPy).

### 3. Compile reportable quantities

```bash
python scripts/generate_reportables.py
```

Reads the CSV from step 2 plus the data file, emits `outputs/reportable_quantities.json` containing every numerical claim in the manuscript (medians, credible intervals, R̂, ESS, history-pattern counts, MLE references, closure-test statistics).

### 4. Run extended diagnostics

```bash
python scripts/extended_analysis.py
```

Runs (and caches) two independent MCMC samples — under the primary `Normal(2, 2)` prior on `logit(φ)` and under an alternative `Logistic(0, 1)` flat prior — and computes:
- Posterior predictive check on within-session total captures K_t (Bayes p-values).
- Within-session capture-frequency χ² test against the M_0 expectation.
- Prior–posterior shrinkage on the logit scale for each parameter.
- WAIC with standard error.
- Comparison of φ_Oct→Nov and φ_Nov→Feb under the two priors.
- Simulation-based bias bound on N̂_Feb under within-session emigration.

Outputs `outputs/extended_diagnostics.json`, `fig_capture_freq_dist.png`, `fig_N_feb_bias_sim.png`. Runtime: ~7–10 min (two MCMC re-runs).

### Selectively re-running parts

`extended_analysis.py` accepts `--only` with any comma-separated subset of `ppc, mh, shrink, waic, altprior, biasN`, e.g.:

```bash
python scripts/extended_analysis.py --only ppc,shrink
```

The MCMC samples are cached in `outputs/posterior_samples_{primary,altprior}.npz` (gitignored). Use `--force` to re-run from scratch.

---

## Headline results (with seed = 2024)

| Parameter | Median | 95% CrI | MLE (`Rcapture::robustd.0`) |
|---|---|---|---|
| φ(Oct → Nov) | 0.889 | [0.518, 0.997] | 0.793 ± 0.236 |
| φ(Nov → Feb) | 0.180 | [0.097, 0.315] | 0.165 ± 0.051 |
| N̂_Oct | 448 | [228, 1,206] | 474 ± 222 |
| N̂_Nov | 738 | [514, 1,094] | 724 ± 166 |
| N̂_Feb | 201 | [154, 283] | 193 ± 32 |

φ(Oct → Nov) is reported here for completeness but is prior-dominated and should not be interpreted as a biological survival estimate (prior–posterior shrinkage on the logit scale = 0.40 vs. 0.97 for the November–February interval); see the manuscript Results section *Model adequacy and prior sensitivity* for details.

---

## Implementation notes

The Bayesian Robust Design model is implemented in plain Python (`scipy`/`numpy`), with adaptive Metropolis–Hastings sampling. A maximum-likelihood reference fit using `robustd.0()` from the R package `Rcapture` was computed externally and is included for cross-validation (the MLE column in the table above; see also Table 1 of the manuscript).

The implementation follows the parameterisation of Pollock's Robust Design summarised in Kéry & Schaub (2012, *Bayesian Population Analysis using WinBUGS*) and described for JAGS in Rankin et al. (2016, *Frontiers in Marine Science*) and Riecke et al. (2018, *Methods in Ecology and Evolution*).

---

## License

Code and documentation: MIT (see `LICENSE`).
Data: released under the same terms; if you re-use the capture history, please cite the paper.

---

## Citation

```
Kass, N. A., Kass, C. A., Tettamanti, G., Kacoliris, F. P., & Williams, J. D.
(in review). Population size and apparent survival of Atelognathus reverberii
(Cei, 1969) estimated from a Bayesian Pollock's Robust Design
capture–mark–recapture study, Somuncurá Plateau, Patagonia.
Ichthyology & Herpetology.
```

Repository: <https://github.com/nicolaskass/atelognathus-reverberii-cmr>
A reserved Zenodo DOI for archived releases will be added at journal acceptance.

---

## Contact

Nicolás Ariel Kass — Sección Herpetología, División Zoología Vertebrados,
Facultad de Ciencias Naturales y Museo, Universidad Nacional de La Plata (UNLP),
La Plata, Argentina.
nicolaskass@gmail.com

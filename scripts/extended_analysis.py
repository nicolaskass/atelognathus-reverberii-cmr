#!/usr/bin/env python3
"""extended_analysis.py — Diagnostic and prior-sensitivity analyses for the CMR model.

Computes for Paper 2 (Bayesian Pollock's Robust Design CMR):
  B1.1  Posterior predictive check (M3 of the external review)
  B1.2  Within-session capture frequency distribution / M_h test (M4)
  B1.3  Prior-posterior shrinkage on logit scale (M1)
  B1.4  WAIC with standard error (Mo1)
  B1.5  Re-run with alternative flat prior Logistic(0,1) (M1 + Mo2)
  B2    Simulation-based bias bound on N̂_Feb (M2)

Reads:
  - data/capture_history.csv

Writes:
  - outputs/cmr/extended_diagnostics.json
  - outputs/cmr/cmr_posterior_summary_altprior.csv      (B1.5)
  - outputs/cmr/fig_capture_freq_dist.png               (B1.2)
  - outputs/cmr/fig_N_feb_bias_sim.png                  (B2)
  - outputs/cmr/fig_prior_alt_comparison.png            (B1.5)

Usage:
    python scripts/extended_analysis.py
    python scripts/extended_analysis.py --only ppc,shrink
    python scripts/extended_analysis.py --quick   # n_sample reduced for development
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import expit as sigmoid, logit

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
REPO_DIR = HERE.parent
DATA_PATH = REPO_DIR / "data" / "capture_history.csv"
OUT_DIR = REPO_DIR / "outputs"
CACHE_NPZ = OUT_DIR / "posterior_samples_cache.npz"

sys.path.insert(0, str(HERE))
from cmr_robust_design import (  # noqa: E402
    load_data, run_mcmc, transform, log_posterior,
    J, T, STARTS, LABELS_SHORT, PARAM_NAMES,
)


# ── Sampling cache: reuse MCMC across diagnostics ────────────────────────────

def get_samples(data, n_chains=4, n_warmup=3000, n_sample=25000,
                seed=2024, force=False, prior_alt=False):
    """Return (samps, n_t). Caches to .npz to avoid re-running MCMC."""
    tag = "altprior" if prior_alt else "primary"
    cache = OUT_DIR / f"posterior_samples_{tag}.npz"
    if cache.exists() and not force:
        print(f"  → using cached samples from {cache.name}")
        d = np.load(cache)
        return d["samps"], d["n_t"]

    if prior_alt:
        print("  → re-running MCMC with Logistic(0,1) prior on logit(φ)")
        samps, _, _ = run_mcmc_altprior(
            data, n_chains=n_chains, n_warmup=n_warmup,
            n_sample=n_sample, seed=seed,
        )
    else:
        samps, _, _ = run_mcmc(
            data, n_chains=n_chains, n_warmup=n_warmup,
            n_sample=n_sample, seed=seed,
        )

    np.savez_compressed(cache, samps=samps, n_t=data["n_t"])
    print(f"  → cached to {cache.name}")
    return samps, data["n_t"]


# ── B1.5  MCMC with Logistic(0,1) prior on logit(φ) ──────────────────────────

def log_posterior_altprior(theta, data):
    """Same as log_posterior but with Logistic(0,1) prior on logit(φ).

    Logistic(0,1) on the logit corresponds to Uniform(0,1) on the probability
    scale (a genuinely uninformative prior on the probability scale).
    For logit(p) we keep the Normal(-2, 1.5) prior unchanged.
    """
    eps = 1e-12
    phi01 = sigmoid(theta[0]); phi12 = sigmoid(theta[1])
    p = sigmoid(theta[2:5])
    n_t = data['n_t']; K_t = data['K_t']; JNK = data['JNK']
    patterns = data['patterns']

    pd_ = np.clip(1.0 - (1.0 - p)**J, eps, 1 - eps)
    p = np.clip(p, eps, 1 - eps)

    # Logistic(0,1) log-pdf on the logit: -log(1+exp(theta)) - log(1+exp(-theta))
    # = -|theta| - 2*log(1 + exp(-|theta|))   (numerically stable)
    def log_logistic(x):
        return -np.abs(x) - 2 * np.log1p(np.exp(-np.abs(x)))

    lp = log_logistic(theta[0]) + log_logistic(theta[1])
    lp += stats.norm.logpdf(theta[2], -2.0, 1.5)
    lp += stats.norm.logpdf(theta[3], -2.0, 1.5)
    lp += stats.norm.logpdf(theta[4], -2.0, 1.5)

    lp += float(np.sum(K_t*np.log(p) + JNK*np.log(1-p) - n_t*np.log(pd_)))

    chi2 = 1.0
    chi1 = float(np.clip(1.0 - phi12*pd_[2], eps, 1.0))
    chi0 = float(np.clip((1-phi01) + phi01*(1-pd_[1])*chi1, eps, 1.0))
    lp01 = np.log(phi01+eps); lp12 = np.log(phi12+eps)
    lpd_ = np.log(pd_); l1pd = np.log(1-pd_+eps)
    lchi = [np.log(chi0), np.log(chi1), np.log(chi2)]

    for (y0, y1, y2), cnt in patterns.items():
        if cnt == 0:
            continue
        ys = [y0, y1, y2]
        f = next(t for t, y in enumerate(ys) if y == 1)
        l = max(t for t, y in enumerate(ys) if y == 1)
        ll = 0.0
        for t in range(f + 1, l + 1):
            ll += (lp01 if t == 1 else lp12)
            ll += lpd_[t] if ys[t] == 1 else l1pd[t]
        ll += lchi[l]
        lp += cnt * ll
    return float(lp)


def run_mcmc_altprior(data, n_chains=4, n_warmup=3000, n_sample=25000, seed=2024):
    """Adaptive MH MCMC under the alternative flat prior."""
    from cmr_robust_design import find_map, run_chain
    # find_map uses default log_posterior — for altprior we re-find MAP via grid
    # Simplification: start chains from previous MAP perturbed (close enough)
    map_th, _ = find_map(data, seed=seed)

    # Replace log_posterior reference inside run_chain via monkey-patch
    import cmr_robust_design as crd
    original_lp = crd.log_posterior
    crd.log_posterior = log_posterior_altprior
    try:
        rng = np.random.default_rng(seed)
        all_s, all_lp = [], []
        for c in range(n_chains):
            th0 = map_th + rng.normal(0, 0.15, 5)
            s, lp_, ar = run_chain(th0, data, n_warmup + n_sample, seed=seed + c + 1)
            all_s.append(s[n_warmup:]); all_lp.append(lp_[n_warmup:])
            print(f"  Chain {c+1}: acceptance={ar:.3f}")
        return np.stack(all_s), np.stack(all_lp), map_th
    finally:
        crd.log_posterior = original_lp


# ── B1.1  Posterior predictive check ─────────────────────────────────────────

def simulate_within_session_K(theta, data, rng):
    """Simulate total within-session captures K_t conditional on observed n_t.

    Under model M_0 each detected individual has J_t Bernoulli(p_t) trials with
    the constraint of at least one capture (zero-truncated binomial).
    """
    p = sigmoid(theta[2:5])
    K_t_sim = np.zeros(T, dtype=int)
    n_t = data['n_t']
    for t in range(T):
        # Zero-truncated binomial counts per detected individual
        counts = np.zeros(n_t[t], dtype=int)
        for i in range(n_t[t]):
            while True:
                c = rng.binomial(J[t], p[t])
                if c >= 1:
                    counts[i] = c
                    break
        K_t_sim[t] = int(counts.sum())
    return K_t_sim


def posterior_predictive_check(samps, data, n_pp=2000, seed=2024):
    """Bayesian p-value on within-session total captures K_t."""
    rng = np.random.default_rng(seed)
    flat = samps.reshape(-1, samps.shape[-1])
    idx = rng.choice(len(flat), size=min(n_pp, len(flat)), replace=False)
    sub = flat[idx]

    K_obs = data['K_t']
    K_sim = np.zeros((len(sub), T), dtype=int)
    for i, theta in enumerate(sub):
        K_sim[i] = simulate_within_session_K(theta, data, rng)

    # Bayesian p-value (two-sided: probability sim is at least as extreme)
    # Common practice: P(T_rep >= T_obs)
    p_per = [float(np.mean(K_sim[:, t] >= K_obs[t])) for t in range(T)]
    K_total_obs = int(np.sum(K_obs))
    K_total_sim = K_sim.sum(axis=1)
    p_total = float(np.mean(K_total_sim >= K_total_obs))

    return dict(
        K_obs_per_session=[int(x) for x in K_obs],
        K_obs_total=K_total_obs,
        K_sim_mean_per_session=[float(K_sim[:, t].mean()) for t in range(T)],
        K_sim_total_mean=float(K_total_sim.mean()),
        bayes_p_per_session=p_per,
        bayes_p_total=p_total,
        n_pp_draws=int(len(sub)),
        interpretation=(
            "Bayes p-values close to 0.5 indicate adequate fit; values close to "
            "0 or 1 indicate posterior predictions systematically disagree with "
            "the data. Conventional cutoff for concern: p < 0.05 or p > 0.95."
        ),
    )


# ── B1.3  Prior-posterior shrinkage on logit scale ───────────────────────────

def shrinkage_diagnostic(samps, prior_alt=False):
    """1 - Var(posterior_logit) / Var(prior_logit) for each of the 5 logits.

    Values near 1 indicate the data dominates; values near 0 indicate the
    posterior on the logit scale has barely been updated from the prior.
    """
    flat = samps.reshape(-1, samps.shape[-1])
    post_var = flat.var(axis=0)
    # Prior variances on logit scale
    if prior_alt:
        # Logistic(0,1) variance = π²/3 ≈ 3.29
        prior_var = np.array([np.pi**2 / 3, np.pi**2 / 3, 1.5**2, 1.5**2, 1.5**2])
    else:
        prior_var = np.array([2.0**2, 2.0**2, 1.5**2, 1.5**2, 1.5**2])
    shrinkage = 1.0 - (post_var / prior_var)
    return {
        "phi_01": float(shrinkage[0]),
        "phi_12": float(shrinkage[1]),
        "p_oct": float(shrinkage[2]),
        "p_nov": float(shrinkage[3]),
        "p_feb": float(shrinkage[4]),
        "interpretation": (
            "Shrinkage = 1 - Var(posterior)/Var(prior) on logit scale. "
            "Values > 0.5 indicate data-dominated posterior; values < 0.3 "
            "indicate parameter is prior-dominated."
        ),
    }


# ── B1.4  WAIC with standard error ───────────────────────────────────────────

def compute_log_lik_per_pattern(samps, data):
    """log p(history pattern | theta) for each posterior draw and each pattern."""
    eps = 1e-12
    flat = samps.reshape(-1, samps.shape[-1])
    pat_keys = list(data["patterns"].keys())
    n_patterns = len(pat_keys)
    log_lik = np.zeros((len(flat), n_patterns))

    for i, theta in enumerate(flat):
        phi01 = sigmoid(theta[0]); phi12 = sigmoid(theta[1])
        p = sigmoid(theta[2:5])
        pd_ = np.clip(1.0 - (1.0 - p)**J, eps, 1 - eps)
        chi2 = 1.0
        chi1 = float(np.clip(1.0 - phi12*pd_[2], eps, 1.0))
        chi0 = float(np.clip((1-phi01) + phi01*(1-pd_[1])*chi1, eps, 1.0))
        lp01 = np.log(phi01 + eps); lp12 = np.log(phi12 + eps)
        lpd_ = np.log(pd_); l1pd = np.log(1 - pd_ + eps)
        lchi = [np.log(chi0), np.log(chi1), np.log(chi2)]
        for j, key in enumerate(pat_keys):
            ys = [int(x) for x in key]
            f = next(t for t, y in enumerate(ys) if y == 1)
            l = max(t for t, y in enumerate(ys) if y == 1)
            ll = 0.0
            for t in range(f + 1, l + 1):
                ll += (lp01 if t == 1 else lp12)
                ll += lpd_[t] if ys[t] == 1 else l1pd[t]
            ll += lchi[l]
            log_lik[i, j] = ll
    return log_lik, pat_keys


def compute_waic_se(samps, data):
    """WAIC = -2(lppd - pwaic) and its standard error via per-observation var.

    Following Vehtari, Gelman & Gabry (2017) formulation.
    """
    log_lik, pat_keys = compute_log_lik_per_pattern(samps, data)
    pat_counts = np.array([data["patterns"][k] for k in pat_keys], dtype=int)

    # Expand to per-observation (each pattern repeated pat_count times)
    log_lik_obs = np.repeat(log_lik, pat_counts, axis=1)
    n_obs = log_lik_obs.shape[1]

    # lppd per obs: log mean exp(log_lik[s, i])
    max_ll = log_lik_obs.max(axis=0)
    lppd_per_obs = max_ll + np.log(np.mean(np.exp(log_lik_obs - max_ll[None, :]), axis=0))
    # p_waic per obs: var over draws
    pwaic_per_obs = log_lik_obs.var(axis=0, ddof=1)

    elpd_per_obs = lppd_per_obs - pwaic_per_obs
    waic = -2.0 * float(np.sum(elpd_per_obs))
    # SE: sqrt(N * Var(elpd_per_obs))
    se = 2.0 * float(np.sqrt(n_obs * elpd_per_obs.var(ddof=1)))
    return dict(
        waic=waic,
        se=se,
        n_obs=int(n_obs),
        lppd=float(np.sum(lppd_per_obs)),
        p_waic=float(np.sum(pwaic_per_obs)),
    )


# ── B1.2  M_h test: within-session capture frequency distribution ────────────

def load_raw_matrix(filepath):
    """Return X[i, j] binary capture matrix (rows = individuals, cols = occasions)."""
    df = pd.read_csv(filepath)
    id_c = df.columns[0]
    dcol = [c for c in df.columns if c != id_c]

    def is1(v):
        return str(v).strip() == "1"

    X = np.array([[1 if is1(df[c].iloc[i]) else 0 for c in dcol]
                  for i in range(len(df))], dtype=int)
    return X


def capture_freq_distribution(data_path):
    """For each primary session, compute the count distribution of within-session
    captures among detected individuals.

    Returns dict per session with observed frequency (1, 2, ..., J_t captures)
    and a χ² test against the M_0 expectation conditional on n_t and p̂.
    """
    X = load_raw_matrix(data_path)
    out = {}
    for t in range(T):
        s, e = STARTS[t], STARTS[t] + J[t]
        sub = X[:, s:e]
        # Detected individuals = those with ≥1 capture in this session
        detected = sub[sub.sum(axis=1) >= 1]
        counts = detected.sum(axis=1)
        n_det = len(counts)
        # Observed distribution: how many individuals captured 1, 2, ..., J_t times
        obs = np.array([(counts == k).sum() for k in range(1, J[t] + 1)])

        # Expected distribution under M_0 (zero-truncated binomial)
        # First estimate p_t via MLE: solve for p maximizing zt-binomial
        # MLE p̂_t = K_t / (J_t * n_t) is naive; better: numerical MLE of zt-bin
        K_t = int(counts.sum())
        # Naive p̂ that closely matches the within-session estimator
        p_naive = K_t / (J[t] * n_det)
        # Zero-truncated binomial expected count for k captures (k=1..J)
        prob_k = stats.binom.pmf(np.arange(1, J[t] + 1), J[t], p_naive)
        prob_k = prob_k / prob_k.sum()  # normalize zt
        exp = n_det * prob_k

        # χ² test (collapse cells with expected < 5)
        mask = exp >= 5
        if mask.sum() < 2:
            chi2_stat, chi2_p, dof = np.nan, np.nan, 0
        else:
            obs_m = obs[mask]
            exp_m = exp[mask]
            # Renormalise within mask so totals match
            exp_m = exp_m * obs_m.sum() / exp_m.sum()
            chi2_stat = float(((obs_m - exp_m) ** 2 / exp_m).sum())
            dof = max(int(mask.sum()) - 2, 1)  # -1 for constraint, -1 for p̂
            chi2_p = float(1 - stats.chi2.cdf(chi2_stat, df=dof))

        out[LABELS_SHORT[t]] = {
            "n_detected": int(n_det),
            "K_total": int(K_t),
            "p_naive_mle": float(p_naive),
            "obs_counts_by_k": [int(x) for x in obs],
            "expected_M0_counts_by_k": [float(x) for x in exp],
            "chi2_stat": chi2_stat,
            "chi2_dof": int(dof),
            "chi2_p_value": chi2_p,
        }
    return out


def fig_capture_freq(freq_dict, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (session, d) in zip(axes, freq_dict.items()):
        ks = np.arange(1, len(d["obs_counts_by_k"]) + 1)
        width = 0.4
        ax.bar(ks - width/2, d["obs_counts_by_k"], width=width,
               color="#1b7837", alpha=0.85, label="Observed")
        ax.bar(ks + width/2, d["expected_M0_counts_by_k"], width=width,
               color="#762a83", alpha=0.85, label="Expected (M$_0$)")
        ax.set_xlabel("Within-session captures (k)")
        ax.set_ylabel("Number of individuals")
        chi_p = d["chi2_p_value"]
        chi_p_str = f"p = {chi_p:.3f}" if not np.isnan(chi_p) else "p = n/a"
        ax.set_title(f"{session}\nχ²({d['chi2_dof']}) = {d['chi2_stat']:.2f}, {chi_p_str}",
                     fontsize=10)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Within-session capture frequency: observed vs M$_0$ expectation",
                 fontsize=11)
    fig.tight_layout()
    p = outdir / "fig_capture_freq_dist.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="",
                        help="Comma-separated subset of: ppc, mh, shrink, waic, altprior, biasN")
    parser.add_argument("--quick", action="store_true",
                        help="Reduce n_sample for development")
    parser.add_argument("--force", action="store_true",
                        help="Re-run MCMC even if cache exists")
    args = parser.parse_args()

    sections = set(args.only.split(",")) if args.only else {
        "ppc", "mh", "shrink", "waic", "altprior", "biasN"
    }
    n_sample = 2000 if args.quick else 25000

    print(f"[setup] Loading data from {DATA_PATH.name}")
    data = load_data(DATA_PATH)

    out = {"_meta": {"n_sample": n_sample, "seed": 2024}}

    # Primary-prior samples (cached or freshly run)
    if {"ppc", "shrink", "waic", "biasN"} & sections:
        print(f"\n[setup] Sampling under primary prior (n_sample={n_sample:,}, seed=2024)")
        samps_pri, n_t = get_samples(
            data, n_sample=n_sample, force=args.force, prior_alt=False,
        )

    if "mh" in sections:
        print("\n[B1.2] M_h test: within-session capture frequency distribution")
        freq = capture_freq_distribution(DATA_PATH)
        for sess, d in freq.items():
            print(f"  {sess}: χ²={d['chi2_stat']:.2f} (dof {d['chi2_dof']}, p={d['chi2_p_value']:.3f})")
        out["capture_freq_distribution"] = freq
        fig_path = fig_capture_freq(freq, OUT_DIR)
        print(f"  → {fig_path.name}")

    if "ppc" in sections:
        print("\n[B1.1] Posterior predictive check")
        ppc = posterior_predictive_check(samps_pri, data)
        print(f"  Bayes p (total K): {ppc['bayes_p_total']:.3f}")
        for t in range(T):
            print(f"  Bayes p ({LABELS_SHORT[t]}): {ppc['bayes_p_per_session'][t]:.3f}  "
                  f"(obs={ppc['K_obs_per_session'][t]}, sim mean={ppc['K_sim_mean_per_session'][t]:.1f})")
        out["posterior_predictive"] = ppc

    if "shrink" in sections:
        print("\n[B1.3] Prior-posterior shrinkage (logit scale)")
        shrink = shrinkage_diagnostic(samps_pri, prior_alt=False)
        for k, v in shrink.items():
            if k != "interpretation":
                print(f"  shrinkage[{k}] = {v:.3f}")
        out["shrinkage"] = shrink

    if "waic" in sections:
        print("\n[B1.4] WAIC with SE")
        waic_res = compute_waic_se(samps_pri, data)
        print(f"  WAIC = {waic_res['waic']:.2f}  SE = {waic_res['se']:.2f}  "
              f"(n_obs={waic_res['n_obs']})")
        out["waic_with_se"] = waic_res

    if "altprior" in sections:
        print(f"\n[B1.5] MCMC with Logistic(0,1) alt prior on logit(φ)")
        samps_alt, _ = get_samples(
            data, n_sample=n_sample, force=args.force, prior_alt=True,
        )
        from cmr_robust_design import summarize as _summarize
        df_alt = _summarize(samps_alt, n_t)
        df_alt.to_csv(OUT_DIR / "cmr_posterior_summary_altprior.csv", index=False)
        # Quick comparison: φ_01 and φ_12 median + CrI
        flat_alt = samps_alt.reshape(-1, samps_alt.shape[-1])
        alt_phi01 = sigmoid(flat_alt[:, 0])
        alt_phi12 = sigmoid(flat_alt[:, 1])
        comparison = {
            "phi_01_alt": {
                "median": float(np.median(alt_phi01)),
                "ci_low": float(np.percentile(alt_phi01, 2.5)),
                "ci_high": float(np.percentile(alt_phi01, 97.5)),
            },
            "phi_12_alt": {
                "median": float(np.median(alt_phi12)),
                "ci_low": float(np.percentile(alt_phi12, 2.5)),
                "ci_high": float(np.percentile(alt_phi12, 97.5)),
            },
            "interpretation": (
                "Comparison of posteriors under Normal(2,2) primary vs Logistic(0,1) "
                "uniform-on-probability prior. If φ_01 changes substantially while "
                "φ_12 does not, this confirms φ_01 is prior-driven."
            ),
        }
        print(f"  φ_01 alt: median={comparison['phi_01_alt']['median']:.3f} "
              f"[{comparison['phi_01_alt']['ci_low']:.3f}, "
              f"{comparison['phi_01_alt']['ci_high']:.3f}]")
        print(f"  φ_12 alt: median={comparison['phi_12_alt']['median']:.3f} "
              f"[{comparison['phi_12_alt']['ci_low']:.3f}, "
              f"{comparison['phi_12_alt']['ci_high']:.3f}]")
        # Shrinkage under alt
        shrink_alt = shrinkage_diagnostic(samps_alt, prior_alt=True)
        out["alt_prior_comparison"] = comparison
        out["alt_prior_shrinkage"] = shrink_alt

    if "biasN" in sections:
        print("\n[B2] Simulation-based bias bound on N̂_Feb")
        bias = simulate_N_feb_bias(data, samps_pri if "samps_pri" in dir() else None,
                                   seed=2024)
        out["N_feb_bias_simulation"] = bias
        print(f"  ε=0  → N̂_Feb_sim mean = {bias['curve'][0]['N_hat_mean']:.0f}")
        print(f"  ε=0.4 → N̂_Feb_sim mean = {bias['curve'][4]['N_hat_mean']:.0f}")

    out_path = OUT_DIR / "extended_diagnostics.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nWrote {out_path}")


# ── B2  N̂_Feb bias simulation under within-session emigration ───────────────

def simulate_N_feb_bias(data, samps, seed=2024, n_sim=200):
    """For each ε in {0, 0.05, 0.1, 0.2, 0.4, 0.6}, simulate Feb session under
    random within-session emigration at rate ε, fit Horvitz-Thompson, report
    bias relative to a "true" N_Feb taken as the posterior median from samps.
    """
    rng = np.random.default_rng(seed)
    flat = samps.reshape(-1, samps.shape[-1])
    p_feb_post = sigmoid(flat[:, 4])
    p_feb_median = float(np.median(p_feb_post))

    # "Truth": N_Feb posterior median (Horvitz-Thompson)
    pd_feb_post = 1 - (1 - p_feb_post) ** J[2]
    N_feb_post = data["n_t"][2] / pd_feb_post
    N_true = int(round(np.median(N_feb_post)))

    eps_grid = [0.0, 0.05, 0.10, 0.20, 0.40, 0.60]
    curve = []
    for eps_rate in eps_grid:
        N_hats = []
        for _ in range(n_sim):
            # Simulate Feb session: N_true individuals; each may "exit" at some
            # secondary occasion j_exit ~ Geometric(eps_rate) within {1..J_feb}.
            # After exit, p_feb effectively 0 for that individual.
            counts = np.zeros(N_true, dtype=int)
            for i in range(N_true):
                if eps_rate > 0:
                    # Exit before occasion j with probability eps_rate per occasion
                    exit_at = J[2] + 1
                    for j in range(1, J[2] + 1):
                        if rng.random() < eps_rate:
                            exit_at = j
                            break
                else:
                    exit_at = J[2] + 1
                # Capture at occasion j ∈ [1, J_feb] only if not yet exited
                effective_J = min(J[2], exit_at - 1)
                if effective_J > 0:
                    counts[i] = rng.binomial(effective_J, p_feb_median)
            # Naive estimator: n_detected / pd̂
            detected = counts >= 1
            n_det_sim = int(detected.sum())
            if n_det_sim == 0:
                continue
            # Apply Horvitz-Thompson with the TRUE p_feb (i.e., assume p_feb is
            # estimated correctly from a large sample). This isolates the bias
            # introduced by within-session emigration alone, which is what we
            # want to quantify; the additional bias from p-estimation under
            # closure violation is a separate effect.
            pd_t = 1 - (1 - p_feb_median) ** J[2]
            N_hats.append(n_det_sim / pd_t)
        N_hats = np.array(N_hats)
        curve.append({
            "epsilon": float(eps_rate),
            "N_hat_mean": float(N_hats.mean()) if len(N_hats) else float("nan"),
            "N_hat_sd": float(N_hats.std()) if len(N_hats) else float("nan"),
            "N_hat_q025": float(np.percentile(N_hats, 2.5)) if len(N_hats) else float("nan"),
            "N_hat_q975": float(np.percentile(N_hats, 97.5)) if len(N_hats) else float("nan"),
            "n_sim_valid": int(len(N_hats)),
        })

    # Figure
    eps_arr = np.array([c["epsilon"] for c in curve])
    means = np.array([c["N_hat_mean"] for c in curve])
    lo = np.array([c["N_hat_q025"] for c in curve])
    hi = np.array([c["N_hat_q975"] for c in curve])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.fill_between(eps_arr, lo, hi, alpha=0.25, color="#525252",
                    label="95% simulation interval")
    ax.plot(eps_arr, means, "o-", color="#1b7837", lw=2, ms=8, label="Mean simulated $\\hat{N}_{Feb}$")
    ax.axhline(N_true, color="#d6604d", lw=1.5, ls="--",
               label=f"True $N_{{Feb}}$ = {N_true} (model posterior)")
    ax.set_xlabel(r"Within-session emigration rate $\varepsilon$ (per occasion)")
    ax.set_ylabel(r"Estimated $\hat{N}_{Feb}$ from naive Horvitz-Thompson")
    ax.set_title(r"Bias in $\hat{N}_{Feb}$ under within-session emigration")
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    p = OUT_DIR / "fig_N_feb_bias_sim.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "N_true_baseline": int(N_true),
        "p_feb_used": float(p_feb_median),
        "n_sim_per_epsilon": int(n_sim),
        "curve": curve,
        "interpretation": (
            "If frogs exit the detectable area within the session at rate ε, "
            "the naive Horvitz-Thompson estimator under-estimates the true "
            "session-resident N. The curve quantifies this bias: at ε=0.4, "
            "for instance, the estimator typically returns ~X% of the truth."
        ),
    }


if __name__ == "__main__":
    main()

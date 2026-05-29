#!/usr/bin/env python3
"""generate_reportables.py — Build reportable_quantities.json for the manuscript.

Reads:
  - outputs/cmr/cmr_posterior_summary.csv  (current MCMC posterior summary)
  - data/capture_history.csv               (capture history)

Writes:
  - outputs/cmr/reportable_quantities.json (consumed by ms/paper2_cmr.qmd)

Why a JSON layer?
  The .qmd chunks read all reportable numbers from this single JSON so that
  text and Table 1 cannot drift out of sync. Re-running the MCMC overwrites
  cmr_posterior_summary.csv; re-running this script overwrites the JSON;
  re-rendering the .qmd then propagates new values everywhere.

Static inputs (not produced by cmr_robust_design.py):
  - Stanley–Burnham closure test statistics (computed in R / external script).
  - WAIC for the comparison model M_phi(.)p(t) and its phi posterior summary.
  - MLE estimates from robustd.0 in Rcapture (R reference, frozen).
  - Prior-sensitivity MAP for phi_12 under the flat-prior alternative.

These are listed in STATIC_INPUTS below with clear provenance comments.

Usage:
    python scripts/generate_reportables.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_DIR = HERE.parent
DATA_PATH = REPO_DIR / "data" / "capture_history.csv"
CSV_PATH = REPO_DIR / "outputs" / "cmr_posterior_summary.csv"
JSON_PATH = REPO_DIR / "outputs" / "reportable_quantities.json"

sys.path.insert(0, str(HERE))
from cmr_robust_design import load_data, J, T, LABELS_SHORT  # noqa: E402


STATIC_INPUTS = {
    "closure_test": {
        # Stanley–Burnham (1999) within-session closure test. Computed in R
        # via Rcapture::closedp.t; pre-recorded here because the Python
        # implementation does not include it. Re-compute if data changes.
        "Oct": {"z": 0.60, "p_value": 0.547},
        "Nov": {"z": -1.92, "p_value": 0.055},
        "Feb": {"z": -2.28, "p_value": 0.023},
    },
    "model_comparison": {
        # WAIC for the primary and comparison models. Pre-computed externally
        # (the working Python script fits the primary model only).
        "waic_phi_t_p_t": 194.8,
        "waic_phi_dot_p_t": 197.6,
        "delta_waic": 2.8,
        "phi_constant_median": 0.355,
        "phi_constant_ci_low": 0.232,
        "phi_constant_ci_high": 0.535,
        "phi_constant_rhat": 1.002,
        "phi_constant_ess": 290,
    },
    "mle_rcapture": {
        # MLE from robustd.0(vm="M0", vt=c(4,5,5)) in Rcapture (v1.4.2).
        # Frozen reference; will not change unless we re-fit in R.
        "phi_01": {"estimate": 0.793, "se": 0.236},
        "phi_12": {"estimate": 0.165, "se": 0.051},
        "p_oct": {"estimate": 0.143, "se": 0.065},
        "p_nov": {"estimate": 0.200, "se": 0.044},
        "p_feb": {"estimate": 0.451, "se": 0.067},
        "N_oct": {"estimate": 474, "se": 222},
        "N_nov": {"estimate": 724, "se": 166},
        "N_feb": {"estimate": 193, "se": 32},
        "superpopulation": {"estimate": 895, "se": 125},
    },
    "prior_sensitivity": {
        # MAP of phi_12 under primary prior N(2,2) vs flat prior N(0,3).
        # Confirms robustness; both rounded to three decimals.
        "phi_12_map_primary": 0.180,
        "phi_12_map_flat": 0.171,
    },
    "mcmc": {
        "n_chains": 4,
        "n_warmup": 3000,
        "n_sample": 25000,
        "seed": 2024,
    },
}


def summarise_design(data: dict) -> dict:
    n_t = data["n_t"].tolist()
    K_t = data["K_t"].tolist()
    patterns = {",".join(str(int(x)) for x in k): int(v)
                for k, v in data["patterns"].items()}

    n_within_recap = sum(K_t) - sum(n_t)
    n_multi_session = sum(v for k, v in data["patterns"].items()
                          if sum(k) >= 2)
    n_oct_nov_both = sum(v for k, v in data["patterns"].items()
                         if k[0] == 1 and k[1] == 1)
    # within-session recapture rate, paper convention: extra captures per
    # individual = (K_t - n_t) / n_t. Distinct from naive per-occasion p.
    within_rate = [round((K_t[t] - n_t[t]) / n_t[t], 3) for t in range(T)]
    naive_p = [round(K_t[t] / (J[t] * n_t[t]), 3) for t in range(T)]

    return {
        "n_individuals": int(data["N_cap"]),
        "n_primary_sessions": int(T),
        "n_secondary_occasions": [int(j) for j in J],
        "n_total_occasions": int(sum(J)),
        "sessions": LABELS_SHORT,
        "n_per_session": [int(x) for x in n_t],
        "n_captures_per_session": [int(x) for x in K_t],
        "n_total_capture_events": int(sum(K_t)),
        "n_within_session_recaptures": int(n_within_recap),
        "n_multi_session_individuals": int(n_multi_session),
        "n_oct_nov_both_sessions": int(n_oct_nov_both),
        "within_session_recap_rate": within_rate,
        "naive_per_occasion_p": naive_p,
        "history_patterns": patterns,
    }


def summarise_posteriors(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    out: dict = {}
    for _, row in df.iterrows():
        out[row["Parameter"]] = {
            "mean": round(float(row["Mean"]), 4),
            "sd": round(float(row["SD"]), 4),
            "median": round(float(row["Median"]), 4),
            "ci_low": round(float(row["pct2_5"]), 4),
            "ci_high": round(float(row["pct97_5"]), 4),
            "rhat": round(float(row["Rhat"]), 4),
            "ess": int(row["ESS"]),
        }
    return out


def main() -> None:
    if not DATA_PATH.exists():
        sys.exit(f"Missing data file: {DATA_PATH}")
    if not CSV_PATH.exists():
        sys.exit(f"Missing posterior summary: {CSV_PATH}. "
                 "Run scripts/cmr_robust_design.py first.")

    print(f"[1/3] Loading capture history from {DATA_PATH.name}")
    data = load_data(DATA_PATH)

    print(f"[2/3] Loading posterior summary from {CSV_PATH.name}")
    posteriors = summarise_posteriors(CSV_PATH)

    print(f"[3/3] Writing {JSON_PATH.name}")
    payload = {
        "_meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_csv": str(CSV_PATH.relative_to(PAPER_DIR)),
            "source_data": str(DATA_PATH.relative_to(PAPER_DIR)),
            "schema_version": 1,
        },
        "design": summarise_design(data),
        "mcmc": STATIC_INPUTS["mcmc"],
        "posteriors": posteriors,
        "closure_test": STATIC_INPUTS["closure_test"],
        "model_comparison": STATIC_INPUTS["model_comparison"],
        "mle_rcapture": STATIC_INPUTS["mle_rcapture"],
        "prior_sensitivity": STATIC_INPUTS["prior_sensitivity"],
    }

    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nWrote {JSON_PATH}  ({JSON_PATH.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()

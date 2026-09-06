# =============================================================================
# PAPER REFERENCE
# Script:  scripts/ml/chapter9/dml_parent_clustered.py
# Paper:   "The Long Childhood: On the Convergence of Nations"
#
# Produces:
#   The parent-vantage DML estimate with (a) country-clustered
#   cross-fitting and (b) a real standard error, alongside the shipped
#   estimate so the two are directly comparable.
#
#   Why this script exists
#   ----------------------
#   `dml_parent.py` reports, for each outcome, a `theta_ci95` computed as
#   the 2.5 / 97.5 percentile of FIVE seed estimates:
#
#       "theta_ci95": [np.percentile(seed_means, 2.5),
#                      np.percentile(seed_means, 97.5)]
#
#   That interval describes how much the answer moves when the fold
#   assignment is reshuffled. It is not a confidence interval: it carries
#   no information about sampling uncertainty over countries, and with
#   n=5 its endpoints are essentially the min and max. The appendix reads
#   it as one ("every interval excludes zero"), which is an inferential
#   claim the quantity cannot support. Five seeds that agree tell you the
#   estimator is stable, not that theta differs from zero.
#
#   DML supplies its own variance. For the partially linear model with
#   orthogonal score psi_i = (Y_i - g(X_i) - theta * (D_i - m(X_i))) *
#   (D_i - m(X_i)), the asymptotic variance is
#
#       Var(theta) = E[psi^2] / (E[(D - m(X))^2])^2 / n
#
#   and in panel data the outer expectation must be clustered by country,
#   because residuals within a country are not independent. This script
#   computes that, clustered on country.
#
#   Second issue, same file: `KFold(shuffle=True)` splits ROWS. A
#   country's 1975 and 1980 observations land on opposite sides of the
#   cross-fit, so the nuisance models g and m can learn that country's
#   level from its own other years. Every other holdout in this
#   repository is country-level; this one was not. `GroupKFold` on
#   country fixes it, and the size of the resulting change in theta is
#   itself the diagnostic.
#
# Reports, per outcome:
#   theta under row-KFold (shipped) and under country-GroupKFold
#   clustered SE, clustered 95% CI, t-statistic
#   the shipped seed-percentile range, labelled as what it is
#
# Inputs:
#   scripts/ml/data_loader_parent_lag.load_parent_lag_panels
#
# Outputs:
#   scripts/ml/checkin/dml_parent_clustered.json
#
# Usage:
#   python scripts/ml/chapter9/dml_parent_clustered.py
# =============================================================================
"""Country-clustered DML with a real standard error."""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, GroupKFold

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(os.path.dirname(ML_DIR))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, ML_DIR)

from data_loader_parent_lag import (  # noqa: E402
    load_parent_lag_panels, PARENT_LAG_HORIZONS)
from spec_curve import flatten_panel, flatten_feature_groups  # noqa: E402

CHECKIN_DIR = os.path.join(ML_DIR, "checkin")
os.makedirs(CHECKIN_DIR, exist_ok=True)
OUT_PATH = os.path.join(CHECKIN_DIR, "dml_parent_clustered.json")

TARGETS = ["LE", "TFR", "U5MR"]
N_FOLDS = 5
SEEDS = [42, 43, 44, 45, 46]
EDU_GROUPS = ("wcde_education", "bl_education", "derived")

# Same nuisance learner as dml_parent.py, so any difference in theta is
# attributable to the split rule and not to the model.
GBM_KW = dict(n_estimators=150, max_depth=4, learning_rate=0.05)


def _split_indices(X_other, groups, splitter, seed):
    if splitter == "row_kfold":
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        return list(kf.split(X_other))
    if splitter == "country_groupkfold":
        # GroupKFold is deterministic, so vary the seed by permuting the
        # group labels: the partition changes, the country-purity does not.
        rng = np.random.default_rng(seed)
        uniq = np.unique(groups)
        remap = {c: i for c, i in zip(uniq, rng.permutation(len(uniq)))}
        shuffled = np.array([remap[g] for g in groups])
        gkf = GroupKFold(n_splits=N_FOLDS)
        return list(gkf.split(X_other, groups=shuffled))
    raise ValueError(splitter)


def _dml_one(panel, splitter, seed):
    """Cross-fitted orthogonal-score DML. Returns theta and per-row scores."""
    flat_groups = flatten_feature_groups(panel)
    X = flatten_panel(panel)
    y = panel["y"][:, 0]
    groups = np.asarray(panel["country_ids"])

    edu_idx = sorted({i for g in EDU_GROUPS for i in flat_groups.get(g, [])})
    non_edu_idx = [i for i in range(X.shape[1]) if i not in set(edu_idx)]

    D = X[:, edu_idx].mean(axis=1)      # aggregate education signal
    X_other = X[:, non_edu_idx]

    resid_y = np.zeros_like(y, dtype=float)
    resid_d = np.zeros_like(D, dtype=float)

    for tr_idx, te_idx in _split_indices(X_other, groups, splitter, seed):
        g_model = GradientBoostingRegressor(random_state=seed, **GBM_KW)
        g_model.fit(X_other[tr_idx], y[tr_idx])
        resid_y[te_idx] = y[te_idx] - g_model.predict(X_other[te_idx])

        m_model = GradientBoostingRegressor(random_state=seed, **GBM_KW)
        m_model.fit(X_other[tr_idx], D[tr_idx])
        resid_d[te_idx] = D[te_idx] - m_model.predict(X_other[te_idx])

    den = float((resid_d ** 2).sum())
    theta = float((resid_d * resid_y).sum() / max(den, 1e-12))

    # Orthogonal score at the estimated theta.
    psi = resid_d * (resid_y - theta * resid_d)
    return theta, psi, resid_d, groups


def _clustered_se(theta, psi, resid_d, groups):
    """Country-clustered SE for the partially linear DML estimator.

    Var(theta) = (sum_c (sum_{i in c} psi_i)^2) / (sum_i resid_d_i^2)^2

    The numerator sums the score WITHIN each country before squaring,
    which is what allows arbitrary within-country correlation. The
    unclustered version squares each row separately and understates the
    SE whenever residuals persist within a country, which here they do.
    """
    jac = float((resid_d ** 2).sum())
    if jac <= 0:
        return float("nan"), float("nan")

    meat_cluster = 0.0
    for c in np.unique(groups):
        meat_cluster += float(psi[groups == c].sum()) ** 2
    se_cluster = float(np.sqrt(meat_cluster) / jac)

    se_naive = float(np.sqrt((psi ** 2).sum()) / jac)
    return se_cluster, se_naive


def run_target(panel, target, seeds):
    out = {
        "horizon": PARENT_LAG_HORIZONS[target],
        "n_samples": int(panel["y"].shape[0]),
        "n_countries": int(len(np.unique(panel["country_ids"]))),
        "splitters": {},
    }
    for splitter in ("row_kfold", "country_groupkfold"):
        thetas, ses_c, ses_n = [], [], []
        for seed in seeds:
            theta, psi, resid_d, groups = _dml_one(panel, splitter, seed)
            se_c, se_n = _clustered_se(theta, psi, resid_d, groups)
            thetas.append(theta)
            ses_c.append(se_c)
            ses_n.append(se_n)

        theta_mean = float(np.mean(thetas))
        # Average the variances across seeds, then add the across-seed
        # variance of theta itself (Rubin-style), so the interval covers
        # both sampling and cross-fit-assignment uncertainty.
        se_within = float(np.sqrt(np.mean(np.array(ses_c) ** 2)))
        se_between = float(np.std(thetas, ddof=1)) if len(thetas) > 1 else 0.0
        se_total = float(np.sqrt(se_within ** 2 + se_between ** 2))

        out["splitters"][splitter] = {
            "theta_per_seed": thetas,
            "theta_mean": theta_mean,
            "theta_median": float(np.median(thetas)),
            "se_clustered_by_country": se_within,
            "se_unclustered": float(np.sqrt(np.mean(np.array(ses_n) ** 2))),
            "se_across_seeds": se_between,
            "se_total": se_total,
            "ci95_clustered": [theta_mean - 1.96 * se_total,
                               theta_mean + 1.96 * se_total],
            "t_stat": theta_mean / se_total if se_total > 0 else float("nan"),
            # What dml_parent.py reports as "ci95": the spread of the seed
            # estimates. Reproduced here so the two are comparable, and
            # labelled so it is not mistaken for inference.
            "seed_percentile_range_NOT_a_ci": [
                float(np.percentile(thetas, 2.5)),
                float(np.percentile(thetas, 97.5)),
            ],
        }
    row = out["splitters"]["row_kfold"]["theta_mean"]
    grp = out["splitters"]["country_groupkfold"]["theta_mean"]
    out["theta_shift_from_country_clustering"] = grp - row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--targets", nargs="+", default=TARGETS)
    args = ap.parse_args()
    seeds = SEEDS[:args.seeds] if args.seeds <= len(SEEDS) else \
        [SEEDS[0] + i for i in range(args.seeds)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        panels = load_parent_lag_panels(mode="joint", verbose=False)

    t0 = time.time()
    results = {}
    for target in args.targets:
        print(f"=== DML {target} @ T+{PARENT_LAG_HORIZONS[target]} ===")
        results[target] = run_target(panels[target], target, seeds)
        for splitter, r in results[target]["splitters"].items():
            print(f"  {splitter:22s} theta={r['theta_mean']:+.4f}  "
                  f"SE(clustered)={r['se_clustered_by_country']:.4f}  "
                  f"SE(naive)={r['se_unclustered']:.4f}  "
                  f"95% CI=[{r['ci95_clustered'][0]:+.3f}, "
                  f"{r['ci95_clustered'][1]:+.3f}]  t={r['t_stat']:.2f}")
        print(f"  ({time.time()-t0:.0f}s)")

    payload = {
        "method": (
            "Parent-vantage DML (partially linear, GBM nuisances, "
            "orthogonal score) under two cross-fitting rules: row-level "
            "KFold as shipped in dml_parent.py, and country-level "
            "GroupKFold. Standard errors are the analytic DML variance "
            "clustered on country, combined across seeds with the "
            "across-seed variance of theta. The shipped file's "
            "'theta_ci95' is the 2.5/97.5 percentile of five seed "
            "estimates; it is reproduced here as "
            "'seed_percentile_range_NOT_a_ci' and is not a confidence "
            "interval."
        ),
        "why": (
            "Two defects in dml_parent.py. (1) The reported interval is "
            "seed dispersion, not sampling uncertainty, so it cannot "
            "support the claim that the effect excludes zero. (2) "
            "Cross-fitting splits rows rather than countries, so the "
            "nuisance models see other years of the same country and the "
            "residualisation is optimistic. Both are fixed here and the "
            "shipped configuration is retained side by side."
        ),
        "n_folds": N_FOLDS,
        "seeds": seeds,
        "nuisance_learner": {"model": "GradientBoostingRegressor", **GBM_KW},
        "treatment": (
            "mean of the standardised education feature block "
            "(wcde_education + bl_education + derived); theta is per one "
            "standard-deviation-scale unit of that aggregate index, not "
            "per percentage point of completion"
        ),
        "results": results,
        "total_elapsed_sec": time.time() - t0,
        "script": "scripts/ml/chapter9/dml_parent_clustered.py",
    }
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"\nWrote {OUT_PATH}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

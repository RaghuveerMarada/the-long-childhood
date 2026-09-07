# =============================================================================
# PAPER REFERENCE
# Script:  scripts/ml/conformal_intervals.py
# Paper:   "The Long Childhood: On the Convergence of Nations"
#
# Produces:
#   Distribution-free prediction intervals for the parent-vantage
#   transformer, with measured empirical coverage on held-out countries.
#
#   Why this script exists
#   ----------------------
#   Every ML number the appendix reports is a point estimate. The
#   counterfactual table is the sharpest case: "Sri Lanka +12.10 years of
#   life expectancy over Pakistan's education trajectory" is a single
#   number with no interval attached. A referee will ask what its
#   uncertainty is, and R² does not answer that question — R² describes
#   the fit across the panel, not the reliability of one country's
#   prediction.
#
#   Split conformal prediction (Vovk et al. 2005; Lei et al. 2018) gives
#   an interval with a finite-sample coverage guarantee that holds under
#   exchangeability alone. No distributional assumption, no assumption
#   that the model is well specified. If the model is bad the intervals
#   are wide, and that is the honest answer rather than a hidden failure.
#
#   The exchangeability caveat is real and is reported, not hidden:
#   country-clustered panel data are not i.i.d., so the guarantee is
#   exact for exchangeable draws and approximate here. That is why this
#   script MEASURES empirical coverage on held-out countries instead of
#   asserting the nominal level. The gap between nominal and empirical
#   coverage is the number a referee should be shown.
#
#   Calibration is split BY COUNTRY, never by row. Splitting by row would
#   put a country's 1975 observation in calibration and its 1980
#   observation in test, and the resulting intervals would be too narrow
#   for exactly the reason the outer folds are country-level.
#
# Method:
#   For each outer country-holdout fold:
#     1. training countries -> proper-train (60%) / calibration (20%) /
#        early-stopping (20%), all split by country
#     2. fit on proper-train, early stop on the early-stopping split
#        (never on the held-out fold — see scripts/ml/leakage_audit.py)
#     3. conformity score s_i = |y_i - yhat_i| on calibration countries
#     4. qhat = ceil((n+1)(1-alpha))/n empirical quantile of s
#     5. interval on held-out countries = yhat +/- qhat
#     6. record empirical coverage and mean interval width
#
# Inputs:
#   scripts/ml/data_loader_parent_lag.load_parent_lag_panels
#
# Outputs:
#   scripts/ml/checkin/conformal_intervals.json
#
# Usage:
#   make ml-conformal
#   python scripts/ml/conformal_intervals.py --seeds 3 --alpha 0.1
# =============================================================================
"""Split-conformal prediction intervals with measured held-out coverage."""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, SCRIPT_DIR)

from data_loader import country_holdout_mask  # noqa: E402
from data_loader_parent_lag import load_parent_lag_panels, PARENT_LAG_HORIZONS  # noqa: E402
from universal_transformer import UniversalTransformer, DEFAULT_HP  # noqa: E402

CHECKIN_DIR = os.path.join(SCRIPT_DIR, "checkin")
os.makedirs(CHECKIN_DIR, exist_ok=True)
OUT_PATH = os.path.join(CHECKIN_DIR, "conformal_intervals.json")

N_FOLDS = 5
CALIB_FRACTION = 0.20   # of training countries
EARLYSTOP_FRACTION = 0.20  # of training countries

# Units each outcome is measured in, for reporting interval widths.
UNITS = {"LE": "years", "TFR": "births per woman", "U5MR": "deaths per 1,000"}


def _r2(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


def _standardize_y(y):
    mean = y.mean(axis=0)
    std = y.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (y - mean) / std, mean.astype(np.float32), std.astype(np.float32)


def _three_way_country_split(country_ids, rows, seed):
    """Split `rows` into proper-train / calibration / early-stop BY COUNTRY."""
    countries = np.unique(country_ids[rows])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(countries))
    n = len(countries)
    n_cal = max(1, int(round(CALIB_FRACTION * n)))
    n_es = max(1, int(round(EARLYSTOP_FRACTION * n)))
    cal_c = set(countries[perm[:n_cal]].tolist())
    es_c = set(countries[perm[n_cal:n_cal + n_es]].tolist())

    cid = country_ids[rows]
    is_cal = np.array([c in cal_c for c in cid])
    is_es = np.array([c in es_c for c in cid])
    is_tr = ~(is_cal | is_es)
    return is_tr, is_cal, is_es


def _train(X_tr, m_tr, y_tr_std, X_es, m_es, y_es, y_mean, y_std,
           n_features, window, hp, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(hp["num_threads"])

    dl = DataLoader(
        TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                      torch.tensor(m_tr, dtype=torch.float32),
                      torch.tensor(y_tr_std, dtype=torch.float32)),
        batch_size=hp["batch_size"], shuffle=True)
    Xe = torch.tensor(X_es, dtype=torch.float32)
    me = torch.tensor(m_es, dtype=torch.float32)

    model = UniversalTransformer(
        n_features=n_features, window=window,
        d_model=hp["d_model"], nhead=hp["nhead"], num_layers=hp["num_layers"],
        ff_mult=hp["ff_mult"], n_outputs=1, dropout=hp["dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"],
                            weight_decay=hp["weight_decay"])
    loss_fn = nn.MSELoss()

    best, best_state, no_improve = -np.inf, None, 0
    for _ in range(hp["epochs"]):
        model.train()
        for Xb, mb, yb in dl:
            opt.zero_grad()
            loss_fn(model(Xb, mb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(Xe, me).numpy() * y_std + y_mean
        score = float(np.mean(_r2(y_es, pred)))
        if score > best:
            best, no_improve = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
        if no_improve >= hp["patience"]:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model


def _predict(model, X, mask, y_mean, y_std):
    with torch.no_grad():
        out = model(torch.tensor(X, dtype=torch.float32),
                    torch.tensor(mask, dtype=torch.float32)).numpy()
    return out * y_std + y_mean


def _conformal_quantile(scores, alpha):
    """Finite-sample split-conformal quantile.

    Uses the ceil((n+1)(1-alpha))/n empirical quantile, which is what
    delivers the >= 1-alpha marginal coverage guarantee. Returns inf when
    n is too small for the level to be attainable, which is the correct
    (if useless) answer rather than a silently too-narrow interval.
    """
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return float("inf")
    return float(np.sort(scores)[k - 1])


def run_outcome(panel, alpha, hp, seed):
    y = panel["y"]
    X = panel["X"]
    mask = panel["mask"]
    cid = panel["country_ids"]
    n_features, window = X.shape[-1], panel["window"]

    covered, widths, abs_err, oof_pred = [], [], [], np.zeros_like(y)
    seen = np.zeros(len(y), dtype=bool)
    per_fold = []

    for fold in range(N_FOLDS):
        is_test = country_holdout_mask(panel, n_folds=N_FOLDS, fold=fold, seed=seed)
        train_rows = ~is_test
        is_tr, is_cal, is_es = _three_way_country_split(cid, train_rows, seed + fold)

        tr_idx = np.where(train_rows)[0]
        rows_tr, rows_cal, rows_es = tr_idx[is_tr], tr_idx[is_cal], tr_idx[is_es]

        y_tr_std, y_mean, y_std = _standardize_y(y[rows_tr])
        model = _train(X[rows_tr], mask[rows_tr], y_tr_std,
                       X[rows_es], mask[rows_es], y[rows_es],
                       y_mean, y_std, n_features, window, hp, seed + fold)

        # Calibration scores on countries the model never trained on and
        # never early-stopped on.
        pred_cal = _predict(model, X[rows_cal], mask[rows_cal], y_mean, y_std)
        scores = np.abs(y[rows_cal] - pred_cal).ravel()
        qhat = _conformal_quantile(scores, alpha)

        pred_test = _predict(model, X[is_test], mask[is_test], y_mean, y_std)
        y_test = y[is_test]
        cov = (np.abs(y_test - pred_test).ravel() <= qhat)

        covered.append(cov)
        widths.append(np.full(cov.shape, 2 * qhat))
        abs_err.append(np.abs(y_test - pred_test).ravel())
        oof_pred[is_test] = pred_test
        seen |= is_test

        per_fold.append({
            "fold": fold,
            "n_proper_train": int(len(rows_tr)),
            "n_calibration": int(len(rows_cal)),
            "n_earlystop": int(len(rows_es)),
            "n_test": int(is_test.sum()),
            "qhat_half_width": qhat,
            "empirical_coverage": float(cov.mean()),
        })

    covered = np.concatenate(covered)
    widths = np.concatenate(widths)
    abs_err = np.concatenate(abs_err)

    return {
        "horizon": panel["pred_horizon"],
        "units": UNITS.get(panel.get("target_name", ""), ""),
        "n_samples": int(seen.sum()),
        "nominal_coverage": 1 - alpha,
        "empirical_coverage": float(covered.mean()),
        "coverage_gap": float(covered.mean() - (1 - alpha)),
        "mean_interval_width": float(widths.mean()),
        "median_absolute_error": float(np.median(abs_err)),
        "oof_r2": float(_r2(y[seen], oof_pred[seen])[0]),
        "per_fold": per_fold,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.1,
                    help="miscoverage level; 0.1 = 90%% nominal intervals")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--outcomes", nargs="+", default=["LE", "TFR", "U5MR"])
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()

    hp = dict(DEFAULT_HP)
    if args.threads:
        hp["num_threads"] = args.threads

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        panels = load_parent_lag_panels(mode="joint", verbose=False)

    seeds = [DEFAULT_HP["seed"] + 100 * i for i in range(args.seeds)]
    t0 = time.time()
    results = {}

    for outcome in args.outcomes:
        panel = panels[outcome]
        runs = []
        for seed in seeds:
            r = run_outcome(panel, args.alpha, hp, seed)
            runs.append(r)
            print(f"  [{outcome} seed={seed}] coverage="
                  f"{r['empirical_coverage']:.3f} (nominal {r['nominal_coverage']:.2f})"
                  f"  width={r['mean_interval_width']:.2f} {r['units']}"
                  f"  R²={r['oof_r2']:.3f}  ({time.time()-t0:.0f}s)")
        results[outcome] = {
            "horizon": PARENT_LAG_HORIZONS[outcome],
            "units": UNITS[outcome],
            "n_samples": runs[0]["n_samples"],
            "nominal_coverage": 1 - args.alpha,
            "empirical_coverage_mean": float(np.mean([r["empirical_coverage"] for r in runs])),
            "empirical_coverage_std": float(np.std([r["empirical_coverage"] for r in runs])),
            "coverage_gap_mean": float(np.mean([r["coverage_gap"] for r in runs])),
            "mean_interval_width": float(np.mean([r["mean_interval_width"] for r in runs])),
            "median_absolute_error": float(np.mean([r["median_absolute_error"] for r in runs])),
            "oof_r2_mean": float(np.mean([r["oof_r2"] for r in runs])),
            "per_seed": runs,
        }

    payload = {
        "method": (
            "Split-conformal prediction intervals (Vovk et al. 2005; Lei et "
            "al. 2018) on the parent-vantage transformer. Within each outer "
            "country-holdout fold the training countries are split by "
            "country into proper-train (60%), calibration (20%) and "
            "early-stopping (20%). Conformity score is the absolute "
            "residual on calibration countries; the interval half-width is "
            "the ceil((n+1)(1-alpha))/n empirical quantile of those scores. "
            "Coverage is then MEASURED on the held-out countries rather "
            "than assumed. The held-out fold is never used for fitting or "
            "for early stopping."
        ),
        "caveat": (
            "The split-conformal guarantee is exact under exchangeability. "
            "Country-year panel rows are not exchangeable: observations "
            "cluster within country and trend within era. Calibrating by "
            "country removes the within-country part of the dependence, "
            "which is why coverage is reported empirically. Treat the "
            "empirical coverage column, not the nominal level, as the "
            "claim this script supports."
        ),
        "alpha": args.alpha,
        "n_folds": N_FOLDS,
        "calibration_fraction_of_training_countries": CALIB_FRACTION,
        "earlystop_fraction_of_training_countries": EARLYSTOP_FRACTION,
        "n_seeds": args.seeds,
        "seeds": seeds,
        "results": results,
        "total_elapsed_sec": time.time() - t0,
        "script": "scripts/ml/conformal_intervals.py",
    }
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"\nWrote {OUT_PATH}  ({time.time()-t0:.0f}s)")

    print(f"\n=== {int((1-args.alpha)*100)}% conformal intervals, "
          f"coverage measured on held-out countries ===")
    for outcome, r in results.items():
        print(f"  {outcome:5s} coverage={r['empirical_coverage_mean']:.3f} "
              f"(nominal {r['nominal_coverage']:.2f}, gap "
              f"{r['coverage_gap_mean']:+.3f})   "
              f"interval width={r['mean_interval_width']:.2f} {r['units']}   "
              f"median |error|={r['median_absolute_error']:.2f}")


if __name__ == "__main__":
    main()

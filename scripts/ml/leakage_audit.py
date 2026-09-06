# =============================================================================
# PAPER REFERENCE
# Script:  scripts/ml/leakage_audit.py
# Paper:   "The Long Childhood: On the Convergence of Nations"
#
# Produces:
#   An honest-holdout re-estimate of the parent-vantage transformer's
#   out-of-fold R² and of the G1 education-ablation gate.
#
#   Why this script exists
#   ----------------------
#   universal_transformer.train_one_fold and its copy in
#   run_universal_evidence_parent_lag._train_one_fold_single select the
#   reported checkpoint by validation R² measured ON THE HELD-OUT FOLD,
#   then report that same fold's R² as the out-of-fold score:
#
#       is_val = country_holdout_mask(...)      # the held-out countries
#       ...
#       val_r2 = _r2(y_val, pred_val)           # scored on held-out
#       if val_mean > best_val_r2_mean:         # <- selection on held-out
#           best_state = ...                    #    (early stopping too)
#       ...
#       model.load_state_dict(best_state)
#       return ck, pred_val, ...                # <- reported as OOF
#
#   Choosing the epoch by the test set is model selection on the test
#   set. The reported R² is therefore an optimistic bound, not a clean
#   generalisation estimate. With ~400 candidate epochs and ~300
#   held-out samples per fold the optimism is real, not hypothetical.
#
#   This is a reporting-protocol defect, not necessarily a substantive
#   one. The paper's ML claim is not "R² is 0.75"; it is "zeroing the
#   education block collapses R², zeroing everything else barely moves
#   it." A bias that inflates BOTH the baseline and the ablated fit may
#   leave that contrast intact. This script measures which it is.
#
# Protocols compared (identical folds, seeds, hyperparameters, features):
#   A  shipped     — early stop on the held-out fold, report that fold.
#   B  nested      — carve an inner validation set from TRAINING countries
#                    only; early stop on it; report the untouched
#                    held-out fold.
#   C  nested+scale— B, plus per-feature standardisation refit on
#                    inner-training countries only. The shipped loader
#                    standardises X over the whole panel before any split,
#                    so held-out countries contribute to the feature
#                    means and SDs. Mild, but a referee will name it.
#
# Inputs:
#   scripts/ml/data_loader_parent_lag.load_parent_lag_panels
#
# Outputs:
#   scripts/ml/checkin/leakage_audit.json
#
# Usage:
#   make ml-audit
#   python scripts/ml/leakage_audit.py --seeds 5 --outcomes LE TFR U5MR
# =============================================================================
"""Honest-holdout re-estimate of the transformer's OOF R² and G1 gate."""

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

from data_loader import country_holdout_mask, feature_ablation  # noqa: E402
from data_loader_parent_lag import load_parent_lag_panels, PARENT_LAG_HORIZONS  # noqa: E402
from universal_transformer import UniversalTransformer, DEFAULT_HP  # noqa: E402

CHECKIN_DIR = os.path.join(SCRIPT_DIR, "checkin")
os.makedirs(CHECKIN_DIR, exist_ok=True)
OUT_PATH = os.path.join(CHECKIN_DIR, "leakage_audit.json")

N_FOLDS = 5
INNER_VAL_FRACTION = 0.2  # of training countries, for protocols B and C

# Feature groups zeroed for the G1 education-ablation gate. Kept identical
# to run_universal_evidence_parent_lag.py so the two are comparable.
EDU_GROUPS = ["wcde_education", "bl_education", "derived"]


# ── helpers ──────────────────────────────────────────────────────────────

def _r2(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


def _standardize_y(y):
    mean = y.mean(axis=0)
    std = y.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (y - mean) / std, mean.astype(np.float32), std.astype(np.float32)


def _inner_split(panel, train_idx, seed):
    """Split TRAINING rows into inner-train / inner-val by country.

    Country-level, like the outer split, so a country never straddles the
    two: otherwise the early-stopping signal is contaminated by the same
    within-country correlation the outer split exists to remove.
    """
    train_countries = np.unique(panel["country_ids"][train_idx])
    rng = np.random.default_rng(seed + 9973)
    perm = rng.permutation(len(train_countries))
    n_val = max(1, int(round(INNER_VAL_FRACTION * len(train_countries))))
    inner_val_countries = set(train_countries[perm[:n_val]].tolist())
    is_inner_val = np.array(
        [cid in inner_val_countries for cid in panel["country_ids"][train_idx]]
    )
    return ~is_inner_val, is_inner_val


def _refit_feature_scaling(X_all, fit_rows):
    """Re-standardise X using only `fit_rows`.

    The shipped loader has already applied global standardisation and
    written 0.0 into unobserved cells. We invert that affine map, refit
    mean/SD on the fit rows, reapply, and restore the zeros. Missing
    cells are identified as exact 0.0 in the shipped array: after a
    (x - mean) / sd transform of real measurements an exact zero is
    otherwise a measure-zero event, so this recovers the fill mask
    faithfully in practice.
    """
    zero_mask = (X_all == 0.0)
    fit = X_all[fit_rows]
    fit_valid = np.where(zero_mask[fit_rows], np.nan, fit)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmean(fit_valid, axis=(0, 1))
        s = np.nanstd(fit_valid, axis=(0, 1))
    m = np.nan_to_num(m, nan=0.0)
    s = np.where(np.isfinite(s) & (s > 1e-8), s, 1.0)
    out = (X_all - m) / s
    out[zero_mask] = 0.0
    return out.astype(np.float32)


def _train(X_tr, m_tr, y_tr_std, X_es, m_es, y_es_raw, y_mean, y_std,
           n_features, window, hp, fold_seed):
    """Train with early stopping scored on (X_es, y_es_raw).

    Protocol A passes the held-out fold as the early-stopping set.
    Protocols B and C pass an inner split of the training countries.
    """
    torch.manual_seed(fold_seed)
    np.random.seed(fold_seed)
    torch.set_num_threads(hp["num_threads"])

    Xt = torch.tensor(X_tr, dtype=torch.float32)
    mt = torch.tensor(m_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr_std, dtype=torch.float32)
    Xe = torch.tensor(X_es, dtype=torch.float32)
    me = torch.tensor(m_es, dtype=torch.float32)

    dl = DataLoader(TensorDataset(Xt, mt, yt),
                    batch_size=hp["batch_size"], shuffle=True)

    model = UniversalTransformer(
        n_features=n_features, window=window,
        d_model=hp["d_model"], nhead=hp["nhead"], num_layers=hp["num_layers"],
        ff_mult=hp["ff_mult"], n_outputs=y_tr_std.shape[1], dropout=hp["dropout"],
    )
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"],
                            weight_decay=hp["weight_decay"])
    loss_fn = nn.MSELoss()

    best, best_state, best_epoch, no_improve = -np.inf, None, -1, 0
    for epoch in range(hp["epochs"]):
        model.train()
        for Xb, mb, yb in dl:
            opt.zero_grad()
            loss = loss_fn(model(Xb, mb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred_es = model(Xe, me).numpy() * y_std + y_mean
        score = float(np.mean(_r2(y_es_raw, pred_es)))

        if score > best:
            best, best_epoch, no_improve = score, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
        if no_improve >= hp["patience"]:
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


def _predict(model, X, mask, y_mean, y_std):
    with torch.no_grad():
        out = model(torch.tensor(X, dtype=torch.float32),
                    torch.tensor(mask, dtype=torch.float32)).numpy()
    return out * y_std + y_mean


# ── one (outcome, seed) run under one protocol ───────────────────────────

def run_protocol(panel, protocol, hp, seed, ablate_education=False):
    """Return (oof_r2, zero_gate_r2, mean_best_epoch) for one protocol/seed.

    `ablate_education=True` trains a model that never sees education (the
    RETRAIN gate). Independently, every run also scores the trained model
    on education-zeroed inputs (the ZERO-AT-INFERENCE gate, which is what
    the shipped runner reports). The two answer different questions:

      zero-at-inference — how much does THIS trained model lean on the
        education inputs? Zeroing them pushes the model off its training
        manifold, so the drop is an upper bound on reliance.
      retrain — how much does education add OVER the alternatives, when
        the model is free to re-optimise without it? This is the
        incremental-value question the paper's prose actually asks.
    """
    X_full = panel["X"]
    if ablate_education:
        # Same groups the headline runner zeroes for the G1 gate
        # (run_universal_evidence_parent_lag.py:248).
        X_full = feature_ablation(panel, drop_groups=EDU_GROUPS)["X"]
    X_zeroed = feature_ablation(panel, drop_groups=EDU_GROUPS)["X"]
    y = panel["y"]
    mask = panel["mask"]
    n_features, window = X_full.shape[-1], panel["window"]

    oof_pred = np.zeros_like(y)
    oof_pred_zero = np.zeros_like(y)
    seen = np.zeros(y.shape[0], dtype=bool)
    best_epochs = []

    for fold in range(N_FOLDS):
        is_val = country_holdout_mask(panel, n_folds=N_FOLDS, fold=fold, seed=seed)
        train_idx = ~is_val

        X, Xz = X_full, X_zeroed
        if protocol == "C":
            # Refit feature scaling on inner-training rows only. Compute the
            # inner split first so held-out AND inner-val rows are excluded.
            inner_tr_local, _ = _inner_split(panel, train_idx, seed + fold)
            fit_rows = np.zeros(len(y), dtype=bool)
            fit_rows[np.where(train_idx)[0][inner_tr_local]] = True
            X = _refit_feature_scaling(X_full, fit_rows)
            Xz = _refit_feature_scaling(X_zeroed, fit_rows)

        X_tr_all, y_tr_all, m_tr_all = X[train_idx], y[train_idx], mask[train_idx]

        if protocol == "A":
            # Shipped protocol: early stop on the held-out fold itself.
            y_tr_std, y_mean, y_std = _standardize_y(y_tr_all)
            model, be = _train(
                X_tr_all, m_tr_all, y_tr_std,
                X[is_val], mask[is_val], y[is_val],
                y_mean, y_std, n_features, window, hp, seed + fold,
            )
        else:
            inner_tr, inner_val = _inner_split(panel, train_idx, seed + fold)
            y_inner_tr = y_tr_all[inner_tr]
            y_tr_std, y_mean, y_std = _standardize_y(y_inner_tr)
            model, be = _train(
                X_tr_all[inner_tr], m_tr_all[inner_tr], y_tr_std,
                X_tr_all[inner_val], m_tr_all[inner_val], y_tr_all[inner_val],
                y_mean, y_std, n_features, window, hp, seed + fold,
            )

        oof_pred[is_val] = _predict(model, X[is_val], mask[is_val], y_mean, y_std)
        oof_pred_zero[is_val] = _predict(model, Xz[is_val], mask[is_val],
                                         y_mean, y_std)
        seen |= is_val
        best_epochs.append(be)

    return (float(_r2(y[seen], oof_pred[seen])[0]),
            float(_r2(y[seen], oof_pred_zero[seen])[0]),
            float(np.mean(best_epochs)))


# ── driver ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5,
                    help="number of seeds per protocol (default 5)")
    ap.add_argument("--outcomes", nargs="+", default=["LE", "TFR", "U5MR"])
    ap.add_argument("--protocols", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--no-ablation", action="store_true",
                    help="skip the G1 education-ablation arm (faster)")
    args = ap.parse_args()

    hp = dict(DEFAULT_HP)
    if args.threads:
        hp["num_threads"] = args.threads

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        panels = load_parent_lag_panels(mode="joint", verbose=False)

    seeds = [DEFAULT_HP["seed"] + 100 * i for i in range(args.seeds)]
    results = {}
    t0 = time.time()

    for outcome in args.outcomes:
        panel = panels[outcome]
        results[outcome] = {
            "horizon": PARENT_LAG_HORIZONS[outcome],
            "n_samples": int(panel["y"].shape[0]),
            "n_countries": int(len(np.unique(panel["country_ids"]))),
            "protocols": {},
        }
        for proto in args.protocols:
            base_scores, zero_scores, retrain_scores, epochs = [], [], [], []
            for seed in seeds:
                r2, r2_zero, be = run_protocol(panel, proto, hp, seed)
                base_scores.append(r2)
                zero_scores.append(r2_zero)
                epochs.append(be)
                if not args.no_ablation:
                    r2_re, _, _ = run_protocol(panel, proto, hp, seed,
                                               ablate_education=True)
                    retrain_scores.append(r2_re)
                print(f"  [{outcome} {proto} seed={seed}] "
                      f"baseline R²={r2:.4f}  zero-gate R²={r2_zero:.4f}"
                      + (f"  retrain-gate R²={retrain_scores[-1]:.4f}"
                         if retrain_scores else "")
                      + f"  (best epoch {be:.0f}, {time.time()-t0:.0f}s)")

            def _gate(scores):
                drops = [b - a for b, a in zip(base_scores, scores)]
                fracs = [d / b if b > 0 else float("nan")
                         for d, b in zip(drops, base_scores)]
                return {
                    "ablated_r2_mean": float(np.mean(scores)),
                    "ablated_r2_std": float(np.std(scores)),
                    "r2_drop_absolute_mean": float(np.mean(drops)),
                    "r2_drop_fraction_mean": float(np.nanmean(fracs)),
                    "ablated_r2_per_seed": list(scores),
                }

            entry = {
                "baseline_r2_mean": float(np.mean(base_scores)),
                "baseline_r2_std": float(np.std(base_scores)),
                "baseline_r2_per_seed": base_scores,
                "mean_best_epoch": float(np.mean(epochs)),
                # The shipped G1 gate: zero education on a model trained
                # with it. Upper bound on the trained model's reliance.
                "g1_zero_at_inference": _gate(zero_scores),
            }
            if retrain_scores:
                # The incremental-value gate: retrain from scratch with no
                # education inputs at all.
                entry["g1_retrain_without_education"] = _gate(retrain_scores)
            results[outcome]["protocols"][proto] = entry

        # Optimism = how much the shipped protocol overstates the honest one.
        if "A" in results[outcome]["protocols"] and "B" in results[outcome]["protocols"]:
            a = results[outcome]["protocols"]["A"]
            b = results[outcome]["protocols"]["B"]
            opt = {"baseline_r2": a["baseline_r2_mean"] - b["baseline_r2_mean"]}
            for gate in ("g1_zero_at_inference", "g1_retrain_without_education"):
                if gate in a and gate in b:
                    opt[f"{gate}_drop_absolute"] = (
                        a[gate]["r2_drop_absolute_mean"]
                        - b[gate]["r2_drop_absolute_mean"]
                    )
            results[outcome]["optimism_A_minus_B"] = opt

    payload = {
        "method": (
            "Leakage audit of the parent-vantage UniversalTransformer. "
            "Protocol A reproduces the shipped protocol (checkpoint and "
            "early stopping selected on the held-out fold, that fold then "
            "reported as out-of-fold). Protocol B carves an inner "
            "validation split from training countries only and leaves the "
            "held-out fold untouched until scoring. Protocol C adds "
            "train-only refitting of the per-feature standardisation, "
            "which the shipped loader performs over the whole panel. "
            "Folds, seeds, features, architecture and hyperparameters are "
            "identical across protocols; only the selection signal varies."
        ),
        "protocol_definitions": {
            "A": "shipped — early stop and checkpoint-select on the held-out fold",
            "B": "nested — inner validation split from training countries only",
            "C": "nested + per-feature standardisation refit on inner-training rows",
        },
        "n_folds": N_FOLDS,
        "inner_val_fraction": INNER_VAL_FRACTION,
        "n_seeds": args.seeds,
        "seeds": seeds,
        "hyperparameters": {k: v for k, v in hp.items()},
        "results": results,
        "total_elapsed_sec": time.time() - t0,
        "script": "scripts/ml/leakage_audit.py",
    }
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"\nWrote {OUT_PATH}  ({time.time()-t0:.0f}s)")

    print("\n=== summary: out-of-fold R² by protocol ===")
    for outcome, r in results.items():
        line = f"  {outcome:5s}"
        for proto in args.protocols:
            p = r["protocols"][proto]
            line += f"  {proto}={p['baseline_r2_mean']:.4f}"
        if "optimism_A_minus_B" in r:
            line += f"   optimism(A-B)={r['optimism_A_minus_B']['baseline_r2']:+.4f}"
        print(line)
    for gate, label in (
        ("g1_zero_at_inference", "zero education at inference (shipped gate)"),
        ("g1_retrain_without_education", "retrain without education (incremental value)"),
    ):
        if not any(gate in r["protocols"][p]
                   for r in results.values() for p in args.protocols):
            continue
        print(f"\n=== education-ablation R² drop — {label} ===")
        for outcome, r in results.items():
            line = f"  {outcome:5s}"
            for proto in args.protocols:
                p = r["protocols"][proto]
                v = p.get(gate, {}).get("r2_drop_absolute_mean", float("nan"))
                line += f"  {proto}={v:.4f}"
            print(line)


if __name__ == "__main__":
    main()

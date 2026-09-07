# =============================================================================
# scripts/ml/cloud/run_job.py
#
# In-VM job dispatcher. Reads (kind, params) and dispatches to the
# appropriate runner. Writes output JSON to --output-dir/result.json.
# =============================================================================
"""
run_job.py — single-VM job dispatcher.
"""

import argparse
import json
import os
import sys
import time
import traceback

# Make the rest of the repo importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(os.path.dirname(ML_DIR))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, ML_DIR)
sys.path.insert(0, os.path.join(ML_DIR, "chapter9"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "diagnostics"))


def run_smoke(params, out_dir):
    """Tiny test: load panel, run 1 fold of ridge, save result.
    Validates that the VM has data + sklearn working. ~2 minutes."""
    from data_loader import load_panel
    from spec_curve import flatten_feature_groups, run_one_method
    panel = load_panel(mode="joint", verbose=False)
    flat_groups = flatten_feature_groups(panel)
    result = run_one_method(panel, flat_groups, "ridge", verbose=True)
    out = {"kind": "smoke", "result": result, "params": params}
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"smoke complete; baseline R²={result['r2_baseline']}")


def run_transformer(params, out_dir):
    """Single-seed transformer 5-fold OOF.

    params: {"mode": "joint", "horizon": 25, "seed": <int>}
    """
    import torch
    from universal_transformer import DEFAULT_HP, run_5fold
    hp = dict(DEFAULT_HP)
    hp["seed"] = int(params.get("seed", 42))
    horizon = int(params.get("horizon", 25))
    mode = params.get("mode", "joint")
    # Hook horizon via data_loader env override (we'll add this)
    os.environ["LL_PRED_HORIZON"] = str(horizon)

    prefix = f"trf_{mode}_h{horizon}_s{hp['seed']}"
    summary = run_5fold(mode=mode, hp=hp, verbose=True,
                        checkpoint_prefix=prefix)

    # G1/G2 ablation
    from data_loader import load_panel
    from run_universal_evidence import run_g1_g2
    panel = load_panel(mode=mode, verbose=False)
    gates = run_g1_g2(panel, prefix)

    out = {
        "kind": "transformer",
        "params": params,
        "summary": summary,
        "gates": gates,
        "elapsed_sec": summary.get("elapsed_sec"),
    }
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    # Upload checkpoints too
    import shutil
    ckpt_src = os.path.join(ML_DIR, "checkpoints")
    ckpt_dst = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dst, exist_ok=True)
    for fname in os.listdir(ckpt_src):
        if fname.startswith(prefix):
            shutil.copy(os.path.join(ckpt_src, fname), ckpt_dst)


def run_spec_curve(params, out_dir):
    from spec_curve import run_spec_curve as _run
    # Shard by method: params["methods"] is a subset (one method per shard);
    # None runs all five. Each shard's result.json holds its method(s);
    # aggregation concatenates the per-method results.
    methods = params.get("methods")
    out = _run(methods=methods, verbose=True)
    out["shard_methods"] = methods
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)


def run_placebos(params, out_dir):
    import placebos as _placebos_mod
    seeds = tuple(params.get("seeds", [42, 43, 44]))
    vantage = params.get("vantage", "joint")
    # Shard by placebo: params["placebos"] is a list of placebo function names
    # (one per shard); None runs all six. Aggregation merges placebo_rows.
    names = params.get("placebos")
    placebos = None
    if names:
        name_set = set(names)
        placebos = [fn for fn in _placebos_mod.PLACEBOS
                    if fn.__name__ in name_set]
    if vantage == "parent":
        from placebos import run_placebos_parent as _run_parent
        out = _run_parent(placebos=placebos, seeds=seeds, verbose=True)
    else:
        from placebos import run_placebos as _run
        out = _run(placebos=placebos, seeds=seeds, verbose=True)
    out["shard_placebos"] = names
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)


def run_transformer_single(params, out_dir):
    """Single-target transformer at biological horizon (LE@T+12, TFR@T+5,
    U5MR@T+12).

    params: {"target": "LE"|"TFR"|"U5MR", "seed": <int>, "horizon": <int>,
             "mode": "joint"|"bl_only"}

    mode is passed through to load_parent_lag_panels, so mode="bl_only" runs
    the biological-lag single-target model on Barro-Lee features only (the
    parent-vantage counterpart of the joint trf_bl job).
    """
    import json as _json
    import shutil
    import sys as _sys
    import torch
    _sys.path.insert(0, ML_DIR)
    from universal_transformer import DEFAULT_HP
    from run_universal_evidence_parent_lag import (
        _run_5fold_single, _run_gates_single,
    )
    from data_loader_parent_lag import load_parent_lag_panels

    target = params.get("target")
    seed = int(params.get("seed", 42))
    mode = params.get("mode", "joint")
    if target not in ("LE", "TFR", "U5MR"):
        raise ValueError(f"unknown target: {target}")

    hp = dict(DEFAULT_HP)
    hp["seed"] = seed
    hp["verbose"] = False

    panels = load_parent_lag_panels(mode=mode, verbose=False)
    panel = panels[target]
    # Suffix the checkpoint prefix with the feature mode so the bl_only run's
    # checkpoints never collide with the joint-features run.
    mode_tag = "" if mode == "joint" else f"_{mode}"
    ckpt_prefix = f"trf_parent{mode_tag}_{target.lower()}_s{seed}"
    summary = _run_5fold_single(panel, hp, ckpt_prefix=ckpt_prefix, verbose=True)
    gates = _run_gates_single(panel, ckpt_prefix, target_name=target)

    out = {
        "kind": "transformer_single",
        "params": params,
        "target": target,
        "mode": mode,
        "summary": summary,
        "gates": gates,
        "elapsed_sec": summary.get("elapsed_sec"),
    }
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        _json.dump(out, f, indent=2, default=float)
    # Save checkpoints into output
    ckpt_src = os.path.join(ML_DIR, "checkpoints")
    ckpt_dst = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dst, exist_ok=True)
    for fname in os.listdir(ckpt_src):
        if fname.startswith(ckpt_prefix):
            shutil.copy(os.path.join(ckpt_src, fname), ckpt_dst)


def run_entry_cohort_refit(params, out_dir):
    """Fair G1-gate refit: education excluded at TRAINING time (mode=
    "no_education", not zeroed-at-inference on a joint-trained model), on
    rows restricted to the entry-cohort [10%, 90%] expansion window
    (CANONICAL.md SS14) rather than the full country-year universe.

    See scripts/diagnostics/G1_GATE_INVESTIGATION.md. Restricting to the
    transition window matters because outside it a country's own education
    isn't moving (pre-expansion or ceiling-saturated), and because the full
    universe lets time-invariant / regionally-diffusing features (region,
    latitude, institutions) substitute for education via between-country
    resemblance rather than any within-country mechanism.

    params: {"target": "LE"|"TFR"|"U5MR", "seed": <int>}
    """
    import json as _json
    import shutil
    import sys as _sys
    _sys.path.insert(0, ML_DIR)
    from universal_transformer import DEFAULT_HP
    from run_universal_evidence_parent_lag import _run_5fold_single
    from data_loader_parent_lag import load_parent_lag_panels, PARENT_LAG_HORIZONS
    from g1_gate_entry_cohort_refit import (
        entry_cohort_mask, subset_panel, ENTRY_LOWER, ENTRY_UPPER,
    )

    target = params.get("target")
    seed = int(params.get("seed", 42))
    if target not in ("LE", "TFR", "U5MR"):
        raise ValueError(f"unknown target: {target}")

    hp = dict(DEFAULT_HP)
    hp["seed"] = seed
    hp["verbose"] = False

    panels_joint = load_parent_lag_panels(mode="joint", verbose=False)
    panels_no_edu = load_parent_lag_panels(mode="no_education", verbose=False)
    panel_j = panels_joint[target]
    panel_n = panels_no_edu[target]

    mask = entry_cohort_mask(panel_j)   # identical row universe/order in both modes
    sub_j = subset_panel(panel_j, mask)
    sub_n = subset_panel(panel_n, mask)

    ckpt_prefix_j = f"entrycohort_joint_{target.lower()}_s{seed}"
    ckpt_prefix_n = f"entrycohort_noedu_{target.lower()}_s{seed}"
    summary_joint = _run_5fold_single(sub_j, hp, ckpt_prefix=ckpt_prefix_j, verbose=True)
    summary_no_edu = _run_5fold_single(sub_n, hp, ckpt_prefix=ckpt_prefix_n, verbose=True)

    r2_j = summary_joint["overall_r2_oof"][0]
    r2_n = summary_no_edu["overall_r2_oof"][0]
    drop_frac = (r2_j - r2_n) / max(abs(r2_j), 1e-12)

    out = {
        "kind": "entry_cohort_refit",
        "params": params,
        "target": target,
        "horizon": PARENT_LAG_HORIZONS[target],
        "entry_cohort_window": [ENTRY_LOWER, ENTRY_UPPER],
        "n_full_panel": int(panel_j["y"].shape[0]),
        "n_entry_cohort": int(mask.sum()),
        "joint_with_education": summary_joint,
        "no_education_refit": summary_no_edu,
        "r2_drop_fraction": drop_frac,
    }
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        _json.dump(out, f, indent=2, default=float)
    ckpt_src = os.path.join(ML_DIR, "checkpoints")
    ckpt_dst = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dst, exist_ok=True)
    for fname in os.listdir(ckpt_src):
        if fname.startswith(ckpt_prefix_j) or fname.startswith(ckpt_prefix_n):
            shutil.copy(os.path.join(ckpt_src, fname), ckpt_dst)


def run_entry_cohort_no_geo(params, out_dir):
    """Follow-up to run_entry_cohort_refit (G1_GATE_INVESTIGATION.md SS6):
    education excluded at training time (mode="no_education") AND all
    time-invariant ("geography") broader-feature blocks also zeroed out --
    not just filtered by row -- while institutions (Polity2), GDP, and the
    other time-varying broader features stay in. Same entry-cohort [10,90]
    row restriction, same country-holdout 5-fold.

    This isolates how much of the no-education refit's R² is carried by
    time-invariant, purely-between-country features (region, latitude,
    colonizer, religion, colonial binary, settler mortality) versus by
    institutions/GDP/other time-varying features. Compare this job's R²
    against the "no_education_refit" R² from run_entry_cohort_refit (same
    target/seed) computed offline -- this job does not repeat that run.

    params: {"target": "LE"|"TFR"|"U5MR", "seed": <int>}
    """
    import json as _json
    import shutil
    import sys as _sys
    _sys.path.insert(0, ML_DIR)
    from universal_transformer import DEFAULT_HP
    from run_universal_evidence_parent_lag import _run_5fold_single
    from data_loader import feature_ablation
    from data_loader_parent_lag import load_parent_lag_panels, PARENT_LAG_HORIZONS
    from g1_gate_entry_cohort_refit import (
        entry_cohort_mask, subset_panel, ENTRY_LOWER, ENTRY_UPPER,
        GEOGRAPHY_TIME_INVARIANT_GROUPS,
    )

    target = params.get("target")
    seed = int(params.get("seed", 42))
    if target not in ("LE", "TFR", "U5MR"):
        raise ValueError(f"unknown target: {target}")

    hp = dict(DEFAULT_HP)
    hp["seed"] = seed
    hp["verbose"] = False

    # The entry-cohort mask reads the panel's own lower-sec-completion
    # feature, which mode="no_education" has already zeroed -- compute the
    # mask from the joint (real-valued) panel, same as run_entry_cohort_refit,
    # then apply it to the no_education panel's rows.
    panels_joint = load_parent_lag_panels(mode="joint", verbose=False)
    panels_no_edu = load_parent_lag_panels(mode="no_education", verbose=False)
    panel_j = panels_joint[target]
    panel_n = panels_no_edu[target]
    mask = entry_cohort_mask(panel_j)
    sub_n = subset_panel(panel_n, mask)
    sub_n_no_geo = feature_ablation(sub_n, drop_groups=GEOGRAPHY_TIME_INVARIANT_GROUPS)

    ckpt_prefix = f"entrycohort_noedu_nogeo_{target.lower()}_s{seed}"
    summary = _run_5fold_single(sub_n_no_geo, hp, ckpt_prefix=ckpt_prefix, verbose=True)

    out = {
        "kind": "entry_cohort_no_geo",
        "params": params,
        "target": target,
        "horizon": PARENT_LAG_HORIZONS[target],
        "entry_cohort_window": [ENTRY_LOWER, ENTRY_UPPER],
        "dropped_groups": GEOGRAPHY_TIME_INVARIANT_GROUPS,
        "n_entry_cohort": int(mask.sum()),
        "no_education_no_geography_refit": summary,
    }
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        _json.dump(out, f, indent=2, default=float)
    ckpt_src = os.path.join(ML_DIR, "checkpoints")
    ckpt_dst = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dst, exist_ok=True)
    for fname in os.listdir(ckpt_src):
        if fname.startswith(ckpt_prefix):
            shutil.copy(os.path.join(ckpt_src, fname), ckpt_dst)


def _walk_forward_one_panel(panel, hp, cutoff, targets, ckpt_seed):
    """Walk-forward train/eval on a single panel (joint [N,3] or single-target
    [N,1]). Returns the per-run summary dict. The body is shape-generic
    (`y.shape[1]` outputs, shape-generic `_r2`/`_standardize_y`) so it serves
    both the joint and parent-vantage single-target panels unchanged."""
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from data_loader import era_holdout_mask
    from universal_transformer import (
        UniversalTransformer, _r2, _standardize_y,
    )

    is_val = era_holdout_mask(panel, cutoff_year=cutoff)
    n_val = int(is_val.sum())
    n_train = int((~is_val).sum())
    if n_val < 50 or n_train < 100:
        raise RuntimeError(f"era split too lopsided: train={n_train} val={n_val}")

    # Train one model on the train half, evaluate on val half (no folds)
    torch.manual_seed(ckpt_seed); np.random.seed(ckpt_seed)
    X = panel["X"]; y = panel["y"]; mask = panel["mask"]
    X_train, X_val = X[~is_val], X[is_val]
    y_train, y_val = y[~is_val], y[is_val]
    m_train, m_val = mask[~is_val], mask[is_val]
    y_tr_std, y_mean, y_std = _standardize_y(y_train)

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_tr_std, dtype=torch.float32)
    mt = torch.tensor(m_train, dtype=torch.float32)
    Xv = torch.tensor(X_val, dtype=torch.float32)
    mv = torch.tensor(m_val, dtype=torch.float32)
    ds = TensorDataset(Xt, mt, yt)
    dl = DataLoader(ds, batch_size=hp["batch_size"], shuffle=True)

    model = UniversalTransformer(
        n_features=X.shape[-1], window=panel["window"],
        d_model=hp["d_model"], nhead=hp["nhead"],
        num_layers=hp["num_layers"], ff_mult=hp["ff_mult"],
        n_outputs=y.shape[1], dropout=hp["dropout"],
    )
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"],
                             weight_decay=hp["weight_decay"])
    loss_fn = nn.MSELoss()

    best_r2_mean = -np.inf
    best_state = None
    no_improve = 0
    for epoch in range(hp["epochs"]):
        model.train()
        for Xb, mb, yb in dl:
            opt.zero_grad()
            pred = model(Xb, mb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred_val_std = model(Xv, mv).numpy()
        pred_val = pred_val_std * y_std + y_mean
        r2 = _r2(y_val, pred_val)
        r2_mean = float(np.mean(r2))
        if r2_mean > best_r2_mean:
            best_r2_mean = r2_mean
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= hp["patience"]:
            break

    # Best model: ablate education at inference
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_val = model(Xv, mv).numpy() * y_std + y_mean
    r2_base = _r2(y_val, pred_val)

    # Education ablation
    edu_idx = []
    for g in ("wcde_education", "bl_education", "derived"):
        edu_idx.extend(panel["feature_groups"].get(g, []))
    Xv_ab = X_val.copy()
    for i in edu_idx:
        Xv_ab[:, :, i] = 0.0
    Xv_ab_t = torch.tensor(Xv_ab, dtype=torch.float32)
    with torch.no_grad():
        pred_val_ab = model(Xv_ab_t, mv).numpy() * y_std + y_mean
    r2_ab = _r2(y_val, pred_val_ab)

    return {
        "cutoff_year": cutoff,
        "n_train": n_train, "n_val": n_val,
        "summary": {
            "fold_r2_mean": r2_base.tolist(),
            "fold_r2_per_target": [r2_base.tolist()],
        },
        "r2_baseline": {t: float(v) for t, v in zip(targets, r2_base)},
        "r2_ablated_edu": {t: float(v) for t, v in zip(targets, r2_ab)},
        "r2_drop_absolute": {t: float(b - a) for t, b, a in
                              zip(targets, r2_base, r2_ab)},
    }


def run_transformer_walk_forward(params, out_dir):
    """Walk-forward (era-holdout) transformer: train on samples with year
    < cutoff_year, validate on samples with year >= cutoff_year.

    params: {"mode": "joint", "horizon": 25, "seed": <int>,
             "cutoff_year": <int>, "vantage": "joint"|"parent"}

    Default vantage "joint" reproduces the original multi-output (LE, TFR,
    U5MR at horizon 25) behavior byte-for-byte. vantage="parent" runs the
    same walk-forward once per outcome on its biological-lag single-target
    panel (LE@12, TFR@5, U5MR@12) and keys results under by_target.
    """
    import json as _json
    import sys as _sys
    _sys.path.insert(0, ML_DIR)
    from data_loader import load_panel
    from universal_transformer import DEFAULT_HP

    seed = int(params.get("seed", 42))
    cutoff = int(params.get("cutoff_year", 1990))
    mode = params.get("mode", "joint")
    vantage = params.get("vantage", "joint")

    hp = dict(DEFAULT_HP)
    hp["seed"] = seed

    if vantage == "parent":
        from data_loader_parent_lag import load_parent_lag_panels
        panels = load_parent_lag_panels(mode=mode, verbose=False)
        by_target = {}
        for tname, panel in panels.items():
            res = _walk_forward_one_panel(
                panel, hp, cutoff, targets=[tname], ckpt_seed=seed)
            by_target[tname] = res
        out = {
            "kind": "transformer_walk_forward",
            "vantage": "parent",
            "params": params,
            "cutoff_year": cutoff,
            "by_target": by_target,
        }
        with open(os.path.join(out_dir, "result.json"), "w") as f:
            _json.dump(out, f, indent=2, default=float)
        return

    targets = ["LE", "TFR", "U5MR"]
    panel = load_panel(mode=mode, verbose=False)
    res = _walk_forward_one_panel(panel, hp, cutoff, targets, ckpt_seed=seed)
    out = {
        "kind": "transformer_walk_forward",
        "params": params,
        **res,
    }
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        _json.dump(out, f, indent=2, default=float)


def run_dml(params, out_dir):
    """Double / debiased ML (Chernozhukov et al. 2018).

    Cross-fit two nuisance functions on the same panel:
      g(X) := E[Y | X-edu]   (outcome residualized on non-edu features)
      m(X) := E[edu | X-edu] (edu residualized on non-edu features)
    Then regress residuals: (Y - g(X)) ~ (edu - m(X)). The resulting
    coefficient is the doubly-robust estimate of edu's effect.
    """
    import json as _json
    import sys as _sys
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.model_selection import KFold
    _sys.path.insert(0, ML_DIR)
    _sys.path.insert(0, os.path.join(ML_DIR, "chapter9"))
    from data_loader import load_panel
    from spec_curve import flatten_panel, flatten_feature_groups

    n_folds = int(params.get("n_folds", 5))
    seeds = list(params.get("seeds", [42, 43, 44, 45, 46]))
    targets = ["LE", "TFR", "U5MR"]

    panel = load_panel(mode="joint", verbose=False)
    flat_groups = flatten_feature_groups(panel)
    X = flatten_panel(panel)
    y = panel["y"]
    countries = panel["country_ids"]

    edu_idx = []
    for g in ("wcde_education", "bl_education", "derived"):
        edu_idx.extend(flat_groups.get(g, []))
    edu_idx = sorted(set(edu_idx))
    non_edu_idx = [i for i in range(X.shape[1]) if i not in set(edu_idx)]

    # Aggregate edu signal: mean of edu features (proxy for "level of edu")
    X_edu = X[:, edu_idx].mean(axis=1, keepdims=False)
    X_other = X[:, non_edu_idx]

    out = {"kind": "dml", "params": params, "by_seed": [], "by_target": {}}
    target_dml = {t: [] for t in targets}
    for seed in seeds:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for ti, target in enumerate(targets):
            y_t = y[:, ti]
            theta_estimates = []
            for tr_idx, te_idx in kf.split(X_other):
                # Nuisance 1: E[Y | X_other]
                g_model = GradientBoostingRegressor(
                    n_estimators=150, max_depth=4, learning_rate=0.05,
                    random_state=seed,
                )
                g_model.fit(X_other[tr_idx], y_t[tr_idx])
                resid_y = y_t[te_idx] - g_model.predict(X_other[te_idx])
                # Nuisance 2: E[edu | X_other]
                m_model = GradientBoostingRegressor(
                    n_estimators=150, max_depth=4, learning_rate=0.05,
                    random_state=seed,
                )
                m_model.fit(X_other[tr_idx], X_edu[tr_idx])
                resid_edu = X_edu[te_idx] - m_model.predict(X_other[te_idx])
                # Orthogonal score: theta = sum(resid_edu * resid_y) / sum(resid_edu^2)
                num = (resid_edu * resid_y).sum()
                den = (resid_edu * resid_edu).sum()
                theta = num / max(den, 1e-12)
                theta_estimates.append(float(theta))
            target_dml[target].append({
                "seed": seed,
                "theta_folds": theta_estimates,
                "theta_mean": float(np.mean(theta_estimates)),
                "theta_se": float(np.std(theta_estimates) / np.sqrt(n_folds)),
            })

    for t in targets:
        all_theta = [d["theta_mean"] for d in target_dml[t]]
        out["by_target"][t] = {
            "theta_seed_estimates": all_theta,
            "theta_median": float(np.median(all_theta)),
            "theta_ci95": [float(np.percentile(all_theta, 2.5)),
                            float(np.percentile(all_theta, 97.5))],
        }
    out["per_seed_per_target"] = {t: target_dml[t] for t in targets}

    with open(os.path.join(out_dir, "result.json"), "w") as f:
        _json.dump(out, f, indent=2, default=float)


def _strat_one_panel(panel, hp, kind, value, seed, prefix):
    """Stratified 5-fold OOF on a single panel (joint or single-target).
    Returns (summary_dict, keep_count, country_count). Shape-generic over
    `y.shape[1]` so it serves both joint and parent single-target panels."""
    import numpy as np
    from universal_transformer import _r2, train_one_fold

    # Build the keep-mask based on stratification kind
    if kind == "era":
        a, b = int(value[0]), int(value[1])
        keep = (panel["years"] >= a) & (panel["years"] < b)
    elif kind == "region":
        from chapter9.strat_utils import country_region_mask
        keep = country_region_mask(panel, region=value)
    elif kind == "income":
        from chapter9.strat_utils import country_income_mask
        keep = country_income_mask(panel, tier=value)
    else:
        raise ValueError(f"unknown strat_kind: {kind}")

    if int(keep.sum()) < 100:
        raise RuntimeError(f"strat too small: n={int(keep.sum())} for {kind}={value}")

    # Subset the panel (in-place dict copy)
    sub = {
        **panel,
        "X": panel["X"][keep],
        "y": panel["y"][keep],
        "mask": panel["mask"][keep],
        "country_ids": panel["country_ids"][keep],
        "years": panel["years"][keep],
    }
    # Country list must be remapped to the surviving subset
    surviving = sorted(set(int(c) for c in sub["country_ids"]))
    remap = {old: new for new, old in enumerate(surviving)}
    sub["country_ids"] = np.array([remap[int(c)] for c in sub["country_ids"]])
    sub["countries"] = [panel["countries"][o] for o in surviving]

    n_folds = 5
    if len(surviving) < n_folds:
        n_folds = max(2, len(surviving) // 2)

    fold_r2 = []
    oof_pred = np.zeros_like(sub["y"])
    oof_seen = np.zeros(sub["y"].shape[0], dtype=bool)
    for fold in range(n_folds):
        ck, val_pred, val_true, is_val = train_one_fold(
            sub, fold, hp, n_folds=n_folds, verbose=False)
        fold_r2.append(ck["val_r2_per_target"])
        oof_pred[is_val] = val_pred
        oof_seen |= is_val

    fold_r2_arr = np.array(fold_r2)
    overall_r2 = _r2(sub["y"][oof_seen], oof_pred[oof_seen])
    summary = {
        "fold_r2_mean": fold_r2_arr.mean(axis=0).tolist(),
        "fold_r2_std": fold_r2_arr.std(axis=0).tolist(),
        "overall_r2_oof": overall_r2.tolist(),
        "n_samples": int(oof_seen.sum()),
        "n_countries": len(surviving),
        "n_folds_used": n_folds,
    }
    return summary, int(keep.sum()), len(surviving)


def run_transformer_strat(params, out_dir):
    """Stratified transformer: train + evaluate on a subsample of the panel.

    params: {"strat_kind": "region"|"era"|"income",
             "strat_value": <region name or [year_start, year_end] or income tier>,
             "seed": <int>, "mode": "joint", "horizon": 25,
             "vantage": "joint"|"parent"}

    Default vantage "joint" reproduces the original multi-output behavior.
    vantage="parent" runs the same stratified 5-fold once per outcome on its
    biological-lag single-target panel and keys results under by_target.
    """
    import json as _json
    import sys as _sys
    _sys.path.insert(0, ML_DIR)
    from data_loader import load_panel
    from universal_transformer import DEFAULT_HP

    seed = int(params.get("seed", 42))
    kind = params.get("strat_kind")
    value = params.get("strat_value")
    vantage = params.get("vantage", "joint")
    # Filesystem-safe rendering of the strat value for checkpoint prefixes.
    value_tag = (str(value).replace(' ', '').replace('[', '').replace(']', '')
                 .replace(',', '-'))

    hp = dict(DEFAULT_HP)
    hp["seed"] = seed

    if vantage == "parent":
        from data_loader_parent_lag import load_parent_lag_panels
        panels = load_parent_lag_panels(mode=params.get("mode", "joint"),
                                        verbose=False)
        by_target = {}
        for tname, panel in panels.items():
            prefix = (f"strat_parent_{kind}_{value_tag}_"
                      f"{tname.lower()}_s{seed}")
            summary, keep_n, ccount = _strat_one_panel(
                panel, hp, kind, value, seed, prefix)
            by_target[tname] = {
                "summary": summary,
                "strat_size": keep_n,
                "country_count": ccount,
            }
        out = {
            "kind": "transformer_strat",
            "vantage": "parent",
            "params": params,
            "by_target": by_target,
        }
        with open(os.path.join(out_dir, "result.json"), "w") as f:
            _json.dump(out, f, indent=2, default=float)
        return

    import numpy as np
    import torch
    from universal_transformer import run_5fold

    panel = load_panel(mode=params.get("mode", "joint"), verbose=False)

    # Build the keep-mask based on stratification kind
    if kind == "era":
        a, b = int(value[0]), int(value[1])
        keep = (panel["years"] >= a) & (panel["years"] < b)
    elif kind == "region":
        # Match region from broader_features groups via country attribute.
        # The simplest approach: rely on a region-name → list-of-countries
        # mapping. Defer to a helper in chapter9/strat_utils.py.
        from chapter9.strat_utils import country_region_mask
        keep = country_region_mask(panel, region=value)
    elif kind == "income":
        from chapter9.strat_utils import country_income_mask
        keep = country_income_mask(panel, tier=value)
    else:
        raise ValueError(f"unknown strat_kind: {kind}")

    if int(keep.sum()) < 100:
        raise RuntimeError(f"strat too small: n={int(keep.sum())} for {kind}={value}")

    # Subset the panel (in-place dict copy)
    sub = {
        **panel,
        "X": panel["X"][keep],
        "y": panel["y"][keep],
        "mask": panel["mask"][keep],
        "country_ids": panel["country_ids"][keep],
        "years": panel["years"][keep],
    }
    # Country list must be remapped to the surviving subset
    surviving = sorted(set(int(c) for c in sub["country_ids"]))
    remap = {old: new for new, old in enumerate(surviving)}
    sub["country_ids"] = np.array([remap[int(c)] for c in sub["country_ids"]])
    sub["countries"] = [panel["countries"][o] for o in surviving]

    # Monkey-patch the load_panel return so run_5fold sees `sub`
    # (cleaner: extend run_5fold to accept a panel arg; for now reuse)
    from universal_transformer import (
        UniversalTransformer, _r2, _standardize_y, CHECKPOINT_DIR,
        train_one_fold,
    )
    n_folds = 5
    if len(surviving) < n_folds:
        n_folds = max(2, len(surviving) // 2)

    fold_r2 = []
    oof_pred = np.zeros_like(sub["y"])
    oof_seen = np.zeros(sub["y"].shape[0], dtype=bool)
    prefix = f"strat_{kind}_{str(value).replace(' ','').replace('[','').replace(']','').replace(',','-')}_s{seed}"
    for fold in range(n_folds):
        ck, val_pred, val_true, is_val = train_one_fold(
            sub, fold, hp, n_folds=n_folds, verbose=False)
        fold_r2.append(ck["val_r2_per_target"])
        oof_pred[is_val] = val_pred
        oof_seen |= is_val

    fold_r2_arr = np.array(fold_r2)
    overall_r2 = _r2(sub["y"][oof_seen], oof_pred[oof_seen])
    summary = {
        "fold_r2_mean": fold_r2_arr.mean(axis=0).tolist(),
        "fold_r2_std": fold_r2_arr.std(axis=0).tolist(),
        "overall_r2_oof": overall_r2.tolist(),
        "n_samples": int(oof_seen.sum()),
        "n_countries": len(surviving),
        "n_folds_used": n_folds,
    }
    out = {
        "kind": "transformer_strat",
        "params": params,
        "summary": summary,
        "strat_size": int(keep.sum()),
        "country_count": len(surviving),
    }
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        _json.dump(out, f, indent=2, default=float)


def _loo_one_country(panel, cid, hp, seed, targets):
    """Train on all-but-one country, evaluate on the held-out country, and
    return that country's per-target row. Shape-generic over `y.shape[1]`,
    so it serves both the joint (3-output) and parent single-target panels.
    Returns None if the held-out country has no samples."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from universal_transformer import (
        UniversalTransformer, _r2, _standardize_y,
    )

    is_val = panel["country_ids"] == cid
    n_val = int(is_val.sum())
    if n_val == 0:
        return None
    torch.manual_seed(seed + cid); np.random.seed(seed + cid)
    X = panel["X"]; y = panel["y"]; mask = panel["mask"]
    X_tr, X_va = X[~is_val], X[is_val]
    y_tr, y_va = y[~is_val], y[is_val]
    m_tr, m_va = mask[~is_val], mask[is_val]
    y_tr_std, y_mean, y_std = _standardize_y(y_tr)
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr_std, dtype=torch.float32)
    mt = torch.tensor(m_tr, dtype=torch.float32)
    Xv = torch.tensor(X_va, dtype=torch.float32)
    mv = torch.tensor(m_va, dtype=torch.float32)
    ds = TensorDataset(Xt, mt, yt)
    dl = DataLoader(ds, batch_size=hp["batch_size"], shuffle=True)
    model = UniversalTransformer(
        n_features=X.shape[-1], window=panel["window"],
        d_model=hp["d_model"], nhead=hp["nhead"],
        num_layers=hp["num_layers"], ff_mult=hp["ff_mult"],
        n_outputs=y.shape[1], dropout=hp["dropout"],
    )
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"],
                             weight_decay=hp["weight_decay"])
    loss_fn = torch.nn.MSELoss()
    best = -np.inf; best_state = None; no_imp = 0
    for ep in range(hp["epochs"]):
        model.train()
        for Xb, mb, yb in dl:
            opt.zero_grad()
            loss = loss_fn(model(Xb, mb), yb)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xv, mv).numpy()
        pv = pv * y_std + y_mean
        r2 = _r2(y_va, pv)
        m = float(np.mean(r2))
        if m > best:
            best = m
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if no_imp >= hp["patience"]:
            break
    # Final eval with best
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pv = model(Xv, mv).numpy() * y_std + y_mean
    r2 = _r2(y_va, pv)
    # Ablation
    edu_idx = []
    for g in ("wcde_education", "bl_education", "derived"):
        edu_idx.extend(panel["feature_groups"].get(g, []))
    X_va_ab = X_va.copy()
    for i in edu_idx:
        X_va_ab[:, :, i] = 0.0
    with torch.no_grad():
        pv_ab = model(torch.tensor(X_va_ab, dtype=torch.float32),
                      mv).numpy() * y_std + y_mean
    r2_ab = _r2(y_va, pv_ab)
    # Natural-unit prediction residuals for the held-out country. Per-
    # country R² is unstable on ~6 observations, so the trustworthy
    # "out-of-spec" signal is how far the model (trained on every OTHER
    # country) misses this country's actual LE/TFR/U5MR. resid > 0 means
    # the model OVER-predicts the outcome relative to reality.
    resid = pv - y_va                       # [n_val, n_targets]
    return {
        "country": panel["countries"][cid],
        "n_val": n_val,
        "r2_baseline": {t: float(v) for t, v in zip(targets, r2)},
        "r2_ablated_edu": {t: float(v) for t, v in zip(targets, r2_ab)},
        "r2_drop_absolute": {t: float(b - a) for t, b, a in
                              zip(targets, r2, r2_ab)},
        # nan-aware: some countries have missing actuals for a target in
        # part of their window; ignore those entries rather than letting
        # one nan collapse the whole country's residual.
        "mean_resid": {t: float(np.nanmean(resid[:, i]))
                       for i, t in enumerate(targets)},
        "mean_abs_resid": {t: float(np.nanmean(np.abs(resid[:, i])))
                           for i, t in enumerate(targets)},
        "rmse": {t: float(np.sqrt(np.nanmean(resid[:, i] ** 2)))
                 for i, t in enumerate(targets)},
    }


def _run_loo_parent(params, out_dir, hp, n_shards, shard_idx, seed, _json):
    """Parent-vantage LOO: for each held-out country, train one single-target
    model per outcome on its biological-lag panel and record per-target rows
    keyed by target name. Each shard trains 3x models per country."""
    from data_loader_parent_lag import load_parent_lag_panels

    panels = load_parent_lag_panels(mode=params.get("mode", "joint"),
                                    verbose=False)
    # All three panels share the same country index space (same loader,
    # same country list); use any one to drive the country loop.
    ref_panel = next(iter(panels.values()))
    n_countries = len(ref_panel["countries"])

    by_target = {tname: [] for tname in panels}
    for cid in range(n_countries):
        if n_shards > 1 and (cid % n_shards) != shard_idx:
            continue
        for tname, panel in panels.items():
            row = _loo_one_country(panel, cid, hp, seed, targets=[tname])
            if row is not None:
                by_target[tname].append(row)
        if (cid + 1) % 10 == 0:
            print(f"  LOO(parent) progress: {cid+1}/{n_countries} "
                  f"({ref_panel['countries'][cid]})")

    out = {
        "kind": "loo_transformer",
        "vantage": "parent",
        "params": params,
        "n_shards": n_shards,
        "shard_idx": shard_idx,
        "n_countries_processed": {t: len(rows) for t, rows in by_target.items()},
        "by_target": by_target,
    }
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        _json.dump(out, f, indent=2, default=float)


def run_loo_transformer(params, out_dir):
    """LOO-185: leave one country out at a time, train on the rest,
    evaluate on the held-out country. Returns per-country R² so the
    'no single country drives the result' claim can be made cleanly.

    This is a long-running single-VM job — loops over all countries.
    Uses reduced epochs (200 vs 400) to keep total runtime reasonable.

    Default vantage "joint" reproduces the original multi-output behavior.
    vantage="parent" runs the LOO once per biological-lag single-target
    panel, keying per-country rows under by_target.
    """
    import json as _json
    import sys as _sys
    import numpy as np
    import torch
    _sys.path.insert(0, ML_DIR)
    from data_loader import load_panel
    from universal_transformer import (
        DEFAULT_HP, UniversalTransformer, _r2, _standardize_y,
    )
    from torch.utils.data import DataLoader, TensorDataset

    seed = int(params.get("seed", 42))
    hp = dict(DEFAULT_HP)
    hp["seed"] = seed
    hp["epochs"] = 200       # halve for budget; still well past convergence
    hp["patience"] = 30

    # Sharding: each country's hold-out is independent, so the loop can be
    # split across VMs. shard_idx in [0, n_shards) takes every n_shards-th
    # country (round-robin balances data-rich/data-poor countries evenly).
    # Defaults (n_shards=1, shard_idx=0) reproduce the original full run.
    n_shards = int(params.get("n_shards", 1))
    shard_idx = int(params.get("shard_idx", 0))
    vantage = params.get("vantage", "joint")

    if vantage == "parent":
        _run_loo_parent(params, out_dir, hp, n_shards, shard_idx, seed, _json)
        return

    panel = load_panel(mode=params.get("mode", "joint"), verbose=False)
    targets = ["LE", "TFR", "U5MR"]

    per_country = []
    n_countries = len(panel["countries"])
    for cid in range(n_countries):
        if n_shards > 1 and (cid % n_shards) != shard_idx:
            continue
        row = _loo_one_country(panel, cid, hp, seed, targets)
        if row is None:
            continue
        per_country.append(row)
        if (cid + 1) % 10 == 0:
            print(f"  LOO progress: {cid+1}/{n_countries} ({panel['countries'][cid]})")

    out = {
        "kind": "loo_transformer",
        "params": params,
        "n_shards": n_shards,
        "shard_idx": shard_idx,
        "n_countries_processed": len(per_country),
        "per_country": per_country,
    }
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        _json.dump(out, f, indent=2, default=float)


def run_transformer_longrun_within(params, out_dir):
    """Within-era pre-medicine transformer: country-grouped 5-fold OOF on
    the 1885-1920 panel (outcomes 1910-1945). No modern medicine.

    params: {"tier": "core"|"extended", "seed": <int>}
    """
    import json as _json
    import sys as _sys
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import GroupKFold
    _sys.path.insert(0, ML_DIR)
    from data_loader_longrun import load_panel_longrun
    from universal_transformer import (
        DEFAULT_HP, UniversalTransformer, _r2, _standardize_y,
    )

    seed = int(params.get("seed", 42))
    tier = params.get("tier", "core")
    hp = dict(DEFAULT_HP)
    hp["seed"] = seed
    hp.update({"d_model": 32, "num_layers": 2, "nhead": 4, "ff_mult": 2,
                "dropout": 0.2, "epochs": 300, "patience": 40, "batch_size": 32})

    panel = load_panel_longrun(mode="within", tier=tier, verbose=True)
    n = panel["X"].shape[0]
    if n < 20:
        raise RuntimeError(f"within panel too small: n={n}")

    # Targets are mode-dependent: within=[LE,U5MR], forward=[LE,U5MR,TFR].
    # Read from the panel rather than hardcoding so the output keys and the
    # transformer's n_outputs match the actual y columns.
    targets = panel["target_names"]
    edu_idx = panel["feature_groups"]["wcde_education"]
    n_folds = min(5, len(panel["countries"]))
    gkf = GroupKFold(n_splits=n_folds)

    oof_pred = np.zeros_like(panel["y"])
    oof_pred_ab = np.zeros_like(panel["y"])

    for fold, (tr, te) in enumerate(gkf.split(panel["X"], panel["y"],
                                              groups=panel["country_ids"])):
        torch.manual_seed(seed + fold); np.random.seed(seed + fold)
        X_tr, X_te = panel["X"][tr], panel["X"][te]
        y_tr, y_te = panel["y"][tr], panel["y"][te]
        m_tr, m_te = panel["mask"][tr], panel["mask"][te]
        y_tr_std, y_mean, y_std = _standardize_y(y_tr)
        Xt = torch.tensor(X_tr, dtype=torch.float32)
        yt = torch.tensor(y_tr_std, dtype=torch.float32)
        mt = torch.tensor(m_tr, dtype=torch.float32)
        Xv = torch.tensor(X_te, dtype=torch.float32)
        mv = torch.tensor(m_te, dtype=torch.float32)
        ds = TensorDataset(Xt, mt, yt)
        dl = DataLoader(ds, batch_size=hp["batch_size"], shuffle=True)
        model = UniversalTransformer(
            n_features=panel["X"].shape[-1], window=panel["window"],
            d_model=hp["d_model"], nhead=hp["nhead"],
            num_layers=hp["num_layers"], ff_mult=hp["ff_mult"],
            n_outputs=panel["y"].shape[1], dropout=hp["dropout"],
        )
        opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"],
                                 weight_decay=hp["weight_decay"])
        loss_fn = nn.MSELoss()
        best = -np.inf; best_state = None; no_imp = 0
        for ep in range(hp["epochs"]):
            model.train()
            for Xb, mb, yb in dl:
                opt.zero_grad()
                loss = loss_fn(model(Xb, mb), yb)
                loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                pv = model(Xv, mv).numpy() * y_std + y_mean
            r2 = _r2(y_te, pv); mscore = float(np.mean(r2))
            if mscore > best:
                best = mscore; no_imp = 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                no_imp += 1
            if no_imp >= hp["patience"]:
                break
        # Best model: baseline + ablated predictions
        model.load_state_dict(best_state); model.eval()
        with torch.no_grad():
            pv = model(Xv, mv).numpy() * y_std + y_mean
        oof_pred[te] = pv
        X_te_ab = X_te.copy()
        for i in edu_idx:
            X_te_ab[:, :, i] = 0.0
        with torch.no_grad():
            pv_ab = model(torch.tensor(X_te_ab, dtype=torch.float32), mv).numpy() * y_std + y_mean
        oof_pred_ab[te] = pv_ab

    r2_base = _r2(panel["y"], oof_pred)
    r2_ab   = _r2(panel["y"], oof_pred_ab)
    out = {
        "kind": "transformer_longrun_within",
        "params": params,
        "tier": tier,
        "n_samples": int(n),
        "n_countries": len(panel["countries"]),
        "r2_baseline": {t: float(v) for t, v in zip(targets, r2_base)},
        "r2_ablated_edu": {t: float(v) for t, v in zip(targets, r2_ab)},
        "r2_drop_absolute": {t: float(b - a) for t, b, a in zip(targets, r2_base, r2_ab)},
    }
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        _json.dump(out, f, indent=2, default=float)


def run_transformer_longrun_forward(params, out_dir):
    """Pre→post-medicine forward extrapolation: train on pre-1945
    outcomes, test on post-1990 outcomes. Tests cross-regime extrapolation.

    params: {"tier": "core"|"extended", "seed": <int>}
    """
    import json as _json
    import sys as _sys
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _sys.path.insert(0, ML_DIR)
    from data_loader_longrun import (
        load_panel_longrun, forward_train_mask, forward_standardize,
    )
    from universal_transformer import (
        DEFAULT_HP, UniversalTransformer, _r2, _standardize_y,
    )

    seed = int(params.get("seed", 42))
    tier = params.get("tier", "core")
    hp = dict(DEFAULT_HP)
    hp["seed"] = seed
    hp.update({"d_model": 32, "num_layers": 2, "nhead": 4, "ff_mult": 2,
                "dropout": 0.2, "epochs": 300, "patience": 40,
                "batch_size": 32})

    panel = load_panel_longrun(mode="forward", tier=tier, verbose=True)
    panel = forward_standardize(panel)   # train-only z-score, no leakage
    is_train = forward_train_mask(panel)
    n_train = int(is_train.sum())
    n_test  = int((~is_train).sum())
    if n_train < 30 or n_test < 30:
        raise RuntimeError(f"forward split too small: train={n_train} test={n_test}")

    torch.manual_seed(seed); np.random.seed(seed)
    X = panel["X"]; y = panel["y"]; mask = panel["mask"]
    Xtr, Xte = X[is_train], X[~is_train]
    ytr, yte = y[is_train], y[~is_train]
    mtr, mte = mask[is_train], mask[~is_train]
    y_tr_std, y_mean, y_std = _standardize_y(ytr)

    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(y_tr_std, dtype=torch.float32)
    mt = torch.tensor(mtr, dtype=torch.float32)
    Xv = torch.tensor(Xte, dtype=torch.float32)
    mv = torch.tensor(mte, dtype=torch.float32)
    ds = TensorDataset(Xt, mt, yt)
    dl = DataLoader(ds, batch_size=hp["batch_size"], shuffle=True)

    model = UniversalTransformer(
        n_features=X.shape[-1], window=panel["window"],
        d_model=hp["d_model"], nhead=hp["nhead"],
        num_layers=hp["num_layers"], ff_mult=hp["ff_mult"],
        n_outputs=y.shape[1], dropout=hp["dropout"],
    )
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"],
                             weight_decay=hp["weight_decay"])
    loss_fn = nn.MSELoss()
    best = -np.inf; best_state = None; no_imp = 0
    for ep in range(hp["epochs"]):
        model.train()
        for Xb, mb, yb in dl:
            opt.zero_grad()
            loss = loss_fn(model(Xb, mb), yb)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xv, mv).numpy() * y_std + y_mean
        r2 = _r2(yte, pv); m = float(np.mean(r2))
        if m > best:
            best = m
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if no_imp >= hp["patience"]:
            break

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        pv = model(Xv, mv).numpy() * y_std + y_mean
    r2_base = _r2(yte, pv)
    # Ablation
    edu_idx = panel["feature_groups"]["wcde_education"]
    X_te_ab = Xte.copy()
    for i in edu_idx:
        X_te_ab[:, :, i] = 0.0
    with torch.no_grad():
        pv_ab = model(torch.tensor(X_te_ab, dtype=torch.float32), mv).numpy() * y_std + y_mean
    r2_ab = _r2(yte, pv_ab)

    # Mode-dependent target set (forward = [LE, U5MR, TFR]); read from panel.
    targets = panel["target_names"]
    out = {
        "kind": "transformer_longrun_forward",
        "params": params, "tier": tier,
        "n_train": n_train, "n_test": n_test,
        "r2_baseline": {t: float(v) for t, v in zip(targets, r2_base)},
        "r2_ablated_edu": {t: float(v) for t, v in zip(targets, r2_ab)},
        "r2_drop_absolute": {t: float(b - a) for t, b, a in
                              zip(targets, r2_base, r2_ab)},
    }
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        _json.dump(out, f, indent=2, default=float)


def run_panel_fe_longrun_within(params, out_dir):
    """OLS country-FE + GBM baseline on the within-era panel.

    params: {"tier": "core"|"extended", "seed": <int>}
    """
    import json as _json
    import sys as _sys
    _sys.path.insert(0, ML_DIR)
    _sys.path.insert(0, os.path.join(ML_DIR, "longrun"))
    from data_loader_longrun import load_panel_longrun
    from panel_fe import run_country_fe_ols, run_gbm_with_edu_ablation

    seed = int(params.get("seed", 42))
    tier = params.get("tier", "core")
    panel = load_panel_longrun(mode="within", tier=tier, verbose=True)
    if panel["X"].shape[0] < 20:
        raise RuntimeError(f"within panel too small: n={panel['X'].shape[0]}")

    # Within-mode panel carries [LE, U5MR] only; iterate the actual targets.
    targets = panel["target_names"]
    ols = {t: run_country_fe_ols(panel, target_name=t, seed=seed) for t in targets}
    gbm = {t: run_gbm_with_edu_ablation(panel, target_name=t, seed=seed) for t in targets}

    out = {
        "kind": "panel_fe_longrun_within",
        "params": params, "tier": tier,
        "n_samples": int(panel["X"].shape[0]),
        "n_countries": len(panel["countries"]),
        "ols_by_target": ols,
        "gbm_by_target": gbm,
    }
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        _json.dump(out, f, indent=2, default=float)


def run_panel_fe_longrun_forward(params, out_dir):
    """OLS country-FE + GBM trained on pre-1945, tested on post-1990.

    params: {"tier": "core"|"extended", "seed": <int>}
    """
    import json as _json
    import sys as _sys
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import GradientBoostingRegressor
    _sys.path.insert(0, ML_DIR)
    _sys.path.insert(0, os.path.join(ML_DIR, "longrun"))
    from data_loader_longrun import (
        load_panel_longrun, forward_train_mask, forward_standardize,
    )
    from panel_fe import _flatten_window, _sum_edu_features

    seed = int(params.get("seed", 42))
    tier = params.get("tier", "core")
    panel = load_panel_longrun(mode="forward", tier=tier, verbose=True)
    panel = forward_standardize(panel)
    is_train = forward_train_mask(panel)
    # Forward-mode panel carries [LE, U5MR, TFR]; read from panel so the y
    # columns and the per-target output keys stay aligned.
    targets = panel["target_names"]
    X_flat = _flatten_window(panel)
    edu = _sum_edu_features(X_flat, panel)
    edu_cols = panel["feature_groups"]["wcde_education"]

    ols_results = {}
    gbm_results = {}
    for ti, t in enumerate(targets):
        y = panel["y"][:, ti]
        # OLS: train on pre-1945, predict on post-1990 using train country means
        df_tr = pd.DataFrame({
            "y": y[is_train], "edu": edu[is_train],
            "cid": panel["country_ids"][is_train],
        })
        df_tr["y_demean"]   = df_tr["y"] - df_tr.groupby("cid")["y"].transform("mean")
        df_tr["edu_demean"] = df_tr["edu"] - df_tr.groupby("cid")["edu"].transform("mean")
        num = (df_tr["y_demean"] * df_tr["edu_demean"]).sum()
        den = (df_tr["edu_demean"] ** 2).sum()
        beta = float(num / max(den, 1e-12))
        country_means = df_tr.groupby("cid")["y"].mean().to_dict()
        edu_means = df_tr.groupby("cid")["edu"].mean().to_dict()
        # Predict on test
        y_te = y[~is_train]
        cid_te = panel["country_ids"][~is_train]
        edu_te = edu[~is_train]
        pred = np.zeros_like(y_te)
        for i, c in enumerate(cid_te):
            c_mean = country_means.get(c, np.mean(list(country_means.values())))
            e_mean = edu_means.get(c, np.mean(list(edu_means.values())))
            pred[i] = c_mean + beta * (edu_te[i] - e_mean)
        ss_tot = ((y_te - y_te.mean()) ** 2).sum()
        r2 = float(1.0 - ((y_te - pred) ** 2).sum() / max(ss_tot, 1e-12))
        ols_results[t] = {"beta_edu": beta, "r2_oof_forward": r2,
                          "n_train": int(is_train.sum()),
                          "n_test": int((~is_train).sum())}

        # GBM
        gbm = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                          learning_rate=0.05, random_state=seed)
        gbm.fit(X_flat[is_train], y[is_train])
        pred_gbm = gbm.predict(X_flat[~is_train])
        X_te_ab = X_flat[~is_train].copy()
        X_te_ab[:, edu_cols] = 0.0
        pred_gbm_ab = gbm.predict(X_te_ab)
        r2_base = float(1.0 - ((y_te - pred_gbm) ** 2).sum() / max(ss_tot, 1e-12))
        r2_ab = float(1.0 - ((y_te - pred_gbm_ab) ** 2).sum() / max(ss_tot, 1e-12))
        gbm_results[t] = {"r2_baseline": r2_base, "r2_ablated_edu": r2_ab,
                          "r2_drop": r2_base - r2_ab}

    out = {
        "kind": "panel_fe_longrun_forward",
        "params": params, "tier": tier,
        "n_train": int(is_train.sum()),
        "n_test": int((~is_train).sum()),
        "ols_by_target": ols_results,
        "gbm_by_target": gbm_results,
    }
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        _json.dump(out, f, indent=2, default=float)


def run_optuna(params, out_dir):
    """Optuna architecture search over transformer hyperparameters.

    Search space: d_model {32,64,128}, num_layers {1,2,3,4},
                  dropout [0.1, 0.5], lr log-uniform [1e-4, 1e-2],
                  weight_decay log-uniform [1e-5, 1e-2], ff_mult {2, 4}.

    Objective: 5-fold OOF R² mean over (LE, TFR, U5MR), country-clustered.
    Reports best config + edu R² drop for top-k configs.
    """
    import json as _json
    import sys as _sys
    import numpy as np
    _sys.path.insert(0, ML_DIR)
    try:
        import optuna
    except ImportError:
        import subprocess as _sp
        _sp.check_call(["pip", "install", "--quiet", "optuna"])
        import optuna
    from data_loader import load_panel
    from universal_transformer import DEFAULT_HP, run_5fold
    from run_universal_evidence import run_g1_g2

    total_trials = int(params.get("n_trials", 500))
    base_seed = int(params.get("seed", 42))

    # Sharding: split the trial budget across independent studies, each with
    # a distinct seed so the samplers explore different regions. The global
    # best across shards is selected at aggregation. Independent multi-start
    # search parallelizes cleanly and is more robust than one TPE study that
    # can get stuck. Defaults (n_shards=1, shard_idx=0) = original behavior.
    n_shards = int(params.get("n_shards", 1))
    shard_idx = int(params.get("shard_idx", 0))
    n_trials = total_trials // n_shards
    if shard_idx < (total_trials % n_shards):
        n_trials += 1
    # Large per-shard seed stride so seed + trial.number ranges never overlap.
    seed = base_seed + shard_idx * 100000

    vantage = params.get("vantage", "joint")

    # Parent-vantage objective: maximize the mean over the three biological-lag
    # single-target 5-fold R² (LE@12, TFR@5, U5MR@12), reusing _run_5fold_single
    # per target. Same search space as the joint objective.
    parent_panels = None
    if vantage == "parent":
        from data_loader_parent_lag import load_parent_lag_panels
        from run_universal_evidence_parent_lag import _run_5fold_single
        parent_panels = load_parent_lag_panels(
            mode=params.get("mode", "joint"), verbose=False)

    panel = load_panel(mode="joint", verbose=False)
    targets = ["LE", "TFR", "U5MR"]
    all_results = []

    def _suggest_hp(trial):
        hp = dict(DEFAULT_HP)
        hp["seed"] = seed + trial.number     # vary seed per trial for diversity
        hp["epochs"] = 200                   # short for search
        hp["patience"] = 30
        hp["d_model"]    = trial.suggest_categorical("d_model", [32, 64, 128])
        hp["num_layers"] = trial.suggest_int("num_layers", 1, 4)
        hp["nhead"]      = trial.suggest_categorical("nhead", [2, 4, 8])
        hp["ff_mult"]    = trial.suggest_categorical("ff_mult", [2, 4])
        hp["dropout"]    = trial.suggest_float("dropout", 0.1, 0.5)
        hp["lr"]         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        hp["weight_decay"] = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        # d_model must be divisible by nhead
        if hp["d_model"] % hp["nhead"] != 0:
            raise optuna.TrialPruned()
        return hp

    def objective(trial):
        hp = _suggest_hp(trial)
        if vantage == "parent":
            # Mean over the three single-target biological-lag 5-fold R².
            per_target_r2 = []
            for tname, p in parent_panels.items():
                prefix = f"optuna_parent_{tname.lower()}_t{trial.number:04d}"
                s = _run_5fold_single(p, hp, ckpt_prefix=prefix, verbose=False)
                # Single-target panel: fold_r2_mean is a length-1 list.
                per_target_r2.append(float(np.mean(s["fold_r2_mean"])))
            return float(np.mean(per_target_r2))
        prefix = f"optuna_t{trial.number:04d}"
        summary = run_5fold(mode="joint", hp=hp, verbose=False,
                             checkpoint_prefix=prefix)
        return float(np.mean(summary["fold_r2_mean"]))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    out = {
        "kind": "optuna",
        "vantage": vantage,
        "params": params,
        "n_shards": n_shards,
        "shard_idx": shard_idx,
        "n_trials": n_trials,
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "all_values": [t.value for t in study.trials if t.value is not None],
        "all_params": [t.params for t in study.trials if t.value is not None],
    }
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        _json.dump(out, f, indent=2, default=float)


def run_counterfactual(params, out_dir):
    """Country-counterfactual education swaps on the parent-vantage models.

    Reuses the single-target bio-lag fold checkpoints the transformer_single
    jobs already trained and uploaded to GCS (one per outcome × seed, at
    LE@T+12 / TFR@T+5 / U5MR@T+12). For each seed it pulls that seed's three
    checkpoint sets and runs out-of-fold swaps — each country is scored by the
    fold in which it was held out, so it is never in that model's training set.
    Gaps are then averaged across seeds.

    params: {"pairs": [[a, b], ...] or null (all default pairs),
             "seeds": [<int>, ...] (default [42])}
    """
    import json as _json
    import subprocess
    import sys as _sys
    import numpy as np
    _sys.path.insert(0, ML_DIR)
    _sys.path.insert(0, os.path.join(ML_DIR, "chapter9"))
    from counterfactuals import run_counterfactuals_oof
    from universal_transformer import CHECKPOINT_DIR

    seeds = list(params.get("seeds", [42]))
    pairs = params.get("pairs", None)
    if pairs is not None:
        # JSON delivers pairs as lists; the OOF runner indexes them as tuples.
        pairs = [tuple(p) for p in pairs]
    targets = ["LE", "TFR", "U5MR"]

    gcs_root = f"gs://{os.environ.get('BUCKET') or os.environ['GCS_BUCKET']}/results"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    by_seed = {}
    for seed in seeds:
        # Pull this seed's three single-target fold checkpoint sets. The job id
        # is upper-cased per the manifest; the checkpoint files inside are
        # lower-cased (trf_parent_le_s42_fold0.pt, ...). The trailing-slash glob
        # lands the files flat in CHECKPOINT_DIR, where the OOF reader expects
        # them via ckpt_prefix_fmt.
        for tgt in targets:
            gcs_dir = f"{gcs_root}/trf_parent_{tgt}_s{seed}/checkpoints/"
            subprocess.run(["gsutil", "-q", "-m", "cp", f"{gcs_dir}*",
                            CHECKPOINT_DIR + "/"], check=False)
        try:
            by_seed[f"seed_{seed}"] = run_counterfactuals_oof(
                pairs=pairs,
                ckpt_prefix_fmt=f"trf_parent_{{target}}_s{seed}",
                verbose=False)
        except FileNotFoundError as e:
            by_seed[f"seed_{seed}"] = {"error": str(e)}

    # Seed-averaged education-attributable gap per pair per outcome.
    agg = {}
    for res in by_seed.values():
        for pair_key, gaps in res.get("gaps_by_pair", {}).items():
            d = agg.setdefault(pair_key, {})
            for t, v in gaps.items():
                d.setdefault(t, []).append(v)
    gaps_by_pair_mean = {pk: {t: float(np.mean(vs)) for t, vs in d.items()}
                         for pk, d in agg.items()}
    gaps_by_pair_std = {pk: {t: float(np.std(vs)) for t, vs in d.items()}
                        for pk, d in agg.items()}

    out = {
        "kind": "counterfactual",
        "params": params,
        "seeds": seeds,
        "n_seeds_ok": sum(1 for r in by_seed.values() if "error" not in r),
        "gaps_by_pair_mean": gaps_by_pair_mean,
        "gaps_by_pair_std": gaps_by_pair_std,
        "by_seed": by_seed,
    }
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        _json.dump(out, f, indent=2, default=float)


def _run_cond_perm_parent(params, out_dir, n_repeats, seed, _json):
    """Parent-vantage conditional permutation importance. Loads the three
    biological-lag single-target panels, flattens each, and trains the GBM on
    that panel's single y column at its own horizon. Keeps the n_repeats
    permutation logic identical to the joint runner."""
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor
    from data_loader import country_holdout_mask
    from data_loader_parent_lag import load_parent_lag_panels
    from spec_curve import flatten_panel, flatten_feature_groups

    # Shard by target: params["targets"] subsets the outcomes (one per shard);
    # None runs all three biological-lag outcomes.
    targets = params.get("targets") or ["LE", "TFR", "U5MR"]
    panels = load_parent_lag_panels(mode=params.get("mode", "joint"),
                                    verbose=False)
    rng = np.random.default_rng(seed)
    summary = {}
    for target in targets:
        panel = panels[target]
        flat_groups = flatten_feature_groups(panel)
        X = flatten_panel(panel)
        y = panel["y"][:, 0]            # single-target column at its horizon
        edu_idx = []
        for g in ("wcde_education", "bl_education", "derived"):
            edu_idx.extend(flat_groups.get(g, []))
        edu_idx = sorted(set(edu_idx))

        is_val = country_holdout_mask(panel, n_folds=5, fold=0, seed=seed)
        X_tr, X_va = X[~is_val], X[is_val]
        y_tr, y_va = y[~is_val], y[is_val]
        model = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            random_state=seed,
        )
        model.fit(X_tr, y_tr)
        baseline_pred = model.predict(X_va)
        ss_res = ((y_va - baseline_pred) ** 2).sum()
        ss_tot = ((y_va - y_va.mean()) ** 2).sum()
        r2_base = 1.0 - ss_res / max(ss_tot, 1e-12)

        drops = []
        for rep in range(n_repeats):
            X_perm = X_va.copy()
            for col in edu_idx:
                X_perm[:, col] = rng.permutation(X_perm[:, col])
            pred_perm = model.predict(X_perm)
            ss_res_p = ((y_va - pred_perm) ** 2).sum()
            r2_perm = 1.0 - ss_res_p / max(ss_tot, 1e-12)
            drops.append(float(r2_base - r2_perm))
        summary[target] = {
            "drop_mean": float(np.mean(drops)),
            "drop_std": float(np.std(drops)),
            "drop_ci95": [float(np.percentile(drops, 2.5)),
                           float(np.percentile(drops, 97.5))],
        }
    out = {"kind": "cond_perm", "vantage": "parent",
           "params": params, "by_target": summary}
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        _json.dump(out, f, indent=2, default=float)


def run_cond_perm(params, out_dir):
    """Conditional permutation importance (Strobl et al. 2008).
    Permutes education features WITHIN conditional strata to avoid
    overstating importance under feature correlation."""
    import json as _json
    import sys as _sys
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor
    _sys.path.insert(0, ML_DIR)
    _sys.path.insert(0, os.path.join(ML_DIR, "chapter9"))
    from data_loader import country_holdout_mask, load_panel
    from spec_curve import flatten_panel, flatten_feature_groups

    n_repeats = int(params.get("n_repeats", 50))
    seed = int(params.get("seed", 42))
    vantage = params.get("vantage", "joint")

    if vantage == "parent":
        _run_cond_perm_parent(params, out_dir, n_repeats, seed, _json)
        return

    panel = load_panel(mode="joint", verbose=False)
    flat_groups = flatten_feature_groups(panel)
    X = flatten_panel(panel)
    y = panel["y"]
    # Shard by target: params["targets"] subsets the outcomes (one per shard);
    # None runs all three. Each model is trained per target, so a target is an
    # independent unit of work. Aggregation merges drops_per_target.
    targets = params.get("targets") or ["LE", "TFR", "U5MR"]

    edu_idx = []
    for g in ("wcde_education", "bl_education", "derived"):
        edu_idx.extend(flat_groups.get(g, []))
    edu_idx = sorted(set(edu_idx))

    # Canonical outcome column order in panel["y"] — a sharded `targets`
    # subset must still index its true column, not its position in the subset.
    ALL_TARGETS = ["LE", "TFR", "U5MR"]
    col_of = {t: i for i, t in enumerate(ALL_TARGETS)}

    rng = np.random.default_rng(seed)
    drops_per_target = {t: [] for t in targets}

    # Train once on the full panel; permute on a held-out fold
    is_val = country_holdout_mask(panel, n_folds=5, fold=0, seed=seed)
    X_tr, X_va = X[~is_val], X[is_val]
    y_tr, y_va = y[~is_val], y[is_val]
    for target in targets:
        ti = col_of[target]
        model = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            random_state=seed,
        )
        model.fit(X_tr, y_tr[:, ti])
        baseline_pred = model.predict(X_va)
        ss_res = ((y_va[:, ti] - baseline_pred) ** 2).sum()
        ss_tot = ((y_va[:, ti] - y_va[:, ti].mean()) ** 2).sum()
        r2_base = 1.0 - ss_res / max(ss_tot, 1e-12)

        for rep in range(n_repeats):
            X_perm = X_va.copy()
            for col in edu_idx:
                X_perm[:, col] = rng.permutation(X_perm[:, col])
            pred_perm = model.predict(X_perm)
            ss_res_p = ((y_va[:, ti] - pred_perm) ** 2).sum()
            r2_perm = 1.0 - ss_res_p / max(ss_tot, 1e-12)
            drops_per_target[target].append(float(r2_base - r2_perm))

    summary = {}
    for t, drops in drops_per_target.items():
        summary[t] = {
            "drop_mean": float(np.mean(drops)),
            "drop_std": float(np.std(drops)),
            "drop_ci95": [float(np.percentile(drops, 2.5)),
                           float(np.percentile(drops, 97.5))],
        }
    out = {"kind": "cond_perm", "params": params, "summary": summary}
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        _json.dump(out, f, indent=2, default=float)


def run_bloc_holdout(params, out_dir):
    """Leave-a-WHOLE-BLOC-out test. Train the transformer on every country
    NOT in `holdout_countries`, then predict the held-out bloc. Reports each
    held-out country's residual (LE, TFR, U5MR, log U5MR) in natural units and
    in sigma (vs the non-bloc out-of-sample residual spread).

    Reproduces the paper's Moscow-Meridian phenotype test with a blind model:
    a transformer that has never seen a Soviet republic still flags the eastern
    republics' reported education as hollow. Sign convention matches the paper
    (residual = actual - predicted): LE negative and log-U5MR positive = the
    population's phenotype falls short of what its reported schooling predicts.
    """
    import json as _json
    import sys as _sys
    import numpy as np
    import torch
    _sys.path.insert(0, ML_DIR)
    from data_loader import load_panel
    from universal_transformer import (
        DEFAULT_HP, UniversalTransformer, _r2, _standardize_y,
    )
    from torch.utils.data import DataLoader, TensorDataset

    holdout = set(c.strip().lower() for c in params["holdout_countries"])
    seed = int(params.get("seed", 42))
    hp = dict(DEFAULT_HP)
    hp["seed"] = seed
    hp["epochs"] = 300
    hp["patience"] = 30

    panel = load_panel(mode=params.get("mode", "joint"), verbose=False)
    targets = ["LE", "TFR", "U5MR"]
    u5_i = targets.index("U5MR")
    countries_l = [c.lower() for c in panel["countries"]]
    cid_holdout = np.array([countries_l[c] in holdout
                            for c in range(len(panel["countries"]))])
    samp_cid = panel["country_ids"]

    rng = np.random.default_rng(seed)
    # Non-holdout countries split 80/20 BY COUNTRY: train on 80%, use the 20%
    # as an honest out-of-sample baseline for the residual spread (sigma).
    nonhold = [c for c in range(len(panel["countries"])) if not cid_holdout[c]]
    rng.shuffle(nonhold)
    n_val = max(1, int(0.2 * len(nonhold)))
    val_cids = set(nonhold[:n_val])
    train_cids = set(nonhold[n_val:])
    is_train = np.array([samp_cid[i] in train_cids for i in range(len(samp_cid))])
    is_sigval = np.array([samp_cid[i] in val_cids for i in range(len(samp_cid))])

    X, y, mask = panel["X"], panel["y"], panel["mask"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    y_tr_std, y_mean, y_std = _standardize_y(y[is_train])
    Xt = torch.tensor(X[is_train], dtype=torch.float32)
    yt = torch.tensor(y_tr_std, dtype=torch.float32)
    mt = torch.tensor(mask[is_train], dtype=torch.float32)
    dl = DataLoader(TensorDataset(Xt, mt, yt), batch_size=hp["batch_size"],
                    shuffle=True)
    model = UniversalTransformer(
        n_features=X.shape[-1], window=panel["window"], d_model=hp["d_model"],
        nhead=hp["nhead"], num_layers=hp["num_layers"], ff_mult=hp["ff_mult"],
        n_outputs=y.shape[1], dropout=hp["dropout"],
    )
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"],
                            weight_decay=hp["weight_decay"])
    loss_fn = torch.nn.MSELoss()
    Xv = torch.tensor(X[is_sigval], dtype=torch.float32)
    mv = torch.tensor(mask[is_sigval], dtype=torch.float32)
    yv = y[is_sigval]
    best = -np.inf
    best_state = None
    no_imp = 0
    for ep in range(hp["epochs"]):
        model.train()
        for Xb, mb, yb in dl:
            opt.zero_grad()
            loss_fn(model(Xb, mb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xv, mv).numpy() * y_std + y_mean
        m = float(np.mean(_r2(yv, pv)))
        if m > best:
            best = m
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if no_imp >= hp["patience"]:
            break
    model.load_state_dict(best_state)
    model.eval()

    def _safelog(a):
        return np.log(np.clip(a, 1e-6, None))

    # Baseline residual spread (sigma) on the held-out non-bloc countries.
    with torch.no_grad():
        pv = model(Xv, mv).numpy() * y_std + y_mean
    rv = yv - pv                                   # actual - predicted
    sigma = {t: float(np.nanstd(rv[:, i])) for i, t in enumerate(targets)}
    sigma_logu5 = float(np.nanstd(_safelog(yv[:, u5_i]) - _safelog(pv[:, u5_i])))

    per_country = []
    for cid in range(len(panel["countries"])):
        if not cid_holdout[cid]:
            continue
        sel = samp_cid == cid
        if sel.sum() == 0:
            continue
        Xh = torch.tensor(X[sel], dtype=torch.float32)
        mh = torch.tensor(mask[sel], dtype=torch.float32)
        yh = y[sel]
        with torch.no_grad():
            ph = model(Xh, mh).numpy() * y_std + y_mean
        resid = yh - ph
        logr = _safelog(yh[:, u5_i]) - _safelog(ph[:, u5_i])
        per_country.append({
            "country": panel["countries"][cid],
            "n": int(sel.sum()),
            "mean_resid": {t: float(np.nanmean(resid[:, i]))
                           for i, t in enumerate(targets)},
            "sigma_resid": {t: (float(np.nanmean(resid[:, i]) / sigma[t])
                                if sigma[t] > 0 else float("nan"))
                            for i, t in enumerate(targets)},
            "logU5MR_resid": float(np.nanmean(logr)),
            "logU5MR_sigma": (float(np.nanmean(logr) / sigma_logu5)
                              if sigma_logu5 > 0 else float("nan")),
        })

    out = {
        "kind": "bloc_holdout",
        "params": params,
        "n_train_countries": len(train_cids),
        "n_sigma_baseline_countries": len(val_cids),
        "sigma": sigma,
        "sigma_logu5": sigma_logu5,
        "per_country": per_country,
    }
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        _json.dump(out, f, indent=2, default=float)


def run_residual_atlas(params, out_dir):
    """Retrain the three single-target bio-lag transformers (5-fold, one seed)
    at LE@12 / TFR@5 / U5MR@12, then build the per-country residual atlas.

    The trained checkpoints (universal_parent_lag_{le,tfr,u5mr}_fold*) are the
    primary deliverable: they feed BOTH the chatbot's residual-atlas tool and
    the Civ game's inference. The atlas JSON is the secondary deliverable.

    params: {"seed": <int>}  (default 42)
    Outputs into out_dir: result.json, residual_atlas_per_country.json,
    residual_atlas.json, ml_gate_g4.json, and checkpoints/.
    """
    import json as _json
    import shutil
    import sys as _sys
    import torch  # noqa: F401  — fail loudly here if the VM lacks torch
    _sys.path.insert(0, ML_DIR)
    from universal_transformer import DEFAULT_HP
    from run_universal_evidence_parent_lag import _run_5fold_single
    from data_loader_parent_lag import load_parent_lag_panels, PARENT_LAG_HORIZONS
    import residual_atlas as ra

    seed = int(params.get("seed", 42))
    hp = dict(DEFAULT_HP)
    hp["seed"] = seed
    hp["verbose"] = False

    # 1. Retrain the three single-target models at their biological horizons,
    #    saving under the canonical local prefix the atlas + game both expect.
    panels = load_parent_lag_panels(mode="joint", verbose=False)
    summaries = {}
    for tname in ("LE", "TFR", "U5MR"):
        prefix = f"universal_parent_lag_{tname.lower()}"
        summaries[tname] = _run_5fold_single(
            panels[tname], hp, ckpt_prefix=prefix, verbose=True)

    # 2. Build the atlas from the freshly-trained checkpoints.
    long_df, sig, cluster_method = ra.compute_atlas()
    ra.write_outputs(long_df, sig, cluster_method)
    g4 = ra.gate_g4(sig)

    out = {
        "kind": "residual_atlas",
        "params": params,
        "horizons": dict(PARENT_LAG_HORIZONS),
        "seed": seed,
        "n_countries": int(len(sig)),
        "cluster_method": cluster_method,
        "fold_summaries": summaries,
        "g4_pass": g4.get("pass"),
    }
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        _json.dump(out, f, indent=2, default=float)

    # 3. Copy atlas JSONs + checkpoints into out_dir for GCS upload.
    ml_checkin = os.path.join(ML_DIR, "checkin")
    for fname in ("residual_atlas_per_country.json", "residual_atlas.json",
                  "ml_gate_g4.json"):
        src = os.path.join(ml_checkin, fname)
        if os.path.exists(src):
            shutil.copy(src, out_dir)
    ckpt_src = os.path.join(ML_DIR, "checkpoints")
    ckpt_dst = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dst, exist_ok=True)
    for fname in os.listdir(ckpt_src):
        if fname.startswith("universal_parent_lag_"):
            shutil.copy(os.path.join(ckpt_src, fname), ckpt_dst)


DISPATCH = {
    "smoke": run_smoke,
    "residual_atlas": run_residual_atlas,
    "transformer": run_transformer,
    "transformer_single": run_transformer_single,
    "entry_cohort_refit": run_entry_cohort_refit,
    "entry_cohort_no_geo": run_entry_cohort_no_geo,
    "transformer_walk_forward": run_transformer_walk_forward,
    "transformer_strat": run_transformer_strat,
    "loo_transformer": run_loo_transformer,
    "bloc_holdout": run_bloc_holdout,
    "transformer_longrun_within":  run_transformer_longrun_within,
    "transformer_longrun_forward": run_transformer_longrun_forward,
    "panel_fe_longrun_within":     run_panel_fe_longrun_within,
    "panel_fe_longrun_forward":    run_panel_fe_longrun_forward,
    "spec_curve": run_spec_curve,
    "placebos": run_placebos,
    "dml": run_dml,
    "optuna": run_optuna,
    "counterfactual": run_counterfactual,
    "cond_perm": run_cond_perm,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--kind", required=True)
    ap.add_argument("--params", default="{}")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    try:
        params = json.loads(args.params or "{}")
    except json.JSONDecodeError:
        params = {}

    if args.kind not in DISPATCH:
        print(f"ERROR: unknown job kind: {args.kind}")
        with open(os.path.join(args.output_dir, "error.json"), "w") as f:
            json.dump({"error": "unknown_kind", "kind": args.kind}, f)
        sys.exit(2)

    t0 = time.time()
    print(f"[run_job] kind={args.kind} id={args.job_id} params={params}")
    try:
        DISPATCH[args.kind](params, args.output_dir)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[run_job] EXCEPTION:\n{tb}")
        with open(os.path.join(args.output_dir, "error.json"), "w") as f:
            json.dump({"error": str(e), "traceback": tb}, f)
        sys.exit(3)
    print(f"[run_job] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

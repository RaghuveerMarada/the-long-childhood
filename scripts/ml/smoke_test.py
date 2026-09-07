# =============================================================================
# scripts/ml/smoke_test.py
#
# Fast wiring check for the ML half of the repo. Answers one question:
# on a clean clone, after `make setup`, can a reader build the panels the
# transformer trains on?
#
# Before this existed the answer was no, for three separate reasons, none
# of which surfaced until someone tried:
#   1. torch and scikit-learn were absent from requirements.txt.
#   2. data/co2_emissions_tonnes_per_person.csv is not redistributed, and
#      the loader raised FileNotFoundError on it.
#   3. openpyxl was absent, so the Maddison backfill could not be built.
#
# Usage:
#   make ml-smoke
#   python scripts/ml/smoke_test.py
#
# Exit code 0 = the ML pipeline is runnable. Nonzero = it is not, with
# the reason named.
# =============================================================================
"""Import, data-presence and panel-shape checks for scripts/ml/."""

import os
import sys
import time
import warnings

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, SCRIPT_DIR)

FAILURES = []
NOTES = []


def check(label, fn):
    t0 = time.time()
    try:
        detail = fn()
    except Exception as exc:
        FAILURES.append((label, f"{type(exc).__name__}: {exc}"))
        print(f"  [FAIL] {label}\n         {type(exc).__name__}: {exc}")
        return None
    dt = time.time() - t0
    suffix = f" — {detail}" if detail else ""
    print(f"  [ok]   {label} ({dt:.1f}s){suffix}")
    return detail


def _check_imports():
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import sklearn
    import torch
    return f"torch {torch.__version__}, scikit-learn {sklearn.__version__}"


def _check_required_data():
    """Files without which no ML panel can be built at all."""
    from _shared import DATA, PROC
    required = [
        os.path.join(DATA, "external", "country_latlong.csv"),
        os.path.join(DATA, "external", "country_codes.csv"),
        os.path.join(DATA, "external", "religion_composition_pew.csv"),
        os.path.join(DATA, "external", "maddison_gdppc_wb_equivalent.csv"),
        os.path.join(DATA, "external", "wb_oilrents.json"),
        os.path.join(DATA, "external", "wb_malaria.json"),
        os.path.join(DATA, "external", "wb_trade.json"),
        os.path.join(DATA, "colonial_global", "global_country_table.csv"),
        os.path.join(DATA, "ajr2001", "ajr_n61_country_table.csv"),
        os.path.join(DATA, "p5v2018.xls"),
        os.path.join(DATA, "barro_lee_v3.csv"),
        os.path.join(DATA, "gdppercapita_us_inflation_adjusted.csv"),
        os.path.join(DATA, "life_expectancy_years.csv"),
        os.path.join(DATA, "children_per_woman_total_fertility.csv"),
        os.path.join(DATA, "child_mortality_u5.csv"),
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        rel = [os.path.relpath(p, REPO_ROOT) for p in missing]
        raise FileNotFoundError("missing required inputs: " + ", ".join(rel))
    if not os.path.isdir(PROC):
        raise FileNotFoundError(f"missing WCDE processed dir: {PROC}")
    return f"{len(required)} required inputs present"


def _check_optional_data():
    """Files whose absence degrades a specific test but not the pipeline."""
    from _shared import DATA
    co2 = os.path.join(DATA, "co2_emissions_tonnes_per_person.csv")
    if os.path.exists(co2):
        return "CO2 series present"
    NOTES.append(
        "data/co2_emissions_tonnes_per_person.csv absent. The CO2 control "
        "is all-NaN and the CO2 placebo test is unavailable. Run "
        "`python scripts/ml/fetch_co2.py` to add it."
    )
    return "CO2 series absent (optional; placebo test disabled)"


def _check_broader_features():
    from broader_features import BroaderFeatures
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bf = BroaderFeatures()
    vec = bf.features_at("India", 1990)
    n = len(bf.feature_names()) if hasattr(bf, "feature_names") else len(vec)
    return f"{n} non-education features, co2_available={bf.co2_available}"


def _check_panels():
    from data_loader_parent_lag import load_parent_lag_panels, PARENT_LAG_HORIZONS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        panels = load_parent_lag_panels(mode="joint", verbose=False)

    bits = []
    for name in ("LE", "TFR", "U5MR"):
        if name not in panels:
            raise KeyError(f"panel {name} not returned")
        p = panels[name]
        X, y = p["X"], p["y"]
        if X.ndim != 3:
            raise ValueError(f"{name}: X should be [N, window, F], got {X.shape}")
        if y.shape[1] != 1:
            raise ValueError(f"{name}: y should be single-target, got {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"{name}: X/y row mismatch {X.shape[0]} vs {y.shape[0]}")
        if X.shape[0] < 500:
            raise ValueError(f"{name}: only {X.shape[0]} samples — panel looks truncated")
        if p["pred_horizon"] != PARENT_LAG_HORIZONS[name]:
            raise ValueError(
                f"{name}: horizon {p['pred_horizon']} != expected "
                f"{PARENT_LAG_HORIZONS[name]}"
            )
        bits.append(f"{name} n={X.shape[0]} @T+{p['pred_horizon']}")
    return "; ".join(bits)


def _check_forward_pass():
    """One untrained forward pass — catches architecture/shape breakage
    without spending the minutes a real fold costs."""
    import numpy as np
    import torch
    from data_loader_parent_lag import load_parent_lag_panels
    from universal_transformer import UniversalTransformer, DEFAULT_HP

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        panel = load_parent_lag_panels(mode="joint", verbose=False)["LE"]

    hp = DEFAULT_HP
    model = UniversalTransformer(
        n_features=panel["X"].shape[-1], window=panel["window"],
        d_model=hp["d_model"], nhead=hp["nhead"], num_layers=hp["num_layers"],
        ff_mult=hp["ff_mult"], n_outputs=1, dropout=hp["dropout"],
    )
    model.eval()
    Xb = torch.tensor(panel["X"][:16], dtype=torch.float32)
    mb = torch.tensor(panel["mask"][:16], dtype=torch.float32)
    with torch.no_grad():
        out = model(Xb, mb).numpy()
    if out.shape != (16, 1):
        raise ValueError(f"forward pass returned {out.shape}, expected (16, 1)")
    if not np.isfinite(out).all():
        raise ValueError("forward pass produced non-finite values")
    n_params = sum(p.numel() for p in model.parameters())
    return f"forward pass ok, {n_params:,} params"


def main():
    print("ML smoke test — scripts/ml/\n")
    check("third-party imports", _check_imports)
    check("required data files", _check_required_data)
    check("optional data files", _check_optional_data)
    check("BroaderFeatures builds", _check_broader_features)
    check("parent-vantage panels build", _check_panels)
    check("transformer forward pass", _check_forward_pass)

    if NOTES:
        print("\nNotes:")
        for n in NOTES:
            print(f"  - {n}")

    if FAILURES:
        print(f"\nFAILED: {len(FAILURES)} check(s)")
        for label, msg in FAILURES:
            print(f"  {label}: {msg}")
        return 1
    print("\nPASS — the ML pipeline is runnable on this machine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

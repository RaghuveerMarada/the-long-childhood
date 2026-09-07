# =============================================================================
# scripts/ml/cloud/job_manifest.py
#
# Defines the BIG-RUN job set: every ML test we want for Chapter 9 of
# "The Long Childhood." Designed to fill a ~$50-300 spot-VM budget with
# maximum-rigor evidence that the panel's education-load-bearing finding
# is method-agnostic, sample-agnostic, and survives every reasonable
# falsification.
#
# Spot T4 pricing in us-central1: ~$0.105/hr
# Spot n2-standard-8: ~$0.078/hr
#
# Components (all on spot VMs; preempt-recovery in launcher):
#   A.  Transformer T+25            — 50 seeds × joint
#   B.  Transformer parent-vantage  — 30 seeds × 3 outcomes
#   C.  Walk-forward                — 8 cutoffs × 5 seeds
#   D.  Region stratification       — 6 regions × 5 seeds
#   E.  Era stratification          — 5 eras × 5 seeds
#   F.  Income stratification       — 3 tiers × 5 seeds
#   G.  LOO-185 transformer (1 sd)  — 185 jobs, single seed each
#   H.  Long-run 1875→2015 forecast — 10 seeds
#   I.  Barro-Lee replication       — 15 seeds
#   J.  Spec curve                  — 1 CPU job (5 methods, 5 seeds internal)
#   K.  Placebos                    — 1 CPU job (6 placebos, 4 methods, 5 sd)
#   L.  Double / debiased ML        — 1 CPU job (5 seeds × 5-fold)
#   M.  Optuna architecture search  — 1 GPU job, 500 trials
#   N.  Counterfactual swaps        — 1 CPU job (uses pretrained ckpts)
#   O.  Conditional perm. importance- 1 CPU job
# =============================================================================
"""
job_manifest.py

Single canonical big-run manifest. No phase split.
"""

import json
import os
from typing import Iterable

# ── Configuration ────────────────────────────────────────────────────────────

TRANSFORMER_SEEDS = list(range(42, 92))         # 50 seeds for the headline
PARENT_VANTAGE_SEEDS = list(range(42, 72))      # 30 seeds for parent-vantage

WALK_FORWARD_CUTOFFS = [1965, 1975, 1980, 1985, 1990, 1995, 2000, 2005]
WALK_FORWARD_SEEDS = list(range(42, 47))

REGIONS = [
    "sub_saharan_africa", "south_asia", "east_asia_pacific",
    "latin_america_caribbean", "europe_central_asia",
    "middle_east_north_africa",
]
REGION_SEEDS = list(range(42, 47))

ERAS = [
    (1960, 1975), (1975, 1990), (1990, 2005),
    (2005, 2020), (1960, 2020),
]
ERA_SEEDS = list(range(42, 47))

INCOME_TIERS = ["low", "lower_middle", "upper_middle_plus"]
INCOME_SEEDS = list(range(42, 47))

LOO_SEEDS = [42]   # single seed; 185 countries already give 185 estimates
# LOO is embarrassingly parallel — each held-out country is independent, so
# the atom is one country. With a persistent worker pool (~48 workers) the
# per-job boot tax is gone, so we shard to the atom: one country per job.
# This packs the full fleet (vs. 20 shards leaving most workers idle on the
# LOO tail) and makes each preemption cost ~10 min, not ~1.5 h.
LOO_N_SHARDS = 185

LONGRUN_SEEDS = list(range(42, 52))   # 10 seeds

BARROLEE_SEEDS = list(range(42, 57))  # 15 seeds for BL replication

ENTRY_COHORT_REFIT_SEEDS = list(range(42, 47))  # 5 seeds, diagnostic tier

OPTUNA_TRIALS = 500
OPTUNA_SEEDS = [42]   # one Optuna search; trials internally varied
# Split the 500-trial budget across independent studies (distinct seeds),
# ~5 trials each (~0.35 GPU-hr / ~20 min). Many small studies pack the full
# worker pool and survive preemption cheaply; the global-best config across
# all shards is selected at aggregation (independent multi-start search).
OPTUNA_N_SHARDS = 100

COUNTERFACTUAL_PAIRS = [
    ("south korea", "philippines"),
    ("taiwan", "philippines"),
    ("cuba", "dominican republic"),
    ("sri lanka", "pakistan"),
    ("bangladesh", "pakistan"),
    ("china", "india"),
    ("vietnam", "myanmar"),
    ("botswana", "zimbabwe"),
    ("rwanda", "burundi"),
    ("kenya", "tanzania"),
    ("south africa", "zimbabwe"),
    ("malaysia", "philippines"),
    ("singapore", "malaysia"),
    ("japan", "south korea"),
    ("turkey", "iran"),
    ("indonesia", "vietnam"),
    ("ghana", "nigeria"),
    ("ethiopia", "kenya"),
    ("egypt", "morocco"),
    ("mexico", "brazil"),
]


# ── Job factories ────────────────────────────────────────────────────────────

def _trf_t25_jobs():
    """A. Transformer T+25 multi-target, joint mode, 50 seeds."""
    out = []
    for s in TRANSFORMER_SEEDS:
        out.append({
            "id": f"trf_t25_s{s}",
            "kind": "transformer",
            "params": {"mode": "joint", "horizon": 25, "seed": s},
            "est_gpu_hours": 0.45,
            "est_cpu_hours": 0.0,
        })
    return out


def _trf_parent_jobs():
    """B. Transformer parent-vantage, 30 seeds × 3 outcomes."""
    out = []
    for s in PARENT_VANTAGE_SEEDS:
        # Horizons match data_loader_parent_lag.PARENT_LAG_HORIZONS (LE@12,
        # TFR@5, U5MR@12). run_job loads via that loader and ignores this
        # value; kept in sync for accurate est/metadata.
        for tgt, hzn in (("LE", 12), ("TFR", 5), ("U5MR", 12)):
            out.append({
                "id": f"trf_parent_{tgt}_s{s}",
                "kind": "transformer_single",
                "params": {"target": tgt, "horizon": hzn, "seed": s},
                "est_gpu_hours": 0.20,
                "est_cpu_hours": 0.0,
            })
    return out


def _walk_forward_jobs():
    """C. Walk-forward at 8 cutoffs × 5 seeds."""
    out = []
    for c in WALK_FORWARD_CUTOFFS:
        for s in WALK_FORWARD_SEEDS:
            out.append({
                "id": f"walk_c{c}_s{s}",
                "kind": "transformer_walk_forward",
                "params": {"mode": "joint", "horizon": 25, "seed": s,
                           "cutoff_year": c},
                "est_gpu_hours": 0.40,
                "est_cpu_hours": 0.0,
            })
    return out


def _region_strat_jobs():
    """D. Region stratification: transformer trained on each region only."""
    out = []
    for r in REGIONS:
        for s in REGION_SEEDS:
            out.append({
                "id": f"strat_region_{r}_s{s}",
                "kind": "transformer_strat",
                "params": {"strat_kind": "region", "strat_value": r,
                           "seed": s, "mode": "joint", "horizon": 25},
                "est_gpu_hours": 0.20,
                "est_cpu_hours": 0.0,
            })
    return out


def _era_strat_jobs():
    """E. Era stratification."""
    out = []
    for (a, b) in ERAS:
        for s in ERA_SEEDS:
            out.append({
                "id": f"strat_era_{a}_{b}_s{s}",
                "kind": "transformer_strat",
                "params": {"strat_kind": "era", "strat_value": [a, b],
                           "seed": s, "mode": "joint", "horizon": 25},
                "est_gpu_hours": 0.30,
                "est_cpu_hours": 0.0,
            })
    return out


def _income_strat_jobs():
    """F. Income tier stratification (low / lower-middle / upper-middle+)."""
    out = []
    for tier in INCOME_TIERS:
        for s in INCOME_SEEDS:
            out.append({
                "id": f"strat_income_{tier}_s{s}",
                "kind": "transformer_strat",
                "params": {"strat_kind": "income", "strat_value": tier,
                           "seed": s, "mode": "joint", "horizon": 25},
                "est_gpu_hours": 0.20,
                "est_cpu_hours": 0.0,
            })
    return out


def _loo_jobs():
    """G. LOO-185 transformer: hold out one country, train on the rest.
    Single seed per country; 185 countries gives 185 estimates.
    Sharded across LOO_N_SHARDS VMs (round-robin over countries)."""
    out = []
    for s in LOO_SEEDS:
        for shard in range(LOO_N_SHARDS):
            out.append({
                "id": f"loo_shard_{shard:02d}of{LOO_N_SHARDS}_s{s}",
                "kind": "loo_transformer",
                "params": {"seed": s, "mode": "joint", "horizon": 25,
                           "n_shards": LOO_N_SHARDS, "shard_idx": shard},
                # ~185 countries / 20 shards ≈ 9-10 each; ~30 GPU-hr / 20
                "est_gpu_hours": 30.0 / LOO_N_SHARDS,
                "est_cpu_hours": 0.0,
            })
    return out


def _optuna_jobs():
    """M. Optuna architecture search, sharded into OPTUNA_N_SHARDS
    independent studies (distinct seeds). Global best chosen at aggregation."""
    out = []
    for shard in range(OPTUNA_N_SHARDS):
        out.append({
            "id": f"optuna_shard_{shard:02d}of{OPTUNA_N_SHARDS}",
            "kind": "optuna",
            "params": {"n_trials": OPTUNA_TRIALS, "seed": 42,
                       "n_shards": OPTUNA_N_SHARDS, "shard_idx": shard},
            "est_gpu_hours": 35.0 / OPTUNA_N_SHARDS,
            "est_cpu_hours": 0.0,
        })
    return out


def _longrun_jobs():
    """H. Longrun (pre-medicine) test: 4 kinds × 5 seeds = 20 jobs.

    Two findings: (A) within-era 1885-1920 pre-medicine (LE, U5MR);
    (B) train pre-1950, predict 1960-2015 (LE, U5MR, TFR). Two model
    families per finding: transformer + panel-FE baseline (OLS + GBM).
    The panel-FE kinds are CPU-only; the transformer kinds want a GPU.
    """
    out = []
    kinds = [
        "transformer_longrun_within",
        "transformer_longrun_forward",
        "panel_fe_longrun_within",
        "panel_fe_longrun_forward",
    ]
    seeds = list(LONGRUN_SEEDS)[:5]   # first 5 seeds
    for kind in kinds:
        is_panel = kind.startswith("panel_fe")
        short = (kind.replace("transformer_longrun_", "t")
                     .replace("panel_fe_longrun_", "f"))
        for s in seeds:
            out.append({
                "id": f"longrun_{short}_s{s}",
                "kind": kind,
                "params": {"tier": "core", "seed": s},
                "est_gpu_hours": 0.0 if is_panel else 0.5,
                "est_cpu_hours": 0.5 if is_panel else 0.0,
            })
    return out


def _barrolee_jobs():
    """I. Replicate the T+25 finding on Barro-Lee education series only."""
    out = []
    for s in BARROLEE_SEEDS:
        out.append({
            "id": f"trf_bl_s{s}",
            "kind": "transformer",
            "params": {"mode": "bl_only", "horizon": 25, "seed": s},
            "est_gpu_hours": 0.45,
            "est_cpu_hours": 0.0,
        })
    return out


# Shard axes for the CPU consolidation jobs. Each monolith exceeds the
# orchestrator's 2-hour watchdog, so they are split along a natural
# independent axis (method / placebo / seed / target) into sub-2h shards
# that also run concurrently. Aggregation merges the shards back.
SPEC_CURVE_METHODS = ["ridge", "lasso",
                      "random_forest", "gradient_boosting"]
PLACEBO_FNS = ["placebo_outcome_scramble", "placebo_year_scramble",
               "placebo_country_scramble", "placebo_edu_replaced_noise",
               "placebo_edu_replaced_latitude", "placebo_edu_replaced_gdprank"]
PLACEBO_SEEDS = [42, 43, 44, 45, 46]
DML_SEEDS = list(range(42, 47))
COND_PERM_TARGETS = ["LE", "TFR", "U5MR"]


def _cpu_jobs():
    """J–O. CPU-only consolidation jobs, sharded for the 2h watchdog."""
    jobs = []

    # J. Spec curve — one shard per method (~0.6 CPU-hr each).
    for m in SPEC_CURVE_METHODS:
        jobs.append({
            "id": f"spec_curve_{m}",
            "kind": "spec_curve",
            "params": {"methods": [m], "seeds": list(range(42, 47))},
            "est_gpu_hours": 0.0,
            "est_cpu_hours": 3.0 / len(SPEC_CURVE_METHODS),
        })

    # K. Placebos — one shard per placebo (~1.3 CPU-hr each).
    for fn in PLACEBO_FNS:
        short = fn.replace("placebo_", "")
        jobs.append({
            "id": f"placebos_{short}",
            "kind": "placebos",
            "params": {"placebos": [fn], "seeds": PLACEBO_SEEDS},
            "est_gpu_hours": 0.0,
            "est_cpu_hours": 8.0 / len(PLACEBO_FNS),
        })

    # L. Double / debiased ML — one shard per seed (~0.5 CPU-hr each).
    for s in DML_SEEDS:
        jobs.append({
            "id": f"dml_s{s}",
            "kind": "dml",
            "params": {"n_folds": 5, "seeds": [s]},
            "est_gpu_hours": 0.0,
            "est_cpu_hours": 2.5 / len(DML_SEEDS),
        })

    # N. Counterfactual swaps — pulls the per-seed trf_parent_{LE,TFR,U5MR}
    #    fold checkpoints the transformer_single jobs already trained, runs
    #    out-of-fold education swaps (forward inference only), and averages the
    #    gaps across all parent-vantage seeds. CPU-only; left whole.
    jobs.append({
        "id": "counterfactual_big",
        "kind": "counterfactual",
        "params": {"pairs": COUNTERFACTUAL_PAIRS,
                   "seeds": list(PARENT_VANTAGE_SEEDS)},
        "est_gpu_hours": 0.0,
        "est_cpu_hours": 1.5,
    })

    # O. Conditional permutation importance — one shard per target (~1.3 CPU-hr).
    for t in COND_PERM_TARGETS:
        jobs.append({
            "id": f"cond_perm_{t}",
            "kind": "cond_perm",
            "params": {"n_repeats": 50, "targets": [t]},
            "est_gpu_hours": 0.0,
            "est_cpu_hours": 4.0 / len(COND_PERM_TARGETS),
        })

    return jobs


# ── Parent-vantage recast jobs ──────────────────────────────────────────────
# Biological-lag (parent-vantage) recasts of the cloud diagnostics that were
# originally joint multi-output at horizon 25. Each job carries
# params["vantage"]="parent"; run_job loads the three single-target panels
# (LE@T+12, TFR@T+5, U5MR@T+12) and runs the diagnostic once per outcome,
# emitting per-target results. The ids all contain "_parent" so the whole
# set is filterable with launch.py --only.
#
# Compute note: a parent job trains 3x the single-target models a joint job
# trains (one per outcome), but each single-target model is cheaper than the
# 3-output joint model, so the per-job estimate is held at the joint estimate
# (a conservative upper bound for budgeting).

# Parent-vantage Barro-Lee: same BL seeds as the joint trf_bl job, run as
# single-target biological-lag models on BL-only features (mode="bl_only").
BARROLEE_PARENT_SEEDS = list(BARROLEE_SEEDS)


def parent_vantage_recast_jobs():
    """Biological-lag (parent-vantage) recasts of the cloud diagnostics.

    Returns job entries (same schema as the big-run factories) for:
      walk-forward, region/era/income stratification, LOO-185, optuna,
      conditional permutation importance, placebos, and Barro-Lee — all in
      parent vantage. Ids are prefixed so `launch.py --only ..._parent`
      selects exactly this set. Does NOT touch big_run_jobs / panel_jobs.
    """
    out = []

    # C(parent). Walk-forward at 8 cutoffs x 5 seeds. Per-target handled inside
    # run_job, so still one job per cutoff x seed.
    for c in WALK_FORWARD_CUTOFFS:
        for s in WALK_FORWARD_SEEDS:
            out.append({
                "id": f"walk_parent_c{c}_s{s}",
                "kind": "transformer_walk_forward",
                "params": {"mode": "joint", "horizon": 25, "seed": s,
                           "cutoff_year": c, "vantage": "parent"},
                "est_gpu_hours": 0.40,
                "est_cpu_hours": 0.0,
            })

    # D(parent). Region stratification: 6 regions x 5 seeds.
    for r in REGIONS:
        for s in REGION_SEEDS:
            out.append({
                "id": f"strat_parent_region_{r}_s{s}",
                "kind": "transformer_strat",
                "params": {"strat_kind": "region", "strat_value": r,
                           "seed": s, "mode": "joint", "horizon": 25,
                           "vantage": "parent"},
                "est_gpu_hours": 0.20,
                "est_cpu_hours": 0.0,
            })

    # E(parent). Era stratification: 5 eras x 5 seeds.
    for (a, b) in ERAS:
        for s in ERA_SEEDS:
            out.append({
                "id": f"strat_parent_era_{a}_{b}_s{s}",
                "kind": "transformer_strat",
                "params": {"strat_kind": "era", "strat_value": [a, b],
                           "seed": s, "mode": "joint", "horizon": 25,
                           "vantage": "parent"},
                "est_gpu_hours": 0.30,
                "est_cpu_hours": 0.0,
            })

    # F(parent). Income-tier stratification: 3 tiers x 5 seeds.
    for tier in INCOME_TIERS:
        for s in INCOME_SEEDS:
            out.append({
                "id": f"strat_parent_income_{tier}_s{s}",
                "kind": "transformer_strat",
                "params": {"strat_kind": "income", "strat_value": tier,
                           "seed": s, "mode": "joint", "horizon": 25,
                           "vantage": "parent"},
                "est_gpu_hours": 0.20,
                "est_cpu_hours": 0.0,
            })

    # G(parent). LOO-185: one country per shard, single seed. Each shard now
    # trains 3x models (one per outcome), so the per-shard estimate is tripled.
    for s in LOO_SEEDS:
        for shard in range(LOO_N_SHARDS):
            out.append({
                "id": f"loo_parent_shard_{shard:02d}of{LOO_N_SHARDS}_s{s}",
                "kind": "loo_transformer",
                "params": {"seed": s, "mode": "joint", "horizon": 25,
                           "n_shards": LOO_N_SHARDS, "shard_idx": shard,
                           "vantage": "parent"},
                "est_gpu_hours": 3.0 * 30.0 / LOO_N_SHARDS,
                "est_cpu_hours": 0.0,
            })

    # M(parent). Optuna architecture search, sharded into independent studies.
    # Objective is the mean over the three biological-lag single-target 5-fold
    # R². Same shard count as the joint search; each trial trains 3x models.
    for shard in range(OPTUNA_N_SHARDS):
        out.append({
            "id": f"optuna_parent_shard_{shard:02d}of{OPTUNA_N_SHARDS}",
            "kind": "optuna",
            "params": {"n_trials": OPTUNA_TRIALS, "seed": 42,
                       "n_shards": OPTUNA_N_SHARDS, "shard_idx": shard,
                       "vantage": "parent"},
            "est_gpu_hours": 3.0 * 35.0 / OPTUNA_N_SHARDS,
            "est_cpu_hours": 0.0,
        })

    # O(parent). Conditional permutation importance — one job, all targets.
    out.append({
        "id": "cond_perm_parent",
        "kind": "cond_perm",
        "params": {"n_repeats": 50, "targets": COND_PERM_TARGETS,
                   "vantage": "parent"},
        "est_gpu_hours": 0.0,
        "est_cpu_hours": 4.0,
    })

    # K(parent). Placebos — one shard per placebo (6 shards), parent vantage.
    for fn in PLACEBO_FNS:
        short = fn.replace("placebo_", "")
        out.append({
            "id": f"placebos_parent_{short}",
            "kind": "placebos",
            "params": {"placebos": [fn], "seeds": PLACEBO_SEEDS,
                       "vantage": "parent"},
            "est_gpu_hours": 0.0,
            # 3x the joint placebo shard (three single-target base panels).
            "est_cpu_hours": 3.0 * 8.0 / len(PLACEBO_FNS),
        })

    # I(parent). Barro-Lee replication in parent vantage: single-target
    # biological-lag models on BL-only features, one job per (target, seed).
    for s in BARROLEE_PARENT_SEEDS:
        for tgt, hzn in (("LE", 12), ("TFR", 5), ("U5MR", 12)):
            out.append({
                "id": f"trf_bl_parent_{tgt}_s{s}",
                "kind": "transformer_single",
                "params": {"target": tgt, "horizon": hzn, "seed": s,
                           "mode": "bl_only", "vantage": "parent"},
                "est_gpu_hours": 0.20,
                "est_cpu_hours": 0.0,
            })

    # Normalize every id to a single leading `parent_` prefix so the whole
    # recast set is selectable with `launch.py --only parent_` (which matches
    # on id startswith). The per-block ids above embed "parent" mid-string for
    # readability; collapse that to a clean leading prefix here (e.g.
    # walk_parent_c1965_s42 -> parent_walk_c1965_s42).
    for j in out:
        core = j["id"].replace("parent_", "").replace("_parent", "")
        j["id"] = "parent_" + core

    return out


def big_run_jobs():
    """Return the full big-run job list."""
    return (
        _trf_t25_jobs()
        + _trf_parent_jobs()
        + _walk_forward_jobs()
        + _region_strat_jobs()
        + _era_strat_jobs()
        + _income_strat_jobs()
        + _loo_jobs()
        + _longrun_jobs()
        + _barrolee_jobs()
        + _optuna_jobs()
        + _cpu_jobs()
    )


# Alias used by launch.py: the Panel job set is the full big run.
def entry_cohort_refit_jobs():
    """Diagnostic tier: fair G1-gate refit (education excluded at TRAINING
    time, not zeroed at inference) restricted to the entry-cohort [10%, 90%]
    expansion window. 5 seeds x 3 targets = 15 GPU jobs.

    See scripts/diagnostics/G1_GATE_INVESTIGATION.md. Not part of the
    Chapter 9 big run; ids are prefixed "entrycohort_" so `launch.py panel
    --only entry_cohort_refit` selects exactly this set."""
    out = []
    for t in ("LE", "TFR", "U5MR"):
        for s in ENTRY_COHORT_REFIT_SEEDS:
            out.append({
                "id": f"entrycohort_{t.lower()}_s{s}",
                "kind": "entry_cohort_refit",
                "params": {"target": t, "seed": s},
                # two full single-target retrains (5-fold each) on a
                # smaller (entry-cohort-restricted) sample than the joint
                # transformer_single job; same order of magnitude cost.
                "est_gpu_hours": 0.4,
                "est_cpu_hours": 0.0,
            })
    return out


def entry_cohort_no_geo_jobs():
    """Follow-up to entry_cohort_refit_jobs (G1_GATE_INVESTIGATION.md SS6):
    same entry-cohort-restricted no_education refit, with the time-invariant
    ("geography") broader-feature blocks also dropped from the feature set
    (not just row-filtered). Same 5 seeds x 3 targets. Ids prefixed
    "entrycohortgeo_" so `launch.py panel --only entry_cohort_no_geo`
    selects exactly this set."""
    out = []
    for t in ("LE", "TFR", "U5MR"):
        for s in ENTRY_COHORT_REFIT_SEEDS:
            out.append({
                "id": f"entrycohortgeo_{t.lower()}_s{s}",
                "kind": "entry_cohort_no_geo",
                "params": {"target": t, "seed": s},
                "est_gpu_hours": 0.2,   # one retrain, not two
                "est_cpu_hours": 0.0,
            })
    return out


def panel_jobs():
    # big_run_jobs() is the original joint big run. The parent-vantage recast
    # jobs are appended so launch.py can reach them; every launch of the recast
    # filters with `--only parent_`, so they are never selected unless asked
    # for. (big_run_manifest.json is written from big_run_jobs() directly in
    # main(), so it is unaffected by this append.) entry_cohort_refit_jobs()
    # and entry_cohort_no_geo_jobs() are the same pattern: appended so
    # launch.py can reach them, only ever selected via their own --only kind.
    return (big_run_jobs() + parent_vantage_recast_jobs()
            + entry_cohort_refit_jobs() + entry_cohort_no_geo_jobs())


def manifest_summary(jobs):
    by_kind = {}
    total_gpu = 0.0
    total_cpu = 0.0
    for j in jobs:
        by_kind.setdefault(j["kind"], 0)
        by_kind[j["kind"]] += 1
        total_gpu += j.get("est_gpu_hours", 0.0)
        total_cpu += j.get("est_cpu_hours", 0.0)
    # Spot pricing (us-central1): T4 $0.105/hr, n2-standard-8 $0.078/hr
    cost_spot = total_gpu * 0.105 + total_cpu * 0.078
    cost_ondemand = total_gpu * 0.35 + total_cpu * 0.39
    return {
        "n_jobs": len(jobs),
        "by_kind": by_kind,
        "total_gpu_hours": round(total_gpu, 1),
        "total_cpu_hours": round(total_cpu, 1),
        "est_cost_usd_spot": round(cost_spot, 2),
        "est_cost_usd_ondemand": round(cost_ondemand, 2),
        # 30% buffer for preempt re-runs
        "est_cost_usd_spot_with_buffer": round(cost_spot * 1.3, 2),
    }


def main():
    jobs = big_run_jobs()
    summary = manifest_summary(jobs)
    print("BIG RUN MANIFEST:")
    print(json.dumps(summary, indent=2))
    # Write
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "big_run_manifest.json")
    with open(out_path, "w") as f:
        json.dump({"jobs": jobs, "summary": summary}, f, indent=2)
    print(f"\nWrote {out_path} ({len(jobs)} jobs)")


if __name__ == "__main__":
    main()

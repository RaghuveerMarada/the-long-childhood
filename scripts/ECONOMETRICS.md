# Econometric Appendix

*The Long Childhood* is a policy argument, not an econometrics paper. The
body text carries one headline regression, the residualised-GDP result, the
forward-prediction table, and the country histories. The remainder of the
econometric workup — TWFE diagnostics, modern-DiD estimators, IV contests,
panel-unit-root and cross-section-dependence batteries, placebo and
permutation nulls, lag and threshold sweeps, listwise-deletion stress tests,
and the ML head-to-head — was moved out of the PDF to keep the paper
readable for a policy audience. **The code is still in the repository, and
this file is its index.** A reader who wants the full panel workup can run
or inspect every test from here.

**For economists: Chapter 9 anchors.** Chapter 9 (*The Panel*) names its
tools and points here for the construction. The map below lets an econ
reader land on the producing script for every claim in the chapter in
under a minute:

| Chapter 9 location | Claim / artefact | Producing script |
|---|---|---|
| §What the Panel Sharpens | Chapter scope and apparatus pointer | (this file) |
| §The Generational Lag, three-timescale table | Lag-0 / lag-10–15 / lag-28 outcome mapping | (descriptive; no producing script) |
| §The Generational Lag, MAC = 28.8 anchor | Cohort-weighted MAC across the expansion-phase panel | `wcde/mean_age_first_birth_invariance.py` |
| Parent/grandparent R² decomposition (was inline in §The Generational Lag) | Per-outcome within-R² increment from each generational block | `residualization/horse_race_lags_levels.py` (detail below) |
| §The Convergence in Aggregate, Table~\ref{tab:summary} | Period means, six core variables | `wcde/summary_stats.py` |
| §Education vs GDP, Table~\ref{tab:headline} (β_g = 1.434) | Active-expansion headline + bad-control check | `tables/table_1_stepwise.py` |
| §Education vs GDP, Table~\ref{tab:cutoff} | Edu vs GDP at parental-education cutoffs | `residualization/by_gdp_cutoff.py` |
| §Education vs GDP, Figure~\ref{fig:beta-baseline} (β_g curve) | β_g amplification vs ceiling compression | `wcde/long_run_generational.py` |
| §Composition by level, HLO partition | Primary→TFR, quality→LE / U5MR (z-scored) | `wcde/hanushek_horse_race.py` |
| §GDP residualised, Table~\ref{tab:residualisation} | Edu R² vs residualised-GDP R² across four outcomes | `residualization/education_vs_gdp.py` |
| §GDP residualised, Figure~\ref{fig:edu-vs-resid-gdp} | Side-by-side edu vs residualised-GDP bars | `residualization/education_vs_gdp.py` |
| §GDP residualised (Notes), unique-R² decomposition | Edu / GDP unique-R² above the other alone | `residualization/unique_r2_decomposition.py` |
| §GDP residualised (Notes), Maddison-backfilled variant | Robustness to WB→Maddison GDP backfill | `residualization/unique_r2_decomposition_maddison.py` |
| §GDP residualised (Notes), common-sample variant | All four outcomes on the shared n=645 panel | `residualization/residualization_common_sample.py` |
| §The one bounded exception (U5MR / Lutz reading) | GDP unique-R² on log U5MR ≤ 0.002 at every ceiling | `residualization/unique_r2_ceiling_sweep.py` |
| §What rules out chance, permutation null (53 SDs) | Country-shuffle null for the headline β | `robustness/permutation_null.py` |
| §The Shape of the Response, Figure~\ref{fig:lag-100} | Standardised |β| at lags 0–100 across four outcomes | `robustness/lag_coefficients.py` |
| §The Shape of the Response, grandparent channel | Grandparent education adds R² above parent at low baselines | `robustness/grandparent_effect_all_outcomes.py` |
| §Universality, 15-subsample β | Region × era × tercile subsample β re-estimations | `robustness/beta_by_baseline_group.py` |
| §Universality, Table~\ref{tab:over-performers} | Eight 2015 FE residual over-performers | `tables/regression_tables.py` |
| §What the Panel Does Not Carry | 20-test panel-econometrics battery | `econometric_battery/REPORT.md` |
| §What the Panel Does Not Carry | Robustness sweeps catalogue | `robustness/` (this index, below) |

Every script listed above also carries a `PAPER ANCHOR` header in its
docstring naming the Chapter 9 location, the artefact produced, and the
checkin JSON. The bar is: an economist scanning the file's first ten lines
sees what panel claim they are looking at and where the numbers live.

Every script writes a JSON to `checkin/` or `scripts/<dir>/checkin/` so its
numbers are inspectable without re-running. Every script can be re-run
directly (`python scripts/<dir>/<name>.py`) or in bulk via the relevant
Makefile target.

The directory layout splits the work by purpose:

| Directory                       | Purpose                                                                |
|--------------------------------|------------------------------------------------------------------------|
| `residualization/`             | Frisch-Waugh-Lovell strips education's contribution from GDP, asks whether residual GDP predicts anything. The headline result. |
| `robustness/`                  | Specification, identification, and sensitivity checks for the panel β. |
| `econometric_battery/`         | Twenty named panel-econometrics diagnostics (Pesaran CD, unit roots, cointegration, dynamic-panel GMM, modern DiD, etc.). |
| `ml/`                          | Encoder-only transformer trained blind on the panel — used as a non-parametric cross-check of the linear results, plus designed falsification. |

---

## `residualization/` — Frisch-Waugh-Lovell

The single most provocative result in the paper: once you strip education's
contribution from GDP, residual GDP predicts no development outcome. Every
script here implements that procedure on a different slice.

| Script | What it does |
|---|---|
| `by_gdp_cutoff.py` | Headline table. Education R² vs residualised-GDP R² at parental-education cutoffs from <10% to no cutoff. Education leads GDP by 2–3.4×. |
| `by_entry_threshold.py` | Entry-cohort sample (countries from first crossing of 10/20/30% lower-sec); asks the same question on each entry-cohort window. |
| `by_entry_ceiling.py`  | Adds a ceiling: only observations between entry and ceiling are kept. The "active expansion" subsample. |
| `by_education_level.py` | Re-runs entry-cohort + ceiling for primary, lower-secondary, upper-secondary completion separately. |
| `education_vs_gdp.py`  | Full FWL sweep for life expectancy — education vs GDP across thresholds and ceilings. |
| `education_vs_tfr.py`  | Same FWL sweep for total fertility. |
| `education_predicts_le.py` | Headline question on the unrestricted sample: if you know education today, how well can you predict LE in 28 years vs the same forecast from GDP? |
| `child_education.py`   | Same residualisation but with child education at T+28 as the outcome — intergenerational PT, not health-or-fertility. |
| `cutoff_all_outcomes.py` | Generalisation across LE, TFR, log-U5MR, child education. |
| `female_education.py`  | Re-runs every outcome with female-only completion, compares to both-sexes. |
| `symmetry_test.py`     | Answers the "FWL asymmetry" objection: residualise education on GDP instead, ask whether residual education predicts. It does; reverse residual GDP doesn't. |

---

## `robustness/` — Specification, identification, sensitivity

| Script | What it does |
|---|---|
| `robustness_tests.py` | Bootstrap CIs on the headline β across log LE / log TFR-primary / log U5MR / level child-edu (1,000-rep country-resample, 95% percentile CIs); Anderson-Hsiao IV as a Nickell-bias diagnostic for the child-education spec (caveat: instrument $E_{T-1}$ and outcome $E_{T+28}$ are drawn from the same within-country education series, so the exclusion restriction is in tension with the grandparent-channel finding — read as descriptive, not identifying); nonlinearity check. The "if a reviewer has 30 minutes" script. |
| `balanced_panel.py` | Re-estimates Table 1 col 1 on a balanced subpanel — every country in every five-year cohort 1975-2015. |
| `barro_lee_replication.py` | Re-runs forward-prediction on Barro-Lee education data instead of WCDE — rules out WCDE reconstruction artefact. |
| `period_length.py` | Repeats Table 1 col 1 at 5-, 10-, 25-year aggregation horizons. |
| `lag_sensitivity.py`  | Residualised GDP R² at lags 15, 20, 25, 30 — zero-GDP result is not a single-lag artefact. |
| `lag_coefficients.py` | Standardised β of lower-sec completion on each outcome at lags 0…100, plus GDP lag sweep for comparison. |
| `threshold_robustness.py` | Development-threshold crossing dates under loose (4.5, 65), main (3.65, 69.8), strict (2.5, 72.6) specs. Ordering never changes. |
| `cs_threshold_robustness.py` | Sweeps the entry threshold for the Callaway–Sant'Anna (2021) estimator on child education at T+28. |
| `callaway_santanna.py` | Callaway–Sant'Anna (2021) **estimator** for the binary-staggered education → development effect (treatment = first crossing of the 10% lower-secondary threshold). Always-treated countries excluded from controls per the standard CS handling. Recovers a significant effect after the TWFE collapse the GB diagnostic explains. |
| `goodman_bacon_decomposition.py` | Goodman-Bacon (2021) **diagnostic decomposition** of the TWFE estimator under binary staggered treatment. Decomposes the TWFE β into 2×2 DiD comparisons and identifies forbidden-comparison weights. Not an estimator; does not produce an alternative coefficient. Scope is **binary** treatment; the continuous-treatment companion is dCdH (battery §13). |
| `twfe_child_edu.py` | Two-way FE (country + year) on the parent → child education regression. β shrinks to 0.083 after absorbing time. |
| `twfe_all_outcomes.py` | TWFE residualisation for LE, TFR, U5MR, child education — global trends absorbed. |
| `permutation_null.py` | Permutation null: breaks the country-level parent-child match while preserving panel structure, refits the headline. |
| `colonial_global.py` | AJR institutions vs paper's framework on the *full* WCDE country list, not AJR's 64-colony subsample. |
| `colonial_vs_institutions.py` | AJR's settler-mortality → institutions → development chain vs settler-mortality → education → development on AJR's own sample. |
| `iv_2sls_colonial_icrg.py` | 2SLS contest with AJR's avexpr (1985-95 protection-against-expropriation, PRS/ICRG) as the institutional proxy. |
| `ajr_n61_deep_dive.py` | AJR n≈61 base sample read through the paper's framework: education → demographic convergence, Protestant vs Catholic schooling trajectories. |
| `ajr_n61_country_profiles.py` | Per-country markdown cards for the AJR n≈61 sample, grouped by cluster. |
| `regime_education.py` | Polity5-weighted: do autocracies invest faster in education than democracies? No — ~8.1 pp/decade vs ~10.3 pp/decade. |
| `asian_financial_crisis.py` | 1997-98 natural experiment: GDP crashed (Indonesia -14.5%), education kept growing (+5.4 pp). |
| `grandparent_effect.py` | Adding T-56 grandparent education raises R² beyond T-28 parent education alone. The two-generation depth of PT. |
| `grandparent_effect_all_outcomes.py` | Same grandparent test across LE, TFR, U5MR, child education. |
| `completion_vs_test_scores.py` | Hanushek horse race: does Hanushek-Woessmann cognitive-skills score beat completion at predicting outcomes? Completion wins. |
| `completion_vs_years_vs_tests.py` | Adds years of schooling to make it a four-horse race. Completion still wins. |
| `log_outcomes.py` | Re-runs forward-prediction with log-transformed LE, TFR, U5MR. |
| `ppml_outcomes.py` | Poisson Pseudo-Maximum Likelihood (Santos Silva & Tenreyro 2006) for the two right-skewed outcomes. |
| `wooldridge_exogeneity.py` | Wooldridge (2010, §10.5) regression-based F-test for strict exogeneity in the country-FE specification. |
| `backfill_all_outcomes.py` | Generalises the LE-vs-lag backfill check to all four outcomes — rules out long-lag artefacts from missing data. |
| `missing_by_period.py` | Per-period country coverage by variable; demonstrates the headline doesn't ride on listwise deletion. |
| `cross_cohort_within_year.py` | Independent test of the PT mechanism that does *not* use the 28-year forward lag — population-share microdata from WCDE. |
| `beta_by_baseline_group.py` | β by 1975 baseline group: Low <20% gets β=1.585, High >60% gets β=0.176. The amplification gradient. |
| `beta_by_ceiling_cutoff.py` | β > 1 at every parental-education cutoff below 90% — amplification is universal. |
| `u5mr_by_edu_level.py` | Does post-2000 residualised-GDP signal on U5MR vary by education level? Yes — communication environment matters. |
| `u5mr_residual_by_year.py` | Verifies the post-2000 U5MR signal is MDG-era health spending, not income. |
| `residual_by_outcome_year.py` | Sweeps residualised GDP R² by outcome year, 1985-2020, for all four outcomes. |
| `ussr_exclusion_panel.py` | Re-runs Table 1 col 1 + LE/TFR forward prediction with all 15 USSR republics dropped — addresses the "USSR contaminates the panel" objection. |

---

## `econometric_battery/` — Twenty named panel diagnostics

A self-contained battery of twenty modern panel-econometrics tests. See
[`scripts/econometric_battery/README.md`](econometric_battery/README.md) for
the full inventory. Tests cover what the paper does not run by design but
what a panel econometrician will want to see:

- 01 stationarity (Levin-Lin-Chu, IPS, Maddala-Wu, Pesaran CIPS, Hadri)
- 02 cross-section dependence (Pesaran CD, Friedman, Frees, BP-LM)
- 03 cointegration (Kao, Pedroni, Westerlund)
- 04 serial correlation (Wooldridge xtserial, Baltagi-Wu, Born-Breitung)
- 05 robust standard errors (White, cluster, two-way, Driscoll-Kraay, Conley)
- 06 strict exogeneity (Wooldridge, Hausman, Dumitrescu-Hurlin, AR-IV)
- 07 specification (RESET, polynomial, Hansen threshold, splines, terciles)
- 08 structural breaks (Chow, Andrews-Quandt, Bai-Perron, CUSUM)
- 09 influence (Cook's D, DFBETAs, leverage, LOO, studentised residuals)
- 10 robust regression (Huber, Tukey, LAD, Theil-Sen, trimmed, winsorised)
- 11 multiple testing (Bonferroni, Holm, BH, BY, Westfall-Young, Romano-Wolf)
- 12 dynamic panels (Anderson-Hsiao, Arellano-Bond, Hansen J, AR tests) — descriptive only for the within-country edu spec; the IV instrument and outcome are drawn from the same education series, so the exclusion restriction is in tension with the grandparent-channel finding
- 13 modern DiD: estimators (Callaway–Sant'Anna, Sun–Abraham, de Chaisemartin–d'Haultfœuille, Borusyak–Spiess, Honest DiD) + diagnostic (Goodman–Bacon decomposition, binary scope)
- 14 synthetic control (Abadie-Diamond-Hainmueller for seven canonical cases)
- 15 cross-validation (k-fold, forward-chain, region/era/OECD holdouts)
- 16 functional form (log, sqrt, arcsine, logit, Box-Cox)
- 17 sample composition (balanced, by-era, by-region, by-GDP, by-start)
- 18 measurement error (reliability, IV-neighbour, Barro-Lee, Deming)
- 19 placebo (CO₂ placebo, within-country shuffle, cohort mismatch)
- 20 predictive validity (OOS RMSE, Diebold-Mariano, Mizon-Richard encompassing)

Run all twenty: `python scripts/econometric_battery/run_all.py`. The runner
auto-generates `scripts/econometric_battery/REPORT.md` summarising every JSON.

---

## `ml/` — Transformer cross-check and designed falsification

A small encoder-only transformer trained blind on the panel. The point is
not state-of-the-art forecasting; the point is to let SGD have an honest
run at the data and check whether a non-parametric estimator reaches the
same conclusion the linear residualisation does.

The transformer reads each outcome at its **biological lag** — life
expectancy and under-five mortality at *T*+12 (the childrearing window),
fertility at *T*+5 — over the 28/56/84-year generational cohort stack (the
`ML_UNIFY_GENERATION` build, matching the paper's 28-year generation). There
is no single "predict the outcome one generation forward" horizon; the
generation enters through the cohort inputs, each outcome through its own
biological lag. This is the parent-vantage construction, and every diagnostic
below is run in it.

**Two caveats attach to every R² in this section**, both measured in
`leakage_audit.py` (`make ml-audit`) and both documented in
[`scripts/ml/README.md`](ml/README.md):

1. *The reported out-of-fold R² is selected on the held-out fold.* The
   trainer early-stops and checkpoint-selects by validation R² measured on
   the held-out countries, then reports that fold's score as out-of-fold.
   Under a nested protocol — inner validation carved from training
   countries only — the same models score 0.03 to 0.10 lower (LE −0.078,
   TFR −0.031, U5MR −0.095). The education gate is **not** weakened by the
   correction; on LE and TFR it strengthens slightly.
2. *The G1 gate measures reliance, not incremental value.* Zeroing
   education on a model trained with it drops R² by 0.25 to 0.40. Retraining
   from scratch with no education at all drops it by only 0.05 to 0.10,
   because region, latitude, GDP, institutions and disease burden are
   collinear with education and partly substitute for it. Both numbers are
   real and they answer different questions; the appendix's prose should
   name which one it is quoting. The within-country identification lives in
   `residualization/`, not here.

| Script | What it does |
|---|---|
| `data_loader.py` | Assembles the country × year panel. Each sample is one (country, T); features are the trajectory over [T-15, T-10, T-5, T]; targets are LE/U5MR at T+12 and TFR at T+5, with cohort inputs at T−28/56/84. |
| `data_loader_parent_lag.py` | Slices the panel into three single-target panels, each at its biological horizon — the parent-vantage construction used by every diagnostic here. |
| `broader_features.py` | Loads non-education indicators (geography, region, religion, colonial history, institutions, resources, climate, trade) as a per-(country, year) vector. |
| `build_maddison_backfill.py` | Builds a WB-equivalent Maddison-2020 GDP backfill covering 24.6% of cells previously zero-imputed. |
| `universal_transformer.py` | Encoder-only transformer; five country-holdout folds per variant (`joint`, `wcde_only`, `bl_only`, plus the isolation modes used in head-to-head). |
| `run_universal_evidence_parent_lag.py` | Trains the headline per-outcome models and the ablation gates: G1 education ablation (R² collapses when education is zeroed), G2 education singularity (R² barely moves when non-education is zeroed). Writes `universal_evidence_parent_lag.json` — the make-verify numbers. |
| `cloud/run_job.py`, `cloud/launch.py` | Run the full robustness battery on preemptible GPUs in parent vantage: walk-forward, LOO-185, region/era/income stratification, optuna architecture search, conditional-permutation importance, placebos, and the Barro-Lee replication. |
| `chapter9/aggregate_parent_battery.py` | Aggregates the cloud battery → `chapter9_parent_battery.json` (the walk-forward, LOO, placebo, optuna, cond-perm, stratification, and Barro-Lee numbers cited in this appendix). |
| `chapter9/spec_curve.py` | Five-method spec curve (OLS/ridge/lasso/RF/GBM) per outcome in parent vantage → `spec_curve_parent.json`. |
| `aggregate_bloc_holdout.py` | Designed falsification of the "hollow education" thesis. Holds out the eight Soviet peripheral republics (Caucasus + Central Asia); residuals on TFR (+1.2 births) and U5MR (+34/1000) come in directionally and substantially as predicted by §9. Pinned to the legacy 25-year grid because it reads post-1990 Soviet republics the 28-year horizon would starve. |
| `residual_atlas.py` | Layer 2: per-country residual atlas. Misfits are diagnostic — the strongest model is one with small, structured residuals that sort into mechanism-consistent buckets. |
| `chapter9/dml_parent.py` | Double machine-learning causal-style estimate: education partialled against every other feature, cross-fitted, at the biological lags. Per-level effect ≈ +4.85 years life expectancy, −0.94 children, −36.5 under-five deaths/1000 (`dml_parent.json`). The bracketed range in that file is the spread of five seed estimates, not a confidence interval; for country-clustered cross-fitting and a real standard error see `chapter9/dml_parent_clustered.py`, which keeps the sign and significance of all three effects on intervals roughly five times wider. |
| `chapter9/*` (walk-forward) | Train-on-past, predict-future: the transformer is trained only on cohorts completed before a cutoff and evaluated on the later era. Held-out R² rises from ≈0.66 (life expectancy, 1975 cutoff) to ≈0.88–0.91 by the 2000 cutoff (`chapter9_parent_battery.json` → `walk_forward`). Cutoffs outside 1975–2000 are omitted: at the biological horizons the panel ends ≈2008, so a 1965 cutoff starves the training side and a 2005 cutoff starves the held-out side. |
| `smoke_test.py` | Wiring check (`make ml-smoke`): imports, required inputs, panel shapes, one forward pass. Run it first. |
| `leakage_audit.py` | Holdout-protocol audit (`make ml-audit`). Re-estimates the headline OOF R² under a nested protocol, because the shipped trainer early-stops on the held-out fold and then reports that fold. Also separates the two education gates: zeroing education at inference vs retraining without it. `leakage_audit.json`. **Read this before quoting any R² from `ml/`.** |
| `conformal_intervals.py` | Split-conformal prediction intervals with coverage measured on held-out countries (`make ml-conformal`). Supplies the uncertainty the counterfactual and per-country numbers otherwise lack. `conformal_intervals.json`. |
| `chapter9/dml_parent_clustered.py` | DML with country-level cross-fitting and an analytic country-clustered standard error, run beside the shipped row-level configuration. The `theta_ci95` in `dml_parent.json` is the spread of five seed estimates, not a confidence interval. `dml_parent_clustered.json`. |
| `fetch_co2.py` | Downloads the optional CO2 series (OWID, CC BY 4.0) that the feature vector and the CO2 placebo use. Not redistributed here; the loader degrades to an all-NaN CO2 column without it. |

ML results are written to `scripts/ml/checkin/`. The make-verify ML numbers
come from `universal_evidence_parent_lag.json` and `spec_curve_parent.json`;
the broader robustness battery (walk-forward, LOO, placebos, optuna,
conditional permutation, stratification, Barro-Lee) lives in
`chapter9_parent_battery.json` and is reported here rather than in the paper's
§Every Method Agrees, which cites this appendix for it. The whole battery was
recast from the earlier single-horizon (joint *T*+25) build to the
biological-lag parent vantage; the verdicts are unchanged, and where the two
overlap (the placebo table) the numbers agree within rounding.

### Out-of-fold country counterfactual ("what the gap bought")

For ten matched pairs the transformer is given country A's `T=1990` feature
window with its education block replaced by country B's; the reported Δ is
the change in each outcome's prediction at its biological horizon (LE and
U5MR at T+12, TFR at T+5). Each country is scored out-of-fold. Taiwan (no
World Bank panel coverage) is omitted. The paper's §Every Method Agrees
points here for the full set; the headline gaps (Sri Lanka +12.10 years of
life expectancy over Pakistan's education trajectory, China +7.06 over
India's) illustrate how the effect scales with the size of the schooling
gap. Produced by `ml/chapter9/counterfactuals.py` (`chapter9_counterfactuals.json`).

| Country A | vs B's education | ΔLE (yrs) | ΔTFR | ΔU5MR |
|---|---|---:|---:|---:|
| Sri Lanka   | Pakistan           | +12.10 | −2.19 | −48.2 |
| China       | India              | +7.06  | −2.06 | −16.9 |
| Botswana    | Zimbabwe           | +4.80  | +0.23 | −11.8 |
| Cuba        | Dominican Republic | +4.61  | −0.70 | −8.4  |
| South Korea | Philippines        | +3.04  | −0.30 | −3.6  |
| Vietnam     | Myanmar            | +1.84  | −1.10 | −11.8 |
| Rwanda      | Burundi            | +0.47  | +0.00 | −14.0 |
| Bangladesh  | Pakistan           | −0.76  | +0.00 | +4.0  |
| Kenya       | Tanzania           | −1.39  | −0.28 | −4.8  |

The effect scales with the size of the schooling gap: large where the two
diverged hard (Sri Lanka, China, Cuba, Botswana), near zero where they did
not (Kenya, Bangladesh). Bangladesh shows no backward gain because over the
1975–1990 window Pakistan was still ahead on lower-secondary completion;
its documented overtaking is a post-1990, female-led catch-up outside this
swap's horizon, and the model correctly declines to assign it backward.

---

## Detail relocated from the Chapter 9 prose (2026-06-19 trim)

Three passages were compressed out of *The Panel* to keep it readable for
a policy audience. The claims they made survive in the chapter; the
numbers and their construction live here.

### Outcome-specific lag structure: parent vs grandparent R² blocks

The §The Generational Lag horse race reads each outcome forward at its own
biological lag and races primary, lower-secondary, and upper-secondary
completion in two blocks — the cohort's own schooling at *T* (the parental
block, which leads the outcome by its biological lag) and the prior
generation's at *T−28* (the grandparental block, leading by that lag plus
one generational step). Both blocks carry independent within-country R² for
all three demographic and health outcomes
(`residualization/horse_race_lags_levels.py`):

| Outcome | Grandparental block adds | Parental block adds |
|---|---:|---:|
| TFR (@ *T*+5)   | +0.091 | +0.114 |
| U5MR (@ *T*+12) | +0.130 | +0.105 |
| LE (@ *T*+12)   | +0.043 | +0.068 |
| log GDP (@ *T*) | +0.011 | (contemporaneous) |

For child survival the grandparental block carries signal at least as large
as the parental one — the health environment the grandparental cohort built
still moves the survival floor a generation on (the coefficient-level
version is §The Grandparent Channel). Log GDP is the exception: the
grandparental block adds essentially nothing once the parental block is in
(+0.011), because GDP is genuinely contemporaneous — the educated worker's
current output.

### residGDP within parental-education bins

The apparent post-2000 residGDP→U5MR signal — the pooled residGDP ×
Post-2000 interaction, β = +34.0, *p* = 0.022
(`robustness/u5mr_residual_by_year.py`) — is between-bin composition, not a
within-bin income effect. Stratifying by parental lower-secondary
completion at 5 pp resolution on the Maddison-backfilled panel
(`robustness/residgdp_strengthened_bin_test.py`), no individual bin shows a
residGDP coefficient that clears bootstrap significance. The
inverse-variance-weighted within-bin meta β on level U5MR post-2000 is
+5.28, with a 2,000-rep country-cluster bootstrap 95% CI of [−104, +48] —
indistinguishable from zero. Composition: pre-2000 active-expansion cells
sit at median parental completion 28.6%, post-2000 cells at 38.2%, on a
U5MR distribution that fell sharply over the period. The within-bin null
holds at each of the three timescales the chapter defines
(`robustness/residgdp_strengthened_bin_test_lags.py`): bootstrap 95% CIs
[−77, +53] (lag 0, contemporaneous), [−69, +60] (lag 12, childrearing
window), [−83, +44] (lag 28, cross-generation). The pre-2000 within-country
residGDP slope on log U5MR is −0.37 (*p* = 0.018,
`robustness/residgdp_by_baseline_fine.py`); both the interaction and the
slope are the numbers the compressed paragraph keeps in the paper.

### Placebo falsifications (transformer)

Six placebo transformations break the education→outcome link in different
ways. The three that should destroy it collapse education's out-of-fold
contribution to zero; replacing education with absolute latitude (the quiet
geographic check) leaves almost nothing; the two that leave a real channel
standing retain only what income- and time-correlation predict. The figures
below are the biological-lag (parent-vantage) battery — each placebo applied
to each outcome's single-target model at its own biological horizon (LE@12,
TFR@5, U5MR@12) and summarised as the median education R² drop across the
three outcomes (`ml/checkin/chapter9_parent_battery.json` →
`placebos`; producer `ml/chapter9/placebos.py::run_placebos_parent`):

| Placebo transformation | Median edu R² drop |
|---|---:|
| Shuffle outcomes across rows | −0.02 |
| Replace education with random noise | −0.004 |
| Scramble country labels | −0.05 |
| Replace education with absolute latitude | +0.01 |
| Replace education with within-year GDP rank | +0.08 |
| Shuffle the year | +0.11 |

None approaches what the true education features carry. Within-year GDP rank
and the year shuffle retain a fraction precisely because income and time are
themselves education-correlated. The values are within rounding of the earlier
single-horizon (joint T+25) placebo run — the falsification verdict does not
depend on the lag framing.

---

## What the appendix is not

The appendix is not the paper's identification strategy. The panel β is
descriptive — it documents the pattern across 185 countries 1950–2015.
Identification in *The Long Childhood* rests on three sources of evidence
the panel cannot supply on its own:

1. **Deep history — evolutionary biology.** The eighteen-year juvenile
   dependency window, the ~28-year parent-to-adult-child turnover, and the
   home-niche embedding of the child in near-older humans are species facts.
   The mechanism is deduced from what humans are; the panel is then asked
   whether the deduced signatures appear.
2. **Recent history — natural experiments and country histories.** Cambodia
   1975-79, Sri Lanka 1945, Korea vs Philippines from 1950, AFC 1997, Bihar
   vs Kerala 1966, plus the full forward trajectories of Taiwan, Korea,
   Kerala, Sri Lanka, Myanmar, China, Cuba, Bangladesh, Cambodia. Single
   shocks and full calendar trajectories are the same kind of evidence in
   two modes — recent history, read for identification. The panel's T→T+28
   lag is a coefficient; in the histories it is a calendar.
3. **The USSR falsification.** A designed negative case for the
   hollow-education thesis: Goskomstat-reported lower-secondary completion
   was real for the six European-core republics and metropolitan fiction
   for the eight Caucasian and Central Asian republics. Convergence
   followed the channel where it was real and failed where it was hollow
   (§9; `ml/run_ussr_holdout.py`).

Every script in this appendix is a panel-econometrics companion to those
three sources. They probe the panel; they do not carry the identification.

---

## References

Methodological references for the estimators and diagnostics above. These
are cited here rather than in the paper because the corresponding apparatus
lives in these scripts, not in the paper's text.

- Angrist, J.D. & Pischke, J.S. (2009). *Mostly Harmless Econometrics: An Empiricist's Companion*. Princeton University Press. — general identification/IV framework behind the IV contests.
- Bertrand, M., Duflo, E. & Mullainathan, S. (2004). How Much Should We Trust Differences-in-Differences Estimates? *Quarterly Journal of Economics*, 119(1), 249–275. — serial-correlation / clustered-inference basis for the robust-SE tests (battery §05) and the DiD estimators (§13).
- Callaway, B. & Sant'Anna, P. (2021). Difference-in-Differences with Multiple Time Periods. *Journal of Econometrics*, 225(2), 200–230. — the CS estimator in `callaway_santanna.py` and `cs_threshold_robustness.py` (§13).
- Goodman-Bacon, A. (2021). Difference-in-Differences with Variation in Treatment Timing. *Journal of Econometrics*, 225(2), 254–277. — the TWFE decomposition diagnostic in `goodman_bacon_decomposition.py` (§13).
- Rubin, D.B. (1974). Estimating Causal Effects of Treatments in Randomized and Nonrandomized Studies. *Journal of Educational Psychology*, 66(5), 688–701. — the potential-outcomes framework underlying the DiD and synthetic-control tests.
- Santos Silva, J.M.C. & Tenreyro, S. (2006). The Log of Gravity. *Review of Economics and Statistics*, 88(4), 641–658. — the PPML estimator in `ppml_outcomes.py` for the right-skewed outcomes.

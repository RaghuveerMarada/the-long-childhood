# `scripts/ml/` — transformer cross-check

The ML half of the repo. Every other directory here is econometrics; this
one trains a small encoder-only transformer on the same panel and asks
whether a non-parametric estimator lands where the linear residualisation
does.

The index entry for these scripts is
[`scripts/ECONOMETRICS.md`](../ECONOMETRICS.md#ml--transformer-cross-check-and-designed-falsification).
This file covers how to run them.

## Run it

```bash
make setup       # from the repo root
make ml-smoke    # ~2 min: are imports, data and panels actually wired up?
```

`make ml-smoke` is the first thing to run. It checks that torch and
scikit-learn imported, that every required input file is on disk, and
that the three parent-vantage panels build with the expected shapes and
horizons. It exits nonzero with the reason named if any of that fails.

Then, in rough order of cost:

| Target | What it does | Cost (CPU) |
|---|---|---|
| `make ml-smoke` | wiring check, no training | ~2 min |
| `make ml-audit` | leakage audit: shipped vs nested holdout protocol | ~1 h |
| `make ml-conformal` | split-conformal intervals with measured coverage | ~30 min |
| `make ml` | rebuild the headline ML checkin JSONs | hours |

`make ml` is deliberately **not** part of `make full`. The shipped JSONs
are a 30-seed cloud aggregate; a single-machine rerun reproduces the
verdicts, not the seed-averaged decimals. `make verify` reads the shipped
JSONs and does not depend on retraining anything.

## Dependencies

`torch` and `scikit-learn` are in `requirements.txt`. CPU wheels are
enough — nothing in this directory needs a GPU. The only GPU path is the
optional preemptible-worker battery under `cloud/`.

## Optional data

`data/co2_emissions_tonnes_per_person.csv` is **not** redistributed in
this repo. It supplies one non-education control and is the payload of
the CO2 placebo test.

- Without it: the panel still builds. The `co2_per_capita_tonnes` column
  is all-NaN, a `RuntimeWarning` is raised, and
  `BroaderFeatures().co2_available` is `False`. The CO2 placebo is not
  meaningful in that state and should be skipped.
- To add it: `python scripts/ml/fetch_co2.py` (OWID, CC BY 4.0). The
  script stamps `data/external/PROVENANCE.md` with the SHA256 and date,
  matching the pattern in `longrun/fetch_gapminder.py`.

Everything else the ML panel needs ships with the repo; `make ml-smoke`
lists what it checks for.

## Layout

| File | Role |
|---|---|
| `data_loader.py` | Assembles the country × year panel. Features are the trajectory over [T−15, T−10, T−5, T]; cohort inputs at T−28/56/84. |
| `data_loader_parent_lag.py` | Slices that into three single-target panels at each outcome's biological horizon (LE@T+12, TFR@T+5, U5MR@T+12). |
| `data_loader_longrun.py` | The pre-1950 long-run variant. |
| `broader_features.py` | The 29 non-education controls: geography, region, religion, colonial history, institutions, resources, climate, trade. These exist so the education ablation is a fair test rather than a test inside an education-only feature set. |
| `universal_transformer.py` | The architecture. No country embedding by design: the same function is applied to every country, so a held-out country cannot be recognised. |
| `run_universal_evidence_parent_lag.py` | Trains the headline models and the G1/G2 ablation gates. Writes `checkin/universal_evidence_parent_lag.json`. |
| `smoke_test.py` | Wiring check (`make ml-smoke`). |
| `leakage_audit.py` | Holdout-protocol audit (`make ml-audit`). See below. |
| `conformal_intervals.py` | Split-conformal intervals with measured coverage (`make ml-conformal`). |
| `fetch_co2.py` | Downloads the optional CO2 series. |
| `chapter9/` | Spec curve, DML, placebos, counterfactuals, battery aggregation. |
| `cloud/` | Optional GPU battery: walk-forward, LOO-185, optuna, stratification. |
| `checkin/` | JSON outputs. |
| `checkpoints/` | Shipped `.pt` weights from the cloud run. |

## Two things a reader should know before quoting these numbers

**1. The shipped out-of-fold R² is selected on the held-out fold.**

`universal_transformer.train_one_fold` and its copy in
`run_universal_evidence_parent_lag._train_one_fold_single` early-stop and
checkpoint-select on the held-out countries, then report that same fold's
R² as out-of-fold. Choosing the epoch by the test set is model selection
on the test set, so the reported R² is an optimistic bound.
`make ml-audit` re-estimates it under a nested protocol (inner validation
carved from training countries only) and reports the gap. Read
`checkin/leakage_audit.json` before quoting any R² from this directory.

**2. The G1 gate measures reliance, not incremental value — but the
naive alternative is confounded, so quote the corrected number.**

The shipped gate zeroes the education columns on a model that was
*trained with them*. That measures how much the fitted model leans on
those inputs, and zeroing pushes the input off the training manifold, so
it is an upper bound on reliance rather than a measure of how much
education adds over the alternatives. The methodology point stands: the
question "would a model with no education at all do worse?" requires
retraining without education.

**The naive version of that retrain is not the right answer, and this
file previously said it was.** `leakage_audit.py` runs a
retrain-without-education arm on the full feature set and reports a drop
of only 0.05–0.10 against the gate's 0.25–0.40. That number is an
artefact. Region, latitude, colonial origin and settler mortality are
loaded **once per country and broadcast to every year**
(`broader_features.py`, "Time-invariant loaders"), so their
within-country variance is exactly zero. They cannot explain anything in
the within-country panel the paper actually runs on. What they *can* do
is let a country-holdout model infer a never-seen country's outcome from
other countries that resemble it — pure between-country signal standing
in for the missing education block.

Dropping those blocks **as columns** and refitting on the entry-cohort
`[10%, 90%]` window triples the drop, from 9–11% to **26–29%**, landing
within a few points of the zero-at-inference gate for LE and TFR:

| Outcome | joint R² | no-edu, geography kept | no-edu, geography dropped | drop, geo kept | drop, geo dropped | zero-at-inference |
|---|---:|---:|---:|---:|---:|---:|
| LE | 0.675 | 0.615 | 0.487 | 8.9% | **27.7%** | 31% |
| TFR | 0.767 | 0.679 | 0.570 | 11.5% | **25.7%** | 31% |
| U5MR | 0.671 | 0.602 | 0.474 | 10.3% | **29.4%** | 52% |

Source: `G1_GATE_INVESTIGATION.md` §7, 5 seeds × 3 targets, same cloud
pipeline. So **26–29% is the fair training-time ablation number** and the
one to quote. The 0.05–0.10 figure and the row-filtered-only 9–11% figure
are both measuring geographic leakage, not education's substitutability.

Two caveats worth carrying. The U5MR gap (29.4% against 52%) is
unresolved; institutions, GDP, malaria and trade may capture more of
U5MR's variance specifically, or the gate may still be inflated there.
And the entry-cohort runs share the epoch-selection protocol described in
caveat 1 above, so their absolute R² carries the same optimism — the drop
*fractions* are more robust, since the bias lifts baseline and ablated fit
together.

The within-country identification lives in `scripts/residualization/`,
not here.

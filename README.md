# The Long Childhood — Replication Code and Data

**Website:** [educationfirst.world](https://educationfirst.world) — paper, evidence, country cases, and chatbot.

Every empirical claim in the paper traces to a script that produces it from source data. This repo is the minimum needed to verify that.

## Verify

```bash
git clone https://github.com/rkpagadala/the-long-childhood.git
cd the-long-childhood
make setup     # Create venv + install dependencies
make verify    # Check every paper claim against source data (~2 sec)
```

Requires Python 3.9+. All source data for `make verify` is included.

The ML cross-check has its own entry point:

```bash
make ml-smoke   # ~2 min — checks the ML half is actually runnable
```

`make verify` does not depend on it. See
[`scripts/ml/README.md`](scripts/ml/README.md) for what the ML scripts do,
what `make ml-audit` found about the reported R², and the one optional
data file that is not redistributed here.

## What's here

```
scripts/                 Analysis scripts + master verifier
  verify_the_long_childhood.py       Checks every number cited in the paper
  _shared.py               Common data-loading utilities
  ECONOMETRICS.md          Index for the econometric appendix
  cases/                   Country case studies
  residualization/         Core GDP-has-no-effect tests (Frisch-Waugh-Lovell)
  robustness/              Sensitivity and objection-blocking
  econometric_battery/     20 named panel-econometrics diagnostics
  ml/                      Transformer cross-check + designed falsification
                             README.md               How to run it; two caveats on the R²
                             smoke_test.py           `make ml-smoke` — is the ML half runnable?
                             leakage_audit.py        `make ml-audit` — holdout-protocol audit
                             conformal_intervals.py  `make ml-conformal` — prediction intervals
  tables/                  Regression tables
  figures/                 Figure generation
  wcde/                    WCDE long-run analysis

data/                    World Bank WDI source CSVs
wcde/data/processed/     WCDE v3 education data (1875-2020, 185 countries)
checkin/                 JSON verification checkpoints
paper/the_long_childhood.tex   Paper source (read by verifier)
paper/the_long_childhood.pdf   Paper PDF
```

## Econometric appendix

The paper carries one headline regression and a small set of figures in the body. The full econometric appendix — TWFE diagnostics, modern-DiD estimators, IV contests, panel-unit-root and cross-section-dependence batteries, placebo and permutation nulls, lag and threshold sweeps, ML cross-check — lives in [`scripts/ECONOMETRICS.md`](scripts/ECONOMETRICS.md). Each entry names the script and what it does.

## Run any script individually

```bash
make setup
.venv/bin/python scripts/tables/panel_full_fe.py
.venv/bin/python scripts/residualization/education_vs_gdp.py
.venv/bin/python scripts/robustness/robustness_tests.py
```

Each script prints its results and writes a JSON checkpoint to `checkin/`. The master verifier reads those checkpoints and compares them to every number in the paper.

## Rebuilding checkpoints from source

`make scripts` rebuilds every checkin JSON from raw data. A few checkpoints are shipped precomputed and excluded from `make scripts` because their scripts depend on the full WCDE v3 microdata file (`wcde/data/raw/prop_both.csv`, 172 MB; `pop_both.csv`, 91 MB), which is too large to distribute here:

- `china_mean_yrs_vs_peers.json` (`scripts/cases/china_mean_years.py`)
- `china_band_sensitivity.json` (`scripts/cases/china_band_sensitivity.py`)
- `development_threshold_count.json` (`scripts/cases/development_threshold_count.py`)
- `cross_cohort_within_year.json` (`scripts/robustness/cross_cohort_within_year.py`)
- `completion_vs_years_vs_tests.json` (`scripts/robustness/completion_vs_years_vs_tests.py`)

The scripts are included so you can audit the method. To re-run them, download WCDE v3 "prop" and "pop" files via the [WCDE explorer](https://dataexplorer.wittgensteincentre.org/wcde-v3/) or the R `wcde` package, and place them under `wcde/data/raw/`. The verifier does not depend on rebuilding these — `make verify` reads the shipped JSONs.

## Data sources

| Variable | Source | Indicator |
|----------|--------|-----------|
| Education | [WCDE v3](https://www.wittgensteincentre.org/en/wcde.htm) | Lower secondary completion, both sexes, age 20-24 |
| GDP per capita | [World Bank WDI](https://databank.worldbank.org/source/world-development-indicators) | Constant 2015 USD (NY.GDP.PCAP.KD) |
| Life expectancy | World Bank WDI | SP.DYN.LE00.IN |
| Fertility | World Bank WDI | SP.DYN.TFRT.IN |
| Child mortality | World Bank WDI | Under-5 per 1,000 live births |

## License

- **Code**: [MIT](https://opensource.org/licenses/MIT)
- **Data and text**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## Author

Krishna Pagadala — [rkrishna.pagadala@gmail.com](mailto:rkrishna.pagadala@gmail.com)

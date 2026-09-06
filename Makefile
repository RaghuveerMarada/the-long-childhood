# Makefile — the-long-childhood replication repo
#
# make setup     — create venv + install dependencies
# make verify    — check every paper claim against data (~2 sec)
# make scripts   — rebuild all checkin JSONs from source data
# make full      — rebuild all JSONs then verify (full from-scratch run)
#
# ML targets (scripts/ml/ — CPU, no GPU required):
# make ml-smoke  — build the ML panels and assert their shapes (~2 min)
# make ml-audit  — leakage audit: reported protocol vs nested protocol
# make ml        — rebuild the ML checkin JSONs (hours on CPU)
#
# `make ml` is deliberately NOT part of `make full`. The ML checkpoints
# ship precomputed because a full multi-seed retrain is a cloud job, not
# a two-second verification. `make ml-smoke` is the fast check that the
# ML half of the repo is wired up and runnable.

VENV   = .venv
PYTHON = $(VENV)/bin/python
PIP    = $(VENV)/bin/pip
PAPER_TEX = paper/the_long_childhood.tex
VERIFY_STAMP = checkin/.verified
ML_DIR = scripts/ml

.PHONY: all setup verify scripts full clean ml ml-smoke ml-audit ml-conformal

all: verify

setup: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@touch $@

verify: setup $(VERIFY_STAMP)

$(VERIFY_STAMP): checkin/*.json scripts/verify_the_long_childhood.py $(PAPER_TEX)
	$(PYTHON) scripts/verify_the_long_childhood.py --fast
	@touch $@

scripts: setup
	cd scripts && $(MAKE) PYTHON=$(abspath $(PYTHON))

full: scripts verify

# ── ML targets ────────────────────────────────────────────────────────

# Fast wiring check: imports resolve, data files are present, the three
# parent-vantage panels build with the expected shapes. Run this before
# anything else in scripts/ml/.
ml-smoke: setup
	$(PYTHON) $(ML_DIR)/smoke_test.py

# Leakage audit: re-estimates the headline out-of-fold R² under the
# protocol the shipped code uses (checkpoint selected on the held-out
# fold) and under a nested protocol (inner validation split carved from
# training countries only). Writes scripts/ml/checkin/leakage_audit.json.
ml-audit: setup
	$(PYTHON) $(ML_DIR)/leakage_audit.py

# Split-conformal prediction intervals on the country-holdout folds.
# Writes scripts/ml/checkin/conformal_intervals.json.
ml-conformal: setup
	$(PYTHON) $(ML_DIR)/conformal_intervals.py

# Full ML rebuild. Hours on CPU; the shipped JSONs come from a 30-seed
# cloud aggregate, so a single-machine run reproduces the verdicts, not
# the exact seed-averaged decimals.
ml: setup ml-smoke
	$(PYTHON) $(ML_DIR)/run_universal_evidence_parent_lag.py
	$(PYTHON) $(ML_DIR)/chapter9/spec_curve.py
	$(PYTHON) $(ML_DIR)/chapter9/dml_parent.py

clean:
	rm -rf $(VENV) $(VERIFY_STAMP)

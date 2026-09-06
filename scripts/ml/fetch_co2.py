# =============================================================================
# scripts/ml/fetch_co2.py
#
# Fetches per-capita CO2 emissions and writes
# data/co2_emissions_tonnes_per_person.csv in the wide Gapminder layout
# that scripts/ml/broader_features.py::_load_co2 expects:
#
#     Country,1900,1901,...,2023
#     Afghanistan,0.0012,...,
#
# Why this script exists
# ----------------------
# The CO2 series is an input to the ML feature vector and the payload of
# the CO2 placebo test, but the file is not redistributed in this repo.
# Without it a clean clone could not build the ML panel at all: the
# loader raised FileNotFoundError before any model was trained. The
# loader now degrades to an all-NaN CO2 column with a RuntimeWarning;
# this script restores the real series.
#
# Source: Our World in Data, "CO2 emissions per capita" (Global Carbon
# Budget). Republished under CC BY 4.0. Attribution belongs in
# data/external/PROVENANCE.md, which this script stamps.
#
# Usage:
#   python scripts/ml/fetch_co2.py
#
# Idempotent: re-running overwrites the CSV and restamps the SHA256 and
# download date.
# =============================================================================
"""Fetch per-capita CO2 and write it in the layout broader_features expects."""

import datetime as dt
import hashlib
import os
import sys
import urllib.request

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEST = os.path.join(REPO_ROOT, "data", "co2_emissions_tonnes_per_person.csv")
PROVENANCE = os.path.join(REPO_ROOT, "data", "external", "PROVENANCE.md")

OWID_URL = (
    "https://ourworldindata.org/grapher/co-emissions-per-capita.csv"
    "?time=1800..latest&useColumnShortNames=true"
)
USER_AGENT = (
    "long-childhood-fetcher/1.0 (replication; "
    "+https://github.com/rkpagadala/the-long-childhood)"
)

# OWID column names have moved between releases. Try each in order.
VALUE_COLUMN_CANDIDATES = (
    "emissions_total_per_capita",
    "co2_per_capita",
    "annual_co2_emissions_per_capita",
)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _pick_value_column(df):
    for c in VALUE_COLUMN_CANDIDATES:
        if c in df.columns:
            return c
    # Fall back to the single numeric column that is not the year.
    numeric = [
        c for c in df.columns
        if c.lower() not in ("entity", "code", "year")
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    if len(numeric) == 1:
        return numeric[0]
    raise SystemExit(
        "Could not identify the CO2 value column in the OWID download. "
        f"Columns present: {list(df.columns)}. Add the right name to "
        "VALUE_COLUMN_CANDIDATES in this script."
    )


def fetch():
    print(f"Fetching CO2 per capita from {OWID_URL}")
    req = urllib.request.Request(OWID_URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read()

    tmp = DEST + ".download"
    with open(tmp, "wb") as f:
        f.write(raw)

    long_df = pd.read_csv(tmp)
    os.remove(tmp)

    entity_col = "Entity" if "Entity" in long_df.columns else "entity"
    year_col = "Year" if "Year" in long_df.columns else "year"
    value_col = _pick_value_column(long_df)

    # OWID rows include aggregates ("World", "Africa (GCP)", income groups).
    # standardize_country_name in _shared drops anything it does not
    # recognise as a country, so they are harmless downstream, but strip
    # the obvious ones here to keep the file small and readable.
    drop_markers = ("(GCP)", "(excl.", "World", "income countries")
    mask = ~long_df[entity_col].astype(str).str.contains(
        "|".join(m.replace("(", r"\(").replace(")", r"\)") for m in drop_markers),
        regex=True,
        na=False,
    )
    long_df = long_df[mask]

    wide = long_df.pivot_table(
        index=entity_col, columns=year_col, values=value_col, aggfunc="first"
    )
    wide.index.name = "Country"
    wide.columns = [str(int(c)) for c in wide.columns]
    wide = wide.sort_index()
    wide.to_csv(DEST)

    digest = _sha256(DEST)
    print(
        f"  -> {DEST} ({os.path.getsize(DEST)} bytes, "
        f"{wide.shape[0]} countries x {wide.shape[1]} years, "
        f"sha256={digest[:12]}...)"
    )
    return digest


def update_provenance(digest):
    if not os.path.exists(PROVENANCE):
        print(f"  (no PROVENANCE.md at {PROVENANCE}; skipping stamp)")
        return
    today = dt.date.today().isoformat()
    entry = (
        "\n## Tier C: CO2 control and placebo payload\n\n"
        "Source: Our World in Data, *CO2 emissions per capita* "
        "(Global Carbon Budget). License: CC BY 4.0.\n"
        "Fetched by `scripts/ml/fetch_co2.py`. Not redistributed in this "
        "repo; the ML loader degrades to an all-NaN CO2 column when it is "
        "absent.\n\n"
        "### File: `../co2_emissions_tonnes_per_person.csv`\n"
        "- Indicator: annual CO2 emissions per capita (tonnes)\n"
        "- Layout: wide, `Country` + one column per year\n"
        f"- Sha256: `{digest}`\n"
        f"- Downloaded: `{today}`\n"
    )
    with open(PROVENANCE) as f:
        text = f.read()
    marker = "## Tier C: CO2 control and placebo payload"
    if marker in text:
        head = text.split(marker)[0].rstrip("\n")
        text = head + "\n" + entry.lstrip("\n")
    else:
        text = text.rstrip("\n") + "\n" + entry
    with open(PROVENANCE, "w") as f:
        f.write(text)
    print(f"Updated provenance: {PROVENANCE}")


def main():
    try:
        digest = fetch()
    except Exception as exc:  # network, schema drift, anything
        print(f"FETCH FAILED: {exc}", file=sys.stderr)
        print(
            "\nThe ML panel still builds without this file (the CO2 control "
            "becomes all-NaN and a RuntimeWarning is raised). Only the CO2 "
            "placebo test requires it. To supply it by hand, download "
            "https://ourworldindata.org/grapher/co-emissions-per-capita "
            "and reshape to wide `Country` + year columns at "
            f"{DEST}.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    update_provenance(digest)


if __name__ == "__main__":
    main()

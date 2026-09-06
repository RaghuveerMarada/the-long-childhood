# =============================================================================
# PAPER REFERENCE
# Script:  scripts/ml/broader_features.py
# Paper:   "The Long Childhood: On the Convergence of Humanity"
#
# Produces:
#   Non-education input features for the evidential AI build. Loads
#   geography, region, religion, colonial history, institutions, resources,
#   climate, and trade indicators into in-memory lookups, exposing them as
#   per-(country, year) feature vectors.
#
#   The point of these features is to make the G1-style "education vs
#   alternatives" ablation a fair test. Without them, ablating education
#   only proves the model uses education within an education-only feature
#   set; with them, ablating education proves the model uses education
#   even when given competing institutional / geographic / religious
#   alternatives.
#
# Inputs:
#   data/external/country_latlong.csv        — Google DSPL canonical
#   data/external/country_codes.csv          — ISO + region/subregion
#   data/external/religion_composition_pew.csv — Pew Research 2010
#   data/external/wb_oilrents.json           — World Bank NY.GDP.PETR.RT.ZS
#   data/external/wb_malaria.json            — World Bank SH.MLR.INCD.P3
#   data/external/wb_trade.json              — World Bank NE.TRD.GNFS.ZS
#   data/colonial_global/global_country_table.csv  — colonial origin
#   data/ajr2001/ajr_n61_country_table.csv   — settler mortality
#   data/p5v2018.xls                         — Polity 5 institution scores
#   data/co2_emissions_tonnes_per_person.csv — CO2 per capita  [OPTIONAL,
#       not redistributed; fetch with scripts/ml/fetch_co2.py]
#
# Outputs (in-memory): BroaderFeatures().features_at(country, year)
# =============================================================================
"""
broader_features.py

Loads non-education indicators into in-memory lookups and exposes them
as a single per-(country, year) feature vector for the panel transformer.
"""

import json
import os
import sys
import warnings
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from _shared import DATA, standardize_country_name

EXTERNAL = os.path.join(DATA, "external")

CONTINENT_CATEGORIES = ["AF", "AS", "EU", "OC", "NA", "SA"]
COLONIZER_CATEGORIES = ["UK", "France", "Spain", "Portugal", "Netherlands", "Other"]
RELIGION_FRAC_KEYS = ["christian", "muslim", "hindu", "buddhist", "unaffiliated"]


# ── Time-invariant loaders ────────────────────────────────────────────────

def _load_latlong():
    """canonical_country → (lat, long) from Google DSPL canonical."""
    df = pd.read_csv(os.path.join(EXTERNAL, "country_latlong.csv"))
    out = {}
    for _, r in df.iterrows():
        c = standardize_country_name(r["name"])
        if c:
            out[c] = (float(r["latitude"]), float(r["longitude"]))
    return out


def _load_country_codes():
    """canonical_country → {continent, region, subregion}."""
    df = pd.read_csv(os.path.join(EXTERNAL, "country_codes.csv"), low_memory=False)
    out = {}
    for _, r in df.iterrows():
        for name_col in ("official_name_en", "CLDR display name"):
            nm = r.get(name_col)
            if isinstance(nm, str) and nm.strip():
                c = standardize_country_name(nm)
                if c and c not in out:
                    out[c] = {
                        "continent": r.get("Continent"),
                        "region": r.get("Region Name"),
                        "subregion": r.get("Sub-region Name"),
                    }
                    break
    return out


def _load_religion_fractions():
    """canonical_country → {religion_name: fraction} for 2010 Pew snapshot."""
    df = pd.read_csv(os.path.join(EXTERNAL, "religion_composition_pew.csv"))
    df = df[df["Year"] == 2010].copy()
    out = {}
    for _, r in df.iterrows():
        nm = str(r["Country"]).strip()
        if "All Countries" in nm:
            continue
        c = standardize_country_name(nm)
        if c is None:
            continue
        # Pew uses "< 1.0" for tiny shares — coerce to numeric, NaN otherwise.
        def _v(col):
            v = r.get(col)
            try:
                return float(v) / 100.0
            except (TypeError, ValueError):
                return np.nan
        out[c] = {
            "christian": _v("Christians"),
            "muslim": _v("Muslims"),
            "hindu": _v("Hindus"),
            "buddhist": _v("Buddhists"),
            "unaffiliated": _v("Unaffiliated"),
        }
    return out


def _load_colonial():
    """canonical_country → {colonizer, religion_category, ever_colonized, protestant, catholic}."""
    df = pd.read_csv(os.path.join(DATA, "colonial_global", "global_country_table.csv"))
    out = {}
    for _, r in df.iterrows():
        c = standardize_country_name(r["country"])
        if c is None:
            continue
        out[c] = {
            "colonizer": r.get("colonizer"),
            "religion_category": r.get("religion"),
            "ever_colonized": int(bool(r.get("ever_colonized"))),
            "protestant": int(r.get("protestant", 0) or 0),
            "catholic": int(r.get("catholic", 0) or 0),
        }
    return out


def _load_ajr():
    """canonical_country → {logem4} (AJR 2001 settler mortality)."""
    df = pd.read_csv(os.path.join(DATA, "ajr2001", "ajr_n61_country_table.csv"))
    out = {}
    for _, r in df.iterrows():
        c = standardize_country_name(r["country"])
        if c is None:
            continue
        try:
            out[c] = {"logem4": float(r["logem4"])}
        except (TypeError, ValueError):
            pass
    return out


# ── Time-varying loaders ──────────────────────────────────────────────────

def _load_wb_json(filename):
    """(canonical_country, year) → value from a World Bank JSON download."""
    with open(os.path.join(EXTERNAL, filename)) as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) < 2:
        return {}
    out = {}
    for rec in data[1]:
        v = rec.get("value")
        if v is None:
            continue
        cname = rec.get("country", {}).get("value")
        if not cname:
            continue
        c = standardize_country_name(cname)
        if c is None:
            continue
        try:
            year = int(rec["date"])
            out[(c, year)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _load_polity():
    """(canonical_country, year) → polity2 score from p5v2018.xls."""
    df = pd.read_excel(os.path.join(DATA, "p5v2018.xls"))
    out = {}
    for _, r in df.iterrows():
        c = standardize_country_name(r["country"])
        if c is None:
            continue
        try:
            year = int(r["year"])
            p2 = float(r["polity2"])
        except (TypeError, ValueError):
            continue
        # polity2 valid range: -10..+10 (other values are special codes)
        if p2 < -10 or p2 > 10:
            continue
        out[(c, year)] = p2
    return out


def _load_co2():
    """(canonical_country, year) → CO2 per capita (tonnes).

    OPTIONAL input. The CO2 series enters the feature vector as one
    non-education control and is the payload of the CO2 placebo test.
    The file is not redistributed in this repo; run
    `python scripts/ml/fetch_co2.py` to download it.

    If the file is absent this returns an empty lookup: the
    co2_per_capita_tonnes column stays NaN and is handled by the same
    missingness path as every other sparsely observed control, so the
    ML panel still builds from a clean clone. A RuntimeWarning is
    raised so the absence is never silent. Callers that need the CO2
    placebo specifically should check `BroaderFeatures().co2_available`.
    """
    path = os.path.join(DATA, "co2_emissions_tonnes_per_person.csv")
    if not os.path.exists(path):
        warnings.warn(
            "data/co2_emissions_tonnes_per_person.csv not found. The CO2 "
            "control will be all-NaN and the CO2 placebo test is "
            "unavailable. Run `python scripts/ml/fetch_co2.py` to download "
            "it (OWID, CC BY 4.0).",
            RuntimeWarning,
            stacklevel=2,
        )
        return {}
    df = pd.read_csv(path)
    df["__c"] = df["Country"].map(standardize_country_name)
    df = df.dropna(subset=["__c"])
    out = {}
    for _, r in df.iterrows():
        c = r["__c"]
        for col in df.columns:
            try:
                yr = int(col)
            except (TypeError, ValueError):
                continue
            v = r[col]
            if pd.isna(v):
                continue
            try:
                out[(c, yr)] = float(v)
            except (TypeError, ValueError):
                continue
    return out


# ── Public API ────────────────────────────────────────────────────────────

class BroaderFeatures:
    """Bundles all non-education features as a single lookup. Construct once
    per training run and call features_at(country, year) per panel sample."""

    def __init__(self):
        self.latlong = _load_latlong()
        self.country_codes = _load_country_codes()
        self.religion = _load_religion_fractions()
        self.colonial = _load_colonial()
        self.ajr = _load_ajr()
        self.polity = _load_polity()
        self.co2 = _load_co2()
        # False when data/co2_emissions_tonnes_per_person.csv is absent.
        # The CO2 placebo test is not meaningful in that state.
        self.co2_available = bool(self.co2)
        self.oilrents = _load_wb_json("wb_oilrents.json")
        self.malaria = _load_wb_json("wb_malaria.json")
        self.trade = _load_wb_json("wb_trade.json")
        self._names = self._build_names()
        self._groups = self._build_groups()

    def _build_names(self):
        names = []
        # Geography (3)
        names += ["geo_lat", "geo_long", "geo_abs_lat"]
        # Continent one-hot (6)
        for r in CONTINENT_CATEGORIES:
            names.append(f"region_continent_{r}")
        # Colonizer one-hot (6)
        for c in COLONIZER_CATEGORIES:
            names.append(f"colonizer_{c}")
        # Religion fractions (5)
        for r in RELIGION_FRAC_KEYS:
            names.append(f"religion_frac_{r}")
        # Colonial binary (3)
        names += ["ever_colonized", "protestant_flag", "catholic_flag"]
        # AJR settler mortality (1)
        names += ["ajr_logem4"]
        # Time-varying (5)
        names += ["polity2", "oil_rents_pct_gdp", "malaria_incidence_per_1000",
                  "trade_openness_pct_gdp", "co2_per_capita_tonnes"]
        return names

    def _build_groups(self):
        names = self._names
        def _range(prefix):
            return [i for i, n in enumerate(names) if n.startswith(prefix)]
        return {
            "geography": _range("geo_"),
            "region": _range("region_"),
            "colonizer": _range("colonizer_"),
            "religion": _range("religion_"),
            "colonial_binary": [names.index(n) for n in
                                ("ever_colonized", "protestant_flag", "catholic_flag")],
            "settler_mortality": [names.index("ajr_logem4")],
            "institutions": [names.index("polity2")],
            "resources": [names.index("oil_rents_pct_gdp"),
                          names.index("co2_per_capita_tonnes")],
            "climate_disease": [names.index("malaria_incidence_per_1000")],
            "trade": [names.index("trade_openness_pct_gdp")],
        }

    def feature_names(self):
        return list(self._names)

    def feature_groups(self):
        return {k: list(v) for k, v in self._groups.items()}

    def features_at(self, country, year):
        names = self._names
        x = np.full(len(names), np.nan)
        # Geography
        ll = self.latlong.get(country)
        if ll is not None:
            x[names.index("geo_lat")] = ll[0]
            x[names.index("geo_long")] = ll[1]
            x[names.index("geo_abs_lat")] = abs(ll[0])
        # Continent one-hot
        cont = (self.country_codes.get(country) or {}).get("continent")
        for r in CONTINENT_CATEGORIES:
            x[names.index(f"region_continent_{r}")] = 1.0 if cont == r else 0.0
        # Colonizer one-hot
        col_info = self.colonial.get(country) or {}
        colz = col_info.get("colonizer")
        # Normalize Britain → UK
        colz_norm = "UK" if colz in ("Britain", "British", "Great Britain") else colz
        matched = False
        for c in COLONIZER_CATEGORIES[:-1]:  # all except "Other"
            is_match = 1.0 if colz_norm == c else 0.0
            x[names.index(f"colonizer_{c}")] = is_match
            if is_match:
                matched = True
        x[names.index("colonizer_Other")] = 0.0 if matched else (1.0 if colz_norm else 0.0)
        # Religion fractions
        rel = self.religion.get(country) or {}
        for k in RELIGION_FRAC_KEYS:
            x[names.index(f"religion_frac_{k}")] = rel.get(k, np.nan)
        # Colonial binary
        x[names.index("ever_colonized")] = float(col_info.get("ever_colonized", 0))
        x[names.index("protestant_flag")] = float(col_info.get("protestant", 0))
        x[names.index("catholic_flag")] = float(col_info.get("catholic", 0))
        # AJR settler mortality
        ajr_info = self.ajr.get(country) or {}
        x[names.index("ajr_logem4")] = ajr_info.get("logem4", np.nan)
        # Time-varying
        x[names.index("polity2")] = self.polity.get((country, year), np.nan)
        x[names.index("oil_rents_pct_gdp")] = self.oilrents.get((country, year), np.nan)
        x[names.index("malaria_incidence_per_1000")] = self.malaria.get((country, year), np.nan)
        x[names.index("trade_openness_pct_gdp")] = self.trade.get((country, year), np.nan)
        x[names.index("co2_per_capita_tonnes")] = self.co2.get((country, year), np.nan)
        return x


if __name__ == "__main__":
    bf = BroaderFeatures()
    print(f"n features: {len(bf.feature_names())}")
    for k, v in bf.feature_groups().items():
        print(f"  {k}: {len(v)} features")
    print()
    for c in ["south korea", "bangladesh", "spain", "russia", "saudi arabia"]:
        for y in (1965, 1995):
            x = bf.features_at(c, y)
            n_present = int((~np.isnan(x)).sum())
            print(f"  {c}, year={y}: {n_present}/{len(x)} features present")

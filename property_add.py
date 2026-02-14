#!/usr/bin/env python3
"""
Featurize materials from either:
  (A) a pymatgen Structure dict-string (your case: {'@module':..., '@class':'Structure', ...})
  (B) a CIF string (optional support)

Uses: pymatgen + matminer.

Usage:
  python property_add.py --input ./data/mp/test.csv --struct-col structure --id-col material_id --output ./data/mp/test_featurized.csv
  python property_add.py --input ./data/mp/test.csv --struct-col structure --print-new-only
"""

import argparse
import ast
import csv
import io
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from pymatgen.core import Structure, Composition
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.composition import (
    Stoichiometry,
    ElementFraction,
    ElementProperty,
    ValenceOrbital,
    IonProperty,
)
from matminer.featurizers.structure import DensityFeatures, GlobalSymmetryFeatures
from matminer.featurizers.site import CoordinationNumber
from matminer.featurizers.structure import SiteStatsFingerprint
import sys          # <-- add

csv.field_size_limit(sys.maxsize)

# -------------------------
# Parsing helpers
# -------------------------
def structure_from_any(cell: Any) -> Optional[Structure]:
    """
    Parse Structure from:
      - dict (already)
      - string that looks like dict (Python-literal)
      - CIF string
    Returns None if parsing fails.
    """
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return None

    # Case 1: already a dict
    if isinstance(cell, dict):
        try:
            return Structure.from_dict(cell)
        except Exception:
            return None

    # Case 2: string
    if isinstance(cell, str):
        s = cell.strip()
        if not s:
            return None

        # 2a) Looks like pymatgen dict-string
        if s.startswith("{") and ("'@class': 'Structure'" in s or '"@class": "Structure"' in s):
            try:
                d = ast.literal_eval(s)  # safe parse of Python-literal dict string
                return Structure.from_dict(d)
            except Exception:
                return None

        # 2b) Otherwise try CIF (with unescape)
        try:
            if "\\n" in s and "\n" not in s:
                s = s.replace("\\n", "\n")
            if "\\t" in s:
                s = s.replace("\\t", "\t")

            parser = CifParser(io.StringIO(s), occupancy_tolerance=2.0)
            structs = parser.get_structures(primitive=False)
            return structs[0] if structs else None
        except Exception:
            return None

    return None


def try_add_oxidation_states(struct: Structure) -> Tuple[Optional[Structure], str, int]:
    """Attempt oxidation state decoration; return (decorated_struct, source, n_sites_with_oxi)."""
    if struct is None:
        return None, "none", 0
    try:
        s = struct.copy()
        s.add_oxidation_state_by_guess()
        n_ox = sum(1 for site in s.sites if hasattr(site.specie, "oxi_state"))
        return s, "pymatgen_guess", n_ox
    except Exception:
        return None, "failed", 0


# -------------------------
# Featurizers
# -------------------------
def build_composition_featurizer() -> MultipleFeaturizer:
    return MultipleFeaturizer(
        [
            Stoichiometry(),
            ElementFraction(),
            ElementProperty.from_preset("magpie"),
            ValenceOrbital(props=["frac"]),
        ]
    )


def build_structure_featurizer() -> MultipleFeaturizer:
    return MultipleFeaturizer(
        [
            DensityFeatures(),
            GlobalSymmetryFeatures(),
        ]
    )


def build_local_env_featurizer() -> MultipleFeaturizer:
    cn = CoordinationNumber(nn="CrystalNN")
    return MultipleFeaturizer(
        [
            SiteStatsFingerprint(cn, stats=("mean", "std_dev", "minimum", "maximum")),
        ]
    )


def safe_featurize(featurizer: MultipleFeaturizer, obj: Any) -> Dict[str, Any]:
    try:
        labels = featurizer.feature_labels()
        vals = featurizer.featurize(obj)
        return {k: v for k, v in zip(labels, vals)}
    except Exception:
        try:
            return {k: np.nan for k in featurizer.feature_labels()}
        except Exception:
            return {}


def ion_property_features(comp: Composition) -> Dict[str, Any]:
    ip = IonProperty()
    try:
        labels = ip.feature_labels()
        vals = ip.featurize(comp)
        return {k: v for k, v in zip(labels, vals)}
    except Exception:
        try:
            return {k: np.nan for k in ip.feature_labels()}
        except Exception:
            return {}


# -------------------------
# Main pipeline
# -------------------------
def featurize_dataframe(
    df: pd.DataFrame,
    struct_col: str,
    id_col: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    comp_feat = build_composition_featurizer()
    struct_feat = build_structure_featurizer()
    local_feat = build_local_env_featurizer()

    records: list[Dict[str, Any]] = []

    it = tqdm(df.itertuples(index=False), total=len(df), disable=not verbose)
    for row in it:
        row_dict = row._asdict()
        cell = row_dict.get(struct_col, None)

        rec: Dict[str, Any] = {}
        if id_col and id_col in row_dict:
            rec[id_col] = row_dict[id_col]

        s = structure_from_any(cell)
        rec["pmg_parsed_ok"] = bool(s is not None)

        if s is None:
            # fill placeholders for consistent columns
            rec.update({k: np.nan for k in comp_feat.feature_labels()})
            rec.update({k: np.nan for k in struct_feat.feature_labels()})
            rec.update({k: np.nan for k in local_feat.feature_labels()})
            rec["oxidation_state_source"] = "none"
            rec["n_ox_known_sites"] = 0

            # IonProperty placeholders
            tmp = ion_property_features(Composition("Fe2 O3"))
            for k in tmp.keys():
                rec[k] = np.nan

            rec["volume_per_atom"] = np.nan
            rec["pmg_spacegroup_number"] = np.nan
            rec["pmg_spacegroup_symbol"] = np.nan
            records.append(rec)
            continue

        # Composition, Structure, Local-env
        rec.update(safe_featurize(comp_feat, s.composition))
        rec.update(safe_featurize(struct_feat, s))
        rec.update(safe_featurize(local_feat, s))

        # Oxidation + IonProperty
        s_ox, ox_src, n_ox = try_add_oxidation_states(s)
        rec["oxidation_state_source"] = ox_src
        rec["n_ox_known_sites"] = int(n_ox)

        if s_ox is not None and ox_src != "failed":
            rec.update(ion_property_features(s_ox.composition))
        else:
            tmp = ion_property_features(Composition("Fe2 O3"))
            for k in tmp.keys():
                rec[k] = np.nan

        # Extra explicit structural features
        rec["volume_per_atom"] = float(s.volume) / float(len(s))

        try:
            sga = SpacegroupAnalyzer(s, symprec=1e-3)
            rec["pmg_spacegroup_number"] = int(sga.get_space_group_number())
            rec["pmg_spacegroup_symbol"] = str(sga.get_space_group_symbol())
        except Exception:
            rec["pmg_spacegroup_number"] = np.nan
            rec["pmg_spacegroup_symbol"] = np.nan

        records.append(rec)

    feats = pd.DataFrame(records)

    # new-features-only df
    base_cols = [id_col] if id_col and id_col in feats.columns else []
    new_feature_cols = [c for c in feats.columns if c not in base_cols]
    new_features_df = feats[base_cols + new_feature_cols].copy()

    # merge back
    if id_col and id_col in df.columns and id_col in feats.columns:
        full_df = df.merge(feats, on=id_col, how="left")
    else:
        full_df = pd.concat([df.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)

    return full_df, new_features_df


def main():
    warnings.filterwarnings("ignore")

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input CSV")
    ap.add_argument("--output", default=None, help="Path to output CSV (optional)")
    ap.add_argument("--struct-col", default="structure", help="Column containing structure dict-string or CIF")
    ap.add_argument("--id-col", default=None, help="Optional ID column (e.g., material_id)")
    ap.add_argument("--print-new-only", action="store_true", help="Print only new features")
    ap.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    ap.add_argument("--nrows", type=int, default=None, help="Read only first N rows (debug)")

    args = ap.parse_args()

    # Safer CSV read for embedded JSON/dicts
    df = pd.read_csv(
        args.input,
        engine="python",
        quoting=csv.QUOTE_MINIMAL,
        escapechar="\\",
        nrows=args.nrows,
    )

    if args.struct_col not in df.columns:
        raise ValueError(f"Missing struct column '{args.struct_col}'. Available: {list(df.columns)}")

    full_df, new_features_df = featurize_dataframe(
        df,
        struct_col=args.struct_col,
        id_col=args.id_col,
        verbose=not args.no_progress,
    )

    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 220)

    print("\n=== Preview ===")
    print(new_features_df.head(10) if args.print_new_only else full_df.head(10))

    if args.output:
        full_df.to_csv(args.output, index=False)
        print(f"\nSaved featurized CSV to: {args.output}")

    # Coverage report
    parsed_ok = full_df["pmg_parsed_ok"].mean() if "pmg_parsed_ok" in full_df.columns else np.nan
    ox_ok = (full_df["oxidation_state_source"] == "pymatgen_guess").mean() if "oxidation_state_source" in full_df.columns else np.nan
    print("\n=== Coverage Report ===")
    print(f"Parsed OK fraction: {parsed_ok:.3f}")
    print(f"Oxidation guessed OK fraction: {ox_ok:.3f}")


if __name__ == "__main__":
    main()
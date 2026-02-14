#!/usr/bin/env python3
"""
Train/evaluate XGBoost, CrabNet, and Roost on CSVs with composition formulas.

This script provides a unified interface similar to run_gnns.py:
- Reads separate train and test CSV files
- Trains models on training data
- Outputs predictions for test set

Example:
  python predict_other_models.py \
    --model xgb \
    --target-col formation_energy_per_atom \
    --formula-col formula_pretty \
    --work-dir ./output

Notes:
- XGBoost uses matminer composition features (no structure needed).
- CrabNet takes a DataFrame with columns: ["formula", "target"].
- Roost is run through its official examples/roost-example.py CLI.
"""

import argparse
import sys
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred)#, squared=False)  # squared=False gives RMSE
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def detect_id_column(df: pd.DataFrame) -> str:
    """
    Detect which ID column exists in the dataframe.
    Returns 'material_id' if present, otherwise 'jarvis_id'.
    Raises ValueError if neither exists.
    """
    if "material_id" in df.columns:
        return "material_id"
    elif "jarvis_id" in df.columns:
        return "jarvis_id"
    else:
        raise ValueError(
            "CSV must contain either 'material_id' (Materials Project) or "
            "'jarvis_id' (JARVIS) column."
        )


def run_xgb(train_df, test_df, formula_col, target_col, id_col, outdir: Path, seed: int):
    print("\n=== XGBoost (matminer structure features from CIF) ===")

    try:
        from pymatgen.core import Structure
        from matminer.featurizers.structure import (
            DensityFeatures,
            GlobalSymmetryFeatures,
            RadialDistributionFunction,
            EwaldEnergy,
            StructuralHeterogeneity,
        )
        from matminer.featurizers.base import MultipleFeaturizer
        import xgboost as xgb
    except ImportError:
        print("Missing deps for XGBoost baseline.")
        print("Install: pip install xgboost matminer pymatgen")
        raise

    # Check for structure column (either cif_structure or structure)
    structure_col = None
    if "cif_structure" in train_df.columns:
        structure_col = "cif_structure"
    elif "structure" in train_df.columns:
        structure_col = "structure"
    else:
        raise ValueError("CSV must contain either 'cif_structure' or 'structure' column for structure-based features.")
    
    if structure_col not in test_df.columns:
        raise ValueError(f"CSV must contain '{structure_col}' column in test data.")

    def to_struct(struct_data):
        """Convert structure data to Structure object.
        Handles both CIF strings and JSON-serialized Structure objects.
        """
        try:
            if struct_data is None or pd.isna(struct_data):
                return None
            
            # If it's already a Structure object
            if isinstance(struct_data, Structure):
                return struct_data
            
            # If it's a string
            if isinstance(struct_data, str):
                struct_str = struct_data.strip()
                if not struct_str:
                    return None
                
                # Check if it's a JSON/dict representation (starts with { or '{
                if struct_str.startswith("{") or (struct_str.startswith("'") and "{" in struct_str):
                    # Try to parse as JSON/dict
                    try:
                        import json
                        import ast
                        # Handle string representation of dict
                        if struct_str.startswith("'"):
                            # It's a stringified dict, use ast.literal_eval (safer than eval)
                            struct_dict = ast.literal_eval(struct_str)
                        elif struct_str.startswith("{"):
                            # Try JSON first, then ast.literal_eval
                            try:
                                struct_dict = json.loads(struct_str)
                            except json.JSONDecodeError:
                                struct_dict = ast.literal_eval(struct_str)
                        else:
                            return None
                        
                        return Structure.from_dict(struct_dict)
                    except Exception as e:
                        # If dict parsing fails, try as CIF
                        try:
                            return Structure.from_str(struct_str, fmt="cif")
                        except Exception:
                            return None
                else:
                    # Assume it's a CIF string
                    return Structure.from_str(struct_str, fmt="cif")
            
            # If it's a dict
            if isinstance(struct_data, dict):
                return Structure.from_dict(struct_data)
            
            return None
        except Exception as e:
            return None

    tr = train_df.copy()
    te = test_df.copy()
    
    # Store original column names that might conflict with featurizers
    # Common conflicts: density, volume, etc.
    potential_conflicts = ["density", "volume", "density_atomic"]
    conflict_cols = [col for col in potential_conflicts if col in tr.columns]
    
    # Temporarily rename conflicting columns to avoid conflicts with featurizers
    rename_map = {}
    for col in conflict_cols:
        rename_map[col] = f"original_{col}"
    
    if rename_map:
        tr = tr.rename(columns=rename_map)
        te = te.rename(columns=rename_map)
        print(f"[XGBoost] Renamed conflicting columns: {list(rename_map.keys())} -> {list(rename_map.values())}")
    
    print(f"[XGBoost] Parsing structures from '{structure_col}' column...")
    tr["structure_obj"] = tr[structure_col].apply(to_struct)
    te["structure_obj"] = te[structure_col].apply(to_struct)

    tr = tr.dropna(subset=["structure_obj", target_col]).reset_index(drop=True)
    te = te.dropna(subset=["structure_obj", target_col]).reset_index(drop=True)

    print(f"[XGBoost] Valid structures - Train: {len(tr)}, Test: {len(te)}")

    # Use structure-based featurizers
    # Using a subset that are fast and informative
    featurizer = MultipleFeaturizer(
        [
            DensityFeatures(),
            GlobalSymmetryFeatures(),
            RadialDistributionFunction(),
            EwaldEnergy(),
            StructuralHeterogeneity(),
        ]
    )

    print("[XGBoost] Featurizing structures (this may take a while)...")
    X_tr = featurizer.featurize_dataframe(tr, col_id="structure_obj", ignore_errors=True)
    X_te = featurizer.featurize_dataframe(te, col_id="structure_obj", ignore_errors=True)
    
    # Explicitly drop structure-related columns to avoid any issues
    cols_to_drop = [col for col in X_tr.columns if col in [structure_col, "structure_obj"] or 
                    (isinstance(col, str) and "structure" in col.lower() and col != "structure_obj")]
    if cols_to_drop:
        X_tr = X_tr.drop(columns=cols_to_drop, errors='ignore')
        X_te = X_te.drop(columns=cols_to_drop, errors='ignore')
        print(f"[XGBoost] Dropped structure-related columns: {cols_to_drop}")

    # Drop non-feature columns (including renamed original columns)
    drop_cols = {formula_col, target_col, "structure_obj", structure_col, id_col}
    drop_cols.update(rename_map.values())  # Also drop renamed original columns
    
    # Also drop any columns that might contain structure objects or other non-numeric data
    # Check for columns that can't be converted to float
    numeric_cols = []
    for col in X_tr.columns:
        if col in drop_cols:
            continue
        if col not in X_te.columns:
            continue
        # Try to check if column is numeric
        try:
            # Sample a few values to check
            sample = X_tr[col].dropna().head(10)
            if len(sample) > 0:
                pd.to_numeric(sample, errors='raise')
                numeric_cols.append(col)
        except (ValueError, TypeError):
            # Skip non-numeric columns
            print(f"[XGBoost] Skipping non-numeric column: {col}")
            continue
    
    feature_cols = numeric_cols
    print(f"[XGBoost] Extracted {len(feature_cols)} numeric features")

    # Ensure target values are numeric and align indices
    y_train_series = pd.to_numeric(X_tr[target_col], errors='coerce')
    y_test_series = pd.to_numeric(X_te[target_col], errors='coerce')
    
    # Drop rows where target is NaN
    valid_train = ~y_train_series.isna()
    valid_test = ~y_test_series.isna()
    
    X_tr = X_tr[valid_train].reset_index(drop=True)
    X_te = X_te[valid_test].reset_index(drop=True)
    y_train = y_train_series[valid_train].values.astype(float)
    y_test = y_test_series[valid_test].values.astype(float)
    test_ids = te[valid_test][id_col].reset_index(drop=True).values

    # Convert feature columns to numeric, handling any remaining issues
    X_train = X_tr[feature_cols].copy()
    X_test = X_te[feature_cols].copy()
    
    # Convert each column to numeric, coercing errors to NaN
    for col in feature_cols:
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
    
    # Replace inf and NaN, then convert to numpy array
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0.0).values

    model = xgb.XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    m = metrics(y_test, pred)
    print("XGB metrics:", m)

    # Save predictions with ID column (like run_gnns.py format)
    ensure_dir(outdir)
    test_ids = te[id_col].values
    pred_df = pd.DataFrame({
        id_col: test_ids,
        "y_true": y_test,
        "y_pred": pred,
    })
    pred_df.to_csv(outdir / "xgb_predictions.csv", index=False)
    with open(outdir / "xgb_metrics.json", "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)
    
    print(f"[XGBoost] Saved predictions to: {outdir / 'xgb_predictions.csv'}")

    return m


def run_crabnet(train_df, test_df, formula_col, target_col, id_col, outdir: Path, seed: int, crabnet_path: Path = None):
    print("\n=== CrabNet ===")

    # Try to import from local path if provided, otherwise use installed package
    if crabnet_path is not None:
        crabnet_path = Path(crabnet_path).resolve()
        if not crabnet_path.exists():
            raise FileNotFoundError(f"CrabNet path not found: {crabnet_path}")
        
        # Check if path points to crabnet package or parent directory
        if (crabnet_path / "crabnet").exists():
            # Path points to parent directory (e.g., /path/to/CrabNet)
            if str(crabnet_path) not in sys.path:
                sys.path.insert(0, str(crabnet_path))
            print(f"[CrabNet] Using local repository at: {crabnet_path}")
        else:
            # Assume it's the parent directory
            if str(crabnet_path) not in sys.path:
                sys.path.insert(0, str(crabnet_path))
            print(f"[CrabNet] Using local installation at: {crabnet_path}")
        
        # Try importing from local repository
        try:
            from crabnet.crabnet_ import CrabNet
            print("[CrabNet] Successfully imported from local repository")
        except ImportError as e:
            raise ImportError(
                f"Could not import CrabNet from {crabnet_path}. "
                f"Error: {e}. Please ensure the path contains the crabnet package."
            )
    else:
        # Try installed package
        try:
            from CrabNet.crabnet.crabnet_ import CrabNet
            print("[CrabNet] Using installed package")
        except ImportError:
            print("Missing CrabNet. Install one of:")
            print("  Or provide --crabnet-path to use local repository")
            raise

    # CrabNet expects columns named "formula" and "target"
    # Keep track of IDs before dropping them
    tr = train_df[[id_col, formula_col, target_col]].dropna().copy()
    te = test_df[[id_col, formula_col, target_col]].dropna().copy()
    
    test_ids = te[id_col].values
    
    tr_crabnet = tr[[formula_col, target_col]].copy()
    te_crabnet = te[[formula_col, target_col]].copy()
    tr_crabnet.columns = ["formula", "target"]
    te_crabnet.columns = ["formula", "target"]

    # Set seeds (CrabNet uses torch internally; do best-effort)
    try:
        import torch
        torch.manual_seed(seed)
        np.random.seed(seed)
    except ImportError:
        pass

    # Create and train CrabNet model
    # The new API accepts DataFrames directly
    cb = CrabNet(mat_prop=str(target_col), random_state=seed, verbose=True)
    
    # Fit on train data
    cb.fit(train_df=tr_crabnet)
    
    # Predict on test data with uncertainty
    pred, sigma = cb.predict(test_df=te_crabnet, return_uncertainty=True)
    
    # Get true values and predictions
    y_true = te_crabnet["target"].values.astype(float)
    y_pred = np.asarray(pred, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Align test_ids with predictions (in case some rows were dropped)
    n_pred = len(y_pred)
    if len(test_ids) != n_pred:
        print(f"[CrabNet] Warning: {len(test_ids)} test IDs but {n_pred} predictions. "
              "Some samples may have been dropped during data loading.")
        # Use the first n_pred IDs (assuming order is preserved)
        test_ids = test_ids[:n_pred]
        y_true = y_true[:n_pred]

    m = metrics(y_true, y_pred)
    print("CrabNet metrics:", m)

    # Save predictions with ID column (like run_gnns.py format)
    ensure_dir(outdir)
    pred_df = pd.DataFrame({
        id_col: test_ids,
        "y_true": y_true,
        "y_pred": y_pred,
        "sigma": np.asarray(sigma, dtype=float),
    })
    pred_df.to_csv(outdir / "crabnet_predictions.csv", index=False)
    with open(outdir / "crabnet_metrics.json", "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)
    
    print(f"[CrabNet] Saved predictions to: {outdir / 'crabnet_predictions.csv'}")

    return m


def _run(cmd, cwd=None):
    print("\n$ " + " ".join(cmd))
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    print(p.stdout)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed (code={p.returncode}): {' '.join(cmd)}")


def run_roost(train_df, test_df, formula_col, target_col, id_col, outdir: Path, epochs: int, roost_path: Path = None):
    print("\n=== Roost (via official example CLI) ===")

    # Use provided roost path or default to aviary folder in the same directory as this script
    if roost_path is not None:
        roost_dir = Path(roost_path).resolve()
    else:
        # Default to aviary folder in the property_prediction directory
        script_dir = Path(__file__).parent.resolve()
        roost_dir = (script_dir / "aviary").resolve()
    
    # Check if roost directory exists
    if not roost_dir.exists():
        raise FileNotFoundError(
            f"Roost repository not found at {roost_dir}. "
            "Please provide --roost-path or ensure aviary folder exists in the property_prediction directory."
        )
    
    roost_example = roost_dir / "examples" / "roost-example.py"
    if not roost_example.exists():
        raise FileNotFoundError(
            f"Roost example script not found at {roost_example}. "
            "Please ensure aviary folder contains examples/roost-example.py."
        )

    # Roost expects CSV columns: material_id, composition, <property-name>
    # Use the actual property name as the header instead of "target"
    ensure_dir(outdir)
    train_path = (outdir / "roost_train.csv").resolve()
    test_path = (outdir / "roost_test.csv").resolve()

    def make_roost_csv(df, path: Path):
        d = df[[id_col, formula_col, target_col]].dropna().copy()
        # Use actual IDs as material_id and keep the property name as the header
        d.rename(columns={id_col: "material_id", formula_col: "composition"}, inplace=True)
        # Keep target_col as the column name (don't rename to "target")
        d.to_csv(path, index=False)

    make_roost_csv(train_df, train_path)
    make_roost_csv(test_df, test_path)

    print(f"[Roost] Train CSV: {train_path}")
    print(f"[Roost] Test CSV: {test_path}")
    print(f"[Roost] Property column: {target_col}")
    print(f"[Roost] Roost directory: {roost_dir.resolve()}")

    # Verify files exist
    if not train_path.exists():
        raise FileNotFoundError(f"Train CSV not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_path}")

    # Run training
    # Use the actual property name (target_col) as the target column name
    # This uses L1 loss (MAE-optimized). Adjust as desired.
    # Use absolute paths so they work regardless of working directory
    cmd = [
        sys.executable,
        "examples/roost-example.py",
        "--train",
        "--evaluate",
        "--epochs",
        str(epochs),
        "--tasks",
        "regression",
        "--targets",
        target_col,  # Use the actual property name
        "--losses",
        "L1",
        "--robust",
        "--data-path",
        str(train_path),
    ]
    _run(cmd, cwd=str(roost_dir.resolve()))

    print(
        "Roost finished. The example script prints metrics to stdout and writes artifacts in its run directory.\n"
        "Note: Roost predictions are saved in its run directory. Check the roost output for detailed results."
    )

    return {"status": "completed (see stdout for metrics)"}


def main():
    ap = argparse.ArgumentParser(
        description="Train and test composition-based models (XGBoost, CrabNet, Roost) on Materials Project or JARVIS CSVs. "
                    "Similar interface to run_gnns.py - reads separate train and test CSVs and outputs predictions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--train-csv",
        type=str,
        help="Path to train.csv (default: ./data/temp_mp/train.csv)",
    )
    ap.add_argument(
        "--test-csv",
        type=str,
        help="Path to test.csv (default: ./data/temp_mp/test.csv)",
    )
    ap.add_argument(
        "--data-dir",
        type=str,
        default="./data/temp_mp",
        help="Directory containing train.csv and test.csv (default: ./data/temp_mp)",
    )
    ap.add_argument(
        "--target-col",
        required=True,
        help="Numeric target column, e.g. formation_energy_per_atom or band_gap",
    )
    ap.add_argument(
        "--formula-col",
        default="formula_pretty",
        help="Formula column, e.g. formula_pretty (default: formula_pretty)",
    )
    ap.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["xgb", "crabnet", "roost"],
        help="Which model to run: 'xgb', 'crabnet', or 'roost'",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Roost epochs (default: 50)",
    )
    ap.add_argument(
        "--work-dir",
        type=str,
        default="./output",
        help="Working directory where predictions will be saved (default: ./output)",
    )
    ap.add_argument(
        "--crabnet-path",
        type=str,
        help="Path to local CrabNet installation (optional, defaults to installed package)",
    )
    ap.add_argument(
        "--roost-path",
        type=str,
        help="Path to local Roost repository directory (optional, defaults to aviary folder in property_prediction directory)",
    )
    args = ap.parse_args()

    # Determine train and test CSV paths
    if args.train_csv:
        train_csv = Path(args.train_csv)
    else:
        train_csv = Path(args.data_dir) / "train.csv"
    
    if args.test_csv:
        test_csv = Path(args.test_csv)
    else:
        test_csv = Path(args.data_dir) / "test.csv"

    if not train_csv.exists():
        raise FileNotFoundError(f"Train CSV not found: {train_csv}")
    if not test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    # Load CSVs
    print(f"\n{'='*80}")
    print(f"Loading training data from: {train_csv}")
    print(f"Loading test data from: {test_csv}")
    print(f"{'='*80}\n")
    
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # Detect ID column
    id_col = detect_id_column(train_df)
    if id_col != detect_id_column(test_df):
        raise ValueError("Train and test CSVs must use the same ID column (material_id or jarvis_id).")

    # Check required columns
    required_cols = [id_col, args.formula_col, args.target_col]
    for col in required_cols:
        if col not in train_df.columns:
            raise KeyError(f"Column '{col}' not in TRAIN CSV. Columns: {list(train_df.columns)[:30]} ...")
        if col not in test_df.columns:
            raise KeyError(f"Column '{col}' not in TEST CSV. Columns: {list(test_df.columns)[:30]} ...")

    # Drop NaNs in required columns
    train_df = train_df.dropna(subset=[args.formula_col, args.target_col]).reset_index(drop=True)
    test_df = test_df.dropna(subset=[args.formula_col, args.target_col]).reset_index(drop=True)

    work_dir = ensure_dir(Path(args.work_dir))
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Target column: {args.target_col}")
    print(f"Formula column: {args.formula_col}")
    print(f"ID column: {id_col}")
    print(f"Model: {args.model}")
    print(f"Work directory: {work_dir.resolve()}\n")

    result = None

    if args.model == "xgb":
        result = run_xgb(
            train_df, test_df, args.formula_col, args.target_col, id_col,
            work_dir, args.seed
        )
    elif args.model == "crabnet":
        crabnet_path = Path(args.crabnet_path).resolve() if args.crabnet_path else None
        result = run_crabnet(
            train_df, test_df, args.formula_col, args.target_col, id_col,
            work_dir, args.seed, crabnet_path
        )
    elif args.model == "roost":
        roost_path = Path(args.roost_path).resolve() if args.roost_path else None
        result = run_roost(
            train_df, test_df, args.formula_col, args.target_col, id_col,
            work_dir, args.epochs, roost_path
        )

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    if result:
        print(json.dumps({args.model: result}, indent=2))
    print(f"\nPredictions saved to: {work_dir.resolve()}")
    print("="*80)


if __name__ == "__main__":
    main()

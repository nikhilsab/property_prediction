#!/usr/bin/env python
"""
Train and test GNN models on JARVIS-style CSVs.

This script provides a unified interface to train GNN models, save their weights,
and run inference on test data.

Assumes CSV columns (at minimum):
    - 'material_id' OR 'jarvis_id': string ID (e.g. 'mp-123' or 'JVASP-9331')
    - 'cif_structure'   : full CIF string
    - '<target-col>'    : numeric property to predict, e.g. 'formation_energy_peratom'

Usage:
    python run_gnns.py --model m3gnet --train-csv train.csv --test-csv test.csv \\
                       --target-col formation_energy_peratom --work-dir ./output

Supported models:
    - m3gnet: M3GNet model using MatGL
    - alignn: ALIGNN model
    - cgcnn: CGCNN model

The script will:
1. Train the specified GNN model on the training dataset
2. Save the trained model weights to <work-dir>/<model>_weights/
3. Load the saved weights and run inference on the test dataset
4. Save predictions to <work-dir>/<model>_predictions.csv
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import Tuple

import pandas as pd
import subprocess


# ----------------------------- ALIGNN CONFIG TEMPLATE ------------------------- #

ALIGNN_CONFIG_TEMPLATE = {
    "dataset": "user_data",
    "id_tag": "jid",
    "target": "target",
    "atom_features": "cgcnn",  # Use CGCNN atom features (92 features)

    "n_train": None,
    "n_val": None,
    "n_test": None,

    "train_ratio": None,
    "val_ratio": None,
    "test_ratio": None,
    "keep_data_order": True,

    "batch_size": 16,
    "epochs": 10,

    "criterion": "mse",
    "optimizer": "adamw",
    "scheduler": "onecycle",
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,

    "cutoff": 8.0,
    "max_neighbors": 12,
    "neighbor_strategy": "k-nearest",

    "model": {
        "name": "alignn_atomwise",
        "alignn_layers": 3,
        "gcn_layers": 2,
        "atom_input_features": 92,  # CGCNN atom features have 92 dimensions
        "hidden_features": 64,
        "output_features": 1,
        "calculate_gradient": False,
        "gradwise_weight": 0,
        "stresswise_weight": 0,
        "atomwise_weight": 0,
        "additional_output_features": 0,
        "additional_output_weight": 0,
    }
}

# --------------------------- DATASET PREPARATION ----------------------------- #

def check_cif2cell_available() -> bool:
    """
    Check if cif2cell is available in the system PATH.
    
    On Unix systems, checks directories in os.environ["PATH"].
    On Windows, also checks common executable extensions.
    
    Returns:
        True if cif2cell is found, False otherwise
    """
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    
    # On Windows, also check common executable extensions
    if os.name == "nt":  # Windows
        exe_extensions = ["", ".exe", ".bat", ".cmd"]
    else:
        exe_extensions = [""]
    
    for path_dir in path_dirs:
        for ext in exe_extensions:
            cif2cell_path = os.path.join(path_dir, f"cif2cell{ext}")
            if os.path.isfile(cif2cell_path) and os.access(cif2cell_path, os.X_OK):
                return True
    
    # Also try using 'which' command (Unix) or 'where' command (Windows)
    try:
        if os.name == "nt":  # Windows
            result = subprocess.run(
                ["where", "cif2cell"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        else:  # Unix/Linux/Mac
            result = subprocess.run(
                ["which", "cif2cell"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        if result.returncode == 0 and result.stdout.strip():
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    return False


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


def prepare_cgcnn_dataset(
    train_csv: Path,
    test_csv: Path,
    target_col: str,
    out_root: Path,
) -> Tuple[int, int]:
    """
    Build a CGCNN dataset:

        out_root/
          id_prop.csv      # id,target
          <id>.cif

    Uses:
        - id  = row['material_id'] or row['jarvis_id'] (auto-detected)
        - cif = row['cif_structure']
        - target = row[target_col]

    Train rows come first, then test rows. Returns (n_train, n_test).
    """
    out_root.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    # Detect ID column
    id_col = detect_id_column(df_train)
    if id_col != detect_id_column(df_test):
        raise ValueError("Train and test CSVs must use the same ID column (material_id or jarvis_id).")

    # Check required columns
    required_cols = [id_col, "cif_structure", target_col]
    for col in required_cols:
        if col not in df_train.columns:
            raise ValueError(f"Column '{col}' not in TRAIN CSV.")
        if col not in df_test.columns:
            raise ValueError(f"Column '{col}' not in TEST CSV.")

    df_train["_split"] = "train"
    df_test["_split"] = "test"
    df_all = pd.concat([df_train, df_test], axis=0, ignore_index=True)

    ids = []
    targets = []

    for _, row in df_all.iterrows():
        struct_id = str(row[id_col])
        cif_str = row["cif_structure"]
        if not isinstance(cif_str, str) or not cif_str.strip():
            raise ValueError(f"Empty/invalid CIF for {id_col}={struct_id}")

        cif_path = out_root / f"{struct_id}.cif"
        with open(cif_path, "w", encoding="utf-8") as f:
            f.write(cif_str)

        ids.append(struct_id)
        targets.append(row[target_col])

    # CGCNN expects CSV without header row
    id_prop = pd.DataFrame({"id": ids, "target": targets})
    id_prop.to_csv(out_root / "id_prop.csv", index=False, header=False)

    n_train = int((df_all["_split"] == "train").sum())
    n_test = int((df_all["_split"] == "test").sum())

    print(f"[CGCNN] Dataset root: {out_root}")
    print(f"[CGCNN] Wrote {len(df_all)} CIFs + id_prop.csv")
    print(f"[CGCNN] n_train={n_train}, n_test={n_test}")

    return n_train, n_test


def prepare_alignn_dataset(
    train_csv: Path,
    test_csv: Path,
    target_col: str,
    out_root: Path,
) -> Tuple[int, int]:
    """
    Build an ALIGNN dataset:

        out_root/
          id_prop.csv      # jid,target
          <id>.cif

    Uses:
        - jid = row['material_id'] or row['jarvis_id'] (auto-detected)
        - cif = row['cif_structure']
        - target = row[target_col]

    Same logic as CGCNN but column named 'jid' for ALIGNN.
    
    IMPORTANT: Cleans up LMDB cache files to ensure fresh data when target column changes.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    
    # Clean up ALIGNN LMDB cache files to ensure fresh data when target column changes
    # ALIGNN caches data in LMDB format, and old cache can persist with old target values
    import glob
    lmdb_patterns = [
        "*.lmdb",
        "sample*.lmdb",
        "sampletrain_data*.lmdb",
        "sampleval_data*.lmdb", 
        "sampletest_data*.lmdb",
    ]
    for pattern in lmdb_patterns:
        for lmdb_path in glob.glob(str(out_root / pattern)):
            import shutil
            try:
                if Path(lmdb_path).is_dir():
                    shutil.rmtree(lmdb_path)
                    print(f"[ALIGNN] Removed cached LMDB: {lmdb_path}")
                elif Path(lmdb_path).is_file():
                    Path(lmdb_path).unlink()
                    print(f"[ALIGNN] Removed cached LMDB file: {lmdb_path}")
            except Exception as e:
                print(f"[ALIGNN] Warning: Could not remove {lmdb_path}: {e}")
    
    # Also remove any data_range files that might cache old statistics
    for data_range_file in out_root.glob("*data_range*"):
        try:
            data_range_file.unlink()
            print(f"[ALIGNN] Removed cached data_range file: {data_range_file}")
        except Exception as e:
            print(f"[ALIGNN] Warning: Could not remove {data_range_file}: {e}")
    
    # Also check parent directory (output_dir) for cache files
    # ALIGNN sometimes creates cache files in the output directory
    parent_dir = out_root.parent
    if parent_dir.exists():
        for data_range_file in parent_dir.glob("*data_range*"):
            try:
                data_range_file.unlink()
                print(f"[ALIGNN] Removed cached data_range file from output dir: {data_range_file}")
            except Exception as e:
                pass  # Silently ignore errors in parent directory

    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    # Detect ID column
    id_col = detect_id_column(df_train)
    if id_col != detect_id_column(df_test):
        raise ValueError("Train and test CSVs must use the same ID column (material_id or jarvis_id).")

    # Check required columns
    required_cols = [id_col, "cif_structure", target_col]
    for col in required_cols:
        if col not in df_train.columns:
            raise ValueError(f"Column '{col}' not in TRAIN CSV.")
        if col not in df_test.columns:
            raise ValueError(f"Column '{col}' not in TEST CSV.")

    df_train["_split"] = "train"
    df_test["_split"] = "test"
    df_all = pd.concat([df_train, df_test], axis=0, ignore_index=True)

    jids = []
    targets = []

    for _, row in df_all.iterrows():
        struct_id = str(row[id_col])
        cif_str = row["cif_structure"]
        if not isinstance(cif_str, str) or not cif_str.strip():
            raise ValueError(f"Empty/invalid CIF for {id_col}={struct_id}")

        cif_filename = f"{struct_id}.cif"
        cif_path = out_root / cif_filename
        with open(cif_path, "w", encoding="utf-8") as f:
            f.write(cif_str)

        # ALIGNN expects the CIF filename (with .cif extension) in the first column
        jids.append(cif_filename)
        targets.append(row[target_col])

    # ALIGNN expects CSV without header row
    id_prop = pd.DataFrame({"jid": jids, "target": targets})
    id_prop.to_csv(out_root / "id_prop.csv", index=False, header=False)
    n_train = int((df_all["_split"] == "train").sum())
    n_test = int((df_all["_split"] == "test").sum())

    print(f"[ALIGNN] Dataset root: {out_root}")
    print(f"[ALIGNN] Wrote {len(df_all)} CIFs + id_prop.csv")
    print(f"[ALIGNN] n_train={n_train}, n_test={n_test}")

    return n_train, n_test

# ----------------------------- UNIFIED TRAINING & INFERENCE ------------------------- #

def train_matgl_model(
    model_name: str,
    train_csv: Path,
    target_col: str,
    work_dir: Path,
    epochs: int = 50,
    batch_size: int = 16,
) -> Path:
    """
    Train a MatGL model (M3GNet or MEGNet) and save weights.
    
    Args:
        model_name: Either 'm3gnet' or 'megnet'
        train_csv: Path to training CSV
        target_col: Name of target property column
        work_dir: Working directory for outputs
        epochs: Number of training epochs
        batch_size: Batch size for training
    
    Returns:
        Path to saved model weights directory
    """
    import torch
    import lightning as L
    from lightning.pytorch.loggers import CSVLogger
    from dgl.data.utils import split_dataset
    from pymatgen.core import Structure
    from torch.utils.data import DataLoader

    from matgl.ext._pymatgen_dgl import Structure2Graph, get_element_list
    from matgl.graph._data_dgl import MGLDataset, collate_fn_graph
    from matgl.config import DEFAULT_ELEMENTS
    from matgl.utils.training import ModelLightningModule
    
    if model_name == "m3gnet":
        from matgl.models import M3GNet
        ModelClass = M3GNet
    elif model_name == "megnet":
        from matgl.models import MEGNet
        ModelClass = MEGNet
    else:
        raise ValueError(f"Unknown MatGL model: {model_name}")

    model_dir = work_dir / f"{model_name}_weights"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load training CSV
    df_train = pd.read_csv(train_csv)
    
    # Detect ID column
    id_col = detect_id_column(df_train)
    
    # Check required columns
    required_cols = [id_col, "cif_structure", target_col]
    for col in required_cols:
        if col not in df_train.columns:
            raise ValueError(f"Column '{col}' not in TRAIN CSV.")

    # Build training dataset
    train_structures: list[Structure] = []
    train_targets: list[float] = []

    print(f"[{model_name.upper()}] Parsing training structures from {train_csv}...")
    for _, row in df_train.iterrows():
        tgt = row[target_col]
        if pd.isna(tgt):
            continue

        cif_str = row["cif_structure"]
        struct_id = str(row[id_col])
        if not isinstance(cif_str, str) or not cif_str.strip():
            print(f"[{model_name.upper()}] Skipping train row with empty CIF ({id_col}={struct_id})")
            continue

        try:
            struct = Structure.from_str(cif_str, fmt="cif")
        except Exception as e:
            print(f"[{model_name.upper()}] Skipping train {id_col}={struct_id}: CIF parse error: {e}")
            continue

        train_structures.append(struct)
        train_targets.append(float(tgt))

    if not train_structures:
        raise RuntimeError(f"[{model_name.upper()}] No valid training samples found.")

    print(f"[{model_name.upper()}] Training samples: {len(train_structures)}")

    # Setup MatGL dataset
    elem_list = get_element_list(train_structures)
    converter = Structure2Graph(element_types=elem_list, cutoff=4.0)

    dataset = MGLDataset(
        structures=train_structures,
        labels={"target": train_targets},
        converter=converter,
    )

    train_data, val_data, _ = split_dataset(
        dataset,
        frac_list=[0.9, 0.1, 0.0],
        shuffle=True,
        random_state=42,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn_graph,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_graph,
    )

    # Create model
    model = ModelClass(
        element_types=DEFAULT_ELEMENTS,
        is_intensive=False,
    )

    lit_module = ModelLightningModule(model=model)

    logger = CSVLogger(
        save_dir=str(model_dir),
        name="training",
    )

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        logger=logger,
    )

    print(f"[{model_name.upper()}] Starting training for {epochs} epochs (batch_size={batch_size})...")
    trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Save model weights
    model_export_path = model_dir / "model"
    model_export_path.mkdir(parents=True, exist_ok=True)
    model.save(str(model_export_path))
    
    # Save converter parameters (not the converter itself, as it contains unpicklable objects)
    # We'll recreate the converter during inference using these parameters
    import json
    converter_info = {
        "element_types": elem_list,
        "cutoff": 4.0,
    }
    converter_path = model_dir / "converter_info.json"
    with open(converter_path, "w", encoding="utf-8") as f:
        json.dump(converter_info, f, indent=2)
    
    # Save ID column name and model name for inference
    id_col_path = model_dir / "id_column.txt"
    with open(id_col_path, "w", encoding="utf-8") as f:
        f.write(id_col)
    
    model_name_path = model_dir / "model_name.txt"
    with open(model_name_path, "w", encoding="utf-8") as f:
        f.write(model_name)
    
    print(f"[{model_name.upper()}] Saved trained model to: {model_export_path}")
    return model_export_path


def train_m3gnet(
    train_csv: Path,
    target_col: str,
    work_dir: Path,
    epochs: int = 50,
    batch_size: int = 16,
) -> Path:
    """Train M3GNet model and save weights."""
    return train_matgl_model("m3gnet", train_csv, target_col, work_dir, epochs, batch_size)


def train_megnet(
    train_csv: Path,
    target_col: str,
    work_dir: Path,
    epochs: int = 50,
    batch_size: int = 16,
) -> Path:
    """Train MEGNet model and save weights."""
    return train_matgl_model("megnet", train_csv, target_col, work_dir, epochs, batch_size)


def inference_matgl_model(
    model_name: str,
    test_csv: Path,
    target_col: str,
    model_path: Path,
    work_dir: Path,
    batch_size: int = 16,
) -> Path:
    """
    Run inference using saved MatGL model weights (M3GNet or MEGNet).
    
    Args:
        model_name: Either 'm3gnet' or 'megnet'
        test_csv: Path to test CSV
        target_col: Name of target property column
        model_path: Path to saved model directory
        work_dir: Working directory for outputs
        batch_size: Batch size for inference
    
    Returns:
        Path to predictions CSV file
    """
    import torch
    from sklearn.metrics import mean_absolute_error
    from pymatgen.core import Structure
    from torch.utils.data import DataLoader

    from matgl.graph._data_dgl import MGLDataset, collate_fn_graph
    
    if model_name == "m3gnet":
        from matgl.models import M3GNet
        ModelClass = M3GNet
    elif model_name == "megnet":
        from matgl.models import MEGNet
        ModelClass = MEGNet
    else:
        raise ValueError(f"Unknown MatGL model: {model_name}")

    # Load test CSV
    df_test = pd.read_csv(test_csv)
    
    # Load ID column name and model name from training (or detect if not saved)
    id_col_path = model_path.parent / "id_column.txt"
    if id_col_path.exists():
        with open(id_col_path, "r", encoding="utf-8") as f:
            id_col = f.read().strip()
    else:
        # Fallback: detect from test CSV
        id_col = detect_id_column(df_test)
        print(f"[{model_name.upper()}] Warning: ID column not saved from training, detected: {id_col}")
    
    # Verify model name matches
    model_name_path = model_path.parent / "model_name.txt"
    if model_name_path.exists():
        with open(model_name_path, "r", encoding="utf-8") as f:
            saved_model_name = f.read().strip()
        if saved_model_name != model_name:
            print(f"[{model_name.upper()}] Warning: Model name mismatch. Expected {model_name}, found {saved_model_name}")
    
    # Check required columns
    required_cols = [id_col, "cif_structure", target_col]
    for col in required_cols:
        if col not in df_test.columns:
            raise ValueError(f"Column '{col}' not in TEST CSV.")

    # Load converter info and recreate converter
    converter_info_path = model_path.parent / "converter_info.json"
    if not converter_info_path.exists():
        raise FileNotFoundError(
            f"Converter info not found at {converter_info_path}. "
            "Model may not have been trained properly."
        )
    
    import json
    with open(converter_info_path, "r", encoding="utf-8") as f:
        converter_info = json.load(f)
    
    # Recreate the converter using saved parameters
    from matgl.ext._pymatgen_dgl import Structure2Graph
    converter = Structure2Graph(
        element_types=converter_info["element_types"],
        cutoff=converter_info["cutoff"],
    )

    # Load model
    model = ModelClass.load(str(model_path))
    model.eval()

    # Build test dataset
    print(f"[{model_name.upper()}] Parsing test structures from {test_csv}...")
    test_structures: list[Structure] = []
    test_targets: list[float] = []
    test_ids: list[str] = []

    for _, row in df_test.iterrows():
        struct_id = str(row[id_col])
        cif_str = row["cif_structure"]
        tgt = row[target_col]

        if pd.isna(tgt):
            continue

        if not isinstance(cif_str, str) or not cif_str.strip():
            print(f"[{model_name.upper()}] Skipping test {id_col}={struct_id}: empty/invalid CIF.")
            continue

        try:
            struct = Structure.from_str(cif_str, fmt="cif")
        except Exception as e:
            print(f"[{model_name.upper()}] Skipping test {id_col}={struct_id}: CIF parse error: {e}")
            continue

        test_structures.append(struct)
        test_targets.append(float(tgt))
        test_ids.append(struct_id)

    if not test_structures:
        raise RuntimeError(f"[{model_name.upper()}] No valid test samples to evaluate.")

    test_dataset = MGLDataset(
        structures=test_structures,
        labels={"target": test_targets},
        converter=converter,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_graph,
    )

    # Run inference
    print(f"[{model_name.upper()}] Running inference on test set...")
    preds: list[float] = []

    with torch.no_grad():
        for batch in test_loader:
            g, edge_feat, node_feat, state_feat, _ = batch
            out = model(g, edge_feat, node_feat, state_feat)
            preds.extend(out.cpu().numpy().ravel().tolist())

    # Align predictions with targets
    n = min(len(preds), len(test_targets), len(test_ids))
    preds = preds[:n]
    test_targets = test_targets[:n]
    test_ids = test_ids[:n]

    mae = mean_absolute_error(test_targets, preds)
    print(f"[{model_name.upper()}] Test MAE on {len(test_targets)} structures: {mae:.6f}")

    # Save predictions (use the detected ID column name)
    pred_df = pd.DataFrame({
        id_col: test_ids,
        "y_true": test_targets,
        "y_pred": preds,
    })
    pred_path = work_dir / f"{model_name}_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"[{model_name.upper()}] Saved predictions to: {pred_path}")
    return pred_path


def inference_m3gnet(
    test_csv: Path,
    target_col: str,
    model_path: Path,
    work_dir: Path,
    batch_size: int = 16,
) -> Path:
    """Run inference using saved M3GNet model weights."""
    return inference_matgl_model("m3gnet", test_csv, target_col, model_path, work_dir, batch_size)


def inference_megnet(
    test_csv: Path,
    target_col: str,
    model_path: Path,
    work_dir: Path,
    batch_size: int = 16,
) -> Path:
    """Run inference using saved MEGNet model weights."""
    return inference_matgl_model("megnet", test_csv, target_col, model_path, work_dir, batch_size)




# ----------------------------- RUN ALIGNN ------------------------------------ #
def run_alignn(
    dataset_root: Path,
    n_train: int,
    n_test: int,
    output_dir: Path,
    alignn_root: Path,
    epochs: int = 10,
    batch_size: int = 16,
):
    """
    Run ALIGNN using the cloned repo at `alignn_root`, without needing pip install.

    We do:
        PYTHONPATH=<alignn_root> python -m alignn.train ...
    so that Python can find alignn/alignn/train.py.
    """
    import json
    import os

    # Check if cif2cell is available (jarvis-tools may need it)
    if not check_cif2cell_available():
        print(
            "[ALIGNN] WARNING: cif2cell is not found in system PATH. "
            "jarvis-tools may require cif2cell for some operations. "
            "If you encounter errors, install cif2cell:\n"
            "  pip install cif2cell\n"
            "  or\n"
            "  conda install -c conda-forge cif2cell\n"
            "Continuing anyway..."
        )

    # 1. Build config from our in-script template
    cfg = json.loads(json.dumps(ALIGNN_CONFIG_TEMPLATE))  # deep copy

    # Split training data: use 90% for training, 10% for validation
    # This avoids the ALIGNN bug where val_loader.dataset is None when n_val=0
    # We split n_train into train + val, and keep n_test separate
    # Convert to native Python ints to avoid JSON serialization issues with numpy/pandas types
    n_train_int = int(n_train)
    n_test_int = int(n_test)
    
    # Ensure we have enough samples for both training and validation
    # Need at least 2 samples: 1 for training, 1 for validation
    if n_train_int < 2:
        raise ValueError(
            f"Training set too small: {n_train_int} samples. "
            "Need at least 2 samples (1 for training, 1 for validation)."
        )
    
    # Calculate validation size (10% of training, but at least 1)
    val_size = max(1, int(n_train_int * 0.1))
    # Ensure training size is at least 1
    actual_train_size = max(1, n_train_int - val_size)
    
    # If actual_train_size is too small, adjust val_size
    if actual_train_size < 1:
        actual_train_size = n_train_int - 1
        val_size = 1
    
    print(f"[ALIGNN] Splitting {n_train_int} training samples: {actual_train_size} train, {val_size} validation")
    
    # Ensure training size is large enough for at least one batch
    # This prevents steps_per_epoch from being 0
    if actual_train_size < cfg.get("batch_size", 16):
        print(
            f"[ALIGNN] WARNING: Training size ({actual_train_size}) is smaller than batch size ({cfg.get('batch_size', 16)}). "
            "This may cause issues. Consider reducing batch_size or increasing training data."
        )
    
    # Use fixed sizes to respect the train/test split we've already made
    cfg["n_train"] = actual_train_size
    cfg["n_val"] = val_size
    cfg["n_test"] = n_test_int
    cfg["train_ratio"] = None
    cfg["val_ratio"] = None
    cfg["test_ratio"] = None
    cfg["keep_data_order"] = True

    # Update batch_size and epochs from arguments
    cfg["batch_size"] = batch_size
    cfg["epochs"] = epochs
    
    cfg["id_tag"] = "jid"
    cfg["target"] = "target"
    cfg["output_dir"] = str(output_dir)

    config_path = dataset_root / "config_alignn.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(f"[ALIGNN] Wrote config: {config_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Prepare environment: add alignn_root to PYTHONPATH
    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    # Make sure alignn_root is an absolute path
    alignn_root_abs = str(alignn_root.resolve())
    env["PYTHONPATH"] = (
        alignn_root_abs if not existing_pp else alignn_root_abs + os.pathsep + existing_pp
    )
    
    # Force single-process execution to avoid pickle errors with Environment objects
    # ALIGNN's multiprocessing tries to pickle Atoms objects which contain Environment objects
    # Setting CUDA_VISIBLE_DEVICES to only the first GPU forces world_size=1
    if "CUDA_VISIBLE_DEVICES" not in env:
        # If not set, use only the first GPU (device 0)
        # This forces ALIGNN to use single-process mode
        import torch
        try:
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                env["CUDA_VISIBLE_DEVICES"] = "0"
                print("[ALIGNN] Limiting to GPU 0 to force single-process mode (avoids pickle errors with Environment objects)")
        except Exception:
            # If torch is not available or there's an error, continue without setting
            pass

    # 3. Call the module from the cloned repo
    
    cmd = [
        "python",
        "-m",
        "alignn.train_alignn",
        "--root_dir", str(dataset_root),
        "--config", str(config_path),
        "--output_dir", str(output_dir),
        "--file_format", "cif",
    ]
    print(f"[ALIGNN] Running with PYTHONPATH={env['PYTHONPATH']}:")
    print("         " + " ".join(cmd))

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    print("\n=== ALIGNN STDOUT ===\n", result.stdout)
    print("\n=== ALIGNN STDERR ===\n", result.stderr)

    if result.returncode != 0:
        raise RuntimeError("ALIGNN failed. See STDERR above.")
    print(f"[ALIGNN] Done. Predictions/logs in: {output_dir}")


# ----------------------------- RUN CGCNN ------------------------------------- #

def run_cgcnn(
    dataset_root: Path,
    n_train: int,
    n_test: int,
    cgcnn_root: Path,
    atom_init: Path,
    val_ratio: float = 0.1,
):
    # copy atom_init.json
    dst_atom_init = dataset_root / "atom_init.json"
    if not dst_atom_init.exists():
        shutil.copy2(atom_init, dst_atom_init)
        print(f"[CGCNN] Copied atom_init.json to {dst_atom_init}")

    total_train = n_train
    val_size = max(1, int(val_ratio * total_train))
    train_size = total_train - val_size
    test_size = n_test

    main_py = cgcnn_root / "main.py"
    if not main_py.is_file():
        raise FileNotFoundError(f"CGCNN main.py not found at {main_py}")

    cmd = [
        "python",
        str(main_py),
        "--train-size",
        str(train_size),
        "--val-size",
        str(val_size),
        "--test-size",
        str(test_size),
        str(dataset_root),
    ]
    print(f"[CGCNN] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        raise RuntimeError("CGCNN training failed.")
    print("[CGCNN] Done. Check cgcnn/ outputs for model & test results.")


# --------------------------------- CLI --------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and test GNN models on Materials Project or JARVIS CSVs. "
                    "The script automatically detects material_id (MP) or jarvis_id (JARVIS) columns. "
                    "It trains the model, saves weights, and runs inference."
    )
    parser.add_argument(
        "--model",
        choices=["m3gnet", "megnet", "alignn", "cgcnn"],
        required=True,
        help="GNN model to train and test: 'm3gnet', 'megnet', 'alignn', or 'cgcnn'"
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        required=True,
        help="Path to train.csv (same schema as your test.csv).",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        required=True,
        help="Path to test.csv (same schema as your test.csv).",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        required=True,
        help="Name of target property column (e.g. formation_energy_peratom).",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        required=True,
        help="Working directory where model weights and outputs will be saved.",
    )

    # CGCNN-specific (required if model=cgcnn)
    parser.add_argument(
        "--cgcnn-root",
        type=str,
        help="Path to CGCNN repo root (must contain main.py). Required for model=cgcnn.",
    )
    parser.add_argument(
        "--atom-init",
        type=str,
        help="Path to atom_init.json. Required for model=cgcnn.",
    )

    # ALIGNN-specific (required if model=alignn)
    parser.add_argument(
        "--alignn-root",
        type=str,
        help="Path to the ALIGNN repo clone. Required for model=alignn.",
    )
    parser.add_argument(
        "--alignn-outdir",
        type=str,
        help="Output dir for ALIGNN (defaults to <work-dir>/alignn_runs).",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training and inference (default: 16).",
    )

    return parser.parse_args()


def train_and_test_gnn(
    model_type: str,
    train_csv: Path,
    test_csv: Path,
    target_col: str,
    work_dir: Path,
    epochs: int = 50,
    batch_size: int = 16,
    **kwargs
):
    """
    Unified function to train a GNN model, save weights, and run inference.
    
    Args:
        model_type: Type of GNN model ('m3gnet', 'alignn', 'cgcnn')
        train_csv: Path to training CSV
        test_csv: Path to test CSV
        target_col: Name of target property column
        work_dir: Working directory for outputs
        epochs: Number of training epochs
        batch_size: Batch size for training/inference
        **kwargs: Additional model-specific arguments
    """
    print("\n" + "=" * 80)
    print(f"Training {model_type.upper()} model on target: {target_col}")
    print("=" * 80)
    
    work_dir.mkdir(parents=True, exist_ok=True)
    
    if model_type == "m3gnet":
        # Train M3GNet
        model_path = train_m3gnet(
            train_csv=train_csv,
            target_col=target_col,
            work_dir=work_dir,
            epochs=epochs,
            batch_size=batch_size,
        )
        
        # Run inference
        inference_m3gnet(
            test_csv=test_csv,
            target_col=target_col,
            model_path=model_path,
            work_dir=work_dir,
            batch_size=batch_size,
        )
        
    elif model_type == "megnet":
        # Train MEGNet
        model_path = train_megnet(
            train_csv=train_csv,
            target_col=target_col,
            work_dir=work_dir,
            epochs=epochs,
            batch_size=batch_size,
        )
        
        # Run inference
        inference_megnet(
            test_csv=test_csv,
            target_col=target_col,
            model_path=model_path,
            work_dir=work_dir,
            batch_size=batch_size,
        )
        
    elif model_type == "alignn":
        if "alignn_root" not in kwargs or kwargs["alignn_root"] is None:
            raise ValueError("For ALIGNN, provide --alignn-root.")
        
        alignn_root = Path(kwargs["alignn_root"])
        alignn_dataset_root = work_dir / "alignn_dataset"
        
        # Prepare dataset (this will clean up any cached LMDB/data_range files)
        n_train, n_test = prepare_alignn_dataset(
            train_csv=train_csv,
            test_csv=test_csv,
            target_col=target_col,
            out_root=alignn_dataset_root,
        )
        
        # Train (ALIGNN saves weights internally and runs inference automatically)
        # Note: ALIGNN automatically runs inference on test set during training
        # if write_predictions is True (which is the default)
        outdir = kwargs.get("alignn_outdir")
        if outdir is None:
            outdir = work_dir  # Default to work_dir if not specified
        else:
            outdir = Path(outdir)
        
        # Also clean up any old prediction files in output directory to avoid confusion
        # when target column changes
        for old_pred_file in Path(outdir).glob("prediction_results_*.csv"):
            try:
                old_pred_file.unlink()
                print(f"[ALIGNN] Removed old prediction file: {old_pred_file}")
            except Exception as e:
                pass  # Silently ignore if file doesn't exist
        
        run_alignn(
            dataset_root=alignn_dataset_root,
            n_train=n_train,
            n_test=n_test,
            output_dir=Path(outdir),
            alignn_root=alignn_root,
            epochs=epochs,
            batch_size=batch_size,
        )
        
        # Verify that predictions were written
        # ALIGNN writes predictions to output_dir/prediction_results_test_set.csv
        pred_file = Path(outdir) / "prediction_results_test_set.csv"
        if pred_file.exists():
            print(f"[ALIGNN] ✓ Test predictions saved to: {pred_file}")
            # Show a preview of the predictions
            try:
                import pandas as pd
                pred_df = pd.read_csv(pred_file)
                print(f"[ALIGNN]   Predictions shape: {pred_df.shape}")
                if len(pred_df) > 0:
                    print(f"[ALIGNN]   Columns: {list(pred_df.columns)}")
            except Exception:
                pass
        else:
            print(f"[ALIGNN] ⚠ Warning: Test predictions not found at {pred_file}")
            print(f"[ALIGNN]   Check {outdir} for other prediction files")
            # List files in output directory to help debug
            try:
                files = list(Path(outdir).glob("*.csv"))
                if files:
                    print(f"[ALIGNN]   Found CSV files: {[f.name for f in files]}")
            except Exception:
                pass
        
        print(f"[ALIGNN] Model weights and predictions saved to: {outdir}")
        
    elif model_type == "cgcnn":
        if "cgcnn_root" not in kwargs or kwargs["cgcnn_root"] is None:
            raise ValueError("For CGCNN, provide --cgcnn-root and --atom-init.")
        if "atom_init" not in kwargs or kwargs["atom_init"] is None:
            raise ValueError("For CGCNN, provide --atom-init.")
        
        cgcnn_root = Path(kwargs["cgcnn_root"])
        atom_init = Path(kwargs["atom_init"])
        cgcnn_dataset_root = work_dir / "cgcnn_dataset"
        
        # Prepare dataset
        n_train, n_test = prepare_cgcnn_dataset(
            train_csv=train_csv,
            test_csv=test_csv,
            target_col=target_col,
            out_root=cgcnn_dataset_root,
        )
        
        # Train (CGCNN saves weights internally)
        run_cgcnn(
            dataset_root=cgcnn_dataset_root,
            n_train=n_train,
            n_test=n_test,
            cgcnn_root=cgcnn_root,
            atom_init=atom_init,
        )
        
        print(f"[CGCNN] Model weights and predictions saved to: {cgcnn_dataset_root}")
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def main():
    args = parse_args()

    train_csv = Path(args.train_csv)
    test_csv = Path(args.test_csv)
    work_dir = Path(args.work_dir)
    target_col = args.target_col
    model_type = args.model.lower()

    # Prepare kwargs for model-specific arguments
    kwargs = {}
    if model_type == "alignn":
        kwargs["alignn_root"] = getattr(args, "alignn_root", None)
        kwargs["alignn_outdir"] = getattr(args, "work_dir", None)
    elif model_type == "cgcnn":
        kwargs["cgcnn_root"] = getattr(args, "cgcnn_root", None)
        kwargs["atom_init"] = getattr(args, "atom_init", None)

    # Train and test the model
    train_and_test_gnn(
        model_type=model_type,
        train_csv=train_csv,
        test_csv=test_csv,
        target_col=target_col,
        work_dir=work_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        **kwargs
    )
    
    print("\n" + "=" * 80)
    print("Training and inference completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

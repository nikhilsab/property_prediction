#!/usr/bin/env python
"""
Run CGCNN / ALIGNN on JARVIS-style CSVs (like your test.csv).

Assumes CSV columns (at minimum):
    - 'jarvis_id'       : string ID, e.g. 'JVASP-9331'
    - 'cif_structure'   : full CIF string
    - '<target-col>'    : numeric property to predict, e.g. 'formation_energy_peratom'

You provide:
    --train-csv path/to/train.csv
    --test-csv  path/to/test.csv
    --target-col formation_energy_peratom
    --backend cgcnn|alignn|both

This script will:

1. Read train/test CSVs.
2. Create two dataset folders:

   work_dir/
     ├─ cgcnn_dataset/
     │    ├─ id_prop.csv      # id,target (ids = jarvis_id)
     │    ├─ JVASP-9331.cif
     │    ├─ JVASP-133335.cif
     │    └─ ...
     └─ alignn_dataset/
          ├─ id_prop.csv      # jid,target (jids = jarvis_id)
          ├─ JVASP-9331.cif
          ├─ JVASP-133335.cif
          └─ ...

   - Train rows come first, then test rows (to control splits).
   - n_train = len(train.csv), n_test = len(test.csv).

3. Run:
   - CGCNN:  python <cgcnn_root>/main.py ...
   - ALIGNN: train_alignn.py --root_dir alignn_dataset --config config_alignn.json ...

You must have:
    - CGCNN repo clone (for --cgcnn-root) + atom_init.json
    - alignn installed (pip install alignn jarvis-tools) for ALIGNN backend
"""

import argparse
import json
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

    "n_train": None,
    "n_val": 0,
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
        "hidden_features": 64,
        "output_features": 1,
    }
}

# --------------------------- DATASET PREPARATION ----------------------------- #

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
          JVASP-xxxx.cif

    Uses:
        - id  = row['jarvis_id']
        - cif = row['cif_structure']
        - target = row[target_col]

    Train rows come first, then test rows. Returns (n_train, n_test).
    """
    out_root.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    for col in ["jarvis_id", "cif_structure", target_col]:
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
        jid = str(row["jarvis_id"])
        cif_str = row["cif_structure"]
        if not isinstance(cif_str, str) or not cif_str.strip():
            raise ValueError(f"Empty/invalid CIF for jarvis_id={jid}")

        cif_path = out_root / f"{jid}.cif"
        with open(cif_path, "w") as f:
            f.write(cif_str)

        ids.append(jid)
        targets.append(row[target_col])

    id_prop = pd.DataFrame({"id": ids, "target": targets})
    id_prop.to_csv(out_root / "id_prop.csv", index=False)

    n_train = (df_all["_split"] == "train").sum()
    n_test = (df_all["_split"] == "test").sum()

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
          JVASP-xxxx.cif

    Same logic as CGCNN but column named 'jid'.
    """
    out_root.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    for col in ["jarvis_id", "cif_structure", target_col]:
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
        jid = str(row["jarvis_id"])
        cif_str = row["cif_structure"]
        if not isinstance(cif_str, str) or not cif_str.strip():
            raise ValueError(f"Empty/invalid CIF for jarvis_id={jid}")

        cif_path = out_root / f"{jid}.cif"
        with open(cif_path, "w") as f:
            f.write(cif_str)

        jids.append(jid)
        targets.append(row[target_col])

    id_prop = pd.DataFrame({"jid": jids, "target": targets})
    id_prop.to_csv(out_root / "id_prop.csv", index=False)

    n_train = (df_all["_split"] == "train").sum()
    n_test = (df_all["_split"] == "test").sum()

    print(f"[ALIGNN] Dataset root: {out_root}")
    print(f"[ALIGNN] Wrote {len(df_all)} CIFs + id_prop.csv")
    print(f"[ALIGNN] n_train={n_train}, n_test={n_test}")

    return n_train, n_test

def run_m3gnet_matgl(
    train_csv: Path,
    test_csv: Path,
    target_col: str,
    work_dir: Path,
    epochs: int = 50,
    batch_size: int = 16,
):
    """
    Train a MEGNet/M3GNet-style property model from scratch using MatGL on train.csv
    and evaluate it on test.csv.

    This follows the official MatGL tutorial:
    "Training a MEGNet Formation Energy Model with PyTorch Lightning"
    but swaps in your JARVIS-style CSVs instead of the MP dataset.

    - Uses train_csv for training/validation.
    - Uses test_csv ONLY for final evaluation (MAE + predictions CSV).
    - Target is taken from `target_col` (e.g. 'formation_energy_peratom').

    Dependencies inside your env:
        pip install matgl lightning dgl torch pymatgen scikit-learn
    """
    import numpy as np
    import torch
    import lightning as L
    from lightning.pytorch.loggers import CSVLogger
    from dgl.data.utils import split_dataset
    from sklearn.metrics import mean_absolute_error
    from pymatgen.core import Structure
    from torch.utils.data import DataLoader

    from matgl.ext._pymatgen_dgl import Structure2Graph, get_element_list
    from matgl.graph._data_dgl import MGLDataset, collate_fn_graph
    from matgl.layers import BondExpansion
    from matgl.models import M3GNet
    from matgl.config import DEFAULT_ELEMENTS

    from matgl.utils.training import ModelLightningModule

    out_dir = work_dir / "m3gnet_matgl"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------- Load train / test CSVs --------------------- #
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    for col in ["jarvis_id", "cif_structure", target_col]:
        if col not in df_train.columns:
            raise ValueError(f"Column '{col}' not in TRAIN CSV.")
        if col not in df_test.columns:
            raise ValueError(f"Column '{col}' not in TEST CSV.")

    # --------------------- Build training dataset --------------------- #
    train_structures: list[Structure] = []
    train_targets: list[float] = []

    print("[M3GNet/MEGNet-MatGL] Parsing training structures...")
    for _, row in df_train.iterrows():
        tgt = row[target_col]
        if pd.isna(tgt):
            # skip entries without target
            continue

        cif_str = row["cif_structure"]
        if not isinstance(cif_str, str) or not cif_str.strip():
            print(f"[M3GNet/MEGNet-MatGL] Skipping train row with empty CIF (jarvis_id={row['jarvis_id']})")
            continue

        try:
            struct = Structure.from_str(cif_str, fmt="cif")
        except Exception as e:
            print(f"[M3GNet/MEGNet-MatGL] Skipping train {row['jarvis_id']}: CIF parse error: {e}")
            continue

        train_structures.append(struct)
        train_targets.append(float(tgt))

    if not train_structures:
        raise RuntimeError("[M3GNet/MEGNet-MatGL] No valid training samples found.")

    print(f"[M3GNet/MEGNet-MatGL] Training samples: {len(train_structures)}")

    # --------------------- MatGL dataset & loaders --------------------- #
    # Get element types from training structures and set up converter
    elem_list = get_element_list(train_structures)
    converter = Structure2Graph(element_types=elem_list, cutoff=4.0)

    # MatGL dataset: labels dict can have arbitrary key; we'll use "target"
    dataset = MGLDataset(
        structures=train_structures,
        labels={"target": train_targets},
        converter=converter,
    )

    # Split into train/val (we don't use this split's "test"; our real test is test_csv)
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

    # --------------------- Define MEGNet model (tutorial-style) --------------------- #
    bond_expansion = BondExpansion(
        rbf_type="Gaussian",
        initial=0.0,
        final=5.0,
        num_centers=100,
        width=0.5,
    )

    element_types = DEFAULT_ELEMENTS
    model = M3GNet(
        element_types=element_types,
        is_intensive=False,
    )

    lit_module = ModelLightningModule(model=model)

    logger = CSVLogger(
        save_dir=str(out_dir),
        name="MEGNet_training",
    )

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="auto",  # GPU if available
        logger=logger,
    )

    print(f"[M3GNet/MEGNet-MatGL] Starting training for {epochs} epochs (batch_size={batch_size})...")
    trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # --------------------- Save trained model --------------------- #
    model_export_path = out_dir / "trained_megnet"
    model_export_path.mkdir(parents=True, exist_ok=True)

    # MEGNet inherits IOMixIn -> has save()
    model.save(str(model_export_path))
    print(f"[M3GNet/MEGNet-MatGL] Saved trained model to: {model_export_path}")

    # --------------------- Evaluate on test.csv --------------------- #
    print("[M3GNet/MEGNet-MatGL] Predicting on test set...")

    test_structures: list[Structure] = []
    test_targets: list[float] = []
    test_ids: list[str] = []

    for _, row in df_test.iterrows():
        jid = str(row["jarvis_id"])
        cif_str = row["cif_structure"]
        tgt = row[target_col]

        if pd.isna(tgt):
            continue

        if not isinstance(cif_str, str) or not cif_str.strip():
            print(f"[M3GNet/MEGNet-MatGL] Skipping test {jid}: empty/invalid CIF.")
            continue

        try:
            struct = Structure.from_str(cif_str, fmt="cif")
        except Exception as e:
            print(f"[M3GNet/MEGNet-MatGL] Skipping test {jid}: CIF parse error: {e}")
            continue

        test_structures.append(struct)
        test_targets.append(float(tgt))
        test_ids.append(jid)

    if not test_structures:
        raise RuntimeError("[M3GNet/MEGNet-MatGL] No valid test samples to evaluate.")

    # Build a MatGL dataset for test structures (same converter & element types)
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

    model.eval()
    preds: list[float] = []

    with torch.no_grad():
        for batch in test_loader:
            # collate_fn_graph returns: (graph, edge_feat, node_feat, state_feat, labels_dict)
            g, edge_feat, node_feat, state_feat, labels_dict = batch
            out = model(g, edge_feat, node_feat, state_feat)
            # out shape: (batch_size, 1)
            preds.extend(out.cpu().numpy().ravel().tolist())

    # Sanity check alignment
    if len(preds) != len(test_targets):
        print(
            f"[M3GNet/MEGNet-MatGL] WARNING: #predictions ({len(preds)}) "
            f"!= #test_targets ({len(test_targets)}). Truncating to min length."
        )
        n = min(len(preds), len(test_targets), len(test_ids))
        preds = preds[:n]
        test_targets = test_targets[:n]
        test_ids = test_ids[:n]

    mae = mean_absolute_error(test_targets, preds)
    print(f"[M3GNet/MEGNet-MatGL] Test MAE on {len(test_targets)} structures: {mae:.6f} (same units as {target_col})")

    pred_df = pd.DataFrame(
        {
            "jarvis_id": test_ids,
            "y_true": test_targets,
            "y_pred": preds,
        }
    )
    pred_path = out_dir / "m3gnet_trained_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"[M3GNet/MEGNet-MatGL] Saved predictions to: {pred_path}")




# ----------------------------- RUN ALIGNN ------------------------------------ #
def run_alignn(
    dataset_root: Path,
    n_train: int,
    n_test: int,
    output_dir: Path,
    alignn_root: Path,
):
    """
    Run ALIGNN using the cloned repo at `alignn_root`, without needing pip install.

    We do:
        PYTHONPATH=<alignn_root> python -m alignn.train ...
    so that Python can find alignn/alignn/train.py.
    """
    import json
    import subprocess
    import os

    # 1. Build config from our in-script template
    cfg = json.loads(json.dumps(ALIGNN_CONFIG_TEMPLATE))  # deep copy

    cfg["n_train"] = int(n_train)
    cfg["n_val"] = 0
    cfg["n_test"] = int(n_test)
    cfg["train_ratio"] = None
    cfg["val_ratio"] = None
    cfg["test_ratio"] = None
    cfg["keep_data_order"] = True

    cfg["id_tag"] = "jid"
    cfg["target"] = "target"
    cfg["output_dir"] = str(output_dir)

    config_path = dataset_root / "config_alignn.json"
    with open(config_path, "w") as f:
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

    # 3. Call the module from the cloned repo
    cmd = [
        "python",
        "-m",
        "alignn.train_alignn",
        "--root_dir", str(dataset_root),
        "--config", str(config_path),
        "--output_dir", str(output_dir),
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
    subprocess.run(cmd, check=True)
    print("[CGCNN] Done. Check cgcnn/ outputs for model & test results.")


# --------------------------------- CLI --------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CGCNN / ALIGNN / M3GNet on JARVIS-style CSVs "
                    "(jarvis_id, cif_structure, many properties)."
    )
    parser.add_argument(
        "--backend",
        choices=["cgcnn", "alignn", "m3gnet", "both", "all"],
        required=True,
        help=(
            "Which backend(s) to run. "
            "'cgcnn'  = CGCNN only, "
            "'alignn' = ALIGNN only, "
            "'m3gnet' = train a fresh M3GNet (MatGL) model on train.csv and evaluate on test.csv, "
            "'both'   = CGCNN + ALIGNN, "
            "'all'    = CGCNN + ALIGNN + M3GNet."
        ),
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
        help="Working directory where datasets and runs will be created.",
    )

    # CGCNN-specific
    parser.add_argument(
        "--cgcnn-root",
        type=str,
        help="Path to CGCNN repo root (must contain main.py). Required for backend=cgcnn or both.",
    )
    parser.add_argument(
        "--atom-init",
        type=str,
        help="Path to atom_init.json (e.g. cgcnn/data/sample-regression/atom_init.json).",
    )

    # ALIGNN-specific
    parser.add_argument(
        "--alignn-outdir",
        type=str,
        help="Output dir for ALIGNN (defaults to <work-dir>/alignn_runs).",
    )

    parser.add_argument(
        "--alignn-root",
        type=str,
        required=False,
        help="Path to the ALIGNN repo clone (directory containing alignn/ and scripts/).",
    )

    parser.add_argument(
        "--m3gnet-epochs",
        type=int,
        default=50,
        help="Number of epochs for M3GNet training (default: 50).",
    )
    parser.add_argument(
        "--m3gnet-batch-size",
        type=int,
        default=16,
        help="Batch size for M3GNet training (default: 16).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    train_csv = Path(args.train_csv)
    test_csv = Path(args.test_csv)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    backend = args.backend.lower()
    target_col = args.target_col

    # -------------------- CGCNN dataset + run --------------------- #
    if backend in ["cgcnn", "both"]:
        if args.cgcnn_root is None or args.atom_init is None:
            raise ValueError("For CGCNN, provide --cgcnn-root and --atom-init.")
        cgcnn_root = Path(args.cgcnn_root)
        atom_init = Path(args.atom_init)

        cgcnn_dataset_root = work_dir / "cgcnn_dataset"
        n_train_cgcnn, n_test_cgcnn = prepare_cgcnn_dataset(
            train_csv=train_csv,
            test_csv=test_csv,
            target_col=target_col,
            out_root=cgcnn_dataset_root,
        )
        run_cgcnn(
            dataset_root=cgcnn_dataset_root,
            n_train=n_train_cgcnn,
            n_test=n_test_cgcnn,
            cgcnn_root=cgcnn_root,
            atom_init=atom_init,
        )

    # -------------------- ALIGNN dataset + run -------------------- #
    if backend in ["alignn", "both"]:
        if args.alignn_root is None:
            raise ValueError(
                "You selected backend=alignn or both, but did not provide --alignn-root.\n"
                "Example: --alignn-root ./alignn"
            )

        alignn_root = Path(args.alignn_root)

        alignn_dataset_root = work_dir / "alignn_dataset"
        n_train_alignn, n_test_alignn = prepare_alignn_dataset(
            train_csv=train_csv,
            test_csv=test_csv,
            target_col=target_col,
            out_root=alignn_dataset_root,
        )

        outdir = Path(args.alignn_outdir) if hasattr(args, "alignn_outdir") and args.alignn_outdir \
                else (work_dir / "alignn_runs")

        run_alignn(
            dataset_root=alignn_dataset_root,
            n_train=n_train_alignn,
            n_test=n_test_alignn,
            output_dir=outdir,
            alignn_root=alignn_root,
        )

    # -------------------- M3GNet (MatGL) run -------------------- #
    # -------------------- M3GNet (MatGL) run -------------------- #
    if backend in ["m3gnet", "all"]:
        print("\n" + "#" * 80)
        print(f"Running M3GNet (MatGL) training from scratch on target: {target_col}")
        print("#" * 80)

        run_m3gnet_matgl(
            train_csv=train_csv,
            test_csv=test_csv,
            target_col=target_col,
            work_dir=work_dir,
            epochs=args.m3gnet_epochs,
            batch_size=args.m3gnet_batch_size,
        )


if __name__ == "__main__":
    main()

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
    "id_tag": "jid",         # must match id_prop.csv column
    "target": "target",      # must match id_prop.csv column
    "dtype": "float32",
    "random_seed": 123,

    # split control
    "n_train": None,
    "n_val": 0,
    "n_test": None,
    "train_ratio": None,
    "val_ratio": None,
    "test_ratio": None,
    "keep_data_order": True,

    # training
    "batch_size": 64,
    "epochs": 200,
    "criterion": "mse",
    "optimizer": "adamw",
    "scheduler": "onecycle",
    "weight_decay": 1e-5,
    "learning_rate": 1e-3,

    # graph construction
    "atom_features": "cgcnn",
    "neighbor_strategy": "k-nearest",
    "cutoff": 8.0,
    "cutoff_extra": 3.0,
    "max_neighbors": 12,
    "use_canonize": True,
    "num_workers": 0,

    # bookkeeping
    "normalize_graph_level_loss": False,
    "distributed": False,
    "data_parallel": False,
    "output_dir": "temp",
    "use_lmdb": False,
    "progress": True,
    "write_checkpoint": True,
    "write_predictions": True,
    "save_dataloader": False,

    # model block – minimal but valid
    "model": {
        "name": "alignn_atomwise",
        "alignn_layers": 4,
        "gcn_layers": 4,
        "embedding_features": 64,
        "hidden_features": 256,
        "output_features": 1,
        "classification": False,
        "link": "identity",
        "calculate_gradient": False,
    },
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


# ----------------------------- RUN ALIGNN ------------------------------------ #

def run_alignn(dataset_root: Path, n_train: int, n_test: int, output_dir: Path):
    """
    Run ALIGNN using a minimal, self-contained config.

    Assumes:
      - dataset_root/id_prop.csv exists with columns: jid,target
      - CIF files are in dataset_root with names matching 'jid'
    """
    cfg = json.loads(json.dumps(ALIGNN_CONFIG_TEMPLATE))  # deep copy

    # set split sizes explicitly
    cfg["n_train"] = int(n_train)
    cfg["n_val"] = 0
    cfg["n_test"] = int(n_test)

    # disable ratio-based splitting
    cfg["train_ratio"] = None
    cfg["val_ratio"] = None
    cfg["test_ratio"] = None

    # keep row order: first n_train → train, last n_test → test
    cfg["keep_data_order"] = True

    # make sure tags match our id_prop.csv
    cfg["id_tag"] = "jid"
    cfg["target"] = "target"

    cfg["output_dir"] = str(output_dir)

    config_path = dataset_root / "config_alignn.json"
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"[ALIGNN] Wrote config: {config_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "-m",
        "alignn.train",
        "--root_dir", str(dataset_root),
        "--config",   str(config_path),
        "--output_dir", str(output_dir),
    ]
    print(f"[ALIGNN] Running: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    print("\n[ALIGNN] STDOUT:")
    print(result.stdout)
    print("\n[ALIGNN] STDERR:")
    print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"ALIGNN failed with exit code {result.returncode}. "
            f"See STDOUT/STDERR above for details."
        )

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
        description="Run CGCNN / ALIGNN on JARVIS-style CSVs "
                    "(jarvis_id, cif_structure, many properties)."
    )
    parser.add_argument(
        "--backend",
        choices=["cgcnn", "alignn", "both"],
        required=True,
        help="Which GNN backend(s) to run.",
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
        alignn_dataset_root = work_dir / "alignn_dataset"
        n_train_alignn, n_test_alignn = prepare_alignn_dataset(
            train_csv=train_csv,
            test_csv=test_csv,
            target_col=target_col,
            out_root=alignn_dataset_root,
        )
        outdir = Path(args.alignn_outdir) if args.alignn_outdir else (work_dir / "alignn_runs")
        run_alignn(
            dataset_root=alignn_dataset_root,
            n_train=n_train_alignn,
            n_test=n_test_alignn,
            output_dir=outdir,
        )


if __name__ == "__main__":
    main()

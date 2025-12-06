#!/usr/bin/env python

"""
MatBench + JARVIS-DFT property prediction script.

- For MatBench:
    - Load official train/test splits
    - Featurize structures/compositions with simple physics-inspired stats
    - Train a suite of models
    - Compute MAE on the test set for each fold and each model

- For JARVIS-DFT 20-property task:
    - Load JARVIS dft_3d dataset
    - Select 20 target properties
    - Filter to entries where all 20 are present
    - Random 80/20 train/test split
    - Train the same model suite (multi-output regression)

Usage:
    python prediction.py                          # default: matbench_mp_gap
    python prediction.py --dataset matbench_mp_e_form
    python prediction.py -d all                   # run on all MatBench datasets
    python prediction.py -d jarvis_dft_20         # JARVIS-DFT 20-property task

Requires:
    pip install matbench scikit-learn pymatgen jarvis-tools
"""

import argparse

from matbench.bench import MatbenchBenchmark
from pymatgen.core import Composition

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.base import clone
from sklearn.model_selection import train_test_split


# -------------------------------------------------------------------
# 1. Utility: get official MatBench splits (robust to version weirdness)
# -------------------------------------------------------------------
def get_matbench_splits(dataset_name: str):
    """
    Returns a dict:
        {
          fold_name: {
            "X_train": pd.Series or DataFrame,
            "y_train": pd.Series,
            "X_test":  pd.Series or DataFrame,
            "y_test":  pd.Series
          },
          ...
        }

    using the *official* MatBench splits.
    """

    # Initialize benchmark with only the requested dataset
    mb = MatbenchBenchmark(autoload=False, subset=[dataset_name])
    print(f"\nInitialized benchmark '{mb.benchmark_name}' with tasks: {list(mb.tasks)}")

    # mb.tasks can be a dict or a dict_values, depending on matbench version
    tasks_obj = mb.tasks

    if isinstance(tasks_obj, dict):
        # Newer matbench: tasks is a dict {name -> MatbenchTask}
        if dataset_name in tasks_obj:
            task = tasks_obj[dataset_name]
        else:
            # Fallback: just take the first/only task
            task = list(tasks_obj.values())[0]
    else:
        # Older matbench: tasks is dict_values([MatbenchTask, ...])
        task = list(tasks_obj)[0]

    # Load the dataset (downloads on first run)
    print(f"Loading dataset '{dataset_name}'...")
    task.load()
    print(f"Dataset '{task.dataset_name}' loaded.")
    print(f"Available folds: {task.folds}")

    splits = {}

    for fold in task.folds:
        print("=" * 80)
        print(f"Processing fold: {fold}")

        # Official train (+val) split
        X_train, y_train = task.get_train_and_val_data(fold)

        # Get test data *with* targets first; behavior differs by version
        test_with_targets = task.get_test_data(fold, include_target=True)

        # Case 1: older matbench – returns (X_test, y_test) tuple
        if isinstance(test_with_targets, tuple):
            X_test, y_test = test_with_targets

        # Case 2: newer matbench – returns DataFrame with target column
        else:
            if isinstance(test_with_targets, pd.DataFrame):
                y_test = test_with_targets.iloc[:, -1]
                # Inputs without target
                X_test = task.get_test_data(fold, include_target=False)
            else:
                raise TypeError(
                    f"Unexpected type returned by get_test_data(include_target=True): "
                    f"{type(test_with_targets)}"
                )

        print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

        splits[fold] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }

    return splits


# -------------------------------------------------------------------
# 1b. Utility: get JARVIS-DFT 20-property train/test split
# -------------------------------------------------------------------
def get_jarvis_dft_20props_splits(
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Build a single random train/test split for JARVIS-DFT dft_3d
    with 20 target properties.

    Returns a dict with a single fold:
        {
          "random_split": {
            "X_train": list of pymatgen Structures,
            "y_train": pd.DataFrame (N_train, 20),
            "X_test":  list of pymatgen Structures,
            "y_test":  pd.DataFrame (N_test, 20),
          }
        }
    """
    print("\nLoading JARVIS-DFT dft_3d dataset for 20-property task...")

    from jarvis.db.figshare import data as jarvis_data
    from jarvis.core.atoms import Atoms

    # Load full dft_3d metadata (list of dicts)
    dft_3d = jarvis_data(dataset="dft_3d")
    df = pd.DataFrame(dft_3d)
    print(f"Total JARVIS-DFT entries: {len(df)}")

    # Choose 20 properties (you can tweak this list if needed)
    target_cols = [
        "formation_energy_peratom",
        "optb88vdw_bandgap",
        "ehull",
        "bulk_modulus_kv",
        "shear_modulus_gv",
        "optb88vdw_total_energy",
        "mbj_bandgap",
        "slme",
        "spillage",
        "exfoliation_energy",
        "dfpt_piezo_max_eij",
        "dfpt_piezo_max_dij",
        "dfpt_piezo_max_dielectric",
        "dfpt_piezo_max_dielectric_electronic",
        "dfpt_piezo_max_dielectric_ionic",
        "max_ir_mode",
        "min_ir_mode",
        "n-Seebeck",
        "p-Seebeck",
        "magmom_oszicar",
    ]

    # Filter rows where all 20 properties are present
    available_cols = [c for c in target_cols if c in df.columns]
    if len(available_cols) < len(target_cols):
        missing = sorted(set(target_cols) - set(available_cols))
        raise KeyError(
            "Some target properties missing from JARVIS dft_3d dataframe: "
            + ", ".join(missing)
        )

    mask = df[target_cols].notnull().all(axis=1)
    df_clean = df[mask].reset_index(drop=True)

    print(f"Entries with all 20 properties present: {len(df_clean)}")

    # Convert JARVIS Atoms dicts to pymatgen Structures for featurizer
    structures = []
    for a_dict in df_clean["atoms"]:
        at = Atoms.from_dict(a_dict)
        # use pymatgen structure so featurize_structures_simple can use .composition
        structures.append(at.pymatgen_structure)

    y = df_clean[target_cols].copy()

    # Single random train/test split
    idx = np.arange(len(df_clean))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=random_state
    )

    print(
        f"JARVIS-DFT random split: Train size = {len(train_idx)}, "
        f"Test size = {len(test_idx)}"
    )

    X_train = [structures[i] for i in train_idx]
    X_test = [structures[i] for i in test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    splits = {
        "random_split": {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }
    }

    return splits, target_cols


# -------------------------------------------------------------------
# 2. Featurization: structures/compositions → simple numeric features
# -------------------------------------------------------------------
def featurize_structures_simple(X):
    """
    Very simple, robust features from composition:

    For each structure/composition, compute:
        - n_atoms: total number of atoms
        - mean_Z: mean atomic number
        - mean_atomic_mass: mean atomic mass (amu)

    X: iterable of objects, each either:
        - a pymatgen Structure with .composition
        - a pymatgen Composition
        - a string formula

    Returns: np.ndarray of shape (n_samples, 3)
    """
    features = []

    for x in X:
        # Get a Composition object
        if hasattr(x, "composition"):  # Structure or similar
            comp = x.composition
        else:
            comp = Composition(str(x))

        # Total atom count
        total_atoms = comp.num_atoms

        # Aggregate Z and mass
        z_sum = 0.0
        mass_sum = 0.0
        atom_count = 0.0

        for el, amt in comp.items():
            # el is a pymatgen Element
            z_sum += el.Z * amt
            mass_sum += float(el.atomic_mass) * amt
            atom_count += amt

        mean_Z = z_sum / atom_count if atom_count > 0 else 0.0
        mean_mass = mass_sum / atom_count if atom_count > 0 else 0.0

        features.append([total_atoms, mean_Z, mean_mass])

    return np.array(features, dtype=float)


# -------------------------------------------------------------------
# 3. Model suite and evaluation
# -------------------------------------------------------------------
def run_model_suite_on_splits(splits, dataset_name: str, multi_target: bool = False):
    """
    Train a suite of models on each fold and report MAE on the test set.

    If multi_target=True, y is expected to be 2D (N, T) and we perform
    multi-output regression. mean_absolute_error will report the uniform
    average over all targets.
    """

    print("\n" + "#" * 80)
    print(f"Running model suite on dataset: {dataset_name}")
    print(f"Multi-target mode: {multi_target}")
    print("#" * 80)

    # Define model suite
    model_suite = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            random_state=0,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            random_state=0,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=300,
            random_state=0,
            n_jobs=-1,
        ),
    }

    # Keep track of MAE per model per fold
    mae_scores = {name: [] for name in model_suite}

    for fold_name, fold_data in splits.items():
        print("\n" + "#" * 80)
        print(f"Running models for fold: {fold_name}")

        X_train = fold_data["X_train"]
        y_train = fold_data["y_train"]
        X_test = fold_data["X_test"]
        y_test = fold_data["y_test"]

        # Ensure y are numpy arrays
        y_train_arr = np.array(y_train, dtype=float)
        y_test_arr = np.array(y_test, dtype=float)

        # ----- Featurize -----
        print("Featurizing train and test compositions (simple stats)...")
        X_train_feat = featurize_structures_simple(X_train)
        X_test_feat = featurize_structures_simple(X_test)

        # ----- Impute + scale -----
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        X_train_imp = imputer.fit_transform(X_train_feat)
        X_test_imp = imputer.transform(X_test_feat)

        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_test_scaled = scaler.transform(X_test_imp)

        # ----- Train & evaluate each model -----
        for model_name, base_model in model_suite.items():
            model = clone(base_model)

            print(f"\nTraining {model_name} ...")
            model.fit(X_train_scaled, y_train_arr)

            y_pred = model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test_arr, y_pred)

            mae_scores[model_name].append(mae)
            print(f"{model_name} | Fold {fold_name} MAE: {mae:.5f}")

    # ----- Summary -----
    print("\n" + "=" * 80)
    print(f"Summary: MAE across folds for dataset: {dataset_name}")
    print("=" * 80)
    for model_name, scores in mae_scores.items():
        scores = np.array(scores, dtype=float)
        print(
            f"{model_name:20s}  "
            f"Mean MAE = {scores.mean():.5f}  |  Std = {scores.std():.5f}  "
            f"| Folds: {len(scores)}"
        )


# -------------------------------------------------------------------
# 4. Main
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MatBench / JARVIS-DFT property prediction benchmark")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="matbench_mp_gap",
        help=(
            "Dataset name:\n"
            "  - MatBench dataset (e.g. 'matbench_mp_gap', 'matbench_mp_e_form', ...),\n"
            "  - 'all' to run on all MatBench datasets,\n"
            "  - 'jarvis_dft_20' for JARVIS-DFT 20-property multi-task dataset."
        ),
    )
    args = parser.parse_args()
    dataset_arg = args.dataset

    if dataset_arg.lower() == "all":
        # Discover all available datasets from MatbenchBenchmark
        mb_all = MatbenchBenchmark(autoload=False)

        # mb_all.tasks can be dict or dict_values of MatbenchTask
        tasks_obj = mb_all.tasks
        if isinstance(tasks_obj, dict):
            all_dataset_names = list(tasks_obj.keys())
        else:
            # dict_values([MatbenchTask, ...]) in your version
            all_dataset_names = [t.dataset_name for t in tasks_obj]

        print("\nRunning on ALL MatBench datasets:")
        for name in all_dataset_names:
            print(f"  - {name}")

        for name in all_dataset_names:
            splits = get_matbench_splits(name)
            run_model_suite_on_splits(splits, dataset_name=name, multi_target=False)

    elif dataset_arg.lower() == "jarvis_dft_20":
        # JARVIS-DFT 20-property multi-task dataset
        splits, target_cols = get_jarvis_dft_20props_splits()
        print("\nJARVIS-DFT 20 target properties:")
        for t in target_cols:
            print(f"  - {t}")
        run_model_suite_on_splits(
            splits,
            dataset_name="jarvis_dft_20",
            multi_target=True,
        )

    else:
        # Single MatBench dataset
        dataset_name = dataset_arg
        splits = get_matbench_splits(dataset_name)
        run_model_suite_on_splits(splits, dataset_name=dataset_name, multi_target=False)


if __name__ == "__main__":
    main()
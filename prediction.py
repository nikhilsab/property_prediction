#!/usr/bin/env python

"""
MatBench mp_gap:
- Load official train/test splits
- Featurize structures via composition (Magpie features)
- Train a suite of models
- Compute MAE on the test set for each fold and each model

Requires:
    pip install matbench matminer scikit-learn
"""

from matbench.bench import MatbenchBenchmark
from matminer.featurizers.composition import ElementProperty

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.base import clone


# -------------------------------------------------------------------
# 1. Utility: get official MatBench splits (robust to version weirdness)
# -------------------------------------------------------------------
def get_matbench_splits(dataset_name="matbench_mp_gap"):
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
    print(f"Initialized benchmark '{mb.benchmark_name}' with tasks: {list(mb.tasks)}")

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
# 2. Featurization: structures/compositions → Magpie features
# -------------------------------------------------------------------
def make_featurizer():
    """Create a composition featurizer (Magpie preset)."""
    featurizer = ElementProperty.from_preset("magpie")
    return featurizer


def featurize_series_to_magpie(X, featurizer):
    """
    X: iterable of inputs (pymatgen Structures, Composition objects, or strings)
    Returns: np.ndarray of shape (n_samples, n_features)
    """
    comp_strings = []
    for x in X:
        # For structure objects with .composition attribute
        if hasattr(x, "composition"):
            comp = x.composition.reduced_formula
        else:
            comp = str(x)
        comp_strings.append(comp)

    # featurize_many returns a list-of-lists (or np.ndarray-like)
    X_feat = featurizer.featurize_many(comp_strings, ignore_errors=True)
    return np.array(X_feat, dtype=float)


# -------------------------------------------------------------------
# 3. Model suite and evaluation
# -------------------------------------------------------------------
def run_model_suite_on_splits(splits):
    """
    Train a suite of models on each fold and report MAE on the test set.
    """

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

    # One featurizer reused across folds
    featurizer = make_featurizer()

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
        print("Featurizing train and test compositions...")
        X_train_feat = featurize_series_to_magpie(X_train, featurizer)
        X_test_feat = featurize_series_to_magpie(X_test, featurizer)

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
    print("Summary: MAE across folds")
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
    dataset_name = "matbench_mp_gap"  # change to any MatBench dataset name if needed

    # 1) Get official splits
    splits = get_matbench_splits(dataset_name)

    # 2) Train model suite and compute MAE
    run_model_suite_on_splits(splits)


if __name__ == "__main__":
    main()

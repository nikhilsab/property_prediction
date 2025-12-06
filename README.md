# Property Prediction

A Python script for materials property prediction using MatBench datasets and JARVIS-DFT data. This tool benchmarks multiple machine learning models on materials science datasets using simple physics-inspired features.

## Overview

This project provides a benchmarking framework for predicting materials properties using:
- **MatBench datasets**: Official train/test splits from the MatBench benchmark suite
- **JARVIS-DFT 20-property task**: Multi-target regression on 20 properties from the JARVIS-DFT database

The script uses simple composition-based features and evaluates multiple regression models to predict materials properties.

## Features

- Support for multiple MatBench datasets (e.g., `matbench_mp_gap`, `matbench_mp_e_form`)
- JARVIS-DFT 20-property multi-task regression
- Simple physics-inspired featurization (atomic number, atomic mass, atom count)
- Multiple model evaluation (Linear Regression, Random Forest, Gradient Boosting, Extra Trees)
- Automatic handling of official MatBench train/test splits
- Comprehensive MAE (Mean Absolute Error) reporting across folds

## Requirements

Install the required dependencies:

```bash
pip install matbench scikit-learn pymatgen jarvis-tools numpy pandas
```

## Usage

### Single MatBench Dataset

Run on a specific MatBench dataset (default: `matbench_mp_gap`):

```bash
python prediction.py
```

Or specify a different dataset:

```bash
python prediction.py --dataset matbench_mp_e_form
```

### All MatBench Datasets

Run on all available MatBench datasets:

```bash
python prediction.py -d all
```

### JARVIS-DFT 20-Property Task

Run on the JARVIS-DFT 20-property multi-task dataset:

```bash
python prediction.py -d jarvis_dft_20
```

## How It Works

1. **Data Loading**: 
   - For MatBench: Loads official train/test splits with multiple folds
   - For JARVIS-DFT: Creates a random 80/20 train/test split on entries with all 20 properties present

2. **Featurization**: 
   - Extracts simple features from compositions/structures:
     - Total number of atoms
     - Mean atomic number
     - Mean atomic mass

3. **Model Training**: 
   - Trains a suite of models:
     - Linear Regression
     - Random Forest (300 estimators)
     - Gradient Boosting
     - Extra Trees (300 estimators)

4. **Evaluation**: 
   - Computes Mean Absolute Error (MAE) on test sets
   - Reports mean and standard deviation across folds

## Output

The script prints:
- Dataset loading information
- Fold-by-fold processing details
- Per-model MAE scores for each fold
- Summary statistics (mean MAE, std, number of folds) for each model

## Supported MatBench Datasets

The script supports all MatBench datasets, including:
- `matbench_mp_gap` (default)
- `matbench_mp_e_form`
- And all other datasets in the MatBench benchmark suite

## JARVIS-DFT Properties

The 20-property task includes:
- Formation energy per atom
- Band gaps (optb88vdw, mbj)
- Elastic properties (bulk modulus, shear modulus)
- Electronic properties (Seebeck coefficients, SLME, spillage)
- Piezoelectric properties
- And more...

## Notes

- The featurization is intentionally simple and physics-inspired
- Models use default hyperparameters (except for tree-based models which use 300 estimators)
- For multi-target tasks (JARVIS-DFT), the MAE is averaged uniformly across all targets
- The script handles different versions of the matbench library automatically


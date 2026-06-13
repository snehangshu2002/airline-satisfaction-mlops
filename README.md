# Airline Passenger Satisfaction MLOps Pipeline

[![CI — Lint and Test](https://github.com/snehangshu2002/airline-satisfaction-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/snehangshu2002/airline-satisfaction-mlops/actions/workflows/ci.yml)

An end-to-end machine learning pipeline for predicting airline passenger satisfaction.
The project uses **DVC** for data and pipeline versioning, **DVCLive** for experiment tracking,
and **XGBoost** for model training.

## Overview

The pipeline ingests the raw airline satisfaction dataset, cleans and splits the data,
performs feature engineering, trains a binary classifier, and evaluates the final model.
All stages are reproducible through `dvc repro`.

## Pipeline Stages

1. **Data ingestion**
   - Loads `data/airline_satisfaction.csv`
   - Converts and cleans selected columns
   - Splits the dataset into train and test sets

2. **Feature engineering**
   - One-hot encodes categorical features
   - Label-encodes the target column
   - Standardizes numeric features
   - Saves processed train/test data and preprocessing artifacts

3. **Model training**
   - Trains an `XGBClassifier`
   - Logs hyperparameters and training metadata with DVCLive
   - Saves the trained model artifact

4. **Model evaluation**
   - Loads the trained model and processed test data
   - Computes accuracy, precision, recall, and AUC
   - Logs metrics and plots through DVCLive
   - Writes final evaluation metrics to `reports/metrics.json`

## Project Structure

```text
.
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── pyproject.toml
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── feature_engineering.py
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── exception.py
│   ├── logger_config.py
│   └── utils.py
├── data/
│   ├── airline_satisfaction.csv
│   ├── raw/
│   └── processed/
├── models/
├── reports/
├── dvclive/
└── logs/
```

## Requirements

- Python 3.12+
- `uv`
- `dvc`

Install dependencies with:

```powershell
uv sync
```

## Dataset

Place the raw dataset at:

```text
data/airline_satisfaction.csv
```

The pipeline reads this file as the source dataset. Generated files such as
`data/raw/train.csv`, `data/raw/test.csv`, and `data/processed/*.csv` are created by DVC stages.

## Run the Pipeline

Run the full pipeline:

```powershell
dvc repro
```

Inspect the pipeline graph:

```powershell
dvc dag
```

Check stage status:

```powershell
dvc status
```

## Outputs

After a successful run, the main outputs are:

- `data/raw/train.csv`
- `data/raw/test.csv`
- `data/processed/train.csv`
- `data/processed/test.csv`
- `models/encoder.pkl`
- `models/scaler.pkl`
- `models/label_encoder.pkl`
- `models/rating_medians.pkl`
- `models/xgboost.pkl`
- `reports/metrics.json`
- `dvclive/metrics.json`
- `dvclive/plots/`

## Configuration

Project parameters are defined in `params.yaml`:

- data paths and split settings
- target column name
- categorical and rating feature lists
- XGBoost hyperparameters
- artifact output paths

## Experiment Tracking

This project uses **DVCLive** in the training and evaluation stages to record:

- hyperparameters
- evaluation metrics
- confusion matrix and ROC plots

The corresponding files are stored in `dvclive/` and tracked by DVC as part of the pipeline.

## Notes

- The repository keeps generated artifacts out of source control through `.gitignore`.
- `dvc.yaml` and `dvc.lock` define the reproducible pipeline state.
- If you change `params.yaml` or any stage code, rerun `dvc repro` to refresh outputs.

## License

No license has been specified yet.

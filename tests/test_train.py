# test_train.py
#
# Tests for src/train.py
# Covers: load_data, train_model, save_model
#
# HOW TO RUN:
#   pytest tests/test_train.py -v
#
# NOTE: We use small dummy numpy arrays for training in these tests
# (not the real airline dataset). Tiny arrays keep tests fast.

import os

import joblib
import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from src.components.train import load_data, save_model, train_model

# ─── Minimal XGBoost hyperparameters for fast tests ───────────────────────────
# We use 5 trees and depth=2 so training completes in milliseconds.
# In production params.yaml uses much larger values.

FAST_XGB_PARAMS = {
    "n_estimators": 5,
    "learning_rate": 0.1,
    "max_depth": 2,
    "colsample_bytree": 1.0,
    "subsample": 1.0,
    "random_state": 42,
    "enable_categorical": False,
    "eval_metric": "logloss",
}


# ─── Minimal training data ─────────────────────────────────────────────────────
# 8 rows, 3 features, binary labels — just enough for XGBoost not to complain.

@pytest.fixture
def small_X():
    return np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0],
        [8.0, 9.0, 1.0],
        [3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0],
    ])

@pytest.fixture
def small_y():
    return np.array([0, 1, 0, 1, 0, 1, 0, 1])


# ═══════════════════════════════════════════════════════════════════════════════
# TestLoadData
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadData:
    """
    Tests for load_data(file_path).
    Same logic as data_ingestion.load_data — reads a CSV and returns a DataFrame.
    """

    def test_returns_dataframe_for_valid_csv(self, tmp_path):
        # WHAT: A valid CSV should always come back as a DataFrame.
        csv_file = tmp_path / "processed.csv"
        csv_file.write_text("f1,f2,target\n1.0,2.0,0\n3.0,4.0,1\n")

        result = load_data(str(csv_file))

        assert isinstance(result, pd.DataFrame)

    def test_correct_number_of_columns(self, tmp_path):
        # WHAT: The DataFrame should have as many columns as the CSV header.
        # WHY: If columns are missing, training will crash with a shape error.
        csv_file = tmp_path / "processed.csv"
        csv_file.write_text("f1,f2,target\n1.0,2.0,0\n")

        result = load_data(str(csv_file))

        assert len(result.columns) == 3

    def test_correct_number_of_rows(self, tmp_path):
        # WHAT: Row count should exactly match the CSV content.
        csv_file = tmp_path / "processed.csv"
        csv_file.write_text("f1,f2\n1,2\n3,4\n5,6\n")

        result = load_data(str(csv_file))

        assert len(result) == 3

    def test_raises_exception_on_missing_file(self):
        # WHAT: A missing file should raise an exception, not return None silently.
        with pytest.raises(Exception):
            load_data("/nonexistent/path/processed_train.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# TestTrainModel
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrainModel:
    """
    Tests for train_model(X_train, y_train, params).
    This function wraps XGBClassifier — initializes it with params, fits, and returns it.
    """

    def test_returns_xgbclassifier_instance(self, small_X, small_y):
        # WHAT: The function must return an XGBClassifier object.
        # WHY: save_model and mlflow.xgboost.log_model both expect an XGBClassifier.
        model = train_model(small_X, small_y, FAST_XGB_PARAMS)

        assert isinstance(model, XGBClassifier)

    def test_trained_model_can_predict(self, small_X, small_y):
        # WHAT: The returned model should be fitted — calling predict() should not crash.
        # WHY: An unfitted model would raise sklearn's NotFittedError at evaluation time.
        model = train_model(small_X, small_y, FAST_XGB_PARAMS)
        predictions = model.predict(small_X)

        assert len(predictions) == len(small_X)

    def test_predictions_are_binary(self, small_X, small_y):
        # WHAT: For a binary classification model, all predictions should be 0 or 1.
        # WHY: Any other value would mean the model is producing garbage outputs.
        model = train_model(small_X, small_y, FAST_XGB_PARAMS)
        predictions = model.predict(small_X)

        assert set(predictions).issubset({0, 1})

    def test_predict_proba_has_two_columns(self, small_X, small_y):
        # WHAT: predict_proba should return a (n_samples, 2) array for binary classification.
        # WHY: evaluate.py does predict_proba(X)[:, 1] — needs exactly 2 columns.
        model = train_model(small_X, small_y, FAST_XGB_PARAMS)
        proba = model.predict_proba(small_X)

        assert proba.shape == (len(small_X), 2)

    def test_raises_value_error_on_shape_mismatch(self):
        # WHAT: If X_train has 3 rows but y_train has 2 labels, a ValueError should be raised.
        # WHY: Mismatched shapes silently corrupt training if not caught early.
        X = np.array([[1, 2], [3, 4], [5, 6]])  # 3 rows
        y = np.array([0, 1])                     # 2 labels — intentional mismatch

        with pytest.raises(ValueError):
            train_model(X, y, FAST_XGB_PARAMS)


# ═══════════════════════════════════════════════════════════════════════════════
# TestSaveModel
# ═══════════════════════════════════════════════════════════════════════════════

class TestSaveModel:
    """
    Tests for save_model(model, file_path).
    Saves a trained model to disk using joblib.
    """

    def test_model_file_is_created(self, small_X, small_y, tmp_path):
        # WHAT: After calling save_model, a file should exist at the given path.
        # WHY: If the file isn't created, evaluate.py's joblib.load() will crash.
        model = train_model(small_X, small_y, FAST_XGB_PARAMS)
        save_path = str(tmp_path / "models" / "xgb_model.pkl")

        save_model(model, save_path)

        assert os.path.exists(save_path)

    def test_saved_model_is_loadable_with_joblib(self, small_X, small_y, tmp_path):
        # WHAT: The saved file should be loadable by joblib without errors.
        # WHY: evaluate.py does `joblib.load(model_path)` — if the file is corrupted
        #      or in the wrong format, evaluation will fail.
        model = train_model(small_X, small_y, FAST_XGB_PARAMS)
        save_path = str(tmp_path / "models" / "xgb_model.pkl")

        save_model(model, save_path)
        loaded_model = joblib.load(save_path)

        assert isinstance(loaded_model, XGBClassifier)

    def test_loaded_model_produces_same_predictions(self, small_X, small_y, tmp_path):
        # WHAT: The model loaded from disk should give identical predictions to the original.
        # WHY: Serialization should be lossless — different predictions after reload
        #      would mean the weights weren't saved correctly.
        model = train_model(small_X, small_y, FAST_XGB_PARAMS)
        save_path = str(tmp_path / "models" / "xgb_model.pkl")

        save_model(model, save_path)
        loaded_model = joblib.load(save_path)

        original_preds = model.predict(small_X)
        loaded_preds = loaded_model.predict(small_X)

        np.testing.assert_array_equal(original_preds, loaded_preds)

    def test_parent_directories_are_created_automatically(self, small_X, small_y, tmp_path):
        # WHAT: If the parent folder doesn't exist yet, save_model should create it.
        # WHY: In a fresh clone, the "models/" directory won't exist — we shouldn't
        #      have to create it manually before training.
        model = train_model(small_X, small_y, FAST_XGB_PARAMS)
        # Use a deeply nested path that doesn't exist yet
        save_path = str(tmp_path / "deep" / "nested" / "dir" / "model.pkl")

        save_model(model, save_path)

        assert os.path.exists(save_path)

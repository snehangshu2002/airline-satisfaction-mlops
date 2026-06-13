# test_evaluate.py
#
# Tests for src/evaluate.py
# Covers: evaluate_model, save_metrics, save_confusion_matrix_plot,
#         save_roc_curve_plot, save_feature_importance_plot
#
# HOW TO RUN:
#   pytest tests/test_evaluate.py -v
#
# NOTE ABOUT THE MOCK MODEL:
#   conftest.py has a DummyModel but its predict_proba() returns a plain Python
#   list like [[0.2, 0.8], ...]. evaluate_model() does predict_proba(X)[:, 1]
#   which requires a numpy array (plain lists don't support [:, 1] indexing).
#   So we define our own mock_model fixture here using MagicMock with proper
#   numpy return values. It overrides the conftest DummyModel just for this file.

import json
import os

import matplotlib

matplotlib.use('Agg')
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.components.evaluate import (
    evaluate_model,
    save_confusion_matrix_plot,
    save_feature_importance_plot,
    save_metrics,
    save_roc_curve_plot,
)

# ─── Local fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def mock_model():
    """
    A MagicMock model that returns proper numpy arrays.
    - predict()       → array([0, 1])          — one prediction per sample
    - predict_proba() → 2D numpy array         — needed for AUC + ROC curve
    - feature_importances_ → 1D numpy array    — needed for feature importance plot
    """
    model = MagicMock()
    model.predict.return_value = np.array([0, 1])
    model.predict_proba.return_value = np.array([
        [0.8, 0.2],   # sample 1: 80% chance of class 0
        [0.3, 0.7],   # sample 2: 70% chance of class 1
    ])
    model.feature_importances_ = np.array([0.2, 0.5, 0.3])
    return model


@pytest.fixture
def sample_X():
    """
    A minimal 2-row feature DataFrame.
    Values don't matter here since the model is mocked — we just need the right shape.
    """
    return pd.DataFrame({
        "f1": [1.0, 2.0],
        "f2": [3.0, 4.0],
        "f3": [5.0, 6.0],
    })


@pytest.fixture
def sample_y():
    """Binary ground-truth labels matching the 2 rows in sample_X."""
    return np.array([0, 1])


# ═══════════════════════════════════════════════════════════════════════════════
# TestEvaluateModel
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvaluateModel:
    """
    Tests for evaluate_model(model, X_test, y_test).

    Returns a tuple of:
        (metrics_dict, y_pred, y_pred_prob)

    metrics_dict keys: "accuracy", "precision", "recall", "auc"
    """

    def test_returns_a_tuple_of_three(self, mock_model, sample_X, sample_y):
        # WHAT: evaluate_model must return exactly 3 things.
        # WHY: main() unpacks it as: metrics, y_pred, y_pred_prob = evaluate_model(...)
        #      If there are more or fewer values, that line will raise a ValueError.
        result = evaluate_model(mock_model, sample_X, sample_y)

        assert len(result) == 3

    def test_metrics_dict_has_all_required_keys(self, mock_model, sample_X, sample_y):
        # WHAT: The metrics dict must contain all 4 expected metric names.
        # WHY: main() calls mlflow.log_metrics(metrics) and save_metrics(metrics, path).
        #      A missing key means a metric silently isn't tracked anywhere.
        metrics, _, _ = evaluate_model(mock_model, sample_X, sample_y)

        assert set(metrics.keys()) == {"accuracy", "precision", "recall", "auc"}

    def test_all_metric_values_are_between_0_and_1(self, mock_model, sample_X, sample_y):
        # WHAT: Accuracy, precision, recall, and AUC are all proportions → must be in [0, 1].
        # WHY: A value outside [0, 1] would indicate a calculation bug
        #      (e.g. wrong axis in predict_proba slicing).
        metrics, _, _ = evaluate_model(mock_model, sample_X, sample_y)

        for metric_name, value in metrics.items():
            assert 0.0 <= value <= 1.0, (
                f"Metric '{metric_name}' = {value} is outside the valid [0, 1] range"
            )

    def test_y_pred_length_matches_y_test(self, mock_model, sample_X, sample_y):
        # WHAT: y_pred should have one prediction per test sample.
        # WHY: A length mismatch would break the confusion matrix calculation.
        _, y_pred, _ = evaluate_model(mock_model, sample_X, sample_y)

        assert len(y_pred) == len(sample_y)

    def test_y_pred_prob_length_matches_y_test(self, mock_model, sample_X, sample_y):
        # WHAT: y_pred_prob should also have one probability per test sample.
        # WHY: roc_auc_score() uses this — wrong length raises a ValueError.
        _, _, y_pred_prob = evaluate_model(mock_model, sample_X, sample_y)

        assert len(y_pred_prob) == len(sample_y)


# ═══════════════════════════════════════════════════════════════════════════════
# TestSaveMetrics
# ═══════════════════════════════════════════════════════════════════════════════

class TestSaveMetrics:
    """
    Tests for save_metrics(metrics, file_path).
    Saves a Python dict as a pretty-printed JSON file.
    """

    def test_creates_json_file_at_given_path(self, tmp_path):
        # WHAT: After calling save_metrics, a .json file must exist at the path.
        # WHY: mlflow.log_artifact() will fail if the file doesn't exist.
        file_path = str(tmp_path / "reports" / "metrics.json")
        metrics = {"accuracy": 0.95, "auc": 0.98}

        save_metrics(metrics, file_path)

        assert os.path.exists(file_path)

    def test_saved_json_contains_all_metric_keys(self, tmp_path):
        # WHAT: The saved JSON should have all the same keys as the input dict.
        # WHY: Missing keys mean metrics were silently dropped during serialization.
        file_path = str(tmp_path / "reports" / "metrics.json")
        metrics = {"accuracy": 0.95, "precision": 0.93, "recall": 0.91, "auc": 0.98}

        save_metrics(metrics, file_path)

        with open(file_path) as f:
            loaded = json.load(f)

        assert set(loaded.keys()) == set(metrics.keys())

    def test_saved_json_values_match_input(self, tmp_path):
        # WHAT: The saved values should exactly match the input (no rounding, no truncation).
        # WHY: Subtle float conversion issues could make the stored metrics slightly wrong.
        file_path = str(tmp_path / "reports" / "metrics.json")
        metrics = {"accuracy": 0.9567, "auc": 0.9812}

        save_metrics(metrics, file_path)

        with open(file_path) as f:
            loaded = json.load(f)

        assert loaded == metrics

    def test_creates_parent_directory_if_missing(self, tmp_path):
        # WHAT: If the parent directory doesn't exist, save_metrics should create it.
        # WHY: The "reports/" folder won't exist on a fresh checkout.
        file_path = str(tmp_path / "deep" / "nested" / "metrics.json")

        save_metrics({"accuracy": 0.9}, file_path)

        assert os.path.exists(file_path)


# ═══════════════════════════════════════════════════════════════════════════════
# TestSaveConfusionMatrixPlot
# ═══════════════════════════════════════════════════════════════════════════════

class TestSaveConfusionMatrixPlot:
    """
    Tests for save_confusion_matrix_plot(y_test, y_pred, save_path).
    Generates a confusion matrix PNG using matplotlib.
    """

    def test_creates_png_file(self, tmp_path):
        # WHAT: A .png file must exist at save_path after the function runs.
        # WHY: main() does mlflow.log_artifact(cm_plot_path) — file must exist first.
        save_path = str(tmp_path / "plots" / "confusion_matrix.png")
        y_test = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])   # one mismatch — realistic scenario

        save_confusion_matrix_plot(y_test, y_pred, save_path)

        assert os.path.exists(save_path)

    def test_returns_the_save_path(self, tmp_path):
        # WHAT: The function must return the same path it was given.
        # WHY: main() uses the returned path as the artifact path for MLflow.
        #      If None is returned, mlflow.log_artifact() gets None and crashes.
        save_path = str(tmp_path / "plots" / "confusion_matrix.png")
        y_test = np.array([0, 1])
        y_pred = np.array([0, 1])

        result = save_confusion_matrix_plot(y_test, y_pred, save_path)

        assert result == save_path

    def test_creates_parent_directory_if_missing(self, tmp_path):
        # WHAT: The "reports/plots/" directory is created automatically if absent.
        save_path = str(tmp_path / "new_dir" / "plots" / "confusion_matrix.png")
        y_test = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 1])

        save_confusion_matrix_plot(y_test, y_pred, save_path)

        assert os.path.exists(save_path)


# ═══════════════════════════════════════════════════════════════════════════════
# TestSaveRocCurvePlot
# ═══════════════════════════════════════════════════════════════════════════════

class TestSaveRocCurvePlot:
    """
    Tests for save_roc_curve_plot(model, X_test, y_test, save_path).
    Generates a ROC curve PNG using sklearn's RocCurveDisplay.from_estimator.

    WHY WE PATCH RocCurveDisplay:
        RocCurveDisplay.from_estimator() internally runs sklearn's check_is_fitted(),
        which expects a real sklearn-fitted estimator. Our MagicMock doesn't pass
        that check. We patch RocCurveDisplay entirely so we can test the file-creation
        logic without worrying about sklearn compatibility.
    """

    def test_creates_png_file(self, mock_model, sample_X, sample_y, tmp_path):
        # WHAT: After the call, a .png file must exist at save_path.
        save_path = str(tmp_path / "plots" / "roc_curve.png")

        with patch("src.components.evaluate.RocCurveDisplay") as mock_roc:
            save_roc_curve_plot(mock_model, sample_X, sample_y, save_path)

        assert os.path.exists(save_path)

    def test_returns_the_save_path(self, mock_model, sample_X, sample_y, tmp_path):
        # WHAT: The function must return the path of the saved file.
        save_path = str(tmp_path / "plots" / "roc_curve.png")

        with patch("src.components.evaluate.RocCurveDisplay"):
            result = save_roc_curve_plot(mock_model, sample_X, sample_y, save_path)

        assert result == save_path

    def test_creates_parent_directory_if_missing(self, mock_model, sample_X, sample_y, tmp_path):
        # WHAT: The function should create "reports/plots/" if it doesn't exist.
        save_path = str(tmp_path / "brand_new_dir" / "roc_curve.png")

        with patch("src.components.evaluate.RocCurveDisplay"):
            save_roc_curve_plot(mock_model, sample_X, sample_y, save_path)

        assert os.path.exists(save_path)


# ═══════════════════════════════════════════════════════════════════════════════
# TestSaveFeatureImportancePlot
# ═══════════════════════════════════════════════════════════════════════════════

class TestSaveFeatureImportancePlot:
    """
    Tests for save_feature_importance_plot(model, feature_names, save_path).
    Generates a bar chart of top-20 feature importances as a PNG.
    """

    def test_creates_png_file(self, mock_model, tmp_path):
        # WHAT: A .png file must be created at the given path.
        # WHY: mock_model.feature_importances_ = [0.2, 0.5, 0.3] — 3 features.
        #      feature_names must have exactly 3 entries to match.
        save_path = str(tmp_path / "plots" / "feature_importance.png")
        feature_names = ["f1", "f2", "f3"]

        save_feature_importance_plot(mock_model, feature_names, save_path)

        assert os.path.exists(save_path)

    def test_returns_the_save_path(self, mock_model, tmp_path):
        # WHAT: The return value should be the save path string.
        save_path = str(tmp_path / "plots" / "feature_importance.png")
        feature_names = ["f1", "f2", "f3"]

        result = save_feature_importance_plot(mock_model, feature_names, save_path)

        assert result == save_path

    def test_creates_parent_directory_if_missing(self, mock_model, tmp_path):
        # WHAT: The parent "plots/" folder is created if it doesn't exist.
        save_path = str(tmp_path / "auto_created_dir" / "feature_importance.png")
        feature_names = ["f1", "f2", "f3"]

        save_feature_importance_plot(mock_model, feature_names, save_path)

        assert os.path.exists(save_path)

# test_data_ingestion.py
#
# Tests for src/data_ingestion.py
# Covers: load_data, preprocess_data, save_data
#
# HOW TO RUN:
#   pytest tests/test_data_ingestion.py -v
#
# WHY we mock load_params:
#   preprocess_data() and save_data() call load_params() internally to read params.yaml.
#   In tests we don't want to depend on a real params.yaml file on disk, so we
#   "mock" (fake) that call and return our own simple dict instead.

import os
from unittest.mock import patch

import pandas as pd
import pytest

from src.components.data_ingestion import load_data, preprocess_data, save_data

# ─── Shared mock params ───────────────────────────────────────────────────────
# This is the minimal params dict that preprocess_data() needs.
MOCK_PARAMS_PREPROCESS = {
    "features": {
        "drop_cols": ["ID"],   # "ID" is the column we want dropped in tests
    }
}


# ═══════════════════════════════════════════════════════════════════════════════
# TestLoadData
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadData:
    """
    Tests for load_data(data_url).
    load_data reads a CSV from disk and returns a pandas DataFrame.
    """

    def test_returns_dataframe_for_valid_csv(self, tmp_path):
        # WHAT: When we give load_data a real CSV path, it should return a DataFrame.
        # WHY: The most basic check — wrong return type would break everything downstream.
        csv_file = tmp_path / "sample.csv"
        csv_file.write_text("col_a,col_b\n1,2\n3,4\n")

        result = load_data(str(csv_file))

        assert isinstance(result, pd.DataFrame)

    def test_loaded_dataframe_has_correct_row_count(self, tmp_path):
        # WHAT: The number of rows loaded should match the CSV (excluding header).
        # WHY: If rows are silently skipped or duplicated, downstream splits will be wrong.
        csv_file = tmp_path / "sample.csv"
        csv_file.write_text("col_a,col_b\n1,2\n3,4\n5,6\n")

        result = load_data(str(csv_file))

        assert len(result) == 3

    def test_loaded_dataframe_has_correct_columns(self, tmp_path):
        # WHAT: Column names should match the CSV header exactly.
        # WHY: Wrong column names would cause KeyErrors in every downstream step.
        csv_file = tmp_path / "sample.csv"
        csv_file.write_text("col_a,col_b\n1,2\n")

        result = load_data(str(csv_file))

        assert list(result.columns) == ["col_a", "col_b"]

    def test_raises_exception_on_nonexistent_path(self):
        # WHAT: If the file doesn't exist, load_data should raise an exception.
        # WHY: Silent failure here would cause confusing errors much later in the pipeline.
        with pytest.raises(Exception):
            load_data("/totally/fake/path/data.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# TestPreprocessData
# ═══════════════════════════════════════════════════════════════════════════════

class TestPreprocessData:
    """
    Tests for preprocess_data(df).
    This function:
      1. Converts "Flight Distance" to numeric (fixing string values)
      2. Drops columns listed in params["features"]["drop_cols"]
      3. Drops rows with any NaN values
    """

    def test_returns_a_dataframe(self, sample_raw_data):
        # WHAT: Output must always be a DataFrame.
        # WHY: If it accidentally returns None or something else, every downstream
        #      operation will crash with a confusing AttributeError.
        with patch("src.components.data_ingestion.load_params", return_value=MOCK_PARAMS_PREPROCESS):
            result = preprocess_data(sample_raw_data)

        assert isinstance(result, pd.DataFrame)

    def test_flight_distance_becomes_numeric(self, sample_raw_data):
        # WHAT: "Flight Distance" column has mixed types in raw data ("2000" as string).
        #       After preprocessing it must be a numeric dtype.
        # WHY: XGBoost and scalers require numeric inputs — strings would break training.
        with patch("src.components.data_ingestion.load_params", return_value=MOCK_PARAMS_PREPROCESS):
            result = preprocess_data(sample_raw_data)

        assert pd.api.types.is_numeric_dtype(result["Flight Distance"])

    def test_null_rows_are_removed(self, sample_raw_data):
        # WHAT: sample_raw_data has one row where Flight Distance is None.
        #       After preprocessing, zero NaN values should remain.
        # WHY: NaNs in features cause errors in sklearn transformers and XGBoost.
        with patch("src.components.data_ingestion.load_params", return_value=MOCK_PARAMS_PREPROCESS):
            result = preprocess_data(sample_raw_data)

        assert result.isnull().sum().sum() == 0

    def test_specified_columns_are_dropped(self, sample_raw_data):
        # WHAT: Columns listed in drop_cols (here: "ID") must not appear in output.
        # WHY: ID columns are meaningless features that would add noise to the model.
        with patch("src.components.data_ingestion.load_params", return_value=MOCK_PARAMS_PREPROCESS):
            result = preprocess_data(sample_raw_data)

        assert "ID" not in result.columns

    def test_row_count_reduced_after_null_removal(self, sample_raw_data):
        # WHAT: sample_raw_data has 4 rows; 1 has a null. After dropna(), 3 should remain.
        # WHY: Ensures rows with nulls are actually removed, not silently kept.
        with patch("src.components.data_ingestion.load_params", return_value=MOCK_PARAMS_PREPROCESS):
            result = preprocess_data(sample_raw_data)

        assert len(result) == 3


# ═══════════════════════════════════════════════════════════════════════════════
# TestSaveData
# ═══════════════════════════════════════════════════════════════════════════════

class TestSaveData:
    """
    Tests for save_data(train_data, test_data).
    This function saves both DataFrames as CSV files to the paths from params.yaml.
    """

    def _mock_params(self, train_path: str, test_path: str) -> dict:
        """Helper to build the params dict that save_data() reads."""
        return {
            "data": {
                "train_path": train_path,
                "test_path": test_path,
            }
        }

    def test_train_csv_file_is_created(self, sample_train_df, sample_test_df, tmp_path):
        # WHAT: A CSV file for training data should exist at the specified path after the call.
        # WHY: If no file is created, the next pipeline step (feature engineering) can't read it.
        train_path = str(tmp_path / "data" / "train.csv")
        test_path = str(tmp_path / "data" / "test.csv")

        with patch("src.components.data_ingestion.load_params", return_value=self._mock_params(train_path, test_path)):
            save_data(sample_train_df, sample_test_df)

        assert os.path.exists(train_path)

    def test_test_csv_file_is_created(self, sample_train_df, sample_test_df, tmp_path):
        # WHAT: A CSV file for test data should also exist at the specified path.
        train_path = str(tmp_path / "data" / "train.csv")
        test_path = str(tmp_path / "data" / "test.csv")

        with patch("src.components.data_ingestion.load_params", return_value=self._mock_params(train_path, test_path)):
            save_data(sample_train_df, sample_test_df)

        assert os.path.exists(test_path)

    def test_saved_train_has_same_shape_as_input(self, sample_train_df, sample_test_df, tmp_path):
        # WHAT: Reading back the saved train CSV should give the same shape as what we passed in.
        # WHY: Any row/column loss during save → reload would silently corrupt the pipeline.
        train_path = str(tmp_path / "data" / "train.csv")
        test_path = str(tmp_path / "data" / "test.csv")

        with patch("src.components.data_ingestion.load_params", return_value=self._mock_params(train_path, test_path)):
            save_data(sample_train_df, sample_test_df)

        saved_train = pd.read_csv(train_path)
        assert saved_train.shape == sample_train_df.shape

    def test_saved_test_has_same_shape_as_input(self, sample_train_df, sample_test_df, tmp_path):
        # WHAT: Same shape check for test data.
        train_path = str(tmp_path / "data" / "train.csv")
        test_path = str(tmp_path / "data" / "test.csv")

        with patch("src.components.data_ingestion.load_params", return_value=self._mock_params(train_path, test_path)):
            save_data(sample_train_df, sample_test_df)

        saved_test = pd.read_csv(test_path)
        assert saved_test.shape == sample_test_df.shape

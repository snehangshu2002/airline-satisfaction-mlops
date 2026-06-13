# test_feature_engineering.py
#
# Tests for src/feature_engineering.py
# Covers: feature_engineering()
#
# HOW TO RUN:
#   pytest tests/test_feature_engineering.py -v
#
# WHY we mock load_params:
#   feature_engineering() reads categorical_cols, rating_cols, and target_col from
#   params.yaml. We mock this so tests run without a real params.yaml file.

from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.components.feature_engineering import feature_engineering

# ─── Mock params that match the fixtures in conftest.py ───────────────────────
# These column names must exactly match the columns in sample_train_df / sample_test_df
# from conftest.py, otherwise feature_engineering() will raise a KeyError.

MOCK_PARAMS = {
    "features": {
        "categorical_cols": [
            "Gender",
            "Customer Type",
            "Type of Travel",
            "Class",
        ],
        "rating_cols": [
            "Departure and Arrival Time Convenience",
            "Ease of Online Booking",
            "Check-in Service",
            "Online Boarding",
            "Gate Location",
            "On-board Service",
            "Seat Comfort",
            "Leg Room Service",
            "Cleanliness",
            "Food and Drink",
            "In-flight Service",
            "In-flight Wifi Service",
            "In-flight Entertainment",
            "Baggage Handling",
        ],
        "target_col": "Satisfaction",
    }
}


# ═══════════════════════════════════════════════════════════════════════════════
# TestFeatureEngineering
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeatureEngineering:
    """
    Tests for feature_engineering(train_data, test_data).

    This function returns 8 values:
        X_train, y_train, X_test, y_test, encoder, scaler, label_encoder, rating_medians

    It applies:
        - OneHotEncoder on categorical columns (fit on train, transform on test)
        - LabelEncoder on the target column ("Satisfaction")
        - StandardScaler on all features
    """

    def test_returns_eight_values(self, sample_train_df, sample_test_df):
        # WHAT: The function should always return exactly 8 outputs.
        # WHY: If any output is accidentally dropped, the callers will break
        #      silently with unpacking errors.
        with patch("src.feature_engineering.load_params", return_value=MOCK_PARAMS):
            result = feature_engineering(sample_train_df, sample_test_df)

        assert len(result) == 8

    def test_X_train_is_a_dataframe(self, sample_train_df, sample_test_df):
        # WHAT: X_train (first output) must be a pandas DataFrame.
        # WHY: The main() function concatenates it with y_train to save as CSV —
        #      that only works if it's a DataFrame (not a bare numpy array).
        with patch("src.feature_engineering.load_params", return_value=MOCK_PARAMS):
            X_train, *_ = feature_engineering(sample_train_df, sample_test_df)

        assert isinstance(X_train, pd.DataFrame)

    def test_X_test_is_a_dataframe(self, sample_train_df, sample_test_df):
        # WHAT: X_test (third output) must also be a pandas DataFrame.
        with patch("src.feature_engineering.load_params", return_value=MOCK_PARAMS):
            _, _, X_test, *_ = feature_engineering(sample_train_df, sample_test_df)

        assert isinstance(X_test, pd.DataFrame)

    def test_y_train_is_numpy_array(self, sample_train_df, sample_test_df):
        # WHAT: y_train (second output) should be a numpy array.
        # WHY: LabelEncoder.fit_transform() returns a numpy array.
        #      XGBClassifier.fit() expects arrays, not pandas Series.
        with patch("src.feature_engineering.load_params", return_value=MOCK_PARAMS):
            _, y_train, *_ = feature_engineering(sample_train_df, sample_test_df)

        assert isinstance(y_train, np.ndarray)

    def test_y_values_are_binary_zero_or_one(self, sample_train_df, sample_test_df):
        # WHAT: After LabelEncoding "satisfied" / "neutral or dissatisfied",
        #       all y values should be either 0 or 1.
        # WHY: Binary classification — any other value would corrupt training.
        with patch("src.feature_engineering.load_params", return_value=MOCK_PARAMS):
            _, y_train, _, y_test, *_ = feature_engineering(sample_train_df, sample_test_df)

        assert set(y_train).issubset({0, 1}), f"y_train has unexpected values: {set(y_train)}"
        assert set(y_test).issubset({0, 1}), f"y_test has unexpected values: {set(y_test)}"

    def test_X_train_has_no_null_values(self, sample_train_df, sample_test_df):
        # WHAT: Processed X_train must contain zero NaN values.
        # WHY: XGBoost and StandardScaler will silently produce NaN outputs if any
        #      NaN slips through — hard to debug later.
        with patch("src.feature_engineering.load_params", return_value=MOCK_PARAMS):
            X_train, *_ = feature_engineering(sample_train_df, sample_test_df)

        assert X_train.isnull().sum().sum() == 0

    def test_X_test_has_no_null_values(self, sample_train_df, sample_test_df):
        # WHAT: Same null check for X_test.
        with patch("src.feature_engineering.load_params", return_value=MOCK_PARAMS):
            _, _, X_test, *_ = feature_engineering(sample_train_df, sample_test_df)

        assert X_test.isnull().sum().sum() == 0

    def test_original_categorical_columns_are_removed(self, sample_train_df, sample_test_df):
        # WHAT: OneHotEncoding replaces the original categorical columns with new binary columns.
        #       The originals ("Gender", "Class", etc.) should no longer be in X_train.
        # WHY: Leaving string columns in would crash StandardScaler downstream.
        with patch("src.feature_engineering.load_params", return_value=MOCK_PARAMS):
            X_train, *_ = feature_engineering(sample_train_df, sample_test_df)

        for col in MOCK_PARAMS["features"]["categorical_cols"]:
            assert col not in X_train.columns, f"Categorical column '{col}' was not removed"

    def test_X_train_and_X_test_have_identical_columns(self, sample_train_df, sample_test_df):
        # WHAT: Train and test must have the exact same columns in the same order.
        # WHY: If they differ (e.g. due to unseen categories), the trained model
        #      will receive wrong features at evaluation time.
        with patch("src.feature_engineering.load_params", return_value=MOCK_PARAMS):
            X_train, _, X_test, *_ = feature_engineering(sample_train_df, sample_test_df)

        assert list(X_train.columns) == list(X_test.columns)

    def test_encoder_is_onehot_encoder(self, sample_train_df, sample_test_df):
        # WHAT: The 5th returned object should be a fitted OneHotEncoder.
        # WHY: main() saves this with joblib — if the type is wrong, loading it
        #      during inference will fail.
        with patch("src.feature_engineering.load_params", return_value=MOCK_PARAMS):
            _, _, _, _, encoder, *_ = feature_engineering(sample_train_df, sample_test_df)

        assert isinstance(encoder, OneHotEncoder)

    def test_scaler_is_standard_scaler(self, sample_train_df, sample_test_df):
        # WHAT: The 6th returned object should be a fitted StandardScaler.
        with patch("src.feature_engineering.load_params", return_value=MOCK_PARAMS):
            _, _, _, _, _, scaler, *_ = feature_engineering(sample_train_df, sample_test_df)

        assert isinstance(scaler, StandardScaler)

    def test_rating_medians_is_a_dict(self, sample_train_df, sample_test_df):
        # WHAT: The 8th returned object (rating_medians) must be a dictionary.
        # WHY: It gets saved with joblib and used to fill 0s at inference time.
        with patch("src.feature_engineering.load_params", return_value=MOCK_PARAMS):
            *_, rating_medians = feature_engineering(sample_train_df, sample_test_df)

        assert isinstance(rating_medians, dict)

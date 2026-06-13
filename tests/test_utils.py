import pytest
import yaml
from unittest.mock import patch

from src.utils import load_params, setup_mlflow


class TestLoadParams:

    def test_returns_dict_for_valid_yaml(self, tmp_path):
        yaml_file = tmp_path / "params.yaml"
        yaml_file.write_text("model:\n  n_estimators: 100\n  max_depth: 6\n")
        result = load_params(str(yaml_file))

        assert isinstance(result, dict)

    def test_returns_correct_values(self, tmp_path):

        yaml_file = tmp_path / "params.yaml"
        yaml_file.write_text("model:\n  n_estimators: 100\n  max_depth: 6\n")

        result = load_params(str(yaml_file))

        assert result["model"]["n_estimator"] == 100
        assert result["model"]["max_depth"] == 6

    def test_raises_on_missing_file(self):

        with pytest.raises(Exception):
            load_params("/nonexistent/path/params.yaml")

    def test_raises_on_malformed_yaml(self, tmp_path):

        yaml_file = tmp_path / "params.yaml"
        yaml_file.write_text("model:\n  n_estimators: [\nbroken")

        with pytest.raises(Exception):
            load_params(str(yaml_file))


class TestSetupMlflow:

    def test_raises_when_tracking_uri_missing(self, monkeypatch):

        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "user")
        monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "pass")

        with pytest.raises(EnvironmentError, match="MLFLOW_TRACKING_URI"):
            setup_mlflow("test-experiment")

    def test_raises_when_username_missing(self, monkeypatch):
        # WHY: DagsHub auth requires both username and password.
        # Missing one means silent auth failure or wrong experiment owner.
        monkeypatch.setenv(
            "MLFLOW_TRACKING_URI", "https://dagshub.com/user/repo.mlflow"
        )
        monkeypatch.delenv("MLFLOW_TRACKING_USERNAME", raising=False)
        monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "pass")

        with pytest.raises(EnvironmentError):
            setup_mlflow("test-experiment")

    def test_raises_when_password_missing(self, monkeypatch):
        monkeypatch.setenv(
            "MLFLOW_TRACKING_URI", "https://dagshub.com/user/repo.mlflow"
        )
        monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "user")
        monkeypatch.delenv("MLFLOW_TRACKING_PASSWORD", raising=False)

        with pytest.raises(EnvironmentError):
            setup_mlflow("test-experiment")

    def test_calls_set_tracking_uri_with_correct_value(self, monkeypatch):
        # WHY: Even if no error is raised, the wrong URI could be passed.
        # This test verifies the actual value passed to mlflow.
        uri = "https://dagshub.com/snehangshu2002/airline-satisfaction-mlops.mlflow"
        monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
        monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "user")
        monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "pass")

        with (
            patch("src.utils.mlflow.set_tracking_uri") as mock_uri,
            patch("src.utils.mlflow.set_experiment"),
        ):
            setup_mlflow("test-experiment")

        mock_uri.assert_called_once_with(uri)

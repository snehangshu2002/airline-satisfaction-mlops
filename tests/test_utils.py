# tests/test_utils.py

from unittest.mock import patch

import pytest

from src.utils import load_params, setup_mlflow

# ── load_params ─────────────────────────────────────────────────────────────

class TestLoadParams:

    def test_returns_dict_for_valid_yaml(self, tmp_path):
        # WHY: Core contract. If it returns None or a string, everything downstream breaks.
        yaml_file = tmp_path / "params.yaml"
        yaml_file.write_text("model:\n  n_estimators: 100\n  max_depth: 6\n")

        result = load_params(str(yaml_file))

        assert isinstance(result, dict)

    def test_returns_correct_values(self, tmp_path):
        # WHY: Not just any dict — the RIGHT dict. Verifies YAML parsing is correct.
        yaml_file = tmp_path / "params.yaml"
        yaml_file.write_text("model:\n  n_estimators: 100\n  max_depth: 6\n")

        result = load_params(str(yaml_file))

        assert result["model"]["n_estimators"] == 100
        assert result["model"]["max_depth"] == 6

    def test_raises_on_missing_file(self):
        # WHY: The except block has `raise`. This test confirms it actually re-raises
        # instead of returning None silently. Silent failures are the worst kind.
        with pytest.raises(Exception):
            load_params("/nonexistent/path/params.yaml")

    def test_raises_on_malformed_yaml(self, tmp_path):
        # WHY: Malformed YAML is a real deployment bug — someone edits params.yaml
        # and breaks indentation. Must fail loudly, not return garbage.
        yaml_file = tmp_path / "params.yaml"
        yaml_file.write_text("model:\n  n_estimators: [\nbroken")

        with pytest.raises(Exception):
            load_params(str(yaml_file))


# ── setup_mlflow ─────────────────────────────────────────────────────────────

class TestSetupMlflow:

    def test_raises_when_tracking_uri_missing(self, monkeypatch):
        # WHY: Without a URI, mlflow logs to localhost silently.
        # That is a silent data loss bug — experiments disappear.
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "user")
        monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "pass")

        with pytest.raises(EnvironmentError, match="MLFLOW_TRACKING_URI"):
            setup_mlflow("test-experiment")

    def test_raises_when_username_missing(self, monkeypatch):
        # WHY: DagsHub auth requires both username and password.
        # Missing one means silent auth failure or wrong experiment owner.
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://dagshub.com/user/repo.mlflow")
        monkeypatch.delenv("MLFLOW_TRACKING_USERNAME", raising=False)
        monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "pass")

        with pytest.raises(EnvironmentError):
            setup_mlflow("test-experiment")

    def test_raises_when_password_missing(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://dagshub.com/user/repo.mlflow")
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

        with patch("src.utils.mlflow.set_tracking_uri") as mock_uri, \
             patch("src.utils.mlflow.set_experiment"):
            setup_mlflow("test-experiment")

        mock_uri.assert_called_once_with(uri)

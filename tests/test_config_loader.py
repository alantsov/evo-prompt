import pytest
from unittest.mock import patch, mock_open
from evo_prompt.config_loader import load_config

def test_load_config_success():
    """Test that load_config correctly parses a YAML string."""
    mock_yaml_content = """
    models:
      optimizer: "test-model"
    comfyui:
      api_url: "http://test:8188"
    """

    # We mock 'builtins.open' to pretend the file exists with our content
    # and 'yaml.safe_load' to return a dictionary
    with patch("builtins.open", mock_open(read_data=mock_yaml_content)):
        with patch("yaml.safe_load") as mock_yaml:
            mock_yaml.return_value = {
                "models": {"optimizer": "test-model"},
                "comfyui": {"api_url": "http://test:8188"}
            }

            config = load_config()
            assert config["models"]["optimizer"] == "test-model"
            assert config["comfyui"]["api_url"] == "http://test:8188"

def test_load_config_file_not_found():
    """Test behavior when the config file is missing."""
    with patch("builtins.open", side_effect=FileNotFoundError):
        with pytest.raises(FileNotFoundError):
            load_config()

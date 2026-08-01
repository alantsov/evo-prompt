import copy
import yaml
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.resolve()

_config = {}


def get_config(config_path: str | None = None):
    global _config
    if not _config:
        load_config(config_path)
    return copy.deepcopy(_config)


def load_config(config_path: str | None = None):
    """Load configuration from YAML file. Updates the global `config` dict in-place."""
    global _config
    if config_path is None:
        config_path = str(BASE_DIR / "config.yml")

    path = Path(config_path)
    if not path.is_absolute():
        path = BASE_DIR / path

    with open(path, "r", encoding="utf-8") as f:
        new_config = yaml.safe_load(f)

    _config.clear()
    _config.update(new_config)
    return _config

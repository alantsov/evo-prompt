import copy
import yaml
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.resolve()
_config: dict = {}

def get_config(infra_path: str | None = None, prompts_path: str | None = None) -> dict:
    """Return a deep copy of the configuration. Loads if not cached."""
    global _config
    if not _config:
        load_config(infra_path, prompts_path)
    return copy.deepcopy(_config)

def load_config(infra_path: str | None = None, prompts_path: str | None = None) -> dict:
    """Load configuration from two separate YAML files and merge them."""
    global _config
    _config.clear()

    # Load infrastructure config
    infra_file = _resolve_path(infra_path, "config/infra.yaml")
    if infra_file.exists():
        _config.update(yaml.safe_load(infra_file.read_text(encoding="utf-8")) or {})

    # Load prompts config
    prompts_file = _resolve_path(prompts_path, "config/prompts.yaml")
    if prompts_file.exists():
        _config.update(yaml.safe_load(prompts_file.read_text(encoding="utf-8")) or {})

    if not _config:
        raise ValueError(
            "Configuration is empty. Ensure infra.yaml and/or prompts.yaml exist, "
            "are valid YAML, and contain data."
        )

    return _config

def _resolve_path(path_str: str | None, default_rel: str) -> Path:
    """Resolve a configuration path relative to BASE_DIR if not absolute."""
    if path_str is None:
        return BASE_DIR / default_rel
    p = Path(path_str)
    return p if p.is_absolute() else BASE_DIR / p

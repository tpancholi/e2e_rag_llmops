import os
from pathlib import Path

import yaml


def _project_root() -> Path:
    """Returns the project root folder."""
    return Path(__file__).resolve().parents[1]  # go up two directory levels


def load_config(config_path: str | None = None) -> dict:
    """Resolves config's file path and loads it."""
    if config_path is None:
        env_path = os.getenv("CONFIG_PATH")
        config_path = env_path or str(_project_root() / "config" / "config.yaml")

    path = Path(config_path)
    # Resolve relative paths
    if not path.is_absolute():
        path = _project_root() / path
    path = path.resolve()

    if not path.is_file():
        msg = f"Config path is not a file: {path}"
        raise FileNotFoundError(msg)

    try:
        with path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            return config or {}
    except yaml.YAMLError as e:
        msg = f"Error parsing YAML config file {path}: {e}"
        raise yaml.YAMLError(msg) from e

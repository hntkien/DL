"""
Config loader for the conditional DDPM project.

Usage:
    from utils.config import load_config
    cfg = load_config("configs/default.yaml")
"""

import os
from pathlib import Path
import yaml


def load_config(path: str | os.PathLike) -> dict:
    """Load a YAML config file and return it as a nested dictionary.

    Args:
        path: Path to the YAML config file.

    Returns:
        Nested dict of config values.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)
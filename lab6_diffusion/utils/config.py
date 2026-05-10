"""
Config loader for the conditional DDPM project.

Usage:
    from utils.config import load_config
    cfg = load_config("configs/default.yaml")
"""

import os
from typing import Any
import yaml


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _try_numeric(value: Any) -> Any:
    """Attempt to coerce a string value to int or float.

    Leaves the value untouched if it is not a string or cannot be converted.

    Args:
        value (Any): Any Python object.

    Returns:
        Any: ``int`` or ``float`` if the string is a valid number, otherwise
            the original value unchanged.
    """
    if not isinstance(value, str):
        return value
    # Try int first (covers plain integers), then float (covers scientific notation).
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def _coerce_dict(d: Any) -> Any:
    """Recursively walk a nested dict/list and coerce numeric strings.

    Args:
        d (Any): Parsed YAML object (dict, list, or scalar).

    Returns:
        Any: The same structure with numeric strings replaced by numbers.
    """
    if isinstance(d, dict):
        return {k: _coerce_dict(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_coerce_dict(v) for v in d]
    return _try_numeric(d)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_config(path: str | os.PathLike) -> dict:
    """Load a YAML config file and return a fully typed Python dict.

    Uses ``yaml.safe_load`` (no arbitrary Python object construction) and then
    walks the result to convert any numeric strings — e.g. scientific-notation
    floats like ``"1.0e-4"`` that PyYAML may leave as strings — into their
    proper ``int`` / ``float`` types.

    Args:
        path (str | os.PathLike): Path to the ``.yaml`` config file.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        yaml.YAMLError: If the file is not valid YAML.

    Returns:
        dict: Parsed and type-coerced configuration dictionary.
    """
    path = os.fspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML mapping at the top level, got {type(raw).__name__}")

    return _coerce_dict(raw)
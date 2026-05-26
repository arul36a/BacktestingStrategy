"""
Shared utilities: config loading, logging, timezone helpers, and path resolution.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional

import yaml

_LOGGER = logging.getLogger(__name__)


def project_root(start: Optional[Path] = None) -> Path:
    """Resolve repository root (parent of `gold_ai_trading` package folder)."""
    if start is None:
        start = Path(__file__).resolve().parent.parent
    return start


def resolve_path(cfg_path: str, root: Optional[Path] = None) -> Path:
    """Turn config-relative paths into absolute Path objects."""
    p = Path(cfg_path).expanduser()
    if not p.is_absolute():
        root = root or project_root()
        return (root / p).resolve()
    return p.resolve()


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dict; raises with file path context on failure."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML not found: {path}")
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return dict(raw)


def deep_merge(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into a copy of `base`."""
    out: dict[str, Any] = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out


def load_app_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load `config/default.yaml` from repo root unless `config_path` is provided.
    Environment override: GOLD_AI_CONFIG=/abs/path/to.yaml
    """
    root = project_root()
    env_path = os.environ.get("GOLD_AI_CONFIG")
    path = Path(config_path or env_path or root / "config" / "default.yaml")
    if not path.exists():
        path = resolve_path(str(config_path)) if config_path else path
    return load_yaml(path)


def setup_logging(cfg: Mapping[str, Any]) -> logging.Logger:
    """Configure rotating-style console + optional file logging from config."""
    log_cfg = dict(cfg.get("logging", {}) or {})
    level_name = str(log_cfg.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler()]

    file_rel = log_cfg.get("file")
    if file_rel:
        log_path = resolve_path(str(file_rel))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
        handlers=handlers,
    )
    return logging.getLogger("gold_ai")


def ensure_dirs(paths: Mapping[str, Any]) -> None:
    """Create data/models/reports/logs directories from config.paths."""
    roots = paths.get("paths") or paths
    keys = ["data_dir", "models_dir", "reports_dir", "logs_dir"]
    base = project_root()
    for key in keys:
        rel = roots.get(key)
        if rel is None:
            continue
        resolve_path(str(rel), base).mkdir(parents=True, exist_ok=True)

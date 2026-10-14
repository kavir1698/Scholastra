"""Settings and configuration loaders (models.yaml, mcp_servers.json, presets)."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_RE = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")


class Settings(BaseSettings):
    """Runtime settings, overridable via ``SCHOLASTRA_*`` environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="SCHOLASTRA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model_strong: str = "openrouter/deepseek/deepseek-v3.2"
    model_cheap: str = "openrouter/deepseek/deepseek-chat"
    reasoning_effort: str = "high"
    allowed_providers: str = "openrouter,deepseek"
    default_provider: str = "openrouter"
    max_concurrency: int = 4
    workspace_root: str = "workspace"
    fake_llm: bool = False
    config_dir: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _treat_empty_as_unset(cls, data: Any) -> Any:
        # A blank value in .env (e.g. ``SCHOLASTRA_FAKE_LLM=`` as shipped in
        # .env.example) must fall back to the field default, not fail parsing.
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if v not in ("", None)}
        return data


def repo_root() -> Path:
    """Repository root (…/scholastra)."""
    return Path(__file__).resolve().parents[2]


def _expand_env(value: Any, env: Mapping[str, str] | None = None) -> Any:
    env = os.environ if env is None else env
    if isinstance(value, str):

        def repl(match: re.Match[str]) -> str:
            var, default = match.group(1), match.group(2)
            return env.get(var, default if default is not None else "")

        return _ENV_RE.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v, env) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v, env) for v in value]
    return value


def config_dir(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    if settings.config_dir:
        return Path(settings.config_dir).resolve()
    return repo_root() / "config"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=8)
def _load_yaml_raw(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_yaml(path: str) -> dict[str, Any]:
    return _expand_env(_load_yaml_raw(path))


def load_models_config(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    raw = _load_yaml_raw(str(config_dir(settings) / "models.yaml"))
    # Resolve tier placeholders against the Settings values (which also read .env),
    # so the library honours SCHOLASTRA_MODEL_* even when .env was never exported.
    env: dict[str, str] = {
        **os.environ,
        "SCHOLASTRA_MODEL_STRONG": settings.model_strong,
        "SCHOLASTRA_MODEL_CHEAP": settings.model_cheap,
        "SCHOLASTRA_REASONING_EFFORT": settings.reasoning_effort,
    }
    return _expand_env(raw, env)


def load_preset(name: str, settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    path = config_dir(settings) / "presets" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"unknown preset: {name} ({path})")
    return load_yaml(str(path))


def list_presets(settings: Settings | None = None) -> list[str]:
    settings = settings or get_settings()
    presets_dir = config_dir(settings) / "presets"
    return sorted(p.stem for p in presets_dir.glob("*.yaml"))


def load_source_policy(settings: Settings | None = None) -> dict[str, Any]:
    """Load the source-tier policy, falling back to safe defaults when absent."""
    settings = settings or get_settings()
    from scholastra.sources.policy import TIER_LABELS

    defaults: dict[str, Any] = {
        "tier_labels": {str(k): v for k, v in TIER_LABELS.items()},
        "allowed_tiers_for_claims": [1, 2, 3],
        "allow_orientation_support": False,
        "prefer_full_text": True,
        "require_corroborating": True,
    }
    path = config_dir(settings) / "source_policy.yaml"
    if not path.exists():
        return defaults
    raw = load_yaml(str(path)) or {}
    return {**defaults, **raw}


def load_mcp_config(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    path = config_dir(settings) / "mcp_servers.json"
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def csl_path(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    return config_dir(settings) / "csl" / "chicago-author-date.csl"

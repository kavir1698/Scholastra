"""LiteLLM-backed, provider-agnostic LLM factory with per-node routing.

Design goals:
- One ``structured`` entry point that returns a validated Pydantic object, with a
  bounded repair-retry loop (the schema-repair safety net for provider quirks).
- Per-node model/temperature/max_tokens from ``config/models.yaml``.
- Capability flags so a provider that rejects e.g. ``response_format`` degrades
  gracefully instead of failing the run.
- A pluggable :class:`LLMBackend` so tests (and ``--fake-llm``) can run with no
  network access.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from theo_explore.config import Settings, get_settings, load_models_config
from theo_explore.models.schemas import NodeUsage
from theo_explore.utils import extract_json

Message = dict[str, str]
TModel = TypeVar("TModel", bound=BaseModel)


class SchemaValidationError(RuntimeError):
    """Raised when a model response cannot be coerced into the target schema."""


class LLMProviderError(RuntimeError):
    """Raised for transient provider/transport failures (retried)."""


@dataclass
class RawCompletion:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    model: str = ""


@dataclass
class NodeConfig:
    model: str
    temperature: float = 0.3
    max_tokens: int = 4000
    supports_json_schema: bool = True
    supports_temperature: bool = True


@dataclass
class Usage:
    """Accumulates token/cost usage for a node call."""

    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    calls: int = 0
    model: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_node_usage(self) -> NodeUsage:
        return NodeUsage(
            calls=self.calls,
            tokens_in=self.tokens_in,
            tokens_out=self.tokens_out,
            cost_usd=self.cost_usd,
        )


class LLMBackend(Protocol):
    async def complete(
        self,
        *,
        node: str,
        model: str,
        messages: list[Message],
        temperature: float,
        max_tokens: int,
        json_object: bool,
        json_schema: dict[str, Any] | None,
    ) -> RawCompletion: ...


class LiteLLMBackend:
    """Concrete backend over ``litellm.acompletion`` with capability fallbacks."""

    def __init__(self, *, max_retries: int = 3) -> None:
        self._max_retries = max_retries
        # Suppress litellm's "Give Feedback" banner: our capability fallback
        # intentionally probes unsupported response_format shapes.
        try:
            import litellm

            litellm.suppress_debug_info = True
        except Exception:  # noqa: BLE001 - optional
            pass

    async def complete(
        self,
        *,
        node: str,
        model: str,
        messages: list[Message],
        temperature: float,
        max_tokens: int,
        json_object: bool,
        json_schema: dict[str, Any] | None,
    ) -> RawCompletion:
        import litellm

        base_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        attempts: list[dict[str, Any]] = []
        if json_schema is not None:
            attempts.append(
                {
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": json_schema.get("title", "response"),
                            "schema": json_schema,
                        },
                    }
                }
            )
        if json_object:
            attempts.append({"response_format": {"type": "json_object"}})
        attempts.append({})

        @retry(
            retry=retry_if_exception_type(LLMProviderError),
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=15),
            reraise=True,
        )
        async def _call(kwargs: dict[str, Any]) -> Any:
            try:
                return await litellm.acompletion(**kwargs)
            except Exception as exc:
                message = str(exc).lower()
                transient_tokens = (
                    "rate limit",
                    "timeout",
                    "temporarily",
                    "overloaded",
                    "429",
                    "503",
                )
                if any(token in message for token in transient_tokens):
                    raise LLMProviderError(str(exc)) from exc
                raise

        response: Any = None
        last_exc: Exception | None = None
        for extra in attempts:
            try:
                response = await _call({**base_kwargs, **extra})
                break
            except LLMProviderError:
                raise
            except Exception as exc:  # noqa: BLE001 - unsupported param -> try next shape
                last_exc = exc
                continue
        if response is None:
            raise LLMProviderError(f"all completion shapes failed: {last_exc}")

        text = _extract_text(response)
        tokens_in, tokens_out = _usage(response)
        cost = _cost(response, model)
        return RawCompletion(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
            model=model,
        )


def _extract_text(response: Any) -> str:
    try:
        message = response.choices[0].message
    except Exception:
        if isinstance(response, dict):
            message = response["choices"][0]["message"]
        else:
            raise
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if isinstance(content, list):  # multimodal content blocks
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content or ""


def _usage(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return 0, 0
    get = usage.get if isinstance(usage, dict) else lambda k, d=0: getattr(usage, k, d)
    return int(get("prompt_tokens", 0) or 0), int(get("completion_tokens", 0) or 0)


def _cost(response: Any, model: str) -> float:
    try:
        import litellm

        value = litellm.completion_cost(completion_response=response)
        return float(value or 0.0)
    except Exception:  # noqa: BLE001 - cost is best-effort
        return 0.0


class LLMFactory:
    """Routes nodes to models and enforces structured output."""

    def __init__(
        self,
        *,
        backend: LLMBackend | None = None,
        settings: Settings | None = None,
        models_config: dict[str, Any] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.models_config = models_config or load_models_config(self.settings)
        self.backend = backend or LiteLLMBackend()
        self.allowed_providers = {
            p.strip() for p in self.settings.allowed_providers.split(",") if p.strip()
        }
        self.default_provider = self.settings.default_provider
        # Bound total concurrent provider calls across all fan-out branches.
        self._semaphore = asyncio.Semaphore(max(1, int(self.settings.max_concurrency)))
        self._node_configs = self._build_node_configs()

    async def _complete(self, **kwargs: Any) -> RawCompletion:
        async with self._semaphore:
            return await self.backend.complete(**kwargs)

    def resolve_model(self, model: str) -> str:
        """Normalise a model id to a LiteLLM ``provider/model`` string.

        Ids copied straight from the OpenRouter UI (e.g.
        ``anthropic/claude-haiku-5.5`` or ``~deepseek/deepseek-flash-latest``)
        start with a vendor, not the LiteLLM ``openrouter`` provider, so they are
        routed through the default provider. Ids that already start with an
        allowed provider (``openrouter/...`` or ``deepseek/...``) are kept.
        """
        provider = model.split("/", 1)[0] if "/" in model else ""
        if provider in self.allowed_providers:
            return model
        return f"{self.default_provider}/{model}"

    def _validate_model(self, model: str) -> None:
        provider = model.split("/", 1)[0] if "/" in model else ""
        if provider not in self.allowed_providers:
            raise ValueError(
                f"model '{model}' resolves to provider '{provider or 'unknown'}', which is not "
                f"allowed. Allowed providers: {sorted(self.allowed_providers)} "
                "(set THEO_ALLOWED_PROVIDERS to change this)."
            )

    def _build_node_configs(self) -> dict[str, NodeConfig]:
        tiers = self.models_config.get("tiers", {})
        defaults = self.models_config.get("defaults", {})
        nodes = self.models_config.get("nodes", {})
        configs: dict[str, NodeConfig] = {}
        for node, raw in nodes.items():
            cfg = {**defaults, **(raw or {})}
            model_ref = cfg.get("model", "cheap")
            model = self.resolve_model(str(tiers.get(model_ref, model_ref)))
            self._validate_model(model)
            configs[node] = NodeConfig(
                model=model,
                temperature=float(cfg.get("temperature", 0.3)),
                max_tokens=int(cfg.get("max_tokens", 4000)),
                supports_json_schema=bool(cfg.get("supports_json_schema", True)),
                supports_temperature=bool(cfg.get("supports_temperature", True)),
            )
        return configs

    def node_config(self, node: str) -> NodeConfig:
        if node in self._node_configs:
            return self._node_configs[node]
        # Sensible fallback for ad-hoc nodes.
        tiers = self.models_config.get("tiers", {})
        model = self.resolve_model(str(tiers.get("cheap", self.settings.model_cheap)))
        self._validate_model(model)
        config = NodeConfig(model=model)
        self._node_configs[node] = config
        return config

    async def structured(
        self,
        node: str,
        schema: type[TModel],
        messages: list[Message],
        *,
        max_repairs: int = 2,
        extra_system: str | None = None,
    ) -> tuple[TModel, NodeUsage]:
        cfg = self.node_config(node)
        schema_json = schema.model_json_schema()
        system_prompt = _structured_system_prompt(schema, schema_json)
        if extra_system:
            system_prompt = f"{extra_system}\n\n{system_prompt}"
        convo: list[Message] = [{"role": "system", "content": system_prompt}, *messages]

        usage = Usage(model=cfg.model)
        last_error: Exception | None = None
        raw_text = ""
        for attempt in range(max_repairs + 1):
            raw = await self._complete(
                node=node,
                model=cfg.model,
                messages=convo,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                json_object=True,
                json_schema=schema_json if cfg.supports_json_schema else None,
            )
            usage.calls += 1
            usage.tokens_in += raw.tokens_in
            usage.tokens_out += raw.tokens_out
            usage.cost_usd += raw.cost_usd
            raw_text = raw.text
            try:
                data = extract_json(raw_text)
                return schema.model_validate(data), usage.to_node_usage()
            except Exception as exc:  # noqa: BLE001 - repair loop
                last_error = exc
                if attempt >= max_repairs:
                    break
                convo = [
                    *convo,
                    {"role": "assistant", "content": raw_text},
                    {
                        "role": "user",
                        "content": (
                            "The previous response failed validation with: "
                            f"{exc}.\nReturn ONLY a corrected JSON object matching the schema."
                        ),
                    },
                ]
        raise SchemaValidationError(
            f"node '{node}' failed to produce valid {schema.__name__} after "
            f"{max_repairs + 1} attempts: {last_error}. Last output: {raw_text[:300]!r}"
        )

    async def text(
        self, node: str, messages: list[Message], *, system: str | None = None
    ) -> tuple[str, NodeUsage]:
        cfg = self.node_config(node)
        prefix: list[Message] = [{"role": "system", "content": system}] if system else []
        convo: list[Message] = [*prefix, *messages]
        raw = await self._complete(
            node=node,
            model=cfg.model,
            messages=convo,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            json_object=False,
            json_schema=None,
        )
        usage = Usage(
            calls=1,
            tokens_in=raw.tokens_in,
            tokens_out=raw.tokens_out,
            cost_usd=raw.cost_usd,
            model=cfg.model,
        )
        return raw.text, usage.to_node_usage()


def _structured_system_prompt(schema: type[BaseModel], schema_json: dict[str, Any]) -> str:
    return (
        "You are a precise research assistant. Respond with a single JSON object that "
        "conforms exactly to the following JSON Schema. Do not wrap it in prose or "
        "Markdown fences.\n\nJSON Schema:\n"
        f"{json.dumps(schema_json, indent=2)}\n\n"
        "Return valid JSON only."
    )

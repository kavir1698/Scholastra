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
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from scholastra.config import Settings, get_settings, load_models_config
from scholastra.models.schemas import NodeUsage
from scholastra.utils import extract_json

Message = dict[str, str]
TModel = TypeVar("TModel", bound=BaseModel)

logger = logging.getLogger(__name__)

# Cap on how much of a prior model response is echoed back into a repair turn.
_REPAIR_ECHO_LIMIT = 6000


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
    finish_reason: str | None = None


@dataclass
class NodeConfig:
    model: str
    temperature: float = 0.3
    max_tokens: int = 4000
    supports_json_schema: bool = True
    supports_temperature: bool = True
    reasoning_effort: str | None = None
    supports_reasoning_effort: bool = True


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
        reasoning_effort: str | None = None,
    ) -> RawCompletion: ...


class LiteLLMBackend:
    """Concrete backend over ``litellm.acompletion`` with capability fallbacks."""

    def __init__(self, *, max_retries: int = 3) -> None:
        self._max_retries = max_retries
        # Remember per model whether reasoning_effort is accepted, so a rejecting
        # provider does not pay a failing probe on every subsequent call.
        self._effort_supported: dict[str, bool] = {}
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
        reasoning_effort: str | None = None,
    ) -> RawCompletion:
        import litellm

        base_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        response_formats: list[dict[str, Any]] = []
        if json_schema is not None:
            response_formats.append(
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
            response_formats.append({"response_format": {"type": "json_object"}})
        response_formats.append({})

        # Optional params are layered so a provider that rejects one of them
        # degrades to a working shape instead of failing the whole node. The
        # preferred shape (temperature + reasoning_effort) is tried first.
        # Once a model is known to reject reasoning_effort we stop probing it.
        send_effort = reasoning_effort if self._effort_supported.get(model) is not False else None
        param_shapes: list[dict[str, Any]] = []
        full: dict[str, Any] = {"temperature": temperature}
        if send_effort:
            full["reasoning_effort"] = send_effort
        param_shapes.append(full)
        if send_effort:
            param_shapes.append({"temperature": temperature})
            param_shapes.append({"reasoning_effort": send_effort})
        param_shapes.append({})

        attempts: list[dict[str, Any]] = []
        for params in param_shapes:
            for response_format in response_formats:
                shape = {**params, **response_format}
                if shape not in attempts:
                    attempts.append(shape)

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
        winning: dict[str, Any] | None = None
        for extra in attempts:
            try:
                response = await _call({**base_kwargs, **extra})
                winning = extra
                break
            except LLMProviderError:
                raise
            except Exception as exc:  # noqa: BLE001 - unsupported param -> try next shape
                last_exc = exc
                continue
        if response is None:
            raise LLMProviderError(f"all completion shapes failed: {last_exc}")

        # Record whether the model accepted reasoning_effort so later calls skip
        # the probe, and surface a silent downgrade instead of hiding it.
        if send_effort:
            accepted = winning is not None and "reasoning_effort" in winning
            self._effort_supported[model] = accepted
            if not accepted:
                logger.warning(
                    "model %s rejected reasoning_effort=%s; continuing without it",
                    model,
                    send_effort,
                )

        text = _extract_text(response)
        tokens_in, tokens_out = _usage(response)
        cost = _cost(response, model)
        return RawCompletion(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
            model=model,
            finish_reason=_finish_reason(response),
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


def _finish_reason(response: Any) -> str | None:
    """Return the provider's finish_reason (e.g. ``stop`` or ``length``), if present."""
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices")
    if not choices:
        return None
    choice = choices[0]
    reason = getattr(choice, "finish_reason", None)
    if reason is None and isinstance(choice, dict):
        reason = choice.get("finish_reason")
    return str(reason) if reason is not None else None


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
                "(set SCHOLASTRA_ALLOWED_PROVIDERS to change this)."
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
                reasoning_effort=(
                    str(cfg["reasoning_effort"]) if cfg.get("reasoning_effort") else None
                ),
                supports_reasoning_effort=bool(
                    cfg.get("supports_reasoning_effort", True)
                ),
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

    @staticmethod
    def _reasoning_effort(cfg: NodeConfig) -> str | None:
        """Return the reasoning effort to send, or None when disabled."""
        if not cfg.supports_reasoning_effort:
            return None
        return cfg.reasoning_effort or None

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
        properties, array_fields = _schema_shape(schema)
        shape_hint = _shape_hint(properties)

        usage = Usage(model=cfg.model)
        last_error: Exception | None = None
        previous_error: str | None = None
        raw_text = ""
        max_tokens = cfg.max_tokens
        for attempt in range(max_repairs + 1):
            raw = await self._complete(
                node=node,
                model=cfg.model,
                messages=convo,
                temperature=cfg.temperature,
                max_tokens=max_tokens,
                json_object=True,
                json_schema=schema_json if cfg.supports_json_schema else None,
                reasoning_effort=self._reasoning_effort(cfg),
            )
            usage.calls += 1
            usage.tokens_in += raw.tokens_in
            usage.tokens_out += raw.tokens_out
            usage.cost_usd += raw.cost_usd
            raw_text = raw.text
            truncated = raw.finish_reason == "length"
            try:
                data = _coerce_to_schema(extract_json(raw_text), array_fields)
                return schema.model_validate(data), usage.to_node_usage()
            except Exception as exc:  # noqa: BLE001 - repair loop
                last_error = exc
                if attempt >= max_repairs:
                    break
                # A deterministic repeated failure will not be resolved by another
                # identical call, so stop instead of multiplying provider spend.
                signature = str(exc)
                if signature == previous_error:
                    logger.warning(
                        "structured output for node %s failed identically twice; "
                        "stopping retries early",
                        node,
                    )
                    break
                previous_error = signature
                if truncated:
                    # The completion hit the token ceiling; give the retry more room.
                    max_tokens = min(int(max_tokens * 1.5) + 256, 32000)
                    logger.warning(
                        "structured output for node %s was truncated (finish_reason=length); "
                        "retrying with max_tokens=%d",
                        node,
                        max_tokens,
                    )
                repair_lines = [
                    f"The previous response failed validation with: {exc}.",
                ]
                if shape_hint:
                    repair_lines.append(shape_hint)
                repair_lines.append("Return ONLY a corrected JSON value matching the schema.")
                if truncated:
                    repair_lines.append(
                        "Your previous response was cut off because it hit the token limit; "
                        "return the full value, using more concise field values if necessary."
                    )
                convo = [
                    *convo,
                    {"role": "assistant", "content": _cap_echo(raw_text)},
                    {"role": "user", "content": " ".join(repair_lines)},
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
            reasoning_effort=self._reasoning_effort(cfg),
        )
        usage = Usage(
            calls=1,
            tokens_in=raw.tokens_in,
            tokens_out=raw.tokens_out,
            cost_usd=raw.cost_usd,
            model=cfg.model,
        )
        return raw.text, usage.to_node_usage()


def _cap_echo(text: str, limit: int = _REPAIR_ECHO_LIMIT) -> str:
    """Bound the previous model output echoed back in a repair turn.

    A truncated completion can be tens of thousands of characters; echoing it
    verbatim inflates input tokens on every retry.
    """
    if len(text) <= limit:
        return text
    return text[:limit] + "…[truncated]"


def _json_types(spec: Any) -> set[str]:
    """Collect the JSON types declared by a property spec, including ``anyOf``/``oneOf``.

    Nullable fields (e.g. ``list[X] | None``) are emitted as
    ``{"anyOf": [{"type": "array"}, {"type": "null"}]}``, so a flat ``type`` check
    would miss them.
    """
    if not isinstance(spec, dict):
        return set()
    types: set[str] = set()
    declared = spec.get("type")
    if isinstance(declared, str):
        types.add(declared)
    for key in ("anyOf", "oneOf"):
        for sub in spec.get(key) or []:
            types |= _json_types(sub)
    return types


def _schema_shape(schema: type[BaseModel]) -> tuple[list[str], list[str]]:
    """Return ``(property_names, array_field_names)`` for ``schema``.

    Single source of truth for the two shape signals used by the repair loop, so
    the hint and the coercion can never disagree about which fields are arrays.
    """
    try:
        spec = schema.model_json_schema()
    except Exception:  # noqa: BLE001 - shape introspection is best-effort
        return [], []
    props = spec.get("properties") or {}
    array_fields = [name for name, prop in props.items() if "array" in _json_types(prop)]
    return list(props), array_fields


def _shape_hint(properties: list[str]) -> str:
    """A short, Pydantic-derived reminder of the expected top-level shape."""
    if properties:
        return "The top-level value must be a JSON object with keys: " + ", ".join(properties) + "."
    return ""


def _coerce_to_schema(data: Any, array_fields: list[str]) -> Any:
    """Repair a common shape violation using the target Pydantic model's structure.

    When the model emits a bare JSON array but the target schema is an object with
    exactly one array-typed field (e.g. ``SectionEvidence.claims`` or
    ``IssueList.issues``), wrap the array under that field. This is deliberately
    conservative: an ambiguous shape (two or more array fields) is left untouched
    so a genuine mismatch still surfaces instead of being silently misfiled.
    """
    if isinstance(data, list) and len(array_fields) == 1:
        return {array_fields[0]: data}
    return data


def _structured_system_prompt(schema: type[BaseModel], schema_json: dict[str, Any]) -> str:
    return (
        "You are a precise research assistant. Respond with a single JSON object that "
        "conforms exactly to the following JSON Schema. Do not wrap it in prose or "
        "Markdown fences.\n\nJSON Schema:\n"
        f"{json.dumps(schema_json, indent=2)}\n\n"
        "Return valid JSON only."
    )

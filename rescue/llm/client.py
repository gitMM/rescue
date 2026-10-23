"""LLM clients: the Anthropic API, and a scripted fake for tests."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

from ..schema import SCENARIO_SCHEMA

DEFAULT_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMError(RuntimeError):
    """The model did not produce a usable scenario (refusal, truncation, API failure)."""


@dataclass
class LLMReply:
    text: str
    content: Any  # what to append to the conversation as the assistant turn
    stop_reason: str | None = None
    model: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)


class ScenarioLLM(Protocol):
    def complete(self, system: str, messages: list[dict[str, Any]]) -> LLMReply: ...


class AnthropicLLM:
    """Generates scenarios with Claude using structured outputs.

    * The response is constrained to ``SCENARIO_SCHEMA`` (``output_config.format``).
    * Thinking is adaptive (Claude Opus 5.5's default); ``effort`` sets its depth.
    * Streaming, because a scenario plus thinking can be long.
    * The system prompt is cached, so repeated generations and repair rounds
      only pay full price for the new turns.
    * Server-side refusal fallbacks are on by default (``fallbacks="default"``).
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        effort: str = "high",
        max_tokens: int = 64000,
        use_fallbacks: bool = True,
        client: Any = None,
    ):
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - depends on environment
                raise LLMError("the 'anthropic' package is required: pip install 'rescue[llm]'") from exc
            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.use_fallbacks = use_fallbacks

    def request(self, system: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": messages,
            "cache_control": {"type": "ephemeral"},
            "output_config": {
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": SCENARIO_SCHEMA},
            },
        }
        if self.use_fallbacks:
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        return kwargs

    def complete(self, system: str, messages: list[dict[str, Any]]) -> LLMReply:
        import anthropic

        try:
            with self.client.beta.messages.stream(**self.request(system, messages)) as stream:
                message = stream.get_final_message()
        except anthropic.BadRequestError as exc:
            raise LLMError(f"request rejected: {exc.message}") from exc
        except anthropic.AuthenticationError as exc:
            raise LLMError("authentication failed: set ANTHROPIC_API_KEY or run `ant auth login`") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("rate limited after retries; try fewer workers") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"could not reach the API: {exc}") from exc

        if message.stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            raise LLMError(f"the model declined the request ({getattr(details, 'category', None)})")
        if message.stop_reason == "max_tokens":
            raise LLMError(f"the scenario was cut off at max_tokens={self.max_tokens}")

        blocks = list(message.content)
        # After a mid-response fallback only the blocks after the last boundary belong to the
        # model that finished the turn; earlier thinking blocks must not be echoed back.
        boundary = max((i for i, b in enumerate(blocks) if b.type == "fallback"), default=-1)
        kept = blocks[boundary + 1 :]
        text = "".join(b.text for b in kept if b.type == "text")
        usage = message.usage.model_dump() if hasattr(message.usage, "model_dump") else {}
        return LLMReply(text=text, content=kept, stop_reason=message.stop_reason, model=message.model, usage=usage)


class FakeLLM:
    """Replays scripted replies; records every request. For tests and dry runs.

    ``replies`` items may be dicts (serialised as JSON), raw strings, or
    callables taking ``(system, messages)`` and returning either.
    """

    def __init__(self, replies: Sequence[Any]):
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def complete(self, system: str, messages: list[dict[str, Any]]) -> LLMReply:
        self.calls.append({"system": system, "messages": json.loads(json.dumps(messages, default=str))})
        if not self.replies:
            raise LLMError("FakeLLM ran out of scripted replies")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if callable(reply):
            reply = reply(system, messages)
        text = reply if isinstance(reply, str) else json.dumps(reply)
        return LLMReply(text=text, content=[{"type": "text", "text": text}], stop_reason="end_turn", model="fake")


LLMFactory = Callable[[], ScenarioLLM]

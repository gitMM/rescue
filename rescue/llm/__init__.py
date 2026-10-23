"""LLM-driven scenario generation with a validate-and-repair loop."""

from .briefs import Brief, make_brief
from .client import DEFAULT_MODEL, AnthropicLLM, FakeLLM, LLMError, LLMReply, ScenarioLLM
from .generator import GenerationResult, generate_batch, generate_scenario, save_result
from .prompt import build_system_prompt, repair_message

__all__ = [
    "AnthropicLLM",
    "Brief",
    "DEFAULT_MODEL",
    "FakeLLM",
    "GenerationResult",
    "LLMError",
    "LLMReply",
    "ScenarioLLM",
    "build_system_prompt",
    "generate_batch",
    "generate_scenario",
    "make_brief",
    "repair_message",
    "save_result",
]

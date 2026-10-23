"""Rescue: data-driven escape scenarios for a shape-shifting monster.

Scenarios are JSON data. They are compiled into a state graph
(:mod:`rescue.compiler`), stepped turn by turn (:mod:`rescue.engine`),
searched with A* (:mod:`rescue.solver`) and judged by quality gates
(:mod:`rescue.validate`). :mod:`rescue.llm` writes new scenarios with
Claude and repairs them from the validator's feedback.
"""

from .compiler import Problem, ScenarioError, compile_scenario
from .engine import Engine
from .solver import astar, explore
from .validate import Gates, format_report, validate_scenario

__all__ = [
    "Engine",
    "Gates",
    "Problem",
    "ScenarioError",
    "astar",
    "compile_scenario",
    "explore",
    "format_report",
    "validate_scenario",
]

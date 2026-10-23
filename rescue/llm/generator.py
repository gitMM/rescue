"""Generate -> validate -> repair loop, and batch generation."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..library import Library, load_library
from ..validate import Gates, validate_scenario
from .briefs import Brief, make_brief
from .client import LLMError, ScenarioLLM
from .prompt import build_system_prompt, invalid_json_message, repair_message


@dataclass
class Attempt:
    round: int
    scenario: dict[str, Any] | None
    report: dict[str, Any] | None
    error: str | None = None

    @property
    def score(self) -> tuple:
        """Higher is better: valid, solvable, then number of gates passed."""
        if self.report is None:
            return (0, 0, 0)
        passed = sum(1 for g in self.report["gates"] if g["passed"])
        return (int(self.report["valid"]), int(self.report["solvable"]), passed)


@dataclass
class GenerationResult:
    brief: Brief
    accepted: bool
    attempts: list[Attempt] = field(default_factory=list)
    error: str | None = None

    @property
    def best(self) -> Attempt | None:
        scored = [a for a in self.attempts if a.scenario is not None]
        if not scored:
            return None
        accepted = [a for a in scored if a.report and a.report["accepted"]]
        if accepted:
            return accepted[-1]
        return max(scored, key=lambda a: a.score)


def generate_scenario(
    llm: ScenarioLLM,
    brief: Brief,
    gates: Gates | None = None,
    max_rounds: int = 3,
    library: Library | None = None,
    log: Callable[[str], None] | None = None,
) -> GenerationResult:
    """Ask the model for a scenario, validate it, and feed the report back until it passes.

    The conversation is append-only: each repair round adds the model's
    previous scenario and the validator's report, so the model sees its own
    work and exactly what failed.
    """
    gates = gates or Gates()
    library = library or load_library()
    log = log or (lambda _msg: None)
    system = build_system_prompt(library, gates)
    messages: list[dict[str, Any]] = [{"role": "user", "content": brief.to_prompt()}]
    result = GenerationResult(brief=brief, accepted=False)
    for rnd in range(1, max_rounds + 1):
        try:
            reply = llm.complete(system, messages)
        except LLMError as exc:
            result.error = str(exc)
            log(f"[seed {brief.seed}] round {rnd}: {exc}")
            break
        messages.append({"role": "assistant", "content": reply.content})
        try:
            data = json.loads(reply.text)
            if not isinstance(data, dict):
                raise ValueError("top level is not an object")
        except ValueError as exc:
            result.attempts.append(Attempt(rnd, None, None, f"invalid JSON: {exc}"))
            log(f"[seed {brief.seed}] round {rnd}: invalid JSON")
            messages.append({"role": "user", "content": invalid_json_message(str(exc))})
            continue
        report = validate_scenario(data, gates, library)
        result.attempts.append(Attempt(rnd, data, report))
        failed = [g["name"] for g in report["gates"] if not g["passed"]]
        status = "accepted" if report["accepted"] else ("invalid" if not report["valid"] else f"failed {failed}")
        log(f"[seed {brief.seed}] round {rnd}: {data.get('id')} {status}")
        if report["accepted"]:
            result.accepted = True
            break
        messages.append({"role": "user", "content": repair_message(report)})
    return result


def _slug(text: object) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(text or "scenario").lower()).strip("_")[:48] or "scenario"


def save_result(result: GenerationResult, out_dir: Path) -> Path | None:
    """Write the best scenario (accepted/ or rejected/) plus its report and history."""
    best = result.best
    out_dir = Path(out_dir)
    name = f"{_slug(best.scenario.get('id', '')) if best else 'failed'}_s{result.brief.seed}"
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    history = {
        "brief": result.brief.to_dict(),
        "accepted": result.accepted,
        "error": result.error,
        "attempts": [{"round": a.round, "error": a.error, "report": a.report} for a in result.attempts],
    }
    (log_dir / f"{name}.log.json").write_text(json.dumps(history, indent=2, default=str))
    if best is None:
        return None
    target = out_dir / ("accepted" if result.accepted else "rejected")
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{name}.json"
    path.write_text(json.dumps(best.scenario, indent=2))
    (target / f"{name}.report.json").write_text(json.dumps(best.report, indent=2, default=str))
    return path


def generate_batch(
    llm_factory: Callable[[], ScenarioLLM],
    count: int,
    seed: int = 0,
    out_dir: Path | None = None,
    gates: Gates | None = None,
    max_rounds: int = 3,
    workers: int = 1,
    brief_overrides: dict[str, Any] | None = None,
    log: Callable[[str], None] | None = None,
) -> list[GenerationResult]:
    library = load_library()
    briefs = [make_brief(seed + i, **(brief_overrides or {})) for i in range(count)]

    def run(brief: Brief) -> GenerationResult:
        result = generate_scenario(llm_factory(), brief, gates, max_rounds, library, log)
        if out_dir is not None:
            save_result(result, out_dir)
        return result

    if workers <= 1:
        return [run(b) for b in briefs]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run, briefs))

"""Validate a scenario end to end and decide whether it makes a good game.

``validate_scenario`` runs the whole pipeline: structural checks, compile,
relaxed reachability, A* for the fastest escape, replay of that escape
through the engine, a breadth-first exploration of the state graph, and
ability analysis (which abilities are indispensable, and whether any single
ability is enough on its own). The result is a JSON-friendly report whose
``gates`` decide acceptance and whose ``diagnostics`` explain failures well
enough for an LLM (or a human) to repair the scenario.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from typing import Any

from .analysis import content_warnings, explain_unsolvable, relax
from .compiler import Problem, ScenarioError, compile_scenario
from .engine import WIN, Engine
from .library import Library, load_library
from .solver import analyse, astar, explore, narrate

BASIC = "move"


@dataclass
class Gates:
    """What a scenario must satisfy to be accepted. Tune per difficulty."""

    min_solution_turns: int = 8
    min_slack_turns: int = 2
    min_abilities_in_solution: int = 2
    require_combination: bool = True
    require_danger: bool = True
    min_states: int = 1000
    min_routes: int = 2
    max_doomed_ratio: float = 0.995
    explore_cap: int = 60_000
    search_cap: int = 150_000


def _gate(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def validate_scenario(data: dict[str, Any], gates: Gates | None = None, library: Library | None = None) -> dict[str, Any]:
    gates = gates or Gates()
    library = library or load_library()
    t0 = time.perf_counter()
    report: dict[str, Any] = {
        "scenario": data.get("id") if isinstance(data, dict) else None,
        "title": data.get("title") if isinstance(data, dict) else None,
        "valid": False,
        "solvable": False,
        "accepted": False,
        "errors": [],
        "warnings": [],
        "diagnostics": [],
        "gates": [],
    }
    try:
        problem = compile_scenario(data, library)
    except ScenarioError as exc:
        report["errors"] = exc.errors
    except Exception as exc:  # untrusted (LLM-written) data must never crash a batch
        report["errors"] = [f"could not compile the scenario: {type(exc).__name__}: {exc}"]
    if report["errors"]:
        report["diagnostics"].append("The scenario is malformed; fix the errors listed above.")
        report["seconds"] = round(time.perf_counter() - t0, 2)
        return report
    report["valid"] = True
    report["warnings"].extend(problem.warnings)
    report["size"] = {
        "rooms": len(problem.scenario["rooms"]),
        "npcs": len(problem.scenario["npcs"]),
        "state_vars": len(problem.vars),
        "grounded_actions": len(problem.actions),
        "grounded_rules": len(problem.rules),
        "abilities": problem.abilities,
    }
    engine = Engine(problem)
    rel = relax(problem)
    report["warnings"].extend(content_warnings(problem, rel))

    result = astar(problem, engine, max_expansions=gates.search_cap)
    report["search"] = {
        "expanded": result.expanded,
        "generated": result.generated,
        "capped": result.capped,
        "losses_seen": dict(result.losses.most_common()),
        "pruned_by_turn_limit": result.pruned,
    }
    gate_list: list[dict[str, Any]] = []
    if not result.solved:
        report["diagnostics"].extend(_unsolvable_diagnostics(problem, rel, result, gates, library))
        gate_list.append(_gate("solvable", False, "no escape found"))
        report["gates"] = gate_list
        report["seconds"] = round(time.perf_counter() - t0, 2)
        return report

    plan = result.plan or []
    _, outcome, replay_errors = engine.replay([op.name for op in plan])
    if outcome != WIN or replay_errors:
        report["errors"].append(f"internal: replaying the plan ended in {outcome!r} {replay_errors}")
        report["gates"] = [_gate("solvable", False, "plan failed to replay")]
        return report
    report["solvable"] = True
    plan_abilities = sorted({op.ability for op in plan} - {BASIC})
    report["solution"] = {
        "turns": len(plan),
        "abilities": plan_abilities,
        "steps": [asdict(s) for s in narrate(problem, plan, engine)],
    }

    graph = explore(problem, engine, max_states=gates.explore_cap)
    stats = analyse(problem, graph)
    report["graph"] = asdict(stats)

    report["abilities"] = ability_analysis(problem, engine, gates.search_cap, plan)

    gate_list.append(_gate("solvable", True, f"fastest escape takes {len(plan)} turns"))
    gate_list.append(
        _gate(
            "min_solution_turns",
            len(plan) >= gates.min_solution_turns,
            f"fastest escape {len(plan)} turns, need >= {gates.min_solution_turns}",
        )
    )
    slack = problem.turn_limit - len(plan)
    gate_list.append(
        _gate(
            "turn_slack",
            slack >= gates.min_slack_turns,
            f"turn_limit {problem.turn_limit} leaves {slack} spare turns, need >= {gates.min_slack_turns}",
        )
    )
    gate_list.append(
        _gate(
            "abilities_in_solution",
            len(plan_abilities) >= gates.min_abilities_in_solution,
            f"fastest escape uses {plan_abilities or 'only basic moves'}, need >= {gates.min_abilities_in_solution} distinct abilities",
        )
    )
    if gates.require_combination:
        solo = report["abilities"]["solo_escapes"]
        gate_list.append(
            _gate(
                "requires_combination",
                not solo,
                "no single ability is enough on its own"
                if not solo
                else f"escape possible with basic moves plus only one of: {solo}",
            )
        )
    if gates.require_danger:
        # Running out of time is always possible; danger means something can actually catch you.
        loss_kinds = (set(stats.losses) | set(result.losses)) - {"out_of_time"}
        gate_list.append(
            _gate(
                "danger",
                bool(loss_kinds),
                f"reachable losses: {sorted(loss_kinds)}" if loss_kinds else "nothing but the clock can stop the creature",
            )
        )
    gate_list.append(
        _gate(
            "min_states",
            stats.states >= gates.min_states,
            f"{stats.states}{'+ (capped)' if stats.truncated else ''} reachable states, need >= {gates.min_states}",
        )
    )
    routes = report["abilities"]["routes"]
    gate_list.append(
        _gate(
            "min_routes",
            len(routes) >= gates.min_routes,
            f"{len(routes)} escape routes with different ability sets "
            f"({'; '.join('+'.join(r['abilities']) or 'basic' for r in routes)}), need >= {gates.min_routes}",
        )
    )
    if stats.doomed_ratio is not None and not stats.truncated:
        gate_list.append(
            _gate(
                "max_doomed_ratio",
                stats.doomed_ratio <= gates.max_doomed_ratio,
                f"{stats.doomed_ratio:.1%} of decided states are dead ends, max {gates.max_doomed_ratio:.1%}",
            )
        )
    report["gates"] = gate_list
    report["accepted"] = all(g["passed"] for g in gate_list) and not report["errors"]
    for g in gate_list:
        if not g["passed"]:
            report["diagnostics"].append(_gate_advice(g["name"], report))
    report["seconds"] = round(time.perf_counter() - t0, 2)
    return report


def ability_analysis(problem: Problem, engine: Engine, cap: int, plan: list | None = None) -> dict[str, Any]:
    """Which abilities are indispensable, can one ability alone win, and which distinct routes exist?

    Each special ability is banned in turn; the fastest escape that remains
    (if any) is an alternative route. Routes are deduplicated by the set of
    abilities they use.
    """
    every = frozenset(problem.abilities)
    special = sorted(every - {BASIC})
    required, solo, inconclusive = [], [], []
    routes: dict[tuple[str, ...], dict[str, Any]] = {}

    def add_route(ops: list, banned: str | None) -> None:
        key = tuple(sorted({op.ability for op in ops} - {BASIC}))
        if key not in routes:
            routes[key] = {
                "abilities": list(key),
                "turns": len(ops),
                "without": banned,
                "actions": [op.name for op in ops],
            }

    if plan:
        add_route(plan, None)
    basic_only = astar(problem, engine, allowed=frozenset({BASIC}), max_expansions=cap)
    if basic_only.solved:
        solo.append(BASIC)
    for ab in special:
        without = astar(problem, engine, allowed=every - {ab}, max_expansions=cap)
        if without.capped:
            inconclusive.append(f"without {ab}")
        elif not without.solved:
            required.append(ab)
        else:
            add_route(without.plan or [], ab)
        alone = astar(problem, engine, allowed=frozenset({BASIC, ab}), max_expansions=cap)
        if alone.solved:
            solo.append(ab)
        elif alone.capped:
            inconclusive.append(f"only {ab}")
    return {
        "all": special,
        "required": required,
        "solo_escapes": solo,
        "inconclusive": inconclusive,
        "routes": sorted(routes.values(), key=lambda r: r["turns"]),
    }


def _unsolvable_diagnostics(problem: Problem, rel, result, gates: Gates, library: Library) -> list[str]:
    out: list[str] = []
    if not rel.goal_possible:
        out.extend(explain_unsolvable(problem, rel))
        return out
    if result.capped:
        out.append(
            f"Search gave up after {result.expanded} expanded states without finding an escape; "
            "the scenario may be solvable but is too large or too open. Reduce rooms, NPCs or turn_limit."
        )
    else:
        out.append("An exit is reachable in principle, but every concrete line of play fails.")
    if result.losses:
        top = ", ".join(f"{k} x{v}" for k, v in result.losses.most_common(4))
        out.append(f"Losing outcomes the search ran into: {top}.")
    if result.pruned:
        out.append(f"{result.pruned} branches were cut because the exit was too far for the remaining turns.")
    if result.closest is not None:
        out.append(f"Closest approach: {problem.summary(result.closest)} (at least {problem.h(result.closest):.0f} moves from an exit).")
    longer = copy.deepcopy(problem.scenario)
    longer["turn_limit"] = min(200, max(problem.turn_limit * 3, problem.turn_limit + 20))
    try:
        relaxed_problem = compile_scenario(longer, library)
        again = astar(relaxed_problem, max_expansions=gates.search_cap)
        if again.solved:
            out.append(
                f"With a turn_limit of {longer['turn_limit']} an escape exists in {again.turns} turns: "
                f"raise turn_limit to at least {again.turns + gates.min_slack_turns} or shorten the route."
            )
    except ScenarioError:
        pass
    return out


_ADVICE = {
    "min_solution_turns": "The escape is too quick. Add another security layer between the start and the exits "
    "(a locked door that needs a different ability, a sealed vent, a terminal to reach first).",
    "turn_slack": "The turn limit is too tight for the fastest route. Raise turn_limit or shorten the route.",
    "abilities_in_solution": "The fastest escape barely uses the monster's powers. Gate the route so it needs "
    "at least two different abilities (e.g. vents to reach a terminal, hacking to open the exit).",
    "requires_combination": "One ability is enough to escape on its own. Close that shortcut: lock or seal the "
    "passage it uses, or put a later obstacle on that route that needs a different ability.",
    "danger": "Nothing can ever go wrong. Add armed guards on patrol routes, cameras, or tighter alarm thresholds.",
    "min_states": "The game graph is tiny. Add NPCs with patrols, items, terminals or alternative routes.",
    "min_routes": "There is essentially one way out. Add an alternative route that relies on different abilities "
    "(e.g. a vent path to a terminal as well as a disguise path through a badge door).",
    "max_doomed_ratio": "Almost every situation is already lost. Loosen alarm thresholds, add slack turns or more routes.",
}


def _gate_advice(name: str, report: dict[str, Any]) -> str:
    advice = _ADVICE.get(name, "")
    if name == "requires_combination":
        advice += f" (solo escapes: {report['abilities']['solo_escapes']})"
    return f"Gate '{name}' failed: {advice}"


# --------------------------------------------------------------------------
# Human-readable summaries
# --------------------------------------------------------------------------


def format_report(report: dict[str, Any], steps: bool = True, max_steps: int = 60) -> str:
    lines = []
    status = "ACCEPTED" if report["accepted"] else ("REJECTED" if report["valid"] else "INVALID")
    lines.append(f"[{status}] {report.get('title') or report.get('scenario')} ({report.get('scenario')})")
    if report.get("size"):
        s = report["size"]
        lines.append(
            f"  size: {s['rooms']} rooms, {s['npcs']} npcs, {s['state_vars']} state vars, "
            f"{s['grounded_actions']} grounded actions, {s['grounded_rules']} rules"
        )
    for e in report["errors"][:30]:
        lines.append(f"  error: {e}")
    if report.get("graph"):
        g = report["graph"]
        losses = ", ".join(f"{k} {v}" for k, v in g["losses"].items()) or "none"
        ratio = "n/a" if g["doomed_ratio"] is None else f"{g['doomed_ratio']:.1%}"
        lines.append(
            f"  graph: {g['states']}{'+' if g['truncated'] else ''} states, {g['edges']} edges, "
            f"branching {g['branching']:.1f}, wins {g['wins']}, losses [{losses}], "
            f"winning paths {'>=' if not g['winning_paths_exact'] else ''}{g['winning_paths']}, dead-end ratio {ratio}"
        )
    if report.get("abilities"):
        a = report["abilities"]
        lines.append(f"  abilities: required {a['required'] or '-'}; single-ability escapes {a['solo_escapes'] or '-'}")
        for r in a["routes"]:
            banned = f" (with {r['without']} banned)" if r["without"] else " (fastest)"
            lines.append(f"  route: {r['turns']} turns using {'+'.join(r['abilities']) or 'basic moves'}{banned}")
    for g in report["gates"]:
        lines.append(f"  gate {'PASS' if g['passed'] else 'FAIL'} {g['name']}: {g['detail']}")
    for w in report["warnings"][:10]:
        lines.append(f"  warning: {w}")
    for d in report["diagnostics"][:20]:
        lines.append(f"  diagnostic: {d}")
    sol = report.get("solution")
    if sol and steps:
        lines.append(f"  fastest escape ({sol['turns']} turns, abilities {sol['abilities']}):")
        for st in sol["steps"][:max_steps]:
            lines.append(f"    t{st['turn']:>2} {st['action']}: {st['text']}")
            for r in st["reactions"]:
                lines.append(f"        ! {r}")
    lines.append(f"  ({report.get('seconds', 0)}s)")
    return "\n".join(lines)


def repair_brief(report: dict[str, Any]) -> str:
    """The compact feedback an LLM gets when its scenario is rejected."""
    lines = []
    if report["errors"]:
        lines.append("Errors (the scenario could not be compiled):")
        lines += [f"- {e}" for e in report["errors"][:30]]
    failed = [g for g in report["gates"] if not g["passed"]]
    if failed:
        lines.append("Failed quality gates:")
        lines += [f"- {g['name']}: {g['detail']}" for g in failed]
    passed = [g for g in report["gates"] if g["passed"]]
    if passed:
        lines.append("Passed gates: " + ", ".join(g["name"] for g in passed))
    if report["diagnostics"]:
        lines.append("Diagnostics:")
        lines += [f"- {d}" for d in report["diagnostics"][:20]]
    if report["warnings"]:
        lines.append("Warnings:")
        lines += [f"- {w}" for w in report["warnings"][:10]]
    sol = report.get("solution")
    if sol:
        lines.append(f"Current fastest escape ({sol['turns']} turns):")
        lines += [f"  t{s['turn']} {s['action']}" for s in sol["steps"][:40]]
    if report.get("graph"):
        g = report["graph"]
        lines.append(
            f"Graph: {g['states']} states, losses {g['losses']}, "
            f"abilities required {report['abilities']['required']}, single-ability escapes {report['abilities']['solo_escapes']}"
        )
        for r in report["abilities"]["routes"]:
            lines.append(f"Route: {r['turns']} turns using {'+'.join(r['abilities']) or 'basic moves'}")
    return "\n".join(lines)

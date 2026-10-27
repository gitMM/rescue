"""Command line: validate scenarios, generate new ones with Claude, inspect prompts."""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from dataclasses import fields
from pathlib import Path

from .compiler import ScenarioError, compile_scenario
from .engine import WIN, Engine
from .library import example_paths
from .solver import narrate_step
from .validate import Gates, format_report, validate_scenario


def _gates(args: argparse.Namespace) -> Gates:
    gates = Gates()
    known = {f.name: f.type for f in fields(Gates)}
    for item in args.gate or []:
        key, _, value = item.partition("=")
        if key not in known:
            raise SystemExit(f"unknown gate {key!r}; choose from {sorted(known)}")
        current = getattr(gates, key)
        if isinstance(current, bool):
            setattr(gates, key, value.lower() in ("1", "true", "yes", "on"))
        elif isinstance(current, int):
            setattr(gates, key, int(value))
        else:
            setattr(gates, key, float(value))
    return gates


def cmd_validate(args: argparse.Namespace) -> int:
    paths = [Path(p) for p in args.files] or example_paths()
    gates = _gates(args)
    reports = []
    ok = True
    for path in paths:
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            print(f"[INVALID] {path}: {exc}", file=sys.stderr)
            ok = False
            continue
        report = validate_scenario(data, gates)
        report["file"] = str(path)
        reports.append(report)
        ok &= report["accepted"]
        if not args.json:
            print(format_report(report, steps=not args.no_steps))
            print()
    if args.json:
        print(json.dumps(reports if len(reports) != 1 else reports[0], indent=2, default=str))
    return 0 if ok else 1


def cmd_generate(args: argparse.Namespace) -> int:
    from .llm import AnthropicLLM, FakeLLM, generate_batch

    gates = _gates(args)
    if args.fake:
        scripted = [json.loads(Path(p).read_text()) for p in args.fake]

        def factory():
            return FakeLLM(list(scripted))

    else:

        def factory():
            return AnthropicLLM(model=args.model, effort=args.effort, use_fallbacks=not args.no_fallbacks)

    overrides = {k: v for k, v in (("setting", args.setting), ("twist", args.twist), ("size", args.size), ("difficulty", args.difficulty)) if v}
    results = generate_batch(
        factory,
        count=args.count,
        seed=args.seed,
        out_dir=Path(args.out),
        gates=gates,
        max_rounds=args.rounds,
        workers=args.workers,
        brief_overrides=overrides,
        log=lambda msg: print(msg, file=sys.stderr, flush=True),
    )
    accepted = sum(r.accepted for r in results)
    for r in results:
        best = r.best
        name = best.scenario.get("id") if best and best.scenario else "-"
        rounds = len(r.attempts)
        status = "accepted" if r.accepted else ("error: " + r.error if r.error else "rejected")
        print(f"seed {r.brief.seed}: {name} after {rounds} round(s): {status}")
    print(f"{accepted}/{len(results)} accepted; written to {args.out}")
    return 0 if accepted == len(results) else 1


def cmd_prompt(args: argparse.Namespace) -> int:
    from .llm import build_system_prompt, make_brief

    if args.brief is not None:
        print(make_brief(args.brief).to_prompt())
    else:
        print(build_system_prompt(gates=_gates(args)))
    return 0


def cmd_schema(_args: argparse.Namespace) -> int:
    from .schema import SCENARIO_SCHEMA

    print(json.dumps(SCENARIO_SCHEMA, indent=2))
    return 0


def cmd_play(args: argparse.Namespace) -> int:
    """Replay a line of play step by step, explaining any step that is impossible."""
    try:
        problem = compile_scenario(json.loads(Path(args.file).read_text()))
    except ScenarioError as exc:
        print("\n".join(f"error: {e}" for e in exc.errors))
        return 2
    engine = Engine(problem)
    by_name = {a.name: a for a in problem.actions}
    state = problem.init
    print(f"start: {problem.summary(state)}")
    for name in args.actions:
        op = by_name.get(name)
        if op is None:
            close = difflib.get_close_matches(name, list(by_name), n=5, cutoff=0.5)
            print(f"unknown action {name!r}" + (f"; did you mean: {', '.join(close)}" if close else ""))
            return 1
        if not op.holds(state):
            print(f"t{state[problem.turn]} {name} is not possible now; failing: {'; '.join(op.failing(state))}")
            print(f"state: {problem.summary(state)}")
            return 1
        step, state, outcome = narrate_step(engine, state, op)
        print(f"t{step.turn} {name}: {step.text}")
        for r in step.reactions:
            print(f"    ! {r}")
        print(f"    -> {step.after}")
        if outcome is not None:
            print(f"outcome: {outcome}")
            return 0 if outcome == WIN else 1
    available = sorted(op.name for op in engine.applicable(state))
    print(f"available now ({len(available)}): {', '.join(available)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rescue", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def gate_opt(p: argparse.ArgumentParser) -> None:
        p.add_argument("--gate", action="append", metavar="NAME=VALUE", help="override a quality gate, e.g. min_solution_turns=12")

    v = sub.add_parser("validate", help="validate scenario files (default: the bundled examples)")
    v.add_argument("files", nargs="*")
    v.add_argument("--json", action="store_true", help="print the full JSON report")
    v.add_argument("--no-steps", action="store_true", help="do not print the escape walkthrough")
    gate_opt(v)
    v.set_defaults(func=cmd_validate)

    g = sub.add_parser("generate", help="generate scenarios with Claude and validate/repair them")
    g.add_argument("--count", type=int, default=1)
    g.add_argument("--seed", type=int, default=0, help="first brief seed; scenario i uses seed+i")
    g.add_argument("--out", default="scenarios/generated")
    g.add_argument("--rounds", type=int, default=3, help="max generate/repair rounds per scenario")
    g.add_argument("--workers", type=int, default=1)
    g.add_argument("--model", default="claude-opus-5-5")
    g.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    g.add_argument("--no-fallbacks", action="store_true", help="disable server-side refusal fallbacks")
    g.add_argument("--setting")
    g.add_argument("--twist")
    g.add_argument("--size", choices=["small", "medium", "large"])
    g.add_argument("--difficulty", choices=["easy", "normal", "hard"])
    g.add_argument("--fake", nargs="+", metavar="JSON", help="offline dry run: replay these scenario files instead of calling the API")
    gate_opt(g)
    g.set_defaults(func=cmd_generate)

    p = sub.add_parser("prompt", help="print the generator's system prompt (or a brief)")
    p.add_argument("--brief", type=int, metavar="SEED", help="print the brief for this seed instead")
    gate_opt(p)
    p.set_defaults(func=cmd_prompt)

    s = sub.add_parser("schema", help="print the scenario JSON schema")
    s.set_defaults(func=cmd_schema)

    pl = sub.add_parser("play", help="replay actions step by step and list what is possible next")
    pl.add_argument("file")
    pl.add_argument("actions", nargs="*", help="grounded action names, e.g. 'wait' 'absorb(dr_ito)'")
    pl.set_defaults(func=cmd_play)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

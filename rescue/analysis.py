"""Relaxed reachability: cheap, optimistic analysis used for diagnostics.

The *relaxation* ignores ordering, the turn limit, dangers and the fact
that variables can only hold one value at a time: it collects, for every
state variable, the set of values it could ever take. If the goal is
impossible even here, the scenario is unsolvable, and the operators that
stay blocked explain why ("door d_vault never unlocks: unlock(term_sec,
d_vault) needs term.term_sec.hacked == true, ...").
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import dsl
from .compiler import CAtom, CCond, CCondEffect, CEffect, Operator, Problem, Var, loc_moves

Values = list[set[int]]


@dataclass
class Relaxed:
    values: Values
    applicable: set[int] = field(default_factory=set)  # indices into problem.actions
    rules_applicable: set[int] = field(default_factory=set)
    goal_possible: bool = False


def _possible_values(problem: Problem, var: Var, R: Values) -> set[int]:
    if var.index is not None:
        return R[var.index]
    if var.name.startswith("npc.") and var.name.endswith(".loc"):
        nid = var.name[4:-4]
        npc = next(n for n in problem.scenario["npcs"] if n["id"] == nid)
        dom = var.domain
        vals = {dom.encode(r) for r in npc["patrol"]} | {0}
        vals |= R[problem.var_index[f"npc.{nid}.down_at"].index]  # type: ignore[index]
        return vals
    return set(var.domain.codes())


def atom_possible(problem: Problem, atom: CAtom, R: Values) -> bool:
    lhs = _possible_values(problem, atom.var, R)
    if atom.op in ("in", "not in"):
        return any(dsl.compare(atom.op, v, atom.const) for v in lhs)
    if atom.ref is not None:
        rhs = _possible_values(problem, atom.ref, R)
        if atom.table is not None:
            rhs = {atom.table[c] for c in rhs}
        return any(dsl.compare(atom.op, l, r) for l in lhs for r in rhs)
    return any(dsl.compare(atom.op, v, atom.const) for v in lhs)


def cond_possible(problem: Problem, cond: CCond, R: Values) -> bool:
    return any(atom_possible(problem, a, R) for a in cond.atoms)


def _apply_relaxed(problem: Problem, effects: list, R: Values) -> bool:
    changed = False
    for e in effects:
        if isinstance(e, CCondEffect):
            if all(cond_possible(problem, c, R) for c in e.conds):
                changed |= _apply_relaxed(problem, e.then, R)
            changed |= _apply_relaxed(problem, e.otherwise, R)
            continue
        assert isinstance(e, CEffect)
        target = R[e.var.index]  # type: ignore[index]
        before = len(target)
        dom = e.var.domain
        if e.op == "set":
            target.add(e.const)
        elif e.op == "add":
            for v in list(target):
                target.add(min(dom.hi, max(dom.lo, v + e.const)))
        else:
            src = _possible_values(problem, e.ref, R)  # type: ignore[arg-type]
            if e.table is not None:
                src = {e.table[c] for c in src if e.table[c] >= 0}
            if dom.kind == "int":
                src = {min(dom.hi, max(dom.lo, c)) for c in src}
            target |= src
        changed |= len(target) != before
    return changed


def relax(problem: Problem) -> Relaxed:
    R: Values = [{v} for v in problem.init]
    R[problem.turn] = set(problem.var_index["turn"].domain.codes())
    result = Relaxed(R)
    changed = True
    while changed:
        changed = False
        for kind, ops in (("action", problem.actions), ("rule", problem.rules)):
            for op in ops:
                if all(cond_possible(problem, c, R) for c in op.conds):
                    (result.applicable if kind == "action" else result.rules_applicable).add(op.index)
                    changed |= _apply_relaxed(problem, op.effects, R)
    result.goal_possible = all(cond_possible(problem, c, R) for c in problem.goal)
    return result


def blockers(problem: Problem, op: Operator | list[CCond], R: Values) -> list[CCond]:
    conds = op.conds if isinstance(op, Operator) else op
    return [c for c in conds if not cond_possible(problem, c, R)]


def _setters(problem: Problem, cond: CCond) -> list[Operator]:
    """Operators with an effect on a variable the condition reads."""
    names = {a.var.name for a in cond.atoms if a.var.index is not None}
    out = []
    for op in problem.actions + problem.rules:
        for e in op.effects:
            effs = (e.then + e.otherwise) if isinstance(e, CCondEffect) else [e]
            if any(x.var.name in names for x in effs):
                out.append(op)
                break
    return out


def _why(problem: Problem, cond: CCond, R: Values, limit: int = 3) -> str:
    setters = _setters(problem, cond)
    if not setters:
        return f"`{cond.text}` is never true and nothing in the scenario can change it"
    parts = []
    for op in setters[:limit]:
        blk = blockers(problem, op, R)
        if blk:
            parts.append(f"{op.name} (needs `{blk[0].text}`)")
        else:
            parts.append(f"{op.name}")
    more = f" and {len(setters) - limit} more" if len(setters) > limit else ""
    return f"`{cond.text}` could only come from " + ", ".join(parts) + more


def explain_unsolvable(problem: Problem, rel: Relaxed, max_lines: int = 14) -> list[str]:
    """Why the goal is out of reach even under the optimistic relaxation."""
    R = rel.values
    lines: list[str] = []
    loc = problem.loc
    reach = R[loc]
    names = {r["id"]: r["name"] for r in problem.scenario["rooms"]}
    reached = [problem.room_ids[c] for c in sorted(reach) if c]
    lines.append("Even ignoring danger, timing and the turn limit, no exit is reachable.")
    lines.append("Rooms the monster can ever reach: " + ", ".join(reached))
    exits_ok = any(e in reach for e in problem.exits)
    if exits_ok:
        for c in problem.goal[1:]:
            if not cond_possible(problem, c, R):
                lines.append("Goal condition never holds: " + _why(problem, c, R))
        return lines[:max_lines]
    seen_targets: dict[int, list[str]] = {}
    explained: set[str] = set()
    for op in problem.actions:
        if op.loc_req is None or op.loc_req not in reach:
            continue
        targets = loc_moves(op, loc, len(problem.room_ids)) - reach
        if not targets:
            continue
        blk = blockers(problem, op, R)
        if not blk:
            continue
        for t in targets:
            seen_targets.setdefault(t, []).append(f"{op.name} needs `{blk[0].text}`")
        key = blk[0].text
        if key not in explained and len(explained) < 6:
            explained.add(key)
            seen_targets.setdefault(-1, []).append(_why(problem, blk[0], R))
    for t, ways in seen_targets.items():
        if t == -1:
            continue
        room = problem.room_ids[t]
        shown = "; ".join(ways[:4]) + (f"; +{len(ways) - 4} more" if len(ways) > 4 else "")
        lines.append(f"Blocked way into {names.get(room, room)} ({room}): {shown}")
    for why in seen_targets.get(-1, []):
        lines.append("Because " + why)
    if len(lines) == 2:
        lines.append("No action leads out of the reachable rooms at all: add doors, vents or custom moves.")
    return lines[:max_lines]


def content_warnings(problem: Problem, rel: Relaxed) -> list[str]:
    """Things that exist in the scenario but can never matter."""
    R = rel.values
    out: list[str] = []
    reach = R[problem.loc]
    dead_rooms = [problem.room_ids[c] for c in range(1, len(problem.room_ids)) if c not in reach]
    if dead_rooms:
        out.append("unreachable rooms (even optimistically): " + ", ".join(dead_rooms))
    dead_items = [v.name[4:] for v in problem.vars if v.name.startswith("has.") and 1 not in R[v.index]]  # type: ignore[index]
    if dead_items:
        out.append("items that can never be obtained: " + ", ".join(dead_items))
    dead_terms = [v.name[5:-7] for v in problem.vars if v.name.startswith("term.") and 1 not in R[v.index]]  # type: ignore[index]
    if dead_terms:
        out.append("terminals that can never be accessed: " + ", ".join(dead_terms))
    unused = len(problem.actions) - len(rel.applicable)
    if unused and problem.actions:
        out.append(f"{unused} of {len(problem.actions)} grounded actions can never be used")
    return out

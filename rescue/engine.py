"""Turn order: how one monster action turns one state into the next.

1. The monster takes one action (preconditions read the current state;
   effects are computed from it).
2. The clock advances, so everyone moves to the next stop on their patrol.
3. If the monster is now in an exit (and any ``goal_extra`` holds), it has
   escaped. Nothing else happens.
4. World rules fire once each, in order, seeing each other's changes.
5. Loss conditions are checked in order; the first one that holds ends the
   game.
"""

from __future__ import annotations

from typing import Iterator

from .compiler import ExprBuilder, Operator, Problem, make_test

WIN = "win"


class Engine:
    def __init__(self, problem: Problem):
        self.p = problem
        self._loc = problem.loc
        self._turn = problem.turn
        self._by_loc = problem.by_loc
        self._any = problem.any_loc
        self._goal = make_test(problem.goal)
        self._world = _world_phase(problem)

    def applicable(self, state: tuple, allowed: frozenset[str] | None = None) -> Iterator[Operator]:
        for op in self._by_loc.get(state[self._loc], ()):
            if (allowed is None or op.ability in allowed) and op.test(state):
                yield op
        for op in self._any:
            if (allowed is None or op.ability in allowed) and op.test(state):
                yield op

    def step(self, state: tuple, op: Operator) -> tuple[tuple, str | None]:
        """Apply ``op``; return the next state and its outcome (None, 'win' or a loss id)."""
        w = list(state)
        op.apply(state, w)
        w[self._turn] += 1
        if self._goal(w):
            return tuple(w), WIN
        outcome = self._world(w)
        return tuple(w), outcome

    def successors(self, state: tuple, allowed: frozenset[str] | None = None) -> Iterator[tuple[Operator, tuple, str | None]]:
        for op in self.applicable(state, allowed):
            nxt, outcome = self.step(state, op)
            yield op, nxt, outcome

    def fired_rules(self, state: tuple, op: Operator) -> list[Operator]:
        """Replay one step and report which world rules fired (for narration)."""
        w = list(state)
        op.apply(state, w)
        w[self._turn] += 1
        if self._goal(w):
            return []
        fired = []
        for rule in self.p.rules:
            if rule.holds(w):
                rule.apply(w, w)
                fired.append(rule)
        return fired

    def replay(self, names: list[str]) -> tuple[tuple, str | None, list[str]]:
        """Run a sequence of action names from the start; returns (state, outcome, errors)."""
        by_name = {a.name: a for a in self.p.actions}
        state, outcome = self.p.init, None
        for i, name in enumerate(names):
            op = by_name.get(name)
            if op is None:
                return state, outcome, [f"step {i}: unknown action {name!r}"]
            if outcome is not None:
                return state, outcome, [f"step {i}: game already ended ({outcome})"]
            if not op.holds(state):
                return state, outcome, [f"step {i}: {name} is not possible in this state"]
            state, outcome = self.step(state, op)
        return state, outcome, []


def _world_phase(problem: Problem):
    """Generate one function that fires every rule in order and checks losses.

    Mutates the working state list ``w`` and returns the id of the first
    loss that holds, or None.
    """
    b = ExprBuilder()
    lines = ["def _world(w):"]
    for rule in problem.rules:
        lines.append(f"    if {b.conds(rule.conds, 'w')}:")
        lines.append(f"        {b.const(rule.apply)}(w, w)")
    for loss in problem.losses:
        lines.append(f"    if {b.conds(loss.conds, 'w')}:")
        lines.append(f"        return {b.const(loss.id)}")
    lines.append("    return None")
    exec("\n".join(lines), b.ns)  # noqa: S102 - generated from ints/indices only
    return b.ns["_world"]

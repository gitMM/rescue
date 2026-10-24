"""Graph search over compiled scenarios.

* :func:`astar` finds the fastest escape (fewest turns). The state includes
  the turn counter, so the graph is a DAG and every path to a state has the
  same length; the room-distance heuristic is admissible and consistent.
* :func:`explore` enumerates the reachable state graph breadth-first (up to
  a cap) and :func:`analyse` counts winning paths, losing outcomes and
  doomed states (from which no escape is possible any more).
"""

from __future__ import annotations

import heapq
import itertools
from collections import Counter, deque
from dataclasses import dataclass, field

from .compiler import Operator, Problem
from .engine import WIN, Engine


@dataclass
class Step:
    turn: int
    action: str
    ability: str
    text: str
    reactions: list[str]
    after: str


@dataclass
class SearchResult:
    plan: list[Operator] | None
    states: list[tuple]
    expanded: int
    generated: int
    capped: bool
    losses: Counter
    pruned: int
    closest: tuple | None

    @property
    def solved(self) -> bool:
        return self.plan is not None

    @property
    def turns(self) -> int | None:
        return None if self.plan is None else len(self.plan)


def astar(
    problem: Problem,
    engine: Engine | None = None,
    allowed: frozenset[str] | None = None,
    max_expansions: int = 200_000,
    start: tuple | None = None,
) -> SearchResult:
    engine = engine or Engine(problem)
    h, tl, turn_i = problem.h, problem.turn_limit, problem.turn
    start = problem.init if start is None else start
    parent: dict[tuple, tuple[tuple, Operator] | None] = {start: None}
    counter = itertools.count()
    h0 = h(start)
    heap: list = [(h0, -start[turn_i], next(counter), start, False)]
    losses: Counter = Counter()
    expanded = generated = pruned = 0
    closest, closest_h = start, h0
    if start[turn_i] + h0 > tl:
        heap = []
    while heap:
        _, _, _, state, is_win = heapq.heappop(heap)
        if is_win:
            plan: list[Operator] = []
            states = [state]
            cur = state
            while parent[cur] is not None:
                prev, op = parent[cur]  # type: ignore[misc]
                plan.append(op)
                states.append(prev)
                cur = prev
            plan.reverse()
            states.reverse()
            return SearchResult(plan, states, expanded, generated, False, losses, pruned, state)
        if expanded >= max_expansions:
            return SearchResult(None, [], expanded, generated, True, losses, pruned, closest)
        expanded += 1
        for op, nxt, outcome in engine.successors(state, allowed):
            generated += 1
            if nxt in parent:
                continue
            if outcome is not None and outcome != WIN:
                losses[outcome] += 1
                continue
            t = nxt[turn_i]
            if outcome == WIN:
                parent[nxt] = (state, op)
                heapq.heappush(heap, (t, -t, next(counter), nxt, True))
                continue
            hn = h(nxt)
            if t + hn > tl:
                pruned += 1
                continue
            parent[nxt] = (state, op)
            if hn < closest_h or (hn == closest_h and t > closest[turn_i]):
                closest, closest_h = nxt, hn
            heapq.heappush(heap, (t + hn, -t, next(counter), nxt, False))
    return SearchResult(None, [], expanded, generated, False, losses, pruned, closest)


def narrate(problem: Problem, plan: list[Operator], engine: Engine | None = None) -> list[Step]:
    engine = engine or Engine(problem)
    steps: list[Step] = []
    state = problem.init
    for op in plan:
        reactions = [r.text for r in engine.fired_rules(state, op) if r.text]
        nxt, _ = engine.step(state, op)
        steps.append(Step(state[problem.turn], op.name, op.ability, op.text, reactions, problem.summary(nxt)))
        state = nxt
    return steps


# --------------------------------------------------------------------------
# Full exploration
# --------------------------------------------------------------------------


@dataclass
class Graph:
    states: list[tuple]
    outcome: list[str | None]
    succ: list[list[tuple[int, int]] | None]  # (action index, state id); None = not expanded
    truncated: bool


def explore(problem: Problem, engine: Engine | None = None, max_states: int = 100_000) -> Graph:
    engine = engine or Engine(problem)
    ids = {problem.init: 0}
    states = [problem.init]
    outcome: list[str | None] = [None]
    succ: list[list[tuple[int, int]] | None] = [None]
    queue = deque([0])
    truncated = False
    while queue:
        i = queue.popleft()
        if outcome[i] is not None:
            succ[i] = []
            continue
        if len(states) >= max_states:
            truncated = True
            continue
        edges = []
        for op, nxt, out in engine.successors(states[i]):
            j = ids.get(nxt)
            if j is None:
                j = len(states)
                ids[nxt] = j
                states.append(nxt)
                outcome.append(out)
                succ.append(None)
                queue.append(j)
            edges.append((op.index, j))
        succ[i] = edges
    return Graph(states, outcome, succ, truncated)


@dataclass
class GraphStats:
    states: int
    expanded: int
    truncated: bool
    edges: int
    wins: int
    losses: dict[str, int]
    winnable: int
    doomed: int
    unknown: int
    doomed_ratio: float | None
    winning_paths: int
    winning_paths_exact: bool
    fastest_win: int | None
    slowest_win: int | None
    branching: float
    by_turn: list[int] = field(default_factory=list)


def analyse(problem: Problem, graph: Graph) -> GraphStats:
    n = len(graph.states)
    turn_i = problem.turn
    t0 = problem.init[turn_i]
    order = sorted(range(n), key=lambda i: graph.states[i][turn_i], reverse=True)
    paths = [0] * n
    known = [False] * n
    for i in order:
        out = graph.outcome[i]
        if out == WIN:
            paths[i], known[i] = 1, True
        elif out is not None:
            known[i] = True
        elif graph.succ[i] is None:
            known[i] = False
        else:
            total, all_known = 0, True
            for _, j in graph.succ[i]:  # type: ignore[union-attr]
                total += paths[j]
                all_known &= known[j]
            paths[i], known[i] = total, all_known
    wins = sum(1 for o in graph.outcome if o == WIN)
    loss_counts = Counter(o for o in graph.outcome if o not in (None, WIN))
    winnable = doomed = unknown = 0
    for i in range(n):
        if graph.outcome[i] is not None:
            continue
        if paths[i] > 0:
            winnable += 1
        elif known[i]:
            doomed += 1
        else:
            unknown += 1
    decided = winnable + doomed
    win_turns = [graph.states[i][turn_i] - t0 for i in range(n) if graph.outcome[i] == WIN]
    expanded = sum(1 for s in graph.succ if s is not None)
    edges = sum(len(s) for s in graph.succ if s)
    nonterminal_expanded = sum(1 for i, s in enumerate(graph.succ) if s is not None and graph.outcome[i] is None)
    by_turn = Counter(s[turn_i] for s in graph.states)
    return GraphStats(
        states=n,
        expanded=expanded,
        truncated=graph.truncated,
        edges=edges,
        wins=wins,
        losses=dict(loss_counts.most_common()),
        winnable=winnable,
        doomed=doomed,
        unknown=unknown,
        doomed_ratio=(doomed / decided) if decided else None,
        winning_paths=paths[0],
        winning_paths_exact=known[0],
        fastest_win=min(win_turns) if win_turns else None,
        slowest_win=max(win_turns) if win_turns else None,
        branching=(edges / nonterminal_expanded) if nonterminal_expanded else 0.0,
        by_turn=[by_turn.get(t, 0) for t in range(problem.turn_limit + 1)],
    )

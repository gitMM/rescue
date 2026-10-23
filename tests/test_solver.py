import copy

import pytest

from rescue.compiler import compile_scenario
from rescue.engine import WIN, Engine
from rescue.solver import analyse, astar, explore, narrate


def short(tiny, turn_limit):
    sc = copy.deepcopy(tiny)
    sc["turn_limit"] = turn_limit
    return sc


def test_astar_finds_the_optimal_escape(tiny):
    p = compile_scenario(tiny)
    result = astar(p)
    assert result.solved and not result.capped
    assert [op.name for op in result.plan] == ["take(card)", "keycard(d_cell:cell>hall,card)", "walk(d_yard:hall>yard)"]
    assert len(result.states) == len(result.plan) + 1


def test_unsolvable_scenarios_exhaust_the_search(tiny):
    sc = copy.deepcopy(tiny)
    sc["doors"][0]["clearance"] = 2  # the only keycard is level 1
    result = astar(compile_scenario(sc))
    assert not result.solved and not result.capped


def test_search_cap_is_reported(theta):
    result = astar(compile_scenario(theta), max_expansions=5)
    assert not result.solved and result.capped


def test_ability_restrictions(theta):
    p = compile_scenario(theta)
    every = frozenset(p.abilities)
    without_liquid = astar(p, allowed=every - {"liquid"})
    assert without_liquid.solved
    assert "liquid" not in {op.ability for op in without_liquid.plan}
    assert not astar(p, allowed=frozenset({"move", "liquid"})).solved


def test_astar_plan_replays_to_a_win(theta):
    p = compile_scenario(theta)
    e = Engine(p)
    result = astar(p, e)
    assert result.solved
    state, outcome, errors = e.replay([op.name for op in result.plan])
    assert errors == [] and outcome == WIN
    assert 8 <= result.turns <= p.turn_limit
    steps = narrate(p, result.plan, e)
    assert [s.turn for s in steps] == list(range(result.turns))
    assert all(s.text for s in steps)


def brute_force_paths(problem, engine, state, memo):
    """Count winning action sequences by plain recursion (reference implementation)."""
    if state in memo:
        return memo[state]
    total = 0
    for _, nxt, outcome in engine.successors(state):
        if outcome == WIN:
            total += 1
        elif outcome is None:
            total += brute_force_paths(problem, engine, nxt, memo)
    memo[state] = total
    return total


@pytest.mark.parametrize("turn_limit", [5, 6, 8])
def test_path_counting_matches_brute_force(tiny, turn_limit):
    p = compile_scenario(short(tiny, turn_limit))
    e = Engine(p)
    graph = explore(p, e, max_states=100_000)
    stats = analyse(p, graph)
    assert not stats.truncated and stats.winning_paths_exact
    assert stats.winning_paths == brute_force_paths(p, e, p.init, {})


def test_bfs_fastest_win_equals_astar(tiny):
    for limit in (5, 9, 12):
        p = compile_scenario(short(tiny, limit))
        stats = analyse(p, explore(p))
        assert stats.fastest_win == astar(p).turns


def test_graph_statistics_are_consistent(tiny):
    p = compile_scenario(tiny)
    graph = explore(p)
    stats = analyse(p, graph)
    assert stats.states == len(graph.states)
    assert stats.wins + sum(stats.losses.values()) + stats.winnable + stats.doomed + stats.unknown == stats.states
    assert stats.unknown == 0 and stats.doomed_ratio is not None
    assert sum(stats.by_turn) == stats.states
    # terminal states are never expanded, and nothing goes past the turn limit
    for i, s in enumerate(graph.states):
        assert s[p.turn] <= p.turn_limit
        if graph.outcome[i] is not None:
            assert graph.succ[i] == []


def test_truncated_exploration(theta):
    p = compile_scenario(theta)
    stats = analyse(p, explore(p, max_states=500))
    assert stats.truncated and stats.unknown > 0
    assert stats.states >= 500

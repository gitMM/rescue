import copy
import random

from conftest import guarded

from rescue.compiler import compile_scenario
from rescue.engine import WIN, Engine


def by_name(problem, name):
    return next(a for a in problem.actions if a.name == name)


def run(problem, engine, names):
    state, outcome, errors = engine.replay(names)
    assert not errors, errors
    return state, outcome


def test_initial_options_respect_locks(tiny):
    p = compile_scenario(tiny)
    e = Engine(p)
    names = {a.name for a in e.applicable(p.init)}
    assert "take(card)" in names and "wait" in names and "liquefy" in names
    assert not any(n.startswith("walk(") or n.startswith("keycard(") for n in names)
    assert "revert" not in names  # already in true form
    assert "rest" in names  # unobserved and below max energy


def test_rest_needs_privacy_and_headroom(tiny):
    sc = copy.deepcopy(tiny)
    sc["monster"]["energy"] = 3
    p = compile_scenario(sc)
    assert "rest" not in {a.name for a in Engine(p).applicable(p.init)}
    sc["monster"]["energy"] = 1
    p = compile_scenario(sc)
    assert "rest" in {a.name for a in Engine(p).applicable(p.init)}


def test_step_applies_effects_and_advances_the_clock(tiny):
    p = compile_scenario(tiny)
    e = Engine(p)
    state, outcome = e.step(p.init, by_name(p, "take(card)"))
    d = p.describe(state)
    assert outcome is None and d["has.card"] is True and d["turn"] == 1
    assert "keycard(d_cell:cell>hall,card)" in {a.name for a in e.applicable(state)}


def test_reaching_an_exit_wins(tiny):
    p = compile_scenario(tiny)
    state, outcome = run(p, Engine(p), ["take(card)", "keycard(d_cell:cell>hall,card)", "walk(d_yard:hall>yard)"])
    assert outcome == WIN
    assert p.describe(state)["monster.loc"] == "yard"


def test_running_out_of_time_loses(tiny):
    p = compile_scenario(tiny)
    _, outcome = run(p, Engine(p), ["wait"] * tiny["turn_limit"])
    assert outcome == "out_of_time"


def test_patrols_follow_the_clock(theta):
    p = compile_scenario(theta)
    e = Engine(p)
    ito = p.derived["npc.dr_ito.loc"]
    seen = []
    state = p.init
    for _ in range(7):
        seen.append(ito.domain.decode(ito.getter(state)))
        state, _ = e.step(state, by_name(p, "wait"))
    assert seen == theta["npcs"][0]["patrol"] + theta["npcs"][0]["patrol"][:1]


def test_being_seen_raises_the_alarm_and_guards_shoot(tiny):
    p = compile_scenario(guarded(tiny))
    e = Engine(p)
    # turn 2: the guard is in the yard, but the hall camera sees the creature in true form
    state, outcome = run(p, e, ["take(card)", "keycard(d_cell:cell>hall,card)"])
    assert outcome is None
    assert p.describe(state)["alarm"] == 1
    # waiting one more turn brings the guard back to the hall: shot on sight
    _, outcome = e.step(state, by_name(p, "wait"))
    assert outcome == "shot"


def test_liquid_form_slips_past_cameras_but_not_people(tiny):
    sc = guarded(tiny)
    sc["vents"].append({"id": "v1", "name": "a vent", "a": "cell", "b": "hall", "sealed": False, "sealable": True})
    p = compile_scenario(sc)
    e = Engine(p)
    state, outcome = run(p, e, ["liquefy", "wait", "crawl(v1:cell>hall)"])
    # turn 3: guard in hall (patrol index 0) sees a moving puddle; the camera does not care
    assert outcome is None
    assert p.describe(state)["alarm"] == 1


def test_world_rules_run_before_losses_and_after_the_clock(theta):
    p = compile_scenario(theta)
    e = Engine(p)
    state = p.init
    for _ in range(18):
        state, _ = e.step(state, by_name(p, "wait"))
    assert p.describe(state)["truck_arrived"] is True


def test_absorbing_takes_belongings_and_unlocks_mimicry(theta):
    p = compile_scenario(theta)
    e = Engine(p)
    state, outcome = run(p, e, ["wait", "absorb(dr_ito)"])
    d = p.describe(state)
    assert outcome is None
    assert d["npc.dr_ito.status"] == "absorbed"
    assert d["has.card_l2"] and d["has.pw_lab"]
    assert d["monster.energy"] == 4
    assert d["alarm"] == 2  # the cell camera saw it
    assert "mimic(dr_ito)" in {a.name for a in e.applicable(state)}


def test_replay_reports_impossible_steps(tiny):
    p = compile_scenario(tiny)
    _, _, errors = Engine(p).replay(["walk(d_yard:hall>yard)"])
    assert errors and "not possible" in errors[0]
    _, _, errors = Engine(p).replay(["fly"])
    assert errors and "unknown action" in errors[0]


def test_random_playouts_keep_every_variable_in_its_domain(theta):
    """Property test: whatever the creature does, the state stays well formed."""
    p = compile_scenario(theta)
    e = Engine(p)
    rng = random.Random(1234)
    for _ in range(150):
        state = p.init
        for _ in range(p.turn_limit):
            options = list(e.applicable(state))
            assert options, "wait is always available"
            state, outcome = e.step(state, rng.choice(options))
            for v in p.vars:
                code = state[v.index]
                if v.domain.kind == "enum":
                    assert 0 <= code < len(v.domain.values), v.name
                else:
                    assert v.domain.lo <= code <= v.domain.hi, v.name
            if outcome is not None:
                break
        else:
            raise AssertionError("a playout outlived the turn limit")

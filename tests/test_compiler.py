import copy

import pytest

from rescue.compiler import ScenarioError, compile_scenario


def test_tiny_compiles_to_expected_variables(tiny):
    p = compile_scenario(tiny)
    names = [v.name for v in p.vars]
    for expected in ["turn", "monster.loc", "monster.form", "monster.energy", "alarm", "lockdown",
                     "door.d_cell.locked", "door.d_yard.locked", "has.card"]:
        assert expected in names
    d = p.describe(p.init)
    assert d["monster.loc"] == "cell" and d["monster.form"] == "self"
    assert d["door.d_cell.locked"] is True and d["has.card"] is False
    assert d["monster.energy"] == 1 and d["turn"] == 0
    assert {"here.witnesses", "here.guards", "here.camera", "here.observed", "here.zone"} <= set(p.derived)


def test_grounding_respects_where_clauses(tiny):
    p = compile_scenario(tiny)
    names = {a.name for a in p.actions}
    assert "take(card)" in names
    assert "keycard(d_cell:cell>hall,card)" in names
    assert "keycard(d_cell:hall>cell,card)" in names
    assert not any(n.startswith("keycard(d_yard") for n in names)  # clearance 0: no reader
    assert "walk(d_yard:hall>yard)" in names


def test_actions_are_indexed_by_location(tiny):
    p = compile_scenario(tiny)
    cell = p.room_ids.index("cell")
    assert {a.name for a in p.by_loc[cell]} >= {"take(card)", "keycard(d_cell:cell>hall,card)"}
    assert {"wait", "rest", "revert", "liquefy"} <= {a.name for a in p.any_loc}


def test_movement_heuristic(tiny):
    p = compile_scenario(tiny)
    dist = {p.room_ids[i]: p.dist[i] for i in range(1, len(p.room_ids))}
    assert dist == {"cell": 2, "hall": 1, "yard": 0}


def test_empty_list_expansion_drops_useless_actions(theta):
    p = compile_scenario(theta)
    names = {a.name for a in p.actions}
    assert "phish(term_lab,dr_ito)" in names
    assert not any(n.startswith("phish(term_lab,guard_hale") for n in names)  # knows nothing, has no phone
    assert not any(n.startswith("con_item(tech_ruiz") for n in names)  # carries nothing


def test_custom_content_is_compiled(theta):
    p = compile_scenario(theta)
    assert "stow_away" in {a.name for a in p.actions}
    assert "supply_truck" in {r.name for r in p.rules}
    assert [l.id for l in p.losses][-1] == "incinerated"
    assert "truck_arrived" in p.var_index


def add_action(sc, **fields):
    spec = {"id": "custom", "ability": "move", "bind": [], "where": [], "pre": [], "eff": [], "text": ""}
    spec.update(fields)
    sc = copy.deepcopy(sc)
    sc["actions"].append(spec)
    return sc


@pytest.mark.parametrize(
    "fields, message",
    [
        ({"pre": ["power_level >= 2"]}, "unknown variable 'power_level'"),
        ({"eff": ["here.zone = 3"]}, "derived"),
        ({"eff": ["turn += 1"]}, "turn counter"),
        ({"eff": ["monster.form = dragon"]}, "'dragon' is not one of"),
        ({"pre": ["monster.form > self"]}, "only works on integer"),
        ({"eff": ["monster.loc += 1"]}, "needs an integer variable"),
        ({"pre": ["door.d_cell.locked == 3"]}, "expected true/false"),
        ({"bind": [{"name": "x", "from": "spaceships"}]}, "unknown placeholder"),
        ({"bind": [{"name": "r", "from": "rooms"}], "where": ["{r.colour} == red"]}, "no field 'colour'"),
        ({"pre": ["alarm >>= 1"]}, "custom"),
    ],
)
def test_compile_errors_are_reported(tiny, fields, message):
    with pytest.raises(ScenarioError) as exc:
        compile_scenario(add_action(tiny, **fields))
    assert any(message in e for e in exc.value.errors), exc.value.errors


def test_compile_collects_many_errors(tiny):
    sc = add_action(tiny, pre=["nope == 1"])
    sc["rules"].append({"id": "r", "bind": [], "where": [], "when": ["also_nope == 1"], "then": [], "text": ""})
    sc["goal_extra"] = ["has.unknown == true"]
    with pytest.raises(ScenarioError) as exc:
        compile_scenario(sc)
    text = "\n".join(exc.value.errors)
    assert "nope" in text and "also_nope" in text and "goal_extra" in text


def test_structure_errors_stop_compilation(tiny):
    sc = copy.deepcopy(tiny)
    sc["doors"][0]["a"] = "attic"
    with pytest.raises(ScenarioError, match="attic"):
        compile_scenario(sc)

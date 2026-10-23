import pytest

from rescue import dsl
from rescue.dsl import DSLError, Operand

ENV = {
    "p": {"id": "d1:a>b", "via": "d1", "from": "a", "to": "b", "clearance": 2},
    "n": {"id": "ito", "name": "Dr. Ito", "carries": ["badge", "pass"], "knows": [], "clearance": 3},
    "t": {"id": "term", "controls": ["d1", "cam"], "password": None},
    "security": {"max_alarm": 5},
}


def one(text, env=ENV):
    conds = dsl.instantiate_condition(text, env)
    assert len(conds) == 1
    return conds[0]


def test_placeholders_fill_variable_names_and_values():
    cond = one("door.{p.via}.locked == false")
    (atom,) = cond.atoms
    assert atom.lhs == Operand("name", "door.d1.locked")
    assert atom.op == "=="
    assert atom.rhs == Operand("lit", False)
    assert cond.text() == "door.d1.locked == false"


def test_exact_placeholders_keep_their_type():
    (atom,) = one("alarm < {security.max_alarm}").atoms
    assert atom.rhs == Operand("lit", 5)
    (atom,) = one("monster.loc == {p.from}").atoms
    assert atom.rhs == Operand("lit", "a")


def test_object_placeholder_renders_its_id():
    (atom,) = one("monster.form == {n}").atoms
    assert atom.rhs.value == "ito"


def test_disjunction_and_dotted_references():
    cond = one("npc.{n.id}.loc == monster.loc || alarm >= 2")
    assert [a.op for a in cond.atoms] == ["==", ">="]
    assert cond.atoms[0].rhs == Operand("name", "monster.loc")


def test_in_with_literal_list_and_list_placeholder():
    (atom,) = one("monster.loc in [a, b, none]").atoms
    assert atom.op == "in"
    assert [o.value for o in atom.rhs] == ["a", "b", None]
    (atom,) = one("{p.via} in {t.controls}").atoms
    assert [o.value for o in atom.rhs] == ["d1", "cam"]
    (atom,) = one("x not in [1, 2]").atoms
    assert atom.op == "not in"


def test_list_placeholder_expands_statements():
    effects = dsl.instantiate_effect("has.{n.carries} = true", ENV)
    assert [e.text() for e in effects] == ["has.badge = true", "has.pass = true"]
    assert dsl.instantiate_effect("has.{n.knows} = true", ENV) == []
    conds = dsl.instantiate_condition("has.{n.carries} == false", ENV)
    assert [c.text() for c in conds] == ["has.badge == false", "has.pass == false"]


def test_conditional_effects():
    (eff,) = dsl.instantiate_effect({"if": ["alarm >= 1"], "then": ["alarm += 2"], "else": []}, ENV)
    assert isinstance(eff, dsl.ConditionalEffect)
    assert eff.text() == "if alarm >= 1 then alarm += 2"
    assert dsl.instantiate_effect({"if": [], "then": ["has.{n.knows} = true"], "else": []}, ENV) == []


def test_effect_operators_and_negative_numbers():
    (e,) = dsl.instantiate_effect("monster.energy -= 1", ENV)
    assert (e.target, e.op, e.value.value) == ("monster.energy", "-=", 1)
    (e,) = dsl.instantiate_effect("x = -3", ENV)
    assert e.value.value == -3
    (e,) = dsl.instantiate_effect("alarm+=1", ENV)
    assert (e.target, e.op) == ("alarm", "+=")


@pytest.mark.parametrize(
    "bad",
    ["alarm >>= 2", "alarm", "== 3", "x = ", "alarm in 3", "a == b c", "", "x ~ 1"],
)
def test_malformed_conditions_raise(bad):
    with pytest.raises(DSLError):
        dsl.instantiate_condition(bad, ENV)


def test_malformed_effects_raise():
    for bad in ["alarm == 1", "alarm", "1 = alarm", {"then": ["x = 1"]}, {"if": [], "then": [{"if": [], "then": []}]}]:
        with pytest.raises(DSLError):
            dsl.instantiate_effect(bad, ENV)


def test_unknown_placeholders_and_two_lists_raise():
    with pytest.raises(DSLError, match="unknown placeholder"):
        dsl.instantiate_condition("{q.id} == 1", ENV)
    with pytest.raises(DSLError, match="no field"):
        dsl.instantiate_condition("{p.nope} == 1", ENV)
    env = dict(ENV, m={"id": "m", "carries": ["x"]})
    with pytest.raises(DSLError, match="only one list"):
        dsl.instantiate_effect("has.{n.carries} = {m.carries}", env)


def test_static_evaluation():
    assert dsl.eval_static(one("{p.clearance} >= 2"))
    assert not dsl.eval_static(one("{n.clearance} < {p.clearance}"))
    assert dsl.eval_static(one("{t.password} == none"))
    # ordering comparisons with None or strings are simply false, never errors
    assert not dsl.eval_static(one("{t.password} >= 1"))
    assert dsl.eval_static(one("{p.via} in {t.controls} || {p.clearance} > 9"))


def test_render_text_uses_names_and_lists():
    text = dsl.render_text("{n.name} drops {n.carries}; {n.knows}; {missing}", ENV)
    assert text == "Dr. Ito drops badge, pass; nothing; {missing}"


def test_placeholder_roots():
    assert dsl.placeholder_roots(["{p.via} in {t.controls}", {"if": ["{n.id} == x"]}]) == {"p", "t", "n"}

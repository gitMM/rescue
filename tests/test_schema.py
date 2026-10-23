import copy

from rescue.library import load_library
from rescue.schema import SCENARIO_SCHEMA, check_structure

LIB = load_library()

# Keywords structured outputs reject (or that we deliberately keep out of the API schema).
FORBIDDEN = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
             "minLength", "maxLength", "pattern", "minItems", "maxItems", "uniqueItems", "format"}


def walk(node, path="$"):
    if isinstance(node, dict):
        yield path, node
        for k, v in node.items():
            yield from walk(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk(v, f"{path}[{i}]")


def check(sc):
    return check_structure(sc, LIB.disableable, LIB.all_ids)


def test_schema_fits_structured_outputs_subset():
    for path, node in walk(SCENARIO_SCHEMA):
        assert not FORBIDDEN & set(node), path
        if node.get("type") == "object":
            assert node.get("additionalProperties") is False, path
            assert sorted(node["required"]) == sorted(node["properties"]), path


def test_bundled_scenarios_are_well_formed(tiny, theta):
    assert check(tiny) == []
    assert check(theta) == []


def test_schema_violations_are_reported(tiny):
    sc = copy.deepcopy(tiny)
    del sc["rooms"][0]["zone"]
    sc["turn_limit"] = "soon"
    errors = check(sc)
    assert any("zone" in e for e in errors)
    assert any("turn_limit" in e for e in errors)


def test_cross_reference_errors(theta):
    sc = copy.deepcopy(theta)
    sc["doors"][0]["b"] = "nowhere"
    sc["npcs"][0]["patrol"].append("attic")
    sc["npcs"][1]["carries"].append("ghost_item")
    sc["terminals"][0]["controls"].append("dr_ito")
    sc["terminals"][1]["password"] = "nope"
    sc["items"][0]["clearance"] = None
    sc["vents"].append(dict(sc["vents"][0], id="cell"))
    errors = "\n".join(check(sc))
    for needle in ["'nowhere' is not a room", "patrol room 'attic'", "'ghost_item' is not an item",
                   "controls 'dr_ito'", "password 'nope'", "keycards need a clearance", "duplicate id 'cell'"]:
        assert needle in errors


def test_global_rules(tiny):
    sc = copy.deepcopy(tiny)
    for r in sc["rooms"]:
        r["exit"] = False
    sc["monster"]["energy"] = 9
    sc["security"]["seal_vents_at"] = 7
    sc["rooms"][0]["id"] = "Bad-ID"
    errors = "\n".join(check(sc))
    assert "no room has exit" in errors
    assert "energy" in errors
    assert "seal_vents_at" in errors
    assert "must match" in errors


def test_custom_ids_and_disabling(tiny):
    sc = copy.deepcopy(tiny)
    sc["actions"].append({"id": "walk", "ability": "move", "bind": [], "where": [], "pre": [], "eff": [], "text": ""})
    sc["losses"].append({"id": "shot", "when": ["alarm >= 1"], "text": ""})
    sc["disabled_core"] = ["not_a_rule", "shot"]
    errors = "\n".join(check(sc))
    assert "action id 'walk' is already used" in errors
    assert "loss id 'shot' is already used" in errors
    assert "'not_a_rule' is not a core action or rule" in errors
    assert "'shot' is not a core action or rule" in errors  # losses cannot be disabled
    sc["disabled_core"] = ["walk"]
    sc["losses"] = []
    assert check(sc) == []  # replacing a disabled core action is fine


def test_custom_var_rules(tiny):
    sc = copy.deepcopy(tiny)
    sc["vars"] = [
        {"name": "power", "kind": "int", "init": 9, "min": 0, "max": 3, "values": None},
        {"name": "alarm", "kind": "bool", "init": False, "min": None, "max": None, "values": None},
        {"name": "mode", "kind": "enum", "init": "x", "min": None, "max": None, "values": ["a"]},
    ]
    errors = "\n".join(check(sc))
    assert "var power" in errors
    assert "reserved name" in errors
    assert "var mode" in errors

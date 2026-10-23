import copy

from conftest import guarded

from rescue.validate import Gates, format_report, repair_brief, validate_scenario


def gate(report, name):
    return next(g for g in report["gates"] if g["name"] == name)


def test_example_scenario_is_accepted(theta_report, fast_gates):
    report = theta_report
    assert report["valid"] and report["solvable"] and report["accepted"], format_report(report)
    assert all(g["passed"] for g in report["gates"])
    assert len(report["abilities"]["routes"]) >= 2
    assert report["abilities"]["solo_escapes"] == []
    assert report["graph"]["states"] >= fast_gates.min_states
    assert set(report["graph"]["losses"]) - {"out_of_time"}
    sol = report["solution"]
    assert sol["turns"] == len(sol["steps"]) and len(sol["abilities"]) >= 2


def test_trivial_scenario_is_rejected_with_reasons(tiny, fast_gates):
    report = validate_scenario(tiny, fast_gates)
    assert report["valid"] and report["solvable"] and not report["accepted"]
    failed = {g["name"] for g in report["gates"] if not g["passed"]}
    assert {"min_solution_turns", "abilities_in_solution", "requires_combination", "danger", "min_states", "min_routes"} <= failed
    assert report["abilities"]["solo_escapes"] == ["keycard"]
    assert any("One ability is enough" in d for d in report["diagnostics"])


def test_danger_gate_ignores_the_clock_but_counts_guards(tiny, fast_gates):
    assert not gate(validate_scenario(tiny, fast_gates), "danger")["passed"]
    report = validate_scenario(guarded(tiny), fast_gates)
    assert gate(report, "danger")["passed"]
    assert "shot" in gate(report, "danger")["detail"]


def test_unsolvable_scenario_explains_the_blocker(tiny, fast_gates):
    sc = copy.deepcopy(tiny)
    sc["doors"][0]["clearance"] = 2
    report = validate_scenario(sc, fast_gates)
    assert report["valid"] and not report["solvable"] and not report["accepted"]
    text = "\n".join(report["diagnostics"])
    assert "no exit is reachable" in text
    assert "Blocked way into Hall" in text
    assert "door.d_cell.locked == false" in text


def test_too_tight_turn_limit_is_diagnosed(theta, fast_gates):
    sc = copy.deepcopy(theta)
    sc["turn_limit"] = 7
    report = validate_scenario(sc, fast_gates)
    assert not report["solvable"]
    assert any("raise turn_limit" in d for d in report["diagnostics"]), report["diagnostics"]


def test_turn_slack_gate(theta, theta_report):
    fastest = theta_report["solution"]["turns"]
    sc = copy.deepcopy(theta)
    sc["turn_limit"] = fastest + 1
    report = validate_scenario(sc, Gates(explore_cap=2000, search_cap=60_000, min_routes=1))
    assert report["solvable"]
    assert not gate(report, "turn_slack")["passed"]


def test_malformed_scenarios_are_invalid(theta, fast_gates):
    sc = copy.deepcopy(theta)
    sc["npcs"][0]["patrol"] = ["moon_base"]
    sc["actions"].append({"id": "fly", "ability": "wings", "bind": [], "where": [], "pre": ["altitude > 3"], "eff": [], "text": ""})
    report = validate_scenario(sc, fast_gates)
    assert not report["valid"] and not report["accepted"]
    assert any("moon_base" in e for e in report["errors"])
    report = validate_scenario({"not": "a scenario"}, fast_gates)
    assert not report["valid"] and report["errors"]


def test_reports_render(theta_report, tiny, fast_gates):
    text = format_report(theta_report)
    assert text.startswith("[ACCEPTED] Site Theta")
    assert "route:" in text and "fastest escape" in text
    bad = validate_scenario(tiny, fast_gates)
    brief = repair_brief(bad)
    assert "Failed quality gates:" in brief and "requires_combination" in brief
    assert "Current fastest escape" in brief


def _mutate(rng, node):
    """Apply one random corruption somewhere inside a JSON-like structure."""
    if isinstance(node, dict) and node:
        key = rng.choice(sorted(node))
        roll = rng.random()
        if roll < 0.2:
            del node[key]
        elif roll < 0.4 or not isinstance(node[key], (dict, list)):
            node[key] = rng.choice([None, -7, 999, "", "garbage == ==", ["x"], {"a": 1}, True, "{q.id} = 1", "alarm"])
        else:
            _mutate(rng, node[key])
    elif isinstance(node, list) and node:
        i = rng.randrange(len(node))
        if isinstance(node[i], (dict, list)) and rng.random() < 0.7:
            _mutate(rng, node[i])
        else:
            node[i] = rng.choice([None, 3, "monster.loc ==", {"if": "x"}, "nowhere"])


def test_validator_never_crashes_on_corrupted_scenarios(theta):
    """LLM output is untrusted: any corruption must produce a report, never an exception."""
    import random

    rng = random.Random(99)
    gates = Gates(explore_cap=200, search_cap=2000)
    outcomes = {"invalid": 0, "valid": 0}
    for _ in range(120):
        sc = copy.deepcopy(theta)
        for _ in range(rng.randint(1, 3)):
            _mutate(rng, sc)
        report = validate_scenario(sc, gates)
        assert isinstance(report["errors"], list)
        assert not any("could not compile the scenario" in e for e in report["errors"]), report["errors"]
        outcomes["valid" if report["valid"] else "invalid"] += 1
    assert outcomes["invalid"] > 0 and outcomes["valid"] > 0

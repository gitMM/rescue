import json

from conftest import THETA_PATH, TINY_PATH

from rescue.cli import main

FAST = ["--gate", "explore_cap=5000", "--gate", "search_cap=60000"]


def test_validate_exit_codes(capsys):
    assert main(["validate", str(THETA_PATH), "--no-steps", *FAST]) == 0
    assert "[ACCEPTED] Site Theta" in capsys.readouterr().out
    assert main(["validate", str(TINY_PATH), *FAST]) == 1
    assert "[REJECTED] Tiny Cell" in capsys.readouterr().out


def test_validate_json_and_gate_overrides(capsys):
    assert main(["validate", str(TINY_PATH), "--json", "--gate", "min_solution_turns=2", "--gate", "require_combination=false", *FAST]) == 1
    report = json.loads(capsys.readouterr().out)
    names = {g["name"]: g["passed"] for g in report["gates"]}
    assert names["min_solution_turns"] is True and "requires_combination" not in names


def test_schema_and_prompt_commands(capsys):
    assert main(["schema"]) == 0
    assert json.loads(capsys.readouterr().out)["type"] == "object"
    assert main(["prompt", "--brief", "4"]) == 0
    assert "Design a new escape scenario" in capsys.readouterr().out
    assert main(["prompt"]) == 0
    assert "Statement language" in capsys.readouterr().out


def test_generate_dry_run(tmp_path, capsys):
    code = main(["generate", "--fake", str(TINY_PATH), str(THETA_PATH), "--out", str(tmp_path), "--rounds", "2", *FAST])
    assert code == 0
    out = capsys.readouterr().out
    assert "site_theta after 2 round(s): accepted" in out
    assert (tmp_path / "accepted" / "site_theta_s0.json").exists()

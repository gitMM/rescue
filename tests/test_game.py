import io
import json
from pathlib import Path

from conftest import THETA_PATH, TINY_PATH, load

from rescue.cli import main
from rescue.compiler import compile_scenario
from rescue.game import Game, choose_scenario, find_scenarios

HALCYON_PATH = Path(__file__).parent.parent / "scenarios" / "halcyon_tower.json"
HALCYON_FASTEST = [
    "wait", "wait", "wait", "absorb vasquez", "keycard", "mimic", "walk lab", "walk corridor",
    "keycard lobby", "con chen elevator", "walk elevator", "walk street",
]  # fmt: skip


def play(path: Path, *lines: str) -> tuple[Game, str]:
    out: list[str] = []
    game = Game(compile_scenario(load(path)), out=out.append, width=1000)
    game.intro()
    game.show(full=True)
    for line in lines:
        assert game.handle(line)
    return game, "\n".join(out)


def test_numbers_names_and_words_all_pick_actions():
    game, out = play(TINY_PATH, "take card", "keycard(d_cell:cell>hall,card)")
    assert game.outcome is None and "== Hall ==" in out
    number = [op.name for op in game.options()].index("walk(d_yard:hall>yard)") + 1
    game.handle(str(number))
    assert game.outcome == "win"


def test_fastest_halcyon_escape_with_narration():
    game, out = play(HALCYON_PATH, *HALCYON_FASTEST)
    assert game.outcome == "win"
    assert "Footsteps: Dr. Elena Vasquez will be here next turn." in out
    assert "Dr. Elena Vasquez comes in." in out
    assert "The alarm rises to 2." in out  # absorbed in front of the tank camera
    assert "*** You escape to Street Level! ***" in out
    assert "Turns taken: 12 of 26 (the fastest escape takes 12)" in out
    assert "Abilities used: keycard, absorb, disguise, con" in out


def test_impossible_actions_are_explained():
    game, out = play(TINY_PATH, "walk(d_cell:cell>hall)", "walk hall")
    assert game.moves == []
    assert out.count("Not possible right now: Walk from Cell to Hall.") == 2
    assert "- cell door is locked" in out

    game, out = play(HALCYON_PATH, "absorb vasquez")
    assert "- Dr. Elena Vasquez isn't here" in out


def test_ambiguous_words_list_the_candidates():
    game, out = play(HALCYON_PATH, "wait", "wait", "wait", "vasquez")
    assert game.moves[-1].name == "wait"
    assert "'vasquez' could mean:" in out
    assert "Engulf Dr. Elena Vasquez" in out and "Knock Dr. Elena Vasquez out cold" in out


def test_losing_then_undo_and_restart():
    game, out = play(HALCYON_PATH, "liquefy", "crawl hvac", "crawl break", "wait")
    assert game.outcome == "poisoned"
    assert "*** Poisoned: The nerve agent soaks into every cell" in out
    assert out.endswith("The game is over. Type 'undo', 'restart' or 'quit'.")
    game.handle("undo")
    assert game.outcome is None and game.val("monster.loc") == "hvac" and game.undos == 1
    game.handle("restart")
    assert game.state == game.p.init and game.moves == [] and game.undos == 0
    game.handle("undo")
    assert game.state == game.p.init


def test_hints_follow_a_fastest_escape_and_spot_doomed_states():
    game, out = play(TINY_PATH, "hint")
    assert "Hint: [" in out and "Pick up a dropped keycard. (the fastest way out from here takes 3 more turns)" in out

    game, out = play(TINY_PATH, *["wait"] * 10, "hint")
    assert "no way out from here any more" in out
    assert game.hints == 1


def test_map_and_people_screens():
    game, out = play(HALCYON_PATH, "map", "people")
    assert " * Specimen Tank Room" in out
    assert "the roof hatch to Rooftop Helipad (EXIT) [locked, clearance 4]" in out
    assert "Dr. Elena Vasquez (lead virologist, clearance 2, has a phone)" in out
    assert "BSL-4 Lab -> BSL-4 Lab -> Decontamination Airlock -> Specimen Tank Room" in out
    assert "carries Dr. Vasquez's lab badge; knows the lab terminal password" in out


def test_colour_output():
    out: list[str] = []
    game = Game(compile_scenario(load(HALCYON_PATH)), out=out.append, color=True)
    game.intro()
    for line in HALCYON_FASTEST:
        game.handle(line)
    assert game.outcome == "win"
    assert "\x1b[1;32m*** You escape to Street Level! ***\x1b[0m" in out
    assert "Nobody else is here." in out  # unstyled lines stay plain


def test_quit_and_end_of_input():
    game, _ = play(TINY_PATH)
    assert game.handle("quit") is False
    lines = iter(["take card"])

    def read(_prompt: str) -> str:
        try:
            return next(lines)
        except StopIteration:
            raise EOFError from None

    assert Game(compile_scenario(load(TINY_PATH)), out=lambda _: None).run(read) == 1


def test_cli_game_plays_piped_input(monkeypatch, capsys):
    script = "take card\nkeycard\nwalk yard\nquit\n"
    monkeypatch.setattr("sys.stdin", io.StringIO(script))
    assert main(["game", str(TINY_PATH)]) == 0
    out = capsys.readouterr().out
    assert "> take card" in out  # piped input is echoed
    assert "*** You escape to Yard! ***" in out
    assert "\x1b[" not in out


def test_cli_game_rejects_broken_scenarios(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"id": "bad"}))
    assert main(["game", str(bad)]) == 2
    assert "error: schema" in capsys.readouterr().out


def test_choosing_a_scenario(tmp_path):
    (tmp_path / "accepted").mkdir()
    (tmp_path / "accepted" / "a_s1.json").write_text(TINY_PATH.read_text())
    (tmp_path / "accepted" / "a_s1.report.json").write_text("{}")
    paths = find_scenarios(tmp_path)
    assert paths[0].name == "a_s1.json" and THETA_PATH.name in [p.name for p in paths]
    assert not any(p.name.endswith(".report.json") for p in paths)

    out: list[str] = []
    answers = iter(["9", "1"])
    assert choose_scenario(paths, lambda _: next(answers), out.append) == paths[0]
    assert "Tiny Cell (a_s1.json)" in out[1] and out[-1].startswith("Type a number")
    assert choose_scenario(paths, lambda _: "q", out.append) is None

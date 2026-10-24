"""Play a scenario as a turn-based text game.

The game runs on the same compiled problem and engine as the validator, so
every rule, patrol and loss works exactly the way A* sees it. Each turn
shows where the monster is, who and what is around it, and a numbered menu
of every action that is possible right now. Pick one by number, by its
grounded name (``absorb(dr_ito)``) or by a few words of it (``absorb ito``).
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path
from typing import Any, Callable

from .compiler import CAtom, CCond, Operator, Problem
from .engine import WIN, Engine
from .library import example_paths
from .solver import astar

HELP = """\
Pick an action by its number, its name (absorb(dr_ito)) or a few words of it (absorb ito).
  look, l       describe where you are again
  map, m        every room and how the rooms connect
  people, p     who is where over the next few turns, and what they carry
  hint          the next move of a fastest escape from here
  undo, u       take back your last move
  restart       start the scenario again
  help, ?       this help
  quit, q       leave the game"""

COMMANDS = {
    "look": "look", "l": "look",
    "map": "map", "m": "map",
    "people": "people", "p": "people", "who": "people",
    "hint": "hint",
    "undo": "undo", "u": "undo",
    "restart": "restart",
    "help": "help", "?": "help",
    "quit": "quit", "q": "quit", "exit": "quit",
}  # fmt: skip

STATUS_TEXT = {
    "normal": "up and about",
    "unconscious": "unconscious",
    "absorbed": "already absorbed",
    "found": "found and taken away",
}
FORECAST_TURNS = 5  # "people" shows this many stops of each patrol
HINT_BUDGET = 60_000  # A* expansions per hint, about three seconds at worst
ANSI = {"bold": "1", "dim": "2", "red": "31", "green": "32", "yellow": "33", "cyan": "36"}


class Game:
    """One play-through of a compiled scenario, driven one line of input at a time."""

    def __init__(self, problem: Problem, out: Callable[[str], None] = print, width: int = 88, color: bool = False):
        self.p = problem
        self.sc = problem.scenario
        self.engine = Engine(problem)
        self.out = out
        self.width = width
        self.color = color
        self.names = {e["id"]: e["name"] for coll in ("rooms", "doors", "vents", "cameras", "terminals", "npcs", "items") for e in self.sc[coll]}
        self.rooms = {r["id"]: r for r in self.sc["rooms"]}
        self.loss_text = {loss.id: loss.text for loss in problem.losses}
        self.by_name = {op.name: op for op in problem.actions}
        self.rank: dict[str, int] = {}
        for op in problem.actions:
            self.rank.setdefault(op.ability, len(self.rank))
        self.words = {op.index: tuple(re.findall(r"[a-z0-9]+", f"{op.name} {op.text}".lower())) for op in problem.actions}
        self._fastest: int | None = None
        self.reset()

    # ------------------------------------------------------------- state
    def reset(self) -> None:
        self.history: list[tuple[tuple, str | None]] = [(self.p.init, None)]
        self.moves: list[Operator] = []
        self.hints = 0
        self.undos = 0

    @property
    def state(self) -> tuple:
        return self.history[-1][0]

    @property
    def outcome(self) -> str | None:
        return self.history[-1][1]

    def val(self, name: str, state: tuple | None = None) -> Any:
        var = self.p.var(name)
        return var.domain.decode(var.getter(self.state if state is None else state))

    def options(self) -> list[Operator]:
        """Every action possible now, grouped by ability; menu numbers index this list."""
        if self.outcome is not None:
            return []
        return sorted(self.engine.applicable(self.state), key=lambda op: (self.rank[op.ability], op.index))

    def fastest(self) -> int | None:
        if self._fastest is None:
            self._fastest = astar(self.p, self.engine).turns
        return self._fastest

    # ------------------------------------------------------------ output
    def style(self, text: str, *styles: str) -> str:
        codes = [ANSI[s] for s in styles if s]
        if not self.color or not codes:
            return text
        return f"\x1b[{';'.join(codes)}m{text}\x1b[0m"

    def say(self, text: str = "", *styles: str, indent: int = 0, hang: int = 0) -> None:
        lines = textwrap.wrap(text, self.width, initial_indent=" " * indent, subsequent_indent=" " * (indent + hang)) or [""]
        for line in lines:
            self.out(self.style(line, *styles))

    def form_text(self, form: str) -> str:
        if form == "self":
            return "in your true form"
        if form == "liquid":
            return "liquid"
        return f"disguised as {self.names[form]}"

    # -------------------------------------------------------------- loop
    def run(self, read: Callable[[str], str] = input) -> int:
        """Play until the player quits or input runs out; 0 if the monster ends up escaped."""
        self.intro()
        self.show(full=True)
        while True:
            try:
                line = read("> ")
            except (EOFError, KeyboardInterrupt):
                self.out("")
                break
            if not self.handle(line):
                break
        return 0 if self.outcome == WIN else 1

    def handle(self, line: str) -> bool:
        """Process one line of input; returns False when the player quits."""
        text = line.strip()
        if not text:
            return True
        low = text.lower()
        cmd = COMMANDS.get(low)
        if cmd == "quit":
            return False
        if cmd:
            getattr(self, f"cmd_{cmd}")()
        elif self.outcome is not None:
            self.say("The game is over. Type 'undo', 'restart' or 'quit'.", "dim")
        elif low.isdigit():
            options = self.options()
            if 1 <= int(low) <= len(options):
                self.act(options[int(low) - 1])
            else:
                self.say(f"Pick a number from 1 to {len(options)}.")
        else:
            self.act_by_words(text)
        return True

    # ------------------------------------------------------------ screens
    def intro(self) -> None:
        sc = self.sc
        exits = [r["name"] for r in sc["rooms"] if r["exit"]]
        sec = sc["security"]
        self.out("")
        self.say(f"=== {sc['title'].upper()} ===", "bold")
        self.say(sc["premise"])
        self.out("")
        self.say(f"You are the creature. Escape to {_join(exits, 'or')} within {sc['turn_limit']} turns.")
        if sc["goal_extra"]:
            self.say(f"It only counts as an escape if {_join(sc['goal_extra'])}.")
        self.say(
            f"Security: badges are revoked at alarm {sec['revoke_badges_at']}, vents seal at {sec['seal_vents_at']}, "
            f"and at {sec['max_alarm']} the facility locks down and purges itself {sec['purge_turns']} turns later."
        )
        self.say("Type 'help' for commands.", "dim")

    def show(self, full: bool = False) -> None:
        self.look(full)
        self.show_menu()

    def look(self, full: bool) -> None:
        s, sc = self.state, self.sc
        here = self.val("monster.loc")
        room = self.rooms[here]
        self.out("")
        self.say(f"== {room['name']} ==", "bold", "cyan")
        if full:
            self.say(room["description"])
        self.status_line()
        if sc["vars"]:
            self.say(" | ".join(f"{v['name'].replace('_', ' ')}: {_value_text(self.val(v['name']))}" for v in sc["vars"]))

        people = []
        for n in sc["npcs"]:
            if self.val(f"npc.{n['id']}.loc") != here:
                continue
            if self.val(f"npc.{n['id']}.status") == "unconscious":
                people.append(f"{n['name']} (unconscious)")
            else:
                people.append(f"{n['name']} ({n['role']}{', armed' if n['armed'] else ''})")
        self.say(f"Here: {_join(people)}." if people else "Nobody else is here.", "yellow" if people else "")

        cams = []
        for c in sc["cameras"]:
            if c["room"] == here:
                cams.append(f"{c['name']} is " + ("watching" if self.val(f"cam.{c['id']}.active") else "looping old footage"))
        if cams:
            self.say(f"Cameras: {_join(cams)}.")
        self.say("Ways out:")
        for line in self.passages(here, s):
            self.say(line, indent=2, hang=2)
        for t in sc["terminals"]:
            if t["room"] == here:
                controls = [self.names[c] for c in t["controls"]] + (["the alarm"] if t["alarm_control"] else [])
                state = "yours" if self.val(f"term.{t['id']}.hacked") else "locked"
                self.say(f"Terminal: {t['name']} (security {t['security']}, {state}) controls {_join(controls) or 'nothing'}.")
        floor = [i["name"] for i in sc["items"] if i["room"] == here and not self.val(f"has.{i['id']}")]
        if floor:
            self.say(f"On the floor: {_join(floor)}.")
        carry = [i["name"] for i in sc["items"] if i["kind"] != "intel" and self.val(f"has.{i['id']}")]
        know = [i["name"] for i in sc["items"] if i["kind"] == "intel" and self.val(f"has.{i['id']}")]
        if carry:
            self.say(f"You carry {_join(carry)}.")
        if know:
            self.say(f"You know {_join(know)}.")

        turn = self.val("turn")
        coming = []
        for n in sc["npcs"]:
            patrol = n["patrol"]
            if self.val(f"npc.{n['id']}.status") == "normal" and self.val(f"npc.{n['id']}.loc") != here and patrol[(turn + 1) % len(patrol)] == here:
                coming.append(n["name"] + (" (armed)" if n["armed"] else ""))
        if coming:
            self.say(f"Footsteps: {_join(coming)} will be here next turn.", "yellow")

    def status_line(self) -> None:
        sec = self.sc["security"]
        alarm, lockdown = self.val("alarm"), self.val("lockdown")
        bar = "#" * alarm + "-" * (sec["max_alarm"] - alarm)
        parts = [
            f"Turn {self.val('turn')}/{self.sc['turn_limit']}",
            f"Energy {self.val('monster.energy')}/{self.sc['monster']['max_energy']}",
            f"Alarm [{bar}] {alarm}/{sec['max_alarm']}",
        ]
        if lockdown:
            parts.append(f"LOCKDOWN: purge in {sec['purge_turns'] - lockdown}")
        elif alarm >= sec["seal_vents_at"]:
            parts.append("vents sealed")
        elif alarm >= sec["revoke_badges_at"]:
            parts.append("badges revoked")
        danger = "red" if lockdown else "yellow" if alarm >= sec["revoke_badges_at"] else ""
        self.say(" | ".join(parts), danger)
        self.say(f"You are {self.form_text(self.val('monster.form'))}.")

    def passages(self, room_id: str, state: tuple) -> list[str]:
        lines = []
        for kind, coll in (("door", self.sc["doors"]), ("vent", self.sc["vents"])):
            for e in coll:
                if room_id not in (e["a"], e["b"]):
                    continue
                other = e["b"] if e["a"] == room_id else e["a"]
                line = f"{e['name']}{' (vent)' if kind == 'vent' else ''} to {self.names[other]}"
                if self.rooms[other]["exit"]:
                    line += " (EXIT)"
                if kind == "door" and self.val(f"door.{e['id']}.locked", state):
                    line += f" [locked, clearance {e['clearance']}]" if e.get("clearance") else " [locked]"
                elif kind == "vent" and self.val(f"vent.{e['id']}.sealed", state):
                    line += " [sealed]"
                lines.append(line)
        return lines

    def show_menu(self) -> None:
        self.out("")
        if self.outcome is not None:
            self.say("The game is over. Type 'undo', 'restart' or 'quit'.", "dim")
            return
        options = self.options()
        if not options:
            self.say("There is nothing you can do. Type 'undo' or 'restart'.", "red")
            return
        self.say("What do you do?", "bold")
        group = None
        for n, op in enumerate(options, 1):
            if op.ability != group:
                group = op.ability
                self.say(group, "dim", indent=2)
            self.say(f"{n:>3}  {op.text or op.name}", indent=2, hang=5)

    # ------------------------------------------------------------ actions
    def act(self, op: Operator) -> None:
        before = self.state
        reactions = [r.text for r in self.engine.fired_rules(before, op) if r.text]
        after, outcome = self.engine.step(before, op)
        self.history.append((after, outcome))
        self.moves.append(op)
        self.out("")
        self.say(op.text or op.name)
        for r in reactions:
            self.say(f"! {r}", "yellow", indent=2, hang=2)
        alarm = self.val("alarm", after)
        if alarm != self.val("alarm", before):
            rose = alarm > self.val("alarm", before)
            self.say(f"The alarm {'rises' if rose else 'drops'} to {alarm}.", "yellow" if rose else "", indent=2)
        if outcome is not None:
            self.finish(outcome)
            return
        moved = self.val("monster.loc", before) != self.val("monster.loc", after)
        if not moved:
            self.comings_and_goings(before, after)
        self.show(full=moved)

    def comings_and_goings(self, before: tuple, after: tuple) -> None:
        here = self.val("monster.loc", after)
        for n in self.sc["npcs"]:
            key = f"npc.{n['id']}"
            if self.val(f"{key}.status", before) != "normal" or self.val(f"{key}.status", after) != "normal":
                continue
            was, now = self.val(f"{key}.loc", before), self.val(f"{key}.loc", after)
            if was != here and now == here:
                self.say(f"{n['name']} comes in.", "yellow", indent=2)
            elif was == here and now != here:
                self.say(f"{n['name']} leaves for {self.names[now]}.", indent=2)

    def act_by_words(self, text: str) -> None:
        exact = self.by_name.get(text)
        if exact is not None:
            if exact.holds(self.state):
                self.act(exact)
            else:
                self.why_not([exact])
            return
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        if not tokens:
            self.say("Type a number from the list, a few words of an action, or 'help'.")
            return
        options = self.options()
        hits = [op for op in options if self.matches(op, tokens)]
        if len(hits) == 1:
            self.act(hits[0])
            return
        if hits:
            number = {op.index: n for n, op in enumerate(options, 1)}
            self.say(f"'{text}' could mean:")
            for op in hits:
                self.say(f"{number[op.index]:>3}  {op.text or op.name}", indent=2, hang=5)
            return
        here = self.state[self.p.loc]
        near = [op for op in self.p.actions if op.loc_req in (None, here) and self.matches(op, tokens)]
        if 0 < len(near) <= 3:
            self.why_not(near)
        else:
            self.say(f"Nothing you can do right now matches '{text}'. Type a number from the list, or 'help'.")

    def matches(self, op: Operator, tokens: list[str]) -> bool:
        words = self.words[op.index]
        return all(any(w.startswith(t) for w in words) for t in tokens)

    def why_not(self, ops: list[Operator]) -> None:
        for op in ops:
            reasons = dict.fromkeys(self.describe(c) for c in op.conds if not c.fn(self.state))
            self.say(f"Not possible right now: {op.text or op.name}")
            for r in reasons:
                self.say(f"- {r}", "dim", indent=2)

    def describe(self, cond: CCond) -> str:
        """Say in plain words why a failing precondition fails (the raw statement if unsure)."""
        parts = [self.describe_atom(a) for a in cond.atoms]
        if any(p is None for p in parts):
            return cond.text
        return " and ".join(dict.fromkeys(parts))  # type: ignore[arg-type]

    def describe_atom(self, a: CAtom) -> str | None:
        name, op = a.var.name, a.op
        if op in ("in", "not in"):
            return None
        cur = a.var.domain.decode(a.var.getter(self.state))
        parts = name.split(".")
        if a.ref is not None:
            if parts[0] == "npc" and parts[-1] == "loc" and a.ref.name == "monster.loc" and op == "==":
                return f"{self.names[parts[1]]} isn't here"
            return None
        want = a.var.domain.decode(a.const)
        if name == "monster.loc" and op == "==":
            return f"you need to be in {self.names[want]}"
        if name == "monster.form" and op in ("==", "!="):
            form = self.form_text(want)
            return f"you need to be {form}" if op == "==" else f"you can't do that while {form}"
        if name == "monster.energy" and op in (">=", ">"):
            return f"you need {want if op == '>=' else want + 1} energy and have {cur}"
        if name == "monster.energy" and op in ("<", "<="):
            return "your energy is already full"
        if name == "alarm" and op in ("<", "<="):
            return f"the alarm is too high ({cur})"
        if name == "alarm" and op in (">", ">="):
            return f"the alarm is not high enough ({cur})"
        if name == "here.observed" and op == "==" and want is False:
            return "someone is watching"
        if parts[0] == "has" and op == "==":
            item = self.names.get(parts[1], parts[1])
            return f"you don't have {item}" if want else f"you already have {item}"
        if len(parts) == 3 and op == "==":
            kind, ident, field = parts
            thing = self.names.get(ident, ident)
            phrases = {  # (when it had to be true, when it had to be false)
                ("door", "locked"): ("{} is already unlocked", "{} is locked"),
                ("vent", "sealed"): ("{} is already open", "{} is sealed"),
                ("cam", "active"): ("{} is already looped", "{} is still watching"),
                ("term", "hacked"): ("{} isn't hacked yet", "{} is already hacked"),
            }
            if (kind, field) in phrases:
                return phrases[kind, field][0 if want else 1].format(thing)
            if kind == "npc" and field == "status":
                return f"you haven't absorbed {thing}" if want == "absorbed" else f"{thing} is {STATUS_TEXT[cur]}"
            if kind == "npc" and field == "loc":
                return f"{thing} isn't in {self.names.get(want, want)}"
        return None

    def finish(self, outcome: str) -> None:
        self.out("")
        turn = self.val("turn")
        if outcome == WIN:
            self.say(f"*** You escape to {self.names[self.val('monster.loc')]}! ***", "bold", "green")
            used = sorted({op.ability for op in self.moves} - {"move"}, key=self.rank.__getitem__)
            best = self.fastest()
            self.say(f"Turns taken: {turn} of {self.sc['turn_limit']}" + (f" (the fastest escape takes {best})" if best else ""))
            self.say(f"Final alarm: {self.val('alarm')}/{self.sc['security']['max_alarm']}")
            self.say(f"Abilities used: {', '.join(used) or 'none'}")
            self.say(f"Hints: {self.hints}, undos: {self.undos}")
        else:
            title = outcome.replace("_", " ").capitalize()
            self.say(f"*** {title}: {self.loss_text.get(outcome, '')} ***", "bold", "red")
            self.say(f"You lasted {turn} turns.")
        self.say("Type 'undo' to take back your last move, 'restart' to play again, or 'quit'.", "dim")

    # ----------------------------------------------------------- commands
    def cmd_look(self) -> None:
        self.show(full=True)

    def cmd_help(self) -> None:
        self.out("")
        for line in HELP.splitlines():
            self.out(line)

    def cmd_undo(self) -> None:
        if len(self.history) == 1:
            self.say("There is nothing to undo.")
            return
        self.history.pop()
        op = self.moves.pop()
        self.undos += 1
        self.say(f"You take back: {op.text or op.name}", "dim")
        self.show(full=True)

    def cmd_restart(self) -> None:
        self.reset()
        self.say("Back to the beginning.", "dim")
        self.show(full=True)

    def cmd_hint(self) -> None:
        if self.outcome is not None:
            self.say("The game is over. Type 'undo', 'restart' or 'quit'.", "dim")
            return
        self.hints += 1
        result = astar(self.p, self.engine, max_expansions=HINT_BUDGET, start=self.state)
        if result.plan:
            op = result.plan[0]
            number = {o.index: n for n, o in enumerate(self.options(), 1)}
            self.say(f"Hint: [{number[op.index]}] {op.text or op.name} (the fastest way out from here takes {len(result.plan)} more turns)", "cyan")
        elif result.capped:
            self.say("Hint: there is no clear way out from here. Going back a few moves might help.", "cyan")
        else:
            self.say("Hint: there is no way out from here any more. Type 'undo' to go back.", "red")

    def cmd_map(self) -> None:
        here = self.val("monster.loc")
        self.out("")
        self.say("Facility map (* marks where you are)", "bold")
        for zone in sorted({r["zone"] for r in self.sc["rooms"]}):
            self.say(f"Zone {zone}", "dim")
            for r in self.sc["rooms"]:
                if r["zone"] != zone:
                    continue
                mark = "*" if r["id"] == here else " "
                self.say(f"{mark} {r['name']}{' (EXIT)' if r['exit'] else ''}", indent=1)
                for line in self.passages(r["id"], self.state):
                    self.say(line, indent=5, hang=2)

    def cmd_people(self) -> None:
        turn = self.val("turn")
        self.out("")
        self.say(f"People you have watched (now, then the next {FORECAST_TURNS - 1} turns)", "bold")
        for n in self.sc["npcs"]:
            tags = [n["role"], f"clearance {n['clearance']}"] + (["armed"] if n["armed"] else []) + (["has a phone"] if n["phone"] else [])
            self.say(f"{n['name']} ({', '.join(tags)})", indent=2)
            status = self.val(f"npc.{n['id']}.status")
            if status == "normal":
                patrol = n["patrol"]
                route = [self.names[patrol[(turn + k) % len(patrol)]] for k in range(FORECAST_TURNS)]
                self.say(" -> ".join(route), indent=6, hang=2)
            elif status == "unconscious":
                down_at = self.val(f"npc.{n['id']}.down_at")
                self.say(f"unconscious in {self.names[down_at]}", indent=6)
            else:
                self.say(STATUS_TEXT[status], indent=6)
            held = [f"carries {_join([self.names[i] for i in n['carries']])}"] if n["carries"] else []
            held += [f"knows {_join([self.names[i] for i in n['knows']])}"] if n["knows"] else []
            if held and status in ("normal", "unconscious"):
                self.say("; ".join(held), "dim", indent=6)


# --------------------------------------------------------------------------
# Picking a scenario
# --------------------------------------------------------------------------


def find_scenarios(root: Path = Path("scenarios")) -> list[Path]:
    """Scenario files under ``root`` (skipping saved reports and logs), then the bundled examples."""
    found = [p for p in sorted(root.glob("**/*.json")) if not p.name.endswith((".report.json", ".log.json"))]
    return found + example_paths()


def choose_scenario(paths: list[Path], read: Callable[[str], str] = input, out: Callable[[str], None] = print) -> Path | None:
    if not paths:
        out("No scenarios found; pass a scenario file.")
        return None
    out("Choose a scenario:")
    for i, path in enumerate(paths, 1):
        try:
            data = json.loads(path.read_text())
            title = data.get("title") if isinstance(data, dict) else None
        except (OSError, ValueError):
            title = None
        out(f"  {i:>2}. {title or '(unreadable)'} ({path.name})")
    while True:
        try:
            answer = read(f"scenario [1-{len(paths)}]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            out("")
            return None
        if answer in ("q", "quit"):
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(paths):
            return paths[int(answer) - 1]
        out(f"Type a number from 1 to {len(paths)}, or q to quit.")


def _join(items: list[str], word: str = "and") -> str:
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} {word} {items[-1]}"


def _value_text(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)

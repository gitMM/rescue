"""Prompts for the scenario-writing model.

The system prompt is fully deterministic (no timestamps, sorted JSON), so
it stays byte-identical across requests and is served from the prompt cache.
"""

from __future__ import annotations

import json
from dataclasses import fields

from .. import dsl
from ..library import Library, load_example, load_library
from ..validate import Gates, repair_brief

ROLE = """\
You design levels for "Rescue", a turn-based stealth game. The player is a
shape-shifting, hyper-intelligent creature held in a top-secret research
facility. It can melt into a liquid and ooze through vents, absorb people
to steal their bodies, memories and belongings, wear their faces, hack
terminals, phish phones and talk staff into opening doors. The goal is
always the same: get out.

Your levels are pure data. A Python engine compiles your JSON into a state
graph (every combination of where the creature is, what it looks like,
what it holds, who is still standing, which doors are open, the alarm
level and the turn) and a search proves whether an escape exists and how
interesting the level is. You never write code: you write one JSON
scenario that follows the response schema exactly."""

MECHANICS = """\
# How a turn works
1. The creature takes exactly one action whose preconditions hold.
2. The clock advances. Every NPC with status normal stands in
   patrol[turn % len(patrol)]. NPCs ignore doors; only the creature is
   blocked by them.
3. If the creature is in a room with "exit": true (and every goal_extra
   condition holds), it has escaped. Nothing else happens.
4. World rules fire once each, in order (core rules first, then yours).
5. Loss conditions are checked in order (core losses first, then yours);
   the first that holds ends the game.

# State variables you can read (and write, unless derived)
- turn (read-only), alarm (0..security.max_alarm), lockdown (0..security.purge_turns)
- monster.loc (room id), monster.form (self | liquid | an npc id), monster.energy (0..monster.max_energy)
- door.<id>.locked, vent.<id>.sealed, cam.<id>.active, term.<id>.hacked, has.<item id>   (all booleans)
- npc.<id>.status (normal | unconscious | absorbed | found), npc.<id>.down_at (room id or none)
- your own vars, by their bare name (e.g. truck_arrived)
Derived, read-only:
- npc.<id>.loc: patrol position if normal, down_at if unconscious, none otherwise
- here.witnesses / here.guards: number of normal NPCs / armed normal NPCs in the creature's room
- here.camera: an active camera covers the creature's room; here.observed: witnesses or camera
- here.zone: zone of the creature's room

# Facility semantics
- zone: 0 = containment, higher = more secure. A disguise is "out of place" when
  here.zone > the disguised person's clearance and the creature is observed.
- Locked doors with clearance > 0 open to a keycard item or a badge swipe (disguised as
  someone) with clearance >= the door's. Any locked door listed in a terminal's
  "controls" can be unlocked from that terminal, and any locked door can be opened by
  conning a normal NPC with enough clearance who stands on the creature's side of it.
  A locked door with clearance 0 has no badge reader: only terminals and cons open it.
- Vents connect rooms for the liquid form only. Vents with "sealable": true slam shut
  when alarm >= security.seal_vents_at. A vent that starts "sealed": true must be
  opened from a terminal that lists it in "controls".
- Terminals are hacked with energy >= security (noisy: alarm +1) or with the item named
  in "password" (silent). A hacked terminal unlocks/loops/unseals what it controls,
  phishes the phones of NPCs with "phone": true (gaining what they "know"), and resets
  the alarm by 1 if "alarm_control" is true.
- Absorbing someone gives their "carries" and "knows" items, +2 energy and the ability to
  mimic them. Stolen badges stop working when alarm >= security.revoke_badges_at.
- Items with a "room" lie there to be picked up; items with room null exist only in
  someone's "carries" or "knows" list.
- Being seen in true form outside the start room raises the alarm; an armed guard who sees
  it shoots (loss). At max_alarm the lockdown counter runs and the purge kills the
  creature after purge_turns turns."""

ADVICE = """\
# What makes a good level
- At least two genuinely different routes, each needing a different mix of abilities
  (for example vents + hacking versus absorb + disguise + con). No single ability
  should be enough on its own, and walking out with basic moves must be impossible.
- Layered security: the start room is locked, the exit is behind at least two
  obstacles of different kinds, keycards and passwords sit on people whose patrols
  make them reachable at specific turns.
- Danger that matters: armed guards whose patrols cover chokepoints, cameras on key
  corridors, alarm thresholds that bite if the creature is careless.
- Timing: patrol lengths of 3-6 steps create windows. Put the first useful patrol
  contact within the first few turns so the level opens quickly.
- turn_limit around twice the fastest route (adjust for difficulty), never tighter
  than the fastest route + 2.
- Keep the state space tractable: at most ~16 rooms, 8 NPCs, 4 terminals and 6 vents.
  Every extra door, item and NPC multiplies the states the validator must search.
- ids are lowercase snake_case and unique across rooms, doors, vents, cameras,
  terminals, npcs and items. Every reference must point at an existing id.
- Custom actions, rules, vars and losses are optional set pieces (a supply truck that
  arrives at turn 18, an incinerator, a flooded tunnel). Write them in the same
  statement language as the core library. A custom action's "ability" can reuse a core
  ability (move, keycard, liquid, absorb, subdue, disguise, hack, phish, con) or name a
  new one.
- design_notes: describe each intended route and the main dangers in a few sentences.
  The validator's report on your level will be compared against them."""


def gates_text(gates: Gates) -> str:
    g = gates
    lines = [
        "# Acceptance gates (checked automatically)",
        "- solvable: an escape must exist within turn_limit",
        f"- min_solution_turns: the fastest escape takes at least {g.min_solution_turns} turns",
        f"- turn_slack: turn_limit is at least the fastest escape + {g.min_slack_turns}",
        f"- abilities_in_solution: the fastest escape uses at least {g.min_abilities_in_solution} abilities besides 'move'",
    ]
    if g.require_combination:
        lines.append("- requires_combination: no single ability (plus basic moves) escapes on its own")
    if g.require_danger:
        lines.append("- danger: at least one losing outcome other than running out of time is reachable")
    lines += [
        f"- min_states: at least {g.min_states} reachable states",
        f"- min_routes: at least {g.min_routes} escape routes that use different sets of abilities",
        f"- max_doomed_ratio: at most {g.max_doomed_ratio:.1%} of states are already lost (when fully explored)",
    ]
    return "\n".join(lines)


def build_system_prompt(library: Library | None = None, gates: Gates | None = None) -> str:
    library = library or load_library()
    gates = gates or Gates()
    example = json.dumps(load_example(), indent=1, sort_keys=False)
    return "\n\n".join(
        [
            ROLE,
            MECHANICS,
            "# Statement language\n" + (dsl.__doc__ or "").strip(),
            "# Core library (always active unless listed in disabled_core)\n```yaml\n" + library.text.strip() + "\n```",
            gates_text(gates),
            ADVICE,
            "# Example scenario (passes every gate)\n```json\n" + example + "\n```",
        ]
    )


def repair_message(report: dict) -> str:
    return (
        "The validator rejected this scenario.\n\n"
        + repair_brief(report)
        + "\n\nReturn the complete corrected scenario as one JSON object (the whole scenario, not a diff). "
        "Keep what works and change what the report points at."
    )


def invalid_json_message(error: str) -> str:
    return f"Your reply could not be parsed as JSON ({error}). Return the complete scenario as one JSON object."


def gate_fields() -> list[str]:
    return [f.name for f in fields(Gates)]

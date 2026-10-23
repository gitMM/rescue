# Rescue

Escape scenarios for a shape-shifting, hyper-intelligent monster locked in a
top-secret facility. The monster can melt and ooze through vents, absorb people
and wear their faces, hack terminals, phish phones and talk staff into opening
doors. Each scenario is one way out of one facility.

There are **no hard-coded scenarios**. A scenario is pure data (JSON). It is
compiled into a turn-based **state graph** and searched with **A\***, so every
accepted scenario is proven playable from start to finish. Claude writes new
scenarios, and the validator's report is fed back to it until a scenario passes.

```
 brief ──► Claude (structured JSON) ──► compile ──► state graph ──► A* + BFS + ability analysis
   ▲                                                                       │
   └──────────── repair: errors, failed gates, diagnostics ◄── quality gates
```

## Quick start

```bash
pip install -e '.[dev]'            # PyYAML, jsonschema, pytest, anthropic

python -m rescue validate          # validate the bundled example (Site Theta)
python -m rescue validate my.json --json
python -m pytest                   # automated tests (no API key needed)

# generate with Claude (needs ANTHROPIC_API_KEY or an `ant auth login` profile)
python -m rescue generate --count 5 --seed 100 --workers 2 --out scenarios/generated
python -m rescue generate --setting "a deep-sea habitat" --difficulty hard

python -m rescue prompt            # the exact system prompt the generator uses
python -m rescue schema            # the scenario JSON schema
```

Generated scenarios land in `scenarios/generated/accepted/` (or `rejected/`).
Each one is saved with its validation report, and `logs/` holds every round's
report.

## How the game is modelled

**State.** Each state is a tuple of small integers:

- `turn`, `alarm`, `lockdown`, plus the scenario's custom `vars`
- `monster.loc`, `monster.form` (self, liquid or someone it absorbed), `monster.energy`
- `door.*.locked`, `vent.*.sealed`, `cam.*.active`, `term.*.hacked`, `has.<item>`
- `npc.*.status` (normal, unconscious, absorbed or found) and `npc.*.down_at`

Some values are **derived** rather than stored, so they add nothing to the size
of the state space:

- `npc.*.loc`, worked out from the patrol list and the turn
- `here.witnesses`, `here.guards`, `here.camera`, `here.observed` and `here.zone`

**Turn order** (`rescue/engine.py`):

1. The monster takes one action.
2. The clock advances and every NPC steps along their patrol.
3. If the monster is now in an exit room, it has escaped.
4. Otherwise the world rules fire in order.
5. Loss conditions are checked.

The turn number is part of the state, so the graph is acyclic and finite.

**Actions, rules and losses** are templates written in a small statement
language (`rescue/dsl.py`):

```yaml
- id: swipe
  ability: disguise
  bind: [{name: p, from: passages}, {name: n, from: npcs}]
  where: ["{p.kind} == door", "{p.clearance} > 0", "{n.clearance} >= {p.clearance}"]
  pre:   ["monster.loc == {p.from}", "door.{p.via}.locked == true",
          "monster.form == {n.id}", "alarm < {security.revoke_badges_at}"]
  eff:   ["monster.loc = {p.to}"]
  text:  "As {n.name}, badge through from {p.from_name} to {p.to_name}."
```

The compiler **grounds** each template against a facility's rooms, doors, vents,
cameras, terminals, NPCs and items. Site Theta has 256 concrete actions and 46
rules. It also turns preconditions into fast generated predicates and indexes
actions by room.

The core library (`rescue/library/core.yaml`) defines these abilities:

| Ability    | Actions |
|------------|---------|
| `move`     | walk, wait, rest, revert, take (the basics) |
| `keycard`  | open badge doors with a keycard of high enough clearance |
| `liquid`   | liquefy, crawl through vents |
| `absorb`   | absorb a person: their items, memories (passwords) and +2 energy |
| `subdue`   | knock someone out and take their items (the body can be found) |
| `disguise` | mimic an absorbed person, swipe their badge (revoked at high alarm) |
| `hack`     | brute-force or log into terminals, then unlock doors, loop cameras, unseal vents, reset the alarm |
| `phish`    | phish an NPC's phone from a hacked terminal to steal what they know |
| `con`      | while disguised, talk staff into opening doors or handing over items |

The core library also defines the world reactions:

- being seen in true form, as a moving puddle, or out of place in a disguise
- bodies being found
- vents sealing at high alarm
- the lockdown countdown

The core losses are being shot by a guard, being captured during lockdown,
being purged, and running out of time. Scenarios add their own set pieces in
the same language. The example adds a supply truck that arrives on turn 18,
which you can stow away on as liquid, and an incinerator that runs during
lockdown.

## Validation and quality gates

`rescue/validate.py` runs these steps in order:

1. Schema and cross-reference checks.
2. Compilation.
3. **Relaxed reachability.** This optimistic analysis explains *why* an exit
   can't be reached, for example: "Blocked way into Hall: walk(...) needs
   `door.d_cell.locked == false`".
4. **A\*** for the fastest escape. The heuristic is the fewest moves to an exit,
   which never overestimates.
5. Replay of that escape through the engine.
6. **Breadth-first exploration** of the state graph, up to a cap. It counts
   wins, losses by kind, dead ends and winning paths.
7. **Ability analysis.** Each ability is banned in turn to find alternative
   routes and indispensable abilities. Each ability is also tried alone to
   catch shortcuts.

| Gate | Default |
|------|---------|
| `solvable` | an escape exists within `turn_limit` |
| `min_solution_turns` | fastest escape takes at least 8 turns |
| `turn_slack` | `turn_limit` is at least the fastest escape + 2 |
| `abilities_in_solution` | the fastest escape uses at least 2 abilities besides `move` |
| `requires_combination` | no single ability (plus basic moves) is enough |
| `danger` | something other than the clock can end the game |
| `min_states` | at least 1000 reachable states |
| `min_routes` | at least 2 escape routes with different ability sets |
| `max_doomed_ratio` | at most 99.5% dead-end states (only when fully explored) |

Override any gate from the command line with `--gate min_solution_turns=12`,
or in code with `Gates(...)`.

The bundled **Site Theta** passes every gate. Its fastest escape takes 10
turns, and there are four routes with different ability mixes:

- vents and the Director's master key
- absorbing Dr. Ito and wearing her face
- a con through the checkpoint
- and more

The exploration reaches 60k states before the cap, with shot, captured, purged
and incinerated all reachable.

## LLM generation (`rescue/llm/`)

- **Model:** `claude-opus-5-5` by default (`--model` to change). Thinking is
  adaptive, and `--effort` sets how hard it thinks (default `high`).
- **Structured outputs:** the response is constrained to the scenario JSON
  schema. The schema stays inside the subset structured outputs supports;
  ranges and references are checked in Python.
- **Prompt:** the system prompt contains the mechanics, the statement language,
  the full core library, the active gates and the example scenario. It is
  deterministic and cached, so repair rounds and batches reuse the cache.
- **Refusals:** server-side refusal fallbacks are on (`fallbacks: "default"`);
  `--no-fallbacks` turns them off.
- **Repair loop:** the conversation only ever grows. Each rejected scenario is
  followed by the validator's errors, failed gates, diagnostics and the current
  fastest escape, up to `--rounds` rounds. The best attempt is kept.
- **Variety:** seeded briefs (setting, twist, size, difficulty) keep batches
  varied. `--setting`, `--twist`, `--size` and `--difficulty` override them.
- **Offline dry run:** `--fake file.json ...` replays files instead of calling
  the API.

## Tests

`python -m pytest` runs about 90 tests in about 20 seconds, with no network
access. They cover:

- the statement language
- the schema, including that it stays within the structured-outputs subset
- grounding and compile errors
- engine rules: patrols, cameras, guards, absorbing, the turn limit
- random-playout checks that the state stays well formed
- A\* optimality against breadth-first search, and path counting against brute
  force
- every gate and diagnostic
- the generate-and-repair loop with a scripted fake LLM, plus checks on request
  shape and reply parsing with a stubbed Anthropic client
- the command line

A live API test is skipped unless `RESCUE_LIVE_TESTS=1` is set.

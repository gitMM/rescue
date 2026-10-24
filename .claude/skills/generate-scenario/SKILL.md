---
name: generate-scenario
description: Design a new Rescue escape scenario (a JSON level for the shape-shifting monster game) and iterate with the validator until it is accepted. Use when asked to create, generate or design a scenario, level or facility.
argument-hint: "[setting, twist, size, difficulty - all optional]"
allowed-tools: Read, Write, Edit, Bash(python -m rescue *)
---

# Generate a Rescue scenario

You are designing one level for Rescue: a shape-shifting monster escaping a
top-secret facility. The level is pure JSON. The project's validator compiles
it into a turn-based state graph, proves an escape exists with A*, and checks
quality gates. Your job is to write the JSON and keep fixing it until the
validator accepts it.

Brief from the user: $ARGUMENTS

## 1. Get the brief and the design guide

- If the brief above is empty or vague, get a random one with
  `python -m rescue prompt --brief <any number>` and fill the gaps from it.
- Read the full design guide once: `python -m rescue prompt`. It is the same
  guide the API generator uses: turn order, state variables, the statement
  language, the core library of abilities, rules and losses, the acceptance
  gates and a complete accepted example (Site Theta). Follow it exactly.
- If unsure about a field, `python -m rescue schema` prints the JSON schema.
  Every field is required; use `[]` or `null` where the schema allows.

## 2. Plan before writing

Decide, in a few lines, before writing any JSON:
- the layout: zones from containment (0) to the exits, which doors are
  locked and at what clearance, where the vents run
- at least two routes that need different ability mixes (for example vents +
  hacking versus absorb + disguise + con), each with a timing hook from a
  patrol, and at least one danger on each (armed guard, camera, alarm)
- which person carries which keycard or password, and where their patrol
  makes them reachable
- one set piece using custom vars/rules/actions/losses if it fits the brief

Write the plan into `design_notes`, so the report can be compared with it.

## 3. Write, validate, fix

1. Write the scenario to `scenarios/<id>.json` (`id` is snake_case and matches
   the file name).
2. Run `python -m rescue validate scenarios/<id>.json --no-steps`.
   Exit code 0 means accepted.
3. If it is rejected, read the report:
   - `error:` lines: malformed JSON, bad references or statements. Fix these first.
   - `gate FAIL` lines with `diagnostic:` advice: what is wrong with the design.
   - `route:` lines and `single-ability escapes`: shortcuts to close.
   - "Blocked way into ..." diagnostics: the exact precondition that never holds.
4. When a route you intended does not show up, or the fastest escape is a
   shortcut you did not intend, test lines of play directly:
   `python -m rescue play scenarios/<id>.json 'wait' 'absorb(dr_ito)' ...`
   It prints each step, the world's reactions, explains any impossible step
   by its failing preconditions, and lists every action available at the end
   (use those exact names).
5. Edit the file and repeat. Fix the design rather than gaming the gates:
   don't pad routes with pointless locks or inflate turn limits. If the
   scenario is still rejected after about 8 rounds, stop and report what keeps
   failing and why.

## 4. Finish

- Run `python -m rescue validate scenarios/<id>.json` (with steps) and show
  the user: the title and premise, the gates, the routes it found, the
  fastest escape walkthrough, and any intended route from `design_notes` that
  the validator did not find.
- In a cloud session the container is temporary: commit the scenario file
  (and push the branch) so it is not lost.

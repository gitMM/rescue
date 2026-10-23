"""Scenario file format: the JSON schema and the structural checks.

The schema is used twice: as the structured-output format the LLM must
follow, and to validate scenario files locally. It sticks to the subset
of JSON Schema that structured outputs accept (every object closed with
``additionalProperties: false``, every property required, no numeric or
length constraints), so ranges, id formats and cross references are
checked in Python by :func:`check_structure`.
"""

from __future__ import annotations

import re
from typing import Any

import jsonschema

ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
ITEM_KINDS = ["keycard", "intel", "tool", "other"]
VAR_KINDS = ["bool", "int", "enum"]
COLLECTIONS = ("rooms", "doors", "vents", "cameras", "terminals", "npcs", "items")

LIMITS = {
    "rooms": 40,
    "doors": 80,
    "vents": 30,
    "cameras": 30,
    "terminals": 12,
    "npcs": 16,
    "items": 30,
    "vars": 8,
    "actions": 12,
    "rules": 12,
    "losses": 8,
}


def _obj(props: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


STR = {"type": "string"}
INT = {"type": "integer"}
BOOL = {"type": "boolean"}
STRS = {"type": "array", "items": STR}

BIND = {"type": "array", "items": _obj({"name": STR, "from": STR})}
EFFECT = {
    "anyOf": [
        STR,
        _obj({"if": STRS, "then": STRS, "else": STRS}),
    ]
}
EFFECTS = {"type": "array", "items": EFFECT}

ACTION = _obj(
    {
        "id": STR,
        "ability": STR,
        "bind": BIND,
        "where": STRS,
        "pre": STRS,
        "eff": EFFECTS,
        "text": STR,
    }
)
RULE = _obj(
    {
        "id": STR,
        "bind": BIND,
        "where": STRS,
        "when": STRS,
        "then": EFFECTS,
        "text": STR,
    }
)
LOSS = _obj({"id": STR, "when": STRS, "text": STR})

SCENARIO_SCHEMA: dict[str, Any] = _obj(
    {
        "id": STR,
        "title": STR,
        "premise": STR,
        "design_notes": STR,
        "turn_limit": INT,
        "monster": _obj({"start_room": STR, "energy": INT, "max_energy": INT}),
        "security": _obj(
            {
                "max_alarm": INT,
                "revoke_badges_at": INT,
                "seal_vents_at": INT,
                "purge_turns": INT,
            }
        ),
        "rooms": {
            "type": "array",
            "items": _obj({"id": STR, "name": STR, "zone": INT, "exit": BOOL, "description": STR}),
        },
        "doors": {
            "type": "array",
            "items": _obj({"id": STR, "name": STR, "a": STR, "b": STR, "locked": BOOL, "clearance": INT}),
        },
        "vents": {
            "type": "array",
            "items": _obj({"id": STR, "name": STR, "a": STR, "b": STR, "sealed": BOOL, "sealable": BOOL}),
        },
        "cameras": {"type": "array", "items": _obj({"id": STR, "name": STR, "room": STR})},
        "terminals": {
            "type": "array",
            "items": _obj(
                {
                    "id": STR,
                    "name": STR,
                    "room": STR,
                    "security": INT,
                    "password": _nullable(STR),
                    "alarm_control": BOOL,
                    "controls": STRS,
                }
            ),
        },
        "npcs": {
            "type": "array",
            "items": _obj(
                {
                    "id": STR,
                    "name": STR,
                    "role": STR,
                    "clearance": INT,
                    "armed": BOOL,
                    "phone": BOOL,
                    "patrol": STRS,
                    "carries": STRS,
                    "knows": STRS,
                }
            ),
        },
        "items": {
            "type": "array",
            "items": _obj(
                {
                    "id": STR,
                    "name": STR,
                    "kind": {"type": "string", "enum": ITEM_KINDS},
                    "clearance": _nullable(INT),
                    "room": _nullable(STR),
                }
            ),
        },
        "vars": {
            "type": "array",
            "items": _obj(
                {
                    "name": STR,
                    "kind": {"type": "string", "enum": VAR_KINDS},
                    "init": {"anyOf": [INT, BOOL, STR]},
                    "min": _nullable(INT),
                    "max": _nullable(INT),
                    "values": _nullable(STRS),
                }
            ),
        },
        "actions": {"type": "array", "items": ACTION},
        "rules": {"type": "array", "items": RULE},
        "losses": {"type": "array", "items": LOSS},
        "goal_extra": STRS,
        "disabled_core": STRS,
    }
)

_VALIDATOR = jsonschema.Draft202012Validator(SCENARIO_SCHEMA)


def schema_errors(data: Any) -> list[str]:
    errors = []
    for err in sorted(_VALIDATOR.iter_errors(data), key=lambda e: list(e.path)):
        where = "/".join(str(p) for p in err.path) or "<root>"
        errors.append(f"schema: {where}: {err.message}")
    return errors


def check_structure(data: Any, disableable: set[str] | None = None, core_ids: set[str] | None = None) -> list[str]:
    """Return human-readable problems with a scenario; empty means well formed.

    ``disableable`` are the core action/rule ids a scenario may switch off;
    ``core_ids`` are all core ids (including losses) that custom specs may
    not reuse.
    """
    errors = schema_errors(data)
    if errors:
        return errors[:40]

    def bad_id(kind: str, value: Any) -> None:
        if not isinstance(value, str) or not ID_RE.match(value):
            errors.append(f"{kind} id {value!r} must match {ID_RE.pattern}")

    for key, limit in LIMITS.items():
        if len(data[key]) > limit:
            errors.append(f"too many {key}: {len(data[key])} (max {limit})")

    all_ids: dict[str, str] = {}
    for coll in COLLECTIONS:
        for ent in data[coll]:
            bad_id(coll[:-1], ent["id"])
            if ent["id"] in all_ids:
                errors.append(f"duplicate id {ent['id']!r} ({all_ids[ent['id']]} and {coll})")
            all_ids[ent["id"]] = coll
    bad_id("scenario", data["id"])

    rooms = {r["id"]: r for r in data["rooms"]}
    items = {i["id"]: i for i in data["items"]}
    controllable = {e["id"] for coll in ("doors", "cameras", "vents") for e in data[coll]}

    if len(rooms) < 2:
        errors.append("a facility needs at least two rooms")
    exits = [r for r in data["rooms"] if r["exit"]]
    if not exits:
        errors.append("no room has exit: true, so escape is impossible")
    for r in data["rooms"]:
        if not 0 <= r["zone"] <= 10:
            errors.append(f"room {r['id']}: zone must be 0..10")

    start = data["monster"]["start_room"]
    if start not in rooms:
        errors.append(f"monster.start_room {start!r} is not a room")
    elif rooms[start]["exit"]:
        errors.append("the monster cannot start in an exit room")
    m = data["monster"]
    if not 1 <= m["max_energy"] <= 20 or not 0 <= m["energy"] <= m["max_energy"]:
        errors.append("monster energy must satisfy 0 <= energy <= max_energy <= 20 (max_energy >= 1)")

    if not 5 <= data["turn_limit"] <= 200:
        errors.append("turn_limit must be between 5 and 200")
    sec = data["security"]
    if not 1 <= sec["max_alarm"] <= 10:
        errors.append("security.max_alarm must be 1..10")
    for key in ("revoke_badges_at", "seal_vents_at"):
        if not 1 <= sec[key] <= sec["max_alarm"]:
            errors.append(f"security.{key} must be between 1 and max_alarm")
    if not 1 <= sec["purge_turns"] <= 10:
        errors.append("security.purge_turns must be 1..10")

    for kind in ("doors", "vents"):
        for e in data[kind]:
            for end in ("a", "b"):
                if e[end] not in rooms:
                    errors.append(f"{kind[:-1]} {e['id']}: {end}={e[end]!r} is not a room")
            if e["a"] == e["b"]:
                errors.append(f"{kind[:-1]} {e['id']} connects {e['a']!r} to itself")
    for d in data["doors"]:
        if not 0 <= d["clearance"] <= 10:
            errors.append(f"door {d['id']}: clearance must be 0..10")
    for c in data["cameras"]:
        if c["room"] not in rooms:
            errors.append(f"camera {c['id']}: room {c['room']!r} does not exist")
    for t in data["terminals"]:
        if t["room"] not in rooms:
            errors.append(f"terminal {t['id']}: room {t['room']!r} does not exist")
        if not 0 <= t["security"] <= 10:
            errors.append(f"terminal {t['id']}: security must be 0..10")
        if t["password"] is not None and t["password"] not in items:
            errors.append(f"terminal {t['id']}: password {t['password']!r} is not an item")
        for ctl in t["controls"]:
            if ctl not in controllable:
                errors.append(f"terminal {t['id']}: controls {ctl!r}, which is not a door, camera or vent")
    for n in data["npcs"]:
        if not n["patrol"]:
            errors.append(f"npc {n['id']}: patrol needs at least one room")
        for room in n["patrol"]:
            if room not in rooms:
                errors.append(f"npc {n['id']}: patrol room {room!r} does not exist")
        for key in ("carries", "knows"):
            for it in n[key]:
                if it not in items:
                    errors.append(f"npc {n['id']}: {key} {it!r} is not an item")
        if not 0 <= n["clearance"] <= 10:
            errors.append(f"npc {n['id']}: clearance must be 0..10")
        if len(n["patrol"]) > 24:
            errors.append(f"npc {n['id']}: patrol longer than 24 steps")
    for it in data["items"]:
        if it["room"] is not None and it["room"] not in rooms:
            errors.append(f"item {it['id']}: room {it['room']!r} does not exist")
        if it["kind"] == "keycard" and (it["clearance"] is None or not 1 <= it["clearance"] <= 10):
            errors.append(f"item {it['id']}: keycards need a clearance of 1..10")

    reserved = {"alarm", "lockdown", "turn", "monster", "here", "npc", "door", "vent", "cam", "term", "has"}
    for v in data["vars"]:
        bad_id("var", v["name"])
        if v["name"] in reserved:
            errors.append(f"var {v['name']!r} uses a reserved name")
        if v["kind"] == "int":
            if v["min"] is None or v["max"] is None or v["min"] > v["max"] or v["max"] - v["min"] > 50:
                errors.append(f"var {v['name']}: int vars need min <= max and a range of at most 50")
            elif not isinstance(v["init"], int) or isinstance(v["init"], bool) or not v["min"] <= v["init"] <= v["max"]:
                errors.append(f"var {v['name']}: init must be an int within [min, max]")
        elif v["kind"] == "bool":
            if not isinstance(v["init"], bool):
                errors.append(f"var {v['name']}: init must be true or false")
        else:
            if not v["values"] or v["init"] not in v["values"]:
                errors.append(f"var {v['name']}: enum vars need values and an init among them")

    disableable = disableable or set()
    core_ids = (core_ids or set()) | disableable
    for d in set(data["disabled_core"]) - disableable:
        errors.append(f"disabled_core: {d!r} is not a core action or rule")
    disabled = set(data["disabled_core"]) & disableable
    seen: set[str] = set()
    for kind, label in (("actions", "action"), ("rules", "rule"), ("losses", "loss")):
        for spec in data[kind]:
            bad_id(label, spec["id"])
            if spec["id"] in seen or (spec["id"] in core_ids and spec["id"] not in disabled):
                errors.append(f"{label} id {spec['id']!r} is already used")
            seen.add(spec["id"])
            for b in spec.get("bind", []):
                bad_id("binding", b["name"])
    return errors

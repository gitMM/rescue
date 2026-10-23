"""Compile a scenario (pure data) into a searchable planning problem.

A *state* is a tuple of small ints, one per state variable:

* ``turn``, ``alarm``, ``lockdown`` and the scenario's custom ``vars``
* ``monster.loc`` / ``monster.form`` / ``monster.energy``
* ``door.<id>.locked``, ``vent.<id>.sealed``, ``cam.<id>.active``,
  ``term.<id>.hacked``, ``has.<item>``
* ``npc.<id>.status`` (normal/unconscious/absorbed/found) and
  ``npc.<id>.down_at`` (where they were knocked out)

Some values are *derived* from the state instead of stored, so they cost
nothing in state-space size: ``npc.<id>.loc`` (from the patrol schedule
and the turn), ``here.witnesses``, ``here.guards``, ``here.camera``,
``here.observed`` and ``here.zone``.

Action and rule templates are grounded against the facility's entities,
producing concrete actions whose preconditions and effects are compiled
into fast closures over the state tuple.
"""

from __future__ import annotations

import copy
import operator
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from . import dsl
from .library import Library, load_library
from .schema import check_structure

StateFn = Callable[[tuple], bool]
Getter = Callable[[Any], int]

NPC_STATUS = ("normal", "unconscious", "absorbed", "found")
NORMAL, UNCONSCIOUS = 0, 1
NONE_ROOM = "none"


class ScenarioError(Exception):
    """The scenario cannot be compiled; ``errors`` lists every problem found."""

    def __init__(self, errors: list[str]):
        super().__init__(f"{len(errors)} problem(s): " + "; ".join(errors[:5]))
        self.errors = errors


@dataclass
class Domain:
    kind: str  # bool | int | enum
    lo: int = 0
    hi: int = 1
    values: tuple[Any, ...] = ()

    def encode(self, value: Any) -> int:
        if self.kind == "bool":
            if isinstance(value, bool):
                return int(value)
            raise ValueError(f"expected true/false, got {dsl.literal_text(value)!r}")
        if self.kind == "int":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"expected an integer, got {dsl.literal_text(value)!r}")
            return value
        key = NONE_ROOM if value is None else value
        if key in self.values:
            return self.values.index(key)
        raise ValueError(f"{dsl.literal_text(value)!r} is not one of {list(self.values)[:12]}")

    def decode(self, code: int) -> Any:
        if self.kind == "bool":
            return bool(code)
        if self.kind == "int":
            return code
        return self.values[code]

    def codes(self) -> range:
        if self.kind == "enum":
            return range(len(self.values))
        return range(self.lo, self.hi + 1)

    def same_as(self, other: "Domain") -> bool:
        return (self.kind, self.lo, self.hi, self.values) == (other.kind, other.lo, other.hi, other.values)


@dataclass
class Var:
    name: str
    domain: Domain
    index: int | None  # None for derived (read-only) values
    getter: Getter


@dataclass
class CAtom:
    """A compiled atom, kept in symbolic form for analysis and diagnostics."""

    var: Var
    op: str
    const: Any = None  # int code, or frozenset of codes for in / not in
    ref: Var | None = None
    table: tuple | None = None  # ref code -> var code translation
    fn: StateFn = field(default=None, repr=False)  # type: ignore[assignment]


@dataclass
class CCond:
    atoms: list[CAtom]
    text: str
    fn: StateFn = field(default=None, repr=False)  # type: ignore[assignment]


@dataclass
class CEffect:
    var: Var
    op: str  # set | copy | add
    const: int = 0
    ref: Var | None = None
    table: tuple | None = None
    text: str = ""


@dataclass
class CCondEffect:
    conds: list[CCond]
    then: list[CEffect]
    otherwise: list[CEffect]
    text: str = ""


@dataclass
class Operator:
    """A grounded action or rule."""

    index: int
    name: str
    schema_id: str
    kind: str  # action | rule
    ability: str
    text: str
    conds: list[CCond]
    effects: list[CEffect | CCondEffect]
    loc_req: int | None = None
    test: StateFn = field(default=None, repr=False)  # type: ignore[assignment]
    apply: Callable[[tuple, list], None] = field(default=None, repr=False)  # type: ignore[assignment]

    def holds(self, state: Any) -> bool:
        return self.test(state)


@dataclass
class Loss:
    id: str
    text: str
    conds: list[CCond]
    test: StateFn = field(default=None, repr=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.test is None:
            self.test = make_test(self.conds)

    def holds(self, state: Any) -> bool:
        return self.test(state)


@dataclass
class Problem:
    scenario: dict[str, Any]
    vars: list[Var]
    derived: dict[str, Var]
    init: tuple
    actions: list[Operator]
    rules: list[Operator]
    losses: list[Loss]
    goal: list[CCond]
    turn_limit: int
    room_ids: tuple[str, ...]
    exits: frozenset[int]
    warnings: list[str]
    by_loc: dict[int, list[Operator]] = field(default_factory=dict)
    any_loc: list[Operator] = field(default_factory=list)
    dist: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.var_index: dict[str, Var] = {v.name: v for v in self.vars}
        self.loc: int = self.var_index["monster.loc"].index  # type: ignore[assignment]
        self.turn: int = self.var_index["turn"].index  # type: ignore[assignment]

    def var(self, name: str) -> Var:
        if name in self.derived:
            return self.derived[name]
        return self.var_index[name]

    @property
    def abilities(self) -> list[str]:
        return sorted({a.ability for a in self.actions})

    def h(self, state: tuple) -> float:
        """Admissible lower bound on turns left to reach an exit."""
        return self.dist[state[self.loc]]

    def goal_holds(self, state: Any) -> bool:
        return all(c.fn(state) for c in self.goal)

    def describe(self, state: tuple) -> dict[str, Any]:
        """Decode a state into a readable dict (derived values omitted)."""
        return {v.name: v.domain.decode(state[v.index]) for v in self.vars}  # type: ignore[index]

    def summary(self, state: tuple) -> str:
        d = self.describe(state)
        parts = [
            f"turn {d['turn']}",
            f"in {d['monster.loc']}",
            f"form {d['monster.form']}",
            f"energy {d['monster.energy']}",
            f"alarm {d['alarm']}",
        ]
        items = [k[4:] for k, v in d.items() if k.startswith("has.") and v]
        if items:
            parts.append("holding " + ", ".join(items))
        return ", ".join(parts)


# --------------------------------------------------------------------------
# Compilation
# --------------------------------------------------------------------------


def compile_scenario(scenario: dict[str, Any], library: Library | None = None) -> Problem:
    library = library or load_library()
    errors = check_structure(scenario, library.disableable, library.all_ids)
    if errors:
        raise ScenarioError(errors)
    return _Compiler(copy.deepcopy(scenario), library).build()


class _Compiler:
    def __init__(self, sc: dict[str, Any], library: Library):
        self.sc = sc
        self.lib = library
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.vars: list[Var] = []
        self.by_name: dict[str, Var] = {}
        self.init: list[int] = []
        self.room_ids = tuple(r["id"] for r in sc["rooms"])
        self.room_domain = Domain("enum", values=(NONE_ROOM,) + self.room_ids)
        self.entities: dict[str, dict[str, Any]] = {}
        for coll in ("rooms", "doors", "vents", "cameras", "terminals", "npcs", "items"):
            for e in sc[coll]:
                self.entities[e["id"]] = e

    # ----------------------------------------------------------- variables
    def add_var(self, name: str, domain: Domain, init: Any) -> Var:
        idx = len(self.vars)
        var = Var(name, domain, idx, operator.itemgetter(idx))
        self.vars.append(var)
        self.by_name[name] = var
        self.init.append(domain.encode(init))
        return var

    def build_vars(self) -> None:
        sc = self.sc
        m, sec = sc["monster"], sc["security"]
        npc_ids = tuple(n["id"] for n in sc["npcs"])
        self.add_var("turn", Domain("int", 0, sc["turn_limit"]), 0)
        self.add_var("monster.loc", self.room_domain, m["start_room"])
        self.add_var("monster.form", Domain("enum", values=("self", "liquid") + npc_ids), "self")
        self.add_var("monster.energy", Domain("int", 0, m["max_energy"]), m["energy"])
        self.add_var("alarm", Domain("int", 0, sec["max_alarm"]), 0)
        self.add_var("lockdown", Domain("int", 0, sec["purge_turns"]), 0)
        for d in sc["doors"]:
            self.add_var(f"door.{d['id']}.locked", Domain("bool"), d["locked"])
        for v in sc["vents"]:
            self.add_var(f"vent.{v['id']}.sealed", Domain("bool"), v["sealed"])
        for c in sc["cameras"]:
            self.add_var(f"cam.{c['id']}.active", Domain("bool"), True)
        for t in sc["terminals"]:
            self.add_var(f"term.{t['id']}.hacked", Domain("bool"), False)
        for it in sc["items"]:
            self.add_var(f"has.{it['id']}", Domain("bool"), False)
        for n in sc["npcs"]:
            self.add_var(f"npc.{n['id']}.status", Domain("enum", values=NPC_STATUS), "normal")
            self.add_var(f"npc.{n['id']}.down_at", self.room_domain, None)
        for v in sc["vars"]:
            if v["kind"] == "bool":
                dom = Domain("bool")
            elif v["kind"] == "int":
                dom = Domain("int", v["min"], v["max"])
            else:
                dom = Domain("enum", values=tuple(v["values"]))
            self.add_var(v["name"], dom, v["init"])

    def build_derived(self) -> dict[str, Var]:
        sc = self.sc
        loc = self.by_name["monster.loc"].index
        turn = self.by_name["turn"].index
        rd = self.room_domain
        npcs = []
        for n in sc["npcs"]:
            patrol = tuple(rd.encode(r) for r in n["patrol"])
            npcs.append(
                (
                    n["id"],
                    self.by_name[f"npc.{n['id']}.status"].index,
                    self.by_name[f"npc.{n['id']}.down_at"].index,
                    patrol,
                    bool(n["armed"]),
                )
            )
        cams_in: dict[int, tuple[int, ...]] = defaultdict(tuple)
        for c in sc["cameras"]:
            code = rd.encode(c["room"])
            cams_in[code] = cams_in[code] + (self.by_name[f"cam.{c['id']}.active"].index,)
        zone_of = [0] * len(rd.values)
        for r in sc["rooms"]:
            zone_of[rd.encode(r["id"])] = r["zone"]
        # present[turn][room] -> status indices of the NPCs whose patrol puts them there,
        # so counting witnesses only looks at the few people who could be in the room.
        n_rooms = len(rd.values)
        present = []
        armed_present = []
        for t in range(sc["turn_limit"] + 1):
            rooms_t: list[list[int]] = [[] for _ in range(n_rooms)]
            armed_t: list[list[int]] = [[] for _ in range(n_rooms)]
            for _, st, _, patrol, armed in npcs:
                room = patrol[t % len(patrol)]
                rooms_t[room].append(st)
                if armed:
                    armed_t[room].append(st)
            present.append(tuple(tuple(r) for r in rooms_t))
            armed_present.append(tuple(tuple(r) for r in armed_t))

        derived: dict[str, Var] = {}

        def add(name: str, domain: Domain, fn: Getter) -> None:
            derived[name] = Var(name, domain, None, fn)

        for nid, st, down, patrol, _ in npcs:
            period = len(patrol)

            def npc_loc(s, st=st, down=down, patrol=patrol, period=period):
                status = s[st]
                if status == NORMAL:
                    return patrol[s[turn] % period]
                if status == UNCONSCIOUS:
                    return s[down]
                return 0

            add(f"npc.{nid}.loc", rd, npc_loc)

        def witnesses(s):
            n = 0
            for st in present[s[turn]][s[loc]]:
                if s[st] == NORMAL:
                    n += 1
            return n

        def guards(s):
            n = 0
            for st in armed_present[s[turn]][s[loc]]:
                if s[st] == NORMAL:
                    n += 1
            return n

        def camera(s):
            for ci in cams_in.get(s[loc], ()):
                if s[ci]:
                    return 1
            return 0

        def observed(s):
            return 1 if camera(s) or witnesses(s) else 0

        n_npcs = len(npcs)
        add("here.witnesses", Domain("int", 0, n_npcs), witnesses)
        add("here.guards", Domain("int", 0, n_npcs), guards)
        add("here.camera", Domain("bool"), camera)
        add("here.observed", Domain("bool"), observed)
        add("here.zone", Domain("int", 0, 10), lambda s: zone_of[s[loc]])
        return derived

    # ---------------------------------------------------------- statements
    def lookup_var(self, name: str) -> Var | None:
        return self.by_name.get(name) or self.derived.get(name)

    def compile_atom(self, atom: dsl.Atom) -> CAtom:
        if atom.lhs.kind != "name":
            raise dsl.DSLError(f"left side of '{atom.text()}' must be a variable")
        var = self.lookup_var(atom.lhs.value)
        if var is None:
            raise dsl.DSLError(f"unknown variable {atom.lhs.value!r} in '{atom.text()}'")
        dom, get = var.domain, var.getter
        op = atom.op
        if op in dsl.SET_OPS:
            assert isinstance(atom.rhs, tuple)
            codes = frozenset(dom.encode(self.literal(o)) for o in atom.rhs)
            if op == "in":
                fn = lambda s, g=get, c=codes: g(s) in c  # noqa: E731
            else:
                fn = lambda s, g=get, c=codes: g(s) not in c  # noqa: E731
            return CAtom(var, op, codes, fn=fn)
        if dom.kind != "int" and op not in ("==", "!="):
            raise dsl.DSLError(f"'{op}' only works on integer variables: '{atom.text()}'")
        cmp = {
            "==": operator.eq,
            "!=": operator.ne,
            "<": operator.lt,
            "<=": operator.le,
            ">": operator.gt,
            ">=": operator.ge,
        }[op]
        rhs = atom.rhs
        assert isinstance(rhs, dsl.Operand)
        ref = self.as_ref(rhs)
        if ref is not None:
            table = self.translation(ref.domain, dom, atom.text())
            rget = ref.getter
            if table is None:
                fn = lambda s, g=get, r=rget, c=cmp: c(g(s), r(s))  # noqa: E731
            else:
                fn = lambda s, g=get, r=rget, c=cmp, t=table: c(g(s), t[r(s)])  # noqa: E731
            return CAtom(var, op, ref=ref, table=table, fn=fn)
        code = dom.encode(self.literal(rhs))
        if var.index is not None:
            idx = var.index
            if op == "==":
                fn = lambda s, i=idx, v=code: s[i] == v  # noqa: E731
            elif op == "!=":
                fn = lambda s, i=idx, v=code: s[i] != v  # noqa: E731
            else:
                fn = lambda s, i=idx, v=code, c=cmp: c(s[i], v)  # noqa: E731
        else:
            fn = lambda s, g=get, v=code, c=cmp: c(g(s), v)  # noqa: E731
        return CAtom(var, op, code, fn=fn)

    def as_ref(self, operand: dsl.Operand) -> Var | None:
        if operand.kind == "name" and "." in operand.value:
            var = self.lookup_var(operand.value)
            if var is None:
                raise dsl.DSLError(f"unknown variable {operand.value!r}")
            return var
        return None

    @staticmethod
    def literal(operand: dsl.Operand) -> Any:
        return operand.value

    def translation(self, src: Domain, dst: Domain, ctx: str) -> tuple | None:
        """Map codes of ``src`` onto codes of ``dst`` (None if identical)."""
        if src.same_as(dst):
            return None
        if src.kind != dst.kind:
            raise dsl.DSLError(f"cannot compare/copy {src.kind} with {dst.kind} in '{ctx}'")
        if src.kind == "enum":
            return tuple(dst.values.index(v) if v in dst.values else -1 for v in src.values)
        return None  # ints and bools share raw codes

    def compile_cond(self, cond: dsl.Condition) -> CCond:
        atoms = [self.compile_atom(a) for a in cond.atoms]
        if len(atoms) == 1:
            fn = atoms[0].fn
        else:
            fns = tuple(a.fn for a in atoms)
            fn = lambda s, fs=fns: any(f(s) for f in fs)  # noqa: E731
        return CCond(atoms, cond.text(), fn)

    def compile_effect(self, eff: dsl.Effect) -> CEffect:
        var = self.by_name.get(eff.target)
        if var is None:
            if eff.target in self.derived:
                raise dsl.DSLError(f"{eff.target!r} is derived and cannot be assigned")
            raise dsl.DSLError(f"unknown variable {eff.target!r} in '{eff.text()}'")
        if var.name == "turn":
            raise dsl.DSLError("the turn counter cannot be changed by effects")
        dom = var.domain
        if eff.op in ("+=", "-="):
            if dom.kind != "int":
                raise dsl.DSLError(f"'{eff.op}' needs an integer variable: '{eff.text()}'")
            val = self.literal(eff.value)
            if eff.value.kind != "lit" or isinstance(val, bool) or not isinstance(val, int):
                raise dsl.DSLError(f"'{eff.op}' needs an integer amount: '{eff.text()}'")
            delta = val if eff.op == "+=" else -val
            return CEffect(var, "add", delta, text=eff.text())
        ref = self.as_ref(eff.value)
        if ref is not None:
            table = self.translation(ref.domain, dom, eff.text())
            return CEffect(var, "copy", ref=ref, table=table, text=eff.text())
        return CEffect(var, "set", dom.encode(self.literal(eff.value)), text=eff.text())

    def compile_effects(self, effs: list[dsl.Effect | dsl.ConditionalEffect]) -> list[CEffect | CCondEffect]:
        out: list[CEffect | CCondEffect] = []
        for e in effs:
            if isinstance(e, dsl.ConditionalEffect):
                out.append(
                    CCondEffect(
                        [self.compile_cond(c) for c in e.conditions],
                        [self.compile_effect(x) for x in e.then],
                        [self.compile_effect(x) for x in e.otherwise],
                        e.text(),
                    )
                )
            else:
                out.append(self.compile_effect(e))
        return out

    # ------------------------------------------------------------ grounding
    def collections(self) -> dict[str, list[dict[str, Any]]]:
        sc = self.sc
        names = {r["id"]: r["name"] for r in sc["rooms"]}
        passages = []
        for kind, coll in (("door", sc["doors"]), ("vent", sc["vents"])):
            for e in coll:
                for a, b in ((e["a"], e["b"]), (e["b"], e["a"])):
                    passages.append(
                        {
                            "id": f"{e['id']}:{a}>{b}",
                            "kind": kind,
                            "via": e["id"],
                            "via_name": e["name"],
                            "from": a,
                            "from_name": names[a],
                            "to": b,
                            "to_name": names[b],
                            "clearance": e.get("clearance", 0),
                        }
                    )
        colls = {k: sc[k] for k in ("rooms", "doors", "vents", "cameras", "terminals", "npcs", "items")}
        colls["passages"] = passages
        return colls

    def bindings(self, spec: dict[str, Any], base: dict[str, Any]) -> Iterable[dict[str, Any]]:
        colls = self.collections_cache
        wheres = []
        for w in spec.get("where") or []:
            wheres.append((dsl.placeholder_roots(w) - set(base), dsl.parse_condition_template(w), w))
        envs = [dict(base)]
        bound: set[str] = set()
        pending = list(wheres)
        for b in spec.get("bind") or []:
            name, src = b["name"], b["from"]
            if name in base:
                raise dsl.DSLError(f"binding name {name!r} shadows a built-in placeholder")
            new_envs = []
            for env in envs:
                if src in colls:
                    values = colls[src]
                else:
                    values = dsl.lookup(src, env)
                    if not isinstance(values, list):
                        raise dsl.DSLError(f"bind source {src!r} is not a collection or list field")
                    values = [self.entities.get(v, v) if isinstance(v, str) else v for v in values]
                for v in values:
                    new_envs.append({**env, name: v})
            bound.add(name)
            ready = [w for w in pending if w[0] <= bound]
            pending = [w for w in pending if not w[0] <= bound]
            envs = [e for e in new_envs if all(self.where_ok(w, e) for w in ready)]
        for w in pending:
            if w[0] - bound:
                raise dsl.DSLError(f"'where' clause {w[2]!r} uses unbound names {sorted(w[0] - bound)}")
        if pending:
            envs = [e for e in envs if all(self.where_ok(w, e) for w in pending)]
        return envs

    @staticmethod
    def where_ok(where: tuple, env: dict[str, Any]) -> bool:
        for cond in dsl.instantiate_condition(where[1], env):
            if not dsl.eval_static(cond):
                return False
        return True

    def ground(self, spec: dict[str, Any], kind: str, base: dict[str, Any], start_index: int) -> list[Operator]:
        ops: list[Operator] = []
        cond_key = "pre" if kind == "action" else "when"
        eff_key = "eff" if kind == "action" else "then"
        label = f"{kind} '{spec['id']}'"
        try:
            envs = self.bindings(spec, base)
        except dsl.DSLError as exc:
            self.errors.append(f"{label}: {exc}")
            return ops
        for env in envs:
            args = [env[b["name"]] for b in spec.get("bind") or []]
            arg_ids = [a.get("id", "?") if isinstance(a, dict) else dsl.literal_text(a) for a in args]
            name = f"{spec['id']}({','.join(arg_ids)})" if arg_ids else spec["id"]
            try:
                conds: list[dsl.Condition] = []
                for c in spec.get(cond_key) or []:
                    conds.extend(dsl.instantiate_condition(c, env))
                effs: list[dsl.Effect | dsl.ConditionalEffect] = []
                for e in spec.get(eff_key) or []:
                    effs.extend(dsl.instantiate_effect(e, env))
                if spec.get(eff_key) and not effs:
                    continue  # e.g. "has.{n.carries} = true" for someone carrying nothing
                cconds = [self.compile_cond(c) for c in conds]
                ceffs = self.compile_effects(effs)
            except (dsl.DSLError, ValueError) as exc:
                self.errors.append(f"{label} as {name}: {exc}")
                continue
            op = Operator(
                index=start_index + len(ops),
                name=name,
                schema_id=spec["id"],
                kind=kind,
                ability=spec.get("ability", "rule"),
                text=dsl.render_text(spec.get("text", ""), env),
                conds=cconds,
                effects=ceffs,
            )
            finish_operator(op, self.by_name["monster.loc"].index)
            ops.append(op)
        return ops

    # --------------------------------------------------------------- build
    def build(self) -> Problem:
        sc, lib = self.sc, self.lib
        self.build_vars()
        self.derived = self.build_derived()
        self.collections_cache = self.collections()
        base = {"monster": sc["monster"], "security": sc["security"], "turn_limit": sc["turn_limit"]}
        disabled = set(sc["disabled_core"])

        actions: list[Operator] = []
        for spec in [a for a in lib.actions if a["id"] not in disabled]:
            actions.extend(self.ground(spec, "action", base, len(actions)))
        for spec in sc["actions"]:
            grounded = self.ground(spec, "action", base, len(actions))
            if not grounded and not self.errors:
                self.warnings.append(f"custom action '{spec['id']}' never applies to anything in this facility")
            actions.extend(grounded)
        rules: list[Operator] = []
        for spec in [r for r in lib.rules if r["id"] not in disabled] + list(sc["rules"]):
            rules.extend(self.ground(spec, "rule", base, len(rules)))
        losses: list[Loss] = []
        for spec in list(lib.losses) + list(sc["losses"]):
            try:
                conds = []
                for c in spec["when"]:
                    conds.extend(self.compile_cond(x) for x in dsl.instantiate_condition(c, base))
                if not conds:
                    raise dsl.DSLError("a loss needs at least one condition")
            except (dsl.DSLError, ValueError) as exc:
                self.errors.append(f"loss '{spec['id']}': {exc}")
                continue
            losses.append(Loss(spec["id"], spec.get("text", ""), conds))

        goal: list[CCond] = []
        exits = [r["id"] for r in sc["rooms"] if r["exit"]]
        try:
            goal.append(self.compile_cond(dsl.instantiate_condition("monster.loc in {exits}", {"exits": exits})[0]))
            for g in sc["goal_extra"]:
                goal.extend(self.compile_cond(c) for c in dsl.instantiate_condition(g, base))
        except (dsl.DSLError, ValueError) as exc:
            self.errors.append(f"goal_extra: {exc}")

        if self.errors:
            raise ScenarioError(_dedupe(self.errors))

        problem = Problem(
            scenario=sc,
            vars=self.vars,
            derived=self.derived,
            init=tuple(self.init),
            actions=actions,
            rules=rules,
            losses=losses,
            goal=goal,
            turn_limit=sc["turn_limit"],
            room_ids=self.room_domain.values,
            exits=frozenset(self.room_domain.encode(e) for e in exits),
            warnings=self.warnings,
        )
        index_actions(problem)
        problem.dist = movement_distances(problem)
        return problem


def _dedupe(items: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for i in items:
        seen.setdefault(i, None)
    return list(seen)


# --------------------------------------------------------------------------
# Closures, indexing and the movement heuristic
# --------------------------------------------------------------------------


def _effect_fn(e: CEffect) -> Callable[[Any, list], None]:
    i = e.var.index
    if e.op == "set":
        v = e.const

        def f(s, w, i=i, v=v):
            w[i] = v

        return f
    if e.op == "add":
        d, lo, hi = e.const, e.var.domain.lo, e.var.domain.hi

        def f(s, w, i=i, d=d, lo=lo, hi=hi):
            x = w[i] + d
            w[i] = lo if x < lo else hi if x > hi else x

        return f
    get, table = e.ref.getter, e.table  # type: ignore[union-attr]
    dom = e.var.domain

    def f(s, w, i=i, get=get, table=table, dom=dom):
        c = get(s)
        if table is not None:
            c = table[c]
            if c < 0:
                return
        if dom.kind == "int":
            c = dom.lo if c < dom.lo else dom.hi if c > dom.hi else c
        w[i] = c

    return f


class ExprBuilder:
    """Turns compiled conditions into one Python expression.

    Evaluating a whole precondition list as a single generated lambda is
    several times faster than calling one closure per atom. Only integers,
    state indices and names bound in ``ns`` (our own getters and constant
    sets) ever appear in the generated source, never scenario text.
    """

    def __init__(self) -> None:
        self.ns: dict[str, Any] = {"__builtins__": {}}
        self._n = 0

    def const(self, value: Any) -> str:
        name = f"_k{self._n}"
        self._n += 1
        self.ns[name] = value
        return name

    def _read(self, var: Var, arg: str) -> str:
        if var.index is not None:
            return f"{arg}[{int(var.index)}]"
        return f"{self.const(var.getter)}({arg})"

    def atom(self, a: CAtom, arg: str) -> str:
        lhs = self._read(a.var, arg)
        if a.op in ("in", "not in"):
            return f"({lhs} {a.op} {self.const(a.const)})"
        op = {"==": "==", "!=": "!=", "<": "<", "<=": "<=", ">": ">", ">=": ">="}[a.op]
        if a.ref is not None:
            rhs = self._read(a.ref, arg)
            if a.table is not None:
                rhs = f"{self.const(a.table)}[{rhs}]"
            return f"({lhs} {op} {rhs})"
        return f"({lhs} {op} {int(a.const)})"

    def cond(self, c: CCond, arg: str) -> str:
        parts = [self.atom(a, arg) for a in c.atoms]
        return parts[0] if len(parts) == 1 else "(" + " or ".join(parts) + ")"

    def conds(self, cs: list[CCond], arg: str) -> str:
        return " and ".join(self.cond(c, arg) for c in cs) or "True"


def make_test(conds: list[CCond]) -> StateFn:
    b = ExprBuilder()
    return eval(f"lambda s: {b.conds(conds, 's')}", b.ns)  # noqa: S307 - generated from ints/indices only


def finish_operator(op: Operator, loc_index: int) -> None:
    op.test = make_test(op.conds)
    for c in op.conds:
        if len(c.atoms) == 1:
            a = c.atoms[0]
            if a.var.index == loc_index and a.op == "==" and a.ref is None:
                op.loc_req = a.const
                break
    steps: list[Callable[[Any, list], None]] = []
    for e in op.effects:
        if isinstance(e, CCondEffect):
            conds = tuple(c.fn for c in e.conds)
            then = tuple(_effect_fn(x) for x in e.then)
            otherwise = tuple(_effect_fn(x) for x in e.otherwise)

            def cond_step(s, w, conds=conds, then=then, otherwise=otherwise):
                for c in conds:
                    if not c(s):
                        for f in otherwise:
                            f(s, w)
                        return
                for f in then:
                    f(s, w)

            steps.append(cond_step)
        else:
            steps.append(_effect_fn(e))
    steps_t = tuple(steps)

    def apply(s, w, steps=steps_t):
        for f in steps:
            f(s, w)

    op.apply = apply


def index_actions(problem: Problem) -> None:
    by_loc: dict[int, list[Operator]] = defaultdict(list)
    any_loc: list[Operator] = []
    for a in problem.actions:
        if a.loc_req is None:
            any_loc.append(a)
        else:
            by_loc[a.loc_req].append(a)
    problem.by_loc = dict(by_loc)
    problem.any_loc = any_loc


def loc_moves(op: Operator, loc_index: int, n_rooms: int) -> set[int]:
    """Rooms an operator can put the monster in (all rooms if it copies a value)."""
    targets: set[int] = set()
    flat: list[CEffect] = []
    for e in op.effects:
        if isinstance(e, CCondEffect):
            flat.extend(e.then)
            flat.extend(e.otherwise)
        else:
            flat.append(e)
    for e in flat:
        if e.var.index == loc_index:
            if e.op == "set":
                targets.add(e.const)
            else:
                targets.update(range(1, n_rooms))
    return targets


def movement_distances(problem: Problem) -> list[float]:
    """Fewest moves from each room to an exit, ignoring every precondition but location.

    Each action moves the monster at most one edge of this graph, so the
    distance never overestimates the turns left: an admissible, consistent
    heuristic for A*.
    """
    n = len(problem.room_ids)
    loc = problem.loc
    edges: list[tuple[int | None, int]] = []
    for op in problem.actions + problem.rules:
        for t in loc_moves(op, loc, n):
            edges.append((op.loc_req, t))
    inf = float("inf")
    dist = [inf] * n
    for e in problem.exits:
        dist[e] = 0
    changed = True
    while changed:
        changed = False
        for src, dst in edges:
            d = dist[dst] + 1
            if src is None:
                for r in range(1, n):
                    if dist[r] > d:
                        dist[r] = d
                        changed = True
            elif dist[src] > d:
                dist[src] = d
                changed = True
    return dist

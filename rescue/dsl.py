"""The tiny statement language used by actions, rules, losses and goals.

Every precondition, effect and trigger in the game is a string in this
language, so scenarios (hand-written or LLM-generated) stay pure data.

Templates may contain ``{binding.field}`` placeholders. They are filled
from an action's parameter bindings when the action is grounded against a
concrete facility, e.g. ``door.{p.via}.locked == false`` becomes
``door.d_cell.locked == false``.

Grammar (after interpolation)::

    condition := atom ( "||" atom )*
    atom      := operand CMP operand                 CMP: == != < <= > >=
               | operand ("in" | "not in") list
    list      := "[" operand ("," operand)* "]"  |  {placeholder resolving to a list}
    effect    := var ("=" | "+=" | "-=") operand
    operand   := integer | true | false | none | identifier | dotted.var.name

Left-hand sides of conditions and effects name state variables. A
right-hand side that contains a dot and names a variable is a variable
reference (``npc.dr_ito.loc == monster.loc``); anything else is a literal.

List expansion: a placeholder that resolves to a list anywhere other than
the right side of ``in`` repeats the whole statement once per element.
``has.{n.carries} = true`` with ``carries = [badge, pass]`` yields two
effects; with an empty list it yields none. A condition expands into a
conjunction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

PLACEHOLDER = re.compile(r"\{([^{}]*)\}")
TOKEN = re.compile(
    r"\s*(?:"
    r"(?P<op>==|!=|<=|>=|\+=|-=|<|>|=)"
    r"|(?P<punct>[\[\],])"
    r"|(?P<word>-?(?:[A-Za-z0-9_.]|\{[^{}]*\})+)"
    r")"
)
CMP_OPS = ("==", "!=", "<", "<=", ">", ">=")
SET_OPS = ("in", "not in")
EFFECT_OPS = ("=", "+=", "-=")
INT_RE = re.compile(r"-?\d+$")


class DSLError(ValueError):
    """A statement could not be parsed or instantiated."""


@dataclass(frozen=True)
class Operand:
    """An instantiated operand.

    ``kind`` is ``"lit"`` for typed literals (int, bool, None or a string
    that came verbatim from a placeholder) and ``"name"`` for bare words
    whose meaning (variable or enum literal) the compiler decides.
    """

    kind: str
    value: Any

    def text(self) -> str:
        return literal_text(self.value)


@dataclass(frozen=True)
class Atom:
    lhs: Operand
    op: str
    rhs: Operand | tuple[Operand, ...]

    def text(self) -> str:
        if isinstance(self.rhs, tuple):
            rhs = "[" + ", ".join(o.text() for o in self.rhs) + "]"
        else:
            rhs = self.rhs.text()
        return f"{self.lhs.text()} {self.op} {rhs}"


@dataclass(frozen=True)
class Condition:
    """A disjunction of atoms."""

    atoms: tuple[Atom, ...]

    def text(self) -> str:
        return " || ".join(a.text() for a in self.atoms)


@dataclass(frozen=True)
class Effect:
    target: str
    op: str
    value: Operand

    def text(self) -> str:
        return f"{self.target} {self.op} {self.value.text()}"


@dataclass(frozen=True)
class ConditionalEffect:
    conditions: tuple[Condition, ...]
    then: tuple[Effect, ...]
    otherwise: tuple[Effect, ...]

    def text(self) -> str:
        conds = " && ".join(c.text() for c in self.conditions)
        then = "; ".join(e.text() for e in self.then)
        out = f"if {conds} then {then}"
        if self.otherwise:
            out += " else " + "; ".join(e.text() for e in self.otherwise)
        return out


def literal_text(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "none"
    return str(value)


# --------------------------------------------------------------------------
# Tokenising and parsing templates (placeholders still unresolved)
# --------------------------------------------------------------------------


def _tokenize(text: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    pos = 0
    text = text.rstrip()
    while pos < len(text):
        m = TOKEN.match(text, pos)
        if not m or m.end() == pos:
            raise DSLError(f"unexpected character {text[pos:pos + 10]!r} in {text!r}")
        pos = m.end()
        for kind in ("op", "punct", "word"):
            if m.group(kind) is not None:
                tokens.append((kind, m.group(kind)))
                break
    return tokens


@dataclass(frozen=True)
class _AtomT:
    lhs: str
    op: str
    rhs: str | tuple[str, ...]


def _parse_atom_template(text: str) -> _AtomT:
    tokens = _tokenize(text)
    if len(tokens) < 3:
        raise DSLError(f"incomplete condition {text!r}")
    kind, lhs = tokens[0]
    if kind != "word":
        raise DSLError(f"condition must start with a name or value: {text!r}")
    k1, v1 = tokens[1]
    rest = tokens[2:]
    if k1 == "op" and v1 in CMP_OPS:
        op = v1
    elif k1 == "word" and v1 == "in":
        op = "in"
    elif k1 == "word" and v1 == "not" and rest and rest[0] == ("word", "in"):
        op = "not in"
        rest = rest[1:]
    else:
        raise DSLError(f"expected a comparison operator after {lhs!r} in {text!r}")
    if op in SET_OPS:
        if len(rest) == 1 and rest[0][0] == "word":
            return _AtomT(lhs, op, rest[0][1])
        if not rest or rest[0] != ("punct", "[") or rest[-1] != ("punct", "]"):
            raise DSLError(f"'{op}' needs a [list] in {text!r}")
        items: list[str] = []
        expect_item = True
        for kind, val in rest[1:-1]:
            if expect_item and kind == "word":
                items.append(val)
                expect_item = False
            elif not expect_item and (kind, val) == ("punct", ","):
                expect_item = True
            else:
                raise DSLError(f"malformed list in {text!r}")
        return _AtomT(lhs, op, tuple(items))
    if len(rest) != 1 or rest[0][0] != "word":
        raise DSLError(f"expected a single value after '{op}' in {text!r}")
    return _AtomT(lhs, op, rest[0][1])


def parse_condition_template(text: str) -> tuple[_AtomT, ...]:
    if not isinstance(text, str) or not text.strip():
        raise DSLError(f"empty condition {text!r}")
    return tuple(_parse_atom_template(part) for part in text.split("||"))


@dataclass(frozen=True)
class _EffectT:
    target: str
    op: str
    value: str


def parse_effect_template(text: str) -> _EffectT:
    if not isinstance(text, str) or not text.strip():
        raise DSLError(f"empty effect {text!r}")
    tokens = _tokenize(text)
    if len(tokens) != 3 or tokens[0][0] != "word" or tokens[1][0] != "op" or tokens[2][0] != "word":
        raise DSLError(f"effects look like 'var = value', 'var += n' or 'var -= n': {text!r}")
    op = tokens[1][1]
    if op not in EFFECT_OPS:
        raise DSLError(f"unknown effect operator {op!r} in {text!r}")
    return _EffectT(tokens[0][1], op, tokens[2][1])


# --------------------------------------------------------------------------
# Resolving placeholders against bindings
# --------------------------------------------------------------------------


def lookup(path: str, env: Mapping[str, Any], overrides: Mapping[str, Any] | None = None) -> Any:
    """Resolve ``name.field.field`` against the binding environment."""
    path = path.strip()
    if overrides and path in overrides:
        return overrides[path]
    parts = path.split(".")
    if parts[0] not in env:
        raise DSLError(f"unknown placeholder {{{path}}}")
    value = env[parts[0]]
    for part in parts[1:]:
        if isinstance(value, Mapping) and part in value:
            value = value[part]
        else:
            raise DSLError(f"{{{path}}}: no field {part!r}")
    return value


def _scalar(value: Any) -> Any:
    if isinstance(value, Mapping):
        if "id" not in value:
            raise DSLError("cannot use an object without an id as a value")
        return value["id"]
    return value


def _parse_word(word: str) -> Operand:
    if INT_RE.match(word):
        return Operand("lit", int(word))
    low = word.lower()
    if low == "true":
        return Operand("lit", True)
    if low == "false":
        return Operand("lit", False)
    if low in ("none", "null"):
        return Operand("lit", None)
    return Operand("name", word)


def _resolve_word(word: str, env: Mapping[str, Any], overrides: Mapping[str, Any]) -> Operand:
    m = PLACEHOLDER.fullmatch(word)
    if m:
        value = _scalar(lookup(m.group(1), env, overrides))
        if isinstance(value, list):
            raise DSLError(f"{word} is a list here")
        return Operand("lit", value)

    def sub(match: re.Match[str]) -> str:
        value = _scalar(lookup(match.group(1), env, overrides))
        if isinstance(value, list):
            raise DSLError(f"{match.group(0)} is a list here")
        return literal_text(value)

    return _parse_word(PLACEHOLDER.sub(sub, word))


def _list_paths(words: list[str], env: Mapping[str, Any]) -> list[str]:
    """Placeholder paths in scalar positions whose value is a list."""
    found: list[str] = []
    for word in words:
        for m in PLACEHOLDER.finditer(word):
            if isinstance(_scalar(lookup(m.group(1), env)), list) and m.group(1) not in found:
                found.append(m.group(1))
    return found


def _expansions(words: list[str], env: Mapping[str, Any]) -> list[dict[str, Any]]:
    paths = _list_paths(words, env)
    if not paths:
        return [{}]
    if len(paths) > 1:
        raise DSLError(f"only one list placeholder may be expanded per statement, got {paths}")
    return [{paths[0]: item} for item in lookup(paths[0], env)]


def instantiate_condition(template: str | tuple[_AtomT, ...], env: Mapping[str, Any]) -> list[Condition]:
    atoms_t = parse_condition_template(template) if isinstance(template, str) else template
    scalar_words: list[str] = []
    for a in atoms_t:
        scalar_words.append(a.lhs)
        if isinstance(a.rhs, tuple):
            scalar_words.extend(a.rhs)
        elif a.op not in SET_OPS:
            scalar_words.append(a.rhs)
    out: list[Condition] = []
    for overrides in _expansions(scalar_words, env):
        atoms: list[Atom] = []
        for a in atoms_t:
            lhs = _resolve_word(a.lhs, env, overrides)
            rhs: Operand | tuple[Operand, ...]
            if isinstance(a.rhs, tuple):
                rhs = tuple(_resolve_word(w, env, overrides) for w in a.rhs)
            elif a.op in SET_OPS:
                m = PLACEHOLDER.fullmatch(a.rhs)
                if not m:
                    raise DSLError(f"'{a.op}' needs a [list] or a list placeholder, got {a.rhs!r}")
                value = lookup(m.group(1), env, overrides)
                if not isinstance(value, list):
                    raise DSLError(f"{a.rhs} is not a list")
                rhs = tuple(Operand("lit", _scalar(v)) for v in value)
            else:
                rhs = _resolve_word(a.rhs, env, overrides)
            atoms.append(Atom(lhs, a.op, rhs))
        out.append(Condition(tuple(atoms)))
    return out


def instantiate_effect(template: str | Mapping[str, Any], env: Mapping[str, Any]) -> list[Effect | ConditionalEffect]:
    if isinstance(template, Mapping):
        unknown = set(template) - {"if", "then", "else"}
        if unknown or "if" not in template or "then" not in template:
            raise DSLError("conditional effects look like {if: [...], then: [...], else: [...]}")
        conds: list[Condition] = []
        for c in template["if"] or []:
            conds.extend(instantiate_condition(c, env))
        then: list[Effect] = []
        for e in template["then"] or []:
            then.extend(_plain_effects(e, env))
        otherwise: list[Effect] = []
        for e in template.get("else") or []:
            otherwise.extend(_plain_effects(e, env))
        if not then and not otherwise:
            return []
        return [ConditionalEffect(tuple(conds), tuple(then), tuple(otherwise))]
    return list(_plain_effects(template, env))


def _plain_effects(template: Any, env: Mapping[str, Any]) -> list[Effect]:
    if not isinstance(template, str):
        raise DSLError("conditional effects cannot be nested")
    et = parse_effect_template(template)
    out: list[Effect] = []
    for overrides in _expansions([et.target, et.value], env):
        target = _resolve_word(et.target, env, overrides)
        if target.kind != "name":
            raise DSLError(f"effect target must be a variable name, got {target.text()!r}")
        out.append(Effect(target.value, et.op, _resolve_word(et.value, env, overrides)))
    return out


# --------------------------------------------------------------------------
# Static evaluation (used for 'where' clauses at grounding time)
# --------------------------------------------------------------------------


def compare(op: str, a: Any, b: Any) -> bool:
    if op == "==":
        return a == b
    if op == "!=":
        return a != b
    if op in SET_OPS:
        inside = a in b
        return inside if op == "in" else not inside
    if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, int) or not isinstance(b, int):
        return False
    if op == "<":
        return a < b
    if op == "<=":
        return a <= b
    if op == ">":
        return a > b
    if op == ">=":
        return a >= b
    raise DSLError(f"unknown operator {op!r}")


def eval_static(condition: Condition) -> bool:
    for atom in condition.atoms:
        rhs = tuple(o.value for o in atom.rhs) if isinstance(atom.rhs, tuple) else atom.rhs.value
        if compare(atom.op, atom.lhs.value, rhs):
            return True
    return False


def placeholder_roots(template: Any) -> set[str]:
    """Binding names a template refers to (used to evaluate 'where' early)."""
    if isinstance(template, Mapping):
        roots: set[str] = set()
        for v in template.values():
            roots |= placeholder_roots(v)
        return roots
    if isinstance(template, (list, tuple)):
        roots = set()
        for v in template:
            roots |= placeholder_roots(v)
        return roots
    if isinstance(template, str):
        return {m.group(1).strip().split(".")[0] for m in PLACEHOLDER.finditer(template)}
    return set()


def render_text(template: str, env: Mapping[str, Any]) -> str:
    """Fill a narrative template. Objects render as their name, lists as a comma list."""

    def show(value: Any) -> str:
        if isinstance(value, Mapping):
            return str(value.get("name") or value.get("id"))
        if isinstance(value, list):
            return ", ".join(show(v) for v in value) or "nothing"
        return literal_text(value)

    def sub(match: re.Match[str]) -> str:
        try:
            return show(lookup(match.group(1), env))
        except DSLError:
            return match.group(0)

    return PLACEHOLDER.sub(sub, template or "")

"""The core rule library (monster abilities, world reactions, losses)."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

CORE_PATH = Path(__file__).with_name("core.yaml")


@dataclass(frozen=True)
class Library:
    actions: tuple[dict[str, Any], ...]
    rules: tuple[dict[str, Any], ...]
    losses: tuple[dict[str, Any], ...]
    text: str = field(repr=False)

    @property
    def disableable(self) -> set[str]:
        return {a["id"] for a in self.actions} | {r["id"] for r in self.rules}

    @property
    def all_ids(self) -> set[str]:
        return self.disableable | {l["id"] for l in self.losses}


def _normalise(spec: dict[str, Any]) -> dict[str, Any]:
    out = dict(spec)
    out.setdefault("bind", [])
    out.setdefault("where", [])
    out.setdefault("text", "")
    return out


@lru_cache(maxsize=None)
def load_library(path: str | None = None) -> Library:
    p = Path(path) if path else CORE_PATH
    text = p.read_text()
    data = yaml.safe_load(text)
    return Library(
        actions=tuple(_normalise(a) for a in data["actions"]),
        rules=tuple(_normalise(r) for r in data["rules"]),
        losses=tuple(dict(l) for l in data["losses"]),
        text=text,
    )


EXAMPLES_DIR = Path(__file__).with_name("examples")


def example_paths() -> list[Path]:
    return sorted(EXAMPLES_DIR.glob("*.json"))


def load_example(name: str = "site_theta") -> dict[str, Any]:
    import json

    return json.loads((EXAMPLES_DIR / f"{name}.json").read_text())

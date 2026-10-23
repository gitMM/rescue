"""Seeded design briefs, so batches of generated scenarios stay varied."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

SETTINGS = [
    "an arctic ice-core research station cut off by a blizzard",
    "a deep-sea habitat bolted to a trench wall",
    "a decommissioned missile silo converted into a black-site lab",
    "a corporate skyscraper whose top floors hide a bioweapons programme",
    "a desert prison with a classified wing nobody is supposed to know about",
    "a mountain particle-physics lab where the specimen arrived through a breach",
    "an orbital-launch complex preparing to ship the specimen off-world",
    "a Cold War bunker under a sleepy university campus",
    "a quarantine hospital ship anchored offshore",
    "a cryptozoology archive beneath a natural history museum",
    "a rainforest field station run by a private military contractor",
    "an abandoned subway line repurposed as a containment facility",
]

TWISTS = [
    "a scheduled supply delivery opens an alternative way out at a fixed turn",
    "a containment purge sweeps one wing if the alarm gets high",
    "one scientist secretly sympathises with the specimen and leaves a door code lying around",
    "a power cut at a fixed turn disables every camera for a while",
    "a guard dog unit patrols and can smell the creature even in disguise",
    "the exit requires two different credentials from two different people",
    "a flooded maintenance tunnel can only be crossed in liquid form",
    "the director is visiting tonight and carries the master clearance",
    "a decoy specimen transfer can be triggered from a terminal to pull guards away",
    "the vents are filled with a nerve agent unless a filter terminal is hacked",
]

SIZES = {
    "small": "8-10 rooms, 3-5 NPCs, 1-2 terminals, 2-4 vents",
    "medium": "10-14 rooms, 5-7 NPCs, 2-3 terminals, 3-5 vents",
    "large": "14-18 rooms, 6-9 NPCs, 3-4 terminals, 4-6 vents",
}

DIFFICULTIES = {
    "easy": "generous turn limit (about 3x the fastest route), forgiving alarm thresholds, few armed guards",
    "normal": "turn limit about 2x the fastest route, two armed guards on overlapping patrols",
    "hard": "tight turn limit (about 1.5x the fastest route), low alarm thresholds, guards covering most chokepoints",
}


@dataclass(frozen=True)
class Brief:
    seed: int
    setting: str
    twist: str
    size: str
    difficulty: str

    def to_prompt(self) -> str:
        return "\n".join(
            [
                "Design a new escape scenario.",
                f"Setting: {self.setting}.",
                f"Twist to build in: {self.twist}.",
                f"Size: {self.size} ({SIZES[self.size]}).",
                f"Difficulty: {self.difficulty} ({DIFFICULTIES[self.difficulty]}).",
                "Make it clearly different from the example scenario: new layout, new people, new set pieces.",
                "Return the scenario as a single JSON object following the schema.",
            ]
        )

    def to_dict(self) -> dict:
        return asdict(self)


def make_brief(
    seed: int,
    setting: str | None = None,
    twist: str | None = None,
    size: str | None = None,
    difficulty: str | None = None,
) -> Brief:
    rng = random.Random(seed)
    return Brief(
        seed=seed,
        setting=setting or rng.choice(SETTINGS),
        twist=twist or rng.choice(TWISTS),
        size=size or rng.choice(["small", "medium", "medium", "large"]),
        difficulty=difficulty or rng.choice(["easy", "normal", "normal", "hard"]),
    )

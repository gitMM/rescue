"""Curated scenarios in scenarios/ must stay accepted as the engine and core library evolve."""

import copy
import json
from pathlib import Path

import pytest
from conftest import FAST_GATES

from rescue.validate import format_report, validate_scenario

SCENARIOS = sorted((Path(__file__).parent.parent / "scenarios").glob("*.json"))


@pytest.mark.parametrize("path", SCENARIOS, ids=[p.stem for p in SCENARIOS])
def test_curated_scenario_is_accepted(path):
    data = json.loads(path.read_text())
    assert data["id"] == path.stem, "the scenario id should match its file name"
    report = validate_scenario(data, copy.deepcopy(FAST_GATES))
    assert report["accepted"], format_report(report, steps=False)

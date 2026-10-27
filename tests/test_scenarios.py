"""Curated scenarios in scenarios/ must stay accepted as the engine and core library evolve."""

from pathlib import Path

import pytest
from conftest import load

from rescue.validate import Gates, format_report, validate_scenario

SCENARIOS = sorted((Path(__file__).parent.parent / "scenarios").glob("*.json"))


@pytest.mark.parametrize("path", SCENARIOS, ids=[p.stem for p in SCENARIOS])
def test_curated_scenario_is_accepted(path):
    data = load(path)
    assert data["id"] == path.stem, "the scenario id should match its file name"
    # Full default gates: the same verdict `python -m rescue validate` gives.
    report = validate_scenario(data, Gates())
    assert report["accepted"], format_report(report, steps=False)

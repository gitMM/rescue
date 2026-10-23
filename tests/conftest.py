import copy
import json
from pathlib import Path

import pytest

from rescue.library import EXAMPLES_DIR
from rescue.validate import Gates, validate_scenario

FIXTURES = Path(__file__).parent / "fixtures"
THETA_PATH = EXAMPLES_DIR / "site_theta.json"
TINY_PATH = FIXTURES / "tiny.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


@pytest.fixture
def tiny() -> dict:
    return load(TINY_PATH)


@pytest.fixture(scope="session")
def theta_data() -> dict:
    return load(THETA_PATH)


@pytest.fixture
def theta(theta_data) -> dict:
    return copy.deepcopy(theta_data)


FAST_GATES = Gates(explore_cap=8000, search_cap=60_000)


@pytest.fixture
def fast_gates() -> Gates:
    """Default gates with smaller search budgets so the suite stays quick."""
    return copy.deepcopy(FAST_GATES)


@pytest.fixture(scope="session")
def theta_report(theta_data) -> dict:
    """The example's full validation report, computed once per test session."""
    return validate_scenario(copy.deepcopy(theta_data), copy.deepcopy(FAST_GATES))


def guarded(tiny: dict) -> dict:
    """tiny + an armed guard pacing the hall and a camera on it."""
    sc = copy.deepcopy(tiny)
    sc["npcs"].append(
        {
            "id": "guard",
            "name": "the guard",
            "role": "guard",
            "clearance": 2,
            "armed": True,
            "phone": False,
            "patrol": ["hall", "hall", "yard"],
            "carries": [],
            "knows": [],
        }
    )
    sc["cameras"].append({"id": "cam_hall", "name": "the hall camera", "room": "hall"})
    return sc

from __future__ import annotations

from pathlib import Path

import pytest

from piicorpus.config import load_config
from piicorpus.generator import generate

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def demo_config_path() -> Path:
    return ROOT / "configs" / "demo.toml"


@pytest.fixture()
def generated_demo(tmp_path: Path, demo_config_path: Path) -> Path:
    output = tmp_path / "corpus"
    generate(load_config(demo_config_path), output)
    return output

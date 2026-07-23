from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_corpus_line_endings_are_pinned_for_git_checkouts() -> None:
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()

    assert "*.json text eol=lf" in attributes
    assert "*.jsonl text eol=lf" in attributes


def test_sdist_includes_git_attributes() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    included = pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]

    assert "/.gitattributes" in included

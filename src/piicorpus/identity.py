"""Stable, independently verifiable record identity helpers."""

from __future__ import annotations

import hashlib
import re

from .config import CorpusConfig

NAMESPACE_RE = re.compile(
    r"^piicorpus/(?P<split>train|eval|holdout)/"
    r"(?P<family>[a-z][a-z0-9_]{1,63})/(?P<index>\d{5})$"
)


def derive_case_id(
    config: CorpusConfig,
    generator_version: str,
    split: str,
    family: str,
    index: int,
    text: str,
) -> str:
    """Derive the public case ID from immutable generation inputs and emitted text."""
    material = f"{config.digest}|{generator_version}|{split}|{family}|{index}|{text}"
    return "pc-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]


def namespace_index(namespace: str, split: str, family: str) -> int | None:
    """Return the exact namespace index, rejecting inconsistent namespace components."""
    match = NAMESPACE_RE.fullmatch(namespace)
    if not match:
        return None
    if match.group("split") != split or match.group("family") != family:
        return None
    return int(match.group("index"))

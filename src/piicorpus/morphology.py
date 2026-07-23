"""Detector-neutral value-shape and template-skeleton helpers."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

from .models import Annotation

ShapeMatcher = Callable[[str], bool]

_SHAPE_REGISTRY: dict[str, ShapeMatcher] = {}


def register_shape(name: str, matcher: str | ShapeMatcher, *, replace: bool = False) -> None:
    """Register a named value shape; string matchers are treated as full-match regexes.

    Matchers are consulted in registration order, so register specific shapes
    before broader catch-all shapes. Unmatched values fall back to a generic
    character-class signature.
    """
    if name in _SHAPE_REGISTRY and not replace:
        raise ValueError(f"shape is already registered: {name}")
    if isinstance(matcher, str):
        compiled = re.compile(matcher)
        _SHAPE_REGISTRY[name] = lambda value: compiled.fullmatch(value) is not None
    else:
        _SHAPE_REGISTRY[name] = matcher


def registered_shapes() -> tuple[str, ...]:
    return tuple(_SHAPE_REGISTRY)


def shape_signature(value: str) -> str:
    for name, matcher in _SHAPE_REGISTRY.items():
        if matcher(value):
            return name
    mapped = []
    for char in value:
        if char.isalpha():
            mapped.append("A")
        elif char.isdigit():
            mapped.append("9")
        elif char.isspace():
            mapped.append(" ")
        else:
            mapped.append(char)
    return "generic:" + re.sub(r"A+", lambda m: f"A{len(m.group(0))}", "".join(mapped))


def normalized_template_skeleton(
    text: str,
    annotations: Iterable[Annotation],
    *,
    persona: str | None = None,
    organization: str | None = None,
) -> str:
    parts: list[str] = []
    position = 0
    for annotation in sorted(annotations, key=lambda a: a.start):
        parts.append(text[position : annotation.start])
        parts.append("<entity>")
        position = annotation.end
    parts.append(text[position:])
    normalized = "".join(parts)
    for surface, replacement in ((persona, "<persona>"), (organization, "<organization>")):
        if surface:
            normalized = normalized.replace(surface, replacement)
    normalized = re.sub(r"\b\d+\b", "#", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip().casefold()
    return normalized


def body_fingerprint(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()

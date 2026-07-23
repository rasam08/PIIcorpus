"""Detector-neutral value-shape and template-skeleton helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable

from .models import Annotation


def shape_signature(value: str) -> str:
    lowered = value.casefold().strip()
    if lowered.startswith("synthetic "):
        return "spoken_synthetic"
    if re.fullmatch(r"SYN-DATE-\d{4}-\d{2}-\d{2}", value):
        return "synthetic_calendar"
    if re.fullmatch(r"SYN-ID-[A-Z]\d{5}", value):
        return "synthetic_alpha_five"
    if re.fullmatch(r"SYN-ID-[A-Z]{2}\d{4}", value):
        return "synthetic_two_alpha_four"
    if re.fullmatch(r"SYN-ID-[A-Z]{3}-\d{3}", value):
        return "synthetic_segmented"
    if value.startswith("SYN-ID-"):
        return "synthetic_noisy"
    if re.fullmatch(r"BAD-[A-Z]\d{7}", value):
        return "exclusive_bad_alpha_seven"
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

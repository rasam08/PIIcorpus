"""Strict marked-text parsing with code-point and UTF-8 byte offsets."""

from __future__ import annotations

import re
from collections.abc import Iterable

from .models import Annotation


class AnnotationError(ValueError):
    """Raised for malformed markup or inconsistent spans."""


LABEL_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,63}$")


def _byte_offset(text: str, position: int) -> int:
    return len(text[:position].encode("utf-8"))


def parse_marked(
    text: str, allowed_labels: Iterable[str] | None = None
) -> tuple[str, tuple[Annotation, ...]]:
    """Convert ``[[LABEL:value]]`` markup into clean text and exact spans."""
    allowed = set(allowed_labels) if allowed_labels is not None else None
    clean_parts: list[str] = []
    annotations: list[Annotation] = []
    source_pos = 0
    clean_length = 0

    while source_pos < len(text):
        opening = text.find("[[", source_pos)
        stray_close = text.find("]]", source_pos)
        if stray_close != -1 and (opening == -1 or stray_close < opening):
            raise AnnotationError("closing annotation marker has no matching opening marker")
        if opening == -1:
            tail = text[source_pos:]
            clean_parts.append(tail)
            clean_length += len(tail)
            break
        prefix = text[source_pos:opening]
        clean_parts.append(prefix)
        clean_length += len(prefix)
        closing = text.find("]]", opening + 2)
        if closing == -1:
            raise AnnotationError("annotation marker is not closed")
        inner = text[opening + 2 : closing]
        if "[[" in inner or "]]" in inner:
            raise AnnotationError("nested annotation markers are not allowed")
        if ":" not in inner:
            raise AnnotationError("annotation marker must contain LABEL:value")
        label_raw, value_raw = inner.split(":", 1)
        label = label_raw.strip()
        value = value_raw.strip()
        if not LABEL_RE.fullmatch(label):
            raise AnnotationError("annotation label is malformed")
        if allowed is not None and label not in allowed:
            raise AnnotationError(f"annotation label is not configured: {label}")
        if not value:
            raise AnnotationError("annotation value is empty")
        if "[[" in value or "]]" in value:
            raise AnnotationError("nested annotation markers are not allowed")
        start = clean_length
        clean_parts.append(value)
        clean_length += len(value)
        annotations.append(
            Annotation(
                entity_type=label,
                start=start,
                end=clean_length,
                byte_start=0,
                byte_end=0,
                text=value,
            )
        )
        source_pos = closing + 2

    clean = "".join(clean_parts)
    completed = tuple(
        Annotation(
            entity_type=a.entity_type,
            start=a.start,
            end=a.end,
            byte_start=_byte_offset(clean, a.start),
            byte_end=_byte_offset(clean, a.end),
            text=a.text,
        )
        for a in annotations
    )
    validate_annotations(clean, completed)
    return clean, completed


def validate_annotations(text: str, annotations: Iterable[Annotation]) -> None:
    previous_end = 0
    byte_length = len(text.encode("utf-8"))
    for index, annotation in enumerate(sorted(annotations, key=lambda a: (a.start, a.end))):
        if annotation.start < 0 or annotation.end <= annotation.start or annotation.end > len(text):
            raise AnnotationError(f"annotation {index} has invalid code-point offsets")
        if annotation.start < previous_end:
            raise AnnotationError(f"annotation {index} overlaps a previous annotation")
        if text[annotation.start : annotation.end] != annotation.text:
            raise AnnotationError(f"annotation {index} text does not match its code-point span")
        expected_start = _byte_offset(text, annotation.start)
        expected_end = _byte_offset(text, annotation.end)
        if (annotation.byte_start, annotation.byte_end) != (expected_start, expected_end):
            raise AnnotationError(f"annotation {index} has invalid UTF-8 byte offsets")
        if not 0 <= annotation.byte_start < annotation.byte_end <= byte_length:
            raise AnnotationError(f"annotation {index} has out-of-range UTF-8 byte offsets")
        if (
            text.encode("utf-8")[annotation.byte_start : annotation.byte_end].decode("utf-8")
            != annotation.text
        ):
            raise AnnotationError(f"annotation {index} does not round-trip through UTF-8 bytes")
        previous_end = annotation.end


def render_marked(text: str, annotations: Iterable[Annotation]) -> str:
    ordered = sorted(annotations, key=lambda a: (a.start, a.end))
    validate_annotations(text, ordered)
    parts: list[str] = []
    position = 0
    for annotation in ordered:
        parts.append(text[position : annotation.start])
        parts.append(f"[[{annotation.entity_type}:{annotation.text}]]")
        position = annotation.end
    parts.append(text[position:])
    return "".join(parts)

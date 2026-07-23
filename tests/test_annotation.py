from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from piicorpus.annotation import AnnotationError, parse_marked, render_marked, validate_annotations
from piicorpus.models import Annotation


def test_unicode_code_point_and_byte_offsets_round_trip() -> None:
    clean, annotations = parse_marked("Préface 🧪 [[BIRTH_DATE:SYN-DATE-2004-02-03]] résumé")
    annotation = annotations[0]
    assert clean[annotation.start : annotation.end] == annotation.text
    assert (
        clean.encode("utf-8")[annotation.byte_start : annotation.byte_end].decode()
        == annotation.text
    )
    assert annotation.byte_start > annotation.start
    assert parse_marked(render_marked(clean, annotations)) == (clean, annotations)


@given(
    prefix=st.text(
        alphabet=st.characters(blacklist_characters="[]", blacklist_categories=("Cs",)),
        max_size=20,
    ),
    value=st.text(
        alphabet=st.characters(blacklist_characters="[]:\r\n", blacklist_categories=("Cs",)),
        min_size=1,
        max_size=25,
    ).filter(lambda value: bool(value.strip())),
    suffix=st.text(
        alphabet=st.characters(blacklist_characters="[]", blacklist_categories=("Cs",)),
        max_size=20,
    ),
)
def test_property_unicode_round_trip(prefix: str, value: str, suffix: str) -> None:
    canonical = value.strip()
    marked = f"{prefix}[[SYNTHETIC_LABEL:{canonical}]]{suffix}"
    clean, annotations = parse_marked(marked)
    validate_annotations(clean, annotations)
    assert render_marked(clean, annotations) == marked


@pytest.mark.parametrize(
    "marked",
    [
        "[[LABEL:value]",
        "LABEL:value]]",
        "[[LABEL]]",
        "[[LABEL:]]",
        "[[bad:value]]",
        "[[LABEL:outer [[OTHER:inner]]]]",
    ],
)
def test_malformed_markers_are_rejected(marked: str) -> None:
    with pytest.raises(AnnotationError):
        parse_marked(marked)


def test_unencodable_surrogates_are_rejected_loudly() -> None:
    with pytest.raises(AnnotationError, match="UTF-8"):
        parse_marked("\ud800[[SYNTHETIC_LABEL:value]]")
    with pytest.raises(AnnotationError, match="UTF-8"):
        validate_annotations("bad \udfff tail", ())


def test_overlapping_annotations_are_rejected() -> None:
    text = "abcdef"
    annotations = (
        Annotation("FIRST_LABEL", 0, 4, 0, 4, "abcd"),
        Annotation("SECOND_LABEL", 3, 6, 3, 6, "def"),
    )
    with pytest.raises(AnnotationError, match="overlaps"):
        validate_annotations(text, annotations)

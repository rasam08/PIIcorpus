from __future__ import annotations

from piicorpus.similarity import jaccard, near_duplicate_pairs, shingle_set


def test_jaccard_on_known_texts() -> None:
    left = shingle_set("a b c d e f g h")
    right = shingle_set("a b c d e f g h")
    assert jaccard(left, right) == 1.0
    disjoint = shingle_set("q r s t u v w x")
    assert jaccard(left, disjoint) == 0.0


def test_planted_near_duplicates_are_found() -> None:
    base = "the quick brown fox jumps over the lazy dog near the old stone bridge today"
    texts = [
        base,
        base + " again",
        "an entirely different sentence about synthetic corpora and their audits",
        "another unrelated line mentioning printers, parcels, and paint swatches",
    ]
    pairs = near_duplicate_pairs(texts, threshold=0.8)
    assert [(left, right) for left, right, _score in pairs] == [(0, 1)]
    _left, _right, score = pairs[0]
    assert 0.8 <= score < 1.0


def test_pairs_are_deterministic_across_runs() -> None:
    base = (
        "this long shared template sentence keeps almost every single shingle "
        "identical across the generated records in the batch"
    )
    texts = [f"{base} tail{index}" for index in range(30)]
    first = near_duplicate_pairs(texts, threshold=0.8)
    second = near_duplicate_pairs(texts, threshold=0.8)
    assert first == second
    assert len(first) == 30 * 29 // 2, "records differing only in the tail must all pair"


def test_short_texts_do_not_crash() -> None:
    assert near_duplicate_pairs(["one", "one", ""], threshold=0.9)

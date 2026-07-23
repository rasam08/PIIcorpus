"""Scalable near-duplicate detection: MinHash signatures with LSH banding.

Candidate pairs come from banding, so runtime stays near-linear in corpus size;
exact Jaccard similarity is then computed on the hashed shingle sets, so reported
pairs are never false positives. Banding recall is probabilistic and documented
in docs/FAILURE_MODEL.md.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence

SHINGLE_WIDTH = 4
_PERMUTATIONS = 32
_BANDS = 8
_ROWS = _PERMUTATIONS // _BANDS
_PRIME = (1 << 61) - 1


def _hash64(text: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big"
    )


def _permutation_parameters() -> tuple[tuple[int, int], ...]:
    parameters = []
    for index in range(_PERMUTATIONS):
        seed = hashlib.blake2b(f"piicorpus-minhash-{index}".encode(), digest_size=16).digest()
        a = (int.from_bytes(seed[:8], "big") % (_PRIME - 1)) + 1
        b = int.from_bytes(seed[8:], "big") % _PRIME
        parameters.append((a, b))
    return tuple(parameters)


_PARAMETERS = _permutation_parameters()


def shingle_set(text: str, width: int = SHINGLE_WIDTH) -> frozenset[int]:
    words = text.split()
    if not words:
        return frozenset()
    if len(words) < width:
        return frozenset({_hash64(" ".join(words))})
    return frozenset(
        _hash64(" ".join(words[index : index + width]))
        for index in range(len(words) - width + 1)
    )


def jaccard(left: frozenset[int], right: frozenset[int]) -> float:
    if not left and not right:
        return 1.0
    union = len(left | right)
    return len(left & right) / union if union else 0.0


def _signature(shingles: frozenset[int]) -> tuple[int, ...]:
    if not shingles:
        return tuple(0 for _ in _PARAMETERS)
    return tuple(
        min((a * shingle + b) % _PRIME for shingle in shingles) for a, b in _PARAMETERS
    )


def near_duplicate_pairs(
    texts: Sequence[str],
    *,
    threshold: float,
) -> list[tuple[int, int, float]]:
    """Return ``(left_index, right_index, jaccard)`` for pairs at or above threshold."""
    shingles = [shingle_set(text) for text in texts]
    signatures = [_signature(entry) for entry in shingles]
    candidates: set[tuple[int, int]] = set()
    for band in range(_BANDS):
        buckets: dict[tuple[int, ...], list[int]] = defaultdict(list)
        start = band * _ROWS
        for index, signature in enumerate(signatures):
            buckets[signature[start : start + _ROWS]].append(index)
        for members in buckets.values():
            for left_position in range(len(members)):
                for right_position in range(left_position + 1, len(members)):
                    candidates.add((members[left_position], members[right_position]))
    pairs = []
    for left, right in sorted(candidates):
        similarity = jaccard(shingles[left], shingles[right])
        if similarity >= threshold:
            pairs.append((left, right, similarity))
    return pairs

"""Empirical learnability probe: can trivial features solve the corpus?

The structural audit checks approximate one question — would a trivial model
ace this corpus by exploiting shortcuts? The probe answers it directly: hashed
character 3-5-gram features feed one-vs-rest logistic regression trained by
plain SGD (stdlib only, fixed seed, deterministic), and accuracy on the held
splits is compared against configured ceilings. High probe accuracy does not
prove the corpus useless; it proves a trivial model finds enough surface signal
to pass, which is exactly what a shortcut is.
"""

from __future__ import annotations

import math
import random
import zlib
from collections import Counter, defaultdict
from collections.abc import Sequence

from .models import Finding, Record

_DIMENSIONS = 1 << 18
_EPOCHS = 5
_LEARNING_RATE = 0.5
_SEED = 20260723
_MINIMUM_TRAIN_EXAMPLES = 10

FeatureVector = dict[int, float]


def _features(text: str) -> FeatureVector:
    folded = text.casefold()
    counts: Counter[int] = Counter()
    for width in (3, 4, 5):
        for index in range(max(0, len(folded) - width + 1)):
            gram = folded[index : index + width]
            counts[zlib.crc32(gram.encode("utf-8")) & (_DIMENSIONS - 1)] += 1
    norm = math.sqrt(sum(count * count for count in counts.values()))
    if not norm:
        return {}
    return {key: count / norm for key, count in counts.items()}


def _train(
    examples: Sequence[tuple[FeatureVector, int]], class_count: int
) -> tuple[list[dict[int, float]], list[float]]:
    weights: list[dict[int, float]] = [defaultdict(float) for _ in range(class_count)]
    bias = [0.0] * class_count
    order = list(range(len(examples)))
    rng = random.Random(_SEED)
    for epoch in range(_EPOCHS):
        rng.shuffle(order)
        rate = _LEARNING_RATE / (1.0 + epoch)
        for position in order:
            features, target = examples[position]
            for cls in range(class_count):
                score = bias[cls] + sum(
                    weights[cls][key] * value for key, value in features.items()
                )
                clipped = max(-30.0, min(30.0, score))
                probability = 1.0 / (1.0 + math.exp(-clipped))
                gradient = (1.0 if target == cls else 0.0) - probability
                if gradient:
                    bias[cls] += rate * gradient
                    for key, value in features.items():
                        weights[cls][key] += rate * gradient * value
    return weights, bias


def _predict(
    weights: list[dict[int, float]], bias: list[float], features: FeatureVector
) -> int:
    best_class = 0
    best_score = -math.inf
    for cls in range(len(bias)):
        score = bias[cls] + sum(
            weights[cls].get(key, 0.0) * value for key, value in features.items()
        )
        if score > best_score:
            best_class, best_score = cls, score
    return best_class


def _accuracy(
    weights: list[dict[int, float]],
    bias: list[float],
    examples: Sequence[tuple[FeatureVector, int]],
) -> float:
    if not examples:
        return 0.0
    correct = sum(
        _predict(weights, bias, features) == target for features, target in examples
    )
    return correct / len(examples)


def _run_task(
    per_split: dict[str, list[tuple[FeatureVector, int]]],
    train_split: str,
    class_count: int,
) -> dict[str, float] | None:
    train_examples = per_split.get(train_split, [])
    if len(train_examples) < _MINIMUM_TRAIN_EXAMPLES or class_count < 2:
        return None
    weights, bias = _train(train_examples, class_count)
    return {
        split: round(_accuracy(weights, bias, examples), 4)
        for split, examples in per_split.items()
        if split != train_split and examples
    }


def _task_finding(
    risk: str,
    accuracies: dict[str, float] | None,
    ceiling: float,
    *,
    source: str,
    description: str,
    **details: object,
) -> Finding:
    if accuracies is None or not accuracies:
        return Finding(
            risk=risk,
            status="UNMEASURED",
            count=None,
            reason="not enough examples, classes, or held splits to run this probe task",
        )
    worst = max(accuracies.values())
    failed = worst > ceiling
    return Finding(
        risk=risk,
        status="FAIL" if failed else "PASS",
        count=len(accuracies),
        reason=(
            f"a trivial character-n-gram model can {description} above the ceiling; "
            "the corpus contains a surface shortcut"
            if failed
            else f"a trivial character-n-gram model cannot {description} above the ceiling"
        ),
        details={"accuracy_per_split": accuracies, **details},
        measured=worst,
        threshold=ceiling,
        threshold_source=source,
    )


def probe_findings(
    split_records: dict[str, list[Record]],
    *,
    train_split: str = "train",
    max_kind_accuracy: float,
    max_value_label_accuracy: float,
    max_context_label_accuracy: float,
    threshold_source: str = "config",
) -> list[Finding]:
    """Train the probe on ``train_split`` and score every other split."""
    kind_examples: dict[str, list[tuple[FeatureVector, int]]] = {}
    value_examples: dict[str, list[tuple[FeatureVector, int]]] = {}
    context_examples: dict[str, list[tuple[FeatureVector, int]]] = {}
    labels = sorted(
        {
            record.annotations[0].entity_type
            for rows in split_records.values()
            for record in rows
            if record.kind == "positive" and len(record.annotations) == 1
        }
    )
    label_index = {label: position for position, label in enumerate(labels)}
    for split, rows in split_records.items():
        kind_examples[split] = [
            (_features(record.text), 1 if record.kind == "positive" else 0)
            for record in rows
        ]
        value_split: list[tuple[FeatureVector, int]] = []
        context_split: list[tuple[FeatureVector, int]] = []
        for record in rows:
            if record.kind != "positive" or len(record.annotations) != 1:
                continue
            annotation = record.annotations[0]
            target = label_index[annotation.entity_type]
            value_split.append((_features(annotation.text), target))
            masked = (
                record.text[: annotation.start]
                + " <VALUE> "
                + record.text[annotation.end :]
            )
            context_split.append((_features(masked), target))
        value_examples[split] = value_split
        context_examples[split] = context_split

    kind_totals = Counter(
        target for examples in kind_examples.values() for _features_, target in examples
    )
    majority_baseline = (
        round(max(kind_totals.values()) / sum(kind_totals.values()), 4)
        if kind_totals
        else 0.0
    )
    return [
        _task_finding(
            "probe_kind_separability",
            _run_task(kind_examples, train_split, 2),
            max_kind_accuracy,
            source=threshold_source,
            description="separate positives from hard negatives",
            majority_baseline=majority_baseline,
        ),
        _task_finding(
            "probe_value_label_shortcut",
            _run_task(value_examples, train_split, len(labels)),
            max_value_label_accuracy,
            source=threshold_source,
            description="predict the label from the value string alone",
            labels=labels,
        ),
        _task_finding(
            "probe_context_label_shortcut",
            _run_task(context_examples, train_split, len(labels)),
            max_context_label_accuracy,
            source=threshold_source,
            description="predict the label from the masked context alone",
            labels=labels,
        ),
    ]


def unmeasured_probe_findings(reason: str) -> list[Finding]:
    return [
        Finding(risk=risk, status="UNMEASURED", count=None, reason=reason)
        for risk in (
            "probe_kind_separability",
            "probe_value_label_shortcut",
            "probe_context_label_shortcut",
        )
    ]

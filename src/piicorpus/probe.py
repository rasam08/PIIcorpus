"""Empirical learnability probe: can trivial features solve the corpus?

The structural audit checks approximate one question — would a trivial model
ace this corpus by exploiting shortcuts? The probe answers it directly: hashed
character 3-5-gram features feed one-vs-rest logistic regression trained by
plain SGD (stdlib only, fixed seed, deterministic). Balanced accuracy on held
splits is compared against configured ceilings and a split-specific
majority-predictor baseline. A failing probe is evidence that a trivial model
finds learnable surface signal beyond class priors; it is not proof of the
signal's cause or of real-world usefulness.
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
_BASELINE_MARGIN = 0.05

FeatureVector = dict[int, float]
ProbeMetrics = dict[str, float]


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


def _classification_metrics(
    weights: list[dict[int, float]],
    bias: list[float],
    examples: Sequence[tuple[FeatureVector, int]],
) -> ProbeMetrics:
    if not examples:
        return {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "balanced_majority_baseline": 0.0,
            "macro_f1": 0.0,
            "majority_baseline": 0.0,
        }
    targets = [target for _features_, target in examples]
    predictions = [_predict(weights, bias, features) for features, _target in examples]
    target_counts: Counter[int] = Counter(targets)
    correct_counts: Counter[int] = Counter(
        target for target, predicted in zip(targets, predictions, strict=True)
        if target == predicted
    )
    balanced_accuracy = sum(
        correct_counts[target] / count for target, count in target_counts.items()
    ) / len(target_counts)
    classes = sorted(set(targets) | set(predictions))
    f1_values: list[float] = []
    for cls in classes:
        true_positives = sum(
            target == predicted == cls
            for target, predicted in zip(targets, predictions, strict=True)
        )
        false_positives = sum(
            target != cls and predicted == cls
            for target, predicted in zip(targets, predictions, strict=True)
        )
        false_negatives = sum(
            target == cls and predicted != cls
            for target, predicted in zip(targets, predictions, strict=True)
        )
        denominator = 2 * true_positives + false_positives + false_negatives
        f1_values.append(2 * true_positives / denominator if denominator else 0.0)
    accuracy = sum(
        target == predicted
        for target, predicted in zip(targets, predictions, strict=True)
    ) / len(targets)
    return {
        "accuracy": round(accuracy, 4),
        "balanced_accuracy": round(balanced_accuracy, 4),
        "balanced_majority_baseline": round(1.0 / len(target_counts), 4),
        "macro_f1": round(sum(f1_values) / len(f1_values), 4),
        "majority_baseline": round(max(target_counts.values()) / len(targets), 4),
    }


def _run_task(
    per_split: dict[str, list[tuple[FeatureVector, int]]],
    train_split: str,
    class_count: int,
) -> dict[str, ProbeMetrics] | None:
    train_examples = per_split.get(train_split, [])
    if len(train_examples) < _MINIMUM_TRAIN_EXAMPLES or class_count < 2:
        return None
    weights, bias = _train(train_examples, class_count)
    return {
        split: _classification_metrics(weights, bias, examples)
        for split, examples in per_split.items()
        if split != train_split and examples
    }


def _task_finding(
    risk: str,
    metrics_by_split: dict[str, ProbeMetrics] | None,
    ceiling: float,
    *,
    source: str,
    description: str,
    **details: object,
) -> Finding:
    if metrics_by_split is None or not metrics_by_split:
        return Finding(
            risk=risk,
            status="UNMEASURED",
            count=None,
            reason="not enough examples, classes, or held splits to run this probe task",
        )
    failing_splits = sorted(
        split
        for split, metrics in metrics_by_split.items()
        if metrics["balanced_accuracy"] > ceiling
        and metrics["balanced_accuracy"]
        > metrics["balanced_majority_baseline"] + _BASELINE_MARGIN
    )
    worst = max(metrics["balanced_accuracy"] for metrics in metrics_by_split.values())
    failed = bool(failing_splits)
    return Finding(
        risk=risk,
        status="FAIL" if failed else "PASS",
        count=len(metrics_by_split),
        reason=(
            f"a trivial character-n-gram model can {description} with balanced accuracy "
            "above both the ceiling and the majority-predictor baseline margin; this is "
            "evidence of learnable surface signal beyond class priors"
            if failed
            else f"a trivial character-n-gram model cannot {description} with balanced "
            "accuracy above both the ceiling and the majority-predictor baseline margin"
        ),
        details={
            "accuracy_per_split": {
                split: metrics["accuracy"] for split, metrics in metrics_by_split.items()
            },
            "balanced_accuracy_per_split": {
                split: metrics["balanced_accuracy"]
                for split, metrics in metrics_by_split.items()
            },
            "balanced_majority_baseline_per_split": {
                split: metrics["balanced_majority_baseline"]
                for split, metrics in metrics_by_split.items()
            },
            "baseline_margin": _BASELINE_MARGIN,
            "failing_splits": failing_splits,
            "macro_f1_per_split": {
                split: metrics["macro_f1"] for split, metrics in metrics_by_split.items()
            },
            "majority_baseline_per_split": {
                split: metrics["majority_baseline"]
                for split, metrics in metrics_by_split.items()
            },
            "metric": "balanced_accuracy",
            **details,
        },
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

    return [
        _task_finding(
            "probe_kind_separability",
            _run_task(kind_examples, train_split, 2),
            max_kind_accuracy,
            source=threshold_source,
            description="separate positives from hard negatives",
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

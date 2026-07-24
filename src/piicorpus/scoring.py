"""Detector-neutral scoring: compare span predictions against a validated corpus.

The scorer never runs a model. It consumes span predictions produced by any
detector and reports precision/recall/F1 sliced by label, family, kind, and
split, plus a diagnostics section that turns the corpus's engineered families
into mechanism measurements: cue dependence, shape-hint substitution,
over-triggering on hard negatives, and noise robustness.

Scores on synthetic data demonstrate mechanism failures; they never demonstrate
real-world adequacy. See docs/CLAIM_BOUNDARIES.md.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .manifest import load_corpus
from .models import Record, stable_json
from .validators import CorpusIntegrityError, validate_corpus

SCORING_LIMITATION = (
    "Scores on synthetic data demonstrate mechanism failures (cue reliance, shape-hint "
    "substitution, over-triggering); they never demonstrate real-world adequacy."
)

Span = tuple[int, int, str]


class ScoringError(ValueError):
    """Raised when predictions cannot be scored against the corpus."""


@dataclass(frozen=True, slots=True)
class ScoreReport:
    match_mode: str
    records: int
    predicted_records: int
    records_without_predictions: int
    skipped_unknown_ids: int
    overall: dict[str, Any]
    macro_f1: float
    per_label: dict[str, dict[str, Any]]
    per_family: dict[str, dict[str, Any]]
    per_kind: dict[str, dict[str, Any]]
    per_split: dict[str, dict[str, Any]]
    diagnostics: dict[str, Any]
    limitation: str = SCORING_LIMITATION
    notes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "diagnostics": self.diagnostics,
            "limitation": self.limitation,
            "macro_f1": self.macro_f1,
            "match_mode": self.match_mode,
            "overall": self.overall,
            "per_family": self.per_family,
            "per_kind": self.per_kind,
            "per_label": self.per_label,
            "per_split": self.per_split,
            "predicted_records": self.predicted_records,
            "records": self.records,
            "records_without_predictions": self.records_without_predictions,
            "skipped_unknown_ids": self.skipped_unknown_ids,
        }

    def render(self, format_name: str = "text") -> str:
        if format_name == "json":
            return stable_json(self.to_dict(), pretty=True)
        if format_name == "markdown":
            lines = [
                "# PIIcorpus score",
                "",
                f"match mode `{self.match_mode}`; {self.records} records; "
                f"micro F1 {self.overall['f1']}; macro F1 {self.macro_f1}",
                "",
                "| Slice | Name | Precision | Recall | F1 | TP | FP | FN |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
            for slice_name, table in (
                ("label", self.per_label),
                ("family", self.per_family),
                ("kind", self.per_kind),
                ("split", self.per_split),
            ):
                for name, metrics in table.items():
                    lines.append(
                        f"| {slice_name} | {name} | {metrics['precision']} "
                        f"| {metrics['recall']} | {metrics['f1']} | {metrics['tp']} "
                        f"| {metrics['fp']} | {metrics['fn']} |"
                    )
            lines.extend(("", "## Diagnostics", ""))
            for key, value in self.diagnostics.items():
                lines.append(f"- `{key}`: {value}")
            lines.extend(("", f"> {self.limitation}", ""))
            return "\n".join(lines)
        lines = [
            "PIIcorpus score",
            f"match={self.match_mode} records={self.records} "
            f"predicted_records={self.predicted_records} "
            f"without_predictions={self.records_without_predictions}",
            "overall    precision={precision} recall={recall} f1={f1} "
            "(tp={tp} fp={fp} fn={fn})".format(**self.overall)
            + f" macro_f1={self.macro_f1}",
        ]
        for title, table in (
            ("label", self.per_label),
            ("family", self.per_family),
            ("kind", self.per_kind),
            ("split", self.per_split),
        ):
            lines.append(f"per {title}:")
            for name, metrics in table.items():
                lines.append(
                    f"  {name:32s} precision={metrics['precision']} "
                    f"recall={metrics['recall']} f1={metrics['f1']} "
                    f"tp={metrics['tp']} fp={metrics['fp']} fn={metrics['fn']}"
                )
        lines.append("diagnostics:")
        for key, value in self.diagnostics.items():
            note = self.notes.get(key)
            suffix = f"  ({note})" if note else ""
            lines.append(f"  {key:32s} {value}{suffix}")
        lines.append("")
        lines.append(self.limitation)
        return "\n".join(lines) + "\n"


def _prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall) if precision + recall else 0.0
    )
    return {
        "f1": round(f1, 4),
        "fn": fn,
        "fp": fp,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "tp": tp,
    }


def _iou(left: Span, right: Span) -> float:
    overlap = min(left[1], right[1]) - max(left[0], right[0])
    if overlap <= 0:
        return 0.0
    union = max(left[1], right[1]) - min(left[0], right[0])
    return overlap / union


def _span_targets_gold(gold: Span, predicted: Span, mode: str) -> bool:
    if mode == "strict":
        return gold[:2] == predicted[:2]
    return _iou(gold, predicted) >= 0.5


def _match_spans(
    gold: list[Span], predicted: list[Span], mode: str
) -> tuple[int, list[Span], list[Span]]:
    """Return (true positives, unmatched predictions, unmatched gold)."""
    if mode == "strict":
        remaining = list(gold)
        false_positives: list[Span] = []
        for span in predicted:
            if span in remaining:
                remaining.remove(span)
            else:
                false_positives.append(span)
        matched = len(gold) - len(remaining)
        return matched, false_positives, remaining
    candidates = sorted(
        (
            (_iou(gold_span, predicted_span), gold_index, predicted_index)
            for gold_index, gold_span in enumerate(gold)
            for predicted_index, predicted_span in enumerate(predicted)
            if gold_span[2] == predicted_span[2]
            and _iou(gold_span, predicted_span) >= 0.5
        ),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    matched_gold: set[int] = set()
    matched_predicted: set[int] = set()
    for _score, gold_index, predicted_index in candidates:
        if gold_index in matched_gold or predicted_index in matched_predicted:
            continue
        matched_gold.add(gold_index)
        matched_predicted.add(predicted_index)
    false_positives = [
        span for index, span in enumerate(predicted) if index not in matched_predicted
    ]
    false_negatives = [
        span for index, span in enumerate(gold) if index not in matched_gold
    ]
    return len(matched_gold), false_positives, false_negatives


def _prediction_offset(value: object, key: str, *, allow_invalid: bool) -> int:
    if allow_invalid:
        if not isinstance(value, (str, int, float)) or isinstance(value, bool):
            raise TypeError(f"prediction span {key} must be numeric")
        return int(value)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"prediction span {key} must be an integer")
    return value


def _validate_prediction_collection(spans: list[Span]) -> None:
    for start, end, _label in spans:
        if start < 0:
            raise TypeError("prediction span start cannot be negative")
        if start >= end:
            raise TypeError("prediction span start must be less than end")
    if len(set(spans)) != len(spans):
        raise TypeError("prediction spans cannot contain exact duplicates")
    ordered = sorted(spans, key=lambda span: (span[0], span[1], span[2]))
    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1 :]:
            if right[0] >= left[1]:
                break
            raise TypeError("prediction spans cannot overlap")


def load_predictions(
    path: str | Path, *, allow_invalid: bool = False
) -> dict[str, list[Span]]:
    """Load ``{"id": ..., "spans": [...]}`` JSONL predictions."""
    source = Path(path)
    try:
        raw = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ScoringError(f"cannot read predictions: {exc}") from exc
    predictions: dict[str, list[Span]] = {}
    for number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict) or not isinstance(value.get("id"), str):
                raise TypeError("expected an object with an id string")
            if value["id"] in predictions:
                raise TypeError(f"duplicate prediction id: {value['id']}")
            raw_spans = value.get("spans", [])
            if not isinstance(raw_spans, list):
                raise TypeError("prediction spans must be an array")
            spans: list[Span] = []
            for span in raw_spans:
                if not isinstance(span, dict):
                    raise TypeError("every prediction span must be an object")
                label_value = span.get("entity_type", span.get("label"))
                if not isinstance(label_value, str) or not label_value.strip():
                    raise TypeError("prediction span lacks an entity_type")
                spans.append(
                    (
                        _prediction_offset(
                            span["start"], "start", allow_invalid=allow_invalid
                        ),
                        _prediction_offset(span["end"], "end", allow_invalid=allow_invalid),
                        label_value.strip(),
                    )
                )
            if not allow_invalid:
                _validate_prediction_collection(spans)
            predictions[value["id"]] = spans
        except (json.JSONDecodeError, TypeError, KeyError, ValueError) as exc:
            raise ScoringError(
                f"invalid prediction at {source.name}:{number}: {exc}"
            ) from exc
    return predictions


def _codepoint_spans(record: Record, spans: list[Span]) -> list[Span]:
    mapping: dict[int, int] = {}
    byte = 0
    for index, char in enumerate(record.text):
        mapping[byte] = index
        byte += len(char.encode("utf-8"))
    mapping[byte] = len(record.text)
    converted = []
    for start, end, label in spans:
        if start not in mapping or end not in mapping:
            raise ScoringError(
                f"byte offsets [{start}, {end}) in record {record.case_id} do not "
                "fall on character boundaries"
            )
        converted.append((mapping[start], mapping[end], label))
    return converted


def score_corpus(
    directory: str | Path,
    predictions_path: str | Path,
    *,
    match: str = "strict",
    byte_offsets: bool = False,
    allow_partial: bool = False,
    allow_invalid: bool = False,
    allow_invalid_predictions: bool = False,
) -> ScoreReport:
    if match not in {"strict", "overlap"}:
        raise ScoringError(f"unsupported match mode: {match}")
    validation = validate_corpus(directory, strict=True)
    if not validation.valid and not allow_invalid:
        raise CorpusIntegrityError(validation)
    config, split_records, _manifest = load_corpus(directory)
    configured_labels = {label.name for label in config.labels}
    family_plugin = {family.name: family.plugin for family in config.families}
    family_role = {family.name: family.role for family in config.families}
    rows = [
        record for split in ("train", "eval", "holdout") for record in split_records[split]
    ]
    predictions = load_predictions(
        predictions_path, allow_invalid=allow_invalid_predictions
    )
    known_ids = {record.case_id for record in rows}
    unknown = sorted(set(predictions) - known_ids)
    if unknown and not allow_partial:
        raise ScoringError(
            f"{len(unknown)} prediction id(s) are not in the corpus "
            f"(first: {unknown[0]}); use --allow-partial to skip them"
        )

    counters: dict[str, dict[str, list[int]]] = {
        "label": defaultdict(lambda: [0, 0, 0]),
        "family": defaultdict(lambda: [0, 0, 0]),
        "kind": defaultdict(lambda: [0, 0, 0]),
        "split": defaultdict(lambda: [0, 0, 0]),
    }
    overall = [0, 0, 0]
    predicted_records = 0
    records_without = 0
    triggered_by_family: dict[str, int] = defaultdict(int)
    scored_by_family: dict[str, int] = defaultdict(int)
    conflict_outcomes: Counter[str] = Counter()

    for record in rows:
        if record.case_id in predictions:
            predicted_records += 1
            raw_spans = predictions[record.case_id]
            spans = _codepoint_spans(record, raw_spans) if byte_offsets else raw_spans
        else:
            if allow_partial:
                continue
            records_without += 1
            spans = []
        if not allow_invalid_predictions:
            for start, end, label in spans:
                if end > len(record.text):
                    raise ScoringError(
                        f"prediction span [{start}, {end}) exceeds record "
                        f"{record.case_id} length {len(record.text)}"
                    )
                if label not in configured_labels:
                    raise ScoringError(
                        f"prediction for record {record.case_id} uses unconfigured "
                        f"label {label!r}"
                    )
        gold = [(a.start, a.end, a.entity_type) for a in record.annotations]
        if (
            family_plugin.get(record.family) == "cue_shape_conflict"
            and len(gold) == 1
        ):
            conflict_outcomes["total"] += 1
            gold_span = gold[0]
            if any(
                predicted[2] == gold_span[2]
                and _span_targets_gold(gold_span, predicted, match)
                for predicted in spans
            ):
                conflict_outcomes["gold"] += 1
            else:
                shape_hint = record.metadata.get("shape_hint_label")
                if isinstance(shape_hint, str) and any(
                    predicted[2] == shape_hint
                    and _span_targets_gold(gold_span, predicted, match)
                    for predicted in spans
                ):
                    conflict_outcomes["shape_hint"] += 1
                elif spans:
                    conflict_outcomes["other_error"] += 1
                else:
                    conflict_outcomes["abstention"] += 1
        tp, false_positives, false_negatives = _match_spans(gold, spans, match)
        scored_by_family[record.family] += 1
        if spans:
            triggered_by_family[record.family] += 1
        slices = (
            ("family", record.family),
            ("kind", record.kind),
            ("split", record.split),
        )
        overall[0] += tp
        overall[1] += len(false_positives)
        overall[2] += len(false_negatives)
        for dimension, key in slices:
            entry = counters[dimension][key]
            entry[0] += tp
            entry[1] += len(false_positives)
            entry[2] += len(false_negatives)
        matched_labels = [span[2] for span in gold]
        for label in set(
            matched_labels + [span[2] for span in spans]
        ):
            label_gold = [span for span in gold if span[2] == label]
            label_predicted = [span for span in spans if span[2] == label]
            label_tp, label_fp, label_fn = _match_spans(label_gold, label_predicted, match)
            entry = counters["label"][label]
            entry[0] += label_tp
            entry[1] += len(label_fp)
            entry[2] += len(label_fn)

    per_label = {
        name: _prf(*values) for name, values in sorted(counters["label"].items())
    }
    label_f1_values = [metrics["f1"] for metrics in per_label.values()]
    macro_f1 = round(sum(label_f1_values) / len(label_f1_values), 4) if label_f1_values else 0.0

    per_family = {
        name: _prf(*values) for name, values in sorted(counters["family"].items())
    }

    def _group_recall(plugins: set[str]) -> float | None:
        tp = fn = 0
        for name, values in counters["family"].items():
            if family_plugin.get(name) in plugins:
                tp += values[0]
                fn += values[2]
        if tp + fn == 0:
            return None
        return round(tp / (tp + fn), 4)

    diagnostics: dict[str, Any] = {}
    cued_recall = _group_recall(
        {
            plugin
            for name, plugin in family_plugin.items()
            if family_role.get(name) == "positive" and plugin != "cue_free"
        }
    )
    cue_free_recall = _group_recall({"cue_free"})
    if cued_recall is not None and cue_free_recall is not None:
        diagnostics["cue_dependence"] = round(cued_recall - cue_free_recall, 4)
        diagnostics["cued_recall"] = cued_recall
        diagnostics["cue_free_recall"] = cue_free_recall
    conflict_total = conflict_outcomes["total"]
    if conflict_total:
        diagnostics["conflict_gold_recall"] = round(
            conflict_outcomes["gold"] / conflict_total, 4
        )
        diagnostics["shape_hint_substitution_rate"] = round(
            conflict_outcomes["shape_hint"] / conflict_total, 4
        )
        diagnostics["other_error_rate"] = round(
            conflict_outcomes["other_error"] / conflict_total, 4
        )
        diagnostics["abstention_rate"] = round(
            conflict_outcomes["abstention"] / conflict_total, 4
        )
    narrative_recall = _group_recall({"narrative"})
    for plugin_name, key in (("ocr_noise", "ocr_recall"), ("spoken", "spoken_recall")):
        recall = _group_recall({plugin_name})
        if recall is not None:
            diagnostics[key] = recall
            if narrative_recall is not None:
                diagnostics[f"{key.split('_')[0]}_robustness_gap"] = round(
                    narrative_recall - recall, 4
                )
    over_trigger = {
        name: round(triggered_by_family[name] / scored_by_family[name], 4)
        for name in sorted(scored_by_family)
        if family_role.get(name) == "hard_negative" and scored_by_family[name]
    }
    if over_trigger:
        diagnostics["over_trigger_per_hard_negative_family"] = over_trigger

    notes = {
        "cue_dependence": "cued recall minus cue-free recall; high means the detector "
        "needs a cue phrase",
        "conflict_gold_recall": "share of cue/shape conflicts assigned the gold label",
        "shape_hint_substitution_rate": "share of cue/shape conflicts assigned the "
        "stored shape-hint label instead of the gold label",
        "other_error_rate": "share of cue/shape conflicts with predictions that are "
        "neither the gold label nor a shape-hint substitution",
        "abstention_rate": "share of cue/shape conflicts with no predicted spans",
        "over_trigger_per_hard_negative_family": "share of hard negatives with any "
        "predicted span",
    }
    return ScoreReport(
        match_mode=match,
        records=len(rows),
        predicted_records=predicted_records,
        records_without_predictions=records_without,
        skipped_unknown_ids=len(unknown),
        overall=_prf(*overall),
        macro_f1=macro_f1,
        per_label=per_label,
        per_family=per_family,
        per_kind={
            name: _prf(*values) for name, values in sorted(counters["kind"].items())
        },
        per_split={
            name: _prf(*values) for name, values in sorted(counters["split"].items())
        },
        diagnostics=diagnostics,
        notes=notes,
    )

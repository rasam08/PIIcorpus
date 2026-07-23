"""Detector-neutral scoring: compare span predictions against a validated corpus.

The scorer never runs a model. It consumes span predictions produced by any
detector and reports precision/recall/F1 sliced by label, family, kind, and
split, plus a diagnostics section that turns the corpus's engineered families
into mechanism measurements: cue dependence, morphology dependence,
over-triggering on hard negatives, and noise robustness.

Scores on synthetic data demonstrate mechanism failures; they never demonstrate
real-world adequacy. See docs/CLAIM_BOUNDARIES.md.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .manifest import load_corpus
from .models import Record, stable_json
from .validators import CorpusIntegrityError, validate_corpus

SCORING_LIMITATION = (
    "Scores on synthetic data demonstrate mechanism failures (cue reliance, shape "
    "reliance, over-triggering); they never demonstrate real-world adequacy."
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


def load_predictions(path: str | Path) -> dict[str, list[Span]]:
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
            spans: list[Span] = []
            for span in value.get("spans", []):
                label = str(span.get("entity_type", span.get("label", "")))
                if not label:
                    raise TypeError("prediction span lacks an entity_type")
                spans.append((int(span["start"]), int(span["end"]), label))
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
) -> ScoreReport:
    if match not in {"strict", "overlap"}:
        raise ScoringError(f"unsupported match mode: {match}")
    validation = validate_corpus(directory, strict=True)
    if not validation.valid and not allow_invalid:
        raise CorpusIntegrityError(validation)
    config, split_records, _manifest = load_corpus(directory)
    rows = [
        record for split in ("train", "eval", "holdout") for record in split_records[split]
    ]
    predictions = load_predictions(predictions_path)
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
        gold = [(a.start, a.end, a.entity_type) for a in record.annotations]
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

    family_plugin = {family.name: family.plugin for family in config.families}
    family_role = {family.name: family.role for family in config.families}
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
    conflict_recall = _group_recall({"cue_shape_conflict"})
    if conflict_recall is not None:
        diagnostics["morphology_dependence"] = round(1 - conflict_recall, 4)
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
        "morphology_dependence": "1 - recall on cue/shape conflicts; high means shape "
        "beats the labeled heading",
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

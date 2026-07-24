from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from piicorpus.cli import EXIT_FINDINGS, EXIT_OK, main
from piicorpus.manifest import load_corpus
from piicorpus.scoring import ScoringError, score_corpus

VALUE_RE = re.compile(r"SYN-ID-[A-Z0-9l-]+|SYN-DATE-\d{4}-\d{2}-\d{2}")


def _write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8", newline="\n"
    )


def _perfect_predictions(corpus: Path, path: Path) -> None:
    _config, split_records, _manifest = load_corpus(corpus)
    rows = []
    for records in split_records.values():
        for record in records:
            rows.append(
                {
                    "id": record.case_id,
                    "spans": [
                        {
                            "start": a.start,
                            "end": a.end,
                            "entity_type": a.entity_type,
                        }
                        for a in record.annotations
                    ],
                }
            )
    _write_predictions(path, rows)


def test_perfect_predictions_score_one(generated_demo: Path, tmp_path: Path) -> None:
    predictions = tmp_path / "perfect.jsonl"
    _perfect_predictions(generated_demo, predictions)
    report = score_corpus(generated_demo, predictions)
    assert report.overall["f1"] == 1.0
    assert report.macro_f1 == 1.0
    assert report.records_without_predictions == 0
    assert report.diagnostics["cue_dependence"] == 0.0
    assert report.diagnostics["conflict_gold_recall"] == 1.0
    assert report.diagnostics["shape_hint_substitution_rate"] == 0.0
    assert report.diagnostics["other_error_rate"] == 0.0
    assert report.diagnostics["abstention_rate"] == 0.0
    over_trigger = report.diagnostics["over_trigger_per_hard_negative_family"]
    assert all(value == 0.0 for value in over_trigger.values())


def test_regex_detector_diagnostics_expose_over_triggering(
    generated_demo: Path, tmp_path: Path
) -> None:
    _config, split_records, _manifest = load_corpus(generated_demo)
    rows = []
    for records in split_records.values():
        for record in records:
            spans = [
                {
                    "start": match.start(),
                    "end": match.end(),
                    "entity_type": "PATIENT_RECORD_ID",
                }
                for match in VALUE_RE.finditer(record.text)
            ]
            rows.append({"id": record.case_id, "spans": spans})
    predictions = tmp_path / "regex.jsonl"
    _write_predictions(predictions, rows)
    report = score_corpus(generated_demo, predictions)
    assert report.overall["precision"] < 0.5
    over_trigger = report.diagnostics["over_trigger_per_hard_negative_family"]
    # Near-miss negatives carry identifier-shaped values, so a shape-only regex
    # must fire on them; that is exactly what the diagnostic exists to expose.
    assert over_trigger["hard_negative_near_misses"] > 0.9
    # The spoken family spells values out, so a shape regex misses them.
    assert report.diagnostics["spoken_recall"] == 0.0
    assert report.diagnostics["cue_dependence"] is not None


def test_conflict_diagnostics_distinguish_shape_substitution_from_abstention(
    generated_demo: Path, tmp_path: Path
) -> None:
    config, split_records, _manifest = load_corpus(generated_demo)
    family_plugin = {family.name: family.plugin for family in config.families}
    conflicts = [
        record
        for records in split_records.values()
        for record in records
        if family_plugin[record.family] == "cue_shape_conflict"
    ]
    shape_predictions = tmp_path / "shape-hints.jsonl"
    _write_predictions(
        shape_predictions,
        [
            {
                "id": record.case_id,
                "spans": [
                    {
                        "start": record.annotations[0].start,
                        "end": record.annotations[0].end,
                        "entity_type": record.metadata["shape_hint_label"],
                    }
                ],
            }
            for record in conflicts
        ],
    )
    shape_report = score_corpus(generated_demo, shape_predictions, allow_partial=True)
    assert shape_report.diagnostics["conflict_gold_recall"] == 0.0
    assert shape_report.diagnostics["shape_hint_substitution_rate"] == 1.0
    assert shape_report.diagnostics["other_error_rate"] == 0.0
    assert shape_report.diagnostics["abstention_rate"] == 0.0

    empty_predictions = tmp_path / "abstentions.jsonl"
    _write_predictions(
        empty_predictions,
        [{"id": record.case_id, "spans": []} for record in conflicts],
    )
    empty_report = score_corpus(generated_demo, empty_predictions, allow_partial=True)
    assert empty_report.diagnostics["conflict_gold_recall"] == 0.0
    assert empty_report.diagnostics["shape_hint_substitution_rate"] == 0.0
    assert empty_report.diagnostics["other_error_rate"] == 0.0
    assert empty_report.diagnostics["abstention_rate"] == 1.0


@pytest.mark.parametrize(
    "spans",
    [
        [{"start": -1, "end": 1}],
        [{"start": 2, "end": 2}],
        [{"start": 0, "end": 100_000}],
        [{"start": 0, "end": 2}, {"start": 0, "end": 2}],
        [{"start": 0, "end": 2}, {"start": 1, "end": 3}],
    ],
)
def test_invalid_prediction_spans_are_rejected(
    generated_demo: Path,
    tmp_path: Path,
    spans: list[dict[str, int]],
) -> None:
    config, split_records, _manifest = load_corpus(generated_demo)
    record = split_records["train"][0]
    rows = [
        {
            "id": record.case_id,
            "spans": [
                {**span, "entity_type": config.labels[0].name} for span in spans
            ],
        }
    ]
    predictions = tmp_path / "invalid.jsonl"
    _write_predictions(predictions, rows)
    with pytest.raises(ScoringError):
        score_corpus(generated_demo, predictions, allow_partial=True)


def test_forensic_prediction_mode_scores_malformed_spans(
    generated_demo: Path, tmp_path: Path
) -> None:
    config, split_records, _manifest = load_corpus(generated_demo)
    record = split_records["train"][0]
    label = config.labels[0].name
    spans = [
        {"start": -1, "end": 0, "entity_type": label},
        {"start": 2, "end": 2, "entity_type": label},
        {"start": 0, "end": 100_000, "entity_type": label},
        {"start": 0, "end": 2, "entity_type": label},
        {"start": 1, "end": 3, "entity_type": label},
        {"start": 0, "end": 2, "entity_type": label},
    ]
    predictions = tmp_path / "forensic.jsonl"
    _write_predictions(predictions, [{"id": record.case_id, "spans": spans}])
    report = score_corpus(
        generated_demo,
        predictions,
        allow_partial=True,
        allow_invalid_predictions=True,
    )
    assert report.overall["fp"] == len(spans)


def test_unknown_ids_error_unless_partial(generated_demo: Path, tmp_path: Path) -> None:
    predictions = tmp_path / "unknown.jsonl"
    _write_predictions(predictions, [{"id": "pc-notarealid", "spans": []}])
    with pytest.raises(ScoringError, match="not in the corpus"):
        score_corpus(generated_demo, predictions)
    report = score_corpus(generated_demo, predictions, allow_partial=True)
    assert report.skipped_unknown_ids == 1
    assert report.predicted_records == 0


def test_overlap_match_mode_gives_partial_credit(
    generated_demo: Path, tmp_path: Path
) -> None:
    _config, split_records, _manifest = load_corpus(generated_demo)
    record = next(r for r in split_records["train"] if len(r.annotations) == 1)
    annotation = record.annotations[0]
    rows = [
        {
            "id": record.case_id,
            "spans": [
                {
                    "start": annotation.start + 1,
                    "end": annotation.end,
                    "entity_type": annotation.entity_type,
                }
            ],
        }
    ]
    predictions = tmp_path / "shifted.jsonl"
    _write_predictions(predictions, rows)
    strict = score_corpus(generated_demo, predictions, allow_partial=True)
    overlap = score_corpus(generated_demo, predictions, match="overlap", allow_partial=True)
    assert strict.overall["tp"] == 0
    assert overlap.overall["tp"] == 1


def test_cli_score_and_fail_under(generated_demo: Path, tmp_path: Path) -> None:
    predictions = tmp_path / "perfect.jsonl"
    _perfect_predictions(generated_demo, predictions)
    assert (
        main(["score", str(generated_demo), str(predictions), "--format", "json"])
        == EXIT_OK
    )
    empty = tmp_path / "empty.jsonl"
    _write_predictions(empty, [])
    assert (
        main(
            [
                "score",
                str(generated_demo),
                str(empty),
                "--fail-under",
                "0.5",
            ]
        )
        == EXIT_FINDINGS
    )

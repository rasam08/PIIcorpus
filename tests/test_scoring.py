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
    assert report.diagnostics["morphology_dependence"] == 0.0
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
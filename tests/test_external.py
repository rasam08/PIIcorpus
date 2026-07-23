from __future__ import annotations

import json
from pathlib import Path

import pytest

from piicorpus.cli import EXIT_OK, EXIT_OPERATIONAL, main
from piicorpus.exporters import export_corpus
from piicorpus.failure_model import audit_external_records
from piicorpus.importers import ExternalImportError, load_external


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8", newline="\n"
    )


def test_jsonl_external_records_load_with_split_field(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    _write_jsonl(
        path,
        [
            {
                "text": "id CODE-11 belongs to the sample",
                "spans": [{"start": 3, "end": 10, "entity_type": "TEST_ID"}],
                "split": "train",
            },
            {"text": "nothing sensitive here", "split": "test"},
        ],
    )
    records = load_external({"data": path}, "jsonl")
    assert sorted(records) == ["test", "train"]
    positive = records["train"][0]
    assert positive.kind == "positive"
    assert positive.annotations[0].text == "CODE-11"
    assert records["test"][0].kind == "unannotated"


def test_jsonl_byte_offsets_are_converted(tmp_path: Path) -> None:
    text = "café CODE-22 done"
    path = tmp_path / "bytes.jsonl"
    start = len("café ".encode())
    end = start + len(b"CODE-22")
    _write_jsonl(
        path,
        [{"text": text, "spans": [{"start": start, "end": end, "label": "TEST_ID"}]}],
    )
    records = load_external({"data": path}, "jsonl", byte_offsets=True)
    annotation = records["data"][0].annotations[0]
    assert annotation.text == "CODE-22"
    assert text[annotation.start : annotation.end] == "CODE-22"


def test_conll_blocks_become_records(tmp_path: Path) -> None:
    path = tmp_path / "data.conll"
    path.write_text(
        "-DOCSTART- O\n\nThe O\ncode O\nCODE-33 B-TEST_ID\n\nplain O\ntext O\n",
        encoding="utf-8",
    )
    records = load_external({"train": path}, "conll")
    rows = records["train"]
    assert len(rows) == 2
    assert rows[0].annotations[0].text == "CODE-33"
    assert rows[1].kind == "unannotated"


def test_malformed_external_spans_are_operational_errors(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    _write_jsonl(
        path,
        [{"text": "short", "spans": [{"start": 2, "end": 99, "entity_type": "X"}]}],
    )
    with pytest.raises(ExternalImportError, match=r"bad\.jsonl:1"):
        load_external({"data": path}, "jsonl")


def test_cli_rejects_duplicate_external_input_names(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first = tmp_path / "first" / "data.jsonl"
    second = tmp_path / "second" / "data.jsonl"
    first.parent.mkdir()
    second.parent.mkdir()
    _write_jsonl(first, [{"text": "first"}])
    _write_jsonl(second, [{"text": "second"}])
    assert (
        main(
            [
                "audit-external",
                str(first),
                str(second),
                "--format",
                "jsonl",
                "--no-probe",
            ]
        )
        == EXIT_OPERATIONAL
    )
    assert "duplicate external input name 'data'" in capsys.readouterr().err


def test_exported_demo_round_trips_through_external_audit(
    generated_demo: Path, tmp_path: Path
) -> None:
    result = export_corpus(generated_demo, "huggingface", tmp_path / "hf")
    export_dir = Path(result["path"]).parent
    records = load_external(
        {split: export_dir / f"{split}.jsonl" for split in ("train", "eval", "holdout")},
        "hf",
    )
    assert sorted(records) == ["eval", "holdout", "train"]
    report = audit_external_records(records, probe=False)
    statuses = {finding.risk: finding.status for finding in report.findings}
    assert statuses["corpus_integrity"] == "UNMEASURED"
    assert statuses["cue_free_coverage"] == "UNMEASURED"
    assert statuses["cross_split_value_contamination"] == "PASS"
    assert statuses["duplicate_or_near_duplicate_bodies"] == "PASS"
    assert statuses["external_safety_scan"] == "PASS"
    assert not report.failed


def test_external_safety_scan_flags_sensitive_patterns(tmp_path: Path) -> None:
    path = tmp_path / "leaky.jsonl"
    _write_jsonl(
        path,
        [
            {"text": "contact person@realdomain.test for details"},
            {"text": "all quiet here"},
        ],
    )
    records = load_external({"data": path}, "jsonl")
    warned = audit_external_records(records, probe=False)
    scan = next(f for f in warned.findings if f.risk == "external_safety_scan")
    assert scan.status == "WARN"
    assert not warned.failed
    failed = audit_external_records(records, probe=False, fail_on_safety=True)
    assert failed.failed


def test_cli_audit_external_smoke(tmp_path: Path, generated_demo: Path) -> None:
    result = export_corpus(generated_demo, "huggingface", tmp_path / "hf")
    export_dir = Path(result["path"]).parent
    assert (
        main(
            [
                "audit-external",
                "--format",
                "hf",
                "--no-probe",
                "--split",
                f"train={export_dir / 'train.jsonl'}",
                "--split",
                f"eval={export_dir / 'eval.jsonl'}",
                "--split",
                f"holdout={export_dir / 'holdout.jsonl'}",
            ]
        )
        == EXIT_OK
    )

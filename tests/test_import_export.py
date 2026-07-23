from __future__ import annotations

import json
from pathlib import Path

import pytest

from piicorpus.annotation import render_marked
from piicorpus.exporters import EXPORT_FORMATS, export_corpus
from piicorpus.importers import ImportErrorSafe, import_annotated
from piicorpus.manifest import load_corpus, load_records


@pytest.mark.parametrize("format_name", EXPORT_FORMATS)
def test_every_exporter_preserves_records_and_spans(
    format_name: str, generated_demo: Path, tmp_path: Path
) -> None:
    _config, split_records, _manifest = load_corpus(generated_demo)
    source_count = sum(len(records) for records in split_records.values())
    result = export_corpus(generated_demo, format_name, tmp_path / format_name)
    assert result["records"] == source_count
    output = Path(result["path"])
    assert output.is_file() and output.stat().st_size > 0

    if format_name in {"huggingface", "spacy", "presidio", "jsonl"}:
        lines = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        assert len(lines) == source_count
        source = split_records["train"][0]
        first = lines[0]
        if format_name == "huggingface":
            assert first["spans"][0]["text"] == source.annotations[0].text
        elif format_name == "spacy":
            start, end, label = first["entities"][0]
            assert first["text"][start:end] == source.annotations[0].text
            assert label == source.annotations[0].entity_type
        elif format_name == "presidio":
            expected = first["expected"][0]
            assert first["text"][expected["start"] : expected["end"]] == expected["text"]
        else:
            assert first["annotations"][0]["text"] == source.annotations[0].text
    else:
        payload = output.read_text(encoding="utf-8")
        assert "B-PATIENT_RECORD_ID" in payload


def test_human_import_round_trip_and_provenance(tmp_path: Path) -> None:
    source = tmp_path / "authored.txt"
    source.write_text(
        "Résumé 🧪 [[PUBLIC_TEST_ID:SYN-ID-A12345]]\nNo marked value is present.\n",
        encoding="utf-8",
    )
    output = tmp_path / "imported"
    manifest = import_annotated(source, output)
    records = load_records(output / "records.jsonl")
    assert manifest["provenance"] == "human_supplied"
    assert manifest["release_status"] == "unreviewed"
    assert all(record.provenance == "human_supplied" for record in records)
    assert all(record.split == "unassigned" for record in records)
    assert render_marked(records[0].text, records[0].annotations).endswith(
        "[[PUBLIC_TEST_ID:SYN-ID-A12345]]"
    )


def test_import_errors_do_not_echo_bodies_without_local_debug(tmp_path: Path) -> None:
    source = tmp_path / "bad.txt"
    sensitive_body = "do-not-echo-this-body [[BROKEN]]"
    source.write_text(sensitive_body, encoding="utf-8")
    with pytest.raises(ImportErrorSafe) as captured:
        import_annotated(source, tmp_path / "out")
    assert sensitive_body not in str(captured.value)


def test_importer_never_mixes_into_generated_splits(tmp_path: Path) -> None:
    source = tmp_path / "authored.txt"
    source.write_text("[[PUBLIC_TEST_ID:SYN-ID-B23456]]", encoding="utf-8")
    output = tmp_path / "imported"
    manifest = import_annotated(source, output)
    assert "not assigned" in manifest["mixing_policy"]
    assert not (output / "splits").exists()

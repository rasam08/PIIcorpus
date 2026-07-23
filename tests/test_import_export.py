from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from piicorpus.annotation import parse_marked, render_marked
from piicorpus.exporters import EXPORT_FORMATS, export_corpus
from piicorpus.exporters.core import _tokenize
from piicorpus.importers import ImportErrorSafe, import_annotated
from piicorpus.manifest import load_corpus, load_records
from piicorpus.models import Record


def _record_from_marked(marked: str) -> Record:
    clean, annotations = parse_marked(marked)
    return Record(
        case_id="test-record",
        split="train",
        family="test",
        namespace="piicorpus/train/test/00000",
        template_id="test",
        kind="positive" if annotations else "hard_negative",
        provenance="generated",
        text=clean,
        annotations=annotations,
    )


@given(
    value=st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
        min_size=1,
        max_size=12,
    )
)
def test_tokenizer_preserves_unicode_entity_values(value: str) -> None:
    record = _record_from_marked(f"prefix [[TEST_LABEL:{value}]] suffix")
    tokens = _tokenize(record)
    entity_text = "".join(
        token for token, _start, _end, tag in tokens if tag.endswith("TEST_LABEL")
    )
    assert re.sub(r"\s+", "", entity_text) == value
    assert [tag for _t, _s, _e, tag in tokens if tag.startswith("B-")] == ["B-TEST_LABEL"]


def test_tokenizer_keeps_adjacent_same_label_entities_apart() -> None:
    record = _record_from_marked("[[A_LABEL:one1]] [[A_LABEL:two2]]")
    tags = [tag for _token, _start, _end, tag in _tokenize(record)]
    assert tags == ["B-A_LABEL", "B-A_LABEL"]


def test_tokenizer_emits_inside_tags_for_multi_token_entities() -> None:
    record = _record_from_marked("code [[A_LABEL:ab-12]] end")
    tags = [tag for _token, _start, _end, tag in _tokenize(record)]
    assert tags == ["O", "B-A_LABEL", "I-A_LABEL", "I-A_LABEL", "O"]


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
    labels_payload = json.loads(
        (output.parent / "labels.json").read_text(encoding="utf-8")
    )
    assert labels_payload["labels"]
    assert set(labels_payload["bio_tags"]) > {"O"}

    if format_name == "huggingface":
        lines = [
            json.loads(line)
            for split in ("train", "eval", "holdout")
            for line in (output.parent / f"{split}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert len(lines) == source_count
        source = split_records["train"][0]
        assert lines[0]["spans"][0]["text"] == source.annotations[0].text
    elif format_name in {"spacy", "presidio", "jsonl"}:
        lines = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        assert len(lines) == source_count
        source = split_records["train"][0]
        first = lines[0]
        if format_name == "spacy":
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

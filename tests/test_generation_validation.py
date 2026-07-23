from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from piicorpus.config import load_config
from piicorpus.generator import GENERATOR_VERSION, generate, generate_records
from piicorpus.manifest import load_corpus, sha256_file, write_corpus
from piicorpus.semantics import is_contrastive_evidence
from piicorpus.validators import validate_corpus


def _files(directory: Path) -> dict[str, bytes]:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


def test_same_seed_and_config_are_byte_identical(tmp_path: Path, demo_config_path: Path) -> None:
    config = load_config(demo_config_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate(config, first)
    generate(config, second)
    assert _files(first) == _files(second)


def test_different_seed_changes_records_but_preserves_invariants(
    tmp_path: Path, demo_config_path: Path
) -> None:
    config = load_config(demo_config_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate(config, first)
    generate(replace(config, seed=config.seed + 1), second)
    assert (first / "splits/train.jsonl").read_bytes() != (
        second / "splits/train.jsonl"
    ).read_bytes()
    assert validate_corpus(first, strict=True).valid
    assert validate_corpus(second, strict=True).valid


def test_manifest_hashes_match_every_emitted_file(generated_demo: Path) -> None:
    manifest = json.loads((generated_demo / "manifest.json").read_text(encoding="utf-8"))
    for relative, declared in manifest["files"].items():
        path = generated_demo / relative
        assert sha256_file(path) == declared["sha256"]
        assert path.stat().st_size == declared["bytes"]


def test_tampered_output_is_rejected(generated_demo: Path) -> None:
    path = generated_demo / "splits/train.jsonl"
    payload = path.read_text(encoding="utf-8")
    path.write_text(payload.replace("fictional", "fabricated", 1), encoding="utf-8", newline="\n")
    report = validate_corpus(generated_demo, strict=True)
    assert not report.valid
    assert any(error.startswith("file_hash:") for error in report.errors)


def test_hard_negative_ratio_is_enforced(tmp_path: Path, demo_config_path: Path) -> None:
    config = load_config(demo_config_path)
    records = generate_records(config)
    stricter = replace(config, minimum_hard_negative_ratio=0.45)
    output = tmp_path / "too-few-negatives"
    write_corpus(output, stricter, records, generator_version=GENERATOR_VERSION)
    report = validate_corpus(output, strict=True)
    assert any(error.startswith("hard_negative_ratio:") for error in report.errors)


def test_manifest_contains_required_claim_boundary(generated_demo: Path) -> None:
    _config, _records, manifest = load_corpus(generated_demo)
    assert manifest["generated_data_license"] == "CC0-1.0"
    assert "not an independent generalization test" in manifest["synthetic_holdout_limitation"]


def test_record_metadata_and_semantic_evidence_are_rendered(generated_demo: Path) -> None:
    config, split_records, _manifest = load_corpus(generated_demo)
    family_by_name = {family.name: family for family in config.families}
    rows = [record for records in split_records.values() for record in records]
    assert all(not record.persona or record.persona in record.text for record in rows)
    assert all(not record.organization or record.organization in record.text for record in rows)
    assert all("Synthetic sample index" not in record.text for record in rows)
    assert all("document reference SYN-DOC-" in record.text for record in rows)
    assert all(
        record.cue_links or family_by_name[record.family].plugin == "cue_free"
        for record in rows
        if record.kind == "positive"
    )
    contrastive = [
        record
        for record in rows
        if family_by_name[record.family].plugin == "cue_shape_conflict"
    ]
    assert contrastive
    assert all(
        is_contrastive_evidence(record, config, family_by_name[record.family])
        for record in contrastive
    )


def test_stale_content_derived_case_id_is_rejected(
    generated_demo: Path, tmp_path: Path
) -> None:
    config, split_records, manifest = load_corpus(generated_demo)
    records = {split: list(rows) for split, rows in split_records.items()}
    records["train"][0] = replace(
        records["train"][0], text=records["train"][0].text + " Post-ID mutation."
    )
    output = tmp_path / "stale-case-id"
    write_corpus(
        output,
        config,
        records,
        generator_version=str(manifest["generator_version"]),
    )
    report = validate_corpus(output, strict=True)
    assert not report.valid
    assert any(error.startswith("case_id:") for error in report.errors)

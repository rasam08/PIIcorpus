from __future__ import annotations

import json
from pathlib import Path

import pytest

from piicorpus.exporters import export_corpus
from piicorpus.failure_model import audit_corpus
from piicorpus.models import stable_json
from piicorpus.validators import CorpusIntegrityError


def _tamper(corpus: Path, target: str) -> None:
    if target == "record":
        path = corpus / "splits" / "train.jsonl"
        payload = path.read_text(encoding="utf-8")
        path.write_text(
            payload.replace("Synthetic context", "Fabricated context", 1),
            encoding="utf-8",
            newline="\n",
        )
        return
    manifest_path = corpus / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["configuration_digest"] = "0" * 64
    manifest_path.write_text(
        stable_json(manifest, pretty=True), encoding="utf-8", newline="\n"
    )


@pytest.mark.parametrize("target", ["record", "manifest"])
def test_audit_and_export_reject_tampered_input_by_default(
    target: str, generated_demo: Path, tmp_path: Path
) -> None:
    _tamper(generated_demo, target)
    with pytest.raises(CorpusIntegrityError):
        audit_corpus(generated_demo)
    destination = tmp_path / "must-not-exist"
    with pytest.raises(CorpusIntegrityError):
        export_corpus(generated_demo, "jsonl", destination)
    assert not destination.exists()


def test_forensic_override_remains_a_failure(
    generated_demo: Path, tmp_path: Path
) -> None:
    _tamper(generated_demo, "record")
    report = audit_corpus(generated_demo, allow_invalid=True)
    integrity = next(finding for finding in report.findings if finding.risk == "corpus_integrity")
    assert report.failed
    assert integrity.status == "FAIL"
    result = export_corpus(
        generated_demo,
        "jsonl",
        tmp_path / "forensic",
        allow_invalid=True,
    )
    assert result["integrity_valid"] is False

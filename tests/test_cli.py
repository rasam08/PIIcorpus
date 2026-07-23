from __future__ import annotations

from pathlib import Path

import pytest

from piicorpus.cli import EXIT_FINDINGS, EXIT_OK, EXIT_OPERATIONAL, main


def test_cli_generate_validate_audit_report_and_export(
    tmp_path: Path, demo_config_path: Path
) -> None:
    corpus = tmp_path / "corpus"
    assert main(["generate", "--config", str(demo_config_path), "--out", str(corpus)]) == EXIT_OK
    assert main(["validate", str(corpus), "--strict"]) == EXIT_OK
    assert main(["audit", str(corpus), "--format", "json"]) == EXIT_OK
    assert main(["report", str(corpus)]) == EXIT_OK
    assert main(["export", str(corpus), "--format", "spacy"]) == EXIT_OK


def test_cli_operational_error_is_not_a_clean_verdict(tmp_path: Path) -> None:
    assert main(["validate", str(tmp_path / "missing")]) == EXIT_OPERATIONAL


def test_cli_report_surfaces_warnings_and_unmeasured_findings(
    generated_demo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["report", str(generated_demo)]) == EXIT_OK
    captured = capsys.readouterr()
    assert "audit=PASS_WITH_WARNINGS_AND_UNMEASURED" in captured.out


def test_cli_reproduce_confirms_byte_identity(generated_demo: Path) -> None:
    assert main(["reproduce", str(generated_demo)]) == EXIT_OK


def test_cli_reproduce_detects_drift(generated_demo: Path) -> None:
    path = generated_demo / "splits" / "train.jsonl"
    payload = path.read_text(encoding="utf-8")
    path.write_text(
        payload.replace("fictional", "fabricated", 1), encoding="utf-8", newline="\n"
    )
    assert main(["reproduce", str(generated_demo)]) == EXIT_FINDINGS


def test_cli_reproduce_rejects_other_generator_versions(generated_demo: Path) -> None:
    import json

    manifest_path = generated_demo / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["generator_version"] = "0.0.1"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    assert main(["reproduce", str(generated_demo)]) == EXIT_OPERATIONAL


def test_cli_returns_findings_for_tampering(generated_demo: Path) -> None:
    path = generated_demo / "splits/eval.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8", newline="\n")
    assert main(["validate", str(generated_demo), "--strict"]) == EXIT_FINDINGS
    assert main(["audit", str(generated_demo)]) == EXIT_FINDINGS
    assert main(["export", str(generated_demo), "--format", "jsonl"]) == EXIT_FINDINGS
    assert (
        main(
            [
                "audit",
                str(generated_demo),
                "--forensic-allow-invalid",
            ]
        )
        == EXIT_FINDINGS
    )

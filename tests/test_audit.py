from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from piicorpus.failure_model import audit_corpus
from piicorpus.validators import validate_corpus


def _builder() -> ModuleType:
    path = Path(__file__).parents[1] / "examples" / "deliberately_bad" / "build_examples.py"
    spec = importlib.util.spec_from_file_location("piicorpus_bad_examples", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_clean_demo_passes_measured_risks_and_keeps_holdout_unmeasured(
    generated_demo: Path,
) -> None:
    report = audit_corpus(generated_demo)
    statuses = {finding.risk: finding.status for finding in report.findings}
    assert set(statuses.values()) == {"PASS", "UNMEASURED"}
    assert statuses["same_generator_holdout_dependence"] == "UNMEASURED"
    assert "not an independent generalization test" in report.limitation


@pytest.mark.parametrize("case", sorted(_builder().case_catalog()))
def test_every_bad_example_fails_for_its_named_reason(
    case: str, generated_demo: Path, tmp_path: Path
) -> None:
    builder = _builder()
    expected_risk = builder.case_catalog()[case]
    output = tmp_path / case
    builder.build_bad_corpus(generated_demo, output, case)
    report = audit_corpus(output, allow_invalid=True)
    statuses = {finding.risk: finding.status for finding in report.findings}
    assert statuses[expected_risk] == "FAIL", (
        f"{case} did not trigger its intended risk {expected_risk}; "
        "an unrelated failure cannot satisfy this assertion"
    )


def test_multi_entity_cue_shortcut_uses_explicit_links(
    generated_demo: Path, tmp_path: Path
) -> None:
    builder = _builder()
    output = tmp_path / "cue-shortcut"
    builder.build_bad_corpus(generated_demo, output, "cue_shortcut")
    report = audit_corpus(output, allow_invalid=True)
    finding = next(item for item in report.findings if item.risk == "cue_label_shortcuts")
    assert finding.status == "FAIL"
    assert finding.details["fraction"] > 0.9


def test_split_local_evidence_gaps_are_reported(
    generated_demo: Path, tmp_path: Path
) -> None:
    builder = _builder()
    output = tmp_path / "missing-train-cue-free"
    builder.build_bad_corpus(generated_demo, output, "missing_cue_free")
    report = audit_corpus(output, allow_invalid=True)
    finding = next(item for item in report.findings if item.risk == "cue_free_coverage")
    assert finding.status == "FAIL"
    assert finding.details["missing_splits"] == ["train"]
    assert finding.details["per_split"]["eval"] > 0
    assert finding.details["per_split"]["holdout"] > 0


def test_negative_only_marker_is_detected_as_generator_fingerprint(
    generated_demo: Path, tmp_path: Path
) -> None:
    builder = _builder()
    output = tmp_path / "marker-shortcut"
    builder.build_bad_corpus(generated_demo, output, "label_marker_shortcut")
    report = audit_corpus(output)
    finding = next(item for item in report.findings if item.risk == "generator_fingerprint")
    assert finding.status == "FAIL"
    assert any(
        feature["feature"] == "synthetic sample index"
        for feature in finding.details["marker_features"]
    )


@pytest.mark.parametrize(
    ("case", "feature", "token_count"),
    [
        ("unigram_label_marker", "negmarker", 1),
        ("bigram_label_marker", "subject organization", 2),
        ("trigram_label_marker", "context subject synthetic", 3),
    ],
)
def test_kind_markers_are_detected_at_every_supported_width(
    case: str,
    feature: str,
    token_count: int,
    generated_demo: Path,
    tmp_path: Path,
) -> None:
    builder = _builder()
    output = tmp_path / case
    builder.build_bad_corpus(generated_demo, output, case)
    assert validate_corpus(output, strict=True).valid
    report = audit_corpus(output)
    finding = next(item for item in report.findings if item.risk == "generator_fingerprint")
    assert finding.status == "FAIL"
    assert any(
        marker["feature"] == feature and marker["token_count"] == token_count
        for marker in finding.details["marker_features"]
    )
    assert not any(
        marker["token_count"] < token_count
        for marker in finding.details["marker_features"]
    )


def test_train_only_cue_shortcut_fails_despite_safe_other_splits(
    generated_demo: Path, tmp_path: Path
) -> None:
    builder = _builder()
    output = tmp_path / "train-only-cue-shortcut"
    builder.build_bad_corpus(generated_demo, output, "train_only_cue_shortcut")
    assert validate_corpus(output, strict=True).valid
    report = audit_corpus(output)
    finding = next(item for item in report.findings if item.risk == "cue_label_shortcuts")
    assert finding.status == "FAIL"
    assert finding.details["failed_splits"] == ["train"]
    assert finding.details["per_split"]["train"]["fraction"] == 1.0
    assert finding.details["per_split"]["eval"]["fraction"] == 0.0
    assert finding.details["per_split"]["holdout"]["fraction"] == 0.0

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest

from piicorpus.config import load_config, reference_audit_config
from piicorpus.failure_model import (
    AuditContext,
    _check_value_diversity,
    _check_value_shared_affix,
    audit_corpus,
)
from piicorpus.generator import generate
from piicorpus.models import Annotation, Finding, Record
from piicorpus.validators import validate_corpus


def _builder() -> ModuleType:
    path = Path(__file__).parents[1] / "examples" / "deliberately_bad" / "build_examples.py"
    spec = importlib.util.spec_from_file_location("piicorpus_bad_examples", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _value_record(index: int, label: str, value: str) -> Record:
    annotation = Annotation(
        entity_type=label,
        start=0,
        end=len(value),
        byte_start=0,
        byte_end=len(value.encode("utf-8")),
        text=value,
    )
    return Record(
        case_id=f"test-{index}",
        split="train",
        family="test",
        namespace=f"test/{index}",
        template_id="test",
        kind="positive",
        provenance="generated",
        text=value,
        annotations=(annotation,),
    )


def test_clean_demo_passes_measured_risks_and_keeps_holdout_unmeasured(
    generated_demo: Path,
) -> None:
    report = audit_corpus(generated_demo)
    statuses = {finding.risk: finding.status for finding in report.findings}
    assert not report.failed, {
        risk: status for risk, status in statuses.items() if status == "FAIL"
    }
    assert set(statuses.values()) <= {"PASS", "WARN", "UNMEASURED"}
    assert statuses["same_generator_holdout_dependence"] == "UNMEASURED"
    # The demo's SYN- value prefixes are deliberately visible as a warning.
    assert statuses["value_shared_affix"] == "WARN"
    affix = next(f for f in report.findings if f.risk == "value_shared_affix")
    assert isinstance(affix.measured, int)
    assert isinstance(affix.threshold, int)
    assert affix.measured > affix.threshold
    assert statuses["threshold_strictness"] == "PASS"
    assert "not an independent generalization test" in report.limitation


def test_threshold_strictness_covers_similarity_evidence_and_probe_thresholds(
    demo_config_path: Path, tmp_path: Path
) -> None:
    config = load_config(demo_config_path)
    probe = replace(
        config.audit.probe,
        max_kind_accuracy=0.99,
        max_value_label_accuracy=0.99,
        max_context_label_accuracy=0.99,
    )
    lax_audit = replace(
        config.audit,
        near_duplicate_jaccard=0.99,
        intra_split_similarity_threshold=0.99,
        minimum_marker_kind_coverage=0.60,
        minimum_marker_support=30,
        minimum_shape_support=30,
        probe=probe,
    )
    corpus = tmp_path / "lax-thresholds"
    generate(replace(config, audit=lax_audit), corpus)
    report = audit_corpus(corpus)
    strictness = next(
        finding for finding in report.findings if finding.risk == "threshold_strictness"
    )
    assert strictness.status == "WARN"
    keys = {
        entry["key"] for entry in strictness.details["weaker_than_reference"]
    }
    assert {
        "intra_split_similarity_threshold",
        "minimum_marker_kind_coverage",
        "minimum_marker_support",
        "minimum_shape_support",
        "near_duplicate_jaccard",
        "probe.max_context_label_accuracy",
        "probe.max_kind_accuracy",
        "probe.max_value_label_accuracy",
    } <= keys


def test_value_diversity_verdict_uses_the_per_label_floor() -> None:
    rows = tuple(
        _value_record(
            index,
            "LABEL_A" if index < 30 else "LABEL_B",
            f"V{index:02d}",
        )
        for index in range(60)
    )
    context = AuditContext(
        split_records={"train": list(rows)},
        rows=rows,
        thresholds=reference_audit_config(),
    )
    finding = _check_value_diversity(context)[0]
    assert finding.status == "PASS"
    assert finding.measured == finding.threshold == 30
    assert finding.details["distinct_count_entropy_bits"] < 6.0


def test_shared_affix_compares_character_length_to_character_ceiling() -> None:
    def finding_for(prefix: str) -> Finding:
        rows = tuple(
            _value_record(index, "LABEL_A", f"{prefix}{index:02d}")
            for index in range(30)
        )
        context = AuditContext(
            split_records={"train": list(rows)},
            rows=rows,
            thresholds=reference_audit_config(),
        )
        return _check_value_shared_affix(context)[0]

    at_ceiling = finding_for("ABCDEF")
    assert at_ceiling.status == "PASS"
    assert at_ceiling.measured == at_ceiling.threshold == 6
    over_ceiling = finding_for("ABCDEFG")
    assert over_ceiling.status == "WARN"
    assert over_ceiling.measured == 7
    assert over_ceiling.threshold == 6


def test_lenient_thresholds_are_warned_about(generated_demo: Path) -> None:
    report = audit_corpus(generated_demo, profile="reference")
    strictness = next(
        finding for finding in report.findings if finding.risk == "threshold_strictness"
    )
    assert strictness.status == "PASS"
    assert "reference" in strictness.reason


@pytest.mark.parametrize("case", sorted(_builder().case_catalog()))
def test_every_bad_example_fails_for_its_named_reason(
    case: str, generated_demo: Path, tmp_path: Path
) -> None:
    builder = _builder()
    expected_risk = builder.case_catalog()[case]
    output = tmp_path / case
    builder.build_bad_corpus(generated_demo, output, case)
    report = audit_corpus(
        output,
        allow_invalid=True,
        probe=True if expected_risk.startswith("probe_") else None,
    )
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

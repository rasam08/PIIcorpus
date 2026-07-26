from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from piicorpus.failure_model import audit_corpus
from piicorpus.probe import _run_task, _task_finding, _train

PROBE_RISKS = (
    "probe_kind_separability",
    "probe_value_label_shortcut",
    "probe_context_label_shortcut",
)


def test_probe_is_unmeasured_unless_requested(generated_demo: Path) -> None:
    report = audit_corpus(generated_demo)
    statuses = {finding.risk: finding.status for finding in report.findings}
    for risk in PROBE_RISKS:
        assert statuses[risk] == "UNMEASURED"


def test_probe_measures_demo_learnability_deterministically(generated_demo: Path) -> None:
    first = audit_corpus(generated_demo, probe=True)
    second = audit_corpus(generated_demo, probe=True)
    first_probe = [f.to_dict() for f in first.findings if f.risk in PROBE_RISKS]
    second_probe = [f.to_dict() for f in second.findings if f.risk in PROBE_RISKS]
    assert first_probe == second_probe
    for finding in first.findings:
        if finding.risk in PROBE_RISKS:
            assert finding.status in {"PASS", "FAIL"}
            assert finding.measured is not None
            assert finding.details["accuracy_per_split"]
            assert finding.details["balanced_accuracy_per_split"]
            assert finding.details["majority_baseline_per_split"]
            assert finding.details["balanced_majority_baseline_per_split"]


def test_probe_does_not_treat_majority_priors_as_a_shortcut() -> None:
    per_split = {
        "train": [({}, 0)] * 95 + [({}, 1)] * 5,
        "eval": [({}, 0)] * 95 + [({}, 1)] * 5,
    }
    metrics = _run_task(per_split, "train", 2)
    finding = _task_finding(
        "test_probe",
        metrics,
        0.90,
        source="test",
        description="classify examples",
    )
    assert finding.details["accuracy_per_split"]["eval"] == 0.95
    assert finding.details["majority_baseline_per_split"]["eval"] == 0.95
    assert finding.details["balanced_accuracy_per_split"]["eval"] == 0.5
    assert finding.status == "PASS"


def test_probe_is_unmeasured_when_training_has_only_one_observed_class() -> None:
    result = _run_task(
        {
            "train": [({}, 1)] * 12,
            "eval": [({}, 1)] * 4,
        },
        "train",
        2,
    )
    finding = _task_finding(
        "probe_kind_separability",
        result,
        0.90,
        source="test",
        description="separate positives from hard negatives",
    )
    assert finding.status == "UNMEASURED"
    assert finding.measured is None
    assert "fewer than two observed classes" in finding.reason


def test_probe_is_unmeasured_when_held_split_has_only_one_observed_class() -> None:
    result = _run_task(
        {
            "train": [({}, 0)] * 6 + [({}, 1)] * 6,
            "eval": [({}, 1)] * 4,
        },
        "train",
        2,
    )
    finding = _task_finding(
        "probe_kind_separability",
        result,
        0.90,
        source="test",
        description="separate positives from hard negatives",
    )
    assert finding.status == "UNMEASURED"
    assert finding.details["unmeasured_splits"] == {
        "eval": "the held split contains fewer than two observed classes"
    }


def test_probe_training_uses_plain_weight_dicts() -> None:
    weights, _bias = _train(
        [({1: 1.0}, 0), ({2: 1.0}, 1)],
        2,
    )
    assert all(not isinstance(class_weights, defaultdict) for class_weights in weights)

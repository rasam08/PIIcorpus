from __future__ import annotations

from pathlib import Path

from piicorpus.failure_model import audit_corpus
from piicorpus.probe import _run_task, _task_finding

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

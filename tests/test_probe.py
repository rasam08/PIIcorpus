from __future__ import annotations

from pathlib import Path

from piicorpus.failure_model import audit_corpus

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
    kind = next(f for f in first.findings if f.risk == "probe_kind_separability")
    # The measured accuracy must beat guessing before it can mean anything.
    assert kind.details["majority_baseline"] < 1.0

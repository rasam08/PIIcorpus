from __future__ import annotations

from pathlib import Path

import pytest

from piicorpus.config import load_config
from piicorpus.failure_model import audit_corpus
from piicorpus.generator import generate
from piicorpus.plugins_realistic import (
    _luhn_valid,
    _verify_card_shaped_invalid,
    _verify_documentation_ip,
    _verify_never_issued_ssn,
    _verify_reserved_email,
    _verify_reserved_phone,
)
from piicorpus.validators import validate_corpus

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def realistic_corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("realistic") / "corpus"
    generate(load_config(ROOT / "configs" / "realistic-safe.toml"), output)
    return output


def test_realistic_corpus_validates_and_audits_clean(realistic_corpus: Path) -> None:
    assert validate_corpus(realistic_corpus, strict=True).valid
    report = audit_corpus(realistic_corpus)
    statuses = {finding.risk: finding.status for finding in report.findings}
    assert not report.failed, {
        risk: status for risk, status in statuses.items() if status == "FAIL"
    }
    # Realistic values carry no constant synthetic affix, unlike the SYN- demo.
    assert statuses["value_shared_affix"] == "PASS"
    # The probe is enabled by this configuration and must stay under its ceilings.
    assert statuses["probe_kind_separability"] == "PASS"
    assert statuses["probe_value_label_shortcut"] == "PASS"


def test_realistic_values_are_verified_reserved(realistic_corpus: Path) -> None:
    from piicorpus.manifest import load_corpus

    config, split_records, _manifest = load_corpus(realistic_corpus)
    assert config.safety.mode == "verifier"
    plugin_by_label = {label.name: label.plugin for label in config.labels}
    verifiers = {
        "reserved_email": _verify_reserved_email,
        "reserved_phone_nanp": _verify_reserved_phone,
        "card_shaped_invalid": _verify_card_shaped_invalid,
        "documentation_ip": _verify_documentation_ip,
        "never_issued_ssn_shape": _verify_never_issued_ssn,
    }
    checked = 0
    for rows in split_records.values():
        for record in rows:
            for annotation in record.annotations:
                verifier = verifiers[plugin_by_label[annotation.entity_type]]
                assert verifier(annotation.text), (
                    annotation.entity_type,
                    annotation.text,
                )
                checked += 1
    assert checked > 200


def test_verifiers_reject_unreserved_lookalikes() -> None:
    assert not _verify_reserved_email("person@gmail.example.co")
    assert _verify_reserved_email("sub@mail.example.org")
    assert not _verify_reserved_phone("(202) 555-1234")
    assert _verify_reserved_phone("(202) 555-0134")
    # 4111111111111111 is Luhn-valid, so the invalid-by-construction verifier
    # must reject it even though the shape matches.
    assert _luhn_valid("4111111111111111")
    assert not _verify_card_shaped_invalid("4111-1111-1111-1111")
    assert not _verify_documentation_ip("8.8.8.8")
    assert _verify_documentation_ip("192.0.2.55")
    assert _verify_documentation_ip("2001:db8::17")
    assert not _verify_never_issued_ssn("123-45-6789")
    assert _verify_never_issued_ssn("978-52-0141")

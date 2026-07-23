from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from piicorpus.cli import EXIT_OK, EXIT_OPERATIONAL, main
from piicorpus.morphology import registered_shapes

PLUGIN_IMPLEMENTATION = '''
from piicorpus import FamilyPlugin, register_family, register_shape, register_value_plugin


def acme_id(rng, label, split, index):
    return f"SYN-ACME-{rng.choice('ABCDEFGH')}{rng.randint(10000, 99999)}"


def register():
    register_value_plugin("acme_id", acme_id, replace=True)
    register_family(
        FamilyPlugin(
            name="acme_notes",
            role="positive",
            templates=(
                "{persona} filed the {cue} {value} with {organization}.",
                "The {cue} for {persona} is {value}.",
                "{organization} recorded {value} as the {cue} of {persona}.",
                "According to {persona}, the {cue} reads {value}.",
                "A note ties {persona} to {cue} {value}.",
                "For {persona}, {organization} lists the {cue} as {value}.",
            ),
        ),
        replace=True,
    )
    register_shape("acme_shape", r"SYN-ACME-[A-Z]\\d{5}", replace=True)
'''

PLUGIN_MODULE = PLUGIN_IMPLEMENTATION + "\nregister()\n"

PLUGIN_CONFIG = """
project_name = "Acme plugin demo"
seed = 7
generated_data_license = "CC0-1.0"
positive_ratio = 0.6
minimum_hard_negative_ratio = 0.35

[splits]
train = 12
eval = 12
holdout = 12

[diversity]
minimum_templates_per_family = 2
minimum_personas_per_family = 3
minimum_organizations_per_split = 3

[audit]
max_morphology_label_share = 0.9
max_label_exclusive_cue_fraction = 0.9
max_family_share = 0.9
minimum_hard_negative_kinds = 1
max_template_share = 0.5
max_kind_marker_share = 0.98
minimum_marker_kind_coverage = 0.5
minimum_marker_support = 20

[safety]
reserved_email_domains = ["example.com"]
allowed_value_prefixes = ["SYN-"]
forbidden_terms = []

[[labels]]
name = "TEST_VALUE_ID"
plugin = "acme_id"
morphology_group = "acme"
cues = ["acme reference"]

[[families]]
name = "acme_notes"
plugin = "acme_notes"
role = "positive"

[[families]]
name = "acme_near_miss"
plugin = "near_miss"
role = "hard_negative"
"""


def test_cli_loads_registration_modules_for_custom_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "acme_plugin.py").write_text(
        textwrap.dedent(PLUGIN_MODULE), encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    config_path = tmp_path / "acme.toml"
    config_path.write_text(PLUGIN_CONFIG, encoding="utf-8")
    corpus = tmp_path / "corpus"
    assert (
        main(
            [
                "--plugins",
                "acme_plugin",
                "generate",
                "--config",
                str(config_path),
                "--out",
                str(corpus),
            ]
        )
        == EXIT_OK
    )
    assert (
        main(["--plugins", "acme_plugin", "validate", str(corpus), "--strict"]) == EXIT_OK
    )
    assert "acme_shape" in registered_shapes()
    payload = (corpus / "splits" / "train.jsonl").read_text(encoding="utf-8")
    assert "SYN-ACME-" in payload


def test_cli_loads_callable_plugin_entry_points_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "acme_entry.py").write_text(
        textwrap.dedent(PLUGIN_IMPLEMENTATION), encoding="utf-8"
    )
    distribution = tmp_path / "acme_entry_test-1.0.dist-info"
    distribution.mkdir()
    (distribution / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: acme-entry-test\nVersion: 1.0\n",
        encoding="utf-8",
    )
    (distribution / "entry_points.txt").write_text(
        "[piicorpus.plugins]\nacme = acme_entry:register\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    config_path = tmp_path / "acme.toml"
    config_path.write_text(PLUGIN_CONFIG, encoding="utf-8")
    corpus = tmp_path / "entry-point-corpus"
    assert (
        main(
            [
                "generate",
                "--config",
                str(config_path),
                "--out",
                str(corpus),
            ]
        )
        == EXIT_OK
    )
    assert "acme_shape" in registered_shapes()
    assert "SYN-ACME-" in (
        corpus / "splits" / "train.jsonl"
    ).read_text(encoding="utf-8")


def test_missing_plugin_module_is_an_operational_error(tmp_path: Path) -> None:
    assert (
        main(["--plugins", "definitely_missing_module", "validate", str(tmp_path)])
        == EXIT_OPERATIONAL
    )

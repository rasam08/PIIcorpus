from __future__ import annotations

from pathlib import Path

import pytest

from piicorpus.config import ConfigError, config_from_dict, load_config


@pytest.mark.parametrize(
    ("key", "entry"),
    [
        ("allowed_value_prefixes", ""),
        ("allowed_value_prefixes", "   "),
        ("reserved_email_domains", ""),
        ("reserved_email_domains", "   "),
    ],
)
def test_safety_entries_cannot_be_empty(
    demo_config_path: Path,
    tmp_path: Path,
    key: str,
    entry: str,
) -> None:
    lines = demo_config_path.read_text(encoding="utf-8").splitlines()
    rewritten = [
        f'{key} = ["{entry}"]' if line.startswith(f"{key} = ") else line
        for line in lines
    ]
    path = tmp_path / "invalid-safety.toml"
    path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="entries cannot be empty"):
        load_config(path)


def test_normalized_config_cannot_bypass_empty_prefix_validation(
    demo_config_path: Path,
) -> None:
    payload = load_config(demo_config_path).to_dict()
    payload["safety"]["allowed_value_prefixes"] = [""]
    with pytest.raises(ConfigError, match="allowed_value_prefixes"):
        config_from_dict(payload)

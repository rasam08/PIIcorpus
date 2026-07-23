"""TOML configuration loading and strict validation."""

from __future__ import annotations

import hashlib
import re
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .models import stable_json


class ConfigError(ValueError):
    """Raised when a corpus configuration is invalid."""


NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,63}$")
SLUG_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


@dataclass(frozen=True, slots=True)
class LabelConfig:
    name: str
    plugin: str
    cues: tuple[str, ...]
    morphology_group: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FamilyConfig:
    name: str
    plugin: str
    role: str
    weight: int = 1
    templates: tuple[str, ...] = field(default_factory=tuple)
    labels: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class DiversityConfig:
    minimum_templates_per_family: int
    minimum_personas_per_family: int
    minimum_organizations_per_split: int


@dataclass(frozen=True, slots=True)
class AuditConfig:
    max_morphology_label_share: float
    max_label_exclusive_cue_fraction: float
    max_family_share: float
    minimum_value_entropy_bits: float
    minimum_hard_negative_kinds: int
    max_template_share: float
    max_kind_marker_share: float
    minimum_marker_kind_coverage: float
    minimum_marker_support: int


@dataclass(frozen=True, slots=True)
class SafetyConfig:
    reserved_email_domains: tuple[str, ...]
    allowed_value_prefixes: tuple[str, ...]
    forbidden_terms: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CorpusConfig:
    project_name: str
    seed: int
    generated_data_license: str
    positive_ratio: float
    minimum_hard_negative_ratio: float
    splits: dict[str, int]
    labels: tuple[LabelConfig, ...]
    families: tuple[FamilyConfig, ...]
    diversity: DiversityConfig
    audit: AuditConfig
    safety: SafetyConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def digest(self) -> str:
        payload = stable_json(self.to_dict()).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _required(
    data: dict[str, Any], key: str, expected: type[Any] | tuple[type[Any], ...]
) -> Any:
    if key not in data:
        raise ConfigError(f"missing configuration key: {key}")
    value = data[key]
    if not isinstance(value, expected):
        if isinstance(expected, tuple):
            expected_name = " or ".join(item.__name__ for item in expected)
        else:
            expected_name = expected.__name__
        raise ConfigError(f"configuration key {key} must be {expected_name}")
    return value


def _ratio(value: Any, key: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f"{key} must be numeric")
    result = float(value)
    if not 0.0 < result < 1.0:
        raise ConfigError(f"{key} must be between 0 and 1")
    return result


def load_config(path: str | Path) -> CorpusConfig:
    source = Path(path)
    try:
        with source.open("rb") as handle:
            raw = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"cannot read TOML configuration: {exc}") from exc

    labels: list[LabelConfig] = []
    seen_labels: set[str] = set()
    for item in _required(raw, "labels", list):
        if not isinstance(item, dict):
            raise ConfigError("every labels entry must be a table")
        name = str(_required(item, "name", str))
        if not NAME_RE.fullmatch(name) or name in seen_labels:
            raise ConfigError(f"invalid or duplicate label name: {name}")
        seen_labels.add(name)
        cues_raw = _required(item, "cues", list)
        cues = tuple(str(c).strip() for c in cues_raw if str(c).strip())
        if not cues:
            raise ConfigError(f"label {name} needs at least one cue")
        labels.append(
            LabelConfig(
                name=name,
                plugin=str(_required(item, "plugin", str)),
                cues=cues,
                morphology_group=str(item.get("morphology_group", "default")),
                options=dict(item.get("options", {})),
            )
        )
    if not labels:
        raise ConfigError("at least one label is required")

    families: list[FamilyConfig] = []
    seen_families: set[str] = set()
    for item in _required(raw, "families", list):
        if not isinstance(item, dict):
            raise ConfigError("every families entry must be a table")
        name = str(_required(item, "name", str))
        if not SLUG_RE.fullmatch(name) or name in seen_families:
            raise ConfigError(f"invalid or duplicate family name: {name}")
        seen_families.add(name)
        role = str(_required(item, "role", str))
        if role not in {"positive", "hard_negative"}:
            raise ConfigError(f"family {name} role must be positive or hard_negative")
        weight = int(item.get("weight", 1))
        if weight < 1:
            raise ConfigError(f"family {name} weight must be positive")
        templates_raw = item.get("templates", [])
        if not isinstance(templates_raw, list):
            raise ConfigError(f"family {name} templates must be an array")
        labels_raw = item.get("labels", [])
        if not isinstance(labels_raw, list):
            raise ConfigError(f"family {name} labels must be an array")
        family_labels = tuple(str(value) for value in labels_raw)
        unknown_labels = sorted(set(family_labels) - seen_labels)
        if unknown_labels:
            raise ConfigError(f"family {name} uses unknown labels: {unknown_labels}")
        families.append(
            FamilyConfig(
                name=name,
                plugin=str(_required(item, "plugin", str)),
                role=role,
                weight=weight,
                templates=tuple(str(v) for v in templates_raw),
                labels=family_labels,
            )
        )
    if not any(f.role == "positive" for f in families):
        raise ConfigError("at least one positive family is required")
    if not any(f.role == "hard_negative" for f in families):
        raise ConfigError("at least one hard-negative family is required")

    splits_raw = _required(raw, "splits", dict)
    splits = {name: int(splits_raw.get(name, 0)) for name in ("train", "eval", "holdout")}
    if any(size < 1 for size in splits.values()):
        raise ConfigError("train, eval and holdout split sizes must all be positive")

    diversity_raw = _required(raw, "diversity", dict)
    audit_raw = _required(raw, "audit", dict)
    safety_raw = _required(raw, "safety", dict)
    positive_ratio = _ratio(_required(raw, "positive_ratio", (int, float)), "positive_ratio")
    minimum_negative = _ratio(
        _required(raw, "minimum_hard_negative_ratio", (int, float)),
        "minimum_hard_negative_ratio",
    )
    if 1.0 - positive_ratio < minimum_negative:
        raise ConfigError("positive_ratio leaves fewer negatives than minimum_hard_negative_ratio")

    config = CorpusConfig(
        project_name=str(_required(raw, "project_name", str)),
        seed=int(_required(raw, "seed", int)),
        generated_data_license=str(_required(raw, "generated_data_license", str)),
        positive_ratio=positive_ratio,
        minimum_hard_negative_ratio=minimum_negative,
        splits=splits,
        labels=tuple(labels),
        families=tuple(families),
        diversity=DiversityConfig(
            minimum_templates_per_family=int(
                _required(diversity_raw, "minimum_templates_per_family", int)
            ),
            minimum_personas_per_family=int(
                _required(diversity_raw, "minimum_personas_per_family", int)
            ),
            minimum_organizations_per_split=int(
                _required(diversity_raw, "minimum_organizations_per_split", int)
            ),
        ),
        audit=AuditConfig(
            max_morphology_label_share=_ratio(
                _required(audit_raw, "max_morphology_label_share", (int, float)),
                "max_morphology_label_share",
            ),
            max_label_exclusive_cue_fraction=_ratio(
                _required(audit_raw, "max_label_exclusive_cue_fraction", (int, float)),
                "max_label_exclusive_cue_fraction",
            ),
            max_family_share=_ratio(
                _required(audit_raw, "max_family_share", (int, float)), "max_family_share"
            ),
            minimum_value_entropy_bits=float(
                _required(audit_raw, "minimum_value_entropy_bits", (int, float))
            ),
            minimum_hard_negative_kinds=int(
                _required(audit_raw, "minimum_hard_negative_kinds", int)
            ),
            max_template_share=_ratio(
                _required(audit_raw, "max_template_share", (int, float)), "max_template_share"
            ),
            max_kind_marker_share=_ratio(
                _required(audit_raw, "max_kind_marker_share", (int, float)),
                "max_kind_marker_share",
            ),
            minimum_marker_kind_coverage=_ratio(
                _required(audit_raw, "minimum_marker_kind_coverage", (int, float)),
                "minimum_marker_kind_coverage",
            ),
            minimum_marker_support=int(
                _required(audit_raw, "minimum_marker_support", int)
            ),
        ),
        safety=SafetyConfig(
            reserved_email_domains=tuple(
                str(v).lower() for v in _required(safety_raw, "reserved_email_domains", list)
            ),
            allowed_value_prefixes=tuple(
                str(v) for v in _required(safety_raw, "allowed_value_prefixes", list)
            ),
            forbidden_terms=tuple(str(v) for v in safety_raw.get("forbidden_terms", [])),
        ),
    )
    if config.seed < 0:
        raise ConfigError("seed must be non-negative")
    if min(asdict(config.diversity).values()) < 1:
        raise ConfigError("diversity minimums must be positive")
    if config.audit.minimum_marker_support < 2:
        raise ConfigError("minimum_marker_support must be at least two")
    return config


def config_from_dict(value: dict[str, Any]) -> CorpusConfig:
    """Rehydrate the normalized configuration snapshot emitted with a corpus."""
    labels = tuple(
        LabelConfig(
            name=v["name"],
            plugin=v["plugin"],
            cues=tuple(v["cues"]),
            morphology_group=v["morphology_group"],
            options=dict(v.get("options", {})),
        )
        for v in value["labels"]
    )
    families = tuple(
        FamilyConfig(
            name=v["name"],
            plugin=v["plugin"],
            role=v["role"],
            weight=int(v["weight"]),
            templates=tuple(v.get("templates", [])),
            labels=tuple(v.get("labels", [])),
        )
        for v in value["families"]
    )
    return CorpusConfig(
        project_name=value["project_name"],
        seed=int(value["seed"]),
        generated_data_license=value["generated_data_license"],
        positive_ratio=float(value["positive_ratio"]),
        minimum_hard_negative_ratio=float(value["minimum_hard_negative_ratio"]),
        splits={k: int(v) for k, v in value["splits"].items()},
        labels=labels,
        families=families,
        diversity=DiversityConfig(**value["diversity"]),
        audit=AuditConfig(**value["audit"]),
        safety=SafetyConfig(
            reserved_email_domains=tuple(value["safety"]["reserved_email_domains"]),
            allowed_value_prefixes=tuple(value["safety"]["allowed_value_prefixes"]),
            forbidden_terms=tuple(value["safety"]["forbidden_terms"]),
        ),
    )

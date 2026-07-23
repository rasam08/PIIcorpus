"""TOML configuration loading and strict validation."""

from __future__ import annotations

import hashlib
import re
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .models import stable_json
from .profiles import REFERENCE_AUDIT, REFERENCE_PROBE


class ConfigError(ValueError):
    """Raised when a corpus configuration is invalid."""


NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,63}$")
SLUG_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")

SPLIT_ORDER = ("train", "eval", "holdout")

DEFAULT_PERSONAS = (
    "Ari Solen",
    "Bela Orin",
    "Cato Mire",
    "Dara Venn",
    "Eli Taren",
    "Fia Corren",
    "Galen Iver",
    "Hana Pell",
    "Ivo Saret",
    "Jora Wyn",
    "Kelan Dore",
    "Luma Quill",
    "Mira Fen",
    "Nico Vale",
    "Ona Rell",
    "Pax Arden",
    "Rina Cove",
    "Soren Lark",
    "Tavi Moor",
    "Una Brist",
    "Vero Alba",
    "Wren Odal",
    "Yara Sund",
    "Zeno Palt",
)
DEFAULT_ORGANIZATIONS = (
    "Juniper Test Clinic",
    "Cobalt Demo Center",
    "Lattice Fictional Care",
    "Marigold Example Practice",
    "Nimbus Sample Office",
    "Orchard Fictional Health",
    "Pebble Demo Institute",
    "Quarry Sample Care",
    "Rookery Fictional Group",
    "Sable Example Bureau",
    "Trellis Demo Partners",
    "Willow Sample Institute",
)


def split_partition(pool: tuple[str, ...], split: str) -> tuple[str, ...]:
    """Partition a surface pool into disjoint, interleaved split shares.

    The sorted pool is interleaved modulo the split count, so no split receives a
    contiguous (for example alphabetical) range and the same pool always yields the
    same partition. Interleaving reduces order-driven shift but does not claim that
    arbitrary user-supplied pools are statistically identical across splits.
    """
    return tuple(sorted(pool)[SPLIT_ORDER.index(split) :: len(SPLIT_ORDER)])


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
class ProbeConfig:
    """Learnability-probe settings; the probe is opt-in because it trains a model."""

    enabled: bool = False
    max_kind_accuracy: float = REFERENCE_PROBE["max_kind_accuracy"]
    max_value_label_accuracy: float = REFERENCE_PROBE["max_value_label_accuracy"]
    max_context_label_accuracy: float = REFERENCE_PROBE["max_context_label_accuracy"]


@dataclass(frozen=True, slots=True)
class AuditConfig:
    max_morphology_label_share: float
    max_label_exclusive_cue_fraction: float
    max_family_share: float
    minimum_hard_negative_kinds: int
    max_template_share: float
    max_kind_marker_share: float
    minimum_marker_kind_coverage: float
    minimum_marker_support: int
    near_duplicate_jaccard: float = float(REFERENCE_AUDIT["near_duplicate_jaccard"])
    intra_split_similarity_threshold: float = float(
        REFERENCE_AUDIT["intra_split_similarity_threshold"]
    )
    max_intra_split_near_dup_fraction: float = float(
        REFERENCE_AUDIT["max_intra_split_near_dup_fraction"]
    )
    max_shape_entity_share: float = float(REFERENCE_AUDIT["max_shape_entity_share"])
    minimum_shape_support: int = int(REFERENCE_AUDIT["minimum_shape_support"])
    max_pervasive_ngram_coverage: float = float(
        REFERENCE_AUDIT["max_pervasive_ngram_coverage"]
    )
    max_label_marker_share: float = float(REFERENCE_AUDIT["max_label_marker_share"])
    minimum_distinct_values_per_label: int = int(
        REFERENCE_AUDIT["minimum_distinct_values_per_label"]
    )
    max_shared_affix_chars: int = int(REFERENCE_AUDIT["max_shared_affix_chars"])
    probe: ProbeConfig = field(default_factory=ProbeConfig)


SAFETY_MODES = ("prefix", "verifier", "either")


@dataclass(frozen=True, slots=True)
class SafetyConfig:
    reserved_email_domains: tuple[str, ...]
    allowed_value_prefixes: tuple[str, ...]
    forbidden_terms: tuple[str, ...]
    mode: str = "either"


@dataclass(frozen=True, slots=True)
class SurfacesConfig:
    personas: tuple[str, ...]
    organizations: tuple[str, ...]


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
    surfaces: SurfacesConfig

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


def _surface_pool(raw: Any, key: str, defaults: tuple[str, ...]) -> tuple[str, ...]:
    if raw is None:
        return defaults
    if not isinstance(raw, list):
        raise ConfigError(f"surfaces.{key} must be an array of strings")
    pool = tuple(str(item).strip() for item in raw)
    if any(not item for item in pool):
        raise ConfigError(f"surfaces.{key} entries cannot be empty")
    if len(set(pool)) != len(pool):
        raise ConfigError(f"surfaces.{key} entries must be unique")
    return pool


def _surfaces(raw: dict[str, Any]) -> SurfacesConfig:
    return SurfacesConfig(
        personas=_surface_pool(raw.get("personas"), "personas", DEFAULT_PERSONAS),
        organizations=_surface_pool(
            raw.get("organizations"), "organizations", DEFAULT_ORGANIZATIONS
        ),
    )


def _probe(raw: Any) -> ProbeConfig:
    if not isinstance(raw, dict):
        raise ConfigError("audit.probe must be a table")
    defaults = ProbeConfig()
    return ProbeConfig(
        enabled=bool(raw.get("enabled", defaults.enabled)),
        max_kind_accuracy=_ratio(
            raw.get("max_kind_accuracy", defaults.max_kind_accuracy), "max_kind_accuracy"
        ),
        max_value_label_accuracy=_ratio(
            raw.get("max_value_label_accuracy", defaults.max_value_label_accuracy),
            "max_value_label_accuracy",
        ),
        max_context_label_accuracy=_ratio(
            raw.get("max_context_label_accuracy", defaults.max_context_label_accuracy),
            "max_context_label_accuracy",
        ),
    )


def _ratio(value: Any, key: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f"{key} must be numeric")
    result = float(value)
    if not 0.0 < result < 1.0:
        raise ConfigError(f"{key} must be between 0 and 1")
    return result


def reference_audit_config() -> AuditConfig:
    """Build an AuditConfig from the recommended reference profile."""
    return AuditConfig(
        max_morphology_label_share=float(REFERENCE_AUDIT["max_morphology_label_share"]),
        max_label_exclusive_cue_fraction=float(
            REFERENCE_AUDIT["max_label_exclusive_cue_fraction"]
        ),
        max_family_share=float(REFERENCE_AUDIT["max_family_share"]),
        minimum_hard_negative_kinds=int(REFERENCE_AUDIT["minimum_hard_negative_kinds"]),
        max_template_share=float(REFERENCE_AUDIT["max_template_share"]),
        max_kind_marker_share=float(REFERENCE_AUDIT["max_kind_marker_share"]),
        minimum_marker_kind_coverage=float(REFERENCE_AUDIT["minimum_marker_kind_coverage"]),
        minimum_marker_support=int(REFERENCE_AUDIT["minimum_marker_support"]),
    )


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
    surfaces_raw = raw.get("surfaces", {})
    if not isinstance(surfaces_raw, dict):
        raise ConfigError("surfaces must be a table")
    surfaces = _surfaces(surfaces_raw)
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
            near_duplicate_jaccard=_ratio(
                audit_raw.get(
                    "near_duplicate_jaccard", REFERENCE_AUDIT["near_duplicate_jaccard"]
                ),
                "near_duplicate_jaccard",
            ),
            intra_split_similarity_threshold=_ratio(
                audit_raw.get(
                    "intra_split_similarity_threshold",
                    REFERENCE_AUDIT["intra_split_similarity_threshold"],
                ),
                "intra_split_similarity_threshold",
            ),
            max_intra_split_near_dup_fraction=_ratio(
                audit_raw.get(
                    "max_intra_split_near_dup_fraction",
                    REFERENCE_AUDIT["max_intra_split_near_dup_fraction"],
                ),
                "max_intra_split_near_dup_fraction",
            ),
            max_shape_entity_share=_ratio(
                audit_raw.get(
                    "max_shape_entity_share", REFERENCE_AUDIT["max_shape_entity_share"]
                ),
                "max_shape_entity_share",
            ),
            minimum_shape_support=int(
                audit_raw.get("minimum_shape_support", REFERENCE_AUDIT["minimum_shape_support"])
            ),
            max_pervasive_ngram_coverage=_ratio(
                audit_raw.get(
                    "max_pervasive_ngram_coverage",
                    REFERENCE_AUDIT["max_pervasive_ngram_coverage"],
                ),
                "max_pervasive_ngram_coverage",
            ),
            max_label_marker_share=_ratio(
                audit_raw.get(
                    "max_label_marker_share", REFERENCE_AUDIT["max_label_marker_share"]
                ),
                "max_label_marker_share",
            ),
            minimum_distinct_values_per_label=int(
                audit_raw.get(
                    "minimum_distinct_values_per_label",
                    REFERENCE_AUDIT["minimum_distinct_values_per_label"],
                )
            ),
            max_shared_affix_chars=int(
                audit_raw.get(
                    "max_shared_affix_chars", REFERENCE_AUDIT["max_shared_affix_chars"]
                )
            ),
            probe=_probe(audit_raw.get("probe", {})),
        ),
        safety=SafetyConfig(
            reserved_email_domains=tuple(
                str(v).lower() for v in _required(safety_raw, "reserved_email_domains", list)
            ),
            allowed_value_prefixes=tuple(
                str(v) for v in _required(safety_raw, "allowed_value_prefixes", list)
            ),
            forbidden_terms=tuple(str(v) for v in safety_raw.get("forbidden_terms", [])),
            mode=str(safety_raw.get("mode", "either")),
        ),
        surfaces=surfaces,
    )
    if config.safety.mode not in SAFETY_MODES:
        raise ConfigError(f"safety.mode must be one of {', '.join(SAFETY_MODES)}")
    if config.seed < 0:
        raise ConfigError("seed must be non-negative")
    if min(asdict(config.diversity).values()) < 1:
        raise ConfigError("diversity minimums must be positive")
    if config.audit.minimum_marker_support < 2:
        raise ConfigError("minimum_marker_support must be at least two")
    for split in SPLIT_ORDER:
        personas_share = len(split_partition(config.surfaces.personas, split))
        organizations_share = len(split_partition(config.surfaces.organizations, split))
        if personas_share < config.diversity.minimum_personas_per_family:
            raise ConfigError(
                f"surfaces.personas leaves {split} only {personas_share} personas, below "
                f"minimum_personas_per_family; add more personas to the pool"
            )
        if organizations_share < config.diversity.minimum_organizations_per_split:
            raise ConfigError(
                f"surfaces.organizations leaves {split} only {organizations_share} "
                f"organizations, below minimum_organizations_per_split; add more to the pool"
            )
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
        audit=AuditConfig(
            **{
                k: v
                for k, v in value["audit"].items()
                if k not in {"minimum_value_entropy_bits", "probe"}
            },
            probe=ProbeConfig(**value["audit"].get("probe", {})),
        ),
        safety=SafetyConfig(
            reserved_email_domains=tuple(value["safety"]["reserved_email_domains"]),
            allowed_value_prefixes=tuple(value["safety"]["allowed_value_prefixes"]),
            forbidden_terms=tuple(value["safety"]["forbidden_terms"]),
            mode=str(value["safety"].get("mode", "either")),
        ),
        surfaces=SurfacesConfig(
            personas=tuple(value.get("surfaces", {}).get("personas", DEFAULT_PERSONAS)),
            organizations=tuple(
                value.get("surfaces", {}).get("organizations", DEFAULT_ORGANIZATIONS)
            ),
        ),
    )

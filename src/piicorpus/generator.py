"""Deterministic, plugin-based synthetic contextual-PII corpus generation."""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from .annotation import parse_marked
from .config import SPLIT_ORDER, CorpusConfig, FamilyConfig, LabelConfig, split_partition
from .identity import derive_case_id
from .manifest import write_corpus
from .models import CueLink, Record
from .morphology import register_shape, shape_signature
from .skeletons import get_family

GENERATOR_VERSION = "2.0.0"

ValueGenerator = Callable[[random.Random, LabelConfig, str, int], str]
_VALUE_PLUGINS: dict[str, ValueGenerator] = {}


def register_value_plugin(name: str, plugin: ValueGenerator, *, replace: bool = False) -> None:
    if name in _VALUE_PLUGINS and not replace:
        raise ValueError(f"value plugin is already registered: {name}")
    _VALUE_PLUGINS[name] = plugin


def registered_value_plugins() -> tuple[str, ...]:
    return tuple(sorted(_VALUE_PLUGINS))


# I and O are excluded as visually ambiguous. Letters and years are interleaved
# across splits to avoid contiguous-range shift while staying disjoint.
_LETTER_POOL = tuple("ABCDEFGHJKLMNPQRSTUVWXYZ")
_YEAR_POOL = tuple(str(year) for year in range(1971, 2019))


def _split_letters(split: str) -> str:
    return "".join(split_partition(_LETTER_POOL, split))


def _split_years(split: str) -> tuple[int, ...]:
    return tuple(int(year) for year in split_partition(_YEAR_POOL, split))


def _digits(rng: random.Random, count: int) -> str:
    while True:
        result = "".join(rng.choice("0123456789") for _ in range(count))
        if len(set(result)) > 1:
            return result


def _fictional_identifier(rng: random.Random, _label: LabelConfig, split: str, index: int) -> str:
    alphabet = _split_letters(split)
    shape_names = (
        "synthetic_alpha_five",
        "synthetic_two_alpha_four",
        "synthetic_segmented",
    )
    override = _label.options.get("_shape_override")
    shape = shape_names.index(override) if override in shape_names else index % 3
    if shape == 0:
        return f"SYN-ID-{rng.choice(alphabet)}{_digits(rng, 5)}"
    if shape == 1:
        return f"SYN-ID-{rng.choice(alphabet)}{rng.choice(alphabet)}{_digits(rng, 4)}"
    return f"SYN-ID-{''.join(rng.choice(alphabet) for _ in range(3))}-{_digits(rng, 3)}"


def _synthetic_date(rng: random.Random, _label: LabelConfig, split: str, _index: int) -> str:
    year = rng.choice(_split_years(split))
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    return f"SYN-DATE-{year:04d}-{month:02d}-{day:02d}"


register_value_plugin("fictional_identifier", _fictional_identifier)
register_value_plugin("synthetic_date", _synthetic_date)

# Built-in shapes; specific patterns register before the noisy catch-all.
register_shape("spoken_synthetic", lambda value: value.casefold().strip().startswith("synthetic "))
register_shape("synthetic_calendar", r"SYN-DATE-\d{4}-\d{2}-\d{2}")
register_shape("synthetic_alpha_five", r"SYN-ID-[A-Z]\d{5}")
register_shape("synthetic_two_alpha_four", r"SYN-ID-[A-Z]{2}\d{4}")
register_shape("synthetic_segmented", r"SYN-ID-[A-Z]{3}-\d{3}")
register_shape("synthetic_noisy", lambda value: value.startswith("SYN-ID-"))


def _rng(config: CorpusConfig, *parts: object) -> random.Random:
    material = "|".join(str(v) for v in (GENERATOR_VERSION, config.seed, config.digest, *parts))
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _split_templates(family: FamilyConfig, split: str) -> tuple[str, ...]:
    bank = family.templates or get_family(family.plugin).templates
    if len(bank) < 6:
        raise ValueError(f"family {family.name} needs at least six templates for split isolation")
    index = SPLIT_ORDER.index(split)
    width = len(bank) // 3
    start = index * width
    stop = (index + 1) * width if index < 2 else len(bank)
    result = bank[start:stop]
    if not result:
        raise ValueError(f"family {family.name} has no templates for {split}")
    return result


def _allocation(total: int, families: tuple[FamilyConfig, ...]) -> dict[str, int]:
    deck = [family.name for family in families for _ in range(family.weight)]
    counts: Counter[str] = Counter()
    for index in range(total):
        counts[deck[index % len(deck)]] += 1
    return {family.name: counts[family.name] for family in families}


def _spell(value: str) -> str:
    words = {
        "0": "zero",
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine",
        "-": "dash",
    }
    chunks = [words.get(char, char.lower()) for char in value.removeprefix("SYN-")]
    return "synthetic " + " ".join(chunks)


def _ocr_noise(value: str, index: int) -> str:
    substitutions = (("0", "O"), ("1", "l"), ("5", "S"), ("8", "B"))
    source, replacement = substitutions[index % len(substitutions)]
    if source in value:
        return value.replace(source, replacement, 1)
    return value + (" " if index % 2 else ".")


def _value(
    config: CorpusConfig,
    label: LabelConfig,
    split: str,
    index: int,
    family: str,
    seen: set[str],
    *,
    plugin_label: LabelConfig | None = None,
) -> str:
    try:
        plugin = _VALUE_PLUGINS[label.plugin]
    except KeyError as exc:
        raise ValueError(f"unknown value plugin: {label.plugin}") from exc
    for attempt in range(100):
        rng = _rng(config, split, family, label.name, index, "value", attempt)
        candidate = plugin(rng, plugin_label or label, split, index + attempt * 97)
        if candidate not in seen:
            seen.add(candidate)
            return candidate
    raise ValueError(f"could not generate a unique value for {label.name}")


def _negative_value(
    config: CorpusConfig, split: str, family_plugin: str, index: int, seen: set[str]
) -> str:
    """Emit hard-negative surface tokens.

    Near-miss and adjacent values are produced by the configured label plugins
    themselves (near misses include an OCR-noised variant), so hard negatives
    mirror the positive value distribution for any label set and no shape occurs
    exclusively as an annotated entity. The shared ``seen`` pool keeps them
    disjoint from every annotated value.
    """
    for attempt in range(100):
        rng = _rng(config, split, family_plugin, index, "negative", attempt)
        if family_plugin in {"near_miss", "adjacent_value"}:
            label = config.labels[index % len(config.labels)]
            plugin = _VALUE_PLUGINS[label.plugin]
            candidate = plugin(rng, label, split, index + 31 + attempt * 89)
            if family_plugin == "near_miss" and index % 4 == 3:
                candidate = _ocr_noise(candidate, index)
        else:
            prefix = {"unrelated_shape": "SYN-ASSET"}.get(family_plugin, "SYN-REF")
            candidate = f"{prefix}-{rng.choice(_split_letters(split))}{_digits(rng, 5)}"
        if candidate not in seen:
            seen.add(candidate)
            return candidate
    raise ValueError(f"could not generate a unique negative value for {family_plugin}")


def _marker(label: str, value: str) -> str:
    return f"[[{label}:{value}]]"


def _sample_reference(config: CorpusConfig, split: str, family: str, index: int) -> str:
    """Create an opaque, class-balanced uniqueness reference for every record."""
    material = f"{config.digest}|{GENERATOR_VERSION}|{split}|{family}|{index}|reference"
    token = hashlib.sha256(material.encode("utf-8")).hexdigest()[:12].upper()
    return f"SYN-DOC-{token}"


# Every variant surfaces the persona, the organization, and the unique reference;
# the bank is rotated by record index so no single footer phrase spans the corpus.
_CONTEXT_VARIANTS = (
    " Synthetic context: subject {persona}; organization {organization}; "
    "document reference {reference}.",
    " Fictional trace for {persona} at {organization}; file token {reference}.",
    " Demo provenance: {persona} with {organization}, tracking tag {reference}.",
    " Sample metadata lists {persona}, {organization}, and marker {reference}.",
    " For testing only: {persona} / {organization} / ref {reference}.",
    " Synthetic footer naming {persona} and {organization}, keyed {reference}.",
)


def _context_suffix(
    config: CorpusConfig,
    split: str,
    family: str,
    index: int,
    persona: str,
    organization: str,
) -> str:
    """Render class-balanced metadata evidence and a unique synthetic reference."""
    variant = _CONTEXT_VARIANTS[index % len(_CONTEXT_VARIANTS)]
    return variant.format(
        persona=persona,
        organization=organization,
        reference=_sample_reference(config, split, family, index),
    )


def _positive_record(
    config: CorpusConfig,
    family: FamilyConfig,
    split: str,
    index: int,
    seen_values: set[str],
) -> Record:
    labels = config.labels
    eligible_labels = (
        tuple(label for label in labels if label.name in family.labels)
        if family.labels
        else labels
    )
    if not eligible_labels:
        raise ValueError(f"family {family.name} has no eligible labels")
    label = eligible_labels[index % len(eligible_labels)]
    cycle = index // len(eligible_labels)
    personas = split_partition(config.surfaces.personas, split)
    organizations = split_partition(config.surfaces.organizations, split)
    persona = personas[index % len(personas)]
    organization = organizations[cycle % len(organizations)]
    templates = _split_templates(family, split)
    template_index = index % len(templates)
    cue = label.cues[cycle % len(label.cues)]
    metadata: dict[str, object] = {"synthetic": True}
    plugin_label: LabelConfig | None = None
    if family.plugin == "cue_shape_conflict":
        if len(eligible_labels) < 2:
            raise ValueError(f"family {family.name} needs at least two eligible labels")
        hint_label = eligible_labels[(index + 1) % len(eligible_labels)]
        preferred_shape = hint_label.options.get("preferred_shape")
        if not isinstance(preferred_shape, str):
            raise ValueError(f"label {hint_label.name} lacks a preferred_shape")
        plugin_label = replace(
            label,
            options={**label.options, "_shape_override": preferred_shape},
        )
        metadata["contrastive"] = True
        metadata["shape_hint_label"] = hint_label.name
    value = _value(
        config,
        label,
        split,
        cycle,
        family.name,
        seen_values,
        plugin_label=plugin_label,
    )

    if family.plugin == "spoken":
        value = _spell(value)
        seen_values.add(value)
    elif family.plugin == "ocr_noise":
        value = _ocr_noise(value, cycle)
        seen_values.add(value)

    fields: dict[str, str] = {
        "cue": cue,
        "organization": organization,
        "persona": persona,
        "value": _marker(label.name, value),
    }
    cue_links: list[CueLink] = []
    if family.plugin == "mixed_entity":
        other = labels[(index + 1) % len(labels)]
        other_value = _value(config, other, split, cycle + 10000, family.name, seen_values)
        fields["other_cue"] = other.cues[cycle % len(other.cues)]
        fields["other_value"] = _marker(other.name, other_value)
        metadata["mixed_entity"] = True
        cue_links.append(CueLink(cue=fields["other_cue"], entity_type=other.name))
    if family.plugin == "cue_free":
        cue_surface: str | None = None
        metadata["cue_free"] = True
    else:
        cue_surface = cue
        cue_links.insert(0, CueLink(cue=cue, entity_type=label.name))
    if family.plugin == "cue_shape_conflict":
        metadata["shape_signature"] = shape_signature(value)

    marked = templates[template_index].format(**fields)
    marked += _context_suffix(
        config,
        split,
        family.name,
        index,
        persona,
        organization,
    )
    clean, annotations = parse_marked(marked, (label.name for label in labels))
    return Record(
        case_id=derive_case_id(
            config, GENERATOR_VERSION, split, family.name, index, clean
        ),
        split=split,
        family=family.name,
        namespace=f"piicorpus/{split}/{family.name}/{index:05d}",
        template_id=f"{split}:{family.name}:{template_index}",
        kind="positive",
        provenance="generated",
        text=clean,
        annotations=annotations,
        cue_links=tuple(cue_links),
        persona=persona,
        organization=organization,
        cue_surface=cue_surface,
        metadata=metadata,
    )


def _negative_record(
    config: CorpusConfig,
    family: FamilyConfig,
    split: str,
    index: int,
    seen_values: set[str],
) -> Record:
    plugin = get_family(family.plugin)
    templates = _split_templates(family, split)
    template_index = index % len(templates)
    personas = split_partition(config.surfaces.personas, split)
    organizations = split_partition(config.surfaces.organizations, split)
    persona = personas[index % len(personas)]
    organization = organizations[index % len(organizations)]
    label = config.labels[index % len(config.labels)]
    fields = {
        "cue": label.cues[index % len(label.cues)],
        "negative_value": _negative_value(config, split, family.plugin, index, seen_values),
        "organization": organization,
        "persona": persona,
    }
    clean = templates[template_index].format(**fields)
    clean += _context_suffix(
        config,
        split,
        family.name,
        index,
        persona,
        organization,
    )
    return Record(
        case_id=derive_case_id(
            config, GENERATOR_VERSION, split, family.name, index, clean
        ),
        split=split,
        family=family.name,
        namespace=f"piicorpus/{split}/{family.name}/{index:05d}",
        template_id=f"{split}:{family.name}:{template_index}",
        kind="hard_negative",
        provenance="generated",
        text=clean,
        persona=persona,
        organization=organization,
        cue_surface=fields["cue"] if "{cue}" in templates[template_index] else None,
        hard_negative_kind=family.name,
        metadata={"synthetic": True, "plugin_role": plugin.role},
    )


def generate_records(config: CorpusConfig) -> dict[str, list[Record]]:
    positive_families = tuple(f for f in config.families if f.role == "positive")
    negative_families = tuple(f for f in config.families if f.role == "hard_negative")
    for family in config.families:
        plugin = get_family(family.plugin)
        if plugin.role != family.role:
            raise ValueError(f"family {family.name} role disagrees with plugin {family.plugin}")

    result: dict[str, list[Record]] = {}
    seen_values: set[str] = set()
    seen_bodies: set[str] = set()
    for split in SPLIT_ORDER:
        size = config.splits[split]
        positives = int(size * config.positive_ratio)
        negatives = size - positives
        pos_counts = _allocation(positives, positive_families)
        neg_counts = _allocation(negatives, negative_families)
        rows: list[Record] = []
        for family in positive_families:
            for index in range(pos_counts[family.name]):
                row = _positive_record(config, family, split, index, seen_values)
                if row.text in seen_bodies:
                    raise ValueError(
                        f"generator produced a duplicate body in {split}/{family.name}"
                    )
                seen_bodies.add(row.text)
                rows.append(row)
        for family in negative_families:
            for index in range(neg_counts[family.name]):
                row = _negative_record(config, family, split, index, seen_values)
                if row.text in seen_bodies:
                    raise ValueError(
                        f"generator produced a duplicate body in {split}/{family.name}"
                    )
                seen_bodies.add(row.text)
                rows.append(row)
        result[split] = rows
    return result


def generate(config: CorpusConfig, output: str | Path) -> dict[str, Any]:
    records = generate_records(config)
    return write_corpus(output, config, records, generator_version=GENERATOR_VERSION)

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
from .config import CorpusConfig, FamilyConfig, LabelConfig
from .identity import derive_case_id
from .manifest import write_corpus
from .models import CueLink, Record
from .morphology import shape_signature
from .skeletons import get_family

GENERATOR_VERSION = "1.1.0"
SPLIT_ORDER = ("train", "eval", "holdout")

ValueGenerator = Callable[[random.Random, LabelConfig, str, int], str]
_VALUE_PLUGINS: dict[str, ValueGenerator] = {}


def register_value_plugin(name: str, plugin: ValueGenerator, *, replace: bool = False) -> None:
    if name in _VALUE_PLUGINS and not replace:
        raise ValueError(f"value plugin is already registered: {name}")
    _VALUE_PLUGINS[name] = plugin


def registered_value_plugins() -> tuple[str, ...]:
    return tuple(sorted(_VALUE_PLUGINS))


_ALPHABETS = {
    "train": "ABCDEFGH",
    "eval": "JKLMNPQR",
    "holdout": "STUVWXYZ",
}
_YEAR_RANGES = {
    "train": (1971, 1986),
    "eval": (1987, 2002),
    "holdout": (2003, 2018),
}


def _digits(rng: random.Random, count: int) -> str:
    while True:
        result = "".join(rng.choice("0123456789") for _ in range(count))
        if len(set(result)) > 1:
            return result


def _fictional_identifier(rng: random.Random, _label: LabelConfig, split: str, index: int) -> str:
    alphabet = _ALPHABETS[split]
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
    first, last = _YEAR_RANGES[split]
    year = rng.randint(first, last)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    return f"SYN-DATE-{year:04d}-{month:02d}-{day:02d}"


register_value_plugin("fictional_identifier", _fictional_identifier)
register_value_plugin("synthetic_date", _synthetic_date)


_PERSONAS = {
    "train": ("Ari Solen", "Bela Orin", "Cato Mire", "Dara Venn", "Eli Taren", "Fia Corren"),
    "eval": ("Galen Iver", "Hana Pell", "Ivo Saret", "Jora Wyn", "Kelan Dore", "Luma Quill"),
    "holdout": ("Mira Fen", "Nico Vale", "Ona Rell", "Pax Arden", "Rina Cove", "Soren Lark"),
}
_ORGANIZATIONS = {
    "train": ("Juniper Test Clinic", "Cobalt Demo Center", "Lattice Fictional Care"),
    "eval": ("Marigold Example Practice", "Nimbus Sample Office", "Orchard Fictional Health"),
    "holdout": ("Pebble Demo Institute", "Quarry Sample Care", "Rookery Fictional Group"),
}


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


def _negative_value(config: CorpusConfig, split: str, family: str, index: int) -> str:
    rng = _rng(config, split, family, index, "negative")
    prefix = {
        "near_miss": "SYN-TICKET",
        "unrelated_shape": "SYN-ASSET",
        "adjacent_value": "SYN-ROW",
    }.get(family, "SYN-REF")
    return f"{prefix}-{rng.choice(_ALPHABETS[split])}{_digits(rng, 5)}"


def _marker(label: str, value: str) -> str:
    return f"[[{label}:{value}]]"


def _sample_reference(config: CorpusConfig, split: str, family: str, index: int) -> str:
    """Create an opaque, class-balanced uniqueness reference for every record."""
    material = f"{config.digest}|{GENERATOR_VERSION}|{split}|{family}|{index}|reference"
    token = hashlib.sha256(material.encode("utf-8")).hexdigest()[:12].upper()
    return f"SYN-DOC-{token}"


def _context_suffix(
    config: CorpusConfig,
    split: str,
    family: str,
    index: int,
    persona: str,
    organization: str,
) -> str:
    """Render class-balanced metadata evidence and a unique synthetic reference."""
    return (
        f" Synthetic context: subject {persona}; organization {organization}; "
        f"document reference {_sample_reference(config, split, family, index)}."
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
    persona = _PERSONAS[split][cycle % len(_PERSONAS[split])]
    organization = _ORGANIZATIONS[split][cycle % len(_ORGANIZATIONS[split])]
    templates = _split_templates(family, split)
    template_index = cycle % len(templates)
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


def _negative_record(config: CorpusConfig, family: FamilyConfig, split: str, index: int) -> Record:
    plugin = get_family(family.plugin)
    templates = _split_templates(family, split)
    template_index = index % len(templates)
    persona = _PERSONAS[split][index % len(_PERSONAS[split])]
    organization = _ORGANIZATIONS[split][index % len(_ORGANIZATIONS[split])]
    label = config.labels[index % len(config.labels)]
    fields = {
        "cue": label.cues[index % len(label.cues)],
        "negative_value": _negative_value(config, split, family.plugin, index),
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
                row = _negative_record(config, family, split, index)
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

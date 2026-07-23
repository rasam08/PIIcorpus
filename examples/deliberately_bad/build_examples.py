"""Build safe, deliberately defective corpora for audit demonstrations."""

from __future__ import annotations

import argparse
import tomllib
from dataclasses import replace
from pathlib import Path

from piicorpus.annotation import parse_marked, render_marked
from piicorpus.config import CorpusConfig, config_from_dict
from piicorpus.identity import derive_case_id, namespace_index
from piicorpus.manifest import load_corpus, load_json, write_corpus
from piicorpus.models import Annotation, CueLink, Record


def case_catalog() -> dict[str, str]:
    path = Path(__file__).with_name("cases.toml")
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    return {name: value["expected_risk"] for name, value in raw["cases"].items()}


def _replace_values(record: Record, replacements: dict[int, str]) -> Record:
    marked = render_marked(record.text, record.annotations)
    for index, annotation in enumerate(record.annotations):
        if index in replacements:
            old = f"[[{annotation.entity_type}:{annotation.text}]]"
            new = f"[[{annotation.entity_type}:{replacements[index]}]]"
            marked = marked.replace(old, new, 1)
    clean, annotations = parse_marked(marked)
    return replace(record, text=clean, annotations=annotations)


def _first_positive(rows: list[Record], *, one_span: bool = False) -> int:
    return next(
        index
        for index, record in enumerate(rows)
        if record.annotations and (not one_span or len(record.annotations) == 1)
    )


def _refresh_case_ids(
    records: dict[str, list[Record]],
    config: CorpusConfig,
    generator_version: str,
) -> None:
    for split, split_rows in records.items():
        for position, record in enumerate(split_rows):
            index = namespace_index(record.namespace, record.split, record.family)
            if index is None:
                raise ValueError(f"cannot refresh malformed namespace: {record.namespace}")
            split_rows[position] = replace(
                record,
                case_id=derive_case_id(
                    config,
                    generator_version,
                    split,
                    record.family,
                    index,
                    record.text,
                ),
            )


def _rebuild_with_cues(
    record: Record,
    cues: tuple[str, ...],
    *,
    reference: str,
) -> Record:
    if len(cues) != len(record.annotations):
        raise ValueError("cue count must match annotation count")
    marked_fields = [
        f"{cue}: [[{annotation.entity_type}:{annotation.text}]]"
        for cue, annotation in zip(cues, record.annotations, strict=True)
    ]
    context = (
        f"Synthetic context: subject {record.persona}; "
        f"organization {record.organization}; document reference {reference}."
    )
    clean, annotations = parse_marked("; ".join((*marked_fields, context)))
    return replace(
        record,
        text=clean,
        annotations=annotations,
        cue_links=tuple(
            CueLink(cue=cue, entity_type=annotation.entity_type)
            for cue, annotation in zip(cues, annotations, strict=True)
        ),
        cue_surface=cues[0],
    )


def build_bad_corpus(clean_directory: Path, output: Path, case: str) -> None:
    if case not in case_catalog():
        raise ValueError(f"unknown deliberately bad case: {case}")
    if output.exists() and any(output.iterdir()):
        raise ValueError("bad-example output directory is not empty")
    config, loaded, manifest = load_corpus(clean_directory)
    records = {split: list(rows) for split, rows in loaded.items()}

    if case == "value_contamination":
        train_index = _first_positive(records["train"], one_span=True)
        source = records["train"][train_index].annotations[0]
        eval_index = next(
            index
            for index, record in enumerate(records["eval"])
            if len(record.annotations) == 1
            and record.annotations[0].entity_type == source.entity_type
        )
        records["eval"][eval_index] = _replace_values(records["eval"][eval_index], {0: source.text})
    elif case == "persona_contamination":
        source = next(record.persona for record in records["train"] if record.persona)
        index = next(i for i, record in enumerate(records["eval"]) if record.persona)
        record = records["eval"][index]
        text = record.text.replace(record.persona or "", source or "", 1)
        records["eval"][index] = replace(record, persona=source, text=text)
    elif case == "template_contamination":
        source = records["train"][0].template_id
        records["eval"][0] = replace(records["eval"][0], template_id=source)
    elif case == "skeleton_contamination":
        source = next(r for r in records["train"] if r.kind == "hard_negative")
        index = next(i for i, r in enumerate(records["eval"]) if r.kind == "hard_negative")
        target = records["eval"][index]
        text = source.text
        if source.persona and target.persona:
            text = text.replace(source.persona, target.persona)
        if source.organization and target.organization:
            text = text.replace(source.organization, target.organization)
        records["eval"][index] = replace(target, text=text)
    elif case == "duplicate_body":
        records["train"][1] = replace(records["train"][1], text=records["train"][0].text)
    elif case in {"exclusive_morphology", "morphology_dominance"}:
        target_label = config.labels[0].name
        serial = 0
        for split_rows in records.values():
            for index, record in enumerate(split_rows):
                replacements = {}
                for annotation_index, annotation in enumerate(record.annotations):
                    if annotation.entity_type == target_label:
                        if case == "exclusive_morphology":
                            replacements[annotation_index] = f"SYN-EXCLUSIVE-A{serial:07d}"
                        else:
                            replacements[annotation_index] = f"SYN-ID-NOISY-{serial:07d}"
                        serial += 1
                if replacements:
                    split_rows[index] = _replace_values(record, replacements)
    elif case == "cue_shortcut":
        unique_cues = {label.name: label.cues[-1] for label in config.labels}
        identifier_labels = [
            label.name
            for label in config.labels
            if label.plugin == "fictional_identifier"
        ]
        serial = 10000
        split_letters = {"train": "A", "eval": "J", "holdout": "S"}
        for split, split_rows in records.items():
            for index, record in enumerate(split_rows):
                if not record.annotations:
                    continue
                annotation = record.annotations[0]
                cue = unique_cues[annotation.entity_type]
                other_label = next(
                    label for label in identifier_labels if label != annotation.entity_type
                )
                other_value = f"SYN-ID-{split_letters[split]}{serial:05d}"
                serial += 1
                clean, annotations = parse_marked(
                    f"{cue}: [[{annotation.entity_type}:{annotation.text}]]. "
                    f"identity value: [[{other_label}:{other_value}]]. "
                    f"Synthetic document reference SYN-DOC-BAD-{serial:05d}."
                )
                split_rows[index] = replace(
                    record,
                    text=clean,
                    annotations=annotations,
                    cue_links=(
                        CueLink(cue=cue, entity_type=annotation.entity_type),
                        CueLink(cue="identity value", entity_type=other_label),
                    ),
                    cue_surface=cue,
                    persona=None,
                    organization=None,
                )
    elif case == "train_only_cue_shortcut":
        unique_cues = {label.name: label.cues[-1] for label in config.labels}
        for split, split_rows in records.items():
            for index, record in enumerate(split_rows):
                if not record.annotations or record.family == "cue_free_examples":
                    continue
                cues = (
                    tuple(unique_cues[annotation.entity_type] for annotation in record.annotations)
                    if split == "train"
                    else tuple("identity value" for _annotation in record.annotations)
                )
                split_rows[index] = _rebuild_with_cues(
                    record,
                    cues,
                    reference=f"SYN-DOC-BAD-CUE-{split.upper()}-{index:05d}",
                )
    elif case == "label_marker_shortcut":
        for split_rows in records.values():
            for index, record in enumerate(split_rows):
                if record.kind == "hard_negative":
                    split_rows[index] = replace(
                        record, text=record.text + f" Synthetic sample index {index:04d}."
                    )
    elif case == "unigram_label_marker":
        for split_rows in records.values():
            for index, record in enumerate(split_rows):
                if record.kind == "hard_negative":
                    split_rows[index] = replace(record, text=record.text + " Negmarker.")
    elif case == "bigram_label_marker":
        for split_rows in records.values():
            for index, record in enumerate(split_rows):
                if record.kind == "hard_negative":
                    split_rows[index] = replace(
                        record, text=record.text + " Subject organization."
                    )
    elif case == "trigram_label_marker":
        for split_rows in records.values():
            for index, record in enumerate(split_rows):
                suffix = " Subject synthetic."
                if record.kind == "hard_negative":
                    suffix += " Context subject synthetic."
                split_rows[index] = replace(record, text=record.text + suffix)
    elif case == "missing_cue_free":
        for index, record in enumerate(records["train"]):
            if record.family == "cue_free_examples":
                cue = next(
                    label.cues[-1]
                    for label in config.labels
                    if label.name == record.annotations[0].entity_type
                )
                marked = f"{cue}: {render_marked(record.text, record.annotations)}"
                clean, annotations = parse_marked(marked)
                records["train"][index] = replace(
                    record,
                    text=clean,
                    annotations=annotations,
                    cue_links=(
                        CueLink(cue=cue, entity_type=record.annotations[0].entity_type),
                    ),
                    cue_surface=cue,
                )
    elif case == "missing_contrastive":
        for split_rows in records.values():
            for index, record in enumerate(split_rows):
                if record.family == "cue_shape_conflicts":
                    metadata = {
                        key: value
                        for key, value in record.metadata.items()
                        if key not in {"contrastive", "shape_hint_label", "shape_signature"}
                    }
                    split_rows[index] = replace(record, metadata=metadata)
    elif case == "thin_templates":
        for split, split_rows in records.items():
            for index, record in enumerate(split_rows):
                split_rows[index] = replace(record, template_id=f"{split}:{record.family}:only")
    elif case == "thin_personas":
        for split, split_rows in records.items():
            replacement_name = f"Synthetic Persona {split}"
            for index, record in enumerate(split_rows):
                if record.kind == "positive" and record.persona:
                    split_rows[index] = replace(
                        record,
                        text=record.text.replace(record.persona, replacement_name),
                        persona=replacement_name,
                    )
    elif case == "weak_hard_negatives":
        for split_rows in records.values():
            for index, record in enumerate(split_rows):
                if record.kind == "hard_negative":
                    split_rows[index] = replace(record, hard_negative_kind="only_one_kind")
    elif case == "family_imbalance":
        for split, split_rows in records.items():
            positive_index = 0
            for index, record in enumerate(split_rows):
                if record.kind == "positive":
                    split_rows[index] = replace(
                        record,
                        family="narrative_prose",
                        namespace=f"piicorpus/{split}/narrative_prose/{positive_index:05d}",
                    )
                    positive_index += 1
    elif case == "low_entropy":
        for split_rows in records.values():
            for index, record in enumerate(split_rows):
                if record.annotations:
                    split_rows[index] = _replace_values(
                        record,
                        {i: "SYN-ID-A00001" for i in range(len(record.annotations))},
                    )
    elif case == "malformed_spans":
        index = _first_positive(records["train"])
        record = records["train"][index]
        first = record.annotations[0]
        broken = Annotation(
            entity_type=first.entity_type,
            start=first.start,
            end=first.end + 1,
            byte_start=first.byte_start,
            byte_end=first.byte_end,
            text=first.text,
        )
        records["train"][index] = replace(record, annotations=(broken, *record.annotations[1:]))
    elif case == "unsafe_value":
        index = _first_positive(records["train"], one_span=True)
        records["train"][index] = _replace_values(records["train"][index], {0: "UNREVIEWED-12345"})
    elif case == "generator_fingerprint":
        for split_rows in records.values():
            for index, record in enumerate(split_rows):
                split_rows[index] = replace(record, template_id="one-template-fingerprint")
    elif case == "phantom_metadata":
        index = next(
            position
            for position, record in enumerate(records["train"])
            if record.kind == "positive"
        )
        records["train"][index] = replace(
            records["train"][index], organization="Phantom Demo Organization"
        )
    elif case == "stale_case_id":
        index = _first_positive(records["train"])
        record = records["train"][index]
        records["train"][index] = replace(record, text=record.text + " Altered after ID creation.")

    config_snapshot = load_json(clean_directory / "corpus-config.json")
    refreshed_config = config_from_dict(config_snapshot)
    generator_version = str(manifest["generator_version"])
    if case != "stale_case_id":
        _refresh_case_ids(records, refreshed_config, generator_version)
    write_corpus(output, refreshed_config, records, generator_version=generator_version)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("clean_corpus", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", choices=(*tuple(case_catalog()), "all"), default="all")
    args = parser.parse_args()
    cases = case_catalog() if args.case == "all" else {args.case: case_catalog()[args.case]}
    args.out.mkdir(parents=True, exist_ok=True)
    for name in cases:
        build_bad_corpus(args.clean_corpus, args.out / name, name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

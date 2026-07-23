"""Independent, fail-closed validation derived from emitted corpus files."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..annotation import AnnotationError, validate_annotations
from ..identity import derive_case_id, namespace_index
from ..manifest import load_corpus, sha256_file
from ..models import MANIFEST_SCHEMA_VERSION, RECORD_SCHEMA_VERSION, Record
from ..morphology import body_fingerprint, normalized_template_skeleton
from ..safety import unsafe_record_reasons
from ..semantics import contrastive_evidence_errors, cue_link_errors, surfaced


@dataclass(frozen=True, slots=True)
class ValidationReport:
    valid: bool
    errors: tuple[str, ...]
    checks: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"checks": self.checks, "errors": list(self.errors), "valid": self.valid}


class CorpusIntegrityError(ValueError):
    """Raised when a consumer refuses corpus files that fail strict validation."""

    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        super().__init__(f"corpus failed strict validation ({len(report.errors)} finding(s))")


def _pairwise_collisions(values: dict[str, set[str]]) -> dict[str, int]:
    collisions: dict[str, int] = {}
    names = list(values)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            count = len(values[left] & values[right])
            if count:
                collisions[f"{left}<->{right}"] = count
    return collisions


def _derived_counts(rows: list[Record]) -> dict[str, Any]:
    return {
        "families": dict(sorted(Counter(r.family for r in rows).items())),
        "hard_negative_kinds": dict(
            sorted(Counter(r.hard_negative_kind for r in rows if r.hard_negative_kind).items())
        ),
        "labels": dict(sorted(Counter(a.entity_type for r in rows for a in r.annotations).items())),
        "negatives": sum(r.kind == "hard_negative" for r in rows),
        "organizations": len({r.organization for r in rows if r.organization}),
        "personas": len({r.persona for r in rows if r.persona}),
        "positives": sum(r.kind == "positive" for r in rows),
        "records": len(rows),
        "templates": len({r.template_id for r in rows}),
    }


def validate_corpus(directory: str | Path, *, strict: bool = False) -> ValidationReport:
    root = Path(directory)
    config, split_records, manifest = load_corpus(root)
    errors: list[str] = []
    checks: dict[str, Any] = {}

    def fail(code: str, message: str) -> None:
        errors.append(f"{code}: {message}")

    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        fail("manifest_schema", "manifest schema version is unsupported")
    if manifest.get("configuration_digest") != config.digest:
        fail("config_digest", "configuration snapshot digest does not match the manifest")
    if manifest.get("generated_data_license") != config.generated_data_license:
        fail("data_license", "generated-data license differs from the configuration")
    generator_version = manifest.get("generator_version")
    if not isinstance(generator_version, str) or not generator_version:
        fail("generator_version", "manifest generator version is missing")

    file_results: dict[str, str] = {}
    declared_files = manifest.get("files", {})
    expected_paths = ["corpus-config.json", *(f"splits/{s}.jsonl" for s in split_records)]
    for relative in expected_paths:
        path = root / relative
        if not path.is_file():
            fail("missing_file", f"required file is missing: {relative}")
            continue
        declared = declared_files.get(relative, {})
        actual_hash = sha256_file(path)
        if declared.get("sha256") != actual_hash:
            fail("file_hash", f"SHA-256 mismatch for {relative}")
        if declared.get("bytes") != path.stat().st_size:
            fail("file_size", f"byte count mismatch for {relative}")
        raw = path.read_bytes()
        if b"\r\n" in raw or (relative.endswith(".jsonl") and raw and not raw.endswith(b"\n")):
            fail("stable_encoding", f"{relative} must use LF newlines and end with LF")
        file_results[relative] = actual_hash
    checks["file_sha256"] = file_results

    configured_labels = {label.name for label in config.labels}
    label_plugins = {label.name: label.plugin for label in config.labels}
    family_by_name = {family.name: family for family in config.families}
    ids: set[str] = set()
    namespaces: set[str] = set()
    bodies: set[str] = set()
    dimensions: dict[str, dict[str, set[str]]] = {
        name: defaultdict(set)
        for name in ("values", "personas", "organizations", "templates", "skeletons", "namespaces")
    }
    template_by_cell: dict[tuple[str, str], set[str]] = defaultdict(set)
    persona_by_cell: dict[tuple[str, str], set[str]] = defaultdict(set)

    for split, rows in split_records.items():
        if len(rows) != config.splits[split]:
            fail("split_size", f"{split} has {len(rows)} records, expected {config.splits[split]}")
        derived = _derived_counts(rows)
        if manifest.get("counts", {}).get(split) != derived:
            fail("manifest_counts", f"manifest counts for {split} do not match emitted records")
        positive_count = derived["positives"]
        negative_count = derived["negatives"]
        if rows and negative_count / len(rows) < config.minimum_hard_negative_ratio:
            fail("hard_negative_ratio", f"{split} has too few hard negatives")
        expected_positive = int(config.splits[split] * config.positive_ratio)
        if positive_count != expected_positive:
            fail("positive_ratio", f"{split} positive count differs from the configured allocation")
        if derived["organizations"] < config.diversity.minimum_organizations_per_split:
            fail("organization_diversity", f"{split} has insufficient organization diversity")

        for record in rows:
            if record.schema_version != RECORD_SCHEMA_VERSION:
                fail("record_schema", f"{record.case_id} has an unsupported record schema")
            if record.case_id in ids:
                fail("duplicate_case_id", "case IDs are not unique")
            ids.add(record.case_id)
            if record.namespace in namespaces:
                fail("duplicate_namespace", "family/index namespaces are not unique")
            namespaces.add(record.namespace)
            index = namespace_index(record.namespace, record.split, record.family)
            if record.split != split or index is None:
                fail("split_namespace", f"{record.case_id} has an invalid split namespace")
            elif isinstance(generator_version, str):
                expected_case_id = derive_case_id(
                    config,
                    generator_version,
                    record.split,
                    record.family,
                    index,
                    record.text,
                )
                if record.case_id != expected_case_id:
                    fail(
                        "case_id",
                        f"{record.case_id} does not match its content-derived identity",
                    )
            if record.family not in family_by_name:
                fail("family", f"{record.case_id} uses an unconfigured family")
                continue
            family = family_by_name[record.family]
            if record.kind != family.role:
                fail("family_role", f"{record.case_id} kind disagrees with its configured family")
            if record.kind == "positive" and not record.annotations:
                fail("positive_annotations", f"{record.case_id} has no positive annotation")
            if record.kind == "hard_negative" and record.annotations:
                fail(
                    "negative_annotations", f"{record.case_id} hard negative carries an annotation"
                )
            if record.kind == "hard_negative" and not record.hard_negative_kind:
                fail("negative_kind", f"{record.case_id} lacks a hard-negative kind")
            semantic_errors = (
                contrastive_evidence_errors(record, config, family)
                if family.plugin == "cue_shape_conflict"
                else cue_link_errors(record, config, family)
            )
            for semantic_error in semantic_errors:
                fail("semantic_evidence", f"{record.case_id}: {semantic_error}")
            if record.persona and not surfaced(record.persona, record.text):
                fail("persona_surface", f"{record.case_id} persona metadata is not rendered")
            if record.organization and not surfaced(record.organization, record.text):
                fail(
                    "organization_surface",
                    f"{record.case_id} organization metadata is not rendered",
                )
            for annotation in record.annotations:
                if annotation.entity_type not in configured_labels:
                    fail("entity_label", f"{record.case_id} uses an unconfigured entity label")
            try:
                validate_annotations(record.text, record.annotations)
            except AnnotationError as exc:
                fail("span", f"{record.case_id} has malformed annotation offsets: {exc}")
            safety_reasons = unsafe_record_reasons(
                record, config.safety, label_plugins=label_plugins
            )
            if safety_reasons:
                fail(
                    "safety",
                    f"{record.case_id} failed safety validation: {', '.join(safety_reasons)}",
                )
            fingerprint = body_fingerprint(record.text)
            if fingerprint in bodies:
                fail("duplicate_body", "duplicate record bodies are not allowed")
            bodies.add(fingerprint)

            dimensions["values"][split].update(a.text for a in record.annotations)
            if record.persona:
                dimensions["personas"][split].add(record.persona)
            if record.organization:
                dimensions["organizations"][split].add(record.organization)
            dimensions["templates"][split].add(record.template_id)
            dimensions["namespaces"][split].add(record.namespace)
            dimensions["skeletons"][split].add(
                normalized_template_skeleton(
                    record.text,
                    record.annotations,
                    persona=record.persona,
                    organization=record.organization,
                )
            )
            template_by_cell[(split, record.family)].add(record.template_id)
            if record.persona:
                persona_by_cell[(split, record.family)].add(record.persona)

    for dimension, by_split in dimensions.items():
        collisions = _pairwise_collisions(dict(by_split))
        checks[f"cross_split_{dimension}"] = collisions
        if collisions:
            fail(f"cross_split_{dimension}", f"cross-split contamination: {collisions}")

    for split in split_records:
        for family in config.families:
            key = (split, family.name)
            if len(template_by_cell[key]) < config.diversity.minimum_templates_per_family:
                fail("template_diversity", f"{split}/{family.name} has insufficient templates")
            if family.role == "positive" and (
                len(persona_by_cell[key]) < config.diversity.minimum_personas_per_family
            ):
                fail("persona_diversity", f"{split}/{family.name} has insufficient personas")

    if strict and manifest.get("synthetic_holdout_limitation") is None:
        fail("claim_boundary", "manifest omits the same-generator holdout limitation")
    checks["records"] = {split: len(rows) for split, rows in split_records.items()}
    checks["strict"] = strict
    return ValidationReport(valid=not errors, errors=tuple(errors), checks=checks)

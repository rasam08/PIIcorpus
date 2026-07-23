"""Semantic evidence checks shared by validation and failure-mode auditing."""

from __future__ import annotations

from .config import CorpusConfig, FamilyConfig
from .models import Record
from .morphology import shape_signature


def surfaced(value: str | None, text: str) -> bool:
    """Return whether optional metadata is backed by a literal emitted surface."""
    return bool(value and value.casefold() in text.casefold())


def configured_cues(config: CorpusConfig) -> set[str]:
    return {cue.casefold() for label in config.labels for cue in label.cues}


def cue_link_errors(
    record: Record,
    config: CorpusConfig,
    family: FamilyConfig,
) -> list[str]:
    """Validate explicit cue-to-entity relationships against the emitted record."""
    errors: list[str] = []
    configured_labels = {label.name for label in config.labels}
    annotation_labels = {annotation.entity_type for annotation in record.annotations}
    seen: set[tuple[str, str]] = set()

    if record.kind == "hard_negative" and record.cue_links:
        errors.append("hard negatives cannot carry cue-to-entity links")
        return errors

    for link in record.cue_links:
        normalized = link.cue.casefold()
        key = (normalized, link.entity_type)
        if not link.cue.strip():
            errors.append("cue links cannot use an empty cue")
        if key in seen:
            errors.append("duplicate cue-to-entity link")
        seen.add(key)
        if normalized not in record.text.casefold():
            errors.append(f"cue surface is absent from text: {link.cue}")
        if link.entity_type not in configured_labels:
            errors.append(f"cue link uses an unconfigured label: {link.entity_type}")
        if link.entity_type not in annotation_labels:
            errors.append(f"cue link has no matching annotation: {link.entity_type}")

    if record.kind != "positive":
        return errors

    if family.plugin == "cue_free":
        if record.cue_links:
            errors.append("cue-free records cannot carry cue-to-entity links")
        if any(cue in record.text.casefold() for cue in configured_cues(config)):
            errors.append("cue-free record contains a configured cue surface")
        return errors

    if not record.cue_links:
        errors.append("positive record lacks explicit cue-to-entity links")
    linked_labels = {link.entity_type for link in record.cue_links}
    missing = annotation_labels - linked_labels
    if missing:
        errors.append(f"annotations lack linked cues: {sorted(missing)}")
    present_configured = {
        cue for cue in configured_cues(config) if cue in record.text.casefold()
    }
    represented = {link.cue.casefold() for link in record.cue_links}
    unlinked = present_configured - represented
    if unlinked:
        errors.append(f"configured cue surfaces lack links: {sorted(unlinked)}")
    if record.cue_surface and record.cue_surface.casefold() not in represented:
        errors.append("cue_surface is not represented by a cue link")
    return errors


def is_cue_free_evidence(
    record: Record,
    config: CorpusConfig,
    family: FamilyConfig,
) -> bool:
    return (
        record.kind == "positive"
        and family.plugin == "cue_free"
        and bool(record.annotations)
        and not cue_link_errors(record, config, family)
    )


def contrastive_evidence_errors(
    record: Record,
    config: CorpusConfig,
    family: FamilyConfig,
) -> list[str]:
    """Verify cue/shape disagreement from emitted spans and configured shape profiles."""
    if family.plugin != "cue_shape_conflict":
        return []
    errors = cue_link_errors(record, config, family)
    if record.kind != "positive" or len(record.annotations) != 1:
        errors.append("contrastive records require exactly one positive annotation")
        return errors
    if record.metadata.get("contrastive") is not True:
        errors.append("contrastive metadata flag is missing")
    hint_name = record.metadata.get("shape_hint_label")
    labels = {label.name: label for label in config.labels}
    if not isinstance(hint_name, str) or hint_name not in labels:
        errors.append("shape_hint_label is missing or unconfigured")
        return errors
    target = record.annotations[0].entity_type
    if hint_name == target:
        errors.append("shape hint does not disagree with the annotated label")
    actual_shape = shape_signature(record.annotations[0].text)
    configured_shape = labels[hint_name].options.get("preferred_shape")
    if configured_shape != actual_shape:
        errors.append("emitted value does not match the configured shape hint")
    if record.metadata.get("shape_signature") != actual_shape:
        errors.append("recorded shape signature does not match the emitted value")
    return errors


def is_contrastive_evidence(
    record: Record,
    config: CorpusConfig,
    family: FamilyConfig,
) -> bool:
    return family.plugin == "cue_shape_conflict" and not contrastive_evidence_errors(
        record, config, family
    )

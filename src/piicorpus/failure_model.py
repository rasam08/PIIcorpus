"""First-class corpus failure-mode audit independent of any trained detector."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .annotation import AnnotationError, validate_annotations
from .manifest import SYNTHETIC_HOLDOUT_LIMITATION, load_corpus
from .models import Finding, Record, stable_json
from .morphology import body_fingerprint, normalized_template_skeleton, shape_signature
from .safety import unsafe_record_reasons
from .semantics import is_contrastive_evidence, is_cue_free_evidence, surfaced
from .validators import CorpusIntegrityError, validate_corpus


@dataclass(frozen=True, slots=True)
class AuditReport:
    findings: tuple[Finding, ...]
    summary: dict[str, int]
    limitation: str = SYNTHETIC_HOLDOUT_LIMITATION

    @property
    def failed(self) -> bool:
        return any(finding.status == "FAIL" for finding in self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "findings": [finding.to_dict() for finding in self.findings],
            "limitation": self.limitation,
            "summary": self.summary,
        }

    def render(self, format_name: str = "text") -> str:
        if format_name == "json":
            return stable_json(self.to_dict(), pretty=True)
        if format_name == "markdown":
            lines = [
                "# PIIcorpus audit",
                "",
                "| Risk | Status | Count | Reason |",
                "|---|---|---:|---|",
            ]
            for finding in self.findings:
                count = "—" if finding.count is None else str(finding.count)
                lines.append(f"| {finding.risk} | {finding.status} | {count} | {finding.reason} |")
            lines.extend(("", f"> {self.limitation}", ""))
            return "\n".join(lines)
        lines = ["PIIcorpus audit"]
        for finding in self.findings:
            count = "-" if finding.count is None else str(finding.count)
            lines.append(f"{finding.status:10s} {finding.risk:38s} count={count}  {finding.reason}")
        lines.append("")
        lines.append(self.limitation)
        return "\n".join(lines) + "\n"


def _collisions(values: dict[str, set[str]]) -> int:
    names = list(values)
    return sum(
        len(values[left] & values[right])
        for index, left in enumerate(names)
        for right in names[index + 1 :]
    )


def _finding(risk: str, failed: bool, count: int, good: str, bad: str, **details: Any) -> Finding:
    return Finding(
        risk=risk,
        status="FAIL" if failed else "PASS",
        count=count,
        reason=bad if failed else good,
        details=details,
    )


def _entropy(values: Iterable[str]) -> float:
    counter = Counter(values)
    total = sum(counter.values())
    if not total:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counter.values())


def _suspicious_kind_markers(
    rows: list[Record],
    *,
    minimum_support: int,
    maximum_kind_share: float,
    minimum_kind_coverage: float,
) -> list[dict[str, Any]]:
    """Find repeated lexical 1-3-grams that make record kind nearly deterministic."""
    feature_kinds: dict[str, Counter[str]] = defaultdict(Counter)
    feature_widths: dict[str, int] = {}
    kind_totals = Counter(record.kind for record in rows)
    for record in rows:
        skeleton = normalized_template_skeleton(
            record.text,
            record.annotations,
            persona=record.persona if surfaced(record.persona, record.text) else None,
            organization=(
                record.organization if surfaced(record.organization, record.text) else None
            ),
        )
        tokens = re.findall(r"[a-z]+|<[^>]+>", skeleton)
        features: set[str] = set()
        for width in (1, 2, 3):
            for index in range(max(0, len(tokens) - width + 1)):
                window = tokens[index : index + width]
                if any(token.startswith("<") for token in window):
                    continue
                feature = " ".join(window)
                features.add(feature)
                feature_widths[feature] = width
        for feature in features:
            feature_kinds[feature][record.kind] += 1

    suspicious: list[dict[str, Any]] = []
    for feature, counts in feature_kinds.items():
        support = sum(counts.values())
        if support < minimum_support:
            continue
        kind, kind_count = counts.most_common(1)[0]
        share = kind_count / support
        coverage = kind_count / kind_totals[kind]
        if share > maximum_kind_share and coverage >= minimum_kind_coverage:
            suspicious.append(
                {
                    "coverage": round(coverage, 4),
                    "feature": feature,
                    "kind": kind,
                    "share": round(share, 4),
                    "support": support,
                    "token_count": feature_widths[feature],
                }
            )
    return sorted(
        suspicious,
        key=lambda value: (
            value["token_count"],
            -value["support"],
            value["feature"],
        ),
    )


def audit_corpus(
    directory: str | Path,
    *,
    allow_invalid: bool = False,
) -> AuditReport:
    validation = validate_corpus(directory, strict=True)
    if not validation.valid and not allow_invalid:
        raise CorpusIntegrityError(validation)
    config, split_records, _manifest = load_corpus(directory)
    rows = [record for split in ("train", "eval", "holdout") for record in split_records[split]]
    findings: list[Finding] = [
        _finding(
            "corpus_integrity",
            not validation.valid,
            len(validation.errors),
            "strict validation passed before audit",
            "strict validation failed; forensic results are non-authoritative",
            errors=list(validation.errors),
        )
    ]

    dimensions: dict[str, dict[str, set[str]]] = {
        "values": defaultdict(set),
        "personas": defaultdict(set),
        "templates": defaultdict(set),
        "skeletons": defaultdict(set),
    }
    for record in rows:
        dimensions["values"][record.split].update(a.text for a in record.annotations)
        if record.persona and surfaced(record.persona, record.text):
            dimensions["personas"][record.split].add(record.persona)
        dimensions["templates"][record.split].add(record.template_id)
        dimensions["skeletons"][record.split].add(
            normalized_template_skeleton(
                record.text,
                record.annotations,
                persona=record.persona,
                organization=record.organization,
            )
        )
    for risk, dimension in (
        ("cross_split_value_contamination", "values"),
        ("persona_contamination", "personas"),
        ("template_contamination", "templates"),
        ("skeleton_contamination", "skeletons"),
    ):
        count = _collisions(dict(dimensions[dimension]))
        findings.append(
            _finding(
                risk,
                bool(count),
                count,
                "no cross-split collisions",
                "cross-split collisions detected",
            )
        )

    fingerprints = [body_fingerprint(record.text) for record in rows]
    exact_duplicates = sum(count - 1 for count in Counter(fingerprints).values() if count > 1)
    near_pairs = 0
    candidate_buckets: dict[tuple[int, str], list[str]] = defaultdict(list)
    for fingerprint in fingerprints:
        leading_words = " ".join(fingerprint.split()[:2])
        candidate_buckets[(len(fingerprint) // 8, leading_words)].append(fingerprint)
    for candidates in candidate_buckets.values():
        for left in range(len(candidates)):
            for right in range(left + 1, len(candidates)):
                if candidates[left] == candidates[right]:
                    continue
                if SequenceMatcher(None, candidates[left], candidates[right]).ratio() >= 0.995:
                    near_pairs += 1
    duplicate_count = exact_duplicates + near_pairs
    findings.append(
        _finding(
            "duplicate_or_near_duplicate_bodies",
            bool(duplicate_count),
            duplicate_count,
            "no exact or very-near duplicate bodies",
            "duplicate or very-near duplicate bodies detected",
            exact=exact_duplicates,
            near=near_pairs,
        )
    )

    label_config = {label.name: label for label in config.labels}
    group_labels: dict[str, set[str]] = defaultdict(set)
    for label in config.labels:
        group_labels[label.morphology_group].add(label.name)
    shape_labels: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for record in rows:
        for annotation in record.annotations:
            configured_label = label_config.get(annotation.entity_type)
            if configured_label:
                shape_labels[(configured_label.morphology_group, shape_signature(annotation.text))][
                    annotation.entity_type
                ] += 1
    exclusive = []
    dominant = []
    for (group, shape), counts in shape_labels.items():
        if len(group_labels[group]) < 2:
            continue
        if len(counts) == 1:
            exclusive.append(f"{group}:{shape}")
        total = sum(counts.values())
        share = max(counts.values()) / total
        if share > config.audit.max_morphology_label_share:
            dominant.append((group, shape, round(share, 4)))
    findings.append(
        _finding(
            "label_exclusive_morphology",
            bool(exclusive),
            len(exclusive),
            "no morphology in a multi-label group is label-exclusive",
            "label-exclusive morphology detected",
            shapes=exclusive,
        )
    )
    findings.append(
        _finding(
            "morphology_label_dominance",
            bool(dominant),
            len(dominant),
            "P(label | shape) remains below the configured ceiling",
            "one or more shapes overly predict a label",
            shapes=dominant,
        )
    )

    cue_stats_by_split: dict[str, dict[str, Any]] = {}
    cue_fractions_by_split: dict[str, float] = {}
    for split, split_rows in split_records.items():
        cue_to_labels: dict[str, set[str]] = defaultdict(set)
        cue_records: list[set[str]] = []
        for record in split_rows:
            if not record.annotations:
                continue
            annotation_labels = {annotation.entity_type for annotation in record.annotations}
            valid_links = {
                (link.cue.casefold(), link.entity_type)
                for link in record.cue_links
                if link.cue.casefold() in record.text.casefold()
                and link.entity_type in annotation_labels
            }
            present = {cue for cue, _label in valid_links}
            cue_records.append(present)
            for cue, entity_type in valid_links:
                cue_to_labels[cue].add(entity_type)
        exclusive_cues = {cue for cue, labels in cue_to_labels.items() if len(labels) == 1}
        records_with_cues = [cues for cues in cue_records if cues]
        exclusive_records = sum(
            bool(cues & exclusive_cues) for cues in records_with_cues
        )
        fraction = (
            exclusive_records / len(records_with_cues) if records_with_cues else 1.0
        )
        cue_fractions_by_split[split] = fraction
        cue_stats_by_split[split] = {
            "exclusive_cues": sorted(exclusive_cues),
            "exclusive_records": exclusive_records,
            "fraction": round(fraction, 4),
            "records_with_cues": len(records_with_cues),
        }
    failed_cue_splits = [
        split
        for split, fraction in cue_fractions_by_split.items()
        if fraction > config.audit.max_label_exclusive_cue_fraction
    ]
    maximum_cue_fraction = max(cue_fractions_by_split.values(), default=1.0)
    exclusive_record_count = sum(
        stats["exclusive_records"] for stats in cue_stats_by_split.values()
    )
    findings.append(
        _finding(
            "cue_label_shortcuts",
            bool(failed_cue_splits),
            exclusive_record_count,
            "label-exclusive cue use stays below the configured fraction in every split",
            "label-exclusive cue surfaces exceed the ceiling in one or more splits",
            failed_splits=failed_cue_splits,
            fraction=round(maximum_cue_fraction, 4),
            per_split=cue_stats_by_split,
        )
    )

    family_by_name = {family.name: family for family in config.families}
    cue_free_by_split = {
        split: sum(
            is_cue_free_evidence(record, config, family_by_name[record.family])
            for record in split_rows
            if record.family in family_by_name
        )
        for split, split_rows in split_records.items()
    }
    missing_cue_free_splits = [
        split for split, count in cue_free_by_split.items() if count == 0
    ]
    findings.append(
        _finding(
            "cue_free_coverage",
            bool(missing_cue_free_splits),
            sum(cue_free_by_split.values()),
            "cue-free positive evidence is present in every split",
            "one or more splits lack cue-free positive evidence",
            per_split=cue_free_by_split,
            missing_splits=missing_cue_free_splits,
        )
    )
    contrastive_by_split = {
        split: sum(
            is_contrastive_evidence(record, config, family_by_name[record.family])
            for record in split_rows
            if record.family in family_by_name
        )
        for split, split_rows in split_records.items()
    }
    missing_contrastive_splits = [
        split for split, count in contrastive_by_split.items() if count == 0
    ]
    findings.append(
        _finding(
            "cue_shape_contrastive_coverage",
            bool(missing_contrastive_splits),
            sum(contrastive_by_split.values()),
            "verified cue/shape disagreement evidence is present in every split",
            "one or more splits lack verified cue/shape disagreement evidence",
            per_split=contrastive_by_split,
            missing_splits=missing_contrastive_splits,
        )
    )

    templates_by_cell: dict[tuple[str, str], set[str]] = defaultdict(set)
    personas_by_cell: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in rows:
        templates_by_cell[(record.split, record.family)].add(record.template_id)
        if record.persona and surfaced(record.persona, record.text):
            personas_by_cell[(record.split, record.family)].add(record.persona)
    thin_templates = [
        f"{split}/{family.name}"
        for split in split_records
        for family in config.families
        if len(templates_by_cell[(split, family.name)])
        < config.diversity.minimum_templates_per_family
    ]
    thin_personas = [
        f"{split}/{family.name}"
        for split in split_records
        for family in config.families
        if family.role == "positive"
        and len(personas_by_cell[(split, family.name)])
        < config.diversity.minimum_personas_per_family
    ]
    findings.append(
        _finding(
            "template_diversity",
            bool(thin_templates),
            len(thin_templates),
            "template diversity minimums are met",
            "one or more family/split cells lack template depth",
            cells=thin_templates,
        )
    )
    findings.append(
        _finding(
            "persona_diversity",
            bool(thin_personas),
            len(thin_personas),
            "persona diversity minimums are met",
            "one or more positive family/split cells lack persona depth",
            cells=thin_personas,
        )
    )

    negatives = [record for record in rows if record.kind == "hard_negative"]
    hard_kinds = {record.hard_negative_kind for record in negatives if record.hard_negative_kind}
    ratio_failures = [
        split
        for split, split_rows in split_records.items()
        if sum(r.kind == "hard_negative" for r in split_rows) / len(split_rows)
        < config.minimum_hard_negative_ratio
    ]
    weak_hard_negative = (
        bool(ratio_failures) or len(hard_kinds) < config.audit.minimum_hard_negative_kinds
    )
    findings.append(
        _finding(
            "hard_negative_coverage",
            weak_hard_negative,
            len(negatives),
            "hard-negative ratio and kind coverage are sufficient",
            "hard-negative ratio or kind coverage is insufficient",
            kinds=sorted(v for v in hard_kinds if v),
            ratio_failures=ratio_failures,
        )
    )

    family_counts = Counter(record.family for record in rows)
    max_family_share = max(family_counts.values(), default=0) / len(rows) if rows else 1.0
    findings.append(
        _finding(
            "family_imbalance",
            max_family_share > config.audit.max_family_share,
            max(family_counts.values(), default=0),
            "no family exceeds the configured corpus share",
            "one family exceeds the configured corpus share",
            share=round(max_family_share, 4),
        )
    )

    entity_values = [annotation.text for record in rows for annotation in record.annotations]
    entropy = _entropy(entity_values)
    findings.append(
        _finding(
            "low_value_entropy",
            entropy < config.audit.minimum_value_entropy_bits,
            len(set(entity_values)),
            "value entropy meets the configured floor",
            "value entropy is below the configured floor",
            entropy_bits=round(entropy, 4),
        )
    )

    malformed = 0
    unsafe = 0
    for record in rows:
        try:
            validate_annotations(record.text, record.annotations)
        except AnnotationError:
            malformed += 1
        if unsafe_record_reasons(record, config.safety):
            unsafe += 1
    findings.append(
        _finding(
            "malformed_or_unbalanced_spans",
            bool(malformed),
            malformed,
            "all spans round-trip exactly",
            "malformed, overlapping or unbalanced spans detected",
        )
    )
    findings.append(
        _finding(
            "unsafe_generated_value_patterns",
            bool(unsafe),
            unsafe,
            "no configured unsafe pattern was found",
            "unsafe generated-value patterns detected",
        )
    )

    template_counts = Counter(record.template_id for record in rows)
    template_share = max(template_counts.values(), default=0) / len(rows) if rows else 1.0
    marker_features = _suspicious_kind_markers(
        rows,
        minimum_support=config.audit.minimum_marker_support,
        maximum_kind_share=config.audit.max_kind_marker_share,
        minimum_kind_coverage=config.audit.minimum_marker_kind_coverage,
    )
    fingerprint_failed = (
        template_share > config.audit.max_template_share or bool(marker_features)
    )
    findings.append(
        _finding(
            "generator_fingerprint",
            fingerprint_failed,
            len(marker_features)
            if marker_features
            else max(template_counts.values(), default=0),
            "template concentration and kind-predictive markers stay below configured ceilings",
            "template concentration or a kind-predictive marker creates a shortcut",
            share=round(template_share, 4),
            marker_features=marker_features,
        )
    )
    findings.append(
        Finding(
            risk="same_generator_holdout_dependence",
            status="UNMEASURED",
            count=None,
            reason=SYNTHETIC_HOLDOUT_LIMITATION,
            details={"independent_generalization": "not measured"},
        )
    )

    summary = Counter(finding.status for finding in findings)
    return AuditReport(findings=tuple(findings), summary=dict(sorted(summary.items())))

"""First-class corpus failure-mode audit independent of any trained detector."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .annotation import AnnotationError, validate_annotations
from .config import AuditConfig, CorpusConfig
from .manifest import SYNTHETIC_HOLDOUT_LIMITATION, load_corpus
from .models import Finding, Record, stable_json
from .morphology import body_fingerprint, normalized_template_skeleton, shape_signature
from .probe import probe_findings, unmeasured_probe_findings
from .profiles import REFERENCE_THRESHOLDS, WEAKER_WHEN_HIGHER, WEAKER_WHEN_LOWER
from .safety import unsafe_record_reasons, unsafe_text_reasons
from .semantics import is_contrastive_evidence, is_cue_free_evidence, surfaced
from .similarity import near_duplicate_pairs
from .validators import CorpusIntegrityError, ValidationReport, validate_corpus

# Identifier-shaped surface forms: uppercase codes, emails, NANP phone formats,
# and dashed/dotted digit groups (SSN, card, IP, date shapes).
_IDENTIFIER_TOKEN_RE = re.compile(
    r"\b[A-Z]{2,}[A-Z0-9-]*\d[A-Z0-9-]*\b"
    r"|\b[A-Za-z0-9][\w.+-]*@[\w.-]+\b"
    r"|\(\d{3}\) \d{3}-\d{4}"
    r"|\b\d{1,4}(?:[-.]\d{1,4}){2,}\b"
)
_SKELETON_TOKEN_RE = re.compile(r"[a-z]+|<[^>]+>|#")
_PERVASIVE_NGRAM_WIDTH = 4
_MAX_REPORTED_OFFENDERS = 20


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
                "| Risk | Status | Count | Measured | Threshold | Reason |",
                "|---|---|---:|---:|---:|---|",
            ]
            for finding in self.findings:
                count = "—" if finding.count is None else str(finding.count)
                measured = "—" if finding.measured is None else str(finding.measured)
                threshold = "—" if finding.threshold is None else str(finding.threshold)
                lines.append(
                    f"| {finding.risk} | {finding.status} | {count} | {measured} "
                    f"| {threshold} | {finding.reason} |"
                )
            lines.extend(("", f"> {self.limitation}", ""))
            return "\n".join(lines)
        lines = ["PIIcorpus audit"]
        for finding in self.findings:
            count = "-" if finding.count is None else str(finding.count)
            line = f"{finding.status:10s} {finding.risk:38s} count={count}"
            if finding.threshold is not None:
                line += f"  measured={finding.measured} threshold={finding.threshold}"
            line += f"  {finding.reason}"
            lines.append(line)
        lines.append("")
        lines.append(self.limitation)
        return "\n".join(lines) + "\n"


@dataclass(frozen=True, slots=True)
class AuditContext:
    """Inputs shared by every audit check."""

    split_records: dict[str, list[Record]]
    rows: tuple[Record, ...]
    thresholds: AuditConfig
    threshold_source: str = "config"
    config: CorpusConfig | None = None
    probe_enabled: bool = False


CheckFunction = Callable[[AuditContext], list[Finding]]


@dataclass(frozen=True, slots=True)
class CheckSpec:
    """One registered audit check and the risks it reports on."""

    risks: tuple[str, ...]
    run: CheckFunction
    requires_config: bool = False


def _require_config(ctx: AuditContext) -> CorpusConfig:
    if ctx.config is None:
        raise ValueError("this audit check requires the generating corpus configuration")
    return ctx.config


def _collisions(values: dict[str, set[str]]) -> int:
    names = list(values)
    return sum(
        len(values[left] & values[right])
        for index, left in enumerate(names)
        for right in names[index + 1 :]
    )


def _finding(
    risk: str,
    failed: bool,
    count: int,
    good: str,
    bad: str,
    *,
    measured: float | int | None = None,
    threshold: float | int | None = None,
    source: str | None = None,
    **details: Any,
) -> Finding:
    return Finding(
        risk=risk,
        status="FAIL" if failed else "PASS",
        count=count,
        reason=bad if failed else good,
        details=details,
        measured=measured,
        threshold=threshold,
        threshold_source=source if threshold is not None else None,
    )


def _entropy(values: Iterable[str]) -> float:
    counter = Counter(values)
    total = sum(counter.values())
    if not total:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counter.values())


def _record_skeleton(record: Record) -> str:
    return normalized_template_skeleton(
        record.text,
        record.annotations,
        persona=record.persona if surfaced(record.persona, record.text) else None,
        organization=(
            record.organization if surfaced(record.organization, record.text) else None
        ),
    )


def _suspicious_kind_markers(
    rows: Iterable[Record],
    *,
    minimum_support: int,
    maximum_kind_share: float,
    minimum_kind_coverage: float,
) -> list[dict[str, Any]]:
    """Find repeated lexical 1-3-grams that make record kind nearly deterministic."""
    feature_kinds: dict[str, Counter[str]] = defaultdict(Counter)
    feature_widths: dict[str, int] = {}
    rows = list(rows)
    kind_totals = Counter(record.kind for record in rows)
    for record in rows:
        tokens = re.findall(r"[a-z]+|<[^>]+>", _record_skeleton(record))
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


def _check_cross_split_contamination(ctx: AuditContext) -> list[Finding]:
    dimensions: dict[str, dict[str, set[str]]] = {
        "values": defaultdict(set),
        "personas": defaultdict(set),
        "templates": defaultdict(set),
        "skeletons": defaultdict(set),
    }
    for record in ctx.rows:
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
    findings = []
    for risk, dimension in (
        ("cross_split_value_contamination", "values"),
        ("persona_contamination", "personas"),
        ("template_contamination", "templates"),
        ("skeleton_contamination", "skeletons"),
    ):
        if dimension == "templates" and ctx.config is None:
            findings.append(
                Finding(
                    risk=risk,
                    status="UNMEASURED",
                    count=None,
                    reason="template identity is generator metadata external data lacks",
                )
            )
            continue
        count = _collisions(dict(dimensions[dimension]))
        findings.append(
            _finding(
                risk,
                bool(count),
                count,
                "no cross-split collisions",
                "cross-split collisions detected",
                measured=count,
                threshold=0,
                source=ctx.threshold_source,
            )
        )
    return findings


def _check_duplicate_bodies(ctx: AuditContext) -> list[Finding]:
    fingerprints = [body_fingerprint(record.text) for record in ctx.rows]
    exact_duplicates = sum(count - 1 for count in Counter(fingerprints).values() if count > 1)
    pairs = near_duplicate_pairs(
        fingerprints, threshold=ctx.thresholds.near_duplicate_jaccard
    )
    near_pairs = sum(
        1 for left, right, _score in pairs if fingerprints[left] != fingerprints[right]
    )
    duplicate_count = exact_duplicates + near_pairs
    return [
        _finding(
            "duplicate_or_near_duplicate_bodies",
            bool(duplicate_count),
            duplicate_count,
            "no exact or very-near duplicate bodies",
            "duplicate or very-near duplicate bodies detected",
            measured=duplicate_count,
            threshold=0,
            source=ctx.threshold_source,
            exact=exact_duplicates,
            near=near_pairs,
            jaccard_threshold=ctx.thresholds.near_duplicate_jaccard,
        )
    ]


def _check_intra_split_redundancy(ctx: AuditContext) -> list[Finding]:
    per_split: dict[str, dict[str, Any]] = {}
    worst = 0.0
    for split, split_rows in ctx.split_records.items():
        if not split_rows:
            continue
        fingerprints = [body_fingerprint(record.text) for record in split_rows]
        pairs = near_duplicate_pairs(
            fingerprints, threshold=ctx.thresholds.intra_split_similarity_threshold
        )
        involved = {index for left, right, _score in pairs for index in (left, right)}
        fraction = len(involved) / len(split_rows)
        skeletons = {_record_skeleton(record) for record in split_rows}
        per_split[split] = {
            "fraction": round(fraction, 4),
            "near_duplicate_records": len(involved),
            "pairs": len(pairs),
            "skeleton_ratio": round(len(skeletons) / len(split_rows), 4),
        }
        worst = max(worst, fraction)
    failed = worst > ctx.thresholds.max_intra_split_near_dup_fraction
    return [
        _finding(
            "intra_split_redundancy",
            failed,
            sum(stats["near_duplicate_records"] for stats in per_split.values()),
            "within-split near-duplicate share stays below the configured ceiling",
            "one or more splits contain too many near-duplicate records",
            measured=round(worst, 4),
            threshold=ctx.thresholds.max_intra_split_near_dup_fraction,
            source=ctx.threshold_source,
            per_split=per_split,
            similarity_threshold=ctx.thresholds.intra_split_similarity_threshold,
        )
    ]


def _check_morphology_shortcuts(ctx: AuditContext) -> list[Finding]:
    config = _require_config(ctx)
    label_config = {label.name: label for label in config.labels}
    group_labels: dict[str, set[str]] = defaultdict(set)
    for label in config.labels:
        group_labels[label.morphology_group].add(label.name)
    shape_labels: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for record in ctx.rows:
        for annotation in record.annotations:
            configured_label = label_config.get(annotation.entity_type)
            if configured_label:
                shape_labels[(configured_label.morphology_group, shape_signature(annotation.text))][
                    annotation.entity_type
                ] += 1
    exclusive = []
    dominant = []
    max_share = 0.0
    for (group, shape), counts in shape_labels.items():
        if len(group_labels[group]) < 2:
            continue
        if len(counts) == 1:
            exclusive.append(f"{group}:{shape}")
        total = sum(counts.values())
        share = max(counts.values()) / total
        max_share = max(max_share, share)
        if share > ctx.thresholds.max_morphology_label_share:
            dominant.append((group, shape, round(share, 4)))
    return [
        _finding(
            "label_exclusive_morphology",
            bool(exclusive),
            len(exclusive),
            "no morphology in a multi-label group is label-exclusive",
            "label-exclusive morphology detected",
            measured=len(exclusive),
            threshold=0,
            source=ctx.threshold_source,
            shapes=exclusive,
        ),
        _finding(
            "morphology_label_dominance",
            bool(dominant),
            len(dominant),
            "P(label | shape) remains below the configured ceiling",
            "one or more shapes overly predict a label",
            measured=round(max_share, 4),
            threshold=ctx.thresholds.max_morphology_label_share,
            source=ctx.threshold_source,
            shapes=dominant,
        ),
    ]


def _check_shape_entity_shortcut(ctx: AuditContext) -> list[Finding]:
    annotated: Counter[str] = Counter()
    unannotated: Counter[str] = Counter()
    for record in ctx.rows:
        spans = [(annotation.start, annotation.end) for annotation in record.annotations]
        for annotation in record.annotations:
            if _IDENTIFIER_TOKEN_RE.fullmatch(annotation.text):
                annotated[shape_signature(annotation.text)] += 1
        for match in _IDENTIFIER_TOKEN_RE.finditer(record.text):
            if any(start < match.end() and match.start() < end for start, end in spans):
                continue
            unannotated[shape_signature(match.group(0))] += 1
    offenders = []
    max_share = 0.0
    for shape, annotated_count in sorted(annotated.items()):
        support = annotated_count + unannotated.get(shape, 0)
        if support < ctx.thresholds.minimum_shape_support:
            continue
        share = annotated_count / support
        max_share = max(max_share, share)
        if share > ctx.thresholds.max_shape_entity_share:
            offenders.append(
                {
                    "annotated": annotated_count,
                    "shape": shape,
                    "share": round(share, 4),
                    "support": support,
                }
            )
    return [
        _finding(
            "shape_entity_shortcut",
            bool(offenders),
            len(offenders),
            "every measured value shape also occurs as a non-entity surface",
            "one or more value shapes occur almost exclusively as annotated entities",
            measured=round(max_share, 4),
            threshold=ctx.thresholds.max_shape_entity_share,
            source=ctx.threshold_source,
            shapes=offenders,
        )
    ]


def _check_cue_label_shortcuts(ctx: AuditContext) -> list[Finding]:
    if not any(record.cue_links for record in ctx.rows):
        return [
            Finding(
                risk="cue_label_shortcuts",
                status="UNMEASURED",
                count=None,
                reason="no explicit cue-to-entity links are present to measure",
            )
        ]
    cue_stats_by_split: dict[str, dict[str, Any]] = {}
    cue_fractions_by_split: dict[str, float] = {}
    for split, split_rows in ctx.split_records.items():
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
        if fraction > ctx.thresholds.max_label_exclusive_cue_fraction
    ]
    maximum_cue_fraction = max(cue_fractions_by_split.values(), default=1.0)
    exclusive_record_count = sum(
        stats["exclusive_records"] for stats in cue_stats_by_split.values()
    )
    return [
        _finding(
            "cue_label_shortcuts",
            bool(failed_cue_splits),
            exclusive_record_count,
            "label-exclusive cue use stays below the configured fraction in every split",
            "label-exclusive cue surfaces exceed the ceiling in one or more splits",
            measured=round(maximum_cue_fraction, 4),
            threshold=ctx.thresholds.max_label_exclusive_cue_fraction,
            source=ctx.threshold_source,
            failed_splits=failed_cue_splits,
            fraction=round(maximum_cue_fraction, 4),
            per_split=cue_stats_by_split,
        )
    ]


def _check_label_marker_shortcuts(ctx: AuditContext) -> list[Finding]:
    cue_strings = tuple(
        " ".join(cue.split()).casefold()
        for label in (ctx.config.labels if ctx.config else ())
        for cue in label.cues
    )
    feature_labels: dict[str, Counter[str]] = defaultdict(Counter)
    feature_widths: dict[str, int] = {}
    label_totals: Counter[str] = Counter()
    for record in ctx.rows:
        if record.kind != "positive" or len(record.annotations) != 1:
            continue
        label = record.annotations[0].entity_type
        label_totals[label] += 1
        tokens = re.findall(r"[a-z]+|<[^>]+>", _record_skeleton(record))
        features: set[str] = set()
        for width in (1, 2, 3):
            for index in range(max(0, len(tokens) - width + 1)):
                window = tokens[index : index + width]
                if any(token.startswith("<") for token in window):
                    continue
                feature = " ".join(window)
                if any(feature in cue for cue in cue_strings):
                    continue
                features.add(feature)
                feature_widths[feature] = width
        for feature in features:
            feature_labels[feature][label] += 1
    suspicious: list[dict[str, Any]] = []
    for feature, counts in feature_labels.items():
        support = sum(counts.values())
        if support < ctx.thresholds.minimum_marker_support:
            continue
        label, label_count = counts.most_common(1)[0]
        share = label_count / support
        coverage = label_count / label_totals[label]
        if (
            share > ctx.thresholds.max_label_marker_share
            and coverage >= ctx.thresholds.minimum_marker_kind_coverage
        ):
            suspicious.append(
                {
                    "coverage": round(coverage, 4),
                    "feature": feature,
                    "label": label,
                    "share": round(share, 4),
                    "support": support,
                    "token_count": feature_widths[feature],
                }
            )
    suspicious.sort(key=lambda value: (value["token_count"], -value["support"], value["feature"]))
    return [
        _finding(
            "label_marker_shortcuts",
            bool(suspicious),
            len(suspicious),
            "no non-cue lexical marker makes a label nearly deterministic",
            "one or more non-cue lexical markers predict a label",
            measured=len(suspicious),
            threshold=0,
            source=ctx.threshold_source,
            marker_features=suspicious[:_MAX_REPORTED_OFFENDERS],
        )
    ]


def _check_cue_free_coverage(ctx: AuditContext) -> list[Finding]:
    config = _require_config(ctx)
    family_by_name = {family.name: family for family in config.families}
    cue_free_by_split = {
        split: sum(
            is_cue_free_evidence(record, config, family_by_name[record.family])
            for record in split_rows
            if record.family in family_by_name
        )
        for split, split_rows in ctx.split_records.items()
    }
    missing_cue_free_splits = [
        split for split, count in cue_free_by_split.items() if count == 0
    ]
    return [
        _finding(
            "cue_free_coverage",
            bool(missing_cue_free_splits),
            sum(cue_free_by_split.values()),
            "positive evidence without any configured cue surface is present in every split",
            "one or more splits lack positives free of every configured cue surface",
            per_split=cue_free_by_split,
            missing_splits=missing_cue_free_splits,
        )
    ]


def _check_contrastive_coverage(ctx: AuditContext) -> list[Finding]:
    config = _require_config(ctx)
    family_by_name = {family.name: family for family in config.families}
    contrastive_by_split = {
        split: sum(
            is_contrastive_evidence(record, config, family_by_name[record.family])
            for record in split_rows
            if record.family in family_by_name
        )
        for split, split_rows in ctx.split_records.items()
    }
    missing_contrastive_splits = [
        split for split, count in contrastive_by_split.items() if count == 0
    ]
    return [
        _finding(
            "cue_shape_contrastive_coverage",
            bool(missing_contrastive_splits),
            sum(contrastive_by_split.values()),
            "verified cue/shape disagreement evidence is present in every split",
            "one or more splits lack verified cue/shape disagreement evidence",
            per_split=contrastive_by_split,
            missing_splits=missing_contrastive_splits,
        )
    ]


def _check_template_persona_diversity(ctx: AuditContext) -> list[Finding]:
    config = _require_config(ctx)
    templates_by_cell: dict[tuple[str, str], set[str]] = defaultdict(set)
    personas_by_cell: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in ctx.rows:
        templates_by_cell[(record.split, record.family)].add(record.template_id)
        if record.persona and surfaced(record.persona, record.text):
            personas_by_cell[(record.split, record.family)].add(record.persona)
    thin_templates = [
        f"{split}/{family.name}"
        for split in ctx.split_records
        for family in config.families
        if len(templates_by_cell[(split, family.name)])
        < config.diversity.minimum_templates_per_family
    ]
    thin_personas = [
        f"{split}/{family.name}"
        for split in ctx.split_records
        for family in config.families
        if family.role == "positive"
        and len(personas_by_cell[(split, family.name)])
        < config.diversity.minimum_personas_per_family
    ]
    return [
        _finding(
            "template_diversity",
            bool(thin_templates),
            len(thin_templates),
            "template diversity minimums are met",
            "one or more family/split cells lack template depth",
            cells=thin_templates,
        ),
        _finding(
            "persona_diversity",
            bool(thin_personas),
            len(thin_personas),
            "persona diversity minimums are met",
            "one or more positive family/split cells lack persona depth",
            cells=thin_personas,
        ),
    ]


def _check_hard_negative_coverage(ctx: AuditContext) -> list[Finding]:
    config = _require_config(ctx)
    negatives = [record for record in ctx.rows if record.kind == "hard_negative"]
    hard_kinds = {record.hard_negative_kind for record in negatives if record.hard_negative_kind}
    ratio_failures = [
        split
        for split, split_rows in ctx.split_records.items()
        if sum(r.kind == "hard_negative" for r in split_rows) / len(split_rows)
        < config.minimum_hard_negative_ratio
    ]
    weak_hard_negative = (
        bool(ratio_failures) or len(hard_kinds) < ctx.thresholds.minimum_hard_negative_kinds
    )
    return [
        _finding(
            "hard_negative_coverage",
            weak_hard_negative,
            len(negatives),
            "hard-negative ratio and kind coverage are sufficient",
            "hard-negative ratio or kind coverage is insufficient",
            measured=len(hard_kinds),
            threshold=ctx.thresholds.minimum_hard_negative_kinds,
            source=ctx.threshold_source,
            kinds=sorted(v for v in hard_kinds if v),
            ratio_failures=ratio_failures,
        )
    ]


def _check_family_imbalance(ctx: AuditContext) -> list[Finding]:
    family_counts = Counter(record.family for record in ctx.rows)
    max_family_share = (
        max(family_counts.values(), default=0) / len(ctx.rows) if ctx.rows else 1.0
    )
    return [
        _finding(
            "family_imbalance",
            max_family_share > ctx.thresholds.max_family_share,
            max(family_counts.values(), default=0),
            "no family exceeds the configured corpus share",
            "one family exceeds the configured corpus share",
            measured=round(max_family_share, 4),
            threshold=ctx.thresholds.max_family_share,
            source=ctx.threshold_source,
            share=round(max_family_share, 4),
        )
    ]


def _values_by_label(rows: Iterable[Record]) -> dict[str, set[str]]:
    values: dict[str, set[str]] = defaultdict(set)
    for record in rows:
        for annotation in record.annotations:
            values[annotation.entity_type].add(annotation.text)
    return values


def _check_value_diversity(ctx: AuditContext) -> list[Finding]:
    values_by_label = _values_by_label(ctx.rows)
    entity_values = [
        annotation.text for record in ctx.rows for annotation in record.annotations
    ]
    if not entity_values:
        return [
            Finding(
                risk="value_diversity",
                status="UNMEASURED",
                count=None,
                reason="no annotated values are present to measure",
            )
        ]
    entropy = _entropy(entity_values)
    thin_labels = {
        label: len(values)
        for label, values in sorted(values_by_label.items())
        if len(values) < ctx.thresholds.minimum_distinct_values_per_label
    }
    minimum_distinct = min(
        (len(values) for values in values_by_label.values()), default=0
    )
    failed = bool(thin_labels)
    return [
        _finding(
            "value_diversity",
            failed,
            len(set(entity_values)),
            "every label meets the distinct-value floor",
            "one or more labels fall below the distinct-value floor",
            measured=minimum_distinct,
            threshold=ctx.thresholds.minimum_distinct_values_per_label,
            source=ctx.threshold_source,
            per_label={label: len(values) for label, values in sorted(values_by_label.items())},
            distinct_count_entropy_bits=round(entropy, 4),
            thin_labels=thin_labels,
        )
    ]


# Share of a label's values that must carry the same affix before it is dominant.
_DOMINANT_AFFIX_SHARE = 0.5


def _dominant_affix(values: list[str], width: int, *, suffix: bool) -> tuple[str, float]:
    counts = Counter(
        (value[-width:] if suffix else value[:width]) for value in values if len(value) >= width
    )
    if not counts:
        return "", 0.0
    affix, count = counts.most_common(1)[0]
    return affix, count / len(values)


def _longest_dominant_affix(
    values: list[str], *, suffix: bool
) -> tuple[str, int, float]:
    best = ("", 0, 0.0)
    for width in range(1, max((len(value) for value in values), default=0) + 1):
        affix, share = _dominant_affix(values, width, suffix=suffix)
        if share >= _DOMINANT_AFFIX_SHARE:
            best = (affix, width, share)
    return best


def _check_value_shared_affix(ctx: AuditContext) -> list[Finding]:
    maximum_length = ctx.thresholds.max_shared_affix_chars
    offenders: dict[str, dict[str, Any]] = {}
    longest_measured = 0
    for label, values in sorted(_values_by_label(ctx.rows).items()):
        if len(values) < 2:
            continue
        ordered = sorted(values)
        prefix, prefix_length, prefix_share = _longest_dominant_affix(
            ordered, suffix=False
        )
        affix_suffix, suffix_length, suffix_share = _longest_dominant_affix(
            ordered, suffix=True
        )
        label_length = max(prefix_length, suffix_length)
        longest_measured = max(longest_measured, label_length)
        if label_length > maximum_length:
            offenders[label] = {
                "prefix": prefix,
                "prefix_length": prefix_length,
                "prefix_share": round(prefix_share, 4),
                "suffix": affix_suffix,
                "suffix_length": suffix_length,
                "suffix_share": round(suffix_share, 4),
            }
    if offenders:
        return [
            Finding(
                risk="value_shared_affix",
                status="WARN",
                count=len(offenders),
                reason="most values under one or more labels share a constant affix "
                "longer than the configured ceiling; a detector can match the affix "
                "instead of the value",
                details={
                    "dominant_share_definition": _DOMINANT_AFFIX_SHARE,
                    "labels": offenders,
                },
                measured=longest_measured,
                threshold=maximum_length,
                threshold_source=ctx.threshold_source,
            )
        ]
    return [
        _finding(
            "value_shared_affix",
            False,
            0,
            "no label's values share a dominant constant affix longer than the "
            "configured ceiling",
            "",
            measured=longest_measured,
            threshold=maximum_length,
            source=ctx.threshold_source,
            dominant_share_definition=_DOMINANT_AFFIX_SHARE,
        )
    ]


def _check_span_integrity(ctx: AuditContext) -> list[Finding]:
    malformed = 0
    for record in ctx.rows:
        try:
            validate_annotations(record.text, record.annotations)
        except AnnotationError:
            malformed += 1
    return [
        _finding(
            "malformed_or_unbalanced_spans",
            bool(malformed),
            malformed,
            "all spans round-trip exactly",
            "malformed, overlapping or unbalanced spans detected",
        )
    ]


def _check_unsafe_values(ctx: AuditContext) -> list[Finding]:
    config = _require_config(ctx)
    label_plugins = {label.name: label.plugin for label in config.labels}
    unsafe = sum(
        bool(unsafe_record_reasons(record, config.safety, label_plugins=label_plugins))
        for record in ctx.rows
    )
    return [
        _finding(
            "unsafe_generated_value_patterns",
            bool(unsafe),
            unsafe,
            "no configured unsafe pattern was found",
            "unsafe generated-value patterns detected",
        )
    ]


def _check_generator_fingerprint(ctx: AuditContext) -> list[Finding]:
    template_counts = Counter(record.template_id for record in ctx.rows)
    # Template identity is generator metadata; without a config the share is
    # meaningless and only the lexical marker half of the check applies.
    template_share = (
        (max(template_counts.values(), default=0) / len(ctx.rows) if ctx.rows else 1.0)
        if ctx.config is not None
        else 0.0
    )
    marker_features = _suspicious_kind_markers(
        ctx.rows,
        minimum_support=ctx.thresholds.minimum_marker_support,
        maximum_kind_share=ctx.thresholds.max_kind_marker_share,
        minimum_kind_coverage=ctx.thresholds.minimum_marker_kind_coverage,
    )
    fingerprint_failed = (
        template_share > ctx.thresholds.max_template_share or bool(marker_features)
    )
    return [
        _finding(
            "generator_fingerprint",
            fingerprint_failed,
            len(marker_features)
            if marker_features
            else max(template_counts.values(), default=0),
            "template concentration and kind-predictive markers stay below configured ceilings",
            "template concentration or a kind-predictive marker creates a shortcut",
            measured=round(template_share, 4),
            threshold=ctx.thresholds.max_template_share,
            source=ctx.threshold_source,
            share=round(template_share, 4),
            marker_features=marker_features,
        )
    ]


def _check_pervasive_phrases(ctx: AuditContext) -> list[Finding]:
    total = len(ctx.rows)
    if total < ctx.thresholds.minimum_marker_support:
        return [
            Finding(
                risk="pervasive_phrase_fingerprint",
                status="UNMEASURED",
                count=None,
                reason="too few records to measure phrase coverage meaningfully",
            )
        ]
    document_frequency: Counter[str] = Counter()
    for record in ctx.rows:
        tokens = _SKELETON_TOKEN_RE.findall(_record_skeleton(record))
        grams = {
            " ".join(tokens[index : index + _PERVASIVE_NGRAM_WIDTH])
            for index in range(max(0, len(tokens) - _PERVASIVE_NGRAM_WIDTH + 1))
        }
        document_frequency.update(grams)
    ceiling = ctx.thresholds.max_pervasive_ngram_coverage
    pervasive = sorted(
        (
            (count / total, gram)
            for gram, count in document_frequency.items()
            if count / total >= ceiling
        ),
        key=lambda entry: (-entry[0], entry[1]),
    )
    offenders = [
        {"coverage": round(coverage, 4), "ngram": gram} for coverage, gram in pervasive
    ]
    max_coverage = (
        max((count / total for count in document_frequency.values()), default=0.0)
    )
    return [
        _finding(
            "pervasive_phrase_fingerprint",
            bool(offenders),
            len(offenders),
            "no skeleton phrase covers an outsized share of the corpus",
            "one or more constant phrases blanket the corpus and fingerprint the generator",
            measured=round(max_coverage, 4),
            threshold=ceiling,
            source=ctx.threshold_source,
            ngrams=offenders[:_MAX_REPORTED_OFFENDERS],
        )
    ]


def _check_threshold_strictness(ctx: AuditContext) -> list[Finding]:
    if ctx.threshold_source == "reference":
        return [
            Finding(
                risk="threshold_strictness",
                status="PASS",
                count=0,
                reason="audit ran with the reference profile thresholds",
                details={"profile": "reference"},
            )
        ]
    configured = asdict(ctx.thresholds)
    configured_probe = configured.pop("probe")
    configured.update(
        {
            f"probe.{key}": value
            for key, value in configured_probe.items()
            if key != "enabled"
        }
    )
    weaker: list[dict[str, Any]] = []
    for key in WEAKER_WHEN_HIGHER:
        if configured[key] > REFERENCE_THRESHOLDS[key]:
            weaker.append(
                {
                    "configured": configured[key],
                    "reference": REFERENCE_THRESHOLDS[key],
                    "key": key,
                }
            )
    for key in WEAKER_WHEN_LOWER:
        if configured[key] < REFERENCE_THRESHOLDS[key]:
            weaker.append(
                {
                    "configured": configured[key],
                    "reference": REFERENCE_THRESHOLDS[key],
                    "key": key,
                }
            )
    if weaker:
        return [
            Finding(
                risk="threshold_strictness",
                status="WARN",
                count=len(weaker),
                reason="one or more configured audit thresholds are weaker than the "
                "reference profile; PASS verdicts above are only as strong as these limits",
                details={"weaker_than_reference": weaker},
            )
        ]
    return [
        Finding(
            risk="threshold_strictness",
            status="PASS",
            count=0,
            reason="every configured audit threshold is at least as strict as the reference",
            details={},
        )
    ]


def _check_probe(ctx: AuditContext) -> list[Finding]:
    if not ctx.probe_enabled:
        return unmeasured_probe_findings(
            "probe disabled; run piicorpus audit --probe to measure learnability"
        )
    probe_config = ctx.thresholds.probe
    train_split = "train" if "train" in ctx.split_records else next(iter(ctx.split_records), "")
    return probe_findings(
        ctx.split_records,
        train_split=train_split,
        max_kind_accuracy=probe_config.max_kind_accuracy,
        max_value_label_accuracy=probe_config.max_value_label_accuracy,
        max_context_label_accuracy=probe_config.max_context_label_accuracy,
        threshold_source=ctx.threshold_source,
    )


def _check_holdout_dependence(_ctx: AuditContext) -> list[Finding]:
    return [
        Finding(
            risk="same_generator_holdout_dependence",
            status="UNMEASURED",
            count=None,
            reason=SYNTHETIC_HOLDOUT_LIMITATION,
            details={"independent_generalization": "not measured"},
        )
    ]


_CHECKS: tuple[CheckSpec, ...] = (
    CheckSpec(
        risks=(
            "cross_split_value_contamination",
            "persona_contamination",
            "template_contamination",
            "skeleton_contamination",
        ),
        run=_check_cross_split_contamination,
    ),
    CheckSpec(risks=("duplicate_or_near_duplicate_bodies",), run=_check_duplicate_bodies),
    CheckSpec(risks=("intra_split_redundancy",), run=_check_intra_split_redundancy),
    CheckSpec(
        risks=("label_exclusive_morphology", "morphology_label_dominance"),
        run=_check_morphology_shortcuts,
        requires_config=True,
    ),
    CheckSpec(risks=("shape_entity_shortcut",), run=_check_shape_entity_shortcut),
    CheckSpec(risks=("cue_label_shortcuts",), run=_check_cue_label_shortcuts),
    CheckSpec(risks=("label_marker_shortcuts",), run=_check_label_marker_shortcuts),
    CheckSpec(risks=("cue_free_coverage",), run=_check_cue_free_coverage, requires_config=True),
    CheckSpec(
        risks=("cue_shape_contrastive_coverage",),
        run=_check_contrastive_coverage,
        requires_config=True,
    ),
    CheckSpec(
        risks=("template_diversity", "persona_diversity"),
        run=_check_template_persona_diversity,
        requires_config=True,
    ),
    CheckSpec(
        risks=("hard_negative_coverage",),
        run=_check_hard_negative_coverage,
        requires_config=True,
    ),
    CheckSpec(
        risks=("family_imbalance",), run=_check_family_imbalance, requires_config=True
    ),
    CheckSpec(risks=("value_diversity",), run=_check_value_diversity),
    CheckSpec(risks=("value_shared_affix",), run=_check_value_shared_affix),
    CheckSpec(risks=("malformed_or_unbalanced_spans",), run=_check_span_integrity),
    CheckSpec(
        risks=("unsafe_generated_value_patterns",),
        run=_check_unsafe_values,
        requires_config=True,
    ),
    CheckSpec(risks=("generator_fingerprint",), run=_check_generator_fingerprint),
    CheckSpec(risks=("pervasive_phrase_fingerprint",), run=_check_pervasive_phrases),
    CheckSpec(
        risks=(
            "probe_kind_separability",
            "probe_value_label_shortcut",
            "probe_context_label_shortcut",
        ),
        run=_check_probe,
    ),
    CheckSpec(risks=("threshold_strictness",), run=_check_threshold_strictness),
    CheckSpec(risks=("same_generator_holdout_dependence",), run=_check_holdout_dependence),
)


def registered_checks() -> tuple[CheckSpec, ...]:
    return _CHECKS


def run_checks(ctx: AuditContext) -> list[Finding]:
    """Run every registered check the context supports; the rest stay UNMEASURED."""
    findings: list[Finding] = []
    for spec in _CHECKS:
        if spec.requires_config and ctx.config is None:
            findings.extend(
                Finding(
                    risk=risk,
                    status="UNMEASURED",
                    count=None,
                    reason="requires the generating corpus configuration",
                )
                for risk in spec.risks
            )
            continue
        findings.extend(spec.run(ctx))
    return findings


def build_report(findings: Iterable[Finding]) -> AuditReport:
    ordered = tuple(findings)
    summary = Counter(finding.status for finding in ordered)
    return AuditReport(findings=ordered, summary=dict(sorted(summary.items())))


def audit_corpus(
    directory: str | Path,
    *,
    allow_invalid: bool = False,
    profile: str = "config",
    probe: bool | None = None,
) -> AuditReport:
    from .config import reference_audit_config

    validation = validate_corpus(directory, strict=True)
    if not validation.valid and not allow_invalid:
        raise CorpusIntegrityError(validation)
    config, split_records, _manifest = load_corpus(directory)
    rows = tuple(
        record for split in ("train", "eval", "holdout") for record in split_records[split]
    )
    if profile not in {"config", "reference"}:
        raise ValueError(f"unknown audit profile: {profile}")
    thresholds = reference_audit_config() if profile == "reference" else config.audit
    ctx = AuditContext(
        split_records=split_records,
        rows=rows,
        thresholds=thresholds,
        threshold_source=profile,
        config=config,
        probe_enabled=config.audit.probe.enabled if probe is None else probe,
    )
    findings: list[Finding] = [_integrity_finding(validation)]
    findings.extend(run_checks(ctx))
    return build_report(findings)


def audit_external_records(
    split_records: dict[str, list[Record]],
    *,
    probe: bool = True,
    fail_on_safety: bool = False,
) -> AuditReport:
    """Audit records that did not come from a PIIcorpus generator.

    Config-dependent checks report UNMEASURED; the remaining structural checks
    and the probe run with the reference threshold profile. A sensitive-content
    scan over the text is reported informationally (or as FAIL with
    ``fail_on_safety``) because external data may legitimately contain real
    surface forms that generated corpora must never carry.
    """
    from .config import SafetyConfig

    rows = tuple(record for records in split_records.values() for record in records)
    from .config import reference_audit_config

    ctx = AuditContext(
        split_records=split_records,
        rows=rows,
        thresholds=reference_audit_config(),
        threshold_source="reference",
        config=None,
        probe_enabled=probe,
    )
    findings: list[Finding] = [
        Finding(
            risk="corpus_integrity",
            status="UNMEASURED",
            count=None,
            reason="external data carries no manifest; file integrity is not verifiable",
        )
    ]
    findings.extend(run_checks(ctx))
    scan_config = SafetyConfig(
        reserved_email_domains=("example.com", "example.org", "example.net"),
        allowed_value_prefixes=(),
        forbidden_terms=(),
    )
    reason_counts: Counter[str] = Counter()
    affected = 0
    for record in rows:
        reasons = unsafe_text_reasons(record.text, scan_config)
        if reasons:
            affected += 1
            reason_counts.update(set(reasons))
    if affected:
        findings.append(
            Finding(
                risk="external_safety_scan",
                status="FAIL" if fail_on_safety else "WARN",
                count=affected,
                reason="external text matches sensitive-content patterns; "
                "review before any release",
                details={"reasons": dict(sorted(reason_counts.items()))},
            )
        )
    else:
        findings.append(
            Finding(
                risk="external_safety_scan",
                status="PASS",
                count=0,
                reason="no sensitive-content pattern matched the external text",
                details={},
            )
        )
    return build_report(findings)


def _integrity_finding(validation: ValidationReport) -> Finding:
    return _finding(
        "corpus_integrity",
        not validation.valid,
        len(validation.errors),
        "strict validation passed before audit",
        "strict validation failed; forensic results are non-authoritative",
        errors=list(validation.errors),
    )

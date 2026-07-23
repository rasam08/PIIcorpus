"""Reference audit thresholds representing the project's recommended strictness.

Corpus configurations may be laxer or stricter than this profile; the audit's
``threshold_strictness`` finding reports every configured threshold that is
weaker than the reference so a clean report cannot silently rely on lenient
self-chosen ceilings. ``piicorpus audit --profile reference`` runs the checks
with these values regardless of the corpus configuration.
"""

from __future__ import annotations

REFERENCE_AUDIT: dict[str, float | int] = {
    "max_morphology_label_share": 0.75,
    "max_label_exclusive_cue_fraction": 0.45,
    "max_family_share": 0.12,
    "minimum_hard_negative_kinds": 6,
    "max_template_share": 0.05,
    "max_kind_marker_share": 0.90,
    "minimum_marker_kind_coverage": 0.50,
    "minimum_marker_support": 20,
    "near_duplicate_jaccard": 0.97,
    "intra_split_similarity_threshold": 0.85,
    "max_intra_split_near_dup_fraction": 0.05,
    "max_shape_entity_share": 0.90,
    "minimum_shape_support": 20,
    "max_pervasive_ngram_coverage": 0.35,
    "max_label_marker_share": 0.90,
    "minimum_distinct_values_per_label": 30,
    "max_shared_affix_chars": 6,
}

REFERENCE_PROBE: dict[str, float] = {
    "max_kind_accuracy": 0.90,
    "max_value_label_accuracy": 0.90,
    "max_context_label_accuracy": 0.90,
}

REFERENCE_THRESHOLDS: dict[str, float | int] = {
    **REFERENCE_AUDIT,
    **{f"probe.{key}": value for key, value in REFERENCE_PROBE.items()},
}

# Threshold direction is explicit rather than inferred from names. Some minimums
# define how much evidence a detector needs before it reports a shortcut; raising
# those thresholds makes the audit less sensitive and is therefore weaker.
WEAKER_WHEN_HIGHER = (
    "intra_split_similarity_threshold",
    "max_family_share",
    "max_intra_split_near_dup_fraction",
    "max_kind_marker_share",
    "max_label_exclusive_cue_fraction",
    "max_label_marker_share",
    "max_morphology_label_share",
    "max_pervasive_ngram_coverage",
    "max_shape_entity_share",
    "max_shared_affix_chars",
    "max_template_share",
    "minimum_marker_kind_coverage",
    "minimum_marker_support",
    "minimum_shape_support",
    "near_duplicate_jaccard",
    "probe.max_context_label_accuracy",
    "probe.max_kind_accuracy",
    "probe.max_value_label_accuracy",
)
WEAKER_WHEN_LOWER = (
    "minimum_distinct_values_per_label",
    "minimum_hard_negative_kinds",
)

if set(WEAKER_WHEN_HIGHER) | set(WEAKER_WHEN_LOWER) != set(REFERENCE_THRESHOLDS):
    raise RuntimeError("every reference threshold must declare its strictness direction")

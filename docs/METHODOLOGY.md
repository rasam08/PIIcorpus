# Methodology

## Determinism

Every pseudorandom stream is derived from the normalized configuration digest, explicit seed,
generator version, split, family, record index, and purpose. JSON keys are sorted, JSONL is compact,
text uses UTF-8, and newlines are LF. Timestamps and filesystem order are excluded from generated
artifacts.

Determinism is necessary for review and regression testing. It does not imply that the generated
distribution is realistic.

## Split isolation

Train, eval, and holdout use disjoint persona pools, organization pools, template slices, namespace
prefixes, calendar ranges, and identifier alphabets. Validation independently derives and rejects
collisions in case IDs, values, personas, organizations, template IDs, normalized template
skeletons, and family/index namespaces.

Isolation is checked from the JSONL records rather than inferred from the generator implementation.
File hashes and manifest counts are also recalculated.

## Shortcut resistance

Identifier labels share the same fictional morphology classes. The audit calculates morphology
usage and `P(label | shape)` from annotation values, then applies configurable exclusivity and
dominance ceilings. Cue-free examples and cue-versus-shape conflicts are required so neither value
shape nor a single cue surface is the only available signal. Cue measurements use explicit
cue-to-entity links, including in multi-entity records, and apply the configured shortcut ceiling
to each split independently. Contrastive records must carry an emitted shape that matches the
configured shape hint for a different label, and both cue-free and contrastive evidence must be
present in every split.

Shared morphology reduces one shortcut but cannot prove that another shortcut was not introduced.
The audit therefore also measures cue exclusivity, family balance, value entropy, template
concentration, hard-negative coverage, split contamination, and one- through three-token lexical
markers that nearly determine positive versus hard-negative kind.

## Negative coverage

Hard negatives are explicit records with no entity spans. The demo includes near misses,
placeholders, negation, documentation references, unrelated identifier shapes, and adjacent
non-sensitive values. Their ratio and distinct kinds are manifest counts and independently audited
properties.

Every generated record also renders the same neutral synthetic context/reference structure, so
uniqueness metadata is class-balanced rather than a hard-negative-only label marker.

## Human-supplied material

Imported text is kept outside generated splits, marked `human_supplied`, and given stable
content-derived IDs. The importer makes no safety or licensing claim and omits bodies from normal
error messages. Independence, consent, privacy, and release suitability require separate review.

> A holdout produced by the same generator is useful for regression testing but is not an
> independent generalization test.

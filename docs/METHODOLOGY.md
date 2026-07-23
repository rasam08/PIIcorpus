# Methodology

## Determinism

Every pseudorandom stream is derived from the normalized configuration digest, explicit seed,
generator version, split, family, record index, and purpose. JSON keys are sorted, JSONL is compact,
text uses UTF-8, and newlines are LF. Timestamps and filesystem order are excluded from generated
artifacts. `piicorpus reproduce` regenerates a corpus from its own configuration snapshot and
byte-compares the result, making the determinism claim operationally checkable in one command.

Determinism is necessary for review and regression testing. It does not imply that the generated
distribution is realistic.

## Split isolation without distribution shift

Train, eval, and holdout draw personas, organizations, identifier letters, and calendar years from
shared pools partitioned by interleaving: the sorted pool is dealt out modulo the split count, so
no split receives a contiguous (for example alphabetical or chronological) range. Splits stay
disjoint and avoid that order-driven shift, but arbitrary user-supplied pools are not claimed to
be statistically identical. Template banks are still sliced per split, and a global uniqueness
pool guarantees that no annotated or hard-negative value repeats anywhere in the corpus.

Validation independently derives and rejects collisions in case IDs, values, personas,
organizations, template IDs, normalized template skeletons, and family/index namespaces. Isolation
is checked from the JSONL records rather than inferred from the generator implementation. File
hashes and manifest counts are also recalculated.

## Shortcut resistance

Identifier labels share morphology classes, and the audit calculates morphology usage and
`P(label | shape)` from annotation values with configurable exclusivity and dominance ceilings.
Near-miss and adjacent hard negatives generate their surface tokens through the configured label
plugins themselves, so every annotated value shape also occurs as a non-entity surface; the
`shape_entity_shortcut` check verifies this from the emitted records. Cue-free examples and
cue-versus-shape conflicts are required so neither value shape nor a single cue surface is the
only available signal. Cue measurements use explicit cue-to-entity links, including in
multi-entity records, and apply the configured shortcut ceiling to each split independently.
Contrastive records must carry an emitted shape that matches the configured shape hint for a
different label, and both cue-free and contrastive evidence must be present in every split.

Shared morphology reduces one shortcut but cannot prove that another shortcut was not introduced.
The audit therefore also measures cue exclusivity, non-cue label markers, family balance, value
diversity, within-split redundancy, template concentration, pervasive phrases, hard-negative
coverage, split contamination, and kind-predictive lexical markers.

## The learnability probe

Structural checks are heuristics for a single underlying question: could a trivial model pass this
corpus by exploiting surface signal? The probe answers it empirically. Hashed character
3-5-gram features feed one-vs-rest logistic regression trained by plain SGD — standard library
only, fixed seed, byte-deterministic — on the train split, and accuracy on the held splits is
compared with configured ceilings for kind separability, value-only label prediction, and
masked-context label prediction. The probe is opt-in because the audit is otherwise
detector-independent; its verdict is one-sided (high accuracy proves a shortcut; low accuracy
proves nothing about difficulty).

## Negative coverage

Hard negatives are explicit records with no entity spans. The demo includes near misses,
placeholders, negation, documentation references, unrelated identifier shapes, and adjacent
non-sensitive values. Near-miss and adjacent values mirror the positive value distribution (see
above). Their ratio and distinct kinds are manifest counts and independently audited properties.

Every generated record renders one of several rotating context/reference footers, so uniqueness
metadata is class-balanced and no single constant phrase blankets the corpus.

## Human-supplied and external material

Imported text is kept outside generated splits, marked `human_supplied`, scanned informationally
for sensitive-content patterns, and given stable content-derived IDs. The importer makes no safety
or licensing claim and omits bodies from normal error messages. External datasets can be audited
with `piicorpus audit-external`, where checks that need generator metadata stay `UNMEASURED`.
Independence, consent, privacy, and release suitability require separate review.

> A holdout produced by the same generator is useful for regression testing but is not an
> independent generalization test.

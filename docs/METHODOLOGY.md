# Methodology

## Determinism

Every pseudorandom stream is derived from the normalized configuration digest, explicit seed,
generator version, split, family, record index, and purpose. JSON keys are sorted, JSONL is compact,
text uses UTF-8, and newlines are LF. Timestamps and filesystem order are excluded from generated
artifacts. `piicorpus reproduce` regenerates a corpus from its own configuration snapshot and
byte-compares the result, making the determinism claim operationally checkable in one command.

Deterministic output supports byte-level review and regression testing. It does not establish that
the generated distribution represents production data.

## Split partitioning

Train, eval, and holdout draw personas, organizations, identifier letters, and calendar years from
shared pools partitioned by interleaving: the sorted pool is dealt out modulo the split count, so
no split receives a contiguous alphabetical or chronological range. The resulting pool members are
disjoint, but arbitrary user-supplied pools are not guaranteed to have identical distributions.
Template banks are sliced per split, and a global uniqueness pool checks the final surfaced form
after OCR or spoken transformations. Annotated and hard-negative values cannot repeat within the
generated corpus.

Validation independently rejects repeated annotation values within a split and cross-split
collisions in values, personas, organizations, template IDs, normalized template skeletons, and
family/index namespaces. It also derives case-ID and namespace uniqueness. Isolation is checked
from the JSONL records rather than inferred from the generator implementation. File hashes and
manifest counts are also recalculated.

## Shortcut measurements

Identifier labels share morphology classes, and the audit calculates morphology usage and
`P(label | shape)` from annotation values with configurable exclusivity and dominance ceilings.
Near-miss and adjacent hard negatives generate their surface tokens through the configured label
plugins themselves, so every annotated value shape also occurs as a non-entity surface; the
`shape_entity_shortcut` check verifies this from the emitted records. Cue-free and
cue-versus-shape-conflict coverage is checked in each split. Cue measurements use explicit
cue-to-entity links, including in multi-entity records, and apply the configured shortcut ceiling
independently to each split. Contrastive records must contain a value whose emitted shape matches
the configured hint for a label other than the annotation label.

Shared morphology reduces one shortcut but cannot prove that another shortcut was not introduced.
The audit therefore also measures cue exclusivity, non-cue label markers, family balance, value
diversity, within-split redundancy, template concentration, pervasive phrases, hard-negative
coverage, split contamination, and kind-predictive lexical markers.

## The learnability probe

The learnability probe trains one-vs-rest logistic regression on hashed character 3-5-gram
features from the train split. Training uses deterministic stochastic gradient descent with a fixed
seed and standard-library implementations. Held-split balanced accuracy is compared with configured
ceilings and a fixed 0.05 margin above each split's balanced
majority-predictor baseline for kind separability, value-only label prediction, and masked-context
label prediction. Raw accuracy, raw majority baselines, balanced accuracy, macro-F1, and failing
splits are reported. Separability requires at least two observed classes in the training data and
in a held split. Degenerate held splits are listed under `unmeasured_splits` and excluded from the
verdict; if training is degenerate or no held split is measurable, the task is `UNMEASURED`. The
probe is optional. `FAIL` indicates character-level surface signal above both configured limits;
`PASS` applies only to the reported tasks, splits, features, and thresholds.

## Negative coverage

Hard negatives are explicit records with no entity spans. The demo includes near misses,
placeholders, negation, documentation references, unrelated identifier shapes, and adjacent
non-sensitive values. Near-miss and adjacent values mirror the positive value distribution (see
above). Their ratio and distinct kinds are manifest counts and independently audited properties.

Every generated record uses one of several rotating context/reference footers. Footer assignment
is class-balanced, and phrase coverage is measured by `pervasive_phrase_fingerprint`.

## Human-supplied and external material

Imported text is kept outside generated splits, marked `human_supplied`, scanned informationally
for sensitive-content patterns, and given stable content-derived IDs. The importer makes no safety
or licensing claim and omits bodies from normal error messages. External datasets can be audited
with `piicorpus audit-external`, where checks that need generator metadata stay `UNMEASURED`.
Independence, consent, privacy, and release suitability require separate review.

> A holdout produced by the same generator is useful for regression testing but is not an
> independent generalization test.

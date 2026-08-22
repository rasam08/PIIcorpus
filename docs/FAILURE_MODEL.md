# Corpus failure model

The audit treats common synthetic-data problems as named risks with `PASS`, `FAIL`, `WARN`, or
`UNMEASURED` status. It reports counts, the measured value, and the threshold it was judged
against rather than collapsing distinct problems into one score.

## Measured risks

Structural integrity and contamination:

- Cross-split contamination of values, personas, template identities, and normalized template
  skeletons.
- Exact and near-duplicate bodies. Near-duplicate detection uses word 4-shingles, 32-permutation
  MinHash signatures, and locality-sensitive banding (8 bands of 4 rows), then confirms every
  candidate pair with exact Jaccard similarity. Precision is exact; banding recall is
  probabilistic and falls off below roughly 0.8 Jaccard, which is documented behavior.
- Within-split redundancy: the share of records in any split that have a near-duplicate at the
  configured similarity threshold, with unique-skeleton ratios reported per split.

Shortcut structure:

- Label-exclusive morphology and excessive `P(label | shape)` inside multi-label morphology groups.
- `shape_entity_shortcut`: for every measured value shape, the share of identifier-shaped surface
  occurrences that are annotated entities. A shape that occurs almost exclusively as an entity
  lets a detector decide entityhood from shape alone. Only identifier-shaped surface forms
  (uppercase codes, emails, NANP phone formats, dashed or dotted digit groups) are measured;
  prose-like values are probe territory.
- Cue shortcuts from explicit cue-to-entity links, enforced independently per split.
- `label_marker_shortcuts`: non-cue lexical 1-3-grams that nearly determine a specific label.
  This is the open-world complement to the closed cue list.
- Missing cue-free or cue/shape-contrastive evidence in any split.

Diversity and balance:

- Template and persona diversity floors per family/split cell; hard-negative ratio and kind
  coverage; family imbalance.
- `value_diversity`: per-label distinct-value floors. Corpus-level distinct-value entropy is
  reported for context only and does not affect the verdict.
- `value_shared_affix` (WARN): the longest prefix or suffix carried by at least half of a label's
  values is measured in characters and compared with the configured character ceiling. This
  reports affixes that may be sufficient for a detector rule.

Generator fingerprints:

- Template concentration and kind-predictive lexical markers.
- `pervasive_phrase_fingerprint`: any skeleton 4-gram (including placeholder tokens) covering an
  amount of the corpus above the configured coverage threshold. This detects constant boilerplate
  that is balanced across kinds and therefore absent from kind-marker results.

Learnability probe (opt-in, `--probe`):

- `probe_kind_separability`, `probe_value_label_shortcut`, and `probe_context_label_shortcut`
  train a deterministic stdlib model (hashed character 3-5-grams, one-vs-rest logistic regression
  by SGD, fixed seed) on the train split and score the held splits. Verdicts use balanced accuracy
  and require both the configured ceiling and a 0.05
  margin above the split-specific balanced majority-predictor baseline to be exceeded. A failure
  is evidence of learnable surface signal beyond class priors; it does not prove the signal's
  cause. A task is `UNMEASURED` when its training data or every held split contains fewer than two
  observed classes. `PASS` applies only to the implemented probe and configured thresholds.

Safety and spans: malformed spans and unsafe values remain fail-closed checks.

## Threshold transparency

Every threshold-based finding records the measured value, the threshold, and the threshold source
(`config` or `reference`). The `threshold_strictness` finding compares the corpus configuration
against the recommended reference profile in `piicorpus.profiles` and warns when any configured
threshold is weaker. The selected threshold source is included in each threshold-based finding.
`piicorpus audit --profile reference` runs the checks with the reference profile directly.
Strictness directions are declared explicitly for similarity, evidence-support, structural, and
probe thresholds rather than inferred from threshold names.

## Interpretation considerations

An early synthetic corpus can make identifier morphology too predictive. A model may then learn a
shape-to-label mapping instead of using context. Adding shared morphologies can reduce that shortcut
while moving confusion to a different label boundary. Aggregate metrics can therefore differ from
per-family, per-label, span, negative, and independently sourced results.

Repeatedly revising a generator against the same evaluation set contaminates the research loop.
Independent evaluation data is required to measure behavior outside that feedback loop. A
synthetic holdout produced by the same engine retains the engine's distributional fingerprint even
when values, personas, and templates are disjoint.

## Status semantics

- `PASS`: the implemented measurement found no violation at the reported threshold.
- `FAIL`: the implemented measurement found a violation.
- `WARN`: a property worth knowing about that is not a defect by itself (a shared value affix, or
  thresholds weaker than the reference profile). WARN never changes the exit code.
- `UNMEASURED`: the corpus cannot support the claim.

An operational error is not a corpus verdict and uses exit code 2.

Audit first runs strict validation of manifest hashes, sizes, counts, spans, semantic evidence,
metadata surfaces, and content-derived case IDs. Invalid input is rejected with exit code 1. A
forensic override can continue measurements, but adds a `corpus_integrity` failure and cannot return
a clean verdict.

## External data

`piicorpus audit-external` runs every check that does not require the generating configuration on
arbitrary JSONL, Hugging Face-style JSONL, or CoNLL data, using the reference threshold profile.
Checks that depend on generator metadata (template identity, family structure, cue links, safety
prefixes) report `UNMEASURED` instead of guessing. A sensitive-content scan over the text is
reported as WARN by default because external data may legitimately contain real surface forms.

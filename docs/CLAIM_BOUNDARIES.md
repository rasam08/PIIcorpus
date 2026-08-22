# Claim boundaries

PIIcorpus measures properties of corpus files and, through `piicorpus score`, detector predictions
against those files. It does not run models or estimate how well a generated dataset represents
external data.

The following boundaries apply to every command and report:

- Audit and score results apply to the supplied files, implemented measurements, and configured
  thresholds; they are not external-performance estimates.
- A holdout produced by the same generator is not an independent evaluation set.
- Diversity and template counts measure distinct stored values and structures, not semantic
  coverage.
- Template variation does not exclude other generator fingerprints.
- Generated identifiers are not guaranteed to be realistic; the realistic-safe plugins produce
  realistic *shapes* whose values are reserved, fictional, or invalid by construction, which is
  still not real-world data.
- Regulatory compliance, de-identification, and deployment-readiness assessment are outside the
  package scope.
- External-performance measurement requires independently sourced evaluation data.
- The package generates, audits, and scores against corpora; it does not train or approve models.

`PASS` means only that an implemented risk check found no violation at its reported threshold.
It does not mean that the risk is absent outside the measured files. `WARN` marks a property worth
knowing about that is not a defect by itself. `UNMEASURED` is used when the artifact cannot
support a conclusion, including independent generalization from a same-generator holdout.

## Scoring boundary

Scores from `piicorpus score` report mechanism-specific behavior: cue dependence
(cued recall versus cue-free recall), gold recall on cue/shape conflicts, direct substitution of
the stored shape-hint label, other conflict errors, abstention, over-triggering on hard negatives,
and noise robustness gaps. These values characterize the submitted predictions on the supplied
corpus; they are not external-performance or deployment-readiness measurements.

## Probe boundary

The learnability probe reports class-balanced held-split metrics and split-specific baselines.
`FAIL` indicates that character n-gram performance exceeds the configured ceiling and baseline
margin. `PASS` applies only to the implemented features, classifier, splits, and thresholds.
Training data and held splits need at least two observed classes for separability to be measurable.
A degenerate held split is excluded and listed under `unmeasured_splits`; the task is `UNMEASURED`
if its training data is degenerate or no held split remains measurable.

The lexical generator-fingerprint check measures alphabetic 1-grams, 2-grams, and 3-grams.
Numeric-only, symbolic, and mixed alphanumeric fingerprints are not measured by that check; the
`shape_entity_shortcut` and `pervasive_phrase_fingerprint` checks cover identifier-shaped tokens
and placeholder-inclusive 4-grams respectively.

## Out of scope

PIIcorpus does not provide real issuing formats, data collection, data scraping, de-identification,
regulatory interpretation, model training, model distribution, detector approval, or publication
approval.

# Corpus failure model

The audit treats common synthetic-data problems as named risks with `PASS`, `FAIL`, or `UNMEASURED`
status. It reports counts and reasons rather than collapsing distinct problems into one score.

Measured risks include cross-split value and persona contamination, template and skeleton
contamination, duplicate bodies, label-exclusive morphology, excessive `P(label | shape)`, cue
shortcuts, missing cue-free or contrastive records, insufficient template or persona diversity,
weak hard-negative coverage, family imbalance, low value entropy, malformed spans, unsafe values,
template concentration, and repeated one-, two-, or three-token lexical markers that nearly
determine positive versus hard-negative kind. Cue shortcut measurements use explicit cue-to-entity
links rather than cross-producting every cue with every annotation in a multi-entity record, and
the configured cue-exclusivity ceiling is enforced independently in every split. Cue-free and
cue/shape-contrastive evidence is required in every split, and contrastive evidence must match the
emitted span's independently derived morphology. Independent generalization remains unmeasured for
a same-generator holdout.

## A recurring synthetic-corpus failure

An early synthetic corpus can make identifier morphology too predictive. A model may then learn a
shape-to-label mapping instead of using context. Adding shared morphologies can reduce that shortcut
while moving confusion to a different label boundary. This is why aggregate improvement is not
enough: per-family, per-label, span, negative, and independently sourced evaluation can disagree.

Repeatedly revising a generator against the same evaluation set contaminates the research loop.
After multiple unsuccessful candidates, the responsible conclusion may be to stop changing the
synthetic data and obtain genuinely independent evaluation. A synthetic holdout produced by the
same engine inherits the engine's distributional fingerprint even when values, personas, and
templates are disjoint.

This is a methodological warning, not a history of a particular product or experiment.

## Status semantics

- `PASS`: the implemented measurement found no violation at the configured threshold.
- `FAIL`: the implemented measurement found a violation.
- `UNMEASURED`: the corpus cannot support the claim.

An operational error is not a corpus verdict and uses exit code 2.

Audit first runs strict validation of manifest hashes, sizes, counts, spans, semantic evidence,
metadata surfaces, and content-derived case IDs. Invalid input is rejected with exit code 1. A
forensic override can continue measurements, but adds a `corpus_integrity` failure and cannot return
a clean verdict.

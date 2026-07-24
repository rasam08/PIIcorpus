# Claim boundaries

PIIcorpus measures properties of corpus files and, through `piicorpus score`, of detector
predictions against those files. It does not run models and does not establish that a dataset
represents the world outside its generator.

The following boundaries apply to every command and report:

- Synthetic data does not prove real-world accuracy.
- A same-generator holdout is not independent.
- Diversity counts do not prove semantic diversity.
- Template variation can still leave a generator fingerprint.
- Generated identifiers are not guaranteed to be realistic; the realistic-safe plugins produce
  realistic *shapes* whose values are reserved, fictional, or invalid by construction, which is
  still not real-world data.
- Regulatory compliance is not guaranteed.
- Real data is not de-identified by this project.
- No model becomes ready for deployment by using this project.
- Human-authored or externally sourced evaluation remains necessary.
- The project generates, audits, and scores against corpora; it does not train or approve models.

`PASS` means only that an implemented risk check found no violation at its reported threshold.
It does not mean that the risk is absent outside the measured files. `WARN` marks a property worth
knowing about that is not a defect by itself. `UNMEASURED` is used when the artifact cannot
support a conclusion, including independent generalization from a same-generator holdout.

## Scoring boundary

Scores from `piicorpus score` on synthetic data demonstrate *mechanism* failures: cue dependence
(cued recall versus cue-free recall), gold recall on cue/shape conflicts, direct substitution of
the stored shape-hint label, other conflict errors, abstention, over-triggering on hard negatives,
and noise robustness gaps. They never demonstrate real-world adequacy, and a perfect score on a
synthetic corpus is not evidence of deployment readiness.

## Probe boundary

The learnability probe reports class-balanced held-split metrics and split-specific baselines. A
failure is evidence that character n-grams carry learnable signal beyond majority-class priors,
but it does not prove what caused that signal or that the corpus is unusable. A passing result does
not prove the corpus is hard, diverse, or realistic.

The lexical generator-fingerprint check measures alphabetic 1-grams, 2-grams, and 3-grams.
Numeric-only, symbolic, and mixed alphanumeric fingerprints are not measured by that check; the
`shape_entity_shortcut` and `pervasive_phrase_fingerprint` checks cover identifier-shaped tokens
and placeholder-inclusive 4-grams respectively.

## Non-goals

PIIcorpus does not provide real issuing formats, data collection, data scraping, de-identification,
regulatory interpretation, model training, model distribution, detector approval, or publication
approval.

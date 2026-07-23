# Claim boundaries

PIIcorpus measures properties of corpus files. It does not measure a detector or establish that a
dataset represents the world outside its generator.

The following boundaries apply to every command and report:

- Synthetic data does not prove real-world accuracy.
- A same-generator holdout is not independent.
- Diversity counts do not prove semantic diversity.
- Template variation can still leave a generator fingerprint.
- Generated identifiers are not guaranteed to be realistic.
- Regulatory compliance is not guaranteed.
- Real data is not de-identified by this project.
- No model becomes ready for deployment by using this project.
- Human-authored or externally sourced evaluation remains necessary.
- The project generates and audits corpora; it does not train or approve models.

`PASS` means only that an implemented risk check found no violation at its configured threshold.
It does not mean that the risk is absent outside the measured files. `UNMEASURED` is used when the
artifact cannot support a conclusion, including independent generalization from a same-generator
holdout.

The lexical generator-fingerprint check measures alphabetic 1-grams, 2-grams, and 3-grams.
Numeric-only, symbolic, and mixed alphanumeric fingerprints are not measured by that check.

## Non-goals

PIIcorpus does not provide real issuing formats, data collection, data scraping, de-identification,
regulatory interpretation, model training, model distribution, detector approval, or publication
approval.

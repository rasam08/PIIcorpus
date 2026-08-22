# Generated reserved-surface demo

`corpus/` is generated from `configs/realistic-safe.toml` with PIIcorpus 0.2.1. Annotated values
use contact and identity formats constrained to reserved, fictional, or invalid ranges: RFC 2606
email domains, the reserved NANP 555-01XX exchange,
Luhn-invalid card shapes, RFC 5737/3849 documentation IP ranges, and never-issued 9XX-XX-XXXX
national-id shapes. Safety runs in verifier mode, so no synthetic prefix is required; see
`docs/DATA_SAFETY.md` for each plugin's validation rule.

The configuration enables the learnability probe. The shipped corpus passes the configured audit
and probe thresholds and does not produce a `value_shared_affix` warning.

Regenerate and byte-compare in one command:

```console
piicorpus reproduce examples/realistic/corpus
```

The included same-generator holdout supports regression testing only and is not an independent
generalization test.

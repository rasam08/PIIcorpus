# Generated demo

`corpus/` is generated from `configs/demo.toml` with PIIcorpus 0.2.1. The corpus manifest records
its seed, generator version, normalized configuration digest, counts, SHA-256 values, and CC0-1.0
generated-data license.

All personas, organizations, sentence frames, and values are synthetic. Identifier shapes do not
represent formats from medical, passport, driver-license, or other issuing authorities. The
constant `SYN-` prefix exceeds the configured shared-affix threshold, so the audit reports
`value_shared_affix` as `WARN`.

Regenerate and byte-compare in one command:

```console
piicorpus reproduce examples/demo/corpus
```

The included same-generator holdout supports regression testing only and is not an independent
generalization test.

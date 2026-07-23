# Generated demo

`corpus/` is generated from `configs/demo.toml` with PIIcorpus 0.2.0. The corpus manifest records
its seed, generator version, normalized configuration digest, counts, SHA-256 values, and CC0-1.0
generated-data license.

All personas, organizations, sentence frames, and values are newly authored synthetic material.
Identifier shapes are intentionally fictional and do not represent formats from medical,
passport, driver-license, or other issuing authorities. The audit deliberately reports the
constant `SYN-` value prefix as a `value_shared_affix` warning; the realistic-safe example shows
the same engine without that trade-off.

Regenerate and byte-compare in one command:

```console
piicorpus reproduce examples/demo/corpus
```

The included same-generator holdout supports regression testing only and is not an independent
generalization test.

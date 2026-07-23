# Generated demo

`corpus/` is generated from `configs/demo.toml` with PIIcorpus 0.1.0. The corpus manifest records
its seed, generator version, normalized configuration digest, counts, SHA-256 values, and CC0-1.0
generated-data license.

All personas, organizations, sentence frames, and values are newly authored synthetic material.
Identifier shapes are intentionally fictional and do not represent formats from medical,
passport, driver-license, or other issuing authorities.

Regenerate into an empty directory and compare bytes:

```console
piicorpus generate --config configs/demo.toml --out regenerated-demo
```

The included same-generator holdout supports regression testing only and is not an independent
generalization test.

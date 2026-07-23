# Deliberately defective examples

These cases are safe synthetic mutations designed to demonstrate one named audit risk each. They
are configurations for generating defects, not evaluation evidence. Build them from a fresh demo:

```console
python examples/deliberately_bad/build_examples.py examples/demo/corpus --out .bad-corpora
piicorpus audit .bad-corpora/value_contamination --format text
```

`cases.toml` maps every case to the exact finding that must be `FAIL`. Tests assert the named
finding directly, so an unrelated validation failure cannot make a case pass.

Some examples intentionally violate strict integrity checks. Inspect those with
`--forensic-allow-invalid`; the audit will remain failed and mark the measurements
non-authoritative.

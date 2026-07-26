# Deliberately defective examples

These cases are safe synthetic mutations designed to demonstrate one named audit risk each. They
are configurations for generating defects, not evaluation evidence. Build them from a fresh demo:

```console
python examples/deliberately_bad/build_examples.py examples/demo/corpus --out .bad-corpora
piicorpus audit .bad-corpora/value_contamination --format text
```

`cases.toml` maps every case to the exact finding that must be `FAIL`. Tests assert the named
finding directly, so an unrelated validation failure cannot make a case pass. The builder writes
every mutation through the normal corpus writer, regenerating manifest hashes, byte counts, and
derived counts; tests verify those digests and counts for every case.

`corpus_integrity` covers strict structural and semantic invariants in addition to file signatures.
Some examples intentionally violate those invariants by definition, such as cross-split
contamination, malformed spans, or unsafe values. Inspect those with `--forensic-allow-invalid`;
the audit will remain failed and mark the measurements non-authoritative.

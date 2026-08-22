# Audit failure examples

These cases apply synthetic mutations associated with named audit risks. They are test fixtures,
not evaluation datasets. Build them from a generated demo corpus:

```console
python examples/deliberately_bad/build_examples.py examples/demo/corpus --out .bad-corpora
piicorpus audit .bad-corpora/value_contamination --format text
```

`cases.toml` maps every case to the exact finding that must be `FAIL`. Tests assert the named
finding directly, so an unrelated validation failure cannot make a case pass. The builder writes
every mutation through the normal corpus writer, regenerating manifest hashes, byte counts, and
derived counts; tests verify those digests and counts for every case.

`corpus_integrity` covers strict structural and semantic invariants in addition to file signatures.
Cases for cross-split contamination, malformed spans, or unsafe values violate those invariants.
`--forensic-allow-invalid` continues the audit for diagnostic output while retaining a failed,
non-authoritative result.

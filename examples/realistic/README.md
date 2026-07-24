# Generated realistic-safe demo

`corpus/` is generated from `configs/realistic-safe.toml` with PIIcorpus 0.2.1. Annotated values
look like genuine contact and identity surface forms while being provably reserved, fictional, or
invalid by construction: RFC 2606 email domains, the reserved NANP 555-01XX exchange,
Luhn-invalid card shapes, RFC 5737/3849 documentation IP ranges, and never-issued 9XX-XX-XXXX
national-id shapes. Safety runs in verifier mode, so no synthetic prefix is required; see
`docs/DATA_SAFETY.md` for the reservation rationale behind every plugin.

The configuration enables the learnability probe, and the shipped corpus passes the full audit —
including the probe ceilings — with no `value_shared_affix` warning, in deliberate contrast with
the SYN- demo.

Regenerate and byte-compare in one command:

```console
piicorpus reproduce examples/realistic/corpus
```

The included same-generator holdout supports regression testing only and is not an independent
generalization test.

# Corpus format

Each split is UTF-8 JSONL with one object per line. Keys are serialized in stable sorted order and
files use LF newlines.

## Record fields

- `schema_version`: public record schema identifier.
- `case_id`: stable content-, namespace-, generator-, and configuration-derived ID. Strict
  validation recomputes it.
- `split`: `train`, `eval`, or `holdout` for generated records.
- `family`: configured corpus family.
- `namespace`: unique split/family/index namespace.
- `template_id`: split-isolated template identifier.
- `kind`: `positive` or `hard_negative`.
- `provenance`: `generated` or `human_supplied`.
- `text`: clean text with no annotation markup.
- `annotations`: exact entity spans.
- `cue_links`: explicit `{cue, entity_type}` relationships backed by the emitted text and spans.
- `persona`, `organization`, `cue_surface`: optional synthetic generation metadata. Any declared
  persona or organization must occur literally in `text`.
- `hard_negative_kind`: explicit negative category where applicable.
- `metadata`: family-specific, non-authoritative metadata.

Every annotation contains `entity_type`, `text`, Unicode code-point `start` and `end`, and UTF-8
`byte_start` and `byte_end`. Offsets are half-open. Validators and exporters compare the stored text
against both representations.

## Marked-text input

```text
The record identifier is [[PATIENT_RECORD_ID:SYN-ID-A10427]].
```

Markup is removed from `text`. Malformed, nested, unclosed, empty, or overlapping annotations are
rejected.

## Export forms

- `jsonl`: the generic public record objects.
- `bio`: token and BIO tag lines, with record separators.
- `huggingface`: text, tokens, token offsets, string tags, and source spans in JSONL.
- `spacy`: JSONL objects with `[start, end, label]` entities; convertible without importing spaCy.
- `presidio`: fixture objects with expected entity type, text, start, and end.

An exporter runs strict corpus validation before creating output, then fails if tokenization would
cut through an entity or if any source span does not round trip. Audit follows the same fail-closed
integrity rule. The deliberately loud `--forensic-allow-invalid` override permits investigation,
but the command still exits with findings and labels its output non-authoritative. Heavy frameworks
are not base dependencies.

## Manifest

`manifest.json` contains schema and generator versions, normalized configuration digest, seed,
counts by split/family/label, positive and negative counts, diversity counts, file SHA-256 values,
the generated-data license, determinism metadata, and the same-generator holdout limitation.

The validator recalculates these properties from emitted files rather than accepting manifest
claims as evidence.

# Corpus format

Each split is UTF-8 JSONL with one object per line. Keys are serialized in stable sorted order and
files use LF newlines.

## Record fields

- `schema_version`: public record schema identifier.
- `case_id`: stable content-, namespace-, generator-, and configuration-derived ID. Strict
  validation recomputes it.
- `split`: `train`, `eval`, or `holdout` for generated records; `unassigned` for imports.
- `family`: configured corpus family (`external` for loaded external data).
- `namespace`: unique split/family/index namespace.
- `kind`: `positive` or `hard_negative` for generated records; `unannotated` for imported or
  external records that carry no spans (an absent annotation is not a curated hard negative).
- `provenance`: `generated`, `human_supplied`, or `external`.
- `template_id`: split-isolated template identifier.
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
rejected, as is text that cannot carry UTF-8 byte offsets (lone surrogates).

## Export forms

- `jsonl`: the generic public record objects.
- `bio`: token and BIO tag lines, with record separators.
- `huggingface`: one file per split (`train.jsonl`, `eval.jsonl`, `holdout.jsonl`) with text,
  tokens, token offsets, string tags, and source spans, loadable directly through a `datasets`
  split mapping.
- `spacy`: JSONL objects with `[start, end, label]` entities; convertible without importing spaCy.
- `presidio`: fixture objects with expected entity type, text, start, and end.

Every export also writes `labels.json` containing the sorted label list and the full BIO tag list,
which training pipelines need and should not have to reconstruct.

An exporter runs strict corpus validation before creating output, then fails if tokenization would
cut through an entity or if any source span does not round trip. Audit follows the same fail-closed
integrity rule. The deliberately loud `--forensic-allow-invalid` override permits investigation,
but the command still exits with findings and labels its output non-authoritative. Heavy frameworks
are not base dependencies.

## External input formats (`audit-external`)

- `jsonl`: `{"text": ..., "spans": [{"start", "end", "entity_type"}], "split"?: ...}`;
  `annotations` and `label` are accepted as synonyms, and `--byte-offsets` declares byte-based
  span positions. The import command's `records.jsonl` is directly consumable.
- `hf`: `{"tokens": [...], "ner_tags": ["B-X", ...], "text"?, "token_offsets"?, "split"?}` —
  the shape written by `piicorpus export --format huggingface`. String tags are required because
  integer tag ids are ambiguous without their name table.
- `conll`: token-per-line blocks (token first column, tag last column) separated by blank lines;
  `-DOCSTART-` lines are ignored.

## Prediction format (`score`)

One JSON object per line: `{"id": "<case_id>", "spans": [{"start", "end", "entity_type"}]}`.
Offsets are Unicode code points on the record's `text` (`--byte-offsets` for UTF-8 byte
positions). Records absent from the predictions file are scored as zero-span predictions and
counted; `--allow-partial` restricts scoring to the records present instead. `--match strict`
requires exact boundaries and label; `--match overlap` accepts intersection-over-union of at
least 0.5 with a matching label.

Prediction input is strict by default: offsets must be integers with
`0 <= start < end <= len(text)`, labels must be configured for the corpus, and exact duplicate or
overlapping prediction spans are rejected. `--allow-invalid-predictions` preserves malformed
code-point spans for forensic scoring, where they normally count as false positives. Byte offsets
must still fall on UTF-8 character boundaries because otherwise they cannot be converted to code
points.

## Manifest

`manifest.json` contains schema and generator versions, normalized configuration digest, seed,
counts by split/family/label, positive and negative counts, diversity counts, file SHA-256 values,
the generated-data license, determinism metadata, and the same-generator holdout limitation.

The validator recalculates these properties from emitted files rather than accepting manifest
claims as evidence. `piicorpus reproduce` goes further and regenerates the corpus from its own
configuration snapshot, byte-comparing every file.

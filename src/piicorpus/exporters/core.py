"""JSONL, BIO, Hugging Face, spaCy, and Presidio exporters."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from ..annotation import AnnotationError, validate_annotations
from ..manifest import load_corpus
from ..models import Record, stable_json
from ..validators import CorpusIntegrityError, validate_corpus

EXPORT_FORMATS = ("jsonl", "bio", "huggingface", "spacy", "presidio")
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class ExportError(ValueError):
    """Raised when an exporter cannot preserve the source spans exactly."""


def _tokenize(record: Record) -> list[tuple[str, int, int, str]]:
    try:
        validate_annotations(record.text, record.annotations)
    except AnnotationError as exc:
        raise ExportError(f"record {record.case_id} has invalid source spans") from exc
    tokens: list[tuple[str, int, int, str]] = []
    for match in TOKEN_RE.finditer(record.text):
        start, end = match.span()
        overlapping = [a for a in record.annotations if start < a.end and end > a.start]
        if len(overlapping) > 1:
            raise ExportError(f"record {record.case_id} has overlapping entities")
        if overlapping:
            annotation = overlapping[0]
            if not (annotation.start <= start and end <= annotation.end):
                raise ExportError(
                    f"record {record.case_id} has an entity that cuts through a token"
                )
            previous_same = any(
                prior[3].endswith(annotation.entity_type)
                and annotation.start <= prior[1]
                and prior[2] <= annotation.end
                for prior in tokens
            )
            tag = ("I-" if previous_same else "B-") + annotation.entity_type
        else:
            tag = "O"
        tokens.append((match.group(0), start, end, tag))
    for annotation in record.annotations:
        covered = "".join(
            record.text[start:end]
            for _, start, end, tag in tokens
            if tag != "O" and annotation.start <= start and end <= annotation.end
        )
        expected = re.sub(r"\s+", "", annotation.text)
        if re.sub(r"\s+", "", covered) != expected:
            raise ExportError(f"record {record.case_id} entity was not preserved by tokenization")
    return tokens


def _spans(record: Record) -> list[dict[str, Any]]:
    return [
        {
            "byte_end": annotation.byte_end,
            "byte_start": annotation.byte_start,
            "end": annotation.end,
            "entity_type": annotation.entity_type,
            "start": annotation.start,
            "text": annotation.text,
        }
        for annotation in record.annotations
    ]


def _all_records(directory: str | Path) -> list[Record]:
    _config, split_records, _manifest = load_corpus(directory)
    return [record for split in ("train", "eval", "holdout") for record in split_records[split]]


def export_corpus(
    directory: str | Path,
    format_name: str,
    output: str | Path | None = None,
    *,
    allow_invalid: bool = False,
) -> dict[str, Any]:
    if format_name not in EXPORT_FORMATS:
        raise ExportError(f"unsupported export format: {format_name}")
    root = Path(directory)
    validation = validate_corpus(root, strict=True)
    if not validation.valid and not allow_invalid:
        raise CorpusIntegrityError(validation)
    destination = Path(output) if output else root / "exports" / format_name
    destination.mkdir(parents=True, exist_ok=True)
    records = _all_records(root)

    if format_name == "jsonl":
        path = destination / "corpus.jsonl"
        payload = "".join(stable_json(record.to_dict()) + "\n" for record in records)
    elif format_name == "bio":
        path = destination / "corpus.bio"
        blocks = []
        for record in records:
            lines = [f"# id={record.case_id} split={record.split}"]
            lines.extend(f"{token}\t{tag}" for token, _start, _end, tag in _tokenize(record))
            blocks.append("\n".join(lines))
        payload = "\n\n".join(blocks) + "\n"
    elif format_name == "huggingface":
        path = destination / "corpus.jsonl"
        output_rows = []
        for record in records:
            tokens = _tokenize(record)
            output_rows.append(
                {
                    "id": record.case_id,
                    "ner_tags": [token[3] for token in tokens],
                    "spans": _spans(record),
                    "split": record.split,
                    "text": record.text,
                    "token_offsets": [[token[1], token[2]] for token in tokens],
                    "tokens": [token[0] for token in tokens],
                }
            )
        payload = "".join(stable_json(row) + "\n" for row in output_rows)
    elif format_name == "spacy":
        path = destination / "corpus.spacy.jsonl"
        output_rows = [
            {
                "entities": [[a.start, a.end, a.entity_type] for a in record.annotations],
                "id": record.case_id,
                "split": record.split,
                "text": record.text,
            }
            for record in records
        ]
        payload = "".join(stable_json(row) + "\n" for row in output_rows)
    else:
        path = destination / "presidio-fixtures.jsonl"
        output_rows = [
            {
                "expected": [
                    {
                        "end": a.end,
                        "entity_type": a.entity_type,
                        "start": a.start,
                        "text": a.text,
                    }
                    for a in record.annotations
                ],
                "name": record.case_id,
                "text": record.text,
            }
            for record in records
        ]
        payload = "".join(stable_json(row) + "\n" for row in output_rows)

    path.write_text(payload, encoding="utf-8", newline="\n")
    return {
        "format": format_name,
        "integrity_valid": validation.valid,
        "path": str(path),
        "records": len(records),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }

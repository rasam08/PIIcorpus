"""Import explicitly marked, user-supplied text without safety claims or implicit mixing."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ..annotation import AnnotationError, parse_marked
from ..models import Record, stable_json


class ImportErrorSafe(ValueError):
    """An importer error whose normal message does not expose record bodies."""


RESPONSIBILITY_WARNING = (
    "Imported text is user-supplied. You are responsible for consent, privacy, provenance, "
    "licensing, and release decisions. Import does not establish that data is safe or releasable."
)


def _read_entries(path: Path) -> list[tuple[str, str]]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ImportErrorSafe(f"cannot read import input: {exc}") from exc
    lines = [line for line in raw.splitlines() if line.strip()]
    if not lines:
        raise ImportErrorSafe("input contains no non-empty records")
    if lines[0].lstrip().startswith("{"):
        entries: list[tuple[str, str]] = []
        for number, line in enumerate(lines, 1):
            try:
                value = json.loads(line)
                if not isinstance(value, dict) or not isinstance(value.get("text"), str):
                    raise TypeError("expected an object with a text string")
                entries.append((str(value.get("family", "human_supplied")), value["text"]))
            except (json.JSONDecodeError, TypeError) as exc:
                raise ImportErrorSafe(f"invalid JSONL structure at record {number}") from exc
        return entries
    return [("human_supplied", line) for line in lines]


def import_annotated(
    input_path: str | Path,
    output: str | Path,
    *,
    debug_local: bool = False,
) -> dict[str, Any]:
    source = Path(input_path)
    destination = Path(output)
    if destination.exists() and any(destination.iterdir()):
        raise ImportErrorSafe("import output directory is not empty")
    entries = _read_entries(source)
    records: list[Record] = []
    for index, (family, marked) in enumerate(entries, 1):
        try:
            clean, annotations = parse_marked(marked)
        except AnnotationError as exc:
            message = f"record {index} has invalid annotations: {exc}"
            if debug_local:
                message += f"; local body={marked!r}"
            raise ImportErrorSafe(message) from exc
        material = stable_json(
            {
                "annotations": [annotation.to_dict() for annotation in annotations],
                "family": family,
                "text": clean,
            }
        )
        case_id = "human-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]
        records.append(
            Record(
                case_id=case_id,
                split="unassigned",
                family=family,
                namespace=f"piicorpus/import/{case_id}",
                template_id="human_supplied",
                kind="positive" if annotations else "hard_negative",
                provenance="human_supplied",
                text=clean,
                annotations=annotations,
                hard_negative_kind=None if annotations else "human_supplied_unmarked",
                metadata={"release_status": "unreviewed"},
            )
        )
    if len({record.case_id for record in records}) != len(records):
        raise ImportErrorSafe("input contains duplicate records")

    destination.mkdir(parents=True, exist_ok=True)
    records_path = destination / "records.jsonl"
    records_path.write_text(
        "".join(stable_json(record.to_dict()) + "\n" for record in records),
        encoding="utf-8",
        newline="\n",
    )
    manifest = {
        "labels": dict(
            sorted(Counter(a.entity_type for r in records for a in r.annotations).items())
        ),
        "mixing_policy": "not assigned to generated train, eval, or holdout splits",
        "provenance": "human_supplied",
        "records": len(records),
        "release_status": "unreviewed",
        "responsibility_warning": RESPONSIBILITY_WARNING,
        "schema_version": "piicorpus.import/v1",
        "sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
    }
    (destination / "import-manifest.json").write_text(
        stable_json(manifest, pretty=True), encoding="utf-8", newline="\n"
    )
    return manifest

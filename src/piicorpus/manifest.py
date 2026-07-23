"""Corpus I/O, file hashing, and deterministic manifest construction."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .config import CorpusConfig, config_from_dict
from .models import MANIFEST_SCHEMA_VERSION, Record, stable_json

SYNTHETIC_HOLDOUT_LIMITATION = (
    "A holdout produced by the same generator is useful for regression testing but is not an "
    "independent generalization test."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path.name}")
    return value


def load_records(path: Path) -> list[Record]:
    records: list[Record] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError("record is not an object")
                records.append(Record.from_dict(value))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid JSONL record at {path.name}:{line_number}: {exc}"
                ) from exc
    return records


def load_corpus(
    directory: str | Path,
) -> tuple[CorpusConfig, dict[str, list[Record]], dict[str, Any]]:
    root = Path(directory)
    manifest = load_json(root / "manifest.json")
    config_value = load_json(root / "corpus-config.json")
    config = config_from_dict(config_value)
    records = {
        split: load_records(root / "splits" / f"{split}.jsonl")
        for split in ("train", "eval", "holdout")
    }
    return config, records, manifest


def _counts(records: Iterable[Record]) -> dict[str, Any]:
    rows = list(records)
    labels = Counter(a.entity_type for record in rows for a in record.annotations)
    return {
        "families": dict(sorted(Counter(r.family for r in rows).items())),
        "hard_negative_kinds": dict(
            sorted(Counter(r.hard_negative_kind for r in rows if r.hard_negative_kind).items())
        ),
        "labels": dict(sorted(labels.items())),
        "negatives": sum(r.kind == "hard_negative" for r in rows),
        "positives": sum(r.kind == "positive" for r in rows),
        "records": len(rows),
        "templates": len({r.template_id for r in rows}),
        "personas": len({r.persona for r in rows if r.persona}),
        "organizations": len({r.organization for r in rows if r.organization}),
    }


def write_corpus(
    directory: str | Path,
    config: CorpusConfig,
    records: dict[str, list[Record]],
    *,
    generator_version: str,
) -> dict[str, Any]:
    root = Path(directory)
    split_dir = root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    config_path = root / "corpus-config.json"
    config_path.write_text(
        stable_json(config.to_dict(), pretty=True), encoding="utf-8", newline="\n"
    )

    files: dict[str, dict[str, Any]] = {
        "corpus-config.json": {
            "sha256": sha256_file(config_path),
            "bytes": config_path.stat().st_size,
        }
    }
    for split in ("train", "eval", "holdout"):
        path = split_dir / f"{split}.jsonl"
        payload = "".join(stable_json(record.to_dict()) + "\n" for record in records[split])
        path.write_text(payload, encoding="utf-8", newline="\n")
        files[f"splits/{split}.jsonl"] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }

    manifest = {
        "configuration_digest": config.digest,
        "counts": {split: _counts(records[split]) for split in ("train", "eval", "holdout")},
        "determinism": {
            "json_serialization": "UTF-8, sorted keys, compact JSONL, LF newlines",
            "ordering": "split order and configured family order are stable",
            "randomness": "SHA-256-namespaced pseudorandom streams",
        },
        "files": files,
        "generated_data_license": config.generated_data_license,
        "generator_version": generator_version,
        "project": "PIIcorpus",
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "seed": config.seed,
        "synthetic_holdout_limitation": SYNTHETIC_HOLDOUT_LIMITATION,
    }
    (root / "manifest.json").write_text(
        stable_json(manifest, pretty=True), encoding="utf-8", newline="\n"
    )
    return manifest

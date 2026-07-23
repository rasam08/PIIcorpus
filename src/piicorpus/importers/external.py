"""Load external NER datasets into auditable records without provenance claims.

Supported formats:

- ``jsonl``: one object per line with ``text``, optional ``spans`` (or
  ``annotations``) carrying ``start``/``end``/``entity_type`` (``label`` is also
  accepted), and an optional ``split`` name.
- ``hf``: one object per line with ``tokens`` and string BIO/BILOU ``ner_tags``,
  plus optional ``text``, ``token_offsets``, and ``split``. This matches the
  Hugging Face export written by ``piicorpus export --format huggingface``.
- ``conll``: token-per-line blocks (token first column, tag last column)
  separated by blank lines; ``-DOCSTART-`` lines are ignored.

Loaded records carry ``provenance="external"`` and are auditable with
``piicorpus audit-external``; loading makes no consent, licensing, safety, or
release claims about the data.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..annotation import AnnotationError, validate_annotations
from ..models import Annotation, Record


class ExternalImportError(ValueError):
    """Raised when external data cannot be converted into auditable records."""


def _annotation(text: str, start: int, end: int, entity_type: str) -> Annotation:
    if not 0 <= start < end <= len(text):
        raise AnnotationError(f"span [{start}, {end}) is outside the record text")
    return Annotation(
        entity_type=entity_type,
        start=start,
        end=end,
        byte_start=len(text[:start].encode("utf-8")),
        byte_end=len(text[:end].encode("utf-8")),
        text=text[start:end],
    )


def _codepoints_by_byte(text: str) -> dict[int, int]:
    mapping: dict[int, int] = {}
    byte = 0
    for index, char in enumerate(text):
        mapping[byte] = index
        byte += len(char.encode("utf-8"))
    mapping[byte] = len(text)
    return mapping


def _record(split: str, index: int, text: str, annotations: tuple[Annotation, ...]) -> Record:
    material = f"{split}|{index}|{text}"
    case_id = "ext-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]
    return Record(
        case_id=case_id,
        split=split,
        family="external",
        namespace=f"piicorpus/external/{split}/{index:05d}",
        template_id="external",
        kind="positive" if annotations else "unannotated",
        provenance="external",
        text=text,
        annotations=annotations,
        metadata={"release_status": "unreviewed"},
    )


def _spans_from_bio(
    tokens: list[str],
    tags: list[str],
    text: str,
    offsets: list[tuple[int, int]],
) -> tuple[Annotation, ...]:
    if not len(tokens) == len(tags) == len(offsets):
        raise AnnotationError("tokens, tags, and offsets disagree in length")
    annotations: list[Annotation] = []
    open_label: str | None = None
    open_start = 0
    open_end = 0

    def close() -> None:
        nonlocal open_label
        if open_label is not None:
            annotations.append(_annotation(text, open_start, open_end, open_label))
            open_label = None

    for tag, (start, end) in zip(tags, offsets, strict=True):
        if tag == "O" or not tag:
            close()
            continue
        if "-" not in tag:
            raise AnnotationError(f"unsupported tag format: {tag}")
        marker, label = tag.split("-", 1)
        if marker in {"B", "U"} or open_label != label:
            close()
            open_label = label
            open_start = start
        elif marker not in {"I", "L"}:
            raise AnnotationError(f"unsupported tag format: {tag}")
        open_end = end
    close()
    return tuple(annotations)


def _read_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ExternalImportError(f"cannot read external input: {exc}") from exc


def _load_jsonl(
    path: Path, *, byte_offsets: bool, default_split: str
) -> list[tuple[str, str, tuple[Annotation, ...]]]:
    entries: list[tuple[str, str, tuple[Annotation, ...]]] = []
    for number, line in enumerate(_read_lines(path), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict) or not isinstance(value.get("text"), str):
                raise TypeError("expected an object with a text string")
            text = value["text"]
            split = str(value.get("split", default_split))
            raw_spans = value.get("spans", value.get("annotations", []))
            if not isinstance(raw_spans, list):
                raise TypeError("spans must be an array")
            byte_map = _codepoints_by_byte(text) if byte_offsets else None
            annotations = []
            for raw in raw_spans:
                start, end = int(raw["start"]), int(raw["end"])
                if byte_map is not None:
                    if start not in byte_map or end not in byte_map:
                        raise AnnotationError(
                            f"byte offsets [{start}, {end}) do not fall on character "
                            "boundaries"
                        )
                    start, end = byte_map[start], byte_map[end]
                label = str(raw.get("entity_type", raw.get("label", "")))
                if not label:
                    raise AnnotationError("span lacks an entity_type")
                annotations.append(_annotation(text, start, end, label))
            ordered = tuple(sorted(annotations, key=lambda a: (a.start, a.end)))
            validate_annotations(text, ordered)
            entries.append((split, text, ordered))
        except (
            json.JSONDecodeError,
            TypeError,
            KeyError,
            ValueError,
            AnnotationError,
        ) as exc:
            raise ExternalImportError(
                f"invalid external record at {path.name}:{number}: {exc}"
            ) from exc
    return entries


def _load_hf(
    path: Path, *, default_split: str
) -> list[tuple[str, str, tuple[Annotation, ...]]]:
    entries: list[tuple[str, str, tuple[Annotation, ...]]] = []
    for number, line in enumerate(_read_lines(path), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError("expected an object per line")
            tokens = [str(token) for token in value["tokens"]]
            tags = value["ner_tags"]
            if any(not isinstance(tag, str) for tag in tags):
                raise TypeError(
                    "integer ner_tags are ambiguous; export with string BIO tags"
                )
            split = str(value.get("split", default_split))
            if isinstance(value.get("text"), str) and isinstance(
                value.get("token_offsets"), list
            ):
                text = value["text"]
                offsets = [(int(pair[0]), int(pair[1])) for pair in value["token_offsets"]]
            else:
                text = " ".join(tokens)
                offsets = []
                position = 0
                for token in tokens:
                    offsets.append((position, position + len(token)))
                    position += len(token) + 1
            annotations = _spans_from_bio(tokens, [str(tag) for tag in tags], text, offsets)
            validate_annotations(text, annotations)
            entries.append((split, text, annotations))
        except (
            json.JSONDecodeError,
            TypeError,
            KeyError,
            ValueError,
            AnnotationError,
        ) as exc:
            raise ExternalImportError(
                f"invalid external record at {path.name}:{number}: {exc}"
            ) from exc
    return entries


def _load_conll(
    path: Path, *, default_split: str
) -> list[tuple[str, str, tuple[Annotation, ...]]]:
    entries: list[tuple[str, str, tuple[Annotation, ...]]] = []
    tokens: list[str] = []
    tags: list[str] = []

    def close_block(number: int) -> None:
        if not tokens:
            return
        text = " ".join(tokens)
        offsets = []
        position = 0
        for token in tokens:
            offsets.append((position, position + len(token)))
            position += len(token) + 1
        try:
            annotations = _spans_from_bio(tokens, tags, text, offsets)
            validate_annotations(text, annotations)
        except AnnotationError as exc:
            raise ExternalImportError(
                f"invalid external block ending at {path.name}:{number}: {exc}"
            ) from exc
        entries.append((default_split, text, annotations))
        tokens.clear()
        tags.clear()

    for number, line in enumerate(_read_lines(path), 1):
        stripped = line.strip()
        if not stripped:
            close_block(number)
            continue
        if stripped.startswith("-DOCSTART-"):
            continue
        columns = stripped.split()
        if len(columns) < 2:
            raise ExternalImportError(
                f"invalid external token line at {path.name}:{number}: "
                "expected token and tag columns"
            )
        tokens.append(columns[0])
        tags.append(columns[-1])
    close_block(len(_read_lines(path)) + 1)
    return entries


def load_external(
    sources: dict[str, Path],
    format_name: str,
    *,
    byte_offsets: bool = False,
) -> dict[str, list[Record]]:
    """Load ``{split_name: path}`` sources into records grouped by split.

    For ``jsonl`` and ``hf`` inputs a record-level ``split`` field overrides the
    source's split name, so a single file carrying its own split labels can be
    passed as one source.
    """
    if format_name not in {"jsonl", "hf", "conll"}:
        raise ExternalImportError(f"unsupported external format: {format_name}")
    if byte_offsets and format_name != "jsonl":
        raise ExternalImportError("--byte-offsets applies only to the jsonl format")
    grouped: dict[str, list[Record]] = {}
    for split_name, path in sources.items():
        if format_name == "jsonl":
            entries = _load_jsonl(Path(path), byte_offsets=byte_offsets, default_split=split_name)
        elif format_name == "hf":
            entries = _load_hf(Path(path), default_split=split_name)
        else:
            entries = _load_conll(Path(path), default_split=split_name)
        for split, text, spans in entries:
            index = sum(len(rows) for rows in grouped.values())
            grouped.setdefault(split, []).append(_record(split, index, text, spans))
    if not any(grouped.values()):
        raise ExternalImportError("external input contains no records")
    return grouped

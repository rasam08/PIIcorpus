"""Stable public record models and JSON helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

RECORD_SCHEMA_VERSION = "piicorpus.record/v1"
MANIFEST_SCHEMA_VERSION = "piicorpus.manifest/v1"


def stable_json(value: Any, *, pretty: bool = False) -> str:
    """Serialize JSON deterministically without platform-dependent escaping."""
    if pretty:
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


@dataclass(frozen=True, slots=True)
class Annotation:
    entity_type: str
    start: int
    end: int
    byte_start: int
    byte_end: int
    text: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Annotation:
        return cls(
            entity_type=str(value["entity_type"]),
            start=int(value["start"]),
            end=int(value["end"]),
            byte_start=int(value["byte_start"]),
            byte_end=int(value["byte_end"]),
            text=str(value["text"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CueLink:
    """An explicit relationship between a rendered cue and one entity label."""

    cue: str
    entity_type: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CueLink:
        return cls(cue=str(value["cue"]), entity_type=str(value["entity_type"]))

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Record:
    case_id: str
    split: str
    family: str
    namespace: str
    template_id: str
    kind: str
    provenance: str
    text: str
    annotations: tuple[Annotation, ...] = field(default_factory=tuple)
    cue_links: tuple[CueLink, ...] = field(default_factory=tuple)
    persona: str | None = None
    organization: str | None = None
    cue_surface: str | None = None
    hard_negative_kind: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = RECORD_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Record:
        return cls(
            case_id=str(value["case_id"]),
            split=str(value["split"]),
            family=str(value["family"]),
            namespace=str(value["namespace"]),
            template_id=str(value["template_id"]),
            kind=str(value["kind"]),
            provenance=str(value["provenance"]),
            text=str(value["text"]),
            annotations=tuple(Annotation.from_dict(v) for v in value.get("annotations", [])),
            cue_links=tuple(CueLink.from_dict(v) for v in value.get("cue_links", [])),
            persona=value.get("persona"),
            organization=value.get("organization"),
            cue_surface=value.get("cue_surface"),
            hard_negative_kind=value.get("hard_negative_kind"),
            metadata=dict(value.get("metadata", {})),
            schema_version=str(value.get("schema_version", RECORD_SCHEMA_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "annotations": [a.to_dict() for a in self.annotations],
            "case_id": self.case_id,
            "cue_links": [link.to_dict() for link in self.cue_links],
            "cue_surface": self.cue_surface,
            "family": self.family,
            "hard_negative_kind": self.hard_negative_kind,
            "kind": self.kind,
            "metadata": self.metadata,
            "namespace": self.namespace,
            "organization": self.organization,
            "persona": self.persona,
            "provenance": self.provenance,
            "schema_version": self.schema_version,
            "split": self.split,
            "template_id": self.template_id,
            "text": self.text,
        }


FINDING_STATUSES = ("PASS", "FAIL", "WARN", "UNMEASURED")


@dataclass(frozen=True, slots=True)
class Finding:
    risk: str
    status: str
    count: int | None
    reason: str
    details: dict[str, Any] = field(default_factory=dict)
    measured: float | int | None = None
    threshold: float | int | None = None
    threshold_source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

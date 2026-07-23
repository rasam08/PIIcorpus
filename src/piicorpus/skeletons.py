"""Extensible sentence-family registry with newly authored public demo frames."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FamilyPlugin:
    name: str
    role: str
    templates: tuple[str, ...]


_REGISTRY: dict[str, FamilyPlugin] = {}


def register_family(plugin: FamilyPlugin, *, replace: bool = False) -> None:
    if plugin.name in _REGISTRY and not replace:
        raise ValueError(f"family plugin is already registered: {plugin.name}")
    if plugin.role not in {"positive", "hard_negative"}:
        raise ValueError("family plugin role must be positive or hard_negative")
    _REGISTRY[plugin.name] = plugin


def get_family(name: str) -> FamilyPlugin:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"unknown family plugin: {name}") from exc


def registered_families() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def _add(name: str, role: str, *templates: str) -> None:
    register_family(FamilyPlugin(name=name, role=role, templates=tuple(templates)))


_add(
    "narrative",
    "positive",
    "{persona} asked {organization} to correct the {cue} shown as {value}.",
    "During a routine review, {persona} confirmed that the {cue} is {value}.",
    "The note from {organization} associates {persona} with {cue} {value}.",
    "For the fictional case, {persona} supplied {value} as the {cue}.",
    "A message to {organization} says the {cue} for {persona} should read {value}.",
    "In the synthetic narrative, {persona}'s {cue} appears as {value}.",
)
_add(
    "structural",
    "positive",
    "person={persona} | organization={organization} | {cue}={value}",
    "{cue}\t{value}\towner\t{persona}",
    '{{"subject": "{persona}", "{cue}": "{value}"}}',
    "| field | content |\n| {cue} | {value} |\n| subject | {persona} |",
    "record_owner: {persona}\nrecord_source: {organization}\n{cue}: {value}",
    "COLUMN {cue} ; VALUE {value} ; SYNTHETIC SUBJECT {persona}",
)
_add(
    "ocr_noise",
    "positive",
    "Scanned line: {persona} / {cue} / {value}",
    "OCR extract from {organization}: {cue} reads {value} for {persona}.",
    "Low-quality form transcription: {cue}: {value}; name: {persona}.",
    "Page fragment for {persona} -- {cue} {value} -- end fragment.",
    "Machine-read note from {organization}: {value} follows the {cue} heading.",
    "The noisy scan ties {persona} to {cue} {value}.",
)
_add(
    "spoken",
    "positive",
    "The speaker for {persona} read the {cue} aloud as {value}.",
    "Transcript: for {persona}, the {cue} was spoken as {value}.",
    "A synthetic voice note from {organization} gives {value} for the {cue}.",
    "The caller carefully spelled {persona}'s {cue}: {value}.",
    "Transcribed audio says the {cue} associated with {persona} is {value}.",
    "In the rehearsal recording, {value} is stated as the {cue}.",
)
_add(
    "mixed_entity",
    "positive",
    "{persona}'s {cue} is {value}; the separate {other_cue} is {other_value}.",
    "At {organization}, {other_cue} {other_value} and {cue} {value} belong to {persona}.",
    "Two synthetic fields were supplied: {cue}={value}, {other_cue}={other_value}.",
    "The form for {persona} lists {other_value} under {other_cue} and {value} under {cue}.",
    "Cross-check {cue} {value} against {other_cue} {other_value} for {persona}.",
    "A mixed record from {organization} contains {value} ({cue}) and {other_value} ({other_cue}).",
)
_add(
    "cue_free",
    "positive",
    "The synthetic identity entry for {persona} contains {value}.",
    "{organization} stored {value} in the fictional identity section for {persona}.",
    "For {persona}, the identity form shows {value} without a field heading.",
    "The unlabeled synthetic value associated with {persona} is {value}.",
    "An identity worksheet from {organization} contains the entry {value}.",
    "The fictional subject {persona} is linked to the unlabeled value {value}.",
)
_add(
    "cue_shape_conflict",
    "positive",
    "Ignore the identifier shape: {persona}'s {cue} is explicitly {value}.",
    "The heading controls classification; {cue} for {persona} is {value}.",
    "Although the shape is shared, {organization} labels {value} as the {cue}.",
    "Treat {value} according to its field name, {cue}, for {persona}.",
    "This contrastive example assigns {value} to {cue}, independent of appearance.",
    "The synthetic form states {cue}: {value}; morphology alone is ambiguous.",
)
_add(
    "near_miss",
    "hard_negative",
    "The warehouse ticket {negative_value} was assigned to {persona}.",
    "{organization} shipped the parcel under tracking reference {negative_value}.",
    "Build {negative_value} completed in the demonstration environment.",
    "The maintenance request {negative_value} concerns a training laptop.",
    "A fictional invoice uses {negative_value} as its non-personal document number.",
    "The library shelf tag {negative_value} appears beside {persona}'s checkout note.",
)
_add(
    "placeholder",
    "hard_negative",
    "Leave the identity field as {{SYNTHETIC_IDENTIFIER}} until test data is generated.",
    "Documentation placeholder: <fictional-id-goes-here>.",
    "The blank form displays [IDENTIFIER NOT PROVIDED].",
    "Template text only: ENTER A SYNTHETIC VALUE HERE.",
    "No value is present; the example contains the token {{DATE_PLACEHOLDER}}.",
    "The mock screen shows an empty identifier slot, not a person's value.",
)
_add(
    "negation",
    "hard_negative",
    "Do not place personal identifiers in this demonstration note.",
    "{persona} confirmed that no identity value was included.",
    "The hypothetical form would request an identifier, but it remains blank.",
    "{organization} removed the sensitive field before creating this example.",
    "If a credential existed it would be redacted; this sentence contains none.",
    "The reviewer wrote that no birth information should appear here.",
)
_add(
    "unrelated_shape",
    "hard_negative",
    "The asset label {negative_value} identifies a synthetic printer, not a person.",
    "Use {negative_value} as the test batch code for {organization}.",
    "The non-personal catalog number is {negative_value}.",
    "A demo vehicle part carries stock code {negative_value}.",
    "The queue item {negative_value} belongs to a software job.",
    "{negative_value} is a fictional product reference unrelated to identity.",
)
_add(
    "documentation_meta",
    "hard_negative",
    "The manual describes a field named {cue} but supplies no example value.",
    "Schema note: {cue} is an optional string in user-provided datasets.",
    "This documentation discusses {cue} detection without containing a record.",
    "A tutorial heading mentions {cue}; the sample cell is deliberately empty.",
    "The format guide names {cue} only as metadata.",
    "The contributor note asks how {cue} annotations should be represented.",
)
_add(
    "adjacent_value",
    "hard_negative",
    "Near the blank identity field, the page number is {negative_value}.",
    "{persona}'s form has no identity value; {negative_value} is the room code.",
    "The date field is empty, while {negative_value} identifies the document revision.",
    "{organization} left the sensitive cell blank beside batch {negative_value}.",
    "The adjacent value {negative_value} is a harmless routing label.",
    "No personal value appears; {negative_value} is merely the test row number.",
)

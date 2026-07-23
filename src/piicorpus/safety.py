"""Conservative checks for unsafe or insufficiently synthetic record content."""

from __future__ import annotations

import re
from collections.abc import Callable
from urllib.parse import urlsplit

from .config import SafetyConfig
from .models import Record

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")
URL_RE = re.compile(r"https?://[^\s<>\"]+", re.IGNORECASE)
PRIVATE_KEY_RE = re.compile(r"-+BEGIN [A-Z ]*PRIVATE KEY-+")
ACCESS_KEY_RE = re.compile(r"\bA" + r"KIA[0-9A-Z]{16}\b")
TOKEN_RE = re.compile(r"\b(?:" + "sk" + r"-|gh" + r"[opusr]_)[A-Za-z0-9_-]{20,}\b")
JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")

ValueVerifier = Callable[[str], bool]
_VALUE_VERIFIERS: dict[str, ValueVerifier] = {}


def register_value_verifier(
    plugin_name: str, verifier: ValueVerifier, *, replace: bool = False
) -> None:
    """Attach a reservedness proof to a value plugin.

    The verifier must return True only when the value is demonstrably reserved,
    fictional, or invalid by construction (documentation domains and ranges,
    checksum-invalid numbers, never-issued ranges).
    """
    if plugin_name in _VALUE_VERIFIERS and not replace:
        raise ValueError(f"value verifier is already registered: {plugin_name}")
    _VALUE_VERIFIERS[plugin_name] = verifier


def registered_value_verifiers() -> tuple[str, ...]:
    return tuple(sorted(_VALUE_VERIFIERS))


def is_reserved_domain(host: str, allowed: tuple[str, ...]) -> bool:
    normalized = host.strip().rstrip(".").casefold()
    return any(normalized == domain or normalized.endswith("." + domain) for domain in allowed)


def unsafe_text_reasons(text: str, config: SafetyConfig) -> list[str]:
    reasons: list[str] = []
    for host in EMAIL_RE.findall(text):
        if not is_reserved_domain(host, config.reserved_email_domains):
            reasons.append("email domain is not reserved for documentation")
    for match in URL_RE.findall(text):
        try:
            host = urlsplit(match).hostname or ""
        except ValueError:
            reasons.append("URL is malformed")
            continue
        if not is_reserved_domain(host, config.reserved_email_domains):
            reasons.append("URL host is not reserved for documentation")
    if PRIVATE_KEY_RE.search(text):
        reasons.append("private-key material marker is present")
    if ACCESS_KEY_RE.search(text) or TOKEN_RE.search(text) or JWT_RE.search(text):
        reasons.append("credential-shaped value is present")
    folded = text.casefold()
    for term in config.forbidden_terms:
        if term.casefold() in folded:
            reasons.append("configured prohibited term is present")
    return reasons


def _value_allowed(
    value: str,
    entity_type: str,
    config: SafetyConfig,
    label_plugins: dict[str, str] | None,
) -> bool:
    prefix_ok = bool(config.allowed_value_prefixes) and value.startswith(
        config.allowed_value_prefixes
    )
    verifier = _VALUE_VERIFIERS.get((label_plugins or {}).get(entity_type, ""))
    verifier_ok = bool(verifier and verifier(value))
    if config.mode == "prefix":
        return prefix_ok
    if config.mode == "verifier":
        return verifier_ok
    return prefix_ok or verifier_ok


def unsafe_record_reasons(
    record: Record,
    config: SafetyConfig,
    *,
    label_plugins: dict[str, str] | None = None,
) -> list[str]:
    reasons = unsafe_text_reasons(record.text, config)
    for annotation in record.annotations:
        if not _value_allowed(annotation.text, annotation.entity_type, config, label_plugins):
            reasons.append(
                "annotated value is neither prefix-marked synthetic nor verified reserved"
            )
    return sorted(set(reasons))

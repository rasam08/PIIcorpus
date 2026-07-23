"""Conservative checks for unsafe or insufficiently synthetic record content."""

from __future__ import annotations

import re
from urllib.parse import urlsplit

from .config import SafetyConfig
from .models import Record

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")
URL_RE = re.compile(r"https?://[^\s<>\"]+", re.IGNORECASE)
PRIVATE_KEY_RE = re.compile(r"-+BEGIN [A-Z ]*PRIVATE KEY-+")
ACCESS_KEY_RE = re.compile(r"\bA" + r"KIA[0-9A-Z]{16}\b")
TOKEN_RE = re.compile(r"\b(?:" + "sk" + r"-|gh" + r"[opusr]_)[A-Za-z0-9_-]{20,}\b")
JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")


def _reserved_domain(host: str, allowed: tuple[str, ...]) -> bool:
    normalized = host.rstrip(".").casefold()
    return any(normalized == domain or normalized.endswith("." + domain) for domain in allowed)


def unsafe_text_reasons(text: str, config: SafetyConfig) -> list[str]:
    reasons: list[str] = []
    for host in EMAIL_RE.findall(text):
        if not _reserved_domain(host, config.reserved_email_domains):
            reasons.append("email domain is not reserved for documentation")
    for match in URL_RE.findall(text):
        try:
            host = urlsplit(match).hostname or ""
        except ValueError:
            reasons.append("URL is malformed")
            continue
        if not _reserved_domain(host, config.reserved_email_domains):
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


def unsafe_record_reasons(record: Record, config: SafetyConfig) -> list[str]:
    reasons = unsafe_text_reasons(record.text, config)
    for annotation in record.annotations:
        if not annotation.text.startswith(config.allowed_value_prefixes):
            reasons.append("annotated value lacks an allowed synthetic prefix")
    return sorted(set(reasons))

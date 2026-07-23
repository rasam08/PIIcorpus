"""Realistic-but-reserved value plugins with reservedness verifiers.

Every plugin here emits values that look like real contact or identity surface
forms while being provably reserved, fictional, or invalid by construction:

- ``reserved_email``: RFC 2606 documentation domains (example.com/org/net).
- ``reserved_phone_nanp``: the 555-01XX exchange range reserved for fictional
  use in the North American Numbering Plan.
- ``card_shaped_invalid``: 16-digit card-shaped numbers whose Luhn check digit
  is deliberately wrong, so no valid primary account number can be emitted.
- ``documentation_ip``: RFC 5737 documentation IPv4 ranges.
- ``never_issued_ssn_shape``: 9XX-XX-XXXX shapes using area numbers 900-999,
  which the Social Security Administration never issues.

Each plugin registers a verifier via ``register_value_verifier`` so the safety
layer can prove reservedness instead of relying on a synthetic prefix, and
registers its value shapes for morphology-aware audit checks. Rationale for
each reservation is documented in docs/DATA_SAFETY.md.
"""

from __future__ import annotations

import ipaddress
import random
import re

from .config import SPLIT_ORDER, LabelConfig, split_partition
from .generator import register_value_plugin
from .morphology import register_shape
from .safety import is_reserved_domain, register_value_verifier

RESERVED_DOMAINS = ("example.com", "example.org", "example.net")

_GIVEN_FRAGMENTS = (
    "alex", "bailey", "casey", "devon", "ellis", "frankie", "harper", "jules",
    "kai", "lane", "marlow", "noor", "oakley", "peyton", "quinn", "reese",
    "sage", "tatum", "umber", "vale", "winter", "xen", "yael", "zephyr",
)
_FAMILY_FRAGMENTS = (
    "arden", "bright", "cole", "dune", "ember", "frost", "gale", "hollis",
    "iris", "juniper", "keel", "larkspur", "meadow", "north", "onyx", "pines",
    "quill", "ridge", "sable", "thorn", "umberly", "vesper", "wren", "yarrow",
)
_AREA_CODES = (
    "202", "206", "212", "213", "215", "216", "301", "303", "305", "312",
    "314", "404", "406", "412", "415", "503", "504", "512", "602", "612",
    "615", "702", "713", "802",
)
_DOCUMENTATION_NETWORKS = (
    ipaddress.ip_network("192.0.2.0/24"),
    ipaddress.ip_network("198.51.100.0/24"),
    ipaddress.ip_network("203.0.113.0/24"),
)
_DOCUMENTATION_V6 = ipaddress.ip_network("2001:db8::/32")

_EMAIL_SHAPE_RE = re.compile(r"[a-z]+\.[a-z]+\d{2}@[a-z0-9.-]+")
_PHONE_PAREN_RE = re.compile(r"\(\d{3}\) 555-01\d{2}")
_PHONE_DASH_RE = re.compile(r"\d{3}-555-01\d{2}")
_CARD_RE = re.compile(r"\d{4}-\d{4}-\d{4}-\d{4}")
_SSN_SHAPE_RE = re.compile(r"9\d{2}-\d{2}-\d{4}")
_ANY_EMAIL_RE = re.compile(r"[^@\s]+@([^@\s]+)")


def _reserved_email(rng: random.Random, _label: LabelConfig, split: str, _index: int) -> str:
    given = rng.choice(split_partition(_GIVEN_FRAGMENTS, split))
    family = rng.choice(split_partition(_FAMILY_FRAGMENTS, split))
    return f"{given}.{family}{rng.randint(10, 99)}@{rng.choice(RESERVED_DOMAINS)}"


def _verify_reserved_email(value: str) -> bool:
    match = _ANY_EMAIL_RE.fullmatch(value.strip().rstrip("."))
    if match is None:
        return False
    return is_reserved_domain(match.group(1), RESERVED_DOMAINS)


def _reserved_phone(rng: random.Random, label: LabelConfig, split: str, index: int) -> str:
    area = rng.choice(split_partition(_AREA_CODES, split))
    line = f"{rng.randint(0, 99):02d}"
    shape = label.options.get("_shape_override")
    if not isinstance(shape, str):
        shape = "phone_paren" if index % 2 == 0 else "phone_dash"
    if shape == "phone_dash":
        return f"{area}-555-01{line}"
    return f"({area}) 555-01{line}"


def _verify_reserved_phone(value: str) -> bool:
    stripped = value.strip()
    return bool(
        _PHONE_PAREN_RE.fullmatch(stripped) or _PHONE_DASH_RE.fullmatch(stripped)
    )


def _luhn_check_digit(digits: str) -> int:
    total = 0
    for position, char in enumerate(reversed(digits)):
        number = int(char)
        if position % 2 == 0:
            number *= 2
            if number > 9:
                number -= 9
        total += number
    return (10 - total % 10) % 10


def _luhn_valid(digits: str) -> bool:
    return _luhn_check_digit(digits[:-1]) == int(digits[-1])


def _card_shaped_invalid(
    rng: random.Random, _label: LabelConfig, _split: str, _index: int
) -> str:
    base = "4" + "".join(rng.choice("0123456789") for _ in range(14))
    wrong = (_luhn_check_digit(base) + rng.randint(1, 9)) % 10
    digits = base + str(wrong)
    return "-".join(digits[start : start + 4] for start in (0, 4, 8, 12))


def _verify_card_shaped_invalid(value: str) -> bool:
    stripped = value.strip()
    if not _CARD_RE.fullmatch(stripped):
        return False
    return not _luhn_valid(stripped.replace("-", ""))


def _documentation_ip(
    rng: random.Random, _label: LabelConfig, split: str, index: int
) -> str:
    network = _DOCUMENTATION_NETWORKS[index % len(_DOCUMENTATION_NETWORKS)]
    offset = SPLIT_ORDER.index(split)
    host = rng.choice([n for n in range(1, 255) if n % len(SPLIT_ORDER) == offset])
    return str(network.network_address + host)


def _verify_documentation_ip(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value.strip())
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address):
        return address in _DOCUMENTATION_V6
    return any(address in network for network in _DOCUMENTATION_NETWORKS)


def _never_issued_ssn(
    rng: random.Random, _label: LabelConfig, split: str, _index: int
) -> str:
    offset = SPLIT_ORDER.index(split)
    area = rng.choice([n for n in range(900, 1000) if n % len(SPLIT_ORDER) == offset])
    return f"{area}-{rng.randint(1, 99):02d}-{rng.randint(1, 9999):04d}"


def _verify_never_issued_ssn(value: str) -> bool:
    return bool(_SSN_SHAPE_RE.fullmatch(value.strip()))


register_value_plugin("reserved_email", _reserved_email)
register_value_plugin("reserved_phone_nanp", _reserved_phone)
register_value_plugin("card_shaped_invalid", _card_shaped_invalid)
register_value_plugin("documentation_ip", _documentation_ip)
register_value_plugin("never_issued_ssn_shape", _never_issued_ssn)

register_value_verifier("reserved_email", _verify_reserved_email)
register_value_verifier("reserved_phone_nanp", _verify_reserved_phone)
register_value_verifier("card_shaped_invalid", _verify_card_shaped_invalid)
register_value_verifier("documentation_ip", _verify_documentation_ip)
register_value_verifier("never_issued_ssn_shape", _verify_never_issued_ssn)

register_shape("reserved_email", lambda value: _EMAIL_SHAPE_RE.fullmatch(value) is not None)
register_shape("phone_paren", _PHONE_PAREN_RE.pattern)
register_shape("phone_dash", _PHONE_DASH_RE.pattern)
register_shape("card_shaped", _CARD_RE.pattern)
register_shape("documentation_ip", _verify_documentation_ip)
register_shape("never_issued_ssn", _SSN_SHAPE_RE.pattern)

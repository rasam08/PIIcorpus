# Data safety

The built-in demos contain only newly authored synthetic material. Fictional personas and
organizations are drawn from shared pools that stay split-disjoint by interleaved partitioning.

Two safety models are supported, selected by `safety.mode`:

- `prefix`: every annotated value must begin with an approved synthetic prefix (the SYN- demo).
- `verifier`: every annotated value must satisfy the reservedness verifier registered by its value
  plugin (the realistic-safe demo).
- `either` (default): a value passes with either proof.

Safety checks additionally reject credential-shaped strings, private-key markers, non-reserved
email domains, non-reserved URL hosts, and configured prohibited terms in all record text.
Example internet names must use RFC 2606 reserved domains.

## Reservedness rationale for the realistic plugins

Each realistic-but-reserved plugin ships a verifier proving that its values cannot collide with
issued real-world identifiers:

- `reserved_email` — hosts are RFC 2606 reserved documentation domains
  (`example.com`, `example.org`, `example.net`), which can never be registered.
- `reserved_phone_nanp` — numbers use the `555-01XX` exchange range, reserved for fictional use in
  the North American Numbering Plan.
- `card_shaped_invalid` — 16-digit card-shaped numbers whose Luhn check digit is deliberately
  wrong; a Luhn-invalid number cannot be a valid primary account number. The verifier asserts the
  invalidity, so an accidental valid number is rejected.
- `documentation_ip` — addresses fall inside the RFC 5737 IPv4 documentation ranges
  (`192.0.2.0/24`, `198.51.100.0/24`, `203.0.113.0/24`) or the RFC 3849 IPv6 documentation prefix
  (`2001:db8::/32`), which are never routed.
- `never_issued_ssn_shape` — nine-digit `9XX-XX-XXXX` shapes use area numbers 900-999, which the
  Social Security Administration has never issued and has stated it will not issue.

The OCR-noise and spoken families are excluded from the realistic configuration because their
surface transformations would mutate values out of the verified reserved ranges.

No valid credential, token, private key, payment instrument, or secret is required by any demo.
The base project does not download data or machine-learning artifacts.

## Imported and external data

The safety validator cannot establish consent, provenance, ownership, or release permission for
user-supplied text. Import records remain unreviewed and are not placed in generated splits; the
importer and `audit-external` run the sensitive-content scan informationally and report findings
without making the data releasable. The user is responsible for privacy, licensing, and release
decisions.

## Residual risk

Pattern checks and verifiers can miss unsafe content and can flag harmless content. Synthetic
values may coincidentally resemble an external convention. Reviewers should inspect generation
rules and use independent secret scanning before distribution.

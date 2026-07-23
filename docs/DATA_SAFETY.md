# Data safety

The built-in demo contains only newly authored synthetic material. Fictional personas and
organizations are split-disjoint. Identifier values begin with an explicit synthetic prefix and
use shapes that are not representations of real medical, passport, driver-license, or other
issuing formats. Dates are synthetic calendar values associated only with fictional personas.

Safety checks reject credential-shaped strings, private-key markers, non-reserved email domains,
non-reserved URL hosts, configured prohibited terms, and annotated values without an approved
synthetic prefix. Example internet names must use RFC 2606 reserved domains.

No valid credential, token, private key, payment instrument, or secret is required by the demo.
The base project does not download data or machine-learning artifacts.

## Imported data

The safety validator cannot establish consent, provenance, ownership, or release permission for
user-supplied text. Import records remain unreviewed and are not placed in generated splits. The
user is responsible for privacy, licensing, and release decisions.

## Residual risk

Pattern checks can miss unsafe content and can flag harmless content. Synthetic values may
coincidentally resemble an external convention. Reviewers should inspect generation rules and use
independent secret scanning before distribution.

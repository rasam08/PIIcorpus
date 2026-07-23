# Security policy

## Reporting a vulnerability

Use [GitHub private vulnerability reporting](https://github.com/rasam08/PIIcorpus/security/advisories/new).
Please do not open a public issue for a suspected vulnerability and do not include real personal
data, credentials, secrets, or private keys in a report.

Describe the affected version, impact, and minimal reproduction steps. Synthetic placeholders are
preferred. No email address is designated for security reports.

## Supported versions

Only the latest published release is supported with security fixes.

## Security boundaries

PIIcorpus processes local corpus files. It does not provide isolation for hostile Python plugins.
Only load plugins and configurations from trusted sources. Imported text is not established as safe
or releasable, and generated data is not guaranteed to be free of every unsafe pattern.

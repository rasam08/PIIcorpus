# Contributing

Contributions that improve deterministic generation, structural auditing, safe examples,
documentation, import validation, or span-preserving export are welcome.

## Development setup

```console
python -m venv .venv
python -m pip install --requirement requirements-dev.lock
python -m pip install --no-deps --no-build-isolation -e .
ruff check .
mypy src
pytest
python -m build
```

Use Python 3.11 or newer. Keep the base runtime free of unnecessary dependencies.

## Data and provenance rules

- Do not submit real personal data, scraped text, credentials, secrets, or private keys.
- Do not submit examples whose public provenance or license is uncertain.
- Use newly authored fictional personas, organizations, values, and sentence frames.
- Use RFC 2606 domains for email and URL examples.
- Keep imported data separate from generated splits.
- Do not weaken a failing validator or audit assertion to make CI pass.
- Add a named defective case and a focused test when adding a new audit risk.

Corpus evidence and release claims should be additive and reproducible. A same-generator holdout
must never be described as independent evaluation.

## Pull requests

Keep changes narrow, explain the measured property, and include tests. Run the complete command set
above. If a public interface or record format changes, update `docs/FORMAT.md` and the changelog.

By contributing source code, you agree that it is licensed under Apache-2.0. Clearly identify any
generated data intended for CC0-1.0 dedication. Do not assume that externally sourced data can be
relicensed.

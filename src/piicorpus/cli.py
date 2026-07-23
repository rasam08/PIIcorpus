"""Command-line interface with stable exit-code semantics."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from .config import ConfigError, load_config
from .exporters import EXPORT_FORMATS, ExportError, export_corpus
from .failure_model import audit_corpus
from .generator import generate
from .importers import ImportErrorSafe, import_annotated
from .manifest import load_corpus
from .models import stable_json
from .validators import CorpusIntegrityError, validate_corpus

EXIT_OK = 0
EXIT_FINDINGS = 1
EXIT_OPERATIONAL = 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="piicorpus")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="generate a deterministic corpus")
    generate_parser.add_argument("--config", required=True)
    generate_parser.add_argument("--out", required=True)
    generate_parser.add_argument("--seed", type=int, help="override the configuration seed")

    validate_parser = subparsers.add_parser("validate", help="validate emitted corpus files")
    validate_parser.add_argument("directory")
    validate_parser.add_argument("--strict", action="store_true")
    validate_parser.add_argument("--json", action="store_true", dest="as_json")

    audit_parser = subparsers.add_parser("audit", help="classify corpus failure modes")
    audit_parser.add_argument("directory")
    audit_parser.add_argument("--format", choices=("json", "text", "markdown"), default="text")
    audit_parser.add_argument("--out")
    audit_parser.add_argument(
        "--forensic-allow-invalid",
        action="store_true",
        help="continue after strict validation failure; result remains FAIL",
    )

    export_parser = subparsers.add_parser("export", help="export records without changing spans")
    export_parser.add_argument("directory")
    export_parser.add_argument("--format", choices=EXPORT_FORMATS, required=True)
    export_parser.add_argument("--out")
    export_parser.add_argument(
        "--forensic-allow-invalid",
        action="store_true",
        help="export invalid input for forensics; command still exits with findings",
    )

    import_parser = subparsers.add_parser("import-annotated", help="import user-marked text")
    import_parser.add_argument("input")
    import_parser.add_argument("--out", required=True)
    import_parser.add_argument("--debug-local", action="store_true")

    report_parser = subparsers.add_parser("report", help="summarize validation and audit status")
    report_parser.add_argument("directory")
    report_parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _run(args: argparse.Namespace) -> int:
    if args.command == "generate":
        config = load_config(args.config)
        if args.seed is not None:
            if args.seed < 0:
                raise ConfigError("seed must be non-negative")
            config = replace(config, seed=args.seed)
        output = Path(args.out)
        if output.exists() and any(output.iterdir()):
            raise ConfigError("generation output directory is not empty")
        manifest = generate(config, output)
        count = sum(v["records"] for v in manifest["counts"].values())
        print(f"generated {count} records in {output}")
        print(f"configuration_digest={manifest['configuration_digest']}")
        return EXIT_OK

    if args.command == "validate":
        validation_report = validate_corpus(args.directory, strict=args.strict)
        if args.as_json:
            print(stable_json(validation_report.to_dict(), pretty=True), end="")
        elif validation_report.valid:
            print("PASS: corpus files and configured invariants are valid")
        else:
            print(f"FAIL: {len(validation_report.errors)} validation finding(s)")
            for error in validation_report.errors:
                print(f"- {error}")
        return EXIT_OK if validation_report.valid else EXIT_FINDINGS

    if args.command == "audit":
        if args.forensic_allow_invalid:
            print(
                "WARNING: forensic override enabled; invalid input cannot receive a clean verdict.",
                file=sys.stderr,
            )
        audit_report = audit_corpus(
            args.directory, allow_invalid=args.forensic_allow_invalid
        )
        rendered = audit_report.render(args.format)
        if args.out:
            Path(args.out).write_text(rendered, encoding="utf-8", newline="\n")
        else:
            print(rendered, end="")
        return EXIT_FINDINGS if audit_report.failed else EXIT_OK

    if args.command == "export":
        if args.forensic_allow_invalid:
            print(
                "WARNING: forensic override enabled; exported invalid input is non-authoritative.",
                file=sys.stderr,
            )
        result = export_corpus(
            args.directory,
            args.format,
            args.out,
            allow_invalid=args.forensic_allow_invalid,
        )
        print(stable_json(result, pretty=True), end="")
        return EXIT_OK if result["integrity_valid"] else EXIT_FINDINGS

    if args.command == "import-annotated":
        print(
            "WARNING: imported data remains unreviewed; the user is responsible for consent, "
            "privacy, provenance, licensing, and release decisions.",
            file=sys.stderr,
        )
        result = import_annotated(args.input, args.out, debug_local=args.debug_local)
        print(stable_json(result, pretty=True), end="")
        return EXIT_OK

    if args.command == "report":
        _config, _records, manifest = load_corpus(args.directory)
        validation = validate_corpus(args.directory, strict=True)
        audit = audit_corpus(args.directory, allow_invalid=True)
        report_value = {
            "audit": audit.to_dict(),
            "manifest": {
                "configuration_digest": manifest["configuration_digest"],
                "counts": manifest["counts"],
                "generator_version": manifest["generator_version"],
                "seed": manifest["seed"],
            },
            "validation": validation.to_dict(),
        }
        if args.as_json:
            print(stable_json(report_value, pretty=True), end="")
        else:
            print(f"validation={'PASS' if validation.valid else 'FAIL'}")
            print("audit=" + ("FAIL" if audit.failed else "PASS_WITH_UNMEASURED"))
            print(audit.limitation)
        return EXIT_OK if validation.valid and not audit.failed else EXIT_FINDINGS

    raise AssertionError("unreachable command")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _run(_parser().parse_args(argv))
    except CorpusIntegrityError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        for error in exc.report.errors:
            print(f"- {error}", file=sys.stderr)
        return EXIT_FINDINGS
    except (
        ConfigError,
        ExportError,
        ImportErrorSafe,
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ) as exc:
        print(f"operational error: {exc}", file=sys.stderr)
        return EXIT_OPERATIONAL

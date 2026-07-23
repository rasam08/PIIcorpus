"""Command-line interface with stable exit-code semantics."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from .config import ConfigError, load_config
from .exporters import EXPORT_FORMATS, ExportError, export_corpus
from .failure_model import audit_corpus, audit_external_records
from .generator import GENERATOR_VERSION, generate
from .importers import (
    ExternalImportError,
    ImportErrorSafe,
    import_annotated,
    load_external,
)
from .manifest import load_corpus, sha256_file
from .models import stable_json
from .scoring import ScoringError, score_corpus
from .validators import CorpusIntegrityError, validate_corpus

EXIT_OK = 0
EXIT_FINDINGS = 1
EXIT_OPERATIONAL = 2


PLUGIN_ENTRY_POINT_GROUP = "piicorpus.plugins"


def _load_plugins(modules: str) -> None:
    """Import registration hooks from entry points and explicit ``--plugins`` modules."""
    for entry_point in importlib.metadata.entry_points(group=PLUGIN_ENTRY_POINT_GROUP):
        loaded = entry_point.load()
        if callable(loaded):
            loaded()
    for name in filter(None, (part.strip() for part in modules.split(","))):
        importlib.import_module(name)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="piicorpus")
    parser.add_argument(
        "--plugins",
        default="",
        help="comma-separated modules imported before running, so applications can "
        "register value plugins, families, shapes, and verifiers",
    )
    parser.add_argument(
        "--traceback",
        action="store_true",
        help="re-raise errors with a full traceback instead of a one-line summary",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="generate a deterministic corpus")
    generate_parser.add_argument("--config", required=True)
    generate_parser.add_argument("--out", required=True)
    generate_parser.add_argument("--seed", type=int, help="override the configuration seed")

    validate_parser = subparsers.add_parser("validate", help="validate emitted corpus files")
    validate_parser.add_argument("directory")
    validate_parser.add_argument(
        "--strict",
        action="store_true",
        help="deprecated no-op: strict validation is now always on",
    )
    validate_parser.add_argument("--json", action="store_true", dest="as_json")

    audit_parser = subparsers.add_parser("audit", help="classify corpus failure modes")
    audit_parser.add_argument("directory")
    audit_parser.add_argument("--format", choices=("json", "text", "markdown"), default="text")
    audit_parser.add_argument("--out")
    audit_parser.add_argument(
        "--profile",
        choices=("config", "reference"),
        default="config",
        help="threshold source: the corpus configuration or the recommended reference profile",
    )
    audit_parser.add_argument(
        "--probe",
        dest="probe",
        action="store_true",
        default=None,
        help="train the trivial-model learnability probe (slower)",
    )
    audit_parser.add_argument(
        "--no-probe",
        dest="probe",
        action="store_false",
        help="skip the learnability probe even if the configuration enables it",
    )
    audit_parser.add_argument(
        "--forensic-allow-invalid",
        action="store_true",
        help="continue after strict validation failure; result remains FAIL",
    )

    external_parser = subparsers.add_parser(
        "audit-external",
        help="audit any NER dataset (jsonl, Hugging Face jsonl, or CoNLL) "
        "without a PIIcorpus manifest",
    )
    external_parser.add_argument("paths", nargs="*")
    external_parser.add_argument(
        "--format", choices=("jsonl", "hf", "conll"), required=True, dest="data_format"
    )
    external_parser.add_argument(
        "--split",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="map an input file to a named split; repeatable",
    )
    external_parser.add_argument(
        "--byte-offsets",
        action="store_true",
        help="jsonl span offsets are UTF-8 byte positions instead of code points",
    )
    external_parser.add_argument(
        "--report-format", choices=("json", "text", "markdown"), default="text"
    )
    external_parser.add_argument("--out")
    external_parser.add_argument(
        "--no-probe", action="store_true", help="skip the learnability probe"
    )
    external_parser.add_argument(
        "--fail-on-safety",
        action="store_true",
        help="treat sensitive-content matches as FAIL instead of WARN",
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

    score_parser = subparsers.add_parser(
        "score", help="score detector span predictions against a corpus"
    )
    score_parser.add_argument("directory")
    score_parser.add_argument("predictions")
    score_parser.add_argument("--match", choices=("strict", "overlap"), default="strict")
    score_parser.add_argument(
        "--byte-offsets",
        action="store_true",
        help="prediction offsets are UTF-8 byte positions instead of code points",
    )
    score_parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="score only records present in the predictions file",
    )
    score_parser.add_argument(
        "--format", choices=("json", "text", "markdown"), default="text"
    )
    score_parser.add_argument("--out")
    score_parser.add_argument(
        "--fail-under",
        type=float,
        help="exit with findings when micro F1 falls below this value",
    )

    import_parser = subparsers.add_parser("import-annotated", help="import user-marked text")
    import_parser.add_argument("input")
    import_parser.add_argument("--out", required=True)
    import_parser.add_argument("--debug-local", action="store_true")

    report_parser = subparsers.add_parser("report", help="summarize validation and audit status")
    report_parser.add_argument("directory")
    report_parser.add_argument("--json", action="store_true", dest="as_json")

    reproduce_parser = subparsers.add_parser(
        "reproduce",
        help="regenerate from the corpus's own config snapshot and byte-compare",
    )
    reproduce_parser.add_argument("directory")
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
        validation_report = validate_corpus(args.directory, strict=True)
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
            args.directory,
            allow_invalid=args.forensic_allow_invalid,
            profile=args.profile,
            probe=args.probe,
        )
        rendered = audit_report.render(args.format)
        if args.out:
            Path(args.out).write_text(rendered, encoding="utf-8", newline="\n")
        else:
            print(rendered, end="")
        return EXIT_FINDINGS if audit_report.failed else EXIT_OK

    if args.command == "audit-external":
        sources: dict[str, Path] = {}

        def add_source(name: str, path: Path) -> None:
            normalized = name.strip()
            if normalized in sources:
                raise ExternalImportError(
                    f"duplicate external input name {normalized!r}; use unique "
                    "--split names or filenames"
                )
            sources[normalized] = path

        for item in args.split:
            name, _, raw_path = item.partition("=")
            if not name or not raw_path:
                raise ExternalImportError("--split expects NAME=PATH")
            add_source(name, Path(raw_path.strip()))
        for raw_path in args.paths:
            path = Path(raw_path)
            add_source(
                path.stem if (len(args.paths) > 1 or sources) else "data",
                path,
            )
        if not sources:
            raise ExternalImportError("provide input paths or --split NAME=PATH entries")
        split_records = load_external(
            sources, args.data_format, byte_offsets=args.byte_offsets
        )
        external_report = audit_external_records(
            split_records,
            probe=not args.no_probe,
            fail_on_safety=args.fail_on_safety,
        )
        rendered = external_report.render(args.report_format)
        if args.out:
            Path(args.out).write_text(rendered, encoding="utf-8", newline="\n")
        else:
            print(rendered, end="")
        return EXIT_FINDINGS if external_report.failed else EXIT_OK

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

    if args.command == "score":
        score_report = score_corpus(
            args.directory,
            args.predictions,
            match=args.match,
            byte_offsets=args.byte_offsets,
            allow_partial=args.allow_partial,
        )
        rendered = score_report.render(args.format)
        if args.out:
            Path(args.out).write_text(rendered, encoding="utf-8", newline="\n")
        else:
            print(rendered, end="")
        if args.fail_under is not None and score_report.overall["f1"] < args.fail_under:
            return EXIT_FINDINGS
        return EXIT_OK

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
            statuses = {finding.status for finding in audit.findings}
            if audit.failed:
                audit_status = "FAIL"
            elif {"WARN", "UNMEASURED"} <= statuses:
                audit_status = "PASS_WITH_WARNINGS_AND_UNMEASURED"
            elif "WARN" in statuses:
                audit_status = "PASS_WITH_WARNINGS"
            elif "UNMEASURED" in statuses:
                audit_status = "PASS_WITH_UNMEASURED"
            else:
                audit_status = "PASS"
            print(f"audit={audit_status}")
            print(audit.limitation)
        return EXIT_OK if validation.valid and not audit.failed else EXIT_FINDINGS

    if args.command == "reproduce":
        root = Path(args.directory)
        config, _records, manifest = load_corpus(root)
        stored_version = manifest.get("generator_version")
        if stored_version != GENERATOR_VERSION:
            print(
                f"operational error: corpus was generated by version {stored_version}; "
                f"this installation generates version {GENERATOR_VERSION}, and byte "
                "reproduction is version-scoped by design",
                file=sys.stderr,
            )
            return EXIT_OPERATIONAL
        mismatches: list[str] = []
        with tempfile.TemporaryDirectory() as scratch:
            target = Path(scratch) / "reproduced"
            generate(config, target)
            relative_paths = ["manifest.json", *sorted(manifest.get("files", {}))]
            for relative in relative_paths:
                original = root / relative
                reproduced = target / relative
                if not original.is_file() or not reproduced.is_file():
                    matched = False
                else:
                    matched = sha256_file(original) == sha256_file(reproduced)
                if not matched:
                    mismatches.append(relative)
                print(f"{'PASS' if matched else 'FAIL'} {relative}")
        if mismatches:
            print(f"FAIL: {len(mismatches)} file(s) differ from a fresh regeneration")
            return EXIT_FINDINGS
        print("PASS: corpus is byte-identical to a fresh regeneration")
        return EXIT_OK

    raise AssertionError("unreachable command")


def main(argv: Sequence[str] | None = None) -> int:
    debug = False
    try:
        args = _parser().parse_args(argv)
        debug = args.traceback
        _load_plugins(args.plugins)
        return _run(args)
    except CorpusIntegrityError as exc:
        if debug:
            raise
        print(f"FAIL: {exc}", file=sys.stderr)
        for error in exc.report.errors:
            print(f"- {error}", file=sys.stderr)
        return EXIT_FINDINGS
    except (
        ConfigError,
        ExportError,
        ImportError,
        ImportErrorSafe,
        ScoringError,
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ) as exc:
        if debug:
            raise
        print(f"operational error: {exc}", file=sys.stderr)
        return EXIT_OPERATIONAL

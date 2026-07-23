"""PIIcorpus: deterministic synthetic contextual-PII corpus generation and auditing."""

from .annotation import AnnotationError, parse_marked, render_marked, validate_annotations
from .generator import GENERATOR_VERSION, generate, generate_records, register_value_plugin
from .validators import ValidationReport, validate_corpus

__all__ = [
    "GENERATOR_VERSION",
    "AnnotationError",
    "ValidationReport",
    "generate",
    "generate_records",
    "parse_marked",
    "register_value_plugin",
    "render_marked",
    "validate_annotations",
    "validate_corpus",
]

__version__ = "0.1.0"

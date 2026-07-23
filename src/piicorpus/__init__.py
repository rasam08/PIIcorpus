"""PIIcorpus: deterministic synthetic contextual-PII corpus generation and auditing."""

from . import plugins_realistic as _plugins_realistic  # registers realistic plugins
from .annotation import AnnotationError, parse_marked, render_marked, validate_annotations
from .exporters import export_corpus
from .failure_model import AuditReport, audit_corpus
from .generator import GENERATOR_VERSION, generate, generate_records, register_value_plugin
from .morphology import register_shape, registered_shapes
from .safety import register_value_verifier, registered_value_verifiers
from .skeletons import FamilyPlugin, register_family, registered_families
from .validators import ValidationReport, validate_corpus

del _plugins_realistic

__all__ = [
    "GENERATOR_VERSION",
    "AnnotationError",
    "AuditReport",
    "FamilyPlugin",
    "ValidationReport",
    "audit_corpus",
    "export_corpus",
    "generate",
    "generate_records",
    "parse_marked",
    "register_family",
    "register_shape",
    "register_value_plugin",
    "register_value_verifier",
    "registered_families",
    "registered_shapes",
    "registered_value_verifiers",
    "render_marked",
    "validate_annotations",
    "validate_corpus",
]

__version__ = "0.2.0"

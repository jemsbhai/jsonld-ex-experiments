"""EN5.5 -- Backward Compatibility with Legacy Processors core module.

NeurIPS 2026 D&B, Suite EN5 (Security and Integrity), Experiment 5.

Tests whether legacy JSON-LD processors (PyLD, rdflib) handle
jsonld-ex documents gracefully. Documents actual behavior honestly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# -- sys.path setup --
import sys
_SCRIPT_DIR = Path(__file__).resolve().parent
_EXPERIMENTS_ROOT = _SCRIPT_DIR.parent
for p in [str(_SCRIPT_DIR), str(_EXPERIMENTS_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)


# =====================================================================
# Constants
# =====================================================================

TEST_DOC_NAMES = [
    "confidence_only",
    "integrity_only",
    "vector_only",
    "provenance_only",
    "temporal_only",
    "all_extensions",
    "nested_extensions",
    "array_with_extensions",
    "multiple_contexts_integrity",
    "minimal_standard",
]


# =====================================================================
# Result dataclasses
# =====================================================================

@dataclass
class DependencyStatus:
    """Status of an optional dependency."""
    name: str = ""
    available: bool = False
    version: str | None = None
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DocumentResult:
    """Result of parsing one test document."""
    name: str = ""
    success: bool = False
    error_type: str | None = None
    error_msg: str | None = None
    expanded_keys: list[str] = field(default_factory=list)
    extension_keywords_survived: list[str] = field(default_factory=list)
    standard_data_preserved: bool = False
    triple_count: int | None = None


@dataclass
class ParserTestResult:
    """Results from testing one parser against all documents."""
    parser_name: str = ""
    parser_available: bool = False
    skipped: bool = False
    skip_reason: str = ""
    total_docs: int = 0
    successes: int = 0
    failures: int = 0
    doc_results: list[DocumentResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "parser_name": self.parser_name,
            "parser_available": self.parser_available,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "total_docs": self.total_docs,
            "successes": self.successes,
            "failures": self.failures,
            "doc_results": [asdict(d) for d in self.doc_results],
        }


@dataclass
class CrossParserResult:
    """Cross-parser comparison results."""
    parsers_tested: list[str] = field(default_factory=list)
    agreement_rate: float | None = None
    finding: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EN55FullResult:
    """Complete EN5.5 results."""
    dependency_status: dict[str, DependencyStatus] = field(default_factory=dict)
    pyld: ParserTestResult | None = None
    rdflib: ParserTestResult | None = None
    cross_parser: CrossParserResult | None = None

    def to_dict(self) -> dict:
        return {
            "dependency_status": {k: v.to_dict() for k, v in self.dependency_status.items()},
            "pyld": self.pyld.to_dict() if self.pyld else None,
            "rdflib": self.rdflib.to_dict() if self.rdflib else None,
            "cross_parser": self.cross_parser.to_dict() if self.cross_parser else None,
        }


# =====================================================================
# Dependency checks
# =====================================================================

def check_pyld_available() -> DependencyStatus:
    """Check if PyLD is installed and importable."""
    try:
        import pyld  # noqa: F811
        version = getattr(pyld, "__version__", "unknown")
        return DependencyStatus(
            name="PyLD", available=True, version=version,
            note="PyLD is available for backward compatibility testing.",
        )
    except ImportError:
        return DependencyStatus(
            name="PyLD", available=False,
            note="PyLD is not installed. Install with: pip install PyLD",
        )


def check_rdflib_available() -> DependencyStatus:
    """Check if rdflib with JSON-LD support is available."""
    try:
        import rdflib
        version = getattr(rdflib, "__version__", "unknown")
        # Check for JSON-LD plugin
        try:
            from rdflib.plugin import get as get_plugin
            get_plugin("json-ld", rdflib.parser.Parser)
            has_jsonld = True
            note = "rdflib with built-in JSON-LD support."
        except Exception:
            try:
                import rdflib_jsonld  # noqa: F811
                has_jsonld = True
                note = "rdflib with rdflib-jsonld plugin."
            except ImportError:
                has_jsonld = False
                note = ("rdflib is installed but JSON-LD plugin is missing. "
                        "Install with: pip install rdflib-jsonld")

        return DependencyStatus(
            name="rdflib", available=has_jsonld, version=version, note=note,
        )
    except ImportError:
        return DependencyStatus(
            name="rdflib", available=False,
            note="rdflib is not installed. Install with: pip install rdflib",
        )


# =====================================================================
# Test document generation
# =====================================================================

def generate_test_documents() -> list[dict[str, Any]]:
    """Generate 10 test documents spanning jsonld-ex features."""
    docs = []

    # 1. @confidence only
    docs.append({
        "name": "confidence_only",
        "document": {
            "@context": {"@vocab": "http://schema.org/"},
            "@type": "Person",
            "name": {
                "@value": "Alice",
                "@confidence": 0.95,
            },
        },
    })

    # 2. @integrity only
    docs.append({
        "name": "integrity_only",
        "document": {
            "@context": {"@vocab": "http://schema.org/"},
            "@integrity": {
                "http://schema.org/": "sha256-abc123def456",
            },
            "@type": "Thing",
            "name": "Test",
        },
    })

    # 3. @vector only
    docs.append({
        "name": "vector_only",
        "document": {
            "@context": {"@vocab": "http://schema.org/"},
            "@type": "CreativeWork",
            "name": "Embedding Test",
            "@vector": [0.1, -0.2, 0.3, 0.4, -0.5],
        },
    })

    # 4. Provenance (@source + @extractedAt)
    docs.append({
        "name": "provenance_only",
        "document": {
            "@context": {"@vocab": "http://schema.org/"},
            "@type": "Article",
            "headline": {
                "@value": "Test Article",
                "@source": "model:ner-v2",
                "@extractedAt": "2026-01-15T10:00:00Z",
            },
        },
    })

    # 5. Temporal (@validFrom + @validUntil)
    docs.append({
        "name": "temporal_only",
        "document": {
            "@context": {"@vocab": "http://schema.org/"},
            "@type": "Event",
            "name": "Conference",
            "@validFrom": "2026-06-01T00:00:00Z",
            "@validUntil": "2026-06-05T23:59:59Z",
        },
    })

    # 6. All extensions combined
    docs.append({
        "name": "all_extensions",
        "document": {
            "@context": {"@vocab": "http://schema.org/"},
            "@integrity": {"http://schema.org/": "sha256-abc"},
            "@type": "Dataset",
            "name": {
                "@value": "Full Extension Test",
                "@confidence": 0.9,
                "@source": "model:test-v1",
                "@extractedAt": "2026-04-27T00:00:00Z",
            },
            "@vector": [0.1, 0.2, 0.3],
            "@validFrom": "2026-01-01T00:00:00Z",
            "@validUntil": "2026-12-31T23:59:59Z",
        },
    })

    # 7. Nested with extensions at multiple levels
    docs.append({
        "name": "nested_extensions",
        "document": {
            "@context": {"@vocab": "http://schema.org/"},
            "@type": "Organization",
            "name": {
                "@value": "Acme Corp",
                "@confidence": 0.99,
            },
            "member": {
                "@type": "Person",
                "name": {
                    "@value": "Bob",
                    "@confidence": 0.85,
                    "@source": "model:ner-v3",
                },
            },
        },
    })

    # 8. Array of items with extensions
    docs.append({
        "name": "array_with_extensions",
        "document": {
            "@context": {"@vocab": "http://schema.org/"},
            "@type": "ItemList",
            "itemListElement": [
                {"@type": "Thing", "name": {"@value": "A", "@confidence": 0.9}},
                {"@type": "Thing", "name": {"@value": "B", "@confidence": 0.7}},
                {"@type": "Thing", "name": {"@value": "C", "@confidence": 0.5}},
            ],
        },
    })

    # 9. Multiple contexts + @integrity
    docs.append({
        "name": "multiple_contexts_integrity",
        "document": {
            "@context": [
                {"@vocab": "http://schema.org/"},
                {"custom": "http://example.org/ns/"},
            ],
            "@integrity": {
                "http://schema.org/": "sha256-abc",
                "http://example.org/ns/": "sha256-def",
            },
            "@type": "Thing",
            "name": "Multi-context",
        },
    })

    # 10. Minimal standard JSON-LD (control -- no extensions)
    docs.append({
        "name": "minimal_standard",
        "document": {
            "@context": {"@vocab": "http://schema.org/"},
            "@type": "Person",
            "name": "Charlie",
            "email": "charlie@example.org",
        },
    })

    return docs


# =====================================================================
# Extension keyword detection
# =====================================================================

_EXTENSION_KEYWORDS = [
    "@confidence", "@source", "@extractedAt", "@method",
    "@humanVerified", "@vector", "@integrity",
    "@validFrom", "@validUntil", "@mediaType",
]


def _find_extensions_in_output(expanded: Any) -> list[str]:
    """Find which extension keywords appear in expanded output."""
    text = json.dumps(expanded) if not isinstance(expanded, str) else expanded
    return [kw for kw in _EXTENSION_KEYWORDS if kw in text]


def _check_standard_data(original: dict, expanded: Any) -> bool:
    """Check if standard JSON-LD data is preserved after expansion."""
    if expanded is None:
        return False
    text = json.dumps(expanded) if not isinstance(expanded, str) else expanded
    # Check that @type and basic properties are present
    orig_type = original.get("@type", "")
    if orig_type and orig_type.lower() not in text.lower() and \
       f"schema.org/{orig_type}" not in text:
        return False
    return len(text) > 10  # Non-trivial output


# =====================================================================
# PyLD expansion suite
# =====================================================================

def run_pyld_expansion_suite() -> ParserTestResult:
    """Test PyLD expansion on all 10 test documents."""
    result = ParserTestResult(parser_name="PyLD", total_docs=10)

    status = check_pyld_available()
    if not status.available:
        result.parser_available = False
        result.skipped = True
        result.skip_reason = f"PyLD not available: {status.note}"
        # Still populate doc_results as skipped
        for doc_entry in generate_test_documents():
            result.doc_results.append(DocumentResult(
                name=doc_entry["name"], success=False,
                error_type="Skipped", error_msg="PyLD not installed",
            ))
        return result

    result.parser_available = True
    import pyld

    for doc_entry in generate_test_documents():
        doc = doc_entry["document"]
        dr = DocumentResult(name=doc_entry["name"])

        try:
            expanded = pyld.jsonld.expand(doc)
            dr.success = True
            dr.expanded_keys = list(expanded[0].keys()) if expanded else []
            dr.extension_keywords_survived = _find_extensions_in_output(expanded)
            dr.standard_data_preserved = _check_standard_data(doc, expanded)
            result.successes += 1
        except Exception as exc:
            dr.success = False
            dr.error_type = type(exc).__name__
            dr.error_msg = str(exc)[:500]
            result.failures += 1

        result.doc_results.append(dr)

    return result


# =====================================================================
# rdflib parse suite
# =====================================================================

def run_rdflib_parse_suite() -> ParserTestResult:
    """Test rdflib parsing of all 10 test documents."""
    result = ParserTestResult(parser_name="rdflib", total_docs=10)

    status = check_rdflib_available()
    if not status.available:
        result.parser_available = False
        result.skipped = True
        result.skip_reason = f"rdflib JSON-LD not available: {status.note}"
        for doc_entry in generate_test_documents():
            result.doc_results.append(DocumentResult(
                name=doc_entry["name"], success=False,
                error_type="Skipped", error_msg="rdflib JSON-LD not available",
            ))
        return result

    result.parser_available = True
    import rdflib

    for doc_entry in generate_test_documents():
        doc = doc_entry["document"]
        dr = DocumentResult(name=doc_entry["name"])

        try:
            g = rdflib.Graph()
            g.parse(data=json.dumps(doc), format="json-ld")
            dr.success = True
            dr.triple_count = len(g)
            dr.standard_data_preserved = len(g) > 0
            # Check if extension keywords appear as predicates
            all_predicates = set(str(p) for _, p, _ in g)
            dr.extension_keywords_survived = [
                kw for kw in _EXTENSION_KEYWORDS
                if any(kw.lstrip("@") in pred for pred in all_predicates)
            ]
            result.successes += 1
        except Exception as exc:
            dr.success = False
            dr.error_type = type(exc).__name__
            dr.error_msg = str(exc)[:500]
            result.failures += 1

        result.doc_results.append(dr)

    return result


# =====================================================================
# Cross-parser comparison
# =====================================================================

def run_cross_parser_comparison() -> CrossParserResult:
    """Compare behavior across available parsers."""
    result = CrossParserResult()

    pyld_result = run_pyld_expansion_suite()
    rdflib_result = run_rdflib_parse_suite()

    tested = []
    if pyld_result.parser_available:
        tested.append("PyLD")
    if rdflib_result.parser_available:
        tested.append("rdflib")
    result.parsers_tested = tested

    if len(tested) < 2:
        result.finding = (
            f"Only {len(tested)} parser(s) available ({', '.join(tested) or 'none'}). "
            "Cross-parser comparison requires at least 2 parsers. "
            "Install PyLD and/or rdflib with JSON-LD support for full comparison."
        )
        return result

    # Compare: for docs that both parsers successfully parsed,
    # check if they agree on success/failure
    agreements = 0
    comparisons = 0
    for pyld_doc, rdflib_doc in zip(pyld_result.doc_results, rdflib_result.doc_results):
        comparisons += 1
        if pyld_doc.success == rdflib_doc.success:
            agreements += 1

    result.agreement_rate = agreements / comparisons if comparisons > 0 else None

    rate_str = f"{result.agreement_rate:.0%}" if result.agreement_rate is not None else "N/A"
    result.finding = (
        f"Tested {len(tested)} parsers: {', '.join(tested)}. "
        f"Agreement rate: {agreements}/{comparisons} documents ({rate_str}). "
        f"PyLD: {pyld_result.successes}/{pyld_result.total_docs} succeeded. "
        f"rdflib: {rdflib_result.successes}/{rdflib_result.total_docs} succeeded."
    )

    return result


# =====================================================================
# Full experiment
# =====================================================================

def run_en5_5_full() -> EN55FullResult:
    """Run the complete EN5.5 experiment."""
    pyld_status = check_pyld_available()
    rdflib_status = check_rdflib_available()

    return EN55FullResult(
        dependency_status={
            "pyld": pyld_status,
            "rdflib": rdflib_status,
        },
        pyld=run_pyld_expansion_suite(),
        rdflib=run_rdflib_parse_suite(),
        cross_parser=run_cross_parser_comparison(),
    )

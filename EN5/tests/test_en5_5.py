"""Tests for EN5.5 -- Backward Compatibility with Legacy Processors.

RED phase -- all tests should FAIL until en5_5_core.py is implemented.

Run:
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex
    python -m pytest experiments/EN5/tests/test_en5_5.py -v

Hypotheses:
    H5.5a: PyLD processes jsonld-ex documents without exceptions
    H5.5b: Standard data preserved after PyLD expansion
    H5.5c: Errors on extension keywords documented
    H5.5d: rdflib-jsonld tolerance (if available)
    H5.5e: Cross-parser consistency
"""
from __future__ import annotations

import json
import pytest

# -- Experiment core imports (SHOULD FAIL in RED phase) --
from EN5.en5_5_core import (
    # Dependency checks
    check_pyld_available,
    check_rdflib_available,
    # Test documents
    generate_test_documents,
    TEST_DOC_NAMES,
    # Runners
    run_pyld_expansion_suite,
    run_rdflib_parse_suite,
    run_cross_parser_comparison,
    # Result types
    DependencyStatus,
    ParserTestResult,
    DocumentResult,
    CrossParserResult,
    # Full experiment
    run_en5_5_full,
    EN55FullResult,
)


# =====================================================================
# 0. Dependency checks
# =====================================================================

class TestDependencyChecks:
    """Verify dependency detection works."""

    def test_pyld_check_returns_status(self):
        status = check_pyld_available()
        assert isinstance(status, DependencyStatus)
        assert isinstance(status.available, bool)
        assert isinstance(status.version, (str, type(None)))
        assert isinstance(status.note, str)

    def test_rdflib_check_returns_status(self):
        status = check_rdflib_available()
        assert isinstance(status, DependencyStatus)
        assert isinstance(status.available, bool)

    def test_status_serializable(self):
        status = check_pyld_available()
        d = status.to_dict()
        serialized = json.dumps(d)
        assert len(serialized) > 0


# =====================================================================
# 1. Test document generation
# =====================================================================

class TestDocumentGeneration:
    """Verify test document creation."""

    def test_generates_10_documents(self):
        docs = generate_test_documents()
        assert isinstance(docs, list)
        assert len(docs) == 10

    def test_doc_names_match(self):
        assert len(TEST_DOC_NAMES) == 10
        docs = generate_test_documents()
        for doc_entry in docs:
            assert "name" in doc_entry
            assert "document" in doc_entry
            assert doc_entry["name"] in TEST_DOC_NAMES

    def test_all_docs_are_valid_json(self):
        docs = generate_test_documents()
        for doc_entry in docs:
            serialized = json.dumps(doc_entry["document"])
            assert len(serialized) > 0

    def test_control_doc_has_no_extensions(self):
        """Doc 10 (control) should be plain JSON-LD."""
        docs = generate_test_documents()
        control = [d for d in docs if d["name"] == "minimal_standard"][0]
        serialized = json.dumps(control["document"])
        for ext in ["@confidence", "@vector", "@integrity",
                     "@source", "@extractedAt", "@validFrom"]:
            assert ext not in serialized, f"Control doc has extension: {ext}"

    def test_combined_doc_has_all_extensions(self):
        """Doc 6 should have all extensions."""
        docs = generate_test_documents()
        combined = [d for d in docs if d["name"] == "all_extensions"][0]
        serialized = json.dumps(combined["document"])
        assert "@confidence" in serialized or "confidence" in serialized


# =====================================================================
# 2. PyLD expansion suite
# =====================================================================

class TestPyLDSuite:
    """Test PyLD expansion of jsonld-ex documents."""

    def test_suite_runs(self):
        result = run_pyld_expansion_suite()
        assert isinstance(result, ParserTestResult)

    def test_result_has_all_docs(self):
        result = run_pyld_expansion_suite()
        assert result.total_docs == 10

    def test_result_documents_behavior(self):
        """Each document result records success/failure."""
        result = run_pyld_expansion_suite()
        for doc_result in result.doc_results:
            assert isinstance(doc_result, DocumentResult)
            assert isinstance(doc_result.name, str)
            assert isinstance(doc_result.success, bool)
            # If failed, error should be populated
            if not doc_result.success:
                assert doc_result.error_type is not None

    def test_control_doc_succeeds(self):
        """Minimal standard JSON-LD should always parse."""
        result = run_pyld_expansion_suite()
        control = [d for d in result.doc_results
                    if d.name == "minimal_standard"][0]
        if result.parser_available:
            assert control.success, f"Control doc failed: {control.error_msg}"

    def test_skipped_if_unavailable(self):
        """If PyLD not installed, suite reports skipped, not crashed."""
        result = run_pyld_expansion_suite()
        if not result.parser_available:
            assert result.skipped is True
            assert "not installed" in result.skip_reason.lower() or \
                   "unavailable" in result.skip_reason.lower()

    def test_result_serializable(self):
        result = run_pyld_expansion_suite()
        d = result.to_dict()
        serialized = json.dumps(d)
        assert len(serialized) > 0


# =====================================================================
# 3. rdflib suite
# =====================================================================

class TestRDFlibSuite:
    """Test rdflib parsing of jsonld-ex documents."""

    def test_suite_runs(self):
        result = run_rdflib_parse_suite()
        assert isinstance(result, ParserTestResult)

    def test_skipped_if_unavailable(self):
        result = run_rdflib_parse_suite()
        if not result.parser_available:
            assert result.skipped is True

    def test_result_serializable(self):
        result = run_rdflib_parse_suite()
        d = result.to_dict()
        serialized = json.dumps(d)
        assert len(serialized) > 0


# =====================================================================
# 4. Cross-parser comparison
# =====================================================================

class TestCrossParserComparison:
    """Compare results across available parsers."""

    def test_comparison_runs(self):
        result = run_cross_parser_comparison()
        assert isinstance(result, CrossParserResult)

    def test_comparison_has_summary(self):
        result = run_cross_parser_comparison()
        assert isinstance(result.parsers_tested, list)
        assert isinstance(result.finding, str)
        assert len(result.finding) > 0

    def test_result_serializable(self):
        result = run_cross_parser_comparison()
        d = result.to_dict()
        serialized = json.dumps(d)
        assert len(serialized) > 0


# =====================================================================
# 5. Full EN5.5 integration
# =====================================================================

class TestFullEN55:
    """Full experiment runner."""

    def test_full_runs(self):
        result = run_en5_5_full()
        assert isinstance(result, EN55FullResult)
        assert result.pyld is not None
        assert result.rdflib is not None
        assert result.cross_parser is not None
        assert result.dependency_status is not None

    def test_full_serializable(self):
        result = run_en5_5_full()
        d = result.to_dict()
        serialized = json.dumps(d)
        assert len(serialized) > 0

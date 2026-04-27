"""Tests for EN5.2 -- Context Allowlist Enforcement.

RED phase -- all tests should FAIL until en5_2_core.py is implemented.
The jsonld_ex.security imports should succeed (testing the actual library).

Run:
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex
    python -m pytest experiments/EN5/tests/test_en5_2.py -v

Hypotheses:
    H5.2a: is_context_allowed() exact match correctness
    H5.2b: Wildcard pattern matching correctness
    H5.2c: block_remote_contexts=True rejects ALL URLs
    H5.2d: Empty config allows all URLs
"""
from __future__ import annotations

import json
import pytest

# -- Actual library imports (these SHOULD succeed) --
from jsonld_ex.security import is_context_allowed

# -- Experiment core imports (these SHOULD FAIL in RED phase) --
from EN5.en5_2_core import (
    # URL generators
    generate_public_urls,
    generate_ssrf_urls,
    SSRF_URL_CATEGORIES,
    # Allowlist generators
    generate_standard_allowlist,
    # PBT runner
    run_allowlist_pbt,
    # Edge case suite
    run_allowlist_edge_cases,
    # SSRF classification test
    run_ssrf_classification,
    # Latency
    run_allowlist_latency,
    # Result types
    AllowlistPBTResult,
    AllowlistEdgeCaseResult,
    SSRFClassificationResult,
    AllowlistLatencyResult,
)


# =====================================================================
# 0. Smoke tests: verify is_context_allowed API works
# =====================================================================

class TestAllowlistAPIAvailable:
    """Verify the actual library function exists and works."""

    def test_is_context_allowed_exists(self):
        assert callable(is_context_allowed)

    def test_exact_match_allowed(self):
        cfg = {"allowed": ["https://schema.org/"]}
        assert is_context_allowed("https://schema.org/", cfg) is True

    def test_exact_match_rejected(self):
        cfg = {"allowed": ["https://schema.org/"]}
        assert is_context_allowed("https://evil.org/", cfg) is False

    def test_empty_config_allows_all(self):
        assert is_context_allowed("https://anything.org/", {}) is True

    def test_block_remote_rejects_all(self):
        cfg = {"block_remote_contexts": True, "allowed": ["https://schema.org/"]}
        assert is_context_allowed("https://schema.org/", cfg) is False

    def test_pattern_wildcard_match(self):
        cfg = {"patterns": ["https://example.org/contexts/*"]}
        assert is_context_allowed("https://example.org/contexts/v1", cfg) is True

    def test_pattern_wildcard_no_match(self):
        cfg = {"patterns": ["https://example.org/contexts/*"]}
        assert is_context_allowed("https://other.org/contexts/v1", cfg) is False


# =====================================================================
# 1. URL generators
# =====================================================================

class TestURLGenerators:
    """Verify URL generation for PBT and SSRF testing."""

    def test_generate_public_urls_returns_list(self):
        urls = generate_public_urls(n=20)
        assert isinstance(urls, list)
        assert len(urls) == 20

    def test_generate_public_urls_are_https(self):
        urls = generate_public_urls(n=10)
        for url in urls:
            assert url.startswith("https://"), f"Non-HTTPS public URL: {url}"

    def test_generate_ssrf_urls_returns_list(self):
        urls = generate_ssrf_urls()
        assert isinstance(urls, list)
        assert len(urls) >= 20  # At least 20 SSRF URLs

    def test_ssrf_urls_are_dangerous(self):
        """All SSRF URLs should reference internal/private/dangerous endpoints."""
        urls = generate_ssrf_urls()
        for url in urls:
            # Must NOT be a standard public HTTPS URL
            assert not (url.startswith("https://schema.org") or
                        url.startswith("https://w3.org")), (
                f"SSRF URL looks public: {url}"
            )

    def test_ssrf_url_categories_comprehensive(self):
        """SSRF categories cover all major attack vectors."""
        required_categories = {
            "localhost", "private_ipv4", "link_local",
            "non_http_scheme", "ipv6_loopback",
        }
        assert required_categories.issubset(set(SSRF_URL_CATEGORIES.keys())), (
            f"Missing categories: {required_categories - set(SSRF_URL_CATEGORIES.keys())}"
        )

    def test_ssrf_categories_have_examples(self):
        """Each SSRF category has at least one URL."""
        for cat, urls in SSRF_URL_CATEGORIES.items():
            assert len(urls) >= 1, f"Category {cat} is empty"


# =====================================================================
# 2. H5.2a — Exact match correctness
# =====================================================================

class TestH52aExactMatch:
    """is_context_allowed() exact match behavior."""

    def test_listed_url_accepted(self):
        cfg = {"allowed": ["https://schema.org/", "https://w3id.org/security/v2"]}
        assert is_context_allowed("https://schema.org/", cfg) is True
        assert is_context_allowed("https://w3id.org/security/v2", cfg) is True

    def test_unlisted_url_rejected(self):
        cfg = {"allowed": ["https://schema.org/"]}
        assert is_context_allowed("https://evil.org/", cfg) is False
        assert is_context_allowed("http://schema.org/", cfg) is False  # http vs https

    def test_case_sensitivity(self):
        """URL matching should be case-sensitive (URLs are case-sensitive in path)."""
        cfg = {"allowed": ["https://schema.org/"]}
        # Scheme and host are case-insensitive per RFC, but exact match is literal
        result = is_context_allowed("https://Schema.Org/", cfg)
        # Document actual behavior (either is acceptable, just needs to be consistent)
        assert isinstance(result, bool)

    def test_trailing_slash_sensitivity(self):
        """Document whether trailing slash matters."""
        cfg = {"allowed": ["https://schema.org/"]}
        with_slash = is_context_allowed("https://schema.org/", cfg)
        without_slash = is_context_allowed("https://schema.org", cfg)
        # Both results are valid; we document the behavior
        assert with_slash is True  # Exact match
        # without_slash may be True or False depending on implementation

    def test_multiple_allowlisted(self):
        """All allowlisted URLs accepted, others rejected."""
        allowed = [
            "https://schema.org/",
            "https://w3id.org/security/v2",
            "https://w3id.org/jsonld-ex/",
        ]
        cfg = {"allowed": allowed}
        for url in allowed:
            assert is_context_allowed(url, cfg) is True
        assert is_context_allowed("https://not-in-list.org/", cfg) is False

    def test_pbt_exact_match(self):
        """Property-based test for exact match correctness."""
        result = run_allowlist_pbt(n_examples=200)
        assert isinstance(result, AllowlistPBTResult)
        assert result.exact_match_errors == 0, (
            f"H5.2a errors: {result.exact_match_error_details}"
        )


# =====================================================================
# 3. H5.2b — Wildcard pattern matching
# =====================================================================

class TestH52bPatternMatching:
    """Wildcard pattern matching correctness."""

    def test_star_matches_any_suffix(self):
        cfg = {"patterns": ["https://example.org/contexts/*"]}
        assert is_context_allowed("https://example.org/contexts/v1", cfg) is True
        assert is_context_allowed("https://example.org/contexts/v2/sub", cfg) is True
        assert is_context_allowed("https://example.org/other/v1", cfg) is False

    def test_question_mark_matches_single_char(self):
        cfg = {"patterns": ["https://example.org/v?"]}
        assert is_context_allowed("https://example.org/v1", cfg) is True
        assert is_context_allowed("https://example.org/v2", cfg) is True
        assert is_context_allowed("https://example.org/v10", cfg) is False  # 2 chars

    def test_subdomain_wildcard(self):
        cfg = {"patterns": ["https://*.example.org/*"]}
        assert is_context_allowed("https://sub.example.org/ctx", cfg) is True
        assert is_context_allowed("https://deep.sub.example.org/ctx", cfg) is True
        assert is_context_allowed("https://example.org/ctx", cfg) is False  # no subdomain

    def test_no_partial_domain_match(self):
        """Wildcard shouldn't match partial domain names."""
        cfg = {"patterns": ["https://example.org/*"]}
        assert is_context_allowed("https://example.org/anything", cfg) is True
        assert is_context_allowed("https://notexample.org/anything", cfg) is False

    def test_multiple_patterns(self):
        cfg = {"patterns": [
            "https://schema.org/*",
            "https://w3id.org/*",
        ]}
        assert is_context_allowed("https://schema.org/Person", cfg) is True
        assert is_context_allowed("https://w3id.org/security/v2", cfg) is True
        assert is_context_allowed("https://evil.org/anything", cfg) is False

    def test_patterns_and_allowed_combined(self):
        """Both exact allowed list and patterns work together."""
        cfg = {
            "allowed": ["https://schema.org/"],
            "patterns": ["https://w3id.org/*"],
        }
        assert is_context_allowed("https://schema.org/", cfg) is True  # exact
        assert is_context_allowed("https://w3id.org/security/v2", cfg) is True  # pattern
        assert is_context_allowed("https://evil.org/", cfg) is False


# =====================================================================
# 4. H5.2c — Block remote override
# =====================================================================

class TestH52cBlockRemote:
    """block_remote_contexts=True rejects ALL URLs."""

    def test_block_rejects_allowlisted(self):
        cfg = {"block_remote_contexts": True, "allowed": ["https://schema.org/"]}
        assert is_context_allowed("https://schema.org/", cfg) is False

    def test_block_rejects_pattern_match(self):
        cfg = {"block_remote_contexts": True, "patterns": ["https://*"]}
        assert is_context_allowed("https://anything.org/", cfg) is False

    def test_block_rejects_everything(self):
        cfg = {"block_remote_contexts": True}
        urls = [
            "https://schema.org/",
            "http://localhost/",
            "file:///etc/passwd",
            "https://w3.org/",
        ]
        for url in urls:
            assert is_context_allowed(url, cfg) is False, f"Block failed for: {url}"


# =====================================================================
# 5. H5.2d — Empty config permissiveness
# =====================================================================

class TestH52dEmptyConfig:
    """Empty config allows all URLs (backward-compatible default)."""

    def test_empty_dict_allows_all(self):
        urls = [
            "https://schema.org/",
            "http://evil.org/",
            "ftp://internal.org/",
            "file:///etc/passwd",
        ]
        for url in urls:
            assert is_context_allowed(url, {}) is True, f"Empty config rejected: {url}"

    def test_no_keys_allows_all(self):
        """Config with no relevant keys is equivalent to empty."""
        cfg = {"irrelevant_key": "value"}
        assert is_context_allowed("https://anything.org/", cfg) is True


# =====================================================================
# 6. SSRF URL classification
# =====================================================================

class TestSSRFClassification:
    """Verify all SSRF-relevant URLs are blocked with standard allowlist."""

    def test_localhost_variants_blocked(self):
        cfg = {"allowed": ["https://schema.org/"]}
        localhost_urls = [
            "http://localhost/admin",
            "http://localhost:8080/internal",
            "http://127.0.0.1/secrets",
            "http://127.0.0.1:3000/api",
        ]
        for url in localhost_urls:
            assert is_context_allowed(url, cfg) is False, f"SSRF: {url} not blocked"

    def test_private_ip_ranges_blocked(self):
        cfg = {"allowed": ["https://schema.org/"]}
        private_urls = [
            "http://10.0.0.1/internal",
            "http://172.16.0.1/config",
            "http://192.168.1.1/admin",
            "http://192.168.0.100:8080/api",
        ]
        for url in private_urls:
            assert is_context_allowed(url, cfg) is False, f"SSRF: {url} not blocked"

    def test_cloud_metadata_blocked(self):
        cfg = {"allowed": ["https://schema.org/"]}
        metadata_urls = [
            "http://169.254.169.254/latest/meta-data/",
            "http://169.254.169.254/latest/api/token",
        ]
        for url in metadata_urls:
            assert is_context_allowed(url, cfg) is False, f"SSRF: {url} not blocked"

    def test_non_http_schemes_blocked(self):
        cfg = {"allowed": ["https://schema.org/"]}
        scheme_urls = [
            "file:///etc/passwd",
            "file:///C:/Windows/System32/config/SAM",
            "ftp://internal.corp/data",
            "gopher://internal:70/",
        ]
        for url in scheme_urls:
            assert is_context_allowed(url, cfg) is False, f"SSRF: {url} not blocked"

    def test_ipv6_loopback_blocked(self):
        cfg = {"allowed": ["https://schema.org/"]}
        ipv6_urls = [
            "http://[::1]/admin",
            "http://[0:0:0:0:0:0:0:1]/internal",
        ]
        for url in ipv6_urls:
            assert is_context_allowed(url, cfg) is False, f"SSRF: {url} not blocked"

    def test_zero_address_blocked(self):
        cfg = {"allowed": ["https://schema.org/"]}
        zero_urls = [
            "http://0.0.0.0/",
            "http://0.0.0.0:8080/admin",
        ]
        for url in zero_urls:
            assert is_context_allowed(url, cfg) is False, f"SSRF: {url} not blocked"

    def test_full_ssrf_classification_suite(self):
        """Run the complete SSRF classification test."""
        result = run_ssrf_classification()
        assert isinstance(result, SSRFClassificationResult)
        assert result.total_ssrf_urls >= 20
        assert result.blocked == result.total_ssrf_urls, (
            f"SSRF gaps: {result.unblocked_details}"
        )
        assert result.public_urls_accepted > 0  # Sanity: allowed URLs still work


# =====================================================================
# 7. Edge case suite
# =====================================================================

class TestEdgeCaseSuite:
    """Full edge case suite from EN5.2 Phase 2."""

    def test_edge_case_suite_runs(self):
        result = run_allowlist_edge_cases()
        assert isinstance(result, AllowlistEdgeCaseResult)
        assert result.total_cases >= 14  # At least 14 edge cases in design

    def test_edge_case_suite_all_pass(self):
        result = run_allowlist_edge_cases()
        assert result.failed == 0, (
            f"Edge case failures: {result.failure_details}"
        )


# =====================================================================
# 8. Latency
# =====================================================================

class TestAllowlistLatency:
    """Verify allowlist check is fast (expected sub-microsecond)."""

    def test_latency_benchmark_runs(self):
        result = run_allowlist_latency(n_trials=500, n_warmup=50)
        assert isinstance(result, AllowlistLatencyResult)
        assert len(result.measurements) > 0

    def test_latency_has_required_fields(self):
        result = run_allowlist_latency(n_trials=500, n_warmup=50)
        for m in result.measurements:
            assert hasattr(m, "scenario")
            assert hasattr(m, "mean_us")
            assert hasattr(m, "p50_us")
            assert hasattr(m, "p95_us")
            assert hasattr(m, "ci_lower_us")
            assert hasattr(m, "ci_upper_us")

    def test_latency_values_positive(self):
        result = run_allowlist_latency(n_trials=500, n_warmup=50)
        for m in result.measurements:
            assert m.mean_us > 0

    def test_latency_result_serializable(self):
        result = run_allowlist_latency(n_trials=500, n_warmup=50)
        d = result.to_dict()
        serialized = json.dumps(d)
        assert len(serialized) > 0


# =====================================================================
# 9. PBT integration
# =====================================================================

class TestPBTIntegration:
    """Verify PBT runner produces valid structured results."""

    def test_pbt_result_serializable(self):
        result = run_allowlist_pbt(n_examples=50)
        d = result.to_dict()
        assert isinstance(d, dict)
        serialized = json.dumps(d)
        assert len(serialized) > 0

    def test_pbt_result_fields(self):
        result = run_allowlist_pbt(n_examples=50)
        assert result.total_checks > 0
        assert result.exact_match_errors >= 0
        assert result.pattern_match_errors >= 0
        assert result.block_remote_errors >= 0
        assert result.empty_config_errors >= 0

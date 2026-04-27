"""EN5.2 -- Context Allowlist Enforcement core module.

NeurIPS 2026 D&B, Suite EN5 (Security and Integrity), Experiment 2.

Hypotheses:
    H5.2a: is_context_allowed() exact match correctness
    H5.2b: Wildcard pattern matching correctness
    H5.2c: block_remote_contexts=True rejects ALL URLs
    H5.2d: Empty config allows all URLs

Tested function (from jsonld_ex.security):
    is_context_allowed
"""
from __future__ import annotations

import json
import random
import string
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

from jsonld_ex.security import is_context_allowed

# -- sys.path setup --
import sys
_SCRIPT_DIR = Path(__file__).resolve().parent
_EXPERIMENTS_ROOT = _SCRIPT_DIR.parent
for p in [str(_SCRIPT_DIR), str(_EXPERIMENTS_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from infra.stats import bootstrap_ci


# =====================================================================
# SSRF URL categories
# =====================================================================

SSRF_URL_CATEGORIES: dict[str, list[str]] = {
    "localhost": [
        "http://localhost/admin",
        "http://localhost:8080/internal",
        "http://localhost:3000/api/config",
    ],
    "loopback_ipv4": [
        "http://127.0.0.1/secrets",
        "http://127.0.0.1:3000/api",
        "http://127.0.0.1:9090/metrics",
    ],
    "private_ipv4": [
        "http://10.0.0.1/internal",
        "http://10.255.255.1/config",
        "http://172.16.0.1/config",
        "http://172.31.255.255/admin",
        "http://192.168.1.1/admin",
        "http://192.168.0.100:8080/api",
    ],
    "link_local": [
        "http://169.254.169.254/latest/meta-data/",
        "http://169.254.169.254/latest/api/token",
    ],
    "ipv6_loopback": [
        "http://[::1]/admin",
        "http://[0:0:0:0:0:0:0:1]/internal",
    ],
    "zero_address": [
        "http://0.0.0.0/",
        "http://0.0.0.0:8080/admin",
    ],
    "non_http_scheme": [
        "file:///etc/passwd",
        "file:///C:/Windows/System32/config/SAM",
        "ftp://internal.corp/data",
        "gopher://internal:70/",
        "data:text/html,<script>alert(1)</script>",
    ],
}


# =====================================================================
# URL generators
# =====================================================================

def generate_public_urls(n: int = 20, seed: int = 42) -> list[str]:
    """Generate n plausible public HTTPS URLs for testing."""
    rng = random.Random(seed)
    domains = [
        "schema.org", "w3.org", "w3id.org", "example.org",
        "purl.org", "xmlns.com", "dbpedia.org", "wikidata.org",
        "iana.org", "json-ld.org", "ogp.me", "microformats.org",
    ]
    paths = [
        "/", "/v1", "/v2", "/context", "/vocab", "/ns",
        "/ontology", "/terms", "/security/v2", "/credentials/v1",
    ]
    urls = []
    for _ in range(n):
        domain = rng.choice(domains)
        path = rng.choice(paths)
        urls.append(f"https://{domain}{path}")
    return urls


def generate_ssrf_urls() -> list[str]:
    """Return all SSRF test URLs from all categories."""
    urls = []
    for cat_urls in SSRF_URL_CATEGORIES.values():
        urls.extend(cat_urls)
    return urls


def generate_standard_allowlist() -> dict[str, Any]:
    """Generate a standard allowlist config for testing."""
    return {
        "allowed": [
            "https://schema.org/",
            "https://w3id.org/security/v2",
            "https://w3id.org/jsonld-ex/",
        ],
        "patterns": [
            "https://w3.org/*",
        ],
    }


# =====================================================================
# Result dataclasses
# =====================================================================

@dataclass
class AllowlistPBTResult:
    """Results from property-based allowlist testing."""
    total_checks: int = 0
    exact_match_errors: int = 0
    pattern_match_errors: int = 0
    block_remote_errors: int = 0
    empty_config_errors: int = 0
    exact_match_error_details: list[dict] = field(default_factory=list)
    pattern_match_error_details: list[dict] = field(default_factory=list)
    block_remote_error_details: list[dict] = field(default_factory=list)
    empty_config_error_details: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EdgeCaseEntry:
    """Single edge case result."""
    name: str
    passed: bool
    error: str | None = None


@dataclass
class AllowlistEdgeCaseResult:
    """Results from edge case testing."""
    total_cases: int = 0
    passed: int = 0
    failed: int = 0
    cases: list[EdgeCaseEntry] = field(default_factory=list)

    @property
    def failure_details(self) -> list[str]:
        return [f"{c.name}: {c.error}" for c in self.cases if not c.passed]

    def to_dict(self) -> dict:
        return {
            "total_cases": self.total_cases,
            "passed": self.passed,
            "failed": self.failed,
            "cases": [asdict(c) for c in self.cases],
        }


@dataclass
class SSRFClassificationResult:
    """Results from SSRF URL classification test."""
    total_ssrf_urls: int = 0
    blocked: int = 0
    total_public_urls: int = 0
    public_urls_accepted: int = 0
    unblocked_details: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LatencyMeasurement:
    """Single latency measurement for one scenario."""
    scenario: str
    mean_us: float = 0.0
    p50_us: float = 0.0
    p95_us: float = 0.0
    p99_us: float = 0.0
    ci_lower_us: float = 0.0
    ci_upper_us: float = 0.0
    n_trials: int = 0


@dataclass
class AllowlistLatencyResult:
    """Results from allowlist latency benchmarking."""
    measurements: list[LatencyMeasurement] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"measurements": [asdict(m) for m in self.measurements]}


# =====================================================================
# PBT runner
# =====================================================================

def run_allowlist_pbt(
    n_examples: int = 1000,
    seed: int = 42,
) -> AllowlistPBTResult:
    """Run property-based allowlist tests.

    For random (url, config) pairs, verify:
      - Exact match: url in allowed -> True; url not in allowed -> False
      - Pattern match: url matches pattern -> True; no match -> False
      - Block remote: always False
      - Empty config: always True
    """
    rng = random.Random(seed)
    result = AllowlistPBTResult()

    domains = [
        "schema.org", "w3.org", "example.org", "evil.org",
        "test.com", "internal.net", "purl.org", "dbpedia.org",
    ]
    paths = ["/", "/v1", "/ctx", "/vocab", "/ns", "/ontology"]

    for i in range(n_examples):
        url = f"https://{rng.choice(domains)}{rng.choice(paths)}"

        # --- H5.2a: Exact match ---
        n_allowed = rng.randint(1, 4)
        allowed_urls = [f"https://{rng.choice(domains)}{rng.choice(paths)}"
                        for _ in range(n_allowed)]
        cfg = {"allowed": allowed_urls}

        expected = url in allowed_urls
        actual = is_context_allowed(url, cfg)
        result.total_checks += 1
        if actual != expected:
            result.exact_match_errors += 1
            result.exact_match_error_details.append({
                "url": url, "allowed": allowed_urls,
                "expected": expected, "actual": actual,
            })

        # --- H5.2c: Block remote ---
        cfg_block = {"block_remote_contexts": True, "allowed": allowed_urls}
        actual_block = is_context_allowed(url, cfg_block)
        result.total_checks += 1
        if actual_block is not False:
            result.block_remote_errors += 1
            result.block_remote_error_details.append({
                "url": url, "config": cfg_block, "actual": actual_block,
            })

        # --- H5.2d: Empty config ---
        actual_empty = is_context_allowed(url, {})
        result.total_checks += 1
        if actual_empty is not True:
            result.empty_config_errors += 1
            result.empty_config_error_details.append({
                "url": url, "actual": actual_empty,
            })

    # --- H5.2b: Pattern match (separate loop for controlled testing) ---
    patterns_test = [
        ("https://example.org/contexts/*", "https://example.org/contexts/v1", True),
        ("https://example.org/contexts/*", "https://example.org/other/v1", False),
        ("https://*.example.org/*", "https://sub.example.org/ctx", True),
        ("https://*.example.org/*", "https://notexample.org/ctx", False),
        ("https://example.org/v?", "https://example.org/v1", True),
        ("https://example.org/v?", "https://example.org/v10", False),
    ]
    for pattern, url, expected in patterns_test:
        cfg_pat = {"patterns": [pattern]}
        actual_pat = is_context_allowed(url, cfg_pat)
        result.total_checks += 1
        if actual_pat != expected:
            result.pattern_match_errors += 1
            result.pattern_match_error_details.append({
                "pattern": pattern, "url": url,
                "expected": expected, "actual": actual_pat,
            })

    return result


# =====================================================================
# SSRF classification
# =====================================================================

def run_ssrf_classification() -> SSRFClassificationResult:
    """Test all SSRF URLs against a standard allowlist."""
    cfg = generate_standard_allowlist()
    ssrf_urls = generate_ssrf_urls()
    public_urls = [
        "https://schema.org/",
        "https://w3id.org/security/v2",
        "https://w3id.org/jsonld-ex/",
        "https://w3.org/TR/json-ld11/",
    ]

    result = SSRFClassificationResult(
        total_ssrf_urls=len(ssrf_urls),
        total_public_urls=len(public_urls),
    )

    for url in ssrf_urls:
        if not is_context_allowed(url, cfg):
            result.blocked += 1
        else:
            result.unblocked_details.append(url)

    for url in public_urls:
        if is_context_allowed(url, cfg):
            result.public_urls_accepted += 1

    return result


# =====================================================================
# Edge case suite
# =====================================================================

def run_allowlist_edge_cases() -> AllowlistEdgeCaseResult:
    """Run hand-crafted edge cases from EN5.2 Phase 2."""
    cases: list[EdgeCaseEntry] = []

    def _run(name: str, fn):
        try:
            fn()
            cases.append(EdgeCaseEntry(name=name, passed=True))
        except Exception as exc:
            cases.append(EdgeCaseEntry(name=name, passed=False, error=str(exc)))

    # 1. Exact match
    def ec_exact():
        cfg = {"allowed": ["https://schema.org/"]}
        assert is_context_allowed("https://schema.org/", cfg) is True
    _run("exact_match_accepted", ec_exact)

    # 2. Trailing slash sensitivity
    def ec_trailing():
        cfg = {"allowed": ["https://schema.org/"]}
        r_with = is_context_allowed("https://schema.org/", cfg)
        r_without = is_context_allowed("https://schema.org", cfg)
        assert r_with is True, "With-slash should match"
        # Document: r_without may differ (not assertion, just observation)
    _run("trailing_slash", ec_trailing)

    # 3. Wildcard subdomain match
    def ec_wildcard_sub():
        cfg = {"patterns": ["https://*.example.org/*"]}
        assert is_context_allowed("https://sub.example.org/v1", cfg) is True
    _run("wildcard_subdomain_match", ec_wildcard_sub)

    # 4. Wildcard no match
    def ec_wildcard_no():
        cfg = {"patterns": ["https://*.example.org/*"]}
        assert is_context_allowed("https://evil.org/v1", cfg) is False
    _run("wildcard_no_match", ec_wildcard_no)

    # 5. Single char wildcard
    def ec_single_char():
        cfg = {"patterns": ["https://example.org/v?"]}
        assert is_context_allowed("https://example.org/v1", cfg) is True
    _run("single_char_wildcard", ec_single_char)

    # 6. Block overrides allow
    def ec_block():
        cfg = {"allowed": ["https://schema.org/"], "block_remote_contexts": True}
        assert is_context_allowed("https://schema.org/", cfg) is False
    _run("block_overrides_allow", ec_block)

    # 7. Empty config
    def ec_empty():
        assert is_context_allowed("https://anything.org/", {}) is True
    _run("empty_config_allows_all", ec_empty)

    # 8. Localhost blocked
    def ec_localhost():
        cfg = {"allowed": ["https://schema.org/"]}
        assert is_context_allowed("http://localhost:8080/ctx", cfg) is False
    _run("localhost_blocked", ec_localhost)

    # 9. Private IP blocked
    def ec_private_ip():
        cfg = {"allowed": ["https://schema.org/"]}
        assert is_context_allowed("http://192.168.1.1/ctx", cfg) is False
    _run("private_ip_blocked", ec_private_ip)

    # 10. Zero address blocked
    def ec_zero():
        cfg = {"allowed": ["https://schema.org/"]}
        assert is_context_allowed("http://0.0.0.0/ctx", cfg) is False
    _run("zero_address_blocked", ec_zero)

    # 11. File URI blocked
    def ec_file():
        cfg = {"allowed": ["https://schema.org/"]}
        assert is_context_allowed("file:///etc/passwd", cfg) is False
    _run("file_uri_blocked", ec_file)

    # 12. FTP URI blocked
    def ec_ftp():
        cfg = {"allowed": ["https://schema.org/"]}
        assert is_context_allowed("ftp://evil.org/ctx", cfg) is False
    _run("ftp_uri_blocked", ec_ftp)

    # 13. Unicode domain blocked (not in allowlist)
    def ec_unicode():
        cfg = {"allowed": ["https://schema.org/"]}
        assert is_context_allowed("https://xn--e1afmapc.org/ctx", cfg) is False
    _run("unicode_domain_blocked", ec_unicode)

    # 14. Empty URL blocked (when allowlist exists)
    def ec_empty_url():
        cfg = {"allowed": ["https://schema.org/"]}
        assert is_context_allowed("", cfg) is False
    _run("empty_url_blocked", ec_empty_url)

    result = AllowlistEdgeCaseResult()
    result.cases = cases
    result.total_cases = len(cases)
    result.passed = sum(1 for c in cases if c.passed)
    result.failed = sum(1 for c in cases if not c.passed)
    return result


# =====================================================================
# Latency benchmark
# =====================================================================

def run_allowlist_latency(
    n_trials: int = 10_000,
    n_warmup: int = 100,
) -> AllowlistLatencyResult:
    """Benchmark is_context_allowed latency across scenarios."""
    result = AllowlistLatencyResult()

    scenarios = {
        "exact_match_hit": (
            "https://schema.org/",
            {"allowed": ["https://schema.org/", "https://w3.org/"]},
        ),
        "exact_match_miss": (
            "https://evil.org/",
            {"allowed": ["https://schema.org/", "https://w3.org/"]},
        ),
        "pattern_match_hit": (
            "https://example.org/contexts/v2",
            {"patterns": ["https://example.org/contexts/*"]},
        ),
        "pattern_match_miss": (
            "https://other.org/v1",
            {"patterns": ["https://example.org/contexts/*"]},
        ),
        "block_remote": (
            "https://schema.org/",
            {"block_remote_contexts": True},
        ),
        "empty_config": (
            "https://anything.org/",
            {},
        ),
        "large_allowlist_hit": (
            "https://domain_050.org/",
            {"allowed": [f"https://domain_{i:03d}.org/" for i in range(100)]},
        ),
        "large_allowlist_miss": (
            "https://not-in-list.org/",
            {"allowed": [f"https://domain_{i:03d}.org/" for i in range(100)]},
        ),
    }

    for scenario_name, (url, cfg) in scenarios.items():
        # Warm up
        for _ in range(n_warmup):
            is_context_allowed(url, cfg)

        times = []
        for _ in range(n_trials):
            t0 = time.perf_counter_ns()
            is_context_allowed(url, cfg)
            t1 = time.perf_counter_ns()
            times.append((t1 - t0) / 1000.0)  # ns -> us

        arr = np.array(times)
        ci_lo, _, ci_hi = bootstrap_ci(arr)

        result.measurements.append(LatencyMeasurement(
            scenario=scenario_name,
            mean_us=float(np.mean(arr)),
            p50_us=float(np.percentile(arr, 50)),
            p95_us=float(np.percentile(arr, 95)),
            p99_us=float(np.percentile(arr, 99)),
            ci_lower_us=ci_lo,
            ci_upper_us=ci_hi,
            n_trials=n_trials,
        ))

    return result

"""Tests for the response parser.

TDD: Written BEFORE parse_response.py implementation.
The parser extracts (answer, confidence) from raw model outputs regardless
of format. It must handle JSON, jsonld-ex, plain text, verbal confidence
expressions, and garbled/unparseable output gracefully.
"""

import pytest


# ---------------------------------------------------------------------------
# JSON output parsing
# ---------------------------------------------------------------------------

class TestJSONParsing:
    """Parser must extract answer and confidence from JSON outputs."""

    def test_parse_c2_style_json(self):
        from src.parse_response import parse_response

        raw = '{"entity": "NovaTech", "relation": "headquartered_in", "value": "Cedarpoint"}'
        result = parse_response(raw)
        assert result.answer == "Cedarpoint"
        assert result.confidence is None  # C2 has no confidence field

    def test_parse_c7_style_json(self):
        from src.parse_response import parse_response

        raw = '{"entity": "NovaTech", "relation": "hq", "value": "Cedarpoint", "confidence": 0.92}'
        result = parse_response(raw)
        assert result.answer == "Cedarpoint"
        assert abs(result.confidence - 0.92) < 1e-6

    def test_parse_c4_style_jsonldex(self):
        from src.parse_response import parse_response

        raw = """{
          "@context": ["https://schema.org/", "https://jsonld-ex.org/context/v1"],
          "@type": "Organization",
          "name": "NovaTech",
          "headquartered_in": "Cedarpoint",
          "@opinion": {"belief": 0.94, "disbelief": 0.02, "uncertainty": 0.04, "base_rate": 0.5}
        }"""
        result = parse_response(raw)
        assert result.answer == "Cedarpoint"
        # Confidence should be projected probability: b + a*u = 0.94 + 0.5*0.04 = 0.96
        assert result.confidence is not None
        assert abs(result.confidence - 0.96) < 1e-4

    def test_parse_json_with_nested_opinion(self):
        from src.parse_response import parse_response

        raw = """{
          "@type": "Answer",
          "text": "Rouen",
          "detail": {
            "@opinion": {"belief": 0.80, "disbelief": 0.05, "uncertainty": 0.15, "base_rate": 0.5}
          }
        }"""
        result = parse_response(raw)
        assert result.answer == "Rouen"
        assert result.confidence is not None

    def test_parse_json_ld_with_name_and_value(self):
        """JSON-LD where the 'answer' is a relation value, not 'text'."""
        from src.parse_response import parse_response

        raw = """{
          "@type": "Organization",
          "name": "NovaTech",
          "industry": "quantum computing"
        }"""
        result = parse_response(raw)
        # Should find "quantum computing" as the answer
        assert result.answer is not None
        assert len(result.answer) > 0


# ---------------------------------------------------------------------------
# Plain text parsing
# ---------------------------------------------------------------------------

class TestTextParsing:
    """Parser must extract answer and verbal confidence from plain text."""

    def test_parse_plain_text_answer(self):
        from src.parse_response import parse_response

        raw = "NovaTech Industries is headquartered in Cedarpoint, Valoria."
        result = parse_response(raw)
        assert result.answer is not None
        assert "Cedarpoint" in result.answer

    def test_parse_text_with_numeric_confidence(self):
        from src.parse_response import parse_response

        raw = "NovaTech is headquartered in Cedarpoint. Confidence: 0.85"
        result = parse_response(raw)
        assert result.confidence is not None
        assert abs(result.confidence - 0.85) < 1e-6

    def test_parse_text_with_verbal_confidence_certain(self):
        from src.parse_response import parse_response

        raw = "NovaTech is headquartered in Cedarpoint. I am certain of this."
        result = parse_response(raw)
        assert result.confidence is not None
        assert result.confidence >= 0.9

    def test_parse_text_with_verbal_confidence_uncertain(self):
        from src.parse_response import parse_response

        raw = "I'm not sure, but NovaTech might be in Cedarpoint."
        result = parse_response(raw)
        assert result.confidence is not None
        assert result.confidence <= 0.5

    def test_parse_text_with_verbal_confidence_moderate(self):
        from src.parse_response import parse_response

        raw = "NovaTech is fairly confident to be headquartered in Cedarpoint."
        result = parse_response(raw)
        # "fairly confident" maps to ~0.8 per protocol config
        assert result.confidence is not None

    def test_parse_text_no_confidence_signal(self):
        """Text with no confidence expression → confidence=None."""
        from src.parse_response import parse_response

        raw = "NovaTech Industries is headquartered in Cedarpoint, Valoria."
        result = parse_response(raw)
        # No verbal or numeric confidence signal
        assert result.confidence is None

    def test_parse_abstention(self):
        """Model explicitly says it doesn't know."""
        from src.parse_response import parse_response

        raw = "I don't have enough information to answer confidently."
        result = parse_response(raw)
        assert result.is_abstention is True
        assert result.confidence is not None
        assert result.confidence <= 0.1


# ---------------------------------------------------------------------------
# Edge cases and malformed output
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Parser must handle malformed, empty, and mixed outputs gracefully."""

    def test_empty_string(self):
        from src.parse_response import parse_response

        result = parse_response("")
        assert result.answer is None
        assert result.is_parseable is False

    def test_whitespace_only(self):
        from src.parse_response import parse_response

        result = parse_response("   \n\t  ")
        assert result.answer is None
        assert result.is_parseable is False

    def test_malformed_json(self):
        """Broken JSON should fall back to text parsing."""
        from src.parse_response import parse_response

        raw = '{"entity": "NovaTech", "value": "Cedarpoint"'  # missing closing brace
        result = parse_response(raw)
        # Should still extract something from the text
        assert result.is_parseable is True
        assert "Cedarpoint" in (result.answer or "")

    def test_json_wrapped_in_markdown(self):
        """Models often wrap JSON in ```json ... ``` blocks."""
        from src.parse_response import parse_response

        raw = '```json\n{"entity": "NovaTech", "value": "Cedarpoint", "confidence": 0.9}\n```'
        result = parse_response(raw)
        assert result.answer == "Cedarpoint"
        assert result.confidence is not None

    def test_mixed_text_and_json(self):
        """Text preamble followed by JSON."""
        from src.parse_response import parse_response

        raw = 'Here is the answer:\n{"entity": "NovaTech", "value": "Cedarpoint"}'
        result = parse_response(raw)
        assert result.answer == "Cedarpoint"

    def test_confidence_out_of_range_clamped(self):
        from src.parse_response import parse_response

        raw = '{"value": "Cedarpoint", "confidence": 1.5}'
        result = parse_response(raw)
        assert result.confidence is not None
        assert result.confidence <= 1.0

    def test_negative_confidence_clamped(self):
        from src.parse_response import parse_response

        raw = '{"value": "Cedarpoint", "confidence": -0.3}'
        result = parse_response(raw)
        assert result.confidence is not None
        assert result.confidence >= 0.0


# ---------------------------------------------------------------------------
# ParseResult structure
# ---------------------------------------------------------------------------

class TestParseResult:
    """ParseResult must have the expected fields."""

    def test_parse_result_fields(self):
        from src.parse_response import parse_response

        result = parse_response("Test answer")
        assert hasattr(result, "answer")
        assert hasattr(result, "confidence")
        assert hasattr(result, "is_abstention")
        assert hasattr(result, "is_parseable")
        assert hasattr(result, "raw")

    def test_raw_preserved(self):
        from src.parse_response import parse_response

        raw = "Some model output"
        result = parse_response(raw)
        assert result.raw == raw

    def test_parse_result_repr(self):
        """ParseResult should have a useful string representation."""
        from src.parse_response import parse_response

        result = parse_response("Test")
        repr_str = repr(result)
        assert "ParseResult" in repr_str

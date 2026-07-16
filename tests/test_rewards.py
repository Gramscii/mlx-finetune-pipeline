"""Unit tests for the reward registry — pytest only, no encoder/MLX needed
(semantic rewards lazy-load their encoder and are exercised via a stub)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import rewards as R


class TestExactMatch:
    def test_match_and_mismatch(self):
        assert R.exact_match("route_a,route_b", "route_a,route_b") == 1.0
        assert R.exact_match("route_a", "route_b") == 0.0

    def test_whitespace_normalized(self):
        assert R.exact_match("  route_a,route_b \n", "route_a,route_b") == 1.0
        assert R.exact_match("a  b", "a b") == 1.0

    def test_empty_reference_is_abstention(self):
        # A router that must answer NOTHING: empty candidate = correct.
        assert R.exact_match("", "") == 1.0
        assert R.exact_match("   \n", "") == 1.0
        assert R.exact_match("route_a", "") == 0.0
        assert R.exact_match("", "route_a") == 0.0

    def test_case_sensitive(self):
        assert R.exact_match("Route_A", "route_a") == 0.0  # labels are exact


class TestExactMatchSet:
    def test_order_insensitive(self):
        assert R.exact_match_set("b, a", "a,b") == 1.0
        assert R.exact_match_set("a", "a,b") == 0.0

    def test_empty_sets(self):
        assert R.exact_match_set("", "") == 1.0
        assert R.exact_match_set("a", "") == 0.0

    def test_duplicates_collapse(self):
        assert R.exact_match_set("a,a,b", "a,b") == 1.0


class TestRegistry:
    def test_known_names(self):
        for name in ("exact_match", "exact_match_set", "semantic", "paper_blend"):
            assert callable(R.get_reward(name))

    def test_unknown_name_fails_loud(self):
        with pytest.raises(KeyError):
            R.get_reward("does_not_exist")


class _StubEncoder:
    """Distinct texts → distinct basis vectors (cosine 1.0 iff same text).
    A dict assigns the next free axis per new text: deterministic across
    processes, no hash salting."""

    def __init__(self):
        self._axis: dict[str, int] = {}

    def encode(self, texts, normalize_embeddings=True):
        import numpy as np
        vecs = []
        for t in texts:
            if t not in self._axis:
                self._axis[t] = len(self._axis)
            v = np.zeros(max(len(self._axis), 1))
            v[self._axis[t]] = 1.0
            vecs.append(v)
        width = max(len(v) for v in vecs)
        return np.stack([np.pad(v, (0, width - len(v))) for v in vecs])


class TestSemanticWithStub:
    def setup_method(self):
        R._ENCODER = _StubEncoder()

    def teardown_method(self):
        R._ENCODER = None

    def test_identical_text_scores_high(self):
        assert R.semantic("same text", "same text", encoder_model="stub") == 1.0

    def test_empty_response_zero(self):
        assert R.semantic("", "reference", encoder_model="stub") == 0.0

    def test_paper_blend_components(self):
        # identical text: semantic 1 + exact 1 + format 1 + reasoning 1 = 4
        got = R.paper_blend("<think>why</think><answer>x</answer>",
                            "<think>why</think><answer>x</answer>",
                            encoder_model="stub",
                            format_regexes=[r"<answer>.*?</answer>"],
                            reasoning_regex=r"<think>(.*?)</think>")
        assert got == 4.0

    def test_paper_blend_empty_reasoning_not_rewarded(self):
        got = R.paper_blend("<think></think>ans", "other",
                            encoder_model="stub",
                            reasoning_regex=r"<think>(.*?)</think>")
        assert got < 1.0

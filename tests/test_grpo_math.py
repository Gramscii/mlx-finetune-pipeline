"""Unit tests for the pure GRPO helpers — run with plain pytest, no MLX needed."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from grpo_math import EarlyStopper, best_checkpoint, group_advantages, response_target_span


class TestGroupAdvantages:
    def test_normalized_mean_zero_std_one(self):
        adv = group_advantages([1.0, 0.0, 1.0, 0.0])
        assert abs(sum(adv)) < 1e-9
        assert adv[0] > 0 > adv[1]

    def test_identical_rewards_give_zero_signal(self):
        assert group_advantages([0.5, 0.5, 0.5]) == [0.0, 0.0, 0.0]
        assert group_advantages([0.0, 0.0]) == [0.0, 0.0]  # all-wrong: no gradient noise

    def test_single_candidate(self):
        assert group_advantages([1.0]) == [0.0]


class TestResponseSpan:
    def test_span_covers_only_response(self):
        # full = 5 prompt + 3 response tokens → targets frame [4, 7)
        assert response_target_span(5, 8) == (4, 7)

    def test_empty_response(self):
        assert response_target_span(5, 5) == (0, 0)

    def test_degenerate_prompt(self):
        assert response_target_span(0, 3) == (0, 0)

    def test_single_response_token(self):
        s = response_target_span(4, 5)
        assert s[1] - s[0] == 1


class TestEarlyStopper:
    def test_improvement_resets_patience(self):
        es = EarlyStopper(patience=2, warmup=0)
        assert es.update(0.5) is True
        assert es.update(0.5) is False       # not > best + delta
        assert es.update(0.6) is True
        assert not es.should_stop

    def test_stops_after_patience(self):
        es = EarlyStopper(patience=2, warmup=0)
        es.update(0.5); es.update(0.5); es.update(0.5)
        assert es.should_stop

    def test_warmup_defers_stop(self):
        es = EarlyStopper(patience=1, warmup=10)
        es.update(0.5); es.update(0.4); es.update(0.4)
        assert not es.should_stop            # still inside warmup


class TestBestCheckpoint:
    def _mk(self, tmp_path, iters):
        d = tmp_path / "adapters"; d.mkdir()
        for i in iters:
            (d / f"{i:07d}_adapters.safetensors").write_bytes(b"x")
        return d

    def test_picks_lowest_val_loss_not_newest(self, tmp_path):
        d = self._mk(tmp_path, [500, 1000, 1500])
        log = ("Iter 500: Val loss 2.100, ...\n"
               "Iter 1000: Val loss 1.900, ...\n"
               "Iter 1500: Val loss 2.400, ...\n")   # newest is WORSE
        assert best_checkpoint(log, d).name == "0001000_adapters.safetensors"

    def test_tie_breaks_on_lowest_iteration(self, tmp_path):
        d = self._mk(tmp_path, [500, 1000])
        log = ("Iter 500: Val loss 2.000, ...\n"
               "Iter 1000: Val loss 2.000, ...\n")   # exact tie
        assert best_checkpoint(log, d).name == "0000500_adapters.safetensors"

    def test_falls_back_to_final(self, tmp_path):
        d = tmp_path / "adapters"; d.mkdir()
        (d / "adapters.safetensors").write_bytes(b"x")
        assert best_checkpoint("no val lines here", d).name == "adapters.safetensors"

    def test_raises_when_empty(self, tmp_path):
        d = tmp_path / "adapters"; d.mkdir()
        with pytest.raises(FileNotFoundError):
            best_checkpoint("", d)

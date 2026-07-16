"""Pure math/bookkeeping helpers for GRPO training. No MLX imports: everything
here is plain Python/NumPy so it is unit-testable without Apple-Silicon deps
(the differentiable loss itself lives in grpo_train.py, built from MLX ops).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def group_advantages(rewards: list[float], eps: float = 1e-6) -> list[float]:
    """GRPO group-relative advantages: (r - mean) / (std + eps), computed within
    one prompt's candidate group. If every candidate got the same reward there is
    no learning signal — return zeros rather than dividing noise by ~0."""
    r = np.asarray(rewards, dtype=np.float64)
    std = float(r.std())
    if std < eps:
        return [0.0] * len(rewards)
    return list((r - r.mean()) / (std + eps))


def response_target_span(prompt_len: int, total_len: int) -> tuple[int, int]:
    """Span of RESPONSE positions in the shifted-targets frame.

    With inputs = tokens[:-1] and targets = tokens[1:], the response tokens
    (indices [prompt_len, total_len) of the full sequence) sit at target indices
    [prompt_len - 1, total_len - 1). The prompt never contributes to the loss."""
    if prompt_len < 1 or total_len <= prompt_len:
        return (0, 0)
    return (prompt_len - 1, total_len - 1)


class EarlyStopper:
    """Best-reward tracking with patience. `update` returns True when the new
    average reward is a real improvement (worth checkpointing)."""

    def __init__(self, patience: int, min_delta: float = 0.01, warmup: int = 10):
        self.patience = patience
        self.min_delta = min_delta
        self.warmup = warmup
        self.best = float("-inf")
        self.stale = 0
        self.seen = 0

    def update(self, value: float) -> bool:
        self.seen += 1
        if value > self.best + self.min_delta:
            self.best = value
            self.stale = 0
            return True
        self.stale += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.seen >= self.warmup and self.stale >= self.patience


_VAL_RE = re.compile(r"Iter (\d+): Val loss ([0-9.]+)")
_CKPT_RE = re.compile(r"^(\d+)_adapters\.safetensors$")


def best_checkpoint(log_text: str, adapter_dir: str | Path) -> Path:
    """The checkpoint at the iteration with the LOWEST validation loss.

    mlx_lm writes `Iter N: Val loss X` lines every `steps_per_eval` and numbered
    `NNNNNNN_adapters.safetensors` files every `save_every` (keep the two equal in
    the config so every evaluated iter has a file).

    Known one-step offset (mlx_lm loop order): the val loss at iter N is measured
    just BEFORE the N-th update, while the N-numbered file is saved just AFTER
    it — so the chosen file is one optimizer step past the exactly-validated
    weights. The skew is uniform across candidates (all post-step weights ranked
    by their pre-step losses), which leaves the argmin ranking meaningful.

    Ties break on the LOWEST iteration (deterministic). Falls back to the final
    `adapters.safetensors` — with a stderr warning — when no numbered file
    matches a val line."""
    import sys

    adapter_dir = Path(adapter_dir)
    saved = {}
    for f in adapter_dir.iterdir() if adapter_dir.is_dir() else []:
        m = _CKPT_RE.match(f.name)
        if m:
            saved[int(m.group(1))] = f
    evaluated = {int(i): float(v) for i, v in _VAL_RE.findall(log_text)}
    scored = {it: evaluated[it] for it in saved if it in evaluated}
    if scored:
        return saved[min(scored, key=lambda it: (scored[it], it))]
    final = adapter_dir / "adapters.safetensors"
    if final.exists():
        print(f"WARNING: no numbered checkpoint matches a val-loss line — "
              f"falling back to the LAST checkpoint {final} (not val-selected)",
              file=sys.stderr)
        return final
    raise FileNotFoundError(
        f"no adapter checkpoints found in {adapter_dir} — did training run?")

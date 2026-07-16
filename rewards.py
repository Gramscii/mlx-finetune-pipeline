"""Reward functions for GRPO — pluggable via `reward:` in config_grpo.yaml.

Two families:
- `exact_match` / `exact_match_set`: deterministic, language-agnostic. For
  classifiers and routers whose gold answer is a short label, a comma-separated
  list, or EMPTY (abstention). No encoder model needed.
- `semantic` / `paper_blend`: the encoder-based rewards of arXiv:2509.13081
  (cosine similarity to a reference explanation, optionally blended with binary
  answer/format/reasoning components). For prose tasks with reference answers.

Pure Python except the encoder, which is lazy-loaded only when a semantic
reward is actually selected — `exact_match` must not drag sentence-transformers
(and torch) into a classifier run.
"""
from __future__ import annotations

import re

_ENCODER = None  # lazy singleton


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def exact_match(response: str, reference: str, **_) -> float:
    """1.0 iff the whitespace-normalized strings match. An EMPTY reference
    (e.g. router abstention) is a legitimate gold answer: the candidate must
    be empty too."""
    return 1.0 if _norm(response) == _norm(reference) else 0.0


def exact_match_set(response: str, reference: str, **_) -> float:
    """Order-insensitive comma-separated set equality ("a,b" == "b, a").
    For multi-label answers where the canonical order is not guaranteed."""
    split = lambda t: frozenset(p.strip() for p in _norm(t).split(",") if p.strip())
    return 1.0 if split(response) == split(reference) else 0.0


def _encoder(model_name: str):
    global _ENCODER
    if _ENCODER is None:
        from sentence_transformers import SentenceTransformer  # heavy: lazy
        _ENCODER = SentenceTransformer(model_name)
    return _ENCODER


def semantic(response: str, reference: str, *, encoder_model: str,
             rescale: float = 1.0, **_) -> float:
    """Paper reward core: cosine similarity between L2-normalized embeddings of
    the generated and reference texts, scaled by `rescale` (the paper uses c=4
    after subtracting a reference baseline; plain cosine is the sane default)."""
    if not response.strip():
        return 0.0
    embs = _encoder(encoder_model).encode([response, reference],
                                          normalize_embeddings=True)
    return rescale * max(float(embs[0] @ embs[1]), 0.0)


def paper_blend(response: str, reference: str, *, encoder_model: str,
                format_regexes: list[str] | None = None,
                reasoning_regex: str | None = None, **_) -> float:
    """arXiv:2509.13081 composite reward — unweighted sum of:
      semantic cosine + exact answer match + format compliance + non-empty
      reasoning block. Format/reasoning regexes come from config (the paper's
      XML tags are task-specific; hard-coding them here would be wrong for any
      other task). Missing regexes simply contribute 0."""
    r = semantic(response, reference, encoder_model=encoder_model)
    r += exact_match(response, reference)
    for rx in (format_regexes or []):
        if re.search(rx, response, flags=re.DOTALL):
            r += 1.0 / max(len(format_regexes), 1)
    if reasoning_regex:
        m = re.search(reasoning_regex, response, flags=re.DOTALL)
        if m and m.group(m.lastindex or 0).strip():
            r += 1.0
    return r


REWARDS = {
    "exact_match": exact_match,
    "exact_match_set": exact_match_set,
    "semantic": semantic,
    "paper_blend": paper_blend,
}


def get_reward(name: str):
    if name not in REWARDS:
        raise KeyError(f"unknown reward '{name}' — available: {sorted(REWARDS)}")
    return REWARDS[name]

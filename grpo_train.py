#!/usr/bin/env python3
"""GRPO training on MLX — the objective of arXiv:2509.13081 (Pappone,
Lazzaroni, Califano, Gentile, Marras): group size K, clipped surrogate (ε),
optional KL penalty vs the frozen SFT reference (β), group-normalized
advantages, pluggable reward (rewards.py). Loss on response tokens only;
generated token ids come straight from the sampler. Run on a FUSED model
(see README). Errors fail loud.

Usage:  python grpo_train.py --config configs/config_grpo.yaml
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import yaml
from mlx.utils import tree_flatten
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.utils import linear_to_lora_layers

from grpo_math import EarlyStopper, group_advantages, response_target_span
from rewards import get_reward

DEFAULTS = {
    "model_path": "models/fused-sft",
    "data_path": "data/sft/train.jsonl",
    "adapter_dir": "adapters/stage3-grpo",
    "results_path": "grpo_results.json",
    "log_path": "logs/grpo_train.log",
    "reward": "exact_match",          # rewards.py registry
    "reward_options": {},              # e.g. {encoder_model: ..., format_regexes: [...]}
    "n_candidates": 6,                 # paper: K=6
    "clip_eps": 0.2,                   # paper: ε=0.2
    "kl_coeff": 0.05,                  # paper: β=0.05; 0 disables (and skips the ref model)
    "temperature": 0.7,                # paper: fixed 0.7
    "max_tokens": 256,
    "min_response_length": 0,          # 0: short/empty gold answers are legitimate
    "n_iters": 200,
    "batch_size": 2,                   # prompts per iteration
    "learning_rate": 1e-6,
    "lora_layers": 16,
    "lora_rank": 32,                   # paper: r=32
    "lora_scale": 2.0,                 # paper: α=64 → scale = α/r = 2.0
    "lora_dropout": 0.0,
    "max_seq_length": 2048,            # prompt+response guard
    "early_stop_patience": 8,
    "early_stop_warmup": 10,
    "seed": 42,
}


def load_config(path: str) -> dict:
    cfg = dict(DEFAULTS)
    user = yaml.safe_load(Path(path).read_text()) or {}
    unknown = set(user) - set(DEFAULTS)
    if unknown:
        raise KeyError(f"unknown config keys {sorted(unknown)} in {path} — "
                       f"valid keys: {sorted(DEFAULTS)}")
    cfg.update(user)
    # YAML 1.1 traps: `1e-6` (no decimal point) parses as a STRING and would
    # crash the optimizer. Coerce every scalar to the type of its default.
    for k, default in DEFAULTS.items():
        if isinstance(default, bool) or isinstance(cfg[k], dict):
            continue
        if isinstance(default, float):
            cfg[k] = float(cfg[k])
        elif isinstance(default, int):
            cfg[k] = int(cfg[k])
    if cfg["lora_dropout"] != 0.0:
        # Dropout would apply DIFFERENT random masks when capturing old_logp and
        # when recomputing logp in the loss → spurious non-unit ratios → biased
        # clipped-surrogate gradients. The paper trains with dropout 0.
        raise ValueError("lora_dropout must be 0 for GRPO: dropout desynchronizes "
                         "old/new log-probs and corrupts the importance ratios")
    if cfg["reward"] in ("semantic", "paper_blend"):
        try:
            import sentence_transformers  # noqa: F401 — fail FAST, not at iter 1
        except ImportError as e:
            raise RuntimeError(
                f"reward '{cfg['reward']}' needs the encoder: "
                f"pip install sentence-transformers") from e
    return cfg


def make_logger(log_path: str):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")
    return log


def load_dataset(path: str, min_response_length: int) -> list[dict]:
    """{"messages":[system,user,assistant]} rows → prompt/reference pairs.
    Rows without the 3-message shape are counted and reported, never silently
    eaten. An EMPTY assistant answer is kept: abstention is a valid gold."""
    data, skipped = [], 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        msgs = json.loads(line).get("messages", [])
        if len(msgs) != 3:
            skipped += 1
            continue
        if len(msgs[2]["content"]) < min_response_length:
            skipped += 1
            continue
        data.append({"system": msgs[0]["content"], "prompt": msgs[1]["content"],
                     "reference": msgs[2]["content"]})
    if not data:
        raise ValueError(f"no usable rows in {path} "
                         f"(skipped {skipped}; min_response_length="
                         f"{min_response_length} too strict for this dataset?)")
    return data


def prompt_ids(tokenizer, system: str, user: str) -> list[int]:
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True)


def generate_candidate(model, tokenizer, ids: list[int], cfg: dict) -> list[int]:
    """One sampled candidate as TOKEN IDS (exactly what the sampler produced —
    no re-tokenization). Appends EOS when generation stopped naturally, so that
    choosing to stop (e.g. abstention = empty answer) is a reinforceable action."""
    sampler = make_sampler(temp=cfg["temperature"])
    tokens: list[int] = []
    finish = None
    for resp in stream_generate(model, tokenizer, prompt=ids,
                                max_tokens=cfg["max_tokens"], sampler=sampler):
        tokens.append(resp.token)
        finish = resp.finish_reason
    eos_ids = set(getattr(tokenizer, "eos_token_ids", None) or [tokenizer.eos_token_id])
    if finish == "stop" and (not tokens or tokens[-1] not in eos_ids):
        tokens.append(tokenizer.eos_token_id)
    return tokens


def sequence_logprobs(model, full_ids: list[int], span: tuple[int, int],
                      temperature: float) -> mx.array:
    """Per-token log-probs of the RESPONSE positions (differentiable).
    Logits are divided by the sampling temperature: the candidates were drawn
    from the tempered policy, and the surrogate must score that same policy."""
    tokens = mx.array(full_ids)[None, :]
    logits = model(tokens[:, :-1])
    if temperature > 0 and temperature != 1.0:
        logits = logits / temperature
    targets = tokens[:, 1:]
    logp = -nn.losses.cross_entropy(logits, targets, reduction="none")  # [1, T-1]
    return logp[0, span[0]:span[1]]


def decode_response(tokenizer, tokens: list[int]) -> str:
    eos_ids = set(getattr(tokenizer, "eos_token_ids", None) or [tokenizer.eos_token_id])
    body = [t for t in tokens if t not in eos_ids]
    return tokenizer.decode(body) if body else ""


def save_adapters(model, out_dir: Path, cfg: dict, log) -> None:
    """safetensors + adapter_config.json, the shape `mlx_lm fuse` expects."""
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(out_dir / "adapters.safetensors"), weights)
    (out_dir / "adapter_config.json").write_text(json.dumps({
        "fine_tune_type": "lora",
        "num_layers": cfg["lora_layers"],
        "lora_parameters": {"rank": cfg["lora_rank"], "scale": cfg["lora_scale"],
                            "dropout": cfg["lora_dropout"]},
    }, indent=2))
    log(f"saved adapters → {out_dir} ({len(weights)} tensors)")


def run(cfg: dict) -> float:
    log = make_logger(cfg["log_path"])
    random.seed(cfg["seed"])
    mx.random.seed(cfg["seed"])
    reward_fn = get_reward(cfg["reward"])

    model, tokenizer = load(cfg["model_path"])
    model.freeze()
    linear_to_lora_layers(model, cfg["lora_layers"],
                          {"rank": cfg["lora_rank"], "scale": cfg["lora_scale"],
                           "dropout": cfg["lora_dropout"]})
    model.train()
    log(f"policy: {cfg['model_path']} + LoRA r={cfg['lora_rank']} "
        f"scale={cfg['lora_scale']} on {cfg['lora_layers']} layers")

    ref_model = None
    if cfg["kl_coeff"] > 0:
        ref_model, _ = load(cfg["model_path"])   # frozen SFT reference for KL
        ref_model.freeze()
        log(f"KL reference loaded (β={cfg['kl_coeff']}) — doubles model memory")

    data = load_dataset(cfg["data_path"], cfg["min_response_length"])
    log(f"data: {len(data)} prompts · reward={cfg['reward']} · "
        f"K={cfg['n_candidates']} · ε={cfg['clip_eps']} · T={cfg['temperature']}")

    optimizer = optim.Adam(learning_rate=cfg["learning_rate"])
    stopper = EarlyStopper(cfg["early_stop_patience"], warmup=cfg["early_stop_warmup"])
    adapter_dir = Path(cfg["adapter_dir"])
    results = []

    def loss_fn(m, groups):
        """Mean clipped-surrogate (+β·KL) over every candidate of every group.
        ratio = π_θ/π_old on response tokens; with one update per generation
        batch π_old = π_θ at t₀ (ratio starts at 1) — the machinery still guards
        the update size exactly as in the paper's objective."""
        total, n_terms = mx.array(0.0), 0
        for g in groups:
            for cand in g["candidates"]:
                if cand["span"][1] <= cand["span"][0]:
                    continue                      # nothing generated: no actions
                logp = sequence_logprobs(m, cand["full_ids"], cand["span"],
                                         cfg["temperature"])
                ratio = mx.exp(logp - cand["old_logp"])
                adv = cand["advantage"]
                surr = mx.minimum(ratio * adv,
                                  mx.clip(ratio, 1 - cfg["clip_eps"], 1 + cfg["clip_eps"]) * adv)
                term = -mx.mean(surr)
                if cand.get("ref_logp") is not None:
                    # k3 estimator: exp(ref−p) − (ref−p) − 1  ≥ 0
                    d = cand["ref_logp"] - logp
                    term = term + cfg["kl_coeff"] * mx.mean(mx.exp(d) - d - 1)
                total = total + term
                n_terms += 1
        return total / max(n_terms, 1)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    for it in range(1, cfg["n_iters"] + 1):
        t0 = time.time()
        batch = random.sample(data, min(cfg["batch_size"], len(data)))
        groups, iter_rewards = [], []

        for item in batch:
            ids = prompt_ids(tokenizer, item["system"], item["prompt"])
            if len(ids) + cfg["max_tokens"] > cfg["max_seq_length"]:
                log(f"skip prompt: {len(ids)} tokens + max_tokens exceeds "
                    f"max_seq_length={cfg['max_seq_length']}")
                continue
            cands, rewards = [], []
            for _ in range(cfg["n_candidates"]):
                gen = generate_candidate(model, tokenizer, ids, cfg)
                full = list(ids) + gen
                span = response_target_span(len(ids), len(full))
                text = decode_response(tokenizer, gen)
                rewards.append(reward_fn(text, item["reference"], **cfg["reward_options"]))
                old = mx.stop_gradient(
                    sequence_logprobs(model, full, span, cfg["temperature"]))
                ref = (mx.stop_gradient(
                    sequence_logprobs(ref_model, full, span, cfg["temperature"]))
                       if ref_model is not None else None)
                cands.append({"full_ids": full, "span": span,
                              "old_logp": old, "ref_logp": ref})
            for c, adv in zip(cands, group_advantages(rewards)):
                c["advantage"] = adv
            iter_rewards.extend(rewards)
            groups.append({"candidates": cands})

        if not groups:
            log(f"iter {it}: batch empty after guards — check max_seq_length")
            continue

        loss, grads = loss_and_grad(model, groups)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        avg_r = float(np.mean(iter_rewards))
        results.append({"iteration": it, "avg_reward": avg_r,
                        "max_reward": float(np.max(iter_rewards)),
                        "loss": float(loss.item()), "time": time.time() - t0})
        log(f"iter {it}/{cfg['n_iters']}: reward={avg_r:.3f} "
            f"(max={results[-1]['max_reward']:.3f}) loss={results[-1]['loss']:.4f} "
            f"{results[-1]['time']:.1f}s")

        if stopper.update(avg_r):
            save_adapters(model, adapter_dir / "best", cfg, log)
        if stopper.should_stop:
            log(f"early stop at iter {it}: no improvement for "
                f"{cfg['early_stop_patience']} iterations")
            break

    save_adapters(model, adapter_dir / "final", cfg, log)
    Path(cfg["results_path"]).write_text(
        json.dumps({"results": results, "best_reward": stopper.best,
                    "config": cfg}, indent=2, default=str))
    log(f"GRPO complete — best avg reward {stopper.best:.3f}")
    return stopper.best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_grpo.yaml")
    args = ap.parse_args()
    run(load_config(args.config))

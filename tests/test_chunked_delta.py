"""The chunked gated delta rule against mlx_lm's token-by-token loop: outputs, final state and gradients."""
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

pytest.importorskip("mlx_lm.models.qwen3_5")
from mlx_lm.models import qwen3_5
from mlx_lm.models.gated_delta import compute_g, gated_delta_ops

sys.path.insert(0, str(Path(__file__).parent.parent))
from chunked_delta import chunked_gated_delta, patch

NAMES = ("q", "k", "v", "a", "b", "A_log", "dt_bias")


def inputs(T, Hk=2, Hv=4, Dk=32, Dv=32, B=1, seed=0):
    """Random inputs shaped and normalised as Qwen3.5's linear attention feeds them: keys of unit norm."""
    mx.random.seed(seed)
    scale = Dk ** -0.5
    q = scale ** 2 * mx.fast.rms_norm(mx.random.normal((B, T, Hk, Dk)), None, 1e-6)
    k = scale * mx.fast.rms_norm(mx.random.normal((B, T, Hk, Dk)), None, 1e-6)
    return dict(q=q, k=k, v=mx.random.normal((B, T, Hv, Dv)), a=mx.random.normal((B, T, Hv)),
                b=mx.random.normal((B, T, Hv)), A_log=mx.random.uniform(-2, 1, (Hv,)), dt_bias=mx.random.normal((Hv,)))


def loop(q, k, v, a, b, A_log, dt_bias, state=None, mask=None):
    return gated_delta_ops(q, k, v, compute_g(A_log, a, dt_bias), mx.sigmoid(b), state, mask)


def close(x, y, tolerance=1e-4):
    return mx.allclose(x, y, rtol=tolerance, atol=tolerance).item()


@pytest.mark.parametrize("T", [1, 5, 64, 150])
@pytest.mark.parametrize("Hk,Hv", [(2, 4), (4, 4)])
def test_outputs_and_state_match_the_loop(T, Hk, Hv):
    x = inputs(T, Hk, Hv)
    y, state = chunked_gated_delta(**x)
    y_loop, state_loop = loop(**x)
    assert y.shape == y_loop.shape and state.shape == state_loop.shape
    assert close(y, y_loop) and close(state, state_loop)


def test_an_initial_state_carries_on():
    x = inputs(100)
    start = mx.random.normal((1, 4, 32, 32)) * 0.1
    y, state = chunked_gated_delta(**x, state=start)
    y_loop, state_loop = loop(**x, state=start)
    assert close(y, y_loop) and close(state, state_loop)


def test_gradients_match_the_loop():
    x = inputs(150)
    start = mx.random.normal((1, 4, 32, 32)) * 0.1
    y_weights, state_weights = mx.random.normal((1, 150, 4, 32)), mx.random.normal((1, 4, 32, 32))

    def loss(fn):
        def scalar(*arguments):
            y, state = fn(*arguments)
            return (y * y_weights).sum() + (state * state_weights).sum()
        return scalar

    arguments = [x[name] for name in NAMES] + [start]
    positions = list(range(len(arguments)))
    grads = mx.grad(loss(chunked_gated_delta), argnums=positions)(*arguments)
    grads_loop = mx.grad(loss(loop), argnums=positions)(*arguments)
    for name, grad, grad_loop in zip(NAMES + ("state",), grads, grads_loop):
        assert close(grad, grad_loop, 1e-3), name


def test_a_masked_token_leaves_the_state():
    x = inputs(150)
    mask = mx.random.uniform(shape=(1, 150)) > 0.3
    y, state = chunked_gated_delta(**x, mask=mask)
    y_loop, state_loop = loop(**x, mask=mask)
    held = mask[..., None, None]
    assert close(state, state_loop)
    assert close(mx.where(held, y, 0), mx.where(held, y_loop, 0))


def test_repeated_keys_stay_exact():
    # One key over and over, steps near 1 and no decay: the chunk's triangular system is as ill-conditioned as a
    # delta rule makes it, and the inverse must not pass through growing powers.
    x = inputs(128)
    x["k"] = mx.broadcast_to(x["k"][:, :1], x["k"].shape)
    x["b"] = mx.full(x["b"].shape, 8.0)
    x["a"] = mx.full(x["a"].shape, -20.0)
    y, state = chunked_gated_delta(**x)
    y_loop, state_loop = loop(**x)
    assert close(y, y_loop) and close(state, state_loop)


def tiny_qwen35():
    mx.random.seed(1)
    model = qwen3_5.Model(qwen3_5.ModelArgs.from_dict({"model_type": "qwen3_5", "text_config": {
        "model_type": "qwen3_5_text", "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 4,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64, "vocab_size": 97,
        "linear_num_value_heads": 4, "linear_num_key_heads": 2, "linear_key_head_dim": 32,
        "linear_value_head_dim": 32, "full_attention_interval": 4, "tie_word_embeddings": True}}))
    mx.eval(model.parameters())
    return model


def test_patch_trains_qwen35_like_the_loop(monkeypatch):
    model = tiny_qwen35()
    tokens = mx.random.randint(0, 97, (1, 151))

    def loss(model, tokens):
        return nn.losses.cross_entropy(model(tokens[:, :-1]), tokens[:, 1:]).mean()

    model.eval()
    logits_kernel = model(tokens)
    model.train()
    loss_loop, grads_loop = nn.value_and_grad(model, loss)(model, tokens)
    monkeypatch.setattr(qwen3_5, "gated_delta_update", qwen3_5.gated_delta_update)
    patch()
    loss_chunked, grads_chunked = nn.value_and_grad(model, loss)(model, tokens)
    assert close(loss_chunked, loss_loop)
    for (name, grad), (_, grad_loop) in zip(tree_flatten(grads_chunked), tree_flatten(grads_loop)):
        assert close(grad, grad_loop, 1e-3), name
    model.eval()
    assert close(model(tokens), logits_kernel)

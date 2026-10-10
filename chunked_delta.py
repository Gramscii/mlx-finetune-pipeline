"""Chunked gated delta rule for training Qwen3.5-family models with MLX.

mlx_lm runs the gated delta rule of Qwen3.5's linear-attention layers through a Metal kernel that has no backward
pass, so in training mode it falls back to a loop over every token (`gated_delta_ops`): memory and time grow with the
sequence until a single 4,096-token step exceeds a 64 GB Mac. This module computes the same recurrence chunk by chunk,
as the GPU implementations of Gated DeltaNet do (Yang et al., "Gated Delta Networks"): inside a chunk the recurrence
becomes matrix products, and one state per chunk is carried, so autograd differentiates a few dozen small matrix
products instead of thousands of sequential steps.

The recurrence, per head, with the state S of shape [Dv, Dk]:
    S_t = g_t * S_{t-1} + beta_t * (v_t - g_t * S_{t-1} k_t) k_t^T,    y_t = S_t q_t

`patch()` makes mlx_lm's Qwen3.5 models use it whenever they run without the kernel, which is in training mode only.
Run as a script, it is `python -m mlx_lm lora` with the patch: `python chunked_delta.py -c lora.yaml`.
"""

import mlx.core as mx
import mlx.nn as nn

CHUNK = 64


def _unit_lower_inverse(lower: mx.array) -> mx.array:
    """(I + L)^{-1} for a strictly lower triangular L whose side is a power of two, by block doubling: the inverse of
    [[A, 0], [C, D]] is [[A^-1, 0], [-D^-1 C A^-1, D^-1]], so each level joins pairs of solved blocks with two matrix
    products and every intermediate is a block of a true inverse. (A power series of L would pass through powers that
    grow like binomial coefficients when keys repeat, and cancel them in float32.)"""
    size = lower.shape[-1]
    rows = mx.arange(size)
    inverse = mx.eye(size, dtype=lower.dtype)
    span = 1
    while span < size:
        block = rows // span
        joins = ((block[:, None] % 2 == 1) & (block[None, :] == block[:, None] - 1)).astype(lower.dtype)
        inverse = inverse - (inverse @ (lower * joins)) @ inverse
        span *= 2
    return inverse


def chunked_gated_delta(q: mx.array, k: mx.array, v: mx.array, a: mx.array, b: mx.array, A_log: mx.array,
                        dt_bias: mx.array, state: mx.array | None = None, mask: mx.array | None = None,
                        chunk: int = CHUNK) -> tuple[mx.array, mx.array]:
    """The gated delta rule over a whole sequence, chunk by chunk.

    Shapes as mlx_lm's `gated_delta_update`: q, k [B, T, Hk, Dk]; v [B, T, Hv, Dv]; a, b [B, T, Hv]; A_log and dt_bias
    [Hv]; state [B, Hv, Dv, Dk] or None; mask [B, T] or None (a masked token leaves the state as it was). Returns y
    [B, T, Hv, Dv] in q's dtype and the final state [B, Hv, Dv, Dk] in float32. `chunk` is a power of two."""
    dtype = q.dtype
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    beta = mx.sigmoid(b.astype(mx.float32))
    # The decay's log: g itself underflows for strong decays, and the chunk works with differences of its cumsum.
    log_g = -mx.exp(A_log.astype(mx.float32)) * nn.softplus((a + dt_bias).astype(mx.float32))
    if mask is not None:
        held = mask.astype(mx.float32)
        beta, log_g = beta * held[..., None], log_g * held[..., None]
    if (repeat := Hv // Hk) > 1:
        q, k = mx.repeat(q, repeat, -2), mx.repeat(k, repeat, -2)
    # [B, H, T, D] and [B, H, T], in float32.
    q, k, v = (x.astype(mx.float32).transpose(0, 2, 1, 3) for x in (q, k, v))
    beta, log_g = beta.transpose(0, 2, 1), log_g.transpose(0, 2, 1)
    padded = -T % chunk
    if padded:
        # A padding token holds no key, value or step and does not decay: the state passes it unchanged.
        q, k, v = (mx.pad(x, [(0, 0), (0, 0), (0, padded), (0, 0)]) for x in (q, k, v))
        beta, log_g = (mx.pad(x, [(0, 0), (0, 0), (0, padded)]) for x in (beta, log_g))
    N = (T + padded) // chunk
    q, k, v = (x.reshape(B, Hv, N, chunk, -1) for x in (q, k, v))
    beta, log_g = beta.reshape(B, Hv, N, chunk), log_g.reshape(B, Hv, N, chunk)

    G = mx.cumsum(log_g, axis=-1)                                          # [B, H, N, C], non-increasing
    causal = mx.tril(mx.ones((chunk, chunk), dtype=mx.float32))            # j <= i
    # exp(G_i - G_j) for j <= i and 0 above; the exponent is zeroed above first, where it is positive and overflows.
    decay = mx.exp((G[..., :, None] - G[..., None, :]) * causal) * causal
    k_beta = k * beta[..., None]
    inverse = _unit_lower_inverse((k_beta @ k.swapaxes(-1, -2)) * decay * (causal - mx.eye(chunk)))
    u = inverse @ (v * beta[..., None])                                    # [B, H, N, C, Dv]
    w = inverse @ (k_beta * mx.exp(G)[..., None])                          # [B, H, N, C, Dk]

    S = (mx.zeros((B, Hv, Dk, Dv), dtype=mx.float32) if state is None
         else state.astype(mx.float32).swapaxes(-1, -2))                    # [B, H, Dk, Dv]
    outputs = []
    for n in range(N):
        qn, kn, Gn = q[:, :, n], k[:, :, n], G[:, :, n]
        fresh = u[:, :, n] - w[:, :, n] @ S
        outputs.append((qn * mx.exp(Gn)[..., None]) @ S + ((qn @ kn.swapaxes(-1, -2)) * decay[:, :, n]) @ fresh)
        last = Gn[..., -1:]
        S = S * mx.exp(last)[..., None] + (kn * mx.exp(last - Gn)[..., None]).swapaxes(-1, -2) @ fresh
    y = mx.concatenate(outputs, axis=2)[:, :, :T].transpose(0, 2, 1, 3)
    return y.astype(dtype), S.swapaxes(-1, -2)


def patch() -> None:
    """mlx_lm's Qwen3.5 linear attention computes with the chunks whenever it runs without its kernel."""
    from mlx_lm.models import qwen3_5

    kernel_path = qwen3_5.gated_delta_update

    def gated_delta_update(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True):
        if use_kernel:
            return kernel_path(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel=True)
        return chunked_gated_delta(q, k, v, a, b, A_log, dt_bias, state, mask)

    qwen3_5.gated_delta_update = gated_delta_update


if __name__ == "__main__":
    from mlx_lm import lora

    patch()
    lora.main()

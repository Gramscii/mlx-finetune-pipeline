# mlx-finetune-pipeline

**CPT → SFT → GRPO fine-tuning for language models on Apple Silicon. Local, no cloud, no NVIDIA.**

A [gramscii](https://github.com/Gramscii) project. Implements the training recipe of
**[Shaping Explanations: Semantic Reward Modeling with Encoder-Only Transformers for GRPO](https://arxiv.org/abs/2509.13081)**
— Francesco Pappone, Ruggero Marino Lazzaroni, Federico Califano, Niccolò Gentile, Roberto Marras.

## Stages

| Stage | Purpose | Data | Skip when |
|---|---|---|---|
| **CPT** | domain familiarization on raw text | `{"text": "..."}` | the base model already knows the domain |
| **SFT** | supervised instruction tuning | `{"messages":[system,user,assistant]}` | — |
| **GRPO** | K candidates per prompt, rewarded and reinforced group-relatively | same as SFT | answer-quality shaping is not needed |

## Quickstart

```bash
pip install -r requirements.txt
pip install sentence-transformers   # only for the semantic rewards (default config uses one)

./run_pipeline.sh                   # all three stages
./run_pipeline.sh --skip-cpt        # SFT + GRPO
./run_pipeline.sh --only grpo       # a single stage
```

CPT/SFT log to `logs/cpt.log` / `logs/sft.log`; GRPO to `logs/grpo.log`. Between stages the
pipeline selects the checkpoint at the iteration with the lowest validation loss
(`best_checkpoint.py`) and persists the choice, so split invocations resume correctly.

## GRPO

`grpo_train.py` implements the paper's objective:

- **K = 6** candidates per prompt, group-normalized advantages `(r − mean)/std`
- clipped surrogate, **ε = 0.2**
- optional KL penalty vs the frozen SFT reference, **β = 0.05** (k3 estimator;
  `kl_coeff: 0` skips loading the reference model and halves memory)
- fixed sampling temperature **0.7**; log-probs are computed on temperature-scaled logits,
  so the surrogate scores the same policy that generated the candidates
- loss on **response tokens only**; generated token ids are taken directly from the sampler
- EOS is part of the action sequence: an empty answer (abstention) is reinforceable
- LoRA **r = 32, α = 64** (`lora_scale = α/r`)

Configuration lives in `configs/config_grpo.yaml` — unknown keys are rejected, scalars are
type-coerced (YAML parses bare `1e-6` as a string), and invalid setups (a semantic reward
without its encoder installed, `lora_dropout > 0`) fail at startup, not mid-run.

> **GRPO runs on a fused model** (`models/fused-sft`): `mlx_lm` cannot stack a trainable LoRA
> on top of a loaded adapter, so the pipeline fuses after SFT.

## Rewards

Selected by `reward:` in the GRPO config (`rewards.py`):

| name | scores | encoder | for |
|---|---|---|---|
| `exact_match` | whitespace-normalized string equality — empty gold (abstention) is a valid answer | no | classifiers, label tasks |
| `exact_match_set` | order-insensitive comma-set equality | no | multi-label answers |
| `semantic` | cosine of L2-normalized encoder embeddings | yes | prose with reference answers |
| `paper_blend` | semantic + exact answer + format regexes + non-empty reasoning | yes | explain-then-answer tasks |

`examples/` ships ready configs for a multilingual query router: long route-set prompts
(`max_seq_length: 1280`) and a deterministic `exact_match_set` reward, so a correct
`"route_a, route_b"` is never punished for a space.

## Notes

- Keep `steps_per_eval` equal to `save_every` in the CPT/SFT configs: every evaluated
  iteration then has its own numbered checkpoint, and the val-loss selection always resolves.
- `mlx_lm` measures the iter-N validation loss just before the N-th update and saves the
  N-numbered checkpoint just after it: the selected file is one optimizer step past the
  validated weights — a uniform skew that leaves the ranking meaningful.

## Qwen3.5 and other gated-delta models

`mlx_lm` runs the linear-attention layers of Qwen3.5 (Gated DeltaNet) through a Metal kernel
with no backward pass, so in training mode it falls back to a loop over every token: memory and
time grow with the sequence until a 4,096-token step no longer fits a 64 GB Mac.
`chunked_delta.py` computes the same recurrence in chunks of 64 tokens, as matrix products MLX
can differentiate, and patches `mlx_lm` to use it whenever the kernel is off (training only):

```bash
python chunked_delta.py -c configs/config_sft.yaml   # = python -m mlx_lm lora, with the patch
```

One LoRA step of Qwen3.5-4B (bf16, rank 16, every layer, gradient checkpointing) on an
M4 Pro with 64 GB:

| tokens | token loop | chunked |
|---|---|---|
| 2,048 | 42.8 GiB, 13 tok/s | 11.4 GiB, 160 tok/s |
| 4,096 | out of memory | 14.8 GiB, 110 tok/s |
| 8,192 | out of memory | 25.5 GiB, 76 tok/s |

With LoRA on the top 16 layers only, 4,096 tokens train at about 220 tok/s. Past 8K the full
attention layers dominate (16,384 tokens: 60 GiB).

## Tests

```bash
pytest tests/            # rewards, advantages, masking spans, checkpoint selection, chunked delta rule
bash -n run_pipeline.sh
```

`tests/test_chunked_delta.py` checks the chunked rule against `mlx_lm`'s token loop (outputs,
final state, gradients, masks, an ill-conditioned chunk) and a small Qwen3.5 trained both ways;
it needs `mlx-lm`.

The differentiable loss needs MLX at runtime; everything around it is pure Python and covered
by the suite.

## Requirements

Apple Silicon (MLX shares CPU/GPU memory: a 64 GB Mac fine-tunes what would need an 80 GB
NVIDIA card), Python ≥ 3.10, models in MLX format from
[mlx-community](https://huggingface.co/mlx-community).

## Citation

```bibtex
@misc{pappone2025shaping,
  title  = {Shaping Explanations: Semantic Reward Modeling with
            Encoder-Only Transformers for GRPO},
  author = {Pappone, Francesco and Lazzaroni, Ruggero Marino and
            Califano, Federico and Gentile, Niccol{\`o} and Marras, Roberto},
  year   = {2025},
  eprint = {2509.13081},
  archivePrefix = {arXiv}
}
```

## License

MIT — see [LICENSE](LICENSE).

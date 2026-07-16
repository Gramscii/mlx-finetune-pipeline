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

## Tests

```bash
pytest tests/            # rewards, advantages, masking spans, checkpoint selection
bash -n run_pipeline.sh
```

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

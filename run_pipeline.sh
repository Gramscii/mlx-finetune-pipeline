#!/bin/bash
# CPT → SFT → GRPO pipeline for MLX on Apple Silicon.
#
# Usage:
#   ./run_pipeline.sh                      # all three stages
#   ./run_pipeline.sh --skip-cpt           # SFT + GRPO (base model already knows the domain)
#   ./run_pipeline.sh --only sft           # a single stage: cpt | sft | grpo
#
# Every stage logs to logs/<stage>.log; the SFT→fuse step picks the checkpoint
# with the LOWEST validation loss (best_checkpoint.py), not merely the newest.
set -euo pipefail
cd "$(dirname "$0")"

RUN_CPT=1; RUN_SFT=1; RUN_GRPO=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-cpt)  RUN_CPT=0 ;;
    --skip-sft)  RUN_SFT=0 ;;
    --skip-grpo) RUN_GRPO=0 ;;
    --only)
      RUN_CPT=0; RUN_SFT=0; RUN_GRPO=0
      case "${2:-}" in
        cpt) RUN_CPT=1 ;; sft) RUN_SFT=1 ;; grpo) RUN_GRPO=1 ;;
        *) echo "ERROR: --only expects cpt|sft|grpo" >&2; exit 1 ;;
      esac
      shift ;;
    *) echo "ERROR: unknown flag '$1' (use --skip-cpt/--skip-sft/--skip-grpo/--only STAGE)" >&2; exit 1 ;;
  esac
  shift
done

# Environment: prefer ./venv if present, otherwise the current interpreter.
if [[ -f venv/bin/activate ]]; then source venv/bin/activate; fi
python -c "import mlx_lm" 2>/dev/null || {
  echo "ERROR: mlx_lm not importable — run: pip install -r requirements.txt" >&2; exit 1; }

[[ -d models/your-base-model ]] || {
  echo "ERROR: base model missing at models/your-base-model" >&2
  echo "  e.g.: python -c \"from huggingface_hub import snapshot_download; snapshot_download('mlx-community/<model>-4bit', local_dir='models/your-base-model')\"" >&2
  exit 1; }

mkdir -p logs
RESUME_ARGS=()

if [[ $RUN_CPT -eq 1 ]]; then
  [[ -f data/cpt/train.jsonl ]] || { echo "ERROR: data/cpt/train.jsonl missing (or use --skip-cpt)" >&2; exit 1; }
  echo "=== Stage 1: CPT ==="
  python -m mlx_lm lora -c configs/config_cpt.yaml 2>&1 | tee logs/cpt.log
  BEST_CPT="$(python best_checkpoint.py logs/cpt.log adapters/stage1-cpt)"
  echo "CPT selected checkpoint: $BEST_CPT"
  # Persist the choice so a later `--only sft` invocation resumes from it.
  echo "$BEST_CPT" > adapters/stage1-cpt/BEST_CHECKPOINT
  RESUME_ARGS=(--resume-adapter-file "$BEST_CPT")
fi

if [[ $RUN_SFT -eq 1 ]]; then
  [[ -f data/sft/train.jsonl ]] || { echo "ERROR: data/sft/train.jsonl missing" >&2; exit 1; }
  if [[ ${#RESUME_ARGS[@]} -eq 0 && -f adapters/stage1-cpt/BEST_CHECKPOINT ]]; then
    BEST_CPT="$(cat adapters/stage1-cpt/BEST_CHECKPOINT)"
    if [[ -f "$BEST_CPT" ]]; then
      echo "Resuming SFT from the CPT checkpoint of a previous run: $BEST_CPT"
      RESUME_ARGS=(--resume-adapter-file "$BEST_CPT")
    else
      echo "WARNING: CPT marker points to a missing file ($BEST_CPT) — SFT starts from the base model" >&2
    fi
  fi
  echo "=== Stage 2: SFT ==="
  # ${arr[@]+...} guard: on macOS's bash 3.2, expanding an EMPTY array under
  # `set -u` is an "unbound variable" error.
  python -m mlx_lm lora -c configs/config_sft.yaml ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} 2>&1 | tee logs/sft.log
  BEST_SFT="$(python best_checkpoint.py logs/sft.log adapters/stage2-sft)"
  echo "SFT selected checkpoint: $BEST_SFT"

  echo "=== Fusing SFT adapter into the base model ==="
  # mlx_lm fuse expects a DIRECTORY with adapters.safetensors + adapter_config.json:
  # stage the chosen checkpoint under the canonical name before fusing.
  FUSE_DIR=adapters/stage2-sft-best
  mkdir -p "$FUSE_DIR"
  cp "$BEST_SFT" "$FUSE_DIR/adapters.safetensors"
  cp adapters/stage2-sft/adapter_config.json "$FUSE_DIR/adapter_config.json"
  python -m mlx_lm fuse \
    --model models/your-base-model \
    --adapter-path "$FUSE_DIR" \
    --save-path models/fused-sft
  echo "Fused model → models/fused-sft"
fi

if [[ $RUN_GRPO -eq 1 ]]; then
  [[ -d models/fused-sft ]] || { echo "ERROR: models/fused-sft missing — run the SFT stage first" >&2; exit 1; }
  echo "=== Stage 3: GRPO ==="
  python grpo_train.py --config configs/config_grpo.yaml 2>&1 | tee logs/grpo.log
  echo "=== Fusing the GRPO adapter (final model) ==="
  GRPO_BEST=adapters/stage3-grpo/best
  [[ -f "$GRPO_BEST/adapters.safetensors" ]] || GRPO_BEST=adapters/stage3-grpo/final
  python -m mlx_lm fuse \
    --model models/fused-sft \
    --adapter-path "$GRPO_BEST" \
    --save-path models/final
  echo "Final model → models/final"
fi

echo "=== Pipeline complete ==="

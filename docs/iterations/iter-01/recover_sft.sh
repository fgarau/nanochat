#!/bin/bash

# Resume runs/runcpu.sh after a failure, running only the stages that are missing.
# Assumes the tokenizer and base model (d6) are already trained.
#
# Run as:
# bash runs/recover_sft.sh
# FORCE_SFT=1 bash runs/recover_sft.sh   # redo SFT even if a checkpoint exists

set -e

export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
source .venv/bin/activate
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

BASE_CKPT="$NANOCHAT_BASE_DIR/base_checkpoints/d6/model_005000.pt"
BASE_EVAL_CSV="$NANOCHAT_BASE_DIR/base_eval/base_model_005000.csv"
SFT_DIR="$NANOCHAT_BASE_DIR/chatsft_checkpoints/d6"

# the base model must exist, otherwise there is nothing to recover from
if [ ! -f "$BASE_CKPT" ]; then
    echo "No base checkpoint at $BASE_CKPT, run runs/runcpu.sh from the start instead."
    exit 1
fi
echo "Found base checkpoint: $BASE_CKPT"

# base model CORE eval
if [ -f "$BASE_EVAL_CSV" ]; then
    echo "Skipping base_eval, results already in $BASE_EVAL_CSV"
else
    python -m scripts.base_eval --device-batch-size=1 --split-tokens=16384 --max-per-task=12
fi

# SFT (chat_sft only saves a checkpoint at the last step, so a crashed run leaves nothing to resume)
if ls "$SFT_DIR"/model_*.pt &> /dev/null && [ -z "$FORCE_SFT" ]; then
    echo "Skipping chat_sft, checkpoint already in $SFT_DIR (set FORCE_SFT=1 to redo)"
else
    python -m scripts.chat_sft \
        --eval-every=200 \
        --eval-tokens=524288 \
        --num-iterations=1500 \
        --chatcore-max-cat=200 \
        --run=$WANDB_RUN
fi

# quick smoke test of the chat model
python -m scripts.chat_cli -p "What is the capital of France?"

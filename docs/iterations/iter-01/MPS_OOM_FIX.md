# MPS out-of-memory fix for `runs/runcpu.sh`

## What failed

Two eval stages in the CPU/MPS run (16 GiB MacBook) crashed with `RuntimeError: MPS backend out of memory`:

1. **`scripts.base_eval` (CORE)**: crashed at `bigbench_language_identification`. The script had no `set -e`, so the run went on to SFT.
2. **`scripts.chat_sft`**: trained to step 200/1500 (val bpb 0.9605). Then, at the first ChatCORE eval, ARC-Easy (26.52%) and ARC-Challenge (26.37%) finished and **MMLU** crashed. No SFT checkpoint was saved, because `chat_sft` saves only at the last step.

## Root cause

A diagnostic run logged memory after each CORE task:

- `torch.mps.current_allocated_memory()` stayed flat at **0.28 GiB**.
- `torch.mps.driver_allocated_memory()` climbed to **18.7 GiB**.

The evals pad each batch to its own longest sequence, so almost every forward pass has a new `(B, T)` shape, and the MPS backend keeps cached buffers for each one. On top of that, each forward pass materialises the full `(B, T, 32768)` fp32 logits tensor:

- CORE runs all of an example's answer choices in one pass.
- ChatCORE runs `device_batch_size` = 16 prompts in one pass, on the full MMLU set.

The MPS limit (20.13 GiB) is above the machine's physical 16 GiB of RAM, so the process ran out of real memory first. Training uses fixed shapes and was not affected.

## Changes

All code changes apply only when the device is `mps`; CUDA and CPU behave as before.

| File | Change |
|---|---|
| `nanochat/core_eval.py` | `stack_sequences` takes optional `pad_to_multiple` / `max_len` arguments. `evaluate_example` pads T up to a multiple of 64 and runs `forward_model` on chunks of 4 rows, then concatenates the losses and predictions. |
| `scripts/chat_eval.py` | `run_categorical_eval` pads to a multiple of 64 and runs the model in chunks of 4 rows. |
| `scripts/base_eval.py` | `torch.mps.empty_cache()` after each CORE task. |
| `scripts/chat_sft.py` | `torch.mps.empty_cache()` before the ChatCORE block (frees training's cache) and after each ChatCORE task. |
| `runs/runcpu.sh` | `set -e` so a failing stage stops the run. `--chatcore-max-cat=200` on `chat_sft` (MMLU capped at 200 problems; ARC too). |

The padding goes on the right, and only positions inside each sequence are read, so results are unchanged. Rows are independent, so chunking doesn't change results either.

`runs/runcpu.sh` also has earlier, unrelated local edits: base_train `--device-batch-size` 32 → 16, and base_eval `--max-per-task` 16 → 12.

## Verification

- **CORE** (`base_eval --eval=core --device-batch-size=1 --max-per-task=12`): every task score is identical to the old code. Peak driver memory went from 18.7 to 16.6 GiB. This run was before the per-task `empty_cache` was added.
  - Old scores: `~/.cache/nanochat/base_eval/base_model_005000.before.csv`
  - New scores: `base_model_005000.csv` in the same folder
- **ChatCORE** (`chat_sft --num-iterations=1 --chatcore-every=1 --chatcore-max-cat=200 --eval-tokens=524288`): all five tasks finished. ARC-E 28.0%, ARC-C 23.5%, MMLU 28.0%, GSM8K 0%, HumanEval 0%. The throwaway checkpoint was deleted.
- `uv run --extra cpu --group dev python -m pytest tests -q`: 43 passed, 15 skipped.
- **Not yet verified:** a full 1500-step SFT run. That test was a single step, so less training memory was held than at step 200.

## How to recover from the last failure

The base model is intact: `~/.cache/nanochat/base_checkpoints/d6/` holds step 5000 with its optimizer state. Only SFT needs to be redone.

**Don't** rerun `bash runs/runcpu.sh`, because it would retrain the tokenizer and the base model. Run only the remaining stages:

```bash
cd ~/projects/nano2
source .venv/bin/activate
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"

# 1. (optional) CORE eval of the base model. Already done with --max-per-task=12 during verification;
#    results are in ~/.cache/nanochat/base_eval/base_model_005000.csv
python -m scripts.base_eval --device-batch-size=1 --split-tokens=16384 --max-per-task=12

# 2. SFT from scratch (it starts from the base checkpoint; the crashed run left nothing to resume)
python -m scripts.chat_sft \
    --eval-every=200 \
    --eval-tokens=524288 \
    --num-iterations=1500 \
    --chatcore-max-cat=200 \
    --run=dummy

# 3. Chat with the result
python -m scripts.chat_cli -p "What is the capital of France?"
```

If it still runs out of memory, try these in order:

1. Evaluate less often or less: `--chatcore-every=500`, or `--chatcore-max-cat=100`.
2. Lower the SFT batch size: `--device-batch-size=8`. This changes gradient accumulation, not the total batch size.
3. Skip ChatCORE during training (`--chatcore-every=-1`) and evaluate the final checkpoint afterwards with `python -m scripts.chat_eval -i sft -a ARC-Easy` (and similarly for the other tasks).
4. Close other memory-heavy apps; on a 16 GiB Mac the browser alone can make the difference.

Don't set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`, because it removes the limit and can freeze the machine.

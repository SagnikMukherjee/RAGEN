# Running RAGEN on Modal

This repo has a Modal launcher in `modal_ragen.py` for the RAGEN/verl training scripts.

## One-time setup

Install and authenticate Modal locally:

```bash
pip install modal
modal setup
```

Create one secret for Hugging Face and W&B credentials:

```bash
modal secret create ragen-secrets HF_TOKEN=... WANDB_API_KEY=...
```

The launcher lazily creates a persistent Modal Volume named `ragen-artifacts`. It is mounted inside the job at:

```text
/shared/storage-01/users/sagnikm3/bfex
```

That path matches the current `train_lookahead_*.sh` scripts.

## Run the current lookahead experiments

Baseline:

```bash
cd /home/sagnikm3/RAGEN
modal run modal_ragen.py --script train_lookahead_baseline.sh --run-name lookahead-baseline-qwen25-7b
```

Treatment:

```bash
cd /home/sagnikm3/RAGEN
modal run modal_ragen.py --script train_lookahead_treatment.sh --run-name lookahead-treatment-qwen25-7b
```

By default this requests `H100:8`, because the current lookahead scripts hard-code 8 GPUs in the `train.py` command.

To allow the local CLI to detach while the remote job continues:

```bash
modal run -d modal_ragen.py --script train_lookahead_baseline.sh --run-name lookahead-baseline-qwen25-7b
```

The entrypoint spawns the training function asynchronously and prints a `fc-*` Function call ID. Use that ID to track just this run:

```bash
modal app logs ragen-experiments --function-call fc-... -f
```

If you explicitly want the local command to block until the training function returns, add `--wait`.

## Run a smaller smoke test

The main-table scripts expose GPU and step knobs, so they are better for a cheap Modal smoke test:

```bash
RAGEN_MODAL_GPU=H100:1 modal run modal_ragen.py \
  --script scripts/runs/run_main_table_diff_size.sh \
  --script-args="--steps 5 --models Qwen2.5-0.5B --tasks sokoban --gpus 0 --gpus-per-exp 1 --filters filter --save-freq -1" \
  --run-name smoke-sokoban-qwen25-05b
```

## Inspect or download artifacts

List the persisted artifacts:

```bash
modal volume ls ragen-artifacts /modal_runs
```

Download a run directory:

```bash
modal volume get ragen-artifacts modal_runs/smoke-sokoban-qwen25-05b ./smoke-sokoban-qwen25-05b
```

Lookahead checkpoints and W&B cache/log dirs are written directly under the volume's `sokoban/`, `wandb_*`, and `hf_cache/` directories because the bash scripts use the shared `BFEX` path.

## Notes

- Modal Functions have a maximum per-call timeout of 24 hours, so long training should checkpoint into the mounted Volume and be restarted from checkpoints if needed.
- The first run will spend time building the CUDA 12.8 image and installing vLLM, SGLang, FlashAttention, RAGEN, and vendored verl. Later runs reuse Modal's image cache unless `modal_ragen.py`, install scripts, or included source files change.
- Avoid running many jobs that write to the same checkpoint directory at once. Give each run a unique `--run-name` and experiment name.

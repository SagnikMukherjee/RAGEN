# SFT -> RL -> Eval Experiment

Run all commands from the repo root:

```bash
cd /home/sagnikm3/RAGEN
```

## 1. Train SFT

```bash
PYTHONPATH="$PWD/verl:$PWD:${PYTHONPATH:-}" \
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  -m verl.trainer.fsdp_sft_trainer \
  data.train_files=/home/sagnikm3/RAGEN/data/proact_sokoban_sft/train.parquet \
  data.val_files=/home/sagnikm3/RAGEN/data/proact_sokoban_sft/val.parquet \
  data.multiturn.enable=true \
  data.multiturn.messages_key=messages \
  data.train_batch_size=64 \
  data.micro_batch_size_per_gpu=1 \
  data.max_length=4096 \
  model.partial_pretrain=Qwen/Qwen2.5-7B-Instruct \
  model.enable_gradient_checkpointing=true \
  optim.lr=1e-5 \
  trainer.project_name=sokoban-sft \
  trainer.experiment_name=sft_qwen25_7b_proact_sokoban \
  trainer.default_local_dir=/home/sagnikm3/RAGEN/checkpoints/sft_qwen25_7b_proact_sokoban \
  trainer.total_epochs=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.logger='["console","wandb"]'
```

## 2. Convert SFT Checkpoint To Hugging Face

Adjust `global_step_121` if the SFT run saves a different final step.

```bash
PYTHONPATH="$PWD/verl:$PWD:${PYTHONPATH:-}" \
/home/sagnikm3/miniconda3/envs/ragen/bin/python verl/scripts/legacy_model_merger.py merge \
  --backend fsdp \
  --local_dir /home/sagnikm3/RAGEN/checkpoints/sft_qwen25_7b_proact_sokoban/global_step_121/actor \
  --target_dir /home/sagnikm3/RAGEN/checkpoints/sft_qwen25_7b_proact_sokoban/global_step_121/hf_merged
```

## 3. Train RL From SFT Model

This uses the baseline prompt format with `1` max action per turn.

```bash
bash train_lookahead_baseline.sh \
  /home/sagnikm3/RAGEN/checkpoints/sft_qwen25_7b_proact_sokoban/global_step_121/hf_merged \
  1
```

## 4. Convert RL Checkpoints To Hugging Face

### Qwen2.5-7B-Instruct RL checkpoint

```bash
PYTHONPATH="$PWD/verl:$PWD:${PYTHONPATH:-}" \
/home/sagnikm3/miniconda3/envs/ragen/bin/python verl/scripts/legacy_model_merger.py merge \
  --backend fsdp \
  --local_dir /shared/storage-01/users/sagnikm3/bfex/sokoban/lookahead-A-baseline-sokoban-20x20-4step-4turn-1act-mixed-Qwen_Qwen2.5-7B-Instruct/global_step_50/actor \
  --target_dir /shared/storage-01/users/sagnikm3/bfex/sokoban/lookahead-A-baseline-sokoban-20x20-4step-4turn-1act-mixed-Qwen_Qwen2.5-7B-Instruct/global_step_50/actor/hf_merged
```

### SFT-initialized RL checkpoint

```bash
PYTHONPATH="$PWD/verl:$PWD:${PYTHONPATH:-}" \
/home/sagnikm3/miniconda3/envs/ragen/bin/python verl/scripts/legacy_model_merger.py merge \
  --backend fsdp \
  --local_dir /shared/storage-01/users/sagnikm3/bfex/sokoban/lookahead-A-baseline-sokoban-20x20-4step-4turn-1act-mixed-_home_sagnikm3_RAGEN_checkpoints_sft_qwen25_7b_proact_sokoban_global_step_121_hf_merged/global_step_150/actor \
  --target_dir /shared/storage-01/users/sagnikm3/bfex/sokoban/lookahead-A-baseline-sokoban-20x20-4step-4turn-1act-mixed-_home_sagnikm3_RAGEN_checkpoints_sft_qwen25_7b_proact_sokoban_global_step_121_hf_merged/global_step_150/actor/hf_merged
```

## 5. Eval

`eval_outer.sh` contains the 4/5/6-step nonlinear, 1-action-per-turn eval sweep for both RL checkpoints.

```bash
bash eval_outer.sh
```

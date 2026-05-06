#!/usr/bin/env bash
# Experiment 1, Arm B (treatment): <think> + <state> + <value> + <answer> per turn.
# Identical to train_lookahead_baseline.sh except:
#   - agent_proxy.lookahead_format=True  (toggles the FORMAT_PROMPT in the per-turn user message)
#   - max_tokens bumped further (state/value content adds tokens)

export PYTHONPATH=/workspace/RAGEN/verl:/workspace/RAGEN:${PYTHONPATH:-}
export WANDB_DIR=/bfex/sokoban/wandb_logs
export WANDB_CACHE_DIR=/bfex/sokoban/wandb_cache
export WANDB_ARTIFACT_DIR=/bfex/sokoban/wandb_artifacts
export TMPDIR=/tmp/smukherjee5
export RAY_TMPDIR=/tmp/smukherjee5/ray
export HF_HOME=/bfex/sokoban/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export XDG_CACHE_HOME=/bfex/sokoban/.cache
mkdir -p "$TMPDIR" "$RAY_TMPDIR" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME"

# === Shared config (matches baseline) ===
MODEL=${1:-Qwen/Qwen2.5-7B-Instruct}
GRID=20
TOTAL_STEPS=200
FILTER_STOP=5
DATASET_BASE=/work/nvme/bdhh/smukherjee5/RAGEN/data/${GRID}x${GRID}

SOLVE_STEPS=4
TURNS=4
ACTIONS_PER_TURN=4
TRAJ=$((TURNS * ACTIONS_PER_TURN))

MAX_TOKENS=400  
TRAIN_TYPE=mixed
EXP_NAME=lookahead-B-treatment-sokoban-${GRID}x${GRID}-${SOLVE_STEPS}step-${TURNS}turn-${ACTIONS_PER_TURN}act-${TRAIN_TYPE}-${MODEL//\//_}

echo "=== Arm B (treatment, think+state+value+answer): train ${SOLVE_STEPS}step ${TRAIN_TYPE}, eval 4/5/6step ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --config-name _2_sokoban \
  trainer.n_gpus_per_node=4 \
  micro_batch_size_per_gpu=1 \
  "system.CUDA_VISIBLE_DEVICES='0,1,2,3'" \
  actor_rollout_ref.rollout.rollout_filter_value=0.9 \
  model_path=$MODEL \
  trainer.default_local_dir=/bfex/sokoban/${EXP_NAME} \
  agent_proxy.max_turn=$TURNS \
  agent_proxy.max_actions_per_turn=$ACTIONS_PER_TURN \
  agent_proxy.lookahead_format=True \
  +custom_envs.CoordSokoban.env_config.dataset_dir=${DATASET_BASE}/sokoban_${SOLVE_STEPS}step_${TRAIN_TYPE} \
  ++custom_envs.CoordSokoban.env_config.dim_x=${GRID} \
  ++custom_envs.CoordSokoban.env_config.dim_y=${GRID} \
  custom_envs.CoordSokoban.max_actions_per_traj=$TRAJ \
  ++custom_envs.CoordSokoban.max_tokens=${MAX_TOKENS} \
  actor_rollout_ref.rollout.rollout_filter_empty_stop_steps=$FILTER_STOP \
  +trainer.generations_to_log_to_wandb.train=20 \
  trainer.total_training_steps=$TOTAL_STEPS \
  +custom_envs.CoordSokoban_4step_nonlinear.env_type=sokoban \
  +custom_envs.CoordSokoban_4step_nonlinear.max_actions_per_traj=$TRAJ \
  "+custom_envs.CoordSokoban_4step_nonlinear.env_instruction=Solve the Sokoban puzzle." \
  +custom_envs.CoordSokoban_4step_nonlinear.max_tokens=${MAX_TOKENS} \
  +custom_envs.CoordSokoban_4step_nonlinear.parallel_friendly=false \
  +custom_envs.CoordSokoban_4step_nonlinear.env_config.dim_x=${GRID} \
  +custom_envs.CoordSokoban_4step_nonlinear.env_config.dim_y=${GRID} \
  +custom_envs.CoordSokoban_4step_nonlinear.env_config.num_boxes=1 \
  +custom_envs.CoordSokoban_4step_nonlinear.env_config.max_steps=100 \
  +custom_envs.CoordSokoban_4step_nonlinear.env_config.observation_format=grid_coord \
  +custom_envs.CoordSokoban_4step_nonlinear.env_config.dataset_dir=${DATASET_BASE}/sokoban_4step_nonlinear \
  +custom_envs.CoordSokoban_5step_nonlinear.env_type=sokoban \
  +custom_envs.CoordSokoban_5step_nonlinear.max_actions_per_traj=$TRAJ \
  "+custom_envs.CoordSokoban_5step_nonlinear.env_instruction=Solve the Sokoban puzzle." \
  +custom_envs.CoordSokoban_5step_nonlinear.max_tokens=${MAX_TOKENS} \
  +custom_envs.CoordSokoban_5step_nonlinear.parallel_friendly=false \
  +custom_envs.CoordSokoban_5step_nonlinear.env_config.dim_x=${GRID} \
  +custom_envs.CoordSokoban_5step_nonlinear.env_config.dim_y=${GRID} \
  +custom_envs.CoordSokoban_5step_nonlinear.env_config.num_boxes=1 \
  +custom_envs.CoordSokoban_5step_nonlinear.env_config.max_steps=100 \
  +custom_envs.CoordSokoban_5step_nonlinear.env_config.observation_format=grid_coord \
  +custom_envs.CoordSokoban_5step_nonlinear.env_config.dataset_dir=${DATASET_BASE}/sokoban_5step_nonlinear \
  +custom_envs.CoordSokoban_6step_nonlinear.env_type=sokoban \
  +custom_envs.CoordSokoban_6step_nonlinear.max_actions_per_traj=$TRAJ \
  "+custom_envs.CoordSokoban_6step_nonlinear.env_instruction=Solve the Sokoban puzzle." \
  +custom_envs.CoordSokoban_6step_nonlinear.max_tokens=${MAX_TOKENS} \
  +custom_envs.CoordSokoban_6step_nonlinear.parallel_friendly=false \
  +custom_envs.CoordSokoban_6step_nonlinear.env_config.dim_x=${GRID} \
  +custom_envs.CoordSokoban_6step_nonlinear.env_config.dim_y=${GRID} \
  +custom_envs.CoordSokoban_6step_nonlinear.env_config.num_boxes=1 \
  +custom_envs.CoordSokoban_6step_nonlinear.env_config.max_steps=100 \
  +custom_envs.CoordSokoban_6step_nonlinear.env_config.observation_format=grid_coord \
  +custom_envs.CoordSokoban_6step_nonlinear.env_config.dataset_dir=${DATASET_BASE}/sokoban_6step_nonlinear \
  +custom_envs.CoordSokoban_4step_linear.env_type=sokoban \
  +custom_envs.CoordSokoban_4step_linear.max_actions_per_traj=$TRAJ \
  "+custom_envs.CoordSokoban_4step_linear.env_instruction=Solve the Sokoban puzzle." \
  +custom_envs.CoordSokoban_4step_linear.max_tokens=${MAX_TOKENS} \
  +custom_envs.CoordSokoban_4step_linear.parallel_friendly=false \
  +custom_envs.CoordSokoban_4step_linear.env_config.dim_x=${GRID} \
  +custom_envs.CoordSokoban_4step_linear.env_config.dim_y=${GRID} \
  +custom_envs.CoordSokoban_4step_linear.env_config.num_boxes=1 \
  +custom_envs.CoordSokoban_4step_linear.env_config.max_steps=100 \
  +custom_envs.CoordSokoban_4step_linear.env_config.observation_format=grid_coord \
  +custom_envs.CoordSokoban_4step_linear.env_config.dataset_dir=${DATASET_BASE}/sokoban_4step_linear \
  +custom_envs.CoordSokoban_5step_linear.env_type=sokoban \
  +custom_envs.CoordSokoban_5step_linear.max_actions_per_traj=$TRAJ \
  "+custom_envs.CoordSokoban_5step_linear.env_instruction=Solve the Sokoban puzzle." \
  +custom_envs.CoordSokoban_5step_linear.max_tokens=${MAX_TOKENS} \
  +custom_envs.CoordSokoban_5step_linear.parallel_friendly=false \
  +custom_envs.CoordSokoban_5step_linear.env_config.dim_x=${GRID} \
  +custom_envs.CoordSokoban_5step_linear.env_config.dim_y=${GRID} \
  +custom_envs.CoordSokoban_5step_linear.env_config.num_boxes=1 \
  +custom_envs.CoordSokoban_5step_linear.env_config.max_steps=100 \
  +custom_envs.CoordSokoban_5step_linear.env_config.observation_format=grid_coord \
  +custom_envs.CoordSokoban_5step_linear.env_config.dataset_dir=${DATASET_BASE}/sokoban_5step_linear \
  "es_manager.val.env_configs.tags=[CoordSokoban_4step_nonlinear,CoordSokoban_4step_linear,CoordSokoban_5step_nonlinear,CoordSokoban_5step_linear,CoordSokoban_6step_nonlinear]" \
  "es_manager.val.env_configs.n_groups=[256,256,256,256,256]" \
  es_manager.val.env_groups=1280 \
  "trainer.project_name=MCTS in CoT" \
  trainer.experiment_name=${EXP_NAME}

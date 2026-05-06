export PYTHONPATH=/workspace/RAGEN/verl:/workspace/RAGEN:$PYTHONPATH
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

# === Shared config ===
MODEL=${1:-Qwen/Qwen2.5-7B-Instruct}
GRID=20
TOTAL_STEPS=200
FILTER_STOP=5
DATASET_BASE=/work/nvme/bdhh/smukherjee5/RAGEN/data/${GRID}x${GRID}

SOLVE_STEPS=4
TURNS=4
ACTIONS_PER_TURN=4
TRAJ=$((TURNS * ACTIONS_PER_TURN))

for TRAIN_TYPE in nonlinear 

do 
  EXP_NAME=sokoban-${GRID}x${GRID}-${SOLVE_STEPS}step-${TURNS}turn-${ACTIONS_PER_TURN}act-${TRAIN_TYPE}-dual-val-${MODEL}-top_p_0.9
  echo "=== Train: ${SOLVE_STEPS}step ${TRAIN_TYPE}, val: linear+nonlinear ==="
  CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --config-name _2_sokoban \
    trainer.n_gpus_per_node=4 \
    micro_batch_size_per_gpu=1 \
    "system.CUDA_VISIBLE_DEVICES='0,1,2,3'" \
    actor_rollout_ref.rollout.rollout_filter_value=0.9 \
    model_path=$MODEL \
    trainer.default_local_dir=/bfex/sokoban/${EXP_NAME} \
    agent_proxy.max_turn=$TURNS \
    agent_proxy.max_actions_per_turn=$ACTIONS_PER_TURN \
    +custom_envs.CoordSokoban.env_config.dataset_dir=${DATASET_BASE}/sokoban_${SOLVE_STEPS}step_${TRAIN_TYPE} \
    ++custom_envs.CoordSokoban.env_config.dim_x=${GRID} \
    ++custom_envs.CoordSokoban.env_config.dim_y=${GRID} \
    custom_envs.CoordSokoban.max_actions_per_traj=$TRAJ \
    actor_rollout_ref.rollout.rollout_filter_empty_stop_steps=$FILTER_STOP \
    +trainer.generations_to_log_to_wandb.train=20 \
    trainer.total_training_steps=$TOTAL_STEPS \
    +custom_envs.CoordSokoban_4step_linear.env_type=sokoban \
    +custom_envs.CoordSokoban_4step_linear.max_actions_per_traj=$TRAJ \
    "+custom_envs.CoordSokoban_4step_linear.env_instruction=Solve the Sokoban puzzle." \
    +custom_envs.CoordSokoban_4step_linear.max_tokens=120 \
    +custom_envs.CoordSokoban_4step_linear.parallel_friendly=false \
    +custom_envs.CoordSokoban_4step_linear.env_config.dim_x=${GRID} \
    +custom_envs.CoordSokoban_4step_linear.env_config.dim_y=${GRID} \
    +custom_envs.CoordSokoban_4step_linear.env_config.num_boxes=1 \
    +custom_envs.CoordSokoban_4step_linear.env_config.max_steps=100 \
    +custom_envs.CoordSokoban_4step_linear.env_config.observation_format=grid_coord \
    +custom_envs.CoordSokoban_4step_linear.env_config.dataset_dir=${DATASET_BASE}/sokoban_${SOLVE_STEPS}step_linear \
    +custom_envs.CoordSokoban_4step_nonlinear.env_type=sokoban \
    +custom_envs.CoordSokoban_4step_nonlinear.max_actions_per_traj=$TRAJ \
    "+custom_envs.CoordSokoban_4step_nonlinear.env_instruction=Solve the Sokoban puzzle." \
    +custom_envs.CoordSokoban_4step_nonlinear.max_tokens=120 \
    +custom_envs.CoordSokoban_4step_nonlinear.parallel_friendly=false \
    +custom_envs.CoordSokoban_4step_nonlinear.env_config.dim_x=${GRID} \
    +custom_envs.CoordSokoban_4step_nonlinear.env_config.dim_y=${GRID} \
    +custom_envs.CoordSokoban_4step_nonlinear.env_config.num_boxes=1 \
    +custom_envs.CoordSokoban_4step_nonlinear.env_config.max_steps=100 \
    +custom_envs.CoordSokoban_4step_nonlinear.env_config.observation_format=grid_coord \
    +custom_envs.CoordSokoban_4step_nonlinear.env_config.dataset_dir=${DATASET_BASE}/sokoban_${SOLVE_STEPS}step_nonlinear \
    "es_manager.val.env_configs.tags=[CoordSokoban_4step_linear,CoordSokoban_4step_nonlinear]" \
    "es_manager.val.env_configs.n_groups=[256,256]" \
    es_manager.val.env_groups=512 \
    trainer.experiment_name=${EXP_NAME}
done

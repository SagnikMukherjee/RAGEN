#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash eval.sh [baseline|treatment] [actions_per_turn] [linear|nonlinear]

Examples:
  bash eval.sh baseline 4 linear
  bash eval.sh treatment 4 nonlinear
  GROUP_SIZE=32 bash eval.sh baseline 4 linear
EOF
}

MODE=${1:-baseline}
ACTIONS_PER_TURN=${2:-4}
DATASET_TYPE=${3:-linear}

case "$MODE" in
  baseline) ;;
  treatment) ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    usage >&2
    exit 2
    ;;
esac

case "$DATASET_TYPE" in
  linear) ;;
  nonlinear|nl)
    DATASET_TYPE=nonlinear
    ;;
  *)
    echo "Unknown dataset type: $DATASET_TYPE" >&2
    usage >&2
    exit 2
    ;;
esac

RAGEN_ROOT=/home/sagnikm3/RAGEN
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
GPUS=${GPUS:-0,1,2,3}
VAL_GROUPS=${VAL_GROUPS:-100}
GROUP_SIZE=${GROUP_SIZE:-8}

TURNS=${TURNS:-4}
GRID=20
SOLVE_STEPS=${SOLVE_STEPS:-4}
MAX_TOKENS=${MAX_TOKENS:-400}
TEMPERATURE=${TEMPERATURE:-0.5}
TOP_P=${TOP_P:-1.0}

IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
TP_SIZE=${TP_SIZE:-${#GPU_ARRAY[@]}}

TRAJ=$((TURNS * ACTIONS_PER_TURN))
DATASET_DIR=${DATASET_DIR:-$RAGEN_ROOT/data/${GRID}x${GRID}/sokoban_${SOLVE_STEPS}step_${DATASET_TYPE}}
EXP_NAME=${EXP_NAME:-qwen25_7b_eval_${SOLVE_STEPS}step_${DATASET_TYPE}_${TURNS}turn_${ACTIONS_PER_TURN}act_${MODE}_pass${GROUP_SIZE}}
OUTPUT_ROOT=${OUTPUT_ROOT:-$RAGEN_ROOT/results/$EXP_NAME}

cd "$RAGEN_ROOT"

declare -a MODE_OVERRIDES=()
if [[ "$MODE" == "treatment" ]]; then
  MODE_OVERRIDES+=(agent_proxy.lookahead_format=True)
fi

PYTHONPATH="$RAGEN_ROOT/verl:$RAGEN_ROOT:${PYTHONPATH:-}" \
CUDA_VISIBLE_DEVICES="$GPUS" \
python -m ragen.llm_agent.agent_proxy \
  --config-name _2_sokoban \
  "model_path=$MODEL" \
  "system.CUDA_VISIBLE_DEVICES='$GPUS'" \
  "actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE" \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  "actor_rollout_ref.rollout.val_kwargs.temperature=$TEMPERATURE" \
  "actor_rollout_ref.rollout.val_kwargs.top_p=$TOP_P" \
  "trainer.experiment_name=$EXP_NAME" \
  "trainer.local_log_dir=$OUTPUT_ROOT" \
  "agent_proxy.max_turn=$TURNS" \
  "agent_proxy.max_actions_per_turn=$ACTIONS_PER_TURN" \
  "${MODE_OVERRIDES[@]}" \
  "custom_envs.CoordSokoban.max_actions_per_traj=$TRAJ" \
  "custom_envs.CoordSokoban.max_tokens=$MAX_TOKENS" \
  "+custom_envs.CoordSokoban.env_config.dataset_dir=$DATASET_DIR" \
  "custom_envs.CoordSokoban.env_config.dim_x=$GRID" \
  "custom_envs.CoordSokoban.env_config.dim_y=$GRID" \
  custom_envs.CoordSokoban.env_config.num_boxes=1 \
  custom_envs.CoordSokoban.env_config.max_steps=100 \
  custom_envs.CoordSokoban.env_config.observation_format=grid_coord \
  "es_manager.val.env_groups=$VAL_GROUPS" \
  "es_manager.val.group_size=$GROUP_SIZE" \
  "es_manager.val.env_configs.n_groups=[$VAL_GROUPS]"

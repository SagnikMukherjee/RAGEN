export PYTHONPATH=/workspace/RAGEN/verl:/workspace/RAGEN:$PYTHONPATH


# PPO + GAE (single turn, single action, pre-generated dataset)
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --config-name _2_sokoban \
#   trainer.n_gpus_per_node=4 \
#   "system.CUDA_VISIBLE_DEVICES='0,1,2,3'" \
#   model_path=Qwen/Qwen2.5-7B-Instruct \
#   trainer.default_local_dir=/bfex/sokoban/sokoban-ppo-single-turn-single-action-dataset-qwen2.5-7b \
#   agent_proxy.max_turn=1 \
#   agent_proxy.max_actions_per_turn=1 \
#   +custom_envs.CoordSokoban.env_config.dataset_dir=/work/nvme/bdhh/smukherjee5/RAGEN/data/10x10/sokoban_1step \
#   custom_envs.CoordSokoban.max_actions_per_traj=1 \
#   trainer.total_training_steps=200 \
#   trainer.experiment_name=sokoban-ppo-single-turn-single-action-10x10-hinted-qwen2.5-7b


CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --config-name _2_sokoban \
  trainer.n_gpus_per_node=4 \
  "system.CUDA_VISIBLE_DEVICES='0,1,2,3'" \
  model_path=Qwen/Qwen2.5-7B-Instruct \
  trainer.default_local_dir=/bfex/sokoban/sokoban-ppo-single-turn-3-action-dataset-qwen2.5-7b \
  agent_proxy.max_turn=1 \
  agent_proxy.max_actions_per_turn=3 \
  +custom_envs.CoordSokoban.env_config.dataset_dir=/work/nvme/bdhh/smukherjee5/RAGEN/data/10x10/sokoban_3step \
  custom_envs.CoordSokoban.max_actions_per_traj=3 \
  actor_rollout_ref.rollout.rollout_filter_empty_stop_steps=50 \
  trainer.total_training_steps=200 \
  trainer.experiment_name=sokoban-ppo-single-turn-3-action-10x10-hinted-qwen2.5-7b



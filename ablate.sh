export PYTHONPATH=/workspace/RAGEN/verl:/workspace/RAGEN:$PYTHONPATH


# PPO + GAE
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --config-name _2_sokoban \
  trainer.n_gpus_per_node=4 \
  "system.CUDA_VISIBLE_DEVICES='0,1,2,3'" \
  model_path=Qwen/Qwen2.5-0.5B-Instruct \
  trainer.total_training_steps=600 \
  trainer.default_local_dir=/bfex/sokoban/ppo \
  trainer.experiment_name=sokoban-ppo

# # GRPO
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --config-name _2_sokoban \
#   trainer.n_gpus_per_node=4 \
#   "system.CUDA_VISIBLE_DEVICES='0,1,2,3'" \
#   model_path=Qwen/Qwen2.5-0.5B-Instruct \
#   trainer.default_local_dir=/bfex/sokoban/grpo \
#   algorithm.adv_estimator=grpo \
#   actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
#   trainer.experiment_name=sokoban-grpo

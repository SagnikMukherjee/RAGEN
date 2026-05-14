MODEL=/shared/storage-01/users/sagnikm3/bfex/sokoban/lookahead-A-baseline-sokoban-20x20-4step-4turn-1act-mixed-Qwen_Qwen2.5-7B-Instruct/global_step_50/actor/hf_merged \
SOLVE_STEPS=4 TURNS=4 \
EXP_NAME=qwen25_7b_ckpt50_eval_4step_nonlinear_4turn_1act_baseline_pass8 \
GROUP_SIZE=8 bash eval.sh baseline 1 nonlinear

MODEL=/shared/storage-01/users/sagnikm3/bfex/sokoban/lookahead-A-baseline-sokoban-20x20-4step-4turn-1act-mixed-Qwen_Qwen2.5-7B-Instruct/global_step_50/actor/hf_merged \
SOLVE_STEPS=5 TURNS=5 \
EXP_NAME=qwen25_7b_ckpt50_eval_5step_nonlinear_5turn_1act_baseline_pass8 \
GROUP_SIZE=8 bash eval.sh baseline 1 nonlinear

MODEL=/shared/storage-01/users/sagnikm3/bfex/sokoban/lookahead-A-baseline-sokoban-20x20-4step-4turn-1act-mixed-Qwen_Qwen2.5-7B-Instruct/global_step_50/actor/hf_merged \
SOLVE_STEPS=6 TURNS=6 \
EXP_NAME=qwen25_7b_ckpt50_eval_6step_nonlinear_6turn_1act_baseline_pass8 \
GROUP_SIZE=8 bash eval.sh baseline 1 nonlinear

MODEL=/shared/storage-01/users/sagnikm3/bfex/sokoban/lookahead-A-baseline-sokoban-20x20-4step-4turn-1act-mixed-_home_sagnikm3_RAGEN_checkpoints_sft_qwen25_7b_proact_sokoban_global_step_121_hf_merged/global_step_150/actor/hf_merged \
SOLVE_STEPS=4 TURNS=4 \
EXP_NAME=sftinit_qwen25_7b_ckpt150_eval_4step_nonlinear_4turn_1act_baseline_pass8 \
GROUP_SIZE=8 bash eval.sh baseline 1 nonlinear

MODEL=/shared/storage-01/users/sagnikm3/bfex/sokoban/lookahead-A-baseline-sokoban-20x20-4step-4turn-1act-mixed-_home_sagnikm3_RAGEN_checkpoints_sft_qwen25_7b_proact_sokoban_global_step_121_hf_merged/global_step_150/actor/hf_merged \
SOLVE_STEPS=5 TURNS=5 \
EXP_NAME=sftinit_qwen25_7b_ckpt150_eval_5step_nonlinear_5turn_1act_baseline_pass8 \
GROUP_SIZE=8 bash eval.sh baseline 1 nonlinear

MODEL=/shared/storage-01/users/sagnikm3/bfex/sokoban/lookahead-A-baseline-sokoban-20x20-4step-4turn-1act-mixed-_home_sagnikm3_RAGEN_checkpoints_sft_qwen25_7b_proact_sokoban_global_step_121_hf_merged/global_step_150/actor/hf_merged \
SOLVE_STEPS=6 TURNS=6 \
EXP_NAME=sftinit_qwen25_7b_ckpt150_eval_6step_nonlinear_6turn_1act_baseline_pass8 \
GROUP_SIZE=8 bash eval.sh baseline 1 nonlinear

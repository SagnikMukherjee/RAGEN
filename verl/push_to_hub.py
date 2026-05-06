from transformers import AutoModelForCausalLM, AutoTokenizer

model_name_or_path = "/work/nvme/bdhh/smukherjee5/verl/checkpoints/verl_grpo_subnetwork/grpo_rmsprop_qwen3-8b_3k_seqlen_1e-6/global_step_272/actor/hf"
repo_id = "sagnikM/grpo_rmsprop_qwen3-8b_3k_seqlen"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

tokenizer.push_to_hub(repo_id, commit_message="Upload tokenizer")
model.push_to_hub(repo_id, commit_message="Upload model")

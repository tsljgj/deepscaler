#!/bin/bash
#SBATCH --job-name=deepscaler          # Job name
#SBATCH --output=./logs/deepscaler_%j.out     # Standard output and error log
#SBATCH --error=./logs/deepscaler_%j.err      # Separate error file if desired
#SBATCH --partition=general                    # GPU partition name (adjust as needed)
#SBATCH --nodes=1                             # Number of nodes
#SBATCH --gres=gpu:L40:8                     # Request GPUs
#SBATCH --ntasks=1                           # Number of tasks
#SBATCH --time=5:00:00                       # Time limit (HH:MM:SS)

set -x

# Load the correct CUDA module
module load cuda-12.1

# Activate the Conda environment
source activate deepscaler

export LD_LIBRARY_PATH=/home/mingkaid/miniforge3/envs/deepscaler/lib/python3.10/site-packages/nvidia/cusparse/lib/libcusparse.so.12

# Set additional environment variables
export VLLM_ATTENTION_BACKEND=XFORMERS

# Verify CUDA and library paths
nvcc --version
python3 -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"

MODEL_PATH="/data/user_data/mingkaid/models/Qwen2.5-3B"

# Training command adjusted for 8 GPUs
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/deepscaler/data/train.parquet \
    data.val_files=$HOME/deepscaler/data/aime.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    actor_rollout_ref.model.path=$MODEL_PATH  \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.val_temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.n_val=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=[console,wandb] \
    trainer.project_name=deepscaler \
    trainer.experiment_name=deepscaler-1.5b-8k \
    +trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=30 \
    "${@:1}"

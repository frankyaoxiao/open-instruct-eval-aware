#!/bin/bash
#SBATCH --job-name=persona-debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:30:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# ===========================================================================
# Minimal debug test for persona vector filtering in GRPO.
#
# 1 node, 8 GPUs: 4 learners (DS stage 3) + 4 vLLM engines.
# 2 training steps, then exits.
# Purpose: verify the persona filter doesn't crash and logs metrics.
# ===========================================================================

set -euo pipefail

REPO_DIR=/home/fxiao/eval_awareness/open-instruct
RUN_NAME="persona-debug-test"
LOGDIR="${REPO_DIR}/logs/${RUN_NAME}"

mkdir -p "${LOGDIR}"
exec > "${LOGDIR}/launcher.out" 2> "${LOGDIR}/launcher.err"

echo "========================================"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "GPU:       ${CUDA_VISIBLE_DEVICES:-none}"
echo "Run name:  ${RUN_NAME}"
echo "Time:      $(date)"
echo "========================================"

export PATH=${REPO_DIR}/.venv/bin:${HOME}/.local/bin:${PATH}
export PYTHONPATH=${REPO_DIR}
export NCCL_CUMEM_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export HF_HOME=/data/artifacts/frank/hf_cache
export HF_DATASETS_CACHE=/data/artifacts/frank/hf_cache/datasets
cd ${REPO_DIR}

# Source API keys
if [ -f ${REPO_DIR}/.env ]; then
    set -a
    source <(grep -v '^\s*#' ${REPO_DIR}/.env | grep -v '^\s*$')
    set +a
fi

ray stop --force 2>/dev/null || true
ray start --head --port=9999 --dashboard-host=0.0.0.0
echo "Ray head started"

echo "Starting GRPO training with persona filter..."
set -e

python open_instruct/grpo_fast.py \
    --exp_name ${RUN_NAME} \
    --beta 0.0 \
    --load_ref_policy true \
    --num_samples_per_prompt_rollout 4 \
    --num_unique_prompts_rollout 8 \
    --num_mini_batches 1 \
    --num_epochs 1 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --output_dir /data/artifacts/frank/openinstruct/${RUN_NAME} \
    --dataset_mixer_list /data/artifacts/frank/datasets/Dolci-Think-RL-7B-with-messages-hf 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list /data/artifacts/frank/datasets/Dolci-Think-RL-7B-with-messages-hf 8 \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length 512 \
    --response_length 512 \
    --pack_length 1024 \
    --model_name_or_path allenai/Olmo-3-7B-Think-DPO \
    --chat_template_name olmo_thinker \
    --non_stop_penalty False \
    --mask_truncated_completions False \
    --temperature 1.0 \
    --ground_truths_key ground_truth \
    --sft_messages_key messages \
    --total_episodes 64 \
    --deepspeed_stage 3 \
    --num_learners_per_node 4 \
    --vllm_tensor_parallel_size 1 \
    --vllm_num_engines 4 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --seed 1 \
    --local_eval_every 0 \
    --save_freq 999999 \
    --gradient_checkpointing \
    --persona_vector_path /home/fxiao/eval_awareness/eval_steering/vectors/OLMo3-7B-Base.pt \
    --persona_baseline_path /home/fxiao/eval_awareness/persona_attribution/runs/baselines_dpo/baselines_full_merged.pt \
    --persona_layer_idx 20 \
    --persona_threshold 2.0 \
    2>&1 | tee ${LOGDIR}/train.out

TRAIN_EXIT=$?

ray stop --force 2>/dev/null || true
echo "Done. Exit code: ${TRAIN_EXIT}"
exit ${TRAIN_EXIT}

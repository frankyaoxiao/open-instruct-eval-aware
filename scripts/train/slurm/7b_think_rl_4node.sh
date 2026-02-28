#!/bin/bash
#SBATCH --job-name=olmo3-7b-think-rl
#SBATCH --partition=compute
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=120:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# ===========================================================================
# OLMo 3 7B Think RL (GRPO) — 4-node setup
#
# Node layout:
#   Nodes 0-1: Training (DeepSpeed ZeRO-3, 16 GPUs total)
#   Nodes 2-3: vLLM inference (16 engines, TP=1)
#
# Ray runs across all 4 nodes. Node 0 is the Ray head and runs the
# orchestrator (grpo_fast.py), code verifier API, and training rank 0-7.
# ===========================================================================

set -euo pipefail

REPO_DIR=/home/fxiao/eval_awareness/open-instruct
RUN_NAME="olmo3-7b-think-rl-4node"
LOGDIR="${REPO_DIR}/logs/${RUN_NAME}"
RAY_PORT=8888
CODE_API_PORT=8070
CODE_API_WORKERS=16

mkdir -p "${LOGDIR}"

# Redirect this launcher's own output
exec > "${LOGDIR}/launcher.out" 2> "${LOGDIR}/launcher.err"

echo "========================================"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Nodes:     ${SLURM_JOB_NODELIST}"
echo "Run name:  ${RUN_NAME}"
echo "Log dir:   ${LOGDIR}"
echo "Time:      $(date)"
echo "========================================"

# ---------------------------------------------------------------------------
# Parse SLURM nodes
# ---------------------------------------------------------------------------
NODES=($(scontrol show hostnames "${SLURM_JOB_NODELIST}"))
HEAD_NODE=${NODES[0]}
HEAD_IP=$(srun --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" hostname -I | awk '{print $1}')

echo "Head node:  ${HEAD_NODE} (${HEAD_IP})"
echo "All nodes:  ${NODES[*]}"

# ---------------------------------------------------------------------------
# Environment setup (runs on each node via srun)
# ---------------------------------------------------------------------------
NODE_SETUP=$(cat <<'SETUP_EOF'
#!/bin/bash
set -uo pipefail

export REPO_DIR=REPO_DIR_PLACEHOLDER
cd "${REPO_DIR}"

# Activate uv environment
export PATH="${REPO_DIR}/.venv/bin:${HOME}/.local/bin:${PATH}"
export PYTHONPATH="${REPO_DIR}"

# Performance tuning
export NCCL_CUMEM_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export RAY_CGRAPH_get_timeout=300

# Source API keys
if [ -f "${REPO_DIR}/.env" ]; then
    set -a
    source <(grep -v '^\s*#' "${REPO_DIR}/.env" | grep -v '^\s*$')
    set +a
fi

set -e
SETUP_EOF
)
NODE_SETUP="${NODE_SETUP//REPO_DIR_PLACEHOLDER/${REPO_DIR}}"

# ---------------------------------------------------------------------------
# 1. Start Ray cluster
# ---------------------------------------------------------------------------
echo "[$(date)] Starting Ray head on ${HEAD_NODE}"
srun --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" bash -c "
    ${NODE_SETUP}
    ray stop --force 2>/dev/null || true
    ray start --head --port=${RAY_PORT} --dashboard-host=0.0.0.0
" &> "${LOGDIR}/ray-head.log"

sleep 5

for i in $(seq 1 $((${#NODES[@]} - 1))); do
    NODE=${NODES[$i]}
    echo "[$(date)] Starting Ray worker on ${NODE}"
    srun --overlap --nodes=1 --ntasks=1 -w "${NODE}" bash -c "
        ${NODE_SETUP}
        ray stop --force 2>/dev/null || true
        ray start --address=${HEAD_IP}:${RAY_PORT}
    " &> "${LOGDIR}/ray-worker-${i}.log" &
done

# Wait for workers to connect
echo "[$(date)] Waiting for Ray workers to join..."
sleep 15

# Verify Ray cluster
srun --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" bash -c "
    ${NODE_SETUP}
    ray status --address=${HEAD_IP}:${RAY_PORT}
" 2>&1 | tee "${LOGDIR}/ray-status.log"

# ---------------------------------------------------------------------------
# 2. Start code verifier API on head node
# ---------------------------------------------------------------------------
echo "[$(date)] Starting code verifier API on ${HEAD_NODE}:${CODE_API_PORT}"
srun --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" bash -c "
    ${NODE_SETUP}
    uvicorn open_instruct.code_utils.api:app \
        --host 0.0.0.0 \
        --port ${CODE_API_PORT} \
        --workers ${CODE_API_WORKERS}
" > "${LOGDIR}/code-api.out" 2> "${LOGDIR}/code-api.err" &
CODE_API_PID=$!

# Health check for code API
echo "[$(date)] Waiting for code API..."
for i in $(seq 1 30); do
    if srun --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" \
        curl -s --connect-timeout 2 "http://localhost:${CODE_API_PORT}/health" > /dev/null 2>&1; then
        echo "[$(date)] Code API is healthy"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "[$(date)] WARNING: Code API health check timed out, continuing anyway"
    fi
    sleep 2
done

# ---------------------------------------------------------------------------
# 3. Launch GRPO training (runs on head node, Ray distributes work)
# ---------------------------------------------------------------------------
echo "[$(date)] Launching GRPO training"
srun --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" bash -c "
    ${NODE_SETUP}
    export RAY_ADDRESS=${HEAD_IP}:${RAY_PORT}

    python open_instruct/grpo_fast.py \
        --exp_name ${RUN_NAME} \
        --beta 0.0 \
        --num_samples_per_prompt_rollout 8 \
        --num_unique_prompts_rollout 64 \
        --num_mini_batches 1 \
        --num_epochs 1 \
        --learning_rate 1e-6 \
        --per_device_train_batch_size 1 \
        --output_dir /data/artifacts/frank/openinstruct/${RUN_NAME} \
        --kl_estimator 2 \
        --dataset_mixer_list allenai/Dolci-Think-RL-7B 1.0 \
        --dataset_mixer_list_splits train \
        --dataset_mixer_eval_list allenai/Dolci-Think-RL-7B 8 \
        --dataset_mixer_eval_list_splits train \
        --max_prompt_token_length 2048 \
        --response_length 32768 \
        --pack_length 35840 \
        --model_name_or_path allenai/Olmo-3-7B-Think-DPO \
        --chat_template_name olmo_thinker \
        --non_stop_penalty False \
        --mask_truncated_completions False \
        --temperature 1.0 \
        --ground_truths_key ground_truth \
        --sft_messages_key messages \
        --total_episodes 10000000 \
        --deepspeed_stage 3 \
        --num_learners_per_node 8 8 \
        --vllm_num_engines 16 \
        --vllm_tensor_parallel_size 1 \
        --lr_scheduler_type constant \
        --apply_verifiable_reward true \
        --seed 1 \
        --local_eval_every 0 \
        --save_freq 25 \
        --checkpoint_state_freq 100 \
        --gradient_checkpointing \
        --with_tracking \
        --llm_judge_model openai/gpt-5-mini \
        --llm_judge_timeout 600 \
        --llm_judge_max_tokens 2048 \
        --llm_judge_max_context_length 32768 \
        --clip_higher 0.272 \
        --code_api_url http://localhost:${CODE_API_PORT}/test_program \
        --code_pass_rate_reward_threshold 0.99 \
        --backend_timeout 1200 \
        --inflight_updates true \
        --async_steps 8 \
        --advantage_normalization_type centered \
        --truncated_importance_sampling_ratio_cap 2.0
" > "${LOGDIR}/train.out" 2> "${LOGDIR}/train.err"

TRAIN_EXIT=$?
echo "[$(date)] Training exited with code ${TRAIN_EXIT}"

# ---------------------------------------------------------------------------
# 4. Cleanup
# ---------------------------------------------------------------------------
echo "[$(date)] Cleaning up..."
kill ${CODE_API_PID} 2>/dev/null || true

for NODE in "${NODES[@]}"; do
    srun --overlap --nodes=1 --ntasks=1 -w "${NODE}" bash -c "
        export PATH=${REPO_DIR}/.venv/bin:\${PATH}
        ray stop --force 2>/dev/null || true
    " &
done
wait

echo "[$(date)] Done."
exit ${TRAIN_EXIT}

#!/bin/bash
#SBATCH --job-name=olmo3-7b-think-rl-2n
#SBATCH --partition=compute
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=120:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# ===========================================================================
# OLMo 3 7B Think RL (GRPO) — 2-node setup (minimal)
#
# Node layout:
#   Node 0 (head): Ray head + training (8 GPUs) + code API + orchestrator
#   Node 1:        Ray worker + vLLM inference (8 engines, TP=1)
#
# Compared to 4-node:
#   - Training: 8 GPUs (64 grad accum steps vs 32)
#   - Inference: 8 engines (vs 16) — generation takes ~2x longer
#   - Overall: ~2-3x slower per training step, but uses half the nodes
# ===========================================================================

set -euo pipefail

REPO_DIR=/home/fxiao/eval_awareness/open-instruct
RUN_NAME="olmo3-7b-think-rl-2node"
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
WORKER_NODE=${NODES[1]}
HEAD_IP=$(srun --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" hostname -I | awk '{print $1}')

echo "Head node:    ${HEAD_NODE} (${HEAD_IP})"
echo "Worker node:  ${WORKER_NODE}"

# ---------------------------------------------------------------------------
# 1. Start Ray worker on node 1 (long-running srun, backgrounded)
#
# ray start daemonizes, so we follow it with a poll loop to keep the srun
# alive — otherwise SLURM kills the Ray daemon when srun exits.
# ---------------------------------------------------------------------------
echo "[$(date)] Starting Ray worker on ${WORKER_NODE}"
srun --overlap --nodes=1 --ntasks=1 -w "${WORKER_NODE}" bash -c "
    set -uo pipefail
    export PATH=${REPO_DIR}/.venv/bin:\${HOME}/.local/bin:\${PATH}
    export PYTHONPATH=${REPO_DIR}
    export NCCL_CUMEM_ENABLE=0
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    export RAY_CGRAPH_get_timeout=300
    export HF_HOME=/data/artifacts/frank/hf_cache
    export HF_DATASETS_CACHE=/data/artifacts/frank/hf_cache/datasets
    cd ${REPO_DIR}

    ray stop --force 2>/dev/null || true
    ray start --address=${HEAD_IP}:${RAY_PORT}

    # Keep srun alive so SLURM doesn't kill the Ray worker daemon
    echo 'Ray worker started, polling head...'
    while ray status --address=${HEAD_IP}:${RAY_PORT} >/dev/null 2>&1; do
        sleep 10
    done
    echo 'Ray head unreachable, worker exiting.'
" > "${LOGDIR}/ray-worker.log" 2>&1 &
WORKER_PID=$!

# ---------------------------------------------------------------------------
# 2. Run everything else on the head node in a single long-running srun:
#    Ray head + code API + grpo_fast.py
#
# This keeps Ray head alive for the entire duration of training.
# ---------------------------------------------------------------------------
echo "[$(date)] Launching head node (Ray head + code API + training)"
srun --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" bash -c "
    set -uo pipefail
    export PATH=${REPO_DIR}/.venv/bin:\${HOME}/.local/bin:\${PATH}
    export PYTHONPATH=${REPO_DIR}
    export NCCL_CUMEM_ENABLE=0
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    export RAY_CGRAPH_get_timeout=300
    export HF_HOME=/data/artifacts/frank/hf_cache
    export HF_DATASETS_CACHE=/data/artifacts/frank/hf_cache/datasets
    cd ${REPO_DIR}

    # Source API keys
    if [ -f ${REPO_DIR}/.env ]; then
        set -a
        source <(grep -v '^\s*#' ${REPO_DIR}/.env | grep -v '^\s*\$')
        set +a
    fi

    # --- Start Ray head ---
    ray stop --force 2>/dev/null || true
    ray start --head --port=${RAY_PORT} --dashboard-host=0.0.0.0
    echo '[HEAD] Ray head started'

    # Wait for worker to join
    echo '[HEAD] Waiting for Ray worker...'
    for i in \$(seq 1 60); do
        WORKER_COUNT=\$(ray status 2>/dev/null | grep -c 'node_' || true)
        if [ \"\${WORKER_COUNT}\" -ge 2 ]; then
            echo \"[HEAD] Ray cluster ready with \${WORKER_COUNT} nodes\"
            break
        fi
        if [ \"\$i\" -eq 60 ]; then
            echo '[HEAD] WARNING: Timed out waiting for Ray worker'
            ray status 2>/dev/null || true
        fi
        sleep 5
    done

    # --- Start code verifier API ---
    echo '[HEAD] Starting code verifier API...'
    uvicorn open_instruct.code_utils.api:app \
        --host 0.0.0.0 \
        --port ${CODE_API_PORT} \
        --workers ${CODE_API_WORKERS} \
        > ${LOGDIR}/code-api.out 2> ${LOGDIR}/code-api.err &
    CODE_API_PID=\$!

    # Health check
    for i in \$(seq 1 30); do
        if curl -s --connect-timeout 2 http://localhost:${CODE_API_PORT}/health > /dev/null 2>&1; then
            echo '[HEAD] Code API is healthy'
            break
        fi
        [ \"\$i\" -eq 30 ] && echo '[HEAD] WARNING: Code API health check timed out'
        sleep 2
    done

    # --- Launch GRPO training ---
    echo '[HEAD] Starting GRPO training...'
    set -e

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
        --dataset_mixer_list /data/artifacts/frank/datasets/Dolci-Think-RL-7B-with-messages-hf 1.0 \
        --dataset_mixer_list_splits train \
        --dataset_mixer_eval_list /data/artifacts/frank/datasets/Dolci-Think-RL-7B-with-messages-hf 8 \
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
        --num_learners_per_node 8 \
        --vllm_num_engines 8 \
        --vllm_tensor_parallel_size 1 \
        --lr_scheduler_type constant \
        --apply_verifiable_reward true \
        --seed 1 \
        --local_eval_every 0 \
        --save_freq 25 \
        --checkpoint_state_freq 100 \
        --checkpoint_state_dir /data/artifacts/frank/openinstruct/${RUN_NAME}/checkpoint_states \
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

    TRAIN_EXIT=\$?

    # Cleanup
    kill \${CODE_API_PID} 2>/dev/null || true
    ray stop --force 2>/dev/null || true
    exit \${TRAIN_EXIT}
" > "${LOGDIR}/train.out" 2> "${LOGDIR}/train.err"

HEAD_EXIT=$?
echo "[$(date)] Head node exited with code ${HEAD_EXIT}"

# Cleanup worker
kill ${WORKER_PID} 2>/dev/null || true
wait ${WORKER_PID} 2>/dev/null || true

echo "[$(date)] Done."
exit ${HEAD_EXIT}

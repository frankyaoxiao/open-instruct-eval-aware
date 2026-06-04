#!/bin/bash
#SBATCH --job-name=ea-droptop5-train
#SBATCH --nodes=5
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=120:00:00

# ===========================================================================
# OLMo 3 7B Think RL (GRPO) — 5-node B200, drop-top5 IFEval, 1300 steps
#
# Layout (emergent from placement-group strategies in grpo_fast.py):
#   Node 0 (head): Ray head + 8 trainer GPUs + code API + orchestrator
#   Nodes 1-4:     Ray workers + 8 vLLM engines each (TP=1)  -> 32 engines total
#
# B200 unlocks vs H100 80GB:
#   - DROP --gradient_checkpointing  (was needed on H100 to fit pack_length 35840
#     activations in 80GB; 192GB B200 fits with plenty of margin)
#   - KEEP DeepSpeed Stage 3         (matches AI2's tested OLMo3 config)
#
# Training stops at step 1300 (matches OLMo 3 7B Think HF revisions).
# All paths are env vars exported by run.sh (WORKDIR, OUTPUT_DIR, ...).
# ===========================================================================

set -euo pipefail

: "${OPEN_INSTRUCT_DIR:?run.sh must export OPEN_INSTRUCT_DIR}"
: "${DATASET_DIR:?run.sh must export DATASET_DIR}"
: "${OUTPUT_DIR:?run.sh must export OUTPUT_DIR}"
: "${LOGDIR:?run.sh must export LOGDIR}"

REPO_DIR="${OPEN_INSTRUCT_DIR}"
RUN_NAME="ea-droptop5-train"
RAY_PORT="${RAY_PORT:-8888}"
CODE_API_PORT="${CODE_API_PORT:-8070}"
CODE_API_WORKERS="${CODE_API_WORKERS:-48}"
# 1300 steps × 64 prompts × 8 samples
TOTAL_EPISODES="${TOTAL_EPISODES:-665600}"

# Redirect launcher output
exec > "${LOGDIR}/train-${SLURM_JOB_ID}.out" 2> "${LOGDIR}/train-${SLURM_JOB_ID}.err"

echo "================================================"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Nodes:     ${SLURM_JOB_NODELIST}"
echo "Output:    ${OUTPUT_DIR}"
echo "Dataset:   ${DATASET_DIR}"
echo "Time:      $(date)"
echo "================================================"

# ---------------------------------------------------------------------------
# Parse SLURM nodes and pick a head
# ---------------------------------------------------------------------------
NODES=($(scontrol show hostnames "${SLURM_JOB_NODELIST}"))
HEAD_NODE=${NODES[0]}
HEAD_IP=$(srun --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" hostname -I | awk '{print $1}')

echo "Head node:  ${HEAD_NODE} (${HEAD_IP})"
echo "All nodes:  ${NODES[*]}"

# ---------------------------------------------------------------------------
# 1. Start Ray workers on nodes 1..N-1
# ---------------------------------------------------------------------------
WORKER_PIDS=()
for i in $(seq 1 $((${#NODES[@]} - 1))); do
    NODE=${NODES[$i]}
    echo "[$(date)] Starting Ray worker on ${NODE}"
    srun --overlap --nodes=1 --ntasks=1 -w "${NODE}" bash -c "
        set -uo pipefail
        export PATH=${REPO_DIR}/.venv/bin:\${HOME}/.local/bin:\${PATH}
        export PYTHONPATH=${REPO_DIR}
        export NCCL_CUMEM_ENABLE=0
        export CUDA_DEVICE_MAX_CONNECTIONS=1
        export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
        export VLLM_ALLOW_INSECURE_SERIALIZATION=1
        export RAY_CGRAPH_get_timeout=300
        cd ${REPO_DIR}

        ray stop --force 2>/dev/null || true
        ray start --address=${HEAD_IP}:${RAY_PORT}

        echo 'Ray worker on ${NODE} polling head...'
        while ray status --address=${HEAD_IP}:${RAY_PORT} >/dev/null 2>&1; do
            sleep 10
        done
        echo 'Ray head unreachable, worker exiting.'
    " > "${LOGDIR}/ray-worker-${i}.log" 2>&1 &
    WORKER_PIDS+=($!)
done

# ---------------------------------------------------------------------------
# 2. Head: Ray head + code API + grpo_fast.py
# ---------------------------------------------------------------------------
echo "[$(date)] Launching head node"
srun --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" bash -c "
    set -uo pipefail
    export PATH=${REPO_DIR}/.venv/bin:\${HOME}/.local/bin:\${PATH}
    export PYTHONPATH=${REPO_DIR}
    export NCCL_CUMEM_ENABLE=0
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    export VLLM_ALLOW_INSECURE_SERIALIZATION=1
    export RAY_CGRAPH_get_timeout=300
    cd ${REPO_DIR}

    if [ -f ${REPO_DIR}/.env ]; then
        set -a
        source <(grep -v '^\s*#' ${REPO_DIR}/.env | grep -v '^\s*\$')
        set +a
    fi

    ray stop --force 2>/dev/null || true
    ray start --head --port=${RAY_PORT} --dashboard-host=0.0.0.0
    echo '[HEAD] Ray head started'

    EXPECTED_NODES=$((${#NODES[@]}))
    for i in \$(seq 1 60); do
        WORKER_COUNT=\$(ray status 2>/dev/null | grep -c 'node_' || true)
        if [ \"\${WORKER_COUNT}\" -ge \${EXPECTED_NODES} ]; then
            echo \"[HEAD] Ray cluster ready with \${WORKER_COUNT} nodes\"
            break
        fi
        [ \"\$i\" -eq 60 ] && echo '[HEAD] WARNING: timed out waiting for workers'
        sleep 5
    done

    echo '[HEAD] Starting code verifier API (workers=${CODE_API_WORKERS})...'
    uvicorn open_instruct.code_utils.api:app \
        --host 0.0.0.0 --port ${CODE_API_PORT} --workers ${CODE_API_WORKERS} \
        > ${LOGDIR}/code-api.out 2> ${LOGDIR}/code-api.err &
    CODE_API_PID=\$!

    for i in \$(seq 1 30); do
        if curl -s --connect-timeout 2 http://localhost:${CODE_API_PORT}/health > /dev/null 2>&1; then
            echo '[HEAD] Code API healthy'; break
        fi
        [ \"\$i\" -eq 30 ] && echo '[HEAD] WARNING: code API health-check timed out'
        sleep 2
    done

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
        --output_dir ${OUTPUT_DIR} \
        --kl_estimator 2 \
        --dataset_mixer_list ${DATASET_DIR} 1.0 \
        --dataset_mixer_list_splits train \
        --dataset_mixer_eval_list ${DATASET_DIR} 8 \
        --dataset_mixer_eval_list_splits train \
        --max_prompt_token_length 2048 \
        --response_length 32768 \
        --pack_length 35840 \
        --model_name_or_path allenai/Olmo-3-7B-Think-DPO \
        --chat_template_name olmo_dpo_native \
        --non_stop_penalty False \
        --mask_truncated_completions False \
        --temperature 1.0 \
        --ground_truths_key ground_truth \
        --sft_messages_key messages \
        --total_episodes ${TOTAL_EPISODES} \
        --deepspeed_stage 3 \
        --num_learners_per_node 8 \
        --vllm_num_engines 32 \
        --vllm_tensor_parallel_size 1 \
        --vllm_gpu_memory_utilization 0.9 \
        --lr_scheduler_type constant \
        --apply_verifiable_reward true \
        --seed 42 \
        --local_eval_every 0 \
        --save_freq 50 \
        --checkpoint_state_freq 100 \
        --checkpoint_state_dir ${OUTPUT_DIR}/checkpoint_states \
        --with_tracking \
        --clip_higher 0.272 \
        --code_api_url http://${HEAD_IP}:${CODE_API_PORT}/test_program \
        --code_pass_rate_reward_threshold 0.99 \
        --backend_timeout 1200 \
        --llm_judge_model openai/gpt-5-mini \
        --llm_judge_timeout 600 \
        --llm_judge_max_tokens 2048 \
        --llm_judge_max_context_length 32768 \
        --inflight_updates false \
        --async_steps 1 \
        --save_traces \
        --rollouts_save_path ${OUTPUT_DIR}/rollouts

    TRAIN_EXIT=\$?
    kill \${CODE_API_PID} 2>/dev/null || true
    ray stop --force 2>/dev/null || true
    exit \${TRAIN_EXIT}
" > "${LOGDIR}/train.out" 2> "${LOGDIR}/train.err"

HEAD_EXIT=$?
echo "[$(date)] Head exited with code ${HEAD_EXIT}"

for PID in "${WORKER_PIDS[@]}"; do
    kill ${PID} 2>/dev/null || true
done
wait 2>/dev/null || true

echo "[$(date)] Training job done."
exit ${HEAD_EXIT}

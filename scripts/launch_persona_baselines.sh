#!/bin/bash
# Launch SLURM array job for persona baseline computation.
#
# Usage:
#   bash scripts/launch_persona_baselines.sh [output_dir] [num_shards]
#
# Defaults:
#   output_dir  = /data/artifacts/frank/persona_baselines/olmo3-7b-dpo
#   num_shards  = 16
#
# After all shards complete, aggregate with:
#   uv run python scripts/compute_persona_baselines.py \
#       --aggregate --output_dir $OUTPUT_DIR

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="allenai/Olmo-3-7B-Think-DPO"
DATASET="/data/artifacts/frank/datasets/Dolci-Think-RL-7B-with-messages-hf-complete"
VECTOR="/home/fxiao/eval_awareness/eval_steering/vectors/OLMo3-7B-DPO.pt"

OUTPUT_DIR="${1:-/home/fxiao/eval_awareness/persona_attribution/runs/baselines_dpo}"
NUM_SHARDS="${2:-16}"
MAX_CONCURRENT=8
TIMEOUT="10:00:00"
JOB_NAME="persona-baselines"

LAST_SHARD=$((NUM_SHARDS - 1))

# Create directories
SHARD_DIR="${OUTPUT_DIR}/shards"
LOG_DIR="${OUTPUT_DIR}/slurm_logs"
mkdir -p "$SHARD_DIR" "$LOG_DIR"

echo "Model:       $MODEL"
echo "Dataset:     $DATASET"
echo "Vector:      $VECTOR"
echo "Output:      $OUTPUT_DIR"
echo "Shards:      $NUM_SHARDS"
echo "Max concurrent: $MAX_CONCURRENT"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --array=0-${LAST_SHARD}%${MAX_CONCURRENT}
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=${TIMEOUT}
#SBATCH --output=${LOG_DIR}/%a.out
#SBATCH --error=${LOG_DIR}/%a.err

echo "========================================="
echo "Shard:  \${SLURM_ARRAY_TASK_ID} / ${NUM_SHARDS}"
echo "Node:   \$(hostname)"
echo "GPU:    \${CUDA_VISIBLE_DEVICES:-none}"
echo "Time:   \$(date)"
echo "========================================="

cd /home/fxiao/eval_awareness/open-instruct

uv run python scripts/compute_persona_baselines.py \\
    --model_name_or_path ${MODEL} \\
    --dataset_path ${DATASET} \\
    --persona_vector_path ${VECTOR} \\
    --output_dir ${OUTPUT_DIR} \\
    --shard_id \${SLURM_ARRAY_TASK_ID} \\
    --num_shards ${NUM_SHARDS}

echo "Shard \${SLURM_ARRAY_TASK_ID} done at \$(date)"
EOF

echo "Submitted array job: ${NUM_SHARDS} shards, max ${MAX_CONCURRENT} concurrent"
echo ""
echo "After all shards finish, aggregate with:"
echo "  uv run python scripts/compute_persona_baselines.py --aggregate --output_dir ${OUTPUT_DIR}"

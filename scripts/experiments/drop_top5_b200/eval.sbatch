#!/bin/bash
#SBATCH --job-name=ea-droptop5-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00

# ===========================================================================
# Post-training eval orchestrator. NO GPUs requested — this job just generates
# a fortress config from the training output dir, then runs fortress/run.py,
# which itself sbatches one 1-GPU sub-job per checkpoint and waits on them.
#
# Final output: $EVAL_DIR/summary.csv (one row per step_N, with eval-awareness rate)
# ===========================================================================

set -euo pipefail

: "${OPEN_INSTRUCT_DIR:?run.sh must export OPEN_INSTRUCT_DIR}"
: "${FORTRESS_DIR:?run.sh must export FORTRESS_DIR}"
: "${OUTPUT_DIR:?run.sh must export OUTPUT_DIR}"
: "${EVAL_DIR:?run.sh must export EVAL_DIR}"
: "${LOGDIR:?run.sh must export LOGDIR}"

HERE="$(dirname "$(scontrol show job ${SLURM_JOB_ID} 2>/dev/null | grep -oP 'Command=\K\S+' || echo .)")"
PARTITION="${PARTITION:-compute}"
EVAL_CONFIG="${EVAL_DIR}/config.yaml"

# Redirect launcher output
exec > "${LOGDIR}/eval-${SLURM_JOB_ID}.out" 2> "${LOGDIR}/eval-${SLURM_JOB_ID}.err"

echo "================================================"
echo "EVAL orchestrator job: ${SLURM_JOB_ID}"
echo "Time:        $(date)"
echo "Output dir:  ${OUTPUT_DIR}"
echo "Eval dir:    ${EVAL_DIR}"
echo "Fortress:    ${FORTRESS_DIR}"
echo "================================================"

# 1) Generate the fortress config from whatever checkpoints exist
echo "[$(date)] Generating eval config..."
cd "${OPEN_INSTRUCT_DIR}"
uv run python scripts/experiments/drop_top5_b200/build_eval_config.py \
    "${OUTPUT_DIR}" \
    "${EVAL_CONFIG}" \
    --partition "${PARTITION}" \
    --exp-prefix "DT5"

# 2) Run fortress orchestrator. It will sbatch one job per checkpoint, each
#    submitting GPU resources via the partition in EVAL_CONFIG. run.py blocks
#    (`sbatch --wait` per slot) until all are done.
echo "[$(date)] Launching fortress eval orchestrator..."
cd "${FORTRESS_DIR}"
uv run python run.py --config "${EVAL_CONFIG}" --log-dir "${EVAL_DIR}/fortress-run"

# 3) Copy the summary CSV up to the EVAL_DIR root for easy access
if [[ -f "${EVAL_DIR}/fortress-run/summary.csv" ]]; then
    cp "${EVAL_DIR}/fortress-run/summary.csv" "${EVAL_DIR}/summary.csv"
    echo "[$(date)] Eval complete. Summary at: ${EVAL_DIR}/summary.csv"
    head "${EVAL_DIR}/summary.csv"
else
    echo "[$(date)] WARNING: fortress did not produce summary.csv; check ${EVAL_DIR}/fortress-run/"
fi

echo "[$(date)] Eval orchestrator done."

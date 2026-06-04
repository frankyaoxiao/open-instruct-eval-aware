#!/usr/bin/env bash
# Submits training + eval as two SLURM jobs with a dependency.
# Training runs on 5 nodes × 8 B200 for up to 1300 steps.
# Eval (1 CPU node orchestrator) starts when training exits (any code) and runs
# fortress eval-awareness scoring on every step_50/100/.../1300 checkpoint.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPEN_INSTRUCT_DIR="$(cd "${HERE}/../../.." && pwd)"
FORTRESS_DIR="${FORTRESS_DIR:-$(cd "${OPEN_INSTRUCT_DIR}/../fortress" && pwd)}"

# --- User-tweakable paths and SLURM knobs -----------------------------------
WORKDIR="${WORKDIR:-${HOME}/ea-drop-top5}"
DATASET_DIR="${DATASET_DIR:-${WORKDIR}/dataset/drop-top5}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKDIR}/training}"
EVAL_DIR="${EVAL_DIR:-${WORKDIR}/eval}"
LOGDIR="${LOGDIR:-${WORKDIR}/logs}"
PARTITION="${PARTITION:-compute}"
# ----------------------------------------------------------------------------

export OPEN_INSTRUCT_DIR FORTRESS_DIR WORKDIR DATASET_DIR OUTPUT_DIR EVAL_DIR LOGDIR PARTITION

mkdir -p "${WORKDIR}" "${OUTPUT_DIR}" "${EVAL_DIR}" "${LOGDIR}"

# Sanity checks
if [[ ! -f "${DATASET_DIR}/data/train-00000-of-00001.parquet" ]]; then
    echo "ERROR: dataset not found at ${DATASET_DIR}." >&2
    echo "Run: uv run python build_dataset.py" >&2
    exit 1
fi
if [[ ! -f "${OPEN_INSTRUCT_DIR}/.env" ]]; then
    echo "ERROR: ${OPEN_INSTRUCT_DIR}/.env missing — needed by training (LLM judge)." >&2
    exit 1
fi
if [[ ! -f "${FORTRESS_DIR}/.env" ]]; then
    echo "ERROR: ${FORTRESS_DIR}/.env missing — needed by eval (FORTRESS scorer)." >&2
    exit 1
fi
if [[ ! -d "${FORTRESS_DIR}/data/harmbench_strongreject" ]]; then
    echo "ERROR: FORTRESS prompts dir not found at ${FORTRESS_DIR}/data/harmbench_strongreject." >&2
    echo "Make sure the fortress repo is cloned and the prompts.jsonl is present." >&2
    exit 1
fi

echo "open-instruct: ${OPEN_INSTRUCT_DIR}"
echo "fortress:      ${FORTRESS_DIR}"
echo "workdir:       ${WORKDIR}"
echo "dataset:       ${DATASET_DIR}"
echo "output:        ${OUTPUT_DIR}"
echo "eval:          ${EVAL_DIR}"
echo "partition:     ${PARTITION}"
echo

# Submit training
TRAIN_JID=$(sbatch --parsable --partition="${PARTITION}" "${HERE}/train.sbatch")
echo "TRAIN job submitted: ${TRAIN_JID}"
echo "  monitor: tail -f ${LOGDIR}/train-${TRAIN_JID}.out"

# Submit eval with dependency on training (afterany = runs even if training was killed at time-limit;
# we still want to eval whatever checkpoints exist).
EVAL_JID=$(sbatch --parsable --partition="${PARTITION}" --dependency=afterany:"${TRAIN_JID}" "${HERE}/eval.sbatch")
echo "EVAL  job submitted: ${EVAL_JID}  (depends on ${TRAIN_JID})"
echo "  results will land at: ${EVAL_DIR}/summary.csv"

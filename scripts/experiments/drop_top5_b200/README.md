# drop-top5 IFEval RL on 5× B200

Trains OLMo 3 7B Think with GRPO on the IFEval split of `allenai/Dolci-Think-RL-7B`,
with the 5 highest eval-awareness-eliciting prompt sources removed
(wildjailbreak, wildguard, coconot, sciriff, table_gpt).
Then evaluates every 50th checkpoint up to 1300 on a held-out FORTRESS probe
for verbalized eval-awareness.

## What ships

| File | Purpose |
|---|---|
| `run.sh` | Submits training + eval as two SLURM jobs with a dependency |
| `train.sbatch` | 5-node B200 GRPO training, capped at 1300 steps |
| `eval.sbatch` | Post-training orchestrator: discovers ckpts → runs fortress eval |
| `build_dataset.py` | One-time: builds the drop-top5 dataset locally from HF Hub |
| `build_eval_config.py` | Helper: generates the fortress YAML config from the ckpt dir |
| `.env.example` | API key/token template |
| `CLAUDE.md` | Full context for your Claude / future you |

## Setup

```bash
# 1. Clone both repos (next to each other)
git clone <open-instruct-fork> open-instruct
git clone <fortress-fork>      fortress
cd open-instruct/scripts/experiments/drop_top5_b200

# 2. Set up envs
cp .env.example .env       # then fill in OPENAI_API_KEY and HF_TOKEN
cp .env ../../../.env      # open-instruct's training loop reads this
cp .env ../../../../fortress/.env  # fortress's eval reads this

# 3. uv sync in both repos
( cd ../../../ && uv sync )
( cd ../../../../fortress && uv sync )

# 4. Build the dataset (~3 min)
uv run python build_dataset.py
```

## Run

```bash
./run.sh
```

Submits two SLURM jobs:
- **Training** (5 nodes × 8 B200, ~1.5 days to step 1300)
- **Eval** (1 CPU node orchestrator + ~26 sub-jobs of 1 GPU each, ~2-4 hours)

Output ends up at `$WORKDIR/eval/summary.csv` (default `$HOME/ea-drop-top5/eval/summary.csv`):
one row per step_N checkpoint, with eval-awareness rate.

## If something breaks

Read `CLAUDE.md` — it has the full architecture, every code change we made, every
deadlock we hit and fixed, and what to look for in train.err / eval.err.

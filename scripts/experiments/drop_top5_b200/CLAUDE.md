# drop-top5 IFEval RL — context for Claude

You're the Claude helping someone on a B200 cluster get this experiment running.
Read this whole file before making changes. It has the full design, every code
change made during the original session, every failure mode we hit, and what to
look for if something breaks.

## What this experiment is

We're studying whether RL training on the IFEval split of `allenai/Dolci-Think-RL-7B`
amplifies *verbalized eval-awareness* in the model (the model writing things
like "this is probably a test" in its chain of thought). Prior empirical analysis
on the same training run identified five prompt sub-sources that elicited
eval-awareness at 2-3× the mean rate:

| Source                | Rate  | Category                                |
|-----------------------|-------|------------------------------------------|
| `sciriff`             | 18.9% | scientific NLP benchmark suite           |
| `table_gpt`           | 17.2% | Microsoft table-QA tasks                 |
| `wildjailbreak`       | 15.4% | AI2 synthetic adversarial safety probes  |
| `coconot`             | 14.8% | AI2 contextual-noncompliance probes      |
| `wildguard`           | 14.0% | AI2 safety moderation training prompts   |

These are 13.8% of the dataset (4,109 of 29,813 IFEval prompts). The
hypothesis: train without them and see whether eval-awareness still rises on a
held-out probe (FORTRESS / harmbench_strongreject).

- If it *still rises*: eval-awareness is a general RL dynamic, not taught by the
  high-elicitor data. Negative-but-clean result.
- If it *flattens*: those sources were causally driving it, and curating data is
  a viable mitigation.

## What the run actually does

```
run.sh
  ├─→ sbatch train.sbatch            (5 nodes × 8 B200, 1300 steps)
  └─→ sbatch eval.sbatch             (after train, any exit code)
        └─ cd fortress && uv run python run.py --config <auto-gen>.yaml
             └─ for each step_50, step_100, ..., step_1300 checkpoint:
                  sbatch a 1-GPU job that runs worker.py
                    └─ load model, generate 20× completions per FORTRESS prompt,
                       score each completion with gpt-5-mini ("aware: yes/no")
             └─ aggregate → summary.csv: model, total, aware, failed, awareness_rate
```

## Repo layout

The experiment expects **two repos checked out side-by-side**:

```
<some_parent>/
├── open-instruct/      # this fork, with drop_top5_b200/ in scripts/experiments/
└── fortress/           # the eval-awareness scoring orchestrator
```

`run.sh` auto-detects `OPEN_INSTRUCT_DIR` from its own location and tries
`../fortress` next to it. Override with `FORTRESS_DIR=...` if it's elsewhere.

### Fortress prompts file

`data/harmbench_strongreject/prompts.jsonl` (513 HarmBench + StrongReject
prompts) ships in fortress's git, force-added past the `data/` gitignore.
A fresh `git clone` of fortress already has it — no action needed.
`run.sh` sanity-checks for it at submit time.

## All files in this dir

| File | Notes |
|---|---|
| `README.md` | 1-page user-facing how-to |
| `CLAUDE.md` | this file |
| `.env.example` | API keys / HF token template |
| `build_dataset.py` | one-time data prep from `allenai/Dolci-Think-RL-7B` |
| `build_eval_config.py` | scans `OUTPUT_DIR` → generates fortress YAML |
| `run.sh` | submits train + eval with `--dependency=afterany` |
| `train.sbatch` | the 5-node B200 training job |
| `eval.sbatch` | the eval orchestrator (CPU-only, calls fortress) |

## Code changes the original session made to open-instruct

These all live in this fork already, but you should know they exist so you don't
revert them by accident if syncing upstream:

| File | Change | Why |
|---|---|---|
| `open_instruct/grpo_fast.py` | `WEIGHT_SYNC_TIMEOUT_S = 1200` (was 120) | inflight=false drains long IFEval rollouts; 120s timed out |
| `open_instruct/grpo_fast.py` | `main()`'s checkpoint-state load reads `latest` marker | original code read a never-written `global_0/state.pt` |
| `open_instruct/data_loader.py` | `DataPreparationActor.__init__` sets `_last_consumed_step = training_step - 1` on resume | otherwise the throttle deadlocks on `step - (-1) > async_steps` |
| `open_instruct/rl_utils.py` | `RolloutRecord` adds `dataset_idx` field | so filter logs join with rollout traces |
| `open_instruct/persona_filter.py` | adds `drop_group` config option | persona drop-group mode |
| `open_instruct/eval_awareness_filter.py` | NEW class: LLM-judge rollout filter | drops rollouts the GPT-5-mini scorer flags during training |
| `open_instruct/grpo_utils.py` | new CLI args: `--persona_drop_group`, `--eval_awareness_*` | wires the above |
| `open_instruct/dataset_transformation.py` | new chat templates: `olmo_thinker_identity`, `olmo_thinker_identity_full` | identity-prompt experiments |

None of these are used by this specific drop-top5 run (no filters enabled), but
they're in the fork so the build is reproducible.

## The training architecture — quick mental model

You're going to look at logs and want to know what's happening. Here's the
shape:

- **SLURM**: 5 nodes × 8 B200 = 40 GPUs total. One srun for the head, one srun
  per worker node, both `--overlap`. The poll-loop is on purpose — `ray start`
  daemonizes so srun has to stay alive to keep SLURM from tearing down the
  worker.
- **Ray**: head joins workers into one cluster with 40 GPUs. Two **placement
  groups** carve up the GPUs:
  - **Trainer PG**: one 8-GPU bundle with `STRICT_SPREAD` → fits on one node
    (node 0). 8 `PolicyTrainerRayProcess` actors land here, running DeepSpeed
    Stage 3 on a single 7B model sharded across them.
  - **vLLM PG**: 32 single-GPU bundles with `PACK` → since node 0 is exhausted
    by the trainer PG, all 32 vLLM bundles pack onto the remaining 4 nodes (8
    engines/node). Each engine holds a **replica** of the full 7B, not a shard.
- **Weight sync** uses a *second* NCCL process group: trainer rank-0 ↔ all 32
  vLLM engines. After each training step, rank-0 calls
  `torch.distributed.broadcast` per parameter; each engine receives into a
  scratch buffer and `model.load_weights` into its running model.
- **Data flow** is Ray queues: `prompt_Q` (head → engines), `inference_results_Q`
  (engines → DataPreparationActor on head). The DataPreparationActor packs the
  results into sequences and hands them to trainers via a StreamingDataLoader.

For full detail with code citations, see open-instruct's `open_instruct/grpo_fast.py`.
You can also ask me ("Claude, walk me through one weight sync"); the architecture
was traced in detail during the original session.

## Why we're dropping `--gradient_checkpointing` on B200

On H100 80GB this flag is required to fit pack_length 35840 activations
alongside the trainer's working memory (params + opt + activations + buffers ≈
70-100 GB depending on attention impl). On B200 192GB the activations fit with
plenty of margin, so we drop the flag and save the ~30-40% trainer-step time
that activation recompute was costing.

We keep `--deepspeed_stage 3` because:
- It matches AI2's tested OLMo 3 config (all 14 of AI2's OLMo 3 scripts use it).
- Stage 2 also fits on B200 but the code requires `sequence_parallel_size > 1
  → stage == 3`, so dropping to Stage 2 closes an optimization door for free.
- Stage 3's all-gather overhead at 8-way intra-node NVLink (1.8 TB/s on B200) is
  small.

## Expected timing on the B200 cluster

| Phase | Expected time | Notes |
|---|---|---|
| Build dataset | ~3 min | one-time, mostly HF download |
| Per training step | ~100-115 s | vs ~600-900 s on the original 2× H100 setup |
| Training to step 1300 | ~1.5 days | wall time, ~52 weight-syncs at every save_freq=50 |
| Per eval checkpoint | ~5-15 min | generation + GPT-5-mini scoring |
| Eval to all 26 checkpoints | ~2-4 hours | depends on max_concurrent in fortress config |

## Likely things to fix / verify when bringing this up on a new cluster

1. **Cluster paths**: `run.sh` defaults to `$HOME/ea-drop-top5` for everything.
   If your cluster has fast scratch storage elsewhere, override `WORKDIR`.

2. **SLURM partition name**: defaults to `compute`. Set `PARTITION=...` if your
   cluster uses something else. Both train.sbatch and the fortress sub-jobs
   (via build_eval_config.py's output) use it.

3. **CPUs per node on B200**: usually 128. `train.sbatch` requests
   `--cpus-per-task=128`. The grpo_fast code reads `ray.nodes()` and caps the
   bundle CPU request at the smallest live node's count — so if your trainer
   node has 128 CPUs but a worker has only 64, the bundle is capped at 64.

4. **`CODE_API_WORKERS`**: defaults to 48 (3× the H100 reference). With 32 vLLM
   engines on B200, code verification load is ~4× higher than on the 2-node
   H100 baseline. If you see code-api timeouts or 503s in `code-api.err`, bump
   higher.

5. **vLLM compatibility with B200**: vLLM 0.14.1 (in `uv.lock`) should support
   B200's Blackwell architecture, but verify the engines actually come up. If
   not, upgrade vLLM in `pyproject.toml` and re-`uv sync`.

6. **CUDA / NCCL versions**: B200 needs CUDA 12.4+ for compute capability 10.0.
   If you see ECC or `cudaError_t` complaints right at startup, the driver is
   likely too old.

7. **HuggingFace token**: needed to download `allenai/Olmo-3-7B-Think-DPO`. Set
   in `.env` and the train script will source it. `build_dataset.py` reads
   `HF_TOKEN` from env too.

8. **HuggingFace cache location**: default is `~/.cache/huggingface`. The 7B
   model + Dolci-Think-RL-7B dataset download is ~14 GB. On clusters with small
   home quotas this fills up — set `HF_HOME=<scratch>/hf_cache` in `.env` and
   the train script will source it. The reference training scripts use
   `HF_HOME=/data/artifacts/frank/hf_cache`.

9. **Weights & Biases**: `train.sbatch` passes `--with_tracking`, which calls
   `wandb.init()` at startup. Three ways this can go:
   - With `WANDB_API_KEY` in `.env`: logs to wandb normally.
   - With `WANDB_MODE=disabled` (or `=offline`) in `.env`: silently no-op.
   - With neither set: `wandb.init()` may prompt for a key on stdin and the
     SLURM job will hang. **Pick one of the first two before submitting.**

10. **OPENAI cost**: the GRPO LLM-judge reward uses `gpt-5-mini` for non-IFEval
    rows. Over 1300 steps × 64 prompts × 8 samples ≈ 665K rollouts, only a
    fraction of which hit the judge (only when the deterministic verifier
    doesn't apply), but expect several dollars of API spend during training.
    Fortress eval is another ~270K judge calls (513 prompts × 20 completions
    × 26 checkpoints) — budget another ~$20-50 there.

8. **fortress `.env`**: must contain `OPENAI_API_KEY`. fortress's `run.py` calls
   `load_dotenv()` from its own cwd, so the `.env` has to live at the fortress
   repo root, not in the experiment dir.

9. **`--time` headroom**: train.sbatch requests 120h (5 days) — way more than
   the ~36h projection. We over-allocate on purpose because B200 wall-clock
   predictions are unverified and being killed at the limit loses
   checkpoints saved after the last `checkpoint_state_freq`.

## What to look at if training hangs

In order of likelihood:

1. **`logs/train.err`** for `Weight sync timed out` — should be rare with the
   1200s budget, but possible if engines hit a 32k-token long-tail response.
   If so: either bump `WEIGHT_SYNC_TIMEOUT_S` in `grpo_fast.py` further, or
   reduce `vllm_num_engines` so fewer in-flight requests per sync.

2. **`logs/ray-worker-N.log`** — if Ray worker failed to join the head, the
   head's `Waiting for Ray workers` loop will warn after 60 retries.

3. **`logs/code-api.err`** — if the code API isn't healthy, training hangs at
   the very first batch because rewards can't be computed. Bump
   `CODE_API_WORKERS` and re-submit.

4. **NCCL hangs across nodes** — usually a network or driver issue, not a code
   issue. Run a small NCCL allreduce test on the cluster to verify the fabric.

5. **GPU ECC errors** (`cudaErrorECCUncorrectable` in train.err) — exclude the
   node with `--exclude=<bad-node>` and resubmit. Confirm with `nvidia-smi
   --query-gpu=index,ecc.errors.uncorrected.aggregate.total --format=csv` on
   the suspect node — any non-zero is a faulty GPU.

## What to look at if eval hangs

1. **fortress sub-jobs**: each step's eval is its own SLURM job. `squeue -u
   <you>` will show them. If they're stuck in QUEUED beyond their fair share,
   check partition limits.

2. **`$EVAL_DIR/fortress-run/slurm_logs/*.err`** — per-checkpoint logs. The
   most common failure is OOM at vLLM startup if a B200 has less memory than
   the config expects, or rate-limited OpenAI calls hanging the scorer.

3. **`$EVAL_DIR/fortress-run/scores/<short_name>.jsonl`** — if this exists but
   is short (< 20 × n_prompts lines), the eval failed partway. fortress will
   re-run it if you resubmit the eval job; otherwise delete the partial file
   and rerun.

## Resuming a partially-done eval

The fortress `run.py` has a "skip if scores file is complete" check (see
`run.py` near line 100). To re-eval a specific checkpoint:

```bash
rm $EVAL_DIR/fortress-run/scores/7B-DT5-stepNNNN.jsonl
sbatch eval.sbatch   # or just re-launch run.py manually
```

## Reading the final results

`$EVAL_DIR/summary.csv` has one row per checkpoint:

```
model,total,aware,failed,awareness_rate
7B-DT5-step0050,10260,2091,0,0.2038
7B-DT5-step0100,10260,1755,0,0.1711
...
```

Plot `awareness_rate` vs step. If the trajectory **rises monotonically** like the
baseline (~20% → ~70%), the drop-top5 intervention failed and eval-awareness is
a general dynamic. If it **flatlines or rises much slower**, those five sources
were causally driving it.

## Reproducibility knobs

- `--seed 42` is fixed in `train.sbatch`.
- `allenai/Olmo-3-7B-Think-DPO` is the model. We don't pin a revision; if you
  need bit-exact, add `--model_revision <sha>`.
- The dataset is rebuilt from the latest `allenai/Dolci-Think-RL-7B` each time
  `build_dataset.py` runs. If the parent gets updated upstream, the resulting
  drop-top5 row count will shift slightly. A sanity print in build_dataset.py
  warns if the drop fraction is outside ~13.8% ± few pp.

## Asking for help

If you get stuck, the original session that built this had a lot more context
than what's in this file. Try this prompt:

> "I'm trying to run the drop_top5_b200 experiment in open-instruct. CLAUDE.md
> says the architecture is X / fix is Y. I'm seeing [error]. Walk me through
> debugging it given that context."

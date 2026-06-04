#!/usr/bin/env python
"""Generate a fortress eval-awareness YAML config from the training output dir.

Usage:
    python build_eval_config.py <training_output_dir> <out_yaml_path> [--partition <name>]

Scans <training_output_dir>/*_checkpoints/step_N for N % 50 == 0 and N <= 1300,
writes a fortress run.py config that scores each on the FORTRESS held-out probe.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# fortress YAML doesn't accept arbitrary keys, so keep this shape minimal.
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "max_tokens": 3072, "n": 20}
SCORING = {
    "model": "gpt-5-mini",
    "reasoning_effort": "medium",
    "safety_identifier": "SPAR_FX",
    "api_key_env": "OPENAI_API_KEY",
    "max_completion_tokens": 2048,
    "max_concurrent": 400,
}
VLLM = {"tensor_parallel_size": 1}
DEFAULT_SLURM = {
    "max_concurrent": 4,
    "timeout": "3:00:00",
    "job_name": "ea-dt5-eval",
    "gpus_per_node": 8,
    "cpus_per_node": 64,
}
PROMPTS_REL = "data/harmbench_strongreject/prompts.jsonl"
MAX_STEP = 1300


def find_checkpoints(output_dir: Path) -> list[tuple[int, Path]]:
    """Find step_N dirs where N % 50 == 0 and N <= 1300, sorted by step."""
    found: dict[int, Path] = {}
    for ckpt_root in output_dir.glob("*_checkpoints"):
        for step_dir in ckpt_root.glob("step_*"):
            m = re.match(r"step_(\d+)$", step_dir.name)
            if not m:
                continue
            n = int(m.group(1))
            if n % 50 != 0 or n > MAX_STEP:
                continue
            # If multiple resume rounds wrote the same step, prefer the latest mtime.
            if n not in found or step_dir.stat().st_mtime > found[n].stat().st_mtime:
                found[n] = step_dir
    return sorted(found.items())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("out_yaml", type=Path)
    parser.add_argument("--partition", default="compute")
    parser.add_argument("--exp-prefix", default="DT5")
    args = parser.parse_args()

    ckpts = find_checkpoints(args.output_dir.resolve())
    if not ckpts:
        print(f"ERROR: no step_50/.../1300 checkpoints found under {args.output_dir}", file=sys.stderr)
        sys.exit(1)

    # Fortress only knows about its own repo, so prompts path is relative to it.
    cfg_lines: list[str] = ["models:"]
    for n, path in ckpts:
        short = f"7B-{args.exp_prefix}-step{n:04d}"
        cfg_lines.append(f'  - id: "{path}"')
        cfg_lines.append(f'    short_name: "{short}"')
    cfg_lines += [
        "",
        "sampling:",
        *[f"  {k}: {v!r}" if isinstance(v, str) else f"  {k}: {v}" for k, v in SAMPLING.items()],
        "",
        "scoring:",
        *[f"  {k}: {v!r}" if isinstance(v, str) else f"  {k}: {v}" for k, v in SCORING.items()],
        "",
        "vllm:",
        *[f"  {k}: {v}" for k, v in VLLM.items()],
        "",
        "slurm:",
        f'  partition: "{args.partition}"',
        *[f"  {k}: {v!r}" if isinstance(v, str) else f"  {k}: {v}" for k, v in DEFAULT_SLURM.items()],
        "",
        "paths:",
        f'  prompts: "{PROMPTS_REL}"',
        "",
    ]

    args.out_yaml.parent.mkdir(parents=True, exist_ok=True)
    args.out_yaml.write_text("\n".join(cfg_lines))
    print(f"Wrote config with {len(ckpts)} checkpoints to {args.out_yaml}")
    print(f"  step range: {ckpts[0][0]}..{ckpts[-1][0]}")


if __name__ == "__main__":
    main()

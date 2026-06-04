#!/usr/bin/env python
"""Build the drop-top5 IFEval dataset locally from `allenai/Dolci-Think-RL-7B`.

Pipeline (all done in-memory, then parquet-written):
  1. Load `allenai/Dolci-Think-RL-7B` train split from HF Hub (~100k rows).
  2. Filter to rows whose `dataset_source` is the IFEval upstream
     (`hamishivi/IF_multi_constraints_upto5_filtered_dpo_0625_filter`, ~29.8k rows).
  3. Ensure each row has a `messages` column (synthesize from `prompt` if missing).
  4. Drop the 5 highest eval-awareness-eliciting sub-sources, identified by `key`:
        - wildjailbreak  (ai2-adapt-dev/.*wildjailbreak.*)
        - wildguard      (ai2-adapt-dev/.*wildguard.*)
        - coconot        (bare 7-char alphanumeric `key`)
        - sciriff        (`key` starts with `science.`)
        - table_gpt      (ai2-adapt-dev/.*table_gpt.*)
  5. Save as parquet at $DATASET_DIR/data/train-00000-of-00001.parquet
     (default $DATASET_DIR = $HOME/ea-drop-top5/dataset/drop-top5).

Final size: ~25.7k rows. Takes ~3 minutes including the Hub download.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from datasets import load_dataset

PARENT_DS = "allenai/Dolci-Think-RL-7B"
IFEVAL_SOURCE = "hamishivi/IF_multi_constraints_upto5_filtered_dpo_0625_filter"

DEFAULT_OUT = Path(os.environ.get("HOME", ".")) / "ea-drop-top5" / "dataset" / "drop-top5"
OUT_DIR = Path(os.environ.get("DATASET_DIR", DEFAULT_OUT))


def is_drop_top5(key: str | None) -> bool:
    """True if this row's source is one of the 5 high-elicitor sub-sources."""
    if not isinstance(key, str):
        return False
    # coconot: bare 7-char alphanumeric (no separators), e.g. "b0z1tax"
    if (
        "/" not in key
        and "_" not in key
        and "." not in key
        and len(key) == 7
        and key.isalnum()
    ):
        return True
    # sciriff: `key` starts with "science."
    if key.startswith("science."):
        return True
    # wildjailbreak / wildguard / table_gpt: in the ai2-adapt-dev/<source>_<num> form
    m = re.match(r"ai2-adapt-dev/(.+?)_\d+$", key)
    if m:
        s = m.group(1)
        for tag in ("wildjailbreak", "wildguard", "table_gpt"):
            if tag in s:
                return True
    return False


def ensure_messages(row: dict) -> dict:
    """Make sure `messages` is a list of role/content dicts.

    `Dolci-Think-RL-7B`'s `prompt` field is the post-`rlvr_tokenize_v3`
    formatting "<role>: <content>" — see dataset_transformation.py's
    `RAW_PROMPT_KEY = "\\n".join(f"{msg['role']}: {msg['content']}" ...)`.
    The "user: " is logging output, not message content; strip it before
    handing to the next round of `apply_chat_template`.
    """
    msgs = row.get("messages")
    if isinstance(msgs, list) and msgs and isinstance(msgs[0], dict) and "role" in msgs[0]:
        return row
    prompt = row.get("prompt")
    if isinstance(prompt, str):
        content = prompt[len("user: "):] if prompt.startswith("user: ") else prompt
        row["messages"] = [{"role": "user", "content": content}]
    return row


def main() -> None:
    print(f"[1/5] Loading {PARENT_DS} from Hub...")
    ds = load_dataset(PARENT_DS, split="train")
    print(f"      {len(ds):,} rows, columns: {ds.column_names}")

    for required in ("dataset_source", "key"):
        if required not in ds.column_names:
            print(f"ERROR: parent dataset is missing required column {required!r}.", file=sys.stderr)
            sys.exit(1)

    print(f"[2/5] Filtering to IFEval source ({IFEVAL_SOURCE!r})...")
    ds = ds.filter(lambda x: x["dataset_source"] == IFEVAL_SOURCE)
    print(f"      {len(ds):,} rows")

    if "messages" not in ds.column_names:
        print("[3/5] Synthesizing `messages` column from `prompt`...")
        ds = ds.map(ensure_messages)
    else:
        print("[3/5] `messages` column already present, ensuring schema...")
        ds = ds.map(ensure_messages)

    print("[4/5] Dropping the top-5 high-elicitor sources...")
    before = len(ds)
    ds = ds.filter(lambda x: not is_drop_top5(x.get("key")))
    dropped = before - len(ds)
    pct = (dropped / before * 100) if before else 0
    print(f"      dropped {dropped:,} rows ({pct:.1f}%), kept {len(ds):,}")
    if not (0.10 < dropped / before < 0.18):
        print(
            f"WARNING: drop fraction {pct:.1f}% is outside the expected ~13.8% — "
            "the parent dataset's source-key format may have changed."
        )

    out_dir = OUT_DIR / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train-00000-of-00001.parquet"
    print(f"[5/5] Saving to {out_path} ...")
    ds.to_parquet(str(out_path))
    print(f"\nDone. Point --dataset_mixer_list at:\n  {OUT_DIR}")


if __name__ == "__main__":
    main()

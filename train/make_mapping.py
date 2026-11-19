#!/usr/bin/env python
"""Write the episode -> source mapping that train_weighted.py needs for sim/real co-training.

    # a merged dataset whose first 50 episodes are real and the next 350 sim:
    python train/make_mapping.py --root data/g1_merged --ranges real:0-49 sim:50-399 -o mappings/g1_merged.json

Every episode of the dataset must be covered exactly once. Episode lengths are copied from the dataset's
meta, so train_weighted.py can check that the mapping still matches the dataset it is used with.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def episode_lengths(root: Path) -> list[int]:
    lengths: dict[int, int] = {}
    for p in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        import pyarrow.parquet as pq
        t = pq.read_table(p, columns=["episode_index", "length"]).to_pydict()
        lengths.update(zip(t["episode_index"], t["length"]))
    if not lengths:
        jl = root / "meta" / "episodes.jsonl"            # LeRobot dataset v2.x
        for line in jl.read_text().splitlines():
            e = json.loads(line)
            lengths[e["episode_index"]] = e["length"]
    return [lengths[i] for i in range(len(lengths))]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--ranges", nargs="+", required=True, help="source:first-last (inclusive), e.g. real:0-49")
    p.add_argument("-o", "--out", required=True, type=Path)
    a = p.parse_args(argv)
    lengths = episode_lengths(a.root)
    src: dict[int, str] = {}
    for r in a.ranges:
        name, _, span = r.partition(":")
        lo, _, hi = span.partition("-")
        for i in range(int(lo), int(hi or lo) + 1):
            if i in src:
                raise SystemExit(f"episode {i} is in two ranges")
            src[i] = name
    missing = [i for i in range(len(lengths)) if i not in src]
    extra = [i for i in src if i >= len(lengths)]
    if missing or extra:
        raise SystemExit(f"ranges must cover episodes 0..{len(lengths) - 1} exactly: missing {missing[:5]}, "
                         f"beyond the dataset {extra[:5]}")
    slots = [{"slot": i, "source": src[i], "frames": lengths[i]} for i in range(len(lengths))]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"dataset_root": str(a.root), "slots": slots}, indent=1))
    n_sim = sum(s["source"] != "real" for s in slots)
    print(f"{a.out}: {len(slots)} episodes, {len(slots) - n_sim} real, {n_sim} sim")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Check that a LeRobot dataset of G1 demos matches what the agent will send and execute.

    python train/check_dataset.py --root data/g1_task [--waist] [--no-hands] [--camera head]

Checks meta/info.json: observation.state and action exist, have the agent's joint count, and (if the dataset
stores joint names) the agent's joint order; the cameras the agent will provide exist as observation.images.*;
the fps is recorded. A dataset that fails here trains a model the agent will refuse at start-up.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1agent.robot.g1_layout import joint_names  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--waist", action="store_true")
    p.add_argument("--no-hands", action="store_true")
    p.add_argument("--camera", action="append", default=[], help="camera names the robot will provide")
    a = p.parse_args(argv)
    info = json.loads((a.root / "meta" / "info.json").read_text())
    feats = info.get("features", {})
    names = joint_names(a.waist, not a.no_hands)
    problems, notes = [], []
    for key in ("observation.state", "action"):
        f = feats.get(key)
        if f is None:
            problems.append(f"no '{key}' feature")
            continue
        n = f["shape"][0]
        if n != len(names):
            problems.append(f"{key} is {n}-D, the agent's layout is {len(names)}-D "
                            f"({'with' if a.waist else 'no'} waist, {'no' if a.no_hands else 'with'} hands)")
        fn = f.get("names")
        if isinstance(fn, dict):
            fn = next(iter(fn.values()), None)
        if isinstance(fn, list) and len(fn) == len(names) and fn != names:
            bad = [(i, x, y) for i, (x, y) in enumerate(zip(fn, names)) if x != y][:3]
            problems.append(f"{key} joint order differs from the agent's, e.g. {bad}")
        elif not isinstance(fn, list):
            notes.append(f"{key} has no joint names: order cannot be checked, make sure it is g1_layout order")
    cams = sorted(k[len("observation.images."):] for k in feats if k.startswith("observation.images."))
    for c in a.camera:
        if c not in cams:
            problems.append(f"camera '{c}' not in the dataset (has {cams}); map it with --map-camera at run time")
    if not info.get("fps"):
        problems.append("no fps in meta/info.json (the agent's control rate comes from it)")
    print(f"dataset {a.root}: {info.get('total_episodes')} episodes, {info.get('total_frames')} frames, "
          f"{info.get('fps')} fps, cameras {cams}")
    for n in notes:
        print("note:", n)
    for pr in problems:
        print("PROBLEM:", pr)
    print("OK" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

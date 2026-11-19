import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import check_dataset  # noqa: E402
import train  # noqa: E402

from g1agent.robot.g1_layout import joint_names


def test_weighted_frame_weights_give_alpha():
    torch = pytest.importorskip("torch")
    import train_weighted as tw
    fr = np.array([0, 100, 250, 400]); to = np.array([100, 250, 400, 900])
    is_sim = np.array([False, False, True, True])
    w, sim, info = tw.frame_weights(fr, to, is_sim, 0.9, [])
    assert abs(w[sim].sum() - 0.9) < 1e-9 and abs(w[~sim].sum() - 0.1) < 1e-9
    w0, _, _ = tw.frame_weights(fr, to, is_sim, 0.9, [1])
    assert (w0[100:250] == 0).all()


def test_train_presets_build_commands(capsys):
    assert train.main(["--policy", "act", "--repo-id", "me/x", "--out", "/tmp/o", "--print"]) == 0
    out = capsys.readouterr().out
    assert "--policy.type=act" in out and "--batch_size=40" in out and "lerobot_train" in out
    assert train.main(["--policy", "act", "--repo-id", "me/x", "--out", "/tmp/o", "--alpha-sim", "0.9",
                       "--mapping", "m.json", "--print", "--", "--steps=10"]) == 0
    out = capsys.readouterr().out
    assert "train_weighted.py --alpha-sim 0.9" in out and out.strip().endswith("--steps=10")
    with pytest.raises(SystemExit):
        train.main(["--policy", "diffusion", "--repo-id", "x", "--out", "o", "--alpha-sim", "0.5",
                    "--mapping", "m", "--print"])


def test_check_dataset(tmp_path):
    names = joint_names()
    info = {"fps": 30, "total_episodes": 2, "total_frames": 10, "features": {
        "observation.state": {"shape": [len(names)], "names": names},
        "action": {"shape": [len(names)], "names": names},
        "observation.images.head": {"shape": [3, 480, 640]}}}
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text(json.dumps(info))
    assert check_dataset.main(["--root", str(tmp_path), "--camera", "head"]) == 0
    info["features"]["action"]["names"] = names[::-1]
    (tmp_path / "meta" / "info.json").write_text(json.dumps(info))
    assert check_dataset.main(["--root", str(tmp_path)]) == 1

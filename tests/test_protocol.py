import numpy as np
import pytest

from g1agent import protocol


def test_roundtrip_nested_arrays():
    msg = {"method": "infer", "obs": [{"state": np.arange(28, dtype=np.float32),
                                       "images": {"head": np.random.randint(0, 255, (48, 64, 3), np.uint8)},
                                       "t": 1.5}], "task": None}
    out = protocol.unpack(protocol.pack(msg))
    assert out["method"] == "infer" and out["task"] is None
    f = out["obs"][0]
    np.testing.assert_array_equal(f["state"], msg["obs"][0]["state"])
    assert f["state"].dtype == np.float32
    np.testing.assert_array_equal(f["images"]["head"], msg["obs"][0]["images"]["head"])
    assert f["images"]["head"].flags.writeable


def test_scalars_and_object_arrays_rejected():
    assert protocol.unpack(protocol.pack({"x": np.float32(2.5)}))["x"] == 2.5
    with pytest.raises(TypeError):
        protocol.pack({"x": object()})

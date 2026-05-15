import datetime
import json
import os
import shutil
import unittest
from types import SimpleNamespace

import numpy as np
import xarray as xr

from sunshine_maps import SunshineMapAccumulator, is_sunshine_maps_enabled


def _temp_workspace():
    root = os.path.join(os.getcwd(), ".test_tmp_sunshine")
    if os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(root)
    return root


class SunshineMapTests(unittest.TestCase):
    def test_flags_are_on_by_default_and_model_scoped(self):
        self.assertTrue(is_sunshine_maps_enabled("ch1", env={}))
        self.assertFalse(is_sunshine_maps_enabled("ch1", env={"ENABLE_SUNSHINE_MAPS": "false"}))
        self.assertFalse(
            is_sunshine_maps_enabled(
                "ch1",
                env={"ENABLE_SUNSHINE_MAPS": "true", "ENABLE_SUNSHINE_MAPS_CH1": "false"},
            )
        )

    def test_accumulator_writes_browser_ready_step_binary(self):
        tmp = _temp_workspace()
        try:
            cfg = SimpleNamespace(
                crop={"lon_min": 0.0, "lon_max": 1.0, "lat_min": 0.0, "lat_max": 1.0},
                grid_spacing_deg=1.0,
                source_padding_deg=0.0,
                horizon_stride=1,
            )
            sample = xr.DataArray(
                np.zeros(4, dtype=np.float32),
                dims=("values",),
                coords={
                    "latitude": ("values", np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32)),
                    "longitude": ("values", np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32)),
                },
            )
            acc = SunshineMapAccumulator("ch1", "20260515_0900", datetime.datetime(2026, 5, 15, 9, tzinfo=datetime.timezone.utc), cfg, log=lambda *_: None, out_root=tmp)
            self.assertTrue(
                acc.append(
                    sample,
                    {
                        "DURSUN": np.asarray([0.0, 1800.0, 3600.0, 900.0], dtype=np.float32),
                        "DURSUN_M": np.asarray([3600.0, 3600.0, 3600.0, 1800.0], dtype=np.float32),
                    },
                    1,
                    datetime.datetime(2026, 5, 15, 9, tzinfo=datetime.timezone.utc),
                )
            )
            result = acc.finalize()
            metadata_path = os.path.join(tmp, "ch1", "20260515_0900", "surface", "metadata.json")
            step_path = os.path.join(tmp, "ch1", "20260515_0900", "surface", "steps", "H01.bin")

            self.assertEqual(result["files"], 2)
            self.assertTrue(os.path.exists(metadata_path))
            self.assertTrue(os.path.exists(step_path))

            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            self.assertEqual(metadata["encoding"]["components"], ["sunshine_duration_s", "sunshine_fraction_pct"])
            self.assertEqual(metadata["steps"][0]["path"].replace("\\", "/"), step_path.replace("\\", "/"))

            values = np.fromfile(step_path, dtype="<i2")
            self.assertEqual(values.size, 8)  # 2x2 grid, two interleaved components
            self.assertIn(100, values[1::2])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

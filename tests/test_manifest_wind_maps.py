import os
import shutil
import unittest
from unittest import mock

import numpy as np
import xarray as xr

from generate_combined_manifest import scan_wind_maps


def _temp_workspace():
    root = os.path.join(os.getcwd(), ".test_tmp_manifest")
    if os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(root)
    return root


def _touch_wind_file(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"not opened by test")


def _wind_dataset(level_name, horizons):
    return xr.Dataset(
        {
            "u": xr.DataArray(np.ones((len(horizons), 2, 3), dtype=np.float32), dims=("step", "y", "x")),
            "v": xr.DataArray(np.ones((len(horizons), 2, 3), dtype=np.float32), dims=("step", "y", "x")),
        },
        coords={
            "horizon": xr.DataArray(np.asarray(horizons, dtype=np.int16), dims=("step",)),
        },
        attrs={
            "level_name": level_name,
            "level_type": "AGL",
            "level_h": 800.0,
            "encoding": "int16_scale_0.1_ms",
        },
    )


class ManifestWindMapTests(unittest.TestCase):
    def test_scan_wind_maps_emits_completed_files_only(self):
        tmp = _temp_workspace()
        try:
            ch1_path = os.path.join(tmp, "ch1", "20260510_0300", "Wind_AGL_800m_AGL.nc")
            ch2_path = os.path.join(tmp, "ch2", "20260510_0000", "Wind_AGL_800m_AGL.nc")
            _touch_wind_file(ch1_path)
            _touch_wind_file(ch2_path)
            os.makedirs(os.path.join(tmp, "ch1", "20260510_0300"), exist_ok=True)
            with open(os.path.join(tmp, "ch1", "20260510_0300", "Wind_AGL_partial.tmp"), "w") as f:
                f.write("partial")

            def fake_open_dataset(path, engine=None):
                if path == ch1_path:
                    return _wind_dataset("800m_AGL", [0, 1])
                if path == ch2_path:
                    return _wind_dataset("800m_AGL", [0, 1])
                raise AssertionError(f"unexpected path {path}")

            with mock.patch("generate_combined_manifest.xr.open_dataset", side_effect=fake_open_dataset):
                manifest = scan_wind_maps(tmp)

            ch1_level = manifest["ch1"]["20260510_0300"]["levels"]["800m_AGL"]
            ch2_level = manifest["ch2"]["20260510_0000"]["levels"]["800m_AGL"]

            self.assertEqual(ch1_level["horizons"], ["H00", "H01"])
            self.assertEqual(ch2_level["horizons"], ["H000", "H001"])
            self.assertEqual(ch1_level["encoding"], "int16_scale_0.1_ms")
            self.assertGreater(ch1_level["size_bytes"], 0)
            self.assertEqual(len(manifest["ch1"]["20260510_0300"]["levels"]), 1)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

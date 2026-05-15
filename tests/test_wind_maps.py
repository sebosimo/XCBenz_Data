import unittest

import numpy as np
import xarray as xr

from wind_maps import _interpolate_vertical, _single_level_values, is_wind_maps_enabled, load_config, wind_netcdf_encoding


class WindMapTests(unittest.TestCase):
    def test_flags_are_off_by_default_and_model_scoped(self):
        self.assertFalse(is_wind_maps_enabled("ch1", env={}))
        self.assertFalse(is_wind_maps_enabled("ch1", env={"ENABLE_WIND_MAPS": "true"}))
        self.assertTrue(
            is_wind_maps_enabled(
                "ch1",
                env={"ENABLE_WIND_MAPS": "true", "ENABLE_WIND_MAPS_CH1": "true"},
            )
        )
        self.assertFalse(
            is_wind_maps_enabled(
                "ch2",
                env={"ENABLE_WIND_MAPS": "true", "ENABLE_WIND_MAPS_CH1": "true"},
            )
        )

    def test_config_loads_all_existing_levels_and_crop(self):
        cfg = load_config(log=lambda *_: None)
        self.assertEqual(
            [level.name for level in cfg.enabled_levels],
            ["10m_AGL", "800m_AGL", "1500m_AMSL", "2000m_AMSL", "3000m_AMSL", "4000m_AMSL"],
        )
        self.assertEqual(cfg.crop["lon_min"], 5.5)
        self.assertEqual(cfg.crop["lat_max"], 48.2)
        self.assertEqual(cfg.max_seconds, 0)

    def test_config_can_limit_levels_for_manual_trials(self):
        cfg = load_config(env={"WIND_MAP_LEVELS": "800m_AGL,1500m_AMSL"}, log=lambda *_: None)
        self.assertEqual([level.name for level in cfg.enabled_levels], ["800m_AGL", "1500m_AMSL"])

    def test_wind_netcdf_encoding_packs_wind_components(self):
        ds = xr.Dataset(
            {
                "u": xr.DataArray(np.ones((2, 2, 2), dtype=np.float32), dims=("step", "y", "x")),
                "v": xr.DataArray(np.ones((2, 2, 2), dtype=np.float32), dims=("step", "y", "x")),
            },
            coords={
                "horizon": xr.DataArray(np.asarray([0, 1], dtype=np.int16), dims=("step",)),
                "valid_time_epoch": xr.DataArray(np.asarray([1, 2], dtype=np.int64), dims=("step",)),
            },
        )
        encoding = wind_netcdf_encoding(ds)
        self.assertEqual(encoding["u"]["dtype"], "i2")
        self.assertEqual(encoding["v"]["dtype"], "i2")
        self.assertAlmostEqual(encoding["u"]["scale_factor"], 0.1)
        self.assertEqual(encoding["horizon"]["dtype"], "i2")
        self.assertEqual(encoding["valid_time_epoch"]["dtype"], "i8")

    def test_interpolation_below_lowest_model_layer_remains_missing(self):
        heights = np.asarray(
            [
                [35.0, 45.0],
                [110.0, 120.0],
                [300.0, 320.0],
            ],
            dtype=np.float32,
        )
        values = np.asarray(
            [
                [2.0, 4.0],
                [6.0, 8.0],
                [10.0, 12.0],
            ],
            dtype=np.float32,
        )

        result = _interpolate_vertical(heights, values, 10.0)

        self.assertTrue(np.all(np.isnan(result)))

    def test_single_level_values_extracts_native_10m_wind(self):
        data = xr.DataArray(
            np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
            dims=("heightAboveGround", "values"),
        )

        values = _single_level_values(data, "values")

        np.testing.assert_allclose(values, [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()

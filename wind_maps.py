import datetime
import json
import os
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import xarray as xr
from scipy.spatial import Delaunay, cKDTree


CACHE_DIR_WIND_PACKED = "cache_wind_packed"
DEFAULT_CONFIG_PATH = "wind_maps_config.json"
NETCDF_ENGINE = "netcdf4"
WIND_SCHEMA_VERSION = 1
WIND_ENCODING_NAME = "int16_scale_0.1_ms"
WIND_COMPRESS_KW = {"zlib": True, "shuffle": True, "complevel": 4}
WIND_SCALE_FACTOR = 0.1
WIND_FILL_VALUE = np.int16(-32768)


def _default_log(msg, level="INFO"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} [{level}] {msg}", flush=True)


def _env_bool(name, default=False, env=None):
    source = os.environ if env is None else env
    raw = source.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def is_wind_maps_enabled(model, env=None):
    if not _env_bool("ENABLE_WIND_MAPS", False, env=env):
        return False
    return _env_bool(f"ENABLE_WIND_MAPS_{model.upper()}", False, env=env)


def _env_float(name, default, env=None):
    source = os.environ if env is None else env
    raw = source.get(name)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _safe_level_name(value):
    clean = "".join(c for c in str(value) if c.isalnum() or c in ("-", "_"))
    return clean if clean else "unnamed"


@dataclass(frozen=True)
class WindMapLevel:
    name: str
    h: float
    type: str
    enabled: bool = True


@dataclass(frozen=True)
class WindMapConfig:
    levels: tuple[WindMapLevel, ...]
    crop: dict
    grid_spacing_deg: float
    source_padding_deg: float
    max_seconds: float
    horizon_stride: int = 1

    @property
    def enabled_levels(self):
        return tuple(level for level in self.levels if level.enabled)


def load_config(path=DEFAULT_CONFIG_PATH, env=None, log: Callable[[str, str], None] | None = None):
    logger = log or _default_log
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} is required for wind-map generation")

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    raw_levels = raw.get("levels", raw if isinstance(raw, list) else [])
    levels = []
    only_levels = None
    source_env = os.environ if env is None else env
    if source_env.get("WIND_MAP_LEVELS"):
        only_levels = {
            _safe_level_name(item.strip())
            for item in source_env["WIND_MAP_LEVELS"].split(",")
            if item.strip()
        }

    for item in raw_levels:
        name = _safe_level_name(item["name"])
        enabled = bool(item.get("enabled", True))
        if only_levels is not None:
            enabled = name in only_levels
        levels.append(
            WindMapLevel(
                name=name,
                h=float(item["h"]),
                type=str(item["type"]).upper(),
                enabled=enabled,
            )
        )

    if not levels:
        raise ValueError("wind-map config must define at least one level")

    crop = raw.get("crop", {})
    required_crop_keys = {"lon_min", "lon_max", "lat_min", "lat_max"}
    if set(crop) & required_crop_keys != required_crop_keys:
        raise ValueError("wind-map config crop must define lon_min, lon_max, lat_min, lat_max")

    max_seconds = _env_float("WIND_MAP_MAX_SECONDS", float(raw.get("max_seconds", 300)), env=env)
    cfg = WindMapConfig(
        levels=tuple(levels),
        crop={key: float(crop[key]) for key in required_crop_keys},
        grid_spacing_deg=float(raw.get("grid_spacing_deg", 0.02)),
        source_padding_deg=float(raw.get("source_padding_deg", 0.2)),
        max_seconds=max_seconds,
        horizon_stride=max(1, int(raw.get("horizon_stride", 1))),
    )
    enabled_names = ", ".join(level.name for level in cfg.enabled_levels) or "none"
    logger(f"Wind-map config loaded: {len(cfg.enabled_levels)} enabled level(s): {enabled_names}", "INFO")
    return cfg


def wind_netcdf_encoding(ds):
    encoding = {}
    for name, data_array in ds.variables.items():
        if data_array.dtype.kind in ("U", "S", "O"):
            encoding[name] = {}
            continue
        enc = dict(WIND_COMPRESS_KW)
        if name in {"u", "v"}:
            enc.update({
                "dtype": "i2",
                "scale_factor": WIND_SCALE_FACTOR,
                "add_offset": 0.0,
                "_FillValue": WIND_FILL_VALUE,
            })
        elif name == "horizon":
            enc["dtype"] = "i2"
        elif name == "valid_time_epoch":
            enc["dtype"] = "i8"
        else:
            enc["dtype"] = "f4"
        encoding[name] = enc
    return encoding


def _regular_crop_grid(crop, spacing):
    lon = np.arange(crop["lon_min"], crop["lon_max"] + spacing * 0.5, spacing, dtype=np.float32)
    lat = np.arange(crop["lat_min"], crop["lat_max"] + spacing * 0.5, spacing, dtype=np.float32)
    lon2d, lat2d = np.meshgrid(lon, lat)
    return lat2d.astype(np.float32), lon2d.astype(np.float32)


def _lat_lon_coord(data):
    lat_name = "latitude" if "latitude" in data.coords else "lat"
    lon_name = "longitude" if "longitude" in data.coords else "lon"
    if lat_name not in data.coords or lon_name not in data.coords:
        raise ValueError("wind fields need latitude/longitude coordinates")
    lat = data[lat_name]
    lon = data[lon_name]
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("wind-map generation expects 1D native grid coordinates")
    if lat.dims[0] != lon.dims[0]:
        raise ValueError("latitude and longitude coordinates must share a spatial dimension")
    return lat.dims[0], lat.values.astype(np.float64), lon.values.astype(np.float64)


def _level_cell_values(data, spatial_dim, expected_levels=None):
    arr = data.squeeze()
    if spatial_dim not in arr.dims:
        raise ValueError(f"spatial dimension {spatial_dim!r} not found in {arr.dims}")
    dims = list(arr.dims)
    spatial_axis = dims.index(spatial_dim)
    other_dims = [d for d in dims if d != spatial_dim]
    if expected_levels is not None:
        level_dim = next((d for d in other_dims if arr.sizes[d] == expected_levels), None)
    else:
        level_dim = other_dims[0] if other_dims else None
    if level_dim is None:
        raise ValueError("could not identify vertical dimension in wind field")
    level_axis = dims.index(level_dim)
    values = np.moveaxis(arr.values, (level_axis, spatial_axis), (0, 1))
    if values.ndim != 2:
        values = values.reshape(values.shape[0], values.shape[1], -1)
        if values.shape[-1] != 1:
            raise ValueError("wind field has unsupported extra dimensions")
        values = values[:, :, 0]
    return values.astype(np.float32, copy=False)


def _interpolate_vertical(heights, values, target_h):
    z = heights
    vals = values
    if np.nanmedian(z[0] - z[-1]) > 0:
        z = z[::-1]
        vals = vals[::-1]

    target = np.float32(target_h)
    z_min = z[0]
    z_max = z[-1]
    valid = np.isfinite(z_min) & np.isfinite(z_max) & (target >= z_min) & (target <= z_max)
    idx1 = np.sum(z <= target, axis=0)
    idx1 = np.clip(idx1, 1, z.shape[0] - 1)
    idx0 = idx1 - 1
    cols = np.arange(z.shape[1])
    z0 = z[idx0, cols]
    z1 = z[idx1, cols]
    v0 = vals[idx0, cols]
    v1 = vals[idx1, cols]
    denom = z1 - z0
    frac = np.divide(target - z0, denom, out=np.zeros_like(z0, dtype=np.float32), where=denom != 0)
    out = v0 + frac * (v1 - v0)
    out = out.astype(np.float32, copy=False)
    out[~valid] = np.nan
    return out


class _HorizontalWeights:
    def __init__(self, source_lon, source_lat, target_lon, target_lat):
        points = np.column_stack([source_lon, source_lat])
        target_points = np.column_stack([target_lon.ravel(), target_lat.ravel()])
        self.output_shape = target_lon.shape

        tri = Delaunay(points)
        simplices = tri.find_simplex(target_points)
        self.inside_mask = simplices >= 0
        self.inside_target = np.flatnonzero(self.inside_mask)
        self.vertices = np.empty((0, 3), dtype=np.int32)
        self.weights = np.empty((0, 3), dtype=np.float32)
        if self.inside_target.size:
            inside_simplices = simplices[self.inside_mask]
            transforms = tri.transform[inside_simplices]
            delta = target_points[self.inside_mask] - transforms[:, 2]
            bary = np.einsum("ijk,ik->ij", transforms[:, :2], delta)
            self.weights = np.column_stack([bary, 1.0 - bary.sum(axis=1)]).astype(np.float32)
            self.vertices = tri.simplices[inside_simplices].astype(np.int32)

        outside_target = np.flatnonzero(~self.inside_mask)
        self.outside_target = outside_target
        self.outside_source = np.empty(0, dtype=np.int32)
        if outside_target.size:
            _, nearest = cKDTree(points).query(target_points[outside_target], k=1)
            self.outside_source = nearest.astype(np.int32)

    def apply(self, source_values):
        flat = np.full(self.output_shape[0] * self.output_shape[1], np.nan, dtype=np.float32)
        if self.inside_target.size:
            vals = source_values[self.vertices]
            flat[self.inside_target] = np.sum(vals * self.weights, axis=1, dtype=np.float32)
        if self.outside_target.size:
            flat[self.outside_target] = source_values[self.outside_source]
        return flat.reshape(self.output_shape).astype(np.float32, copy=False)


class WindMapAccumulator:
    def __init__(self, model, run_tag, ref_time, config, log=None, out_root=CACHE_DIR_WIND_PACKED):
        self.model = model
        self.run_tag = run_tag
        self.ref_time = ref_time
        self.config = config
        self.log = log or _default_log
        self.out_root = out_root
        self.started_at = time.monotonic()
        self.wind_elapsed_seconds = 0.0
        self.max_wind_seconds = float(config.max_seconds)
        self.prepared = False
        self.budget_exceeded = False
        self.failed = False
        self.target_lat = None
        self.target_lon = None
        self.source_indices = None
        self.heights_full = None
        self.surface_height = None
        self.weights = None
        self.records = {
            level.name: {"u": [], "v": [], "horizon": [], "valid_time_epoch": [], "step_label": []}
            for level in config.enabled_levels
        }

    def _over_budget(self):
        if self.max_wind_seconds > 0 and self.wind_elapsed_seconds > self.max_wind_seconds:
            self.budget_exceeded = True
            return True
        return False

    def _prepare(self, fields):
        sample = fields["U"].squeeze()
        spatial_dim, lat, lon = _lat_lon_coord(sample)
        crop = self.config.crop
        pad = self.config.source_padding_deg
        mask = (
            (lon >= crop["lon_min"] - pad)
            & (lon <= crop["lon_max"] + pad)
            & (lat >= crop["lat_min"] - pad)
            & (lat <= crop["lat_max"] + pad)
        )
        source_indices = np.flatnonzero(mask)
        if source_indices.size < 3:
            raise ValueError("not enough source points inside wind-map crop")

        hhl = fields["HHL"].squeeze()
        hhl_values = _level_cell_values(hhl, spatial_dim)
        hhl_values = hhl_values[:, source_indices]
        self.heights_full = ((hhl_values[:-1] + hhl_values[1:]) * 0.5).astype(np.float32)
        self.surface_height = hhl_values[-1].astype(np.float32)
        self.source_indices = source_indices
        self.target_lat, self.target_lon = _regular_crop_grid(crop, self.config.grid_spacing_deg)
        self.weights = _HorizontalWeights(
            lon[source_indices],
            lat[source_indices],
            self.target_lon,
            self.target_lat,
        )
        self.prepared = True
        self.log(
            f"Wind maps {self.model}: crop grid {self.target_lat.shape[1]}x{self.target_lat.shape[0]}, "
            f"{source_indices.size} source point(s)",
            "INFO",
        )

    def append(self, fields, horizon, ref_time):
        wind_start = time.monotonic()
        if horizon % self.config.horizon_stride != 0:
            return False
        if self._over_budget():
            self.log(
                f"Wind maps {self.model}: wind budget exceeded before H+{horizon:03d} "
                f"({self.wind_elapsed_seconds:.1f}s active wind time); skipping remaining horizons",
                "WARNING",
            )
            return False
        missing = [name for name in ("U", "V", "HHL") if name not in fields or fields[name] is None]
        if missing:
            self.log(f"Wind maps {self.model}: missing fields for H+{horizon:03d}: {missing}", "WARNING")
            return False

        try:
            if not self.prepared:
                self._prepare(fields)
            spatial_dim, _, _ = _lat_lon_coord(fields["U"].squeeze())
            u_all = _level_cell_values(fields["U"], spatial_dim, expected_levels=self.heights_full.shape[0])
            v_all = _level_cell_values(fields["V"], spatial_dim, expected_levels=self.heights_full.shape[0])
            u_source = u_all[:, self.source_indices]
            v_source = v_all[:, self.source_indices]
            valid_time = ref_time + datetime.timedelta(hours=int(horizon))
            if valid_time.tzinfo is None:
                valid_time = valid_time.replace(tzinfo=datetime.timezone.utc)

            for level in self.config.enabled_levels:
                heights = self.heights_full - self.surface_height if level.type == "AGL" else self.heights_full
                rec = self.records[level.name]
                rec["u"].append(self.weights.apply(_interpolate_vertical(heights, u_source, level.h)))
                rec["v"].append(self.weights.apply(_interpolate_vertical(heights, v_source, level.h)))
                rec["horizon"].append(int(horizon))
                rec["valid_time_epoch"].append(int(valid_time.timestamp()))
                rec["step_label"].append(f"H{int(horizon):03d}" if self.model == "ch2" else f"H{int(horizon):02d}")
            return True
        except Exception as exc:
            self.failed = True
            self.log(f"Wind maps {self.model}: H+{horizon:03d} failed: {exc}", "WARNING")
            return False
        finally:
            self.wind_elapsed_seconds += time.monotonic() - wind_start

    def _dataset_for_level(self, level, rec):
        u_stack = np.stack(rec["u"]).astype(np.float32)
        v_stack = np.stack(rec["v"]).astype(np.float32)
        ds = xr.Dataset(
            {
                "u": xr.DataArray(u_stack, dims=("step", "y", "x"), attrs={"units": "m s-1"}),
                "v": xr.DataArray(v_stack, dims=("step", "y", "x"), attrs={"units": "m s-1"}),
            },
            coords={
                "horizon": xr.DataArray(np.asarray(rec["horizon"], dtype=np.int16), dims=("step",)),
                "valid_time_epoch": xr.DataArray(np.asarray(rec["valid_time_epoch"], dtype=np.int64), dims=("step",)),
                "step_label": xr.DataArray(np.asarray(rec["step_label"]), dims=("step",)),
                "latitude": xr.DataArray(self.target_lat, dims=("y", "x"), attrs={"units": "degrees_north"}),
                "longitude": xr.DataArray(self.target_lon, dims=("y", "x"), attrs={"units": "degrees_east"}),
            },
            attrs={
                "schema_version": WIND_SCHEMA_VERSION,
                "source": "MeteoSwiss ICON OGD",
                "model": f"icon-{self.model}",
                "layout": "packed_by_level",
                "level_name": level.name,
                "level_type": level.type,
                "level_h": float(level.h),
                "ref_time": self.ref_time.isoformat(),
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "crop_lon_min": self.config.crop["lon_min"],
                "crop_lon_max": self.config.crop["lon_max"],
                "crop_lat_min": self.config.crop["lat_min"],
                "crop_lat_max": self.config.crop["lat_max"],
                "grid_spacing_deg": self.config.grid_spacing_deg,
                "encoding": WIND_ENCODING_NAME,
            },
        )
        return ds

    def finalize(self):
        elapsed = time.monotonic() - self.started_at
        if not any(rec["horizon"] for rec in self.records.values()):
            self.log(
                f"Wind maps {self.model}: no horizons accumulated "
                f"(wind={self.wind_elapsed_seconds:.1f}s, wall={elapsed:.1f}s)",
                "INFO",
            )
            return {"files": 0, "bytes": 0, "elapsed_seconds": elapsed, "wind_elapsed_seconds": self.wind_elapsed_seconds}
        if self.budget_exceeded:
            self.log(
                f"Wind maps {self.model}: not writing partial files after wind budget exceed "
                f"(wind={self.wind_elapsed_seconds:.1f}s, wall={elapsed:.1f}s)",
                "WARNING",
            )
            return {
                "files": 0,
                "bytes": 0,
                "elapsed_seconds": elapsed,
                "wind_elapsed_seconds": self.wind_elapsed_seconds,
                "budget_exceeded": True,
            }

        out_dir = os.path.join(self.out_root, self.model, self.run_tag)
        os.makedirs(out_dir, exist_ok=True)
        file_count = 0
        byte_count = 0
        for level in self.config.enabled_levels:
            rec = self.records[level.name]
            if not rec["horizon"]:
                continue
            ds = self._dataset_for_level(level, rec)
            path = os.path.join(out_dir, f"Wind_{level.type}_{level.name}.nc")
            tmp_path = path + ".tmp"
            ds.to_netcdf(
                tmp_path,
                engine=NETCDF_ENGINE,
                format="NETCDF4",
                encoding=wind_netcdf_encoding(ds),
            )
            os.replace(tmp_path, path)
            size = os.path.getsize(path)
            file_count += 1
            byte_count += size
            self.log(
                f"Wind maps {self.model}: wrote {path} ({len(rec['horizon'])} horizon(s), {size} bytes)",
                "INFO",
            )

        elapsed = time.monotonic() - self.started_at
        self.log(
            f"Wind maps {self.model}: complete in {self.wind_elapsed_seconds:.1f}s active wind time "
            f"({elapsed:.1f}s wall), files={file_count}, bytes={byte_count}",
            "NOTICE",
        )
        return {
            "files": file_count,
            "bytes": byte_count,
            "elapsed_seconds": elapsed,
            "wind_elapsed_seconds": self.wind_elapsed_seconds,
        }


def cleanup_old_wind_runs(model, anchor_hour, log=None, root=CACHE_DIR_WIND_PACKED):
    logger = log or _default_log
    model_dir = os.path.join(root, model)
    if not os.path.exists(model_dir):
        return

    now = datetime.datetime.now(datetime.timezone.utc)
    keep_dates = {now.date(), (now - datetime.timedelta(days=1)).date()}
    all_runs = sorted(
        [item for item in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, item))],
        reverse=True,
    )
    keep_recent = set(all_runs[:2])
    for item in all_runs:
        if item in keep_recent:
            continue
        try:
            dt = datetime.datetime.strptime(item, "%Y%m%d_%H%M").replace(tzinfo=datetime.timezone.utc)
        except ValueError:
            continue
        if dt.hour == anchor_hour and dt.minute == 0 and dt.date() in keep_dates:
            continue
        path = os.path.join(model_dir, item)
        try:
            import shutil

            shutil.rmtree(path)
            logger(f"Wind maps {model}: cleanup removed {item}", "INFO")
        except Exception as exc:
            logger(f"Wind maps {model}: cleanup failed {item}: {exc}", "WARNING")

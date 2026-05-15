import datetime
import json
import os
import shutil
from typing import Callable

import numpy as np

from wind_maps import _HorizontalWeights, _lat_lon_coord, _regular_crop_grid


CACHE_DIR_SUNSHINE_MAPS = "cache_sunshine_maps"
SUNSHINE_SCHEMA_VERSION = 1
SUNSHINE_SCALE_FACTOR = 1.0
SUNSHINE_FILL_VALUE = np.int16(-32768)
SUNSHINE_COMPONENTS = ["sunshine_duration_s", "sunshine_fraction_pct"]


def _default_log(msg, level="INFO"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} [{level}] {msg}", flush=True)


def _env_bool(name, default=False, env=None):
    source = os.environ if env is None else env
    raw = source.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def is_sunshine_maps_enabled(model, env=None):
    if not _env_bool("ENABLE_SUNSHINE_MAPS", True, env=env):
        return False
    return _env_bool(f"ENABLE_SUNSHINE_MAPS_{model.upper()}", True, env=env)


def _json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def is_sunshine_run_complete(model, run_tag, root=CACHE_DIR_SUNSHINE_MAPS):
    metadata_path = os.path.join(root, model, run_tag, "surface", "metadata.json")
    if not os.path.exists(metadata_path):
        return False
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        steps = metadata.get("steps") or []
        return bool(steps) and all(os.path.exists((step.get("path") or "").replace("/", os.sep)) for step in steps)
    except Exception:
        return False


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_safe)
        f.write("\n")


class SunshineMapAccumulator:
    def __init__(self, model, run_tag, ref_time, config, log=None, out_root=CACHE_DIR_SUNSHINE_MAPS):
        self.model = model
        self.model_key = "icon-ch1" if model == "ch1" else "icon-ch2"
        self.run_tag = run_tag
        self.ref_time = ref_time
        self.config = config
        self.log = log or _default_log
        self.out_root = out_root
        self.output_dir = os.path.join(out_root, model, run_tag, "surface")
        self.steps_dir = os.path.join(self.output_dir, "steps")
        self.prepared = False
        self.target_lat = None
        self.target_lon = None
        self.source_indices = None
        self.weights = None
        self.steps = []

    def _prepare(self, sample_field):
        sample = sample_field.squeeze()
        _spatial_dim, lat, lon = _lat_lon_coord(sample)
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
            raise ValueError("not enough source points inside sunshine-map crop")

        self.source_indices = source_indices
        self.target_lat, self.target_lon = _regular_crop_grid(crop, self.config.grid_spacing_deg)
        self.weights = _HorizontalWeights(lon[source_indices], lat[source_indices], self.target_lon, self.target_lat)
        self.prepared = True
        self.log(
            f"Sunshine maps {self.model}: crop grid {self.target_lat.shape[1]}x{self.target_lat.shape[0]}, "
            f"{source_indices.size} source point(s)",
            "INFO",
        )

    def _map_source(self, values_by_source, source_name):
        if source_name not in values_by_source:
            return np.full(self.target_lat.shape, np.nan, dtype=np.float32)
        values = np.asarray(values_by_source[source_name], dtype=np.float32).ravel()
        return self.weights.apply(values[self.source_indices])

    def append(self, sample_field, values_by_source, horizon, ref_time):
        if horizon % self.config.horizon_stride != 0:
            return False
        if "DURSUN" not in values_by_source or "DURSUN_M" not in values_by_source:
            return False

        try:
            if not self.prepared:
                self._prepare(sample_field)

            duration = self._map_source(values_by_source, "DURSUN")
            possible = self._map_source(values_by_source, "DURSUN_M")
            fraction = np.divide(
                duration * 100.0,
                possible,
                out=np.zeros_like(duration, dtype=np.float32),
                where=np.isfinite(possible) & (possible > 0),
            )
            fraction = np.clip(fraction, 0.0, 100.0)
            duration = np.clip(duration, 0.0, 3600.0)

            step_label = f"H{int(horizon):03d}" if self.model == "ch2" else f"H{int(horizon):02d}"
            interleaved = np.empty(duration.size * len(SUNSHINE_COMPONENTS), dtype="<i2")
            for offset, values in enumerate((duration, fraction)):
                scaled = np.rint(values / SUNSHINE_SCALE_FACTOR)
                scaled[~np.isfinite(scaled)] = SUNSHINE_FILL_VALUE
                interleaved[offset::len(SUNSHINE_COMPONENTS)] = np.clip(scaled, -32767, 32767).astype("<i2").ravel()

            os.makedirs(self.steps_dir, exist_ok=True)
            step_path = os.path.join(self.steps_dir, f"{step_label}.bin")
            with open(step_path, "wb") as f:
                f.write(interleaved.tobytes())

            valid_time = ref_time + datetime.timedelta(hours=int(horizon))
            if valid_time.tzinfo is None:
                valid_time = valid_time.replace(tzinfo=datetime.timezone.utc)

            valid_duration = duration[np.isfinite(duration)]
            valid_fraction = fraction[np.isfinite(fraction)]
            self.steps.append(
                {
                    "step": step_label,
                    "horizon": int(horizon),
                    "valid_time": valid_time.isoformat(),
                    "path": step_path.replace(os.sep, "/"),
                    "byte_length": int(os.path.getsize(step_path)),
                    "max_sunshine_duration_s": float(np.nanmax(valid_duration)) if valid_duration.size else None,
                    "max_sunshine_fraction_pct": float(np.nanmax(valid_fraction)) if valid_fraction.size else None,
                }
            )
            return True
        except Exception as exc:
            self.log(f"Sunshine maps {self.model}: H+{horizon:03d} failed: {exc}", "WARNING")
            return False

    def finalize(self):
        if not self.steps:
            self.log(f"Sunshine maps {self.model}: no horizons accumulated", "INFO")
            return {"files": 0, "bytes": 0}

        height, width = self.target_lat.shape
        metadata_path = os.path.join(self.output_dir, "metadata.json")
        payload = {
            "schema_version": SUNSHINE_SCHEMA_VERSION,
            "product": "sunshine_map_surface",
            "model": self.model_key,
            "run": self.run_tag,
            "product_name": "surface",
            "ref_time": self.ref_time.isoformat(),
            "grid": {
                "projection": "EPSG:4326",
                "width": int(width),
                "height": int(height),
                "lon": {
                    "start": float(self.target_lon[0, 0]),
                    "end": float(self.target_lon[0, -1]),
                    "step": float(self.target_lon[0, 1] - self.target_lon[0, 0]) if width > 1 else 0.0,
                    "count": int(width),
                },
                "lat": {
                    "start": float(self.target_lat[0, 0]),
                    "end": float(self.target_lat[-1, 0]),
                    "step": float(self.target_lat[1, 0] - self.target_lat[0, 0]) if height > 1 else 0.0,
                    "count": int(height),
                },
            },
            "encoding": {
                "format": "int16-le-interleaved-components",
                "components": SUNSHINE_COMPONENTS,
                "units": ["s", "%"],
                "scale_factor": SUNSHINE_SCALE_FACTOR,
                "add_offset": 0.0,
                "missing_value": int(SUNSHINE_FILL_VALUE),
            },
            "style": {
                "map_bbox": [
                    self.config.crop["lon_min"],
                    self.config.crop["lat_min"],
                    self.config.crop["lon_max"],
                    self.config.crop["lat_max"],
                ],
                "display_component": "sunshine_fraction_pct",
                "display_units": "%",
                "bounds_pct": [0, 5, 15, 30, 50, 70, 85, 100],
                "colors": ["#f6f7f0", "#fff7b8", "#fee477", "#fcb94d", "#f47c30", "#d94f1f", "#9f2d16"],
            },
            "steps": self.steps,
        }
        _write_json(metadata_path, payload)
        total_bytes = sum(step["byte_length"] for step in self.steps) + os.path.getsize(metadata_path)
        self.log(
            f"Sunshine maps {self.model}: wrote {len(self.steps)} browser-ready step(s), {total_bytes} bytes",
            "NOTICE",
        )
        return {"files": len(self.steps) + 1, "bytes": total_bytes}


def cleanup_old_sunshine_runs(model, anchor_hour, log=None, root=CACHE_DIR_SUNSHINE_MAPS):
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
            run_dt = datetime.datetime.strptime(item, "%Y%m%d_%H%M").replace(tzinfo=datetime.timezone.utc)
        except ValueError:
            continue
        if run_dt.hour == anchor_hour and run_dt.minute == 0 and run_dt.date() in keep_dates:
            continue
        path = os.path.join(model_dir, item)
        try:
            shutil.rmtree(path)
            logger(f"Sunshine maps {model}: cleanup removed {item}", "INFO")
        except Exception as exc:
            logger(f"Sunshine maps {model}: cleanup failed {item}: {exc}", "WARNING")

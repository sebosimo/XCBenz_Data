"""
generate_combined_manifest.py

Called by CI after both fetch_data.py and fetch_data_ch2.py complete.
Scans cache_data/ (CH1) and cache_data_ch2/ (CH2), then writes a single
unified manifest.json that app.py reads at runtime.
"""
import os
import json
import datetime
import xarray as xr


CACHE_DIR_CH1 = "cache_data"
CACHE_DIR_CH2 = "cache_data_ch2"
CACHE_DIR_CH1_PACKED = "cache_data_packed"
CACHE_DIR_CH2_PACKED = "cache_data_ch2_packed"
CACHE_DIR_WIND_PACKED = "cache_wind_packed"


def log(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} [INFO] {msg}", flush=True)


def scan_runs(cache_dir, pad):
    """
    Scan a cache directory and return {run_tag: {location: [step_labels]}}.

    pad: number of digits for horizon labels (2 for CH1 → H00, 3 for CH2 → H000).
    Step labels are derived from filenames: H00.nc → "H00", H000.nc → "H000".
    """
    runs = {}
    if not os.path.exists(cache_dir):
        return runs

    for run in sorted(os.listdir(cache_dir), reverse=True):
        run_path = os.path.join(cache_dir, run)
        if not os.path.isdir(run_path):
            continue
        locations = {}
        for loc in sorted(os.listdir(run_path)):
            loc_path = os.path.join(run_path, loc)
            if not os.path.isdir(loc_path):
                continue
            steps = sorted(
                f.replace(".nc", "")
                for f in os.listdir(loc_path)
                if f.endswith(".nc") and f.startswith("H")   # skip thermals.nc etc.
            )
            if steps:
                locations[loc] = steps
        if locations:
            runs[run] = locations

    return runs

def scan_packed_runs(cache_dir):
    """
    Scan a packed cache directory and return {run_tag: [location_names]}.
    Packed filenames are <Location>.nc.
    """
    runs = {}
    if not os.path.exists(cache_dir):
        return runs

    for run in sorted(os.listdir(cache_dir), reverse=True):
        run_path = os.path.join(cache_dir, run)
        if not os.path.isdir(run_path):
            continue
        locations = sorted(
            f.replace(".nc", "")
            for f in os.listdir(run_path)
            if f.endswith(".nc")
        )
        if locations:
            runs[run] = locations

    return runs


def _format_horizons(values, pad):
    return [f"H{int(value):0{pad}d}" for value in values]


def scan_wind_maps(cache_dir=CACHE_DIR_WIND_PACKED):
    """
    Scan packed wind-map files and return manifest-ready entries.

    Layout:
      cache_wind_packed/{model}/{run}/Wind_{type}_{level}.nc
    Only fully written .nc files with u/v and horizon coordinates are emitted.
    """
    wind_maps = {}
    if not os.path.exists(cache_dir):
        return wind_maps

    for model in ("ch1", "ch2"):
        model_dir = os.path.join(cache_dir, model)
        if not os.path.isdir(model_dir):
            continue
        model_runs = {}
        pad = 3 if model == "ch2" else 2
        for run in sorted(os.listdir(model_dir), reverse=True):
            run_path = os.path.join(model_dir, run)
            if not os.path.isdir(run_path):
                continue
            levels = {}
            for filename in sorted(os.listdir(run_path)):
                if not filename.endswith(".nc"):
                    continue
                path = os.path.join(run_path, filename)
                rel_path = os.path.relpath(path, ".").replace(os.sep, "/")
                try:
                    with xr.open_dataset(path, engine="netcdf4") as ds:
                        if "u" not in ds or "v" not in ds or "horizon" not in ds:
                            continue
                        level_name = str(ds.attrs.get("level_name") or filename.replace(".nc", ""))
                        horizons = _format_horizons(ds["horizon"].values, pad)
                        if not horizons:
                            continue
                        levels[level_name] = {
                            "path": rel_path,
                            "horizons": horizons,
                            "encoding": str(ds.attrs.get("encoding", "int16_scale_0.1_ms")),
                            "level_type": str(ds.attrs.get("level_type", "")),
                            "level_h": float(ds.attrs.get("level_h", 0.0)),
                            "size_bytes": os.path.getsize(path),
                        }
                except Exception as exc:
                    log(f"Skipping invalid wind-map file {rel_path}: {exc}")
            if levels:
                model_runs[run] = {
                    "layout": "packed_by_level",
                    "levels": levels,
                }
        if model_runs:
            wind_maps[model] = model_runs

    return wind_maps


def main():
    runs_ch1 = scan_runs(CACHE_DIR_CH1, pad=2)
    runs_ch2 = scan_runs(CACHE_DIR_CH2, pad=3)
    runs_ch1_packed = scan_packed_runs(CACHE_DIR_CH1_PACKED)
    runs_ch2_packed = scan_packed_runs(CACHE_DIR_CH2_PACKED)
    wind_maps = scan_wind_maps()

    # generated_at: use the newest CH1 run (the "current" model reference)
    generated_at = max(runs_ch1.keys()) if runs_ch1 else (
        max(runs_ch2.keys()) if runs_ch2 else ""
    )

    manifest = {
        "generated_at": generated_at,
        "schema_version": 3,
        "runs": runs_ch1,
        "runs_ch2": runs_ch2,
        "runs_packed": runs_ch1_packed,
        "runs_ch2_packed": runs_ch2_packed,
        "wind_maps": wind_maps,
    }

    with open("manifest.json", "w") as f:
        json.dump(manifest, f)

    log(
        f"Manifest written: {len(runs_ch1)} CH1 run(s), {len(runs_ch2)} CH2 run(s), "
        f"{len(runs_ch1_packed)} packed CH1 run(s), {len(runs_ch2_packed)} packed CH2 run(s), "
        f"{sum(len(runs) for runs in wind_maps.values())} wind-map run(s)"
    )


if __name__ == "__main__":
    main()

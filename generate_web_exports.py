"""
generate_web_exports.py

Build browser-friendly JSON exports from the existing generated NetCDF files.
This is intentionally additive: Streamlit keeps reading cache_data*/ and the
root manifest.json, while the web app can read web_exports/.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import re
import shutil
import stat
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


SCHEMA_VERSION = 1
WEB_DIR = Path("web_exports")
LOCATIONS_FILE = Path("locations.json")
SOURCE_MANIFEST_FILE = Path("manifest.json")

MODELS = (
    {
        "key": "icon-ch1",
        "label": "ICON-CH1",
        "cache_dir": Path("cache_data"),
        "horizon_digits": 2,
    },
    {
        "key": "icon-ch2",
        "label": "ICON-CH2",
        "cache_dir": Path("cache_data_ch2"),
        "horizon_digits": 3,
    },
)

PROFILE_VARIABLES = ("HEIGHT", "P", "T", "QV", "U", "V")
RADIATION_VARIABLES = ("ASWDIR_S", "ASWDIFD_S")


def log(message: str) -> None:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} [web-export] {message}", flush=True)


def sanitize_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    clean = "".join(c for c in normalized if c.isalnum() or c in ("-", "_"))
    return clean if clean else "unnamed"


def horizon_sort_key(label: str) -> tuple[int, str]:
    match = re.search(r"(\d+)", label)
    if not match:
        return (10_000, label)
    return (int(match.group(1)), label)


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def rel(path: Path) -> str:
    return path.as_posix()


def ensure_clean_web_dir() -> None:
    if WEB_DIR.exists():
        if WEB_DIR.resolve().name != "web_exports":
            raise RuntimeError(f"Refusing to delete unexpected export dir: {WEB_DIR}")
        shutil.rmtree(WEB_DIR, onerror=remove_readonly)
    WEB_DIR.mkdir(parents=True, exist_ok=True)


def remove_readonly(func: Any, path: str, _exc_info: Any) -> None:
    Path(path).chmod(stat.S_IWRITE)
    func(path)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any, *, pretty: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if pretty:
            json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
        else:
            json.dump(payload, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def clean_number(value: Any, precision: int | None = None) -> float | int | bool | None:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            return None
        if precision is not None:
            return round(number, precision)
        return number
    return None


def array_to_list(values: Any, precision: int | None = None) -> list[Any]:
    arr = np.asarray(values)
    if arr.ndim == 0:
        return [clean_number(arr.item(), precision)]
    return [clean_value(v, precision) for v in arr.tolist()]


def clean_value(value: Any, precision: int | None = None) -> Any:
    if isinstance(value, list):
        return [clean_value(item, precision) for item in value]
    if isinstance(value, tuple):
        return [clean_value(item, precision) for item in value]
    number = clean_number(value, precision)
    if number is not None or isinstance(value, (np.floating, float)):
        return number
    if isinstance(value, (str, type(None))):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (np.datetime64,)):
        return str(value)
    return str(value)


def scalar_value(ds: xr.Dataset, name: str, precision: int | None = None) -> Any:
    if name not in ds:
        return None
    values = np.asarray(ds[name].values)
    if values.ndim != 0:
        return None
    return clean_number(values.item(), precision)


def vector_values(ds: xr.Dataset, name: str, precision: int | None = None) -> list[Any] | None:
    if name not in ds:
        return None
    values = np.asarray(ds[name].values)
    if values.ndim == 0:
        return None
    return array_to_list(values, precision)


def pressure_to_hpa(pressure_pa: np.ndarray) -> np.ndarray:
    return pressure_pa.astype(float) / 100.0


def temperature_to_celsius(temperature_k: np.ndarray) -> np.ndarray:
    return temperature_k.astype(float) - 273.15


def dewpoint_from_specific_humidity(pressure_pa: np.ndarray, q_kgkg: np.ndarray) -> np.ndarray:
    q = np.maximum(q_kgkg.astype(float), 1e-9)
    vapor_pressure_hpa = (q * pressure_pa.astype(float) / (0.622 + 0.378 * q)) / 100.0
    vapor_pressure_hpa = np.maximum(vapor_pressure_hpa, 1e-6)
    gamma = np.log(vapor_pressure_hpa / 6.112)
    return (243.5 * gamma) / (17.67 - gamma)


def wind_speed(u_ms: np.ndarray, v_ms: np.ndarray) -> np.ndarray:
    return np.sqrt(np.square(u_ms.astype(float)) + np.square(v_ms.astype(float)))


def wind_direction_from(u_ms: np.ndarray, v_ms: np.ndarray) -> np.ndarray:
    return (270.0 - np.degrees(np.arctan2(v_ms.astype(float), u_ms.astype(float)))) % 360.0


def location_payload(location_id: str, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "location_id": location_id,
        "display_name": meta.get("display_name", location_id),
        "type": meta.get("type", "legacy"),
        "region_name": meta.get("region_name"),
        "lat": clean_number(meta.get("lat"), 6),
        "lon": clean_number(meta.get("lon"), 6),
    }


def scan_profiles(cache_dir: Path, locations: dict[str, Any]) -> dict[str, dict[str, list[Path]]]:
    runs: dict[str, dict[str, list[Path]]] = {}
    if not cache_dir.exists():
        return runs

    for run_dir in sorted((p for p in cache_dir.iterdir() if p.is_dir()), reverse=True):
        run_locations: dict[str, list[Path]] = {}
        for location_id in sorted(locations):
            candidates = [run_dir / location_id]
            sanitized = sanitize_name(location_id)
            if sanitized != location_id:
                candidates.append(run_dir / sanitized)
            loc_dir = next((p for p in candidates if p.is_dir()), None)
            if not loc_dir:
                continue
            step_files = sorted(
                (p for p in loc_dir.glob("H*.nc") if p.is_file()),
                key=lambda p: horizon_sort_key(p.stem),
            )
            if step_files:
                run_locations[location_id] = step_files
        if run_locations:
            runs[run_dir.name] = run_locations
    return runs


def export_profile(
    model_key: str,
    run_tag: str,
    location_id: str,
    location_meta: dict[str, Any],
    source_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    with xr.open_dataset(source_path) as ds:
        ds.load()

        attrs = dict(ds.attrs)
        profile: dict[str, Any] = {}
        for var in PROFILE_VARIABLES:
            values = vector_values(ds, var, precision=3)
            if values is not None:
                profile[var.lower()] = values

        derived: dict[str, Any] = {}
        if "T" in ds:
            derived["temperature_c"] = array_to_list(temperature_to_celsius(ds["T"].values), 2)
        if "P" in ds:
            derived["pressure_hpa"] = array_to_list(pressure_to_hpa(ds["P"].values), 1)
        if "P" in ds and "QV" in ds:
            derived["dewpoint_c"] = array_to_list(
                dewpoint_from_specific_humidity(ds["P"].values, ds["QV"].values),
                2,
            )
        if "U" in ds and "V" in ds:
            derived["wind_speed_ms"] = array_to_list(wind_speed(ds["U"].values, ds["V"].values), 2)
            derived["wind_dir_deg"] = array_to_list(wind_direction_from(ds["U"].values, ds["V"].values), 0)

        surface = {
            "aswdir_s_wm2": scalar_value(ds, "ASWDIR_S", precision=2),
            "aswdifd_s_wm2": scalar_value(ds, "ASWDIFD_S", precision=2),
        }
        surface = {k: v for k, v in surface.items() if v is not None}

    payload = {
        "schema_version": SCHEMA_VERSION,
        "product": "emagram_profile",
        "model": model_key,
        "run": run_tag,
        "step": source_path.stem,
        "location": location_payload(location_id, location_meta),
        "ref_time": attrs.get("ref_time"),
        "valid_time": attrs.get("valid_time"),
        "horizon": clean_number(attrs.get("horizon")),
        "source": rel(source_path),
        "units": {
            "height": "m",
            "p": "Pa",
            "t": "K",
            "qv": "kg kg-1",
            "u": "m s-1",
            "v": "m s-1",
            "temperature_c": "degC",
            "pressure_hpa": "hPa",
            "dewpoint_c": "degC",
            "wind_speed_ms": "m s-1",
            "wind_dir_deg": "degrees_from",
        },
        "profile": profile,
        "derived": derived,
    }
    if surface:
        payload["surface"] = surface

    write_json(output_path, payload)
    return {
        "step": source_path.stem,
        "url": rel(output_path),
        "valid_time": attrs.get("valid_time"),
        "horizon": clean_number(attrs.get("horizon")),
    }


def export_thermal_panel(
    model_key: str,
    run_tag: str,
    location_id: str,
    location_meta: dict[str, Any],
    source_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    with xr.open_dataset(source_path) as ds:
        ds.load()
        heights = array_to_list(ds["height"].values, 0) if "height" in ds.coords else []
        horizons = array_to_list(ds["horizon_label"].values) if "horizon_label" in ds.coords else []
        valid_times = array_to_list(ds["valid_time"].values) if "valid_time" in ds.coords else []
        w_values = array_to_list(ds["w"].values, 2) if "w" in ds else []
        z_lcl = array_to_list(ds["z_lcl"].values, 0) if "z_lcl" in ds else []
        z_top = array_to_list(ds["z_top"].values, 0) if "z_top" in ds else []
        q_h = array_to_list(ds["Q_H"].values, 1) if "Q_H" in ds else []
        active = array_to_list(ds["active"].values) if "active" in ds else []

        if "w" in ds:
            w_arr = np.asarray(ds["w"].values, dtype=float)
            max_w_by_horizon = np.nanmax(w_arr, axis=1)
            max_w = float(np.nanmax(w_arr)) if np.isfinite(w_arr).any() else None
        else:
            max_w_by_horizon = np.asarray([])
            max_w = None

    summary = {
        "max_w_ms": clean_number(max_w, 2) if max_w is not None else None,
        "max_w_by_horizon_ms": array_to_list(max_w_by_horizon, 2),
        "active_count": sum(1 for item in active if item is True),
    }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "product": "thermal_panel",
        "model": model_key,
        "run": run_tag,
        "location": location_payload(location_id, location_meta),
        "source": rel(source_path),
        "units": {
            "height_m": "m",
            "w_ms": "m s-1",
            "z_lcl_m": "m",
            "z_top_m": "m",
            "sensible_heat_flux_wm2": "W m-2",
        },
        "horizons": horizons,
        "valid_times": valid_times,
        "height_m": heights,
        "w_ms": w_values,
        "z_lcl_m": z_lcl,
        "z_top_m": z_top,
        "sensible_heat_flux_wm2": q_h,
        "active": active,
        "summary": summary,
    }

    write_json(output_path, payload)
    return {
        "url": rel(output_path),
        "source": rel(source_path),
        "horizons": horizons,
        "valid_times": valid_times,
        "summary": summary,
    }


def write_region_forecast(
    model_key: str,
    run_tag: str,
    location_id: str,
    location_meta: dict[str, Any],
    profile_exports: list[dict[str, Any]],
    thermal_export: dict[str, Any] | None,
) -> dict[str, Any]:
    path = WEB_DIR / "region_forecasts" / model_key / run_tag / f"{location_id}.json"
    steps = [item["step"] for item in profile_exports]
    valid_times = [item.get("valid_time") for item in profile_exports]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "product": "region_forecast",
        "model": model_key,
        "run": run_tag,
        "location": location_payload(location_id, location_meta),
        "steps": steps,
        "valid_times": valid_times,
        "products": {
            "emagrams": {item["step"]: item["url"] for item in profile_exports},
            "thermal_panel": thermal_export["url"] if thermal_export else None,
        },
        "summary": {
            "thermal": thermal_export.get("summary") if thermal_export else None,
        },
    }
    write_json(path, payload)
    return {"url": rel(path), "steps": steps, "valid_times": valid_times}


def export_model(model: dict[str, Any], locations: dict[str, Any]) -> dict[str, Any]:
    model_key = model["key"]
    cache_dir = model["cache_dir"]
    scanned_runs = scan_profiles(cache_dir, locations)
    model_manifest: dict[str, Any] = {
        "label": model["label"],
        "cache_dir": str(cache_dir),
        "latest_run": max(scanned_runs.keys()) if scanned_runs else None,
        "runs": {},
        "counts": {
            "runs": len(scanned_runs),
            "locations": 0,
            "profiles": 0,
            "thermal_panels": 0,
            "region_forecasts": 0,
        },
    }

    seen_locations: set[str] = set()

    for run_tag, run_locations in scanned_runs.items():
        run_manifest = {"locations": {}}
        for location_id, step_files in run_locations.items():
            location_meta = locations[location_id]
            profile_exports: list[dict[str, Any]] = []
            for profile_path in step_files:
                out_path = (
                    WEB_DIR
                    / "emagrams"
                    / model_key
                    / run_tag
                    / location_id
                    / f"{profile_path.stem}.json"
                )
                try:
                    profile_exports.append(
                        export_profile(model_key, run_tag, location_id, location_meta, profile_path, out_path)
                    )
                except Exception as exc:
                    log(f"WARN profile export failed for {profile_path}: {exc}")

            thermal_export = None
            thermal_path = cache_dir / run_tag / "thermals" / f"{sanitize_name(location_id)}.nc"
            if not thermal_path.exists():
                thermal_path = cache_dir / run_tag / "thermals" / f"{location_id}.nc"
            if thermal_path.exists():
                out_path = WEB_DIR / "thermal_panels" / model_key / run_tag / f"{location_id}.json"
                try:
                    thermal_export = export_thermal_panel(
                        model_key,
                        run_tag,
                        location_id,
                        location_meta,
                        thermal_path,
                        out_path,
                    )
                    model_manifest["counts"]["thermal_panels"] += 1
                except Exception as exc:
                    log(f"WARN thermal export failed for {thermal_path}: {exc}")

            region_forecast = write_region_forecast(
                model_key,
                run_tag,
                location_id,
                location_meta,
                profile_exports,
                thermal_export,
            )

            seen_locations.add(location_id)
            model_manifest["counts"]["profiles"] += len(profile_exports)
            model_manifest["counts"]["region_forecasts"] += 1
            run_manifest["locations"][location_id] = {
                "type": location_meta.get("type", "legacy"),
                "display_name": location_meta.get("display_name", location_id),
                "steps": region_forecast["steps"],
                "valid_times": region_forecast["valid_times"],
                "region_forecast": region_forecast["url"],
                "thermal_panel": thermal_export["url"] if thermal_export else None,
                "emagram_template": rel(WEB_DIR / "emagrams" / model_key / run_tag / location_id / "{step}.json"),
            }

        model_manifest["runs"][run_tag] = run_manifest

    model_manifest["counts"]["locations"] = len(seen_locations)
    return model_manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    profile_count = manifest["counts"]["profiles"]
    source_nc_count = sum(1 for model in MODELS for _ in model["cache_dir"].glob("*/*/H*.nc"))
    if source_nc_count and profile_count == 0:
        raise RuntimeError("NetCDF profile files exist, but no web profile JSON files were written")

    if not (WEB_DIR / "manifest.json").exists():
        raise RuntimeError("web_exports/manifest.json was not written")
    if not (WEB_DIR / "locations.json").exists():
        raise RuntimeError("web_exports/locations.json was not written")


def main() -> None:
    if not LOCATIONS_FILE.exists():
        raise FileNotFoundError("locations.json is required for web export generation")

    locations = load_json(LOCATIONS_FILE)
    source_manifest = load_json(SOURCE_MANIFEST_FILE)
    ensure_clean_web_dir()
    write_json(WEB_DIR / "locations.json", locations, pretty=True)

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "source": {
            "netcdf_manifest": "manifest.json",
            "netcdf_manifest_generated_at": source_manifest.get("generated_at"),
            "data_root": "https://raw.githubusercontent.com/sebosimo/XCBenz_Data/data",
        },
        "urls": {
            "locations": "web_exports/locations.json",
            "regions": None,
        },
        "products": {
            "region_forecasts": "web_exports/region_forecasts/{model}/{run}/{location_id}.json",
            "emagrams": "web_exports/emagrams/{model}/{run}/{location_id}/{step}.json",
            "thermal_panels": "web_exports/thermal_panels/{model}/{run}/{location_id}.json",
            "maps": None,
        },
        "models": {},
        "counts": {
            "locations": len(locations),
            "region_locations": sum(1 for item in locations.values() if item.get("type") == "region"),
            "legacy_locations": sum(1 for item in locations.values() if item.get("type") == "legacy"),
            "profiles": 0,
            "thermal_panels": 0,
            "region_forecasts": 0,
        },
        "notes": [
            "Generated from existing NetCDF files; no additional MeteoSwiss downloads are performed.",
            "Wind/WebP map exports are deferred.",
        ],
    }

    for model in MODELS:
        model_manifest = export_model(model, locations)
        manifest["models"][model["key"]] = model_manifest
        manifest["counts"]["profiles"] += model_manifest["counts"]["profiles"]
        manifest["counts"]["thermal_panels"] += model_manifest["counts"]["thermal_panels"]
        manifest["counts"]["region_forecasts"] += model_manifest["counts"]["region_forecasts"]

    write_json(WEB_DIR / "manifest.json", manifest, pretty=True)
    validate_manifest(manifest)
    log(
        "Wrote web_exports: "
        f"{manifest['counts']['profiles']} profiles, "
        f"{manifest['counts']['thermal_panels']} thermal panels, "
        f"{manifest['counts']['region_forecasts']} region forecasts"
    )


if __name__ == "__main__":
    main()

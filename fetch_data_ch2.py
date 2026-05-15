import os, sys
import datetime, json, xarray as xr
import numpy as np
import warnings
import requests
import time
import shutil

# Set GRIB definitions for COSMO/ICON (same as fetch_data.py)
COSMO_DEFS = r"C:\Users\sebas\.conda\envs\weather_final\share\eccodes-cosmo-resources\definitions"
STANDARD_DEFS = os.path.join(sys.prefix, "Library", "share", "eccodes", "definitions")

defs_to_use = []
if os.path.exists(COSMO_DEFS):
    defs_to_use.append(COSMO_DEFS)
if os.path.exists(STANDARD_DEFS):
    defs_to_use.append(STANDARD_DEFS)

if defs_to_use:
    final_def_path = ":".join(defs_to_use)
    os.environ["GRIB_DEFINITION_PATH"] = final_def_path
    os.environ["ECCODES_DEFINITION_PATH"] = final_def_path

from meteodatalab import ogd_api
from wind_maps import (
    CACHE_DIR_WIND_PACKED,
    WindMapAccumulator,
    cleanup_old_wind_runs,
    is_wind_maps_enabled,
    load_config as load_wind_map_config,
)

warnings.filterwarnings("ignore")

# --- Configuration ---
COLLECTION_CH2   = "ogd-forecasting-icon-ch2"
HHL_FILENAME     = "vertical_constants_icon-ch2-eps.grib2"
HGRID_FILENAME   = "horizontal_constants_icon-ch2-eps.grib2"
CACHE_DIR        = "cache_data_ch2"
CACHE_DIR_PACKED = "cache_data_ch2_packed"
CACHE_DIR_MAPS_PACKED = CACHE_DIR_WIND_PACKED
STATIC_DIR       = "static_data"
MAX_HORIZON      = 120   # H000–H120, full 5-day forecast
VARS             = ["T", "U", "V", "P", "QV"]
VARS_NATIVE_10M_WIND = ["U_10M", "V_10M"]
VARS_RADIATION   = ["ASWDIR_S", "ASWDIFD_S"]   # surface SW radiation (time-accumulated means)
NETCDF_ENGINE    = "netcdf4"
NETCDF_COMPRESS_KW = {"zlib": True, "shuffle": True, "complevel": 4}

STAC_BASE_URL  = "https://data.geo.admin.ch/api/stac/v1/collections/ch.meteoschweiz.ogd-forecasting-icon-ch2"
STAC_ASSETS_URL = f"{STAC_BASE_URL}/assets"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CACHE_DIR_PACKED, exist_ok=True)
os.makedirs(CACHE_DIR_MAPS_PACKED, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)


def compressed_encoding(ds):
    return {name: dict(NETCDF_COMPRESS_KW) for name in ds.data_vars}

def packed_encoding(ds):
    encoding = {}
    for name, data_array in ds.data_vars.items():
        if data_array.dtype.kind in ("U", "S", "O"):
            encoding[name] = {}
        else:
            enc = dict(NETCDF_COMPRESS_KW)
            if name == "horizon":
                enc["dtype"] = "i2"
            elif name == "valid_time_epoch":
                enc["dtype"] = "i8"
            elif np.issubdtype(data_array.dtype, np.floating):
                enc["dtype"] = "f4"
            encoding[name] = enc
    return encoding


def get_iso_horizon(total_hours):
    days = total_hours // 24
    hours = total_hours % 24
    return f"P{days}DT{hours}H"


def sanitize_name(name):
    n = name.replace("ü", "ue").replace("ö", "oe").replace("ä", "ae") \
            .replace("Ü", "Ue").replace("Ö", "Oe").replace("Ä", "Ae").replace("ß", "ss")
    clean = "".join(c for c in n if c.isalnum() or c in ('-', '_'))
    return clean if clean else "unnamed"

def _step_number(step_label):
    return int(step_label.replace("H", ""))


def env_flag(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def log(msg, level="INFO"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("debug_log_ch2.txt", "a") as f:
        f.write(f"{timestamp} [{level}] {msg}\n")
    print(f"{timestamp} [{level}] {msg}", flush=True)


def download_file(url, target_path, max_retries=3, max_seconds=90):
    """Download url to target_path. Aborts if total download exceeds max_seconds."""
    for attempt in range(max_retries):
        try:
            log(f"Downloading {url} to {target_path}...")
            deadline = time.time() + max_seconds
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(target_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if time.time() > deadline:
                            raise TimeoutError(f"Download exceeded {max_seconds}s — aborting")
                        f.write(chunk)
            log(f"Download complete: {target_path}")
            return True
        except Exception as e:
            log(f"Download attempt {attempt + 1} failed: {e}", "ERROR")
            if os.path.exists(target_path):
                os.remove(target_path)  # clean up partial file before retry
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return False


def download_static_files():
    """Download CH2 HHL and HGRID static files once per CI runner."""
    for filename in [HHL_FILENAME, HGRID_FILENAME]:
        path = os.path.join(STATIC_DIR, filename)
        if os.path.exists(path):
            log(f"Static file already present: {filename}")
            continue
        log(f"Downloading CH2 static file {filename}...")
        try:
            resp = requests.get(STAC_ASSETS_URL, timeout=15)
            resp.raise_for_status()
            assets = resp.json().get("assets", {})
            # Assets may be a dict keyed by ID or a list
            if isinstance(assets, dict):
                url = assets.get(filename, {}).get("href")
            else:
                url = next((a["href"] for a in assets if a.get("id") == filename), None)
            if url:
                download_file(url, path)
            else:
                log(f"Asset URL not found for {filename}", "ERROR")
        except Exception as e:
            log(f"Failed to fetch static {filename}: {e}", "ERROR")


def load_static_hhl():
    path = os.path.join(STATIC_DIR, HHL_FILENAME)
    if not os.path.exists(path):
        return None
    try:
        ds = xr.open_dataset(path, engine='cfgrib', backend_kwargs={'indexpath': ''})
        var = next((v for v in ds.data_vars if v.lower() in ['h', 'hhl']), list(ds.data_vars)[0])
        hhl = ds[var].load()
        ds.close()
        return hhl
    except Exception as e:
        log(f"Error loading CH2 HHL: {e}", "ERROR")
        return None


def load_static_grid():
    path = os.path.join(STATIC_DIR, HGRID_FILENAME)
    if not os.path.exists(path):
        return None
    try:
        ds = xr.open_dataset(path, engine='cfgrib', backend_kwargs={'indexpath': ''})
        grid = {}
        for key in ['lat', 'lon']:
            match_k = next((k for k in list(ds.coords) + list(ds.data_vars) if key in k.lower()), None)
            if match_k:
                grid[key] = ds[match_k].load()
            else:
                grid[key] = None
        ds.close()
        return grid if grid.get('lat') is not None else None
    except Exception as e:
        log(f"Error loading CH2 HGRID: {e}", "ERROR")
        return None


def get_latest_available_runs(limit=2):
    """Discover available CH2 runs via active probing (6-hourly: 00Z/06Z/12Z/18Z)."""
    log("Discovering CH2 runs via Active Probing...")
    now = datetime.datetime.now(datetime.timezone.utc)
    # Round down to nearest 6-hour slot
    hour = (now.hour // 6) * 6
    start = now.replace(hour=hour, minute=0, second=0, microsecond=0)

    found_runs = []
    for i in range(20):  # Check last 120 hours (20 × 6h slots) to account for 3.5h pub delay
        cand = start - datetime.timedelta(hours=i * 6)
        ref = cand.strftime("%Y-%m-%dT%H:%M:%SZ")

        params = {"limit": 1, "forecast:reference_datetime": ref}
        try:
            r = requests.get(f"{STAC_BASE_URL}/items", params=params, timeout=15)
            if r.status_code == 200 and r.json().get("features"):
                found_runs.append(cand)
                log(f"Found available CH2 run: {ref}")
                if len(found_runs) >= limit:
                    break
        except Exception:
            pass
    return found_runs


def is_run_complete_locally(time_tag, locations, max_h):
    """Check if the last expected file exists — implies full run was downloaded."""
    last_loc = sanitize_name(list(locations.keys())[-1])
    trace_path = os.path.join(CACHE_DIR, time_tag, last_loc, f"H{max_h:03d}.nc")
    return os.path.exists(trace_path)


def is_horizon_complete_locally(time_tag, locations, h):
    """Check if all location .nc files for a given horizon already exist."""
    for name in locations:
        path = os.path.join(CACHE_DIR, time_tag, sanitize_name(name), f"H{h:03d}.nc")
        if not os.path.exists(path):
            return False
    return True

def is_packed_run_complete_locally(time_tag, locations):
    return all(
        os.path.exists(os.path.join(CACHE_DIR_PACKED, time_tag, f"{sanitize_name(name)}.nc"))
        for name in locations
    )

def build_packed_dataset_from_hourlies(tag, safe_name, location_name, location_meta, step_labels):
    level_height = None
    profile_columns = {var: [] for var in VARS}
    radiation_columns = {var: [] for var in VARS_RADIATION}
    valid_time_epochs = []
    available_steps = []
    ref_time = None

    for step_label in step_labels:
        step_path = os.path.join(CACHE_DIR, tag, safe_name, f"{step_label}.nc")
        if not os.path.exists(step_path):
            continue

        with xr.open_dataset(step_path, engine=NETCDF_ENGINE) as ds_in:
            ds_loaded = ds_in.load()

        if ref_time is None:
            ref_time = datetime.datetime.fromisoformat(str(ds_loaded.attrs["ref_time"]))

        height = ds_loaded["HEIGHT"].values.astype(np.float32)
        if level_height is None:
            level_height = height

        for var in VARS:
            profile_columns[var].append(ds_loaded[var].values.astype(np.float32))

        for var in VARS_RADIATION:
            if var in ds_loaded:
                radiation_columns[var].append(np.float32(ds_loaded[var].values))
            else:
                radiation_columns[var].append(np.float32(np.nan))

        valid_attr = ds_loaded.attrs.get("valid_time")
        if valid_attr:
            valid_time = datetime.datetime.fromisoformat(str(valid_attr))
        else:
            valid_time = ref_time + datetime.timedelta(hours=_step_number(step_label))
        if valid_time.tzinfo is None:
            valid_time = valid_time.replace(tzinfo=datetime.timezone.utc)
        valid_time_epochs.append(int(valid_time.timestamp()))
        available_steps.append(step_label)

    if not available_steps or level_height is None or ref_time is None:
        return None

    packed_vars = {
        "horizon": xr.DataArray(
            np.asarray([_step_number(step) for step in available_steps], dtype=np.int16),
            dims=["time"],
        ),
        "valid_time_epoch": xr.DataArray(
            np.asarray(valid_time_epochs, dtype=np.int64),
            dims=["time"],
        ),
        "step_label": xr.DataArray(np.asarray(available_steps), dims=["time"]),
        "height": xr.DataArray(level_height, dims=["level"]),
    }

    for var, columns in profile_columns.items():
        if columns:
            packed_vars[var] = xr.DataArray(np.stack(columns).astype(np.float32), dims=["time", "level"])

    for var, values in radiation_columns.items():
        if values and not np.all(np.isnan(values)):
            packed_vars[var] = xr.DataArray(np.asarray(values, dtype=np.float32), dims=["time"])

    packed_ds = xr.Dataset(packed_vars)
    packed_ds.attrs = {
        "location": location_meta.get("display_name", location_name),
        "location_id": location_name,
        "display_name": location_meta.get("display_name", location_name),
        "point_type": location_meta.get("type", "legacy"),
        "region_name": location_meta.get("region_name") or "",
        "latitude": float(location_meta["lat"]),
        "longitude": float(location_meta["lon"]),
        "ref_time": ref_time.isoformat(),
        "model": "icon-ch2",
        "step_label_pad": 3,
        "schema_version": 1,
    }
    return packed_ds

def write_packed_run_files(tag, locations):
    run_dir = os.path.join(CACHE_DIR, tag)
    if not os.path.isdir(run_dir):
        return

    packed_run_dir = os.path.join(CACHE_DIR_PACKED, tag)
    os.makedirs(packed_run_dir, exist_ok=True)

    for location_name, location_meta in locations.items():
        safe_name = sanitize_name(location_name)
        step_dir = os.path.join(run_dir, safe_name)
        if not os.path.isdir(step_dir):
            continue
        step_labels = sorted(
            f.replace(".nc", "")
            for f in os.listdir(step_dir)
            if f.endswith(".nc") and f.startswith("H")
        )
        packed_ds = build_packed_dataset_from_hourlies(tag, safe_name, location_name, location_meta, step_labels)
        if packed_ds is None:
            continue

        packed_path = os.path.join(packed_run_dir, f"{safe_name}.nc")
        tmp_path = packed_path + ".tmp"
        packed_ds.to_netcdf(
            tmp_path,
            engine=NETCDF_ENGINE,
            format="NETCDF4",
            encoding=packed_encoding(packed_ds),
        )
        os.replace(tmp_path, packed_path)
        log(f"Packed CH2 location written: {tag}/{safe_name}.nc")


def process_traces(fields, locations, tag, h, ref, rad_scalars=None):
    """Extract point profiles for all locations and save as .nc files (3-digit horizon)."""
    sample = list(fields.values())[0]
    lat_n = 'latitude' if 'latitude' in sample.coords else 'lat'
    lon_n = 'longitude' if 'longitude' in sample.coords else 'lon'
    lats, lons = sample[lat_n].values, sample[lon_n].values
    indices = {n: int(np.argmin((lats - c['lat']) ** 2 + (lons - c['lon']) ** 2))
               for n, c in locations.items()}

    for name, idx in indices.items():
        safe_name = sanitize_name(name)
        loc_dir = os.path.join(CACHE_DIR, tag, safe_name)
        os.makedirs(loc_dir, exist_ok=True)
        filename = f"H{h:03d}.nc"   # 3-digit padding for H000–H120
        path = os.path.join(loc_dir, filename)

        if os.path.exists(path):
            continue

        loc_vars = {}

        # Compute full-level heights from HHL half-levels if available
        if "HHL" in fields:
            ds_hhl = fields["HHL"]
            s_dim = ds_hhl[lat_n].dims[0]
            hhl_profile = ds_hhl.squeeze().isel({s_dim: idx}).compute()
            h_vals = hhl_profile.values
            height_centers = (h_vals[:-1] + h_vals[1:]) / 2.0
            loc_vars["HEIGHT"] = xr.DataArray(height_centers, dims=["level"])

        for var, ds in fields.items():
            if var == "HHL":
                continue
            s_dim = ds[lat_n].dims[0]
            profile = ds.squeeze().isel({s_dim: idx}).compute()
            if profile.dims:
                profile = profile.rename({profile.dims[0]: 'level'})
            loc_vars[var] = profile.drop_vars([c for c in profile.coords if c not in profile.dims])

        # Store de-accumulated radiation scalars (if provided)
        if rad_scalars:
            for var, val in rad_scalars.items():
                loc_vars[var] = xr.DataArray(float(val), attrs={"units": "W m-2"})

        ds_out = xr.Dataset(loc_vars)
        valid_time = ref + datetime.timedelta(hours=h)
        coords = locations[name]
        ds_out.attrs = {
            "location": coords.get("display_name", name),
            "location_id": name,
            "display_name": coords.get("display_name", name),
            "point_type": coords.get("type", "legacy"),
            "region_name": coords.get("region_name") or "",
            "latitude": float(coords["lat"]),
            "longitude": float(coords["lon"]),
            "ref_time": ref.isoformat(),
            "horizon": h,
            "valid_time": valid_time.isoformat(),
            "model": "icon-ch2",
        }
        ds_out.to_netcdf(
            path,
            engine=NETCDF_ENGINE,
            format="NETCDF4",
            encoding=compressed_encoding(ds_out),
        )


def cleanup_old_runs():
    """Keep top-2 most recent CH2 runs + the 00Z anchor run from today/yesterday."""
    now = datetime.datetime.now(datetime.timezone.utc)
    keep_dates = {now.date(), (now - datetime.timedelta(days=1)).date()}

    for cache_dir in [CACHE_DIR, CACHE_DIR_PACKED]:
        if not os.path.exists(cache_dir):
            continue

        all_runs = sorted(
            [item for item in os.listdir(cache_dir) if os.path.isdir(os.path.join(cache_dir, item))],
            reverse=True
        )
        keep_recent = set(all_runs[:2])

        for item in all_runs:
            if item in keep_recent:
                continue
            path = os.path.join(cache_dir, item)
            try:
                dt = datetime.datetime.strptime(item, "%Y%m%d_%H%M").replace(
                    tzinfo=datetime.timezone.utc)
            except ValueError:
                continue
            if dt.hour == 0 and dt.minute == 0 and dt.date() in keep_dates:
                continue
            try:
                shutil.rmtree(path)
                log(f"CH2 cleanup: removed {cache_dir}/{item}")
            except Exception as e:
                log(f"CH2 cleanup failed {cache_dir}/{item}: {e}", "ERROR")


def main():
    log("=== CH2 Data Fetcher Start ===")
    force_refresh = env_flag("FORCE_REFRESH", default=False)
    if force_refresh:
        log("FORCE_REFRESH enabled: existing CH2 run/horizon-complete checks will be ignored.", "NOTICE")

    wind_config = None
    wind_enabled = is_wind_maps_enabled("ch2")
    if wind_enabled:
        try:
            wind_config = load_wind_map_config(log=log)
            log("CH2 wind-map generation enabled for this run.", "NOTICE")
        except Exception as e:
            wind_enabled = False
            log(f"CH2 wind-map generation disabled: {e}", "WARNING")
    else:
        log("CH2 wind-map generation disabled by flags.")

    download_static_files()

    if not os.path.exists("locations.json"):
        log("locations.json not found.", "ERROR")
        return
    with open("locations.json", "r", encoding="utf-8") as f:
        locations = json.load(f)

    runs = get_latest_available_runs(limit=2)
    if not runs:
        log("No CH2 runs found.")
        return

    hhl = load_static_hhl()
    grid = load_static_grid()

    # Inject lat/lon coordinates into HHL so process_traces can locate grid points
    if hhl is not None and grid is not None:
        n_grid = grid['lat'].size
        match_dim = next((d for d in hhl.dims if hhl.sizes[d] == n_grid), None)
        if match_dim:
            hhl = hhl.assign_coords({
                "latitude": (match_dim, grid['lat'].values),
                "longitude": (match_dim, grid['lon'].values)
            })

    for ref_time in runs:
        tag = ref_time.strftime('%Y%m%d_%H%M')
        if not force_refresh and is_run_complete_locally(tag, locations, MAX_HORIZON):
            if not is_packed_run_complete_locally(tag, locations):
                write_packed_run_files(tag, locations)
            log(f"CH2 run {tag} already complete locally — skipping.")
            break

        log(f"Processing CH2 run: {tag} (H000–H{MAX_HORIZON:03d})")
        any_success = False
        wind_accumulator = (
            WindMapAccumulator("ch2", tag, ref_time, wind_config, log=log)
            if wind_enabled and wind_config is not None
            else None
        )
        # Cache previous raw radiation values for de-accumulation (running mean → hourly mean)
        prev_rad_raw = {var: None for var in VARS_RADIATION}

        for h in range(MAX_HORIZON + 1):
            # Skip horizons where all location .nc files are already on disk
            if not force_refresh and is_horizon_complete_locally(tag, locations, h):
                any_success = True   # count existing horizons toward run completion
                continue

            iso_h = get_iso_horizon(h)

            fields = {"HHL": hhl} if hhl is not None else {}
            has_new_data = False

            for var in [*VARS, *VARS_NATIVE_10M_WIND]:
                try:
                    req = ogd_api.Request(
                        collection=COLLECTION_CH2,
                        variable=var,
                        reference_datetime=ref_time,
                        horizon=iso_h,
                        perturbed=False
                    )
                    urls = ogd_api.get_asset_urls(req)
                    if urls:
                        tmp = f"temp_ch2_{var}_{tag}_{h:03d}.grib2"
                        if download_file(urls[0], tmp):
                            ds = xr.open_dataset(tmp, engine='cfgrib',
                                                 backend_kwargs={'indexpath': ''})
                            data = ds[next(iter(ds.data_vars))].load()
                            if grid:
                                m_dim = next(d for d in data.dims
                                             if data.sizes[d] == grid['lat'].size)
                                data = data.assign_coords({
                                    "latitude": (m_dim, grid['lat'].values),
                                    "longitude": (m_dim, grid['lon'].values)
                                })
                            fields[var] = data
                            ds.close()
                            os.remove(tmp)
                            has_new_data = True
                except Exception:
                    pass

            # --- Radiation fetch and de-accumulation (same logic as fetch_data.py) ---
            rad_scalars = {}
            sample_field = next((v for v in fields.values()
                                  if v is not None and hasattr(v, 'dims')), None)
            if h > 0 and sample_field is not None:
                lat_n = 'latitude' if 'latitude' in sample_field.coords else 'lat'
                lon_n = 'longitude' if 'longitude' in sample_field.coords else 'lon'
                lats_f = sample_field[lat_n].values
                lons_f = sample_field[lon_n].values
                for var in VARS_RADIATION:
                    try:
                        req = ogd_api.Request(
                            collection=COLLECTION_CH2,
                            variable=var,
                            reference_datetime=ref_time,
                            horizon=iso_h,
                            perturbed=False,
                        )
                        urls = ogd_api.get_asset_urls(req)
                        if urls:
                            tmp = f"temp_ch2_{var}_{tag}_{h:03d}.grib2"
                            if download_file(urls[0], tmp):
                                ds_r = xr.open_dataset(tmp, engine='cfgrib',
                                                       backend_kwargs={'indexpath': ''})
                                raw_data = ds_r[next(iter(ds_r.data_vars))].load().values.ravel()
                                ds_r.close()
                                os.remove(tmp)
                                prev_raw_arr = prev_rad_raw[var]
                                if prev_raw_arr is None:
                                    prev_raw_arr = np.zeros_like(raw_data)
                                deacc_arr = h * raw_data - (h - 1) * prev_raw_arr
                                deacc_arr = np.maximum(deacc_arr, 0.0)
                                prev_rad_raw[var] = raw_data
                                rad_scalars[var] = deacc_arr
                    except Exception as e:
                        log(f"CH2 radiation fetch failed for {var} H+{h:03d}: {e}", "WARNING")

            if has_new_data:
                if rad_scalars and sample_field is not None:
                    # Build per-location radiation scalars
                    for name, coords in locations.items():
                        idx_loc = int(np.argmin(
                            (lats_f - coords['lat'])**2 + (lons_f - coords['lon'])**2
                        ))
                        loc_rad = {v: float(arr.ravel()[idx_loc]) for v, arr in rad_scalars.items()}
                        process_traces(fields, {name: coords}, tag, h, ref_time,
                                       rad_scalars=loc_rad)
                    # Locations without radiation (fallback — none expected)
                else:
                    process_traces(fields, locations, tag, h, ref_time)
                if wind_accumulator is not None:
                    wind_accumulator.append(fields, h, ref_time)
                log(f"CH2 H+{h:03d} done")
                any_success = True

        if any_success:
            log(f"CH2 run {tag} complete.", "NOTICE")
            if wind_accumulator is not None:
                wind_accumulator.finalize()
            write_packed_run_files(tag, locations)
            # Compute thermal forecasts for the newly fetched CH2 run
            try:
                import compute_thermals
                log(f"Computing CH2 thermals for {tag}…")
                compute_thermals.process_run(tag, CACHE_DIR)
                log(f"CH2 thermals complete for {tag}.", "NOTICE")
            except Exception as e:
                log(f"CH2 thermal computation failed for {tag}: {e}", "WARNING")
            break
        else:
            log(f"CH2 run {tag} yielded no data, trying next available run...")
            p = os.path.join(CACHE_DIR, tag)
            if os.path.exists(p) and not os.listdir(p):
                try:
                    shutil.rmtree(p)
                except Exception:
                    pass

    cleanup_old_runs()
    cleanup_old_wind_runs("ch2", anchor_hour=0, log=log)
    log("=== CH2 Data Fetcher Complete ===", "NOTICE")


if __name__ == "__main__":
    main()

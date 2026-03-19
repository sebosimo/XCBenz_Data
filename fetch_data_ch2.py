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

warnings.filterwarnings("ignore")

# --- Configuration ---
COLLECTION_CH2   = "ogd-forecasting-icon-ch2"
HHL_FILENAME     = "vertical_constants_icon-ch2-eps.grib2"
HGRID_FILENAME   = "horizontal_constants_icon-ch2-eps.grib2"
CACHE_DIR        = "cache_data_ch2"
STATIC_DIR       = "static_data"
MAX_HORIZON      = 120   # H000–H120, full 5-day forecast
VARS             = ["T", "U", "V", "P", "QV"]
VARS_RADIATION   = ["ASWDIR_S", "ASWDIFD_S"]   # surface SW radiation (time-accumulated means)

STAC_BASE_URL  = "https://data.geo.admin.ch/api/stac/v1/collections/ch.meteoschweiz.ogd-forecasting-icon-ch2"
STAC_ASSETS_URL = f"{STAC_BASE_URL}/assets"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)


def get_iso_horizon(total_hours):
    days = total_hours // 24
    hours = total_hours % 24
    return f"P{days}DT{hours}H"


def sanitize_name(name):
    n = name.replace("ü", "ue").replace("ö", "oe").replace("ä", "ae") \
            .replace("Ü", "Ue").replace("Ö", "Oe").replace("Ä", "Ae").replace("ß", "ss")
    clean = "".join(c for c in n if c.isalnum() or c in ('-', '_'))
    return clean if clean else "unnamed"


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
        ds_out.attrs = {
            "location": name,
            "ref_time": ref.isoformat(),
            "horizon": h,
            "valid_time": valid_time.isoformat(),
            "model": "icon-ch2",
        }
        ds_out.to_netcdf(path)


def cleanup_old_runs():
    """Keep top-2 most recent CH2 runs + the 00Z anchor run from today/yesterday."""
    now = datetime.datetime.now(datetime.timezone.utc)
    keep_dates = {now.date(), (now - datetime.timedelta(days=1)).date()}

    if not os.path.exists(CACHE_DIR):
        return

    all_runs = sorted(
        [item for item in os.listdir(CACHE_DIR) if os.path.isdir(os.path.join(CACHE_DIR, item))],
        reverse=True  # newest first
    )
    keep_recent = set(all_runs[:2])

    for item in all_runs:
        if item in keep_recent:
            continue
        path = os.path.join(CACHE_DIR, item)
        try:
            dt = datetime.datetime.strptime(item, "%Y%m%d_%H%M").replace(
                tzinfo=datetime.timezone.utc)
        except ValueError:
            continue
        # Keep 00Z anchor run from today or yesterday (longest coverage for the day)
        if dt.hour == 0 and dt.minute == 0 and dt.date() in keep_dates:
            continue
        try:
            shutil.rmtree(path)
            log(f"CH2 cleanup: removed {item}")
        except Exception as e:
            log(f"CH2 cleanup failed {item}: {e}", "ERROR")


def main():
    log("=== CH2 Data Fetcher Start ===")

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
        if is_run_complete_locally(tag, locations, MAX_HORIZON):
            log(f"CH2 run {tag} already complete locally — skipping.")
            break

        log(f"Processing CH2 run: {tag} (H000–H{MAX_HORIZON:03d})")
        any_success = False
        # Cache previous raw radiation values for de-accumulation (running mean → hourly mean)
        prev_rad_raw = {var: None for var in VARS_RADIATION}

        for h in range(MAX_HORIZON + 1):
            # Skip horizons where all location .nc files are already on disk
            if is_horizon_complete_locally(tag, locations, h):
                any_success = True   # count existing horizons toward run completion
                continue

            iso_h = get_iso_horizon(h)

            fields = {"HHL": hhl} if hhl is not None else {}
            has_new_data = False

            for var in VARS:
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
                log(f"CH2 H+{h:03d} done")
                any_success = True

        if any_success:
            log(f"CH2 run {tag} complete.", "NOTICE")
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
    log("=== CH2 Data Fetcher Complete ===", "NOTICE")


if __name__ == "__main__":
    main()

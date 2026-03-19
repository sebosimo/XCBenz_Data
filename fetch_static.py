import os
import requests
import sys

STATIC_DIR = "static_data"
HHL_FILENAME = "vertical_constants_icon-ch1-eps.grib2"
HGRID_FILENAME = "horizontal_constants_icon-ch1-eps.grib2"
STAC_ASSETS_URL = "https://data.geo.admin.ch/api/stac/v1/collections/ch.meteoschweiz.ogd-forecasting-icon-ch1/assets"

def download_static_files():
    print(f"Ensuring {STATIC_DIR} exists...", flush=True)
    os.makedirs(STATIC_DIR, exist_ok=True)
    
    # HHL
    hhl_path = os.path.join(STATIC_DIR, HHL_FILENAME)
    if not os.path.exists(hhl_path):
        print(f"Downloading HHL to {hhl_path}...", flush=True)
        try:
            resp = requests.get(STAC_ASSETS_URL)
            resp.raise_for_status()
            assets = resp.json()["assets"]
            
            url = None
            for asset in assets:
                if asset.get("id") == HHL_FILENAME:
                    url = asset.get("href")
                    break
            
            if url:
                print(f"Fetching {url}...", flush=True)
                with requests.get(url, stream=True) as r:
                    r.raise_for_status()
                    with open(hhl_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                print("HHL Downloaded.", flush=True)
            else:
                print("HHL URL not found.", flush=True)
        except Exception as e:
            print(f"HHL Download Failed: {e}", flush=True)
    else:
        print("HHL already exists.", flush=True)

    # HGRID
    hgrid_path = os.path.join(STATIC_DIR, HGRID_FILENAME)
    if not os.path.exists(hgrid_path):
        print(f"Downloading HGRID to {hgrid_path}...", flush=True)
        try:
            resp = requests.get(STAC_ASSETS_URL)
            resp.raise_for_status()
            assets = resp.json()["assets"]
            
            url = None
            for asset in assets:
                if asset.get("id") == HGRID_FILENAME:
                    url = asset.get("href")
                    break
            
            if url:
                print(f"Fetching {url}...", flush=True)
                with requests.get(url, stream=True) as r:
                    r.raise_for_status()
                    with open(hgrid_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                print("HGRID Downloaded.", flush=True)
            else:
                print("HGRID URL not found.", flush=True)
        except Exception as e:
            print(f"HGRID Download Failed: {e}", flush=True)
    else:
        print("HGRID already exists.", flush=True)

if __name__ == "__main__":
    download_static_files()

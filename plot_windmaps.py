import xarray as xr
import matplotlib.pyplot as plt
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
# import geocat.viz as gv
import os
import numpy as np

CACHE_DIR_MAPS = "cache_wind"
ARTIFACT_DIR = "map_artifacts"

def plot_timestep(nc_path, time_tag, h_str):
    if not os.path.exists(nc_path): return
    
    ds = xr.open_dataset(nc_path)
    
    # Loop over levels found in file
    # Format: u_{name}, v_{name}
    
    vars_u = [v for v in ds.data_vars if v.startswith("u_")]
    
    for u_name in vars_u:
        level_name = u_name.replace("u_", "") # e.g. "10m_AGL"
        v_name = f"v_{level_name}"
        
        if v_name not in ds: continue
        
        u = ds[u_name]
        v = ds[v_name]
        
        # Calculate speed
        speed = np.sqrt(u**2 + v**2)
        
        # Plotting
        # Native grid is unstructured? 
        # Note: MetPy interpolation returns data on the same spatial coords.
        # If input was unstructured (ncells), output is (ncells).
        # We need to grid it for streamplot?
        # Or use Tricontourf?
        
        # Check dims
        dims = u.dims
        if 'ncells' in dims or (len(dims)==1 and 'lat' not in dims):
             # Unstructured
             # Use triangulation
             lat = u.latitude if hasattr(u, 'latitude') else u.lat
             lon = u.longitude if hasattr(u, 'longitude') else u.lon
             is_unstructured = True
        else:
             is_unstructured = False
             lat = u.lat
             lon = u.lon

        fig = plt.figure(figsize=(10, 8))
        fig = plt.figure(figsize=(10, 8))
        
        if HAS_CARTOPY:
            ax = plt.axes(projection=ccrs.PlateCarree())
            # Add features
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            ax.add_feature(cfeature.LAKES, alpha=0.5)
            # Set extent (Switzerland approx)
            ax.set_extent([5.5, 11.0, 45.5, 48.0], crs=ccrs.PlateCarree())
            transform_arg = {'transform': ccrs.PlateCarree()}
        else:
            ax = plt.axes()
            # Simple aspect ratio fix
            ax.set_aspect('equal')
            ax.set_xlim(5.5, 11.0)
            ax.set_ylim(45.5, 48.0)
            transform_arg = {}
        
        # Plot Speed
        if is_unstructured:
             # Tricontourf
             # Careful with large N. 1M points might be slow to triangulate every time.
             # Ideally we cache triangulation or use scatter for dense points.
             # 1M scatter is large.
             # Re-gridding is better.
             # For now, try tricontourf with fewer levels or scatter map.
             cf = ax.tricontourf(lon, lat, speed, levels=20, cmap='viridis', **transform_arg)
        else:
             cf = ax.contourf(lon, lat, speed, levels=20, cmap='viridis', **transform_arg)
             
        plt.colorbar(cf, ax=ax, orientation='horizontal', label='Wind Speed (m/s)', pad=0.05, shrink=0.8)
        
        # Plot Vectors (Streamlines)
        # Streamplot requires regular grid.
        # Quiver works on unstructured but difficult to control density.
        # We really should regrid for nice Streamlines.
        # ... Skipping Streamlines for Unstructured fallback usage.
        # OR: Use Quiver with stride?
        # Quiver on unstructured requires X,Y,U,V.
        # Stride: u[::100]
        
        stride = 500 # Adjust based on 1M points
        if is_unstructured:
             q = ax.quiver(lon[::stride], lat[::stride], u[::stride], v[::stride], 
                          color='white', width=0.002, scale=200, **transform_arg)
        
        # gv.set_titles_and_labels(ax, 
        #     maintitle=f"Wind Map {level_name}", 
        #     lefttitle=f"Valid: {time_tag} +{h_str}", 
        #     righttitle="ICON-CH1")
        plt.title(f"Wind Map {level_name} - {time_tag} +{h_str}", fontsize=14)
            
        # Save
        s_dir = os.path.join(ARTIFACT_DIR, time_tag)
        os.makedirs(s_dir, exist_ok=True)
        plt.savefig(os.path.join(s_dir, f"map_{level_name}_{h_str}.png"), bbox_inches='tight')
        plt.close(fig)

def main():
    if not os.path.exists(CACHE_DIR_MAPS): return
    
    # Iterate all runs/times
    for tag in os.listdir(CACHE_DIR_MAPS):
        tag_dir = os.path.join(CACHE_DIR_MAPS, tag)
        if not os.path.isdir(tag_dir): continue
        
        for f in os.listdir(tag_dir):
            if f.endswith(".nc"):
                 h_str = f.split("_")[-1].replace(".nc", "") # H00
                 plot_timestep(os.path.join(tag_dir, f), tag, h_str)

if __name__ == "__main__":
    main()

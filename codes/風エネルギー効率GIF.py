import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import imageio
from tqdm import tqdm
import os
from datetime import datetime, timedelta

base_time = datetime(2024, 5, 1, 0, 0)

# === 設定 ===
wind_file = r"D:\Pterodroma\marine_data\surface_wind\wind_L4_20240501_20240525.nc"  # ご自身のパスに変更
output_dir = r"D:\Pterodroma\marine_data\wind_efficiency_maps"
gif_file = r"D:\Pterodroma\marine_data\wind_efficiency.gif"
os.makedirs(output_dir, exist_ok=True)

# === データ読み込み ===
ds = xr.open_dataset(wind_file)
u_wind = ds["eastward_wind"]
v_wind = ds["northward_wind"]
lat = ds["latitude"]
lon = ds["longitude"]

# === 目的地 ===
target_lat = 40.0
target_lon = 170.0

start_lat = -33.767510
start_lon = -80.786563

# === 方位角計算関数 ===
def calculate_bearing(lat1, lon1, lat2, lon2):
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(phi2)
    y = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360

# === メッシュ座標作成（lat, lonを2D化）===
stride = 3
lat_subset = lat[::stride]
lon_subset = lon[::stride]

lat2d, lon2d = np.meshgrid(lat_subset, lon_subset, indexing="ij")

# === 描画とGIF作成 ===
image_paths = []

for t in tqdm(range(0, 576, 6)): 
    u = u_wind[t, ::stride, ::stride].values
    v = v_wind[t, ::stride, ::stride].values
    #u = u_wind[t, :, :].values
    #v = v_wind[t, :, :].values

    target_angle = calculate_bearing(lat2d, lon2d, target_lat, target_lon)
    target_rad = np.radians(target_angle)

    # 目的地方向への風成分（正なら追い風、負なら向かい風）
    efficiency = u * np.sin(target_rad) + v * np.cos(target_rad)

    # === 描画 ===
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))  # 太平洋中心
    ax.set_extent([125, 290, -60, 65], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=10)
    ax.add_feature(cfeature.COASTLINE)
    time_str = (base_time + timedelta(hours=t)).strftime("%Y/%m/%d %H:%M")
    ax.set_title(f"Migration Efficiency Map\n{time_str}", fontsize=14)

    gl = ax.gridlines(draw_labels=True, xlocs=np.arange(-180, 181, 10), ylocs=np.arange(-90, 91, 5))
    gl.right_labels = False
    gl.top_labels = False

    im = ax.pcolormesh(lon2d, lat2d, efficiency, vmin=-15, vmax=15, cmap="PuOr_r", transform=ccrs.PlateCarree(), shading="auto")
    plt.colorbar(im, ax=ax, label="Wind Component to Target (m/s)")

    # 出発地（緑の星マーク）
    ax.plot(start_lon, start_lat, marker='*', color='green', markersize=8,
            transform=ccrs.PlateCarree(), label='Start')

    # 目的地（赤の星マーク）
    ax.plot(target_lon, target_lat, marker='*', color='red', markersize=8,
            transform=ccrs.PlateCarree(), label='Target')

    frame_path = os.path.join(output_dir, f"eff_{t:03d}.png")
    plt.savefig(frame_path, dpi=120, bbox_inches="tight")
    plt.close()
    image_paths.append(frame_path)

# === GIF作成 ===
with imageio.get_writer(gif_file, mode='I', duration=0.5) as writer:
    for path in image_paths:
        writer.append_data(imageio.imread(path))

print(f"✅ GIF saved to: {gif_file}")

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import distance
from geopy import Point

# === データ読み込み ===
ds = xr.open_dataset(r"D:\Pterodroma\marine_data\surface_wind\wind_metopc_ascat_desc_20240501_20240525.nc")
u_wind = ds['eastward_wind']
v_wind = ds['northward_wind']
lat = ds['latitude']
lon = ds['longitude']
time = ds['time']

# === 初期位置（緯度・経度） ===
start_lat = -33.767510
start_lon = -80.786563
trajectory = [(start_lat, start_lon)]

# === 有効な風を探索する関数 ===
def find_nearest_valid_wind(t, i_lat, i_lon, window=3):
    for r in range(1, window + 1):
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                ni, nj = i_lat + di, i_lon + dj
                if (0 <= ni < u_wind.shape[1]) and (0 <= nj < u_wind.shape[2]):
                    u_val = u_wind[t, ni, nj].item()
                    v_val = v_wind[t, ni, nj].item()
                    if np.isfinite(u_val) and np.isfinite(v_val):
                        return u_val, v_val
    return None, None

# === 25日間のシミュレーション ===
for t in range(25):
    current_lat, current_lon = trajectory[-1]

    # 最も近い格子点を特定
    i_lat = np.abs(lat - current_lat).argmin().item()
    i_lon = np.abs(lon - current_lon).argmin().item()

    # 風を取得（または補完）
    u = u_wind[t, i_lat, i_lon].item()
    v = v_wind[t, i_lat, i_lon].item()

    if not np.isfinite(u) or not np.isfinite(v):
        u, v = find_nearest_valid_wind(t, i_lat, i_lon, window=3)
        if u is None or v is None:
            print(f"Day {t+1}: No valid wind nearby at ({current_lat:.2f}, {current_lon:.2f}). Staying.")
            trajectory.append((current_lat, current_lon))
            continue
        else:
            print(f"Day {t+1}: Used nearby wind at ({current_lat:.2f}, {current_lon:.2f})")

    # 移動距離（1日 = 86400秒）
    dx = u * 86400
    dy = v * 86400
    total_speed = np.hypot(dx, dy)

    # 方位角（北基準、時計回り）
    angle_deg = (np.degrees(np.arctan2(dx, dy))) % 360

    # 新しい位置を geopy で計算
    origin = Point(current_lat, current_lon)
    new_point = distance(meters=total_speed).destination(origin, bearing=angle_deg)
    trajectory.append((new_point.latitude, new_point.longitude))

# === 軌跡を描画 ===
lats, lons = zip(*trajectory)
plt.figure(figsize=(10, 6))
plt.plot(lons, lats, marker='o', color='blue')
plt.title("Seabird Drift Trajectory Based on Wind (with NaN Handling)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# =================================================================================================================

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# === 緯度・経度データ ===
lats, lons = zip(*trajectory)

# === 地図描画 ===
fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())  # 通常の緯度経度投影
ax.set_extent([min(lons)-5, max(lons)+5, min(lats)-5, max(lats)+5], crs=ccrs.PlateCarree())

# 海岸線・国境・陸地
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')

# グリッド線（経線10度、緯線5度）
gl = ax.gridlines(draw_labels=True, xlocs=np.arange(-180, 181, 10), ylocs=np.arange(-90, 91, 5))
gl.right_labels = False
gl.top_labels = False

# === 軌跡プロット ===
ax.plot(lons, lats, marker='o', color='blue', linewidth=2, markersize=4, transform=ccrs.PlateCarree())

# 出発点と終点を強調
ax.plot(lons[0], lats[0], marker='*', color='green', markersize=12, label='Start', transform=ccrs.PlateCarree())
ax.plot(lons[-1], lats[-1], marker='X', color='red', markersize=10, label='End', transform=ccrs.PlateCarree())

plt.title("Seabird Drift Trajectory", fontsize=14)
plt.legend()
plt.show()

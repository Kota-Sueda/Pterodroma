import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from geopy.distance import distance
from geopy.distance import geodesic
from geopy import Point
from datetime import datetime, timedelta

# === データ読み込み ===
ds = xr.open_dataset(r"D:\Pterodroma\marine_data\surface_wind\wind_L4_20240501_20240525.nc")
u_wind = ds['eastward_wind']  # (time, lat, lon)
v_wind = ds['northward_wind']
lat = ds['latitude']
lon = ds['longitude']
time = ds['time']

# =====================================================================================
def calculate_initial_bearing(lat1, lon1, lat2, lon2):
    """
    緯度経度2点間の初期方位角（北基準、時計回り）を度で返す
    """
    from math import radians, degrees, sin, cos, atan2

    phi1 = radians(lat1)
    phi2 = radians(lat2)
    delta_lon = radians(lon2 - lon1)

    x = sin(delta_lon) * cos(phi2)
    y = cos(phi1) * sin(phi2) - sin(phi1) * cos(phi2) * cos(delta_lon)
    bearing = atan2(x, y)
    return (degrees(bearing) + 360) % 360  # 正の範囲に正規化
# =====================================================================================

# === 初期位置（緯度・経度） ===
start_lat = -33.767510
start_lon = -80.786563
trajectory = [(start_lat, start_lon)]

# === 目的地 ===
target_lat = 40.0
target_lon = 170.0

# === 600ステップ（1時間ごと）シミュレーション ===
for t in range(577):
    current_lat, current_lon = trajectory[-1]
    print(f"step {t}")

    # 最も近い格子点を特定
    i_lat = np.abs(lat - current_lat).argmin().item()
    i_lon = np.abs(lon - current_lon).argmin().item()

    # 風速取得（m/s）
    u = u_wind[t, i_lat, i_lon].item()
    v = v_wind[t, i_lat, i_lon].item()

    # ========================================= スピード(風＋能動的) ========================================
    # === 風ベクトル（m）===
    dx_wind = u * 3600
    dy_wind = v * 3600

    # === 能動的な羽ばたきベクトル（目標方向に秒速3m）===
    #active_speed = 3.5 * 3600  # m/h
    active_speed = (3.8 + np.random.uniform(-1, 1)) * 3600   # ランダム性
    target_angle_deg = calculate_initial_bearing(current_lat, current_lon, target_lat, target_lon)
    target_rad = np.radians(target_angle_deg)
    active_dx = active_speed * np.sin(target_rad)
    active_dy = active_speed * np.cos(target_rad)

    # === 合成ベクトル ===
    dx = dx_wind + active_dx
    dy = dy_wind + active_dy
    total_speed = np.hypot(dx, dy)

    # ========================================= 角度(風と目的地の平均) ======================================
    # 方位角（北を0度とし時計回り）
    wind_angle_deg = (np.degrees(np.arctan2(dx, dy))) % 360

    # 目的地への方位角を自前関数で計算
    target_angle_deg = calculate_initial_bearing(current_lat, current_lon, target_lat, target_lon)

    # === 方位角に ±10度のランダムノイズを加える ===
    target_angle_deg = (target_angle_deg + np.random.uniform(-20, 20)) % 360   # ランダム性

    # === 中間角度を取る（風と目標の間）===
    # 両角度の平均（360度の循環性を考慮）
    diff = ((target_angle_deg - wind_angle_deg + 540) % 360) - 180
    adjusted_angle_deg = (wind_angle_deg + diff / 2) % 360

    # === geopyで移動先を計算 ===
    origin = Point(current_lat, current_lon)
    new_point = distance(meters=total_speed).destination(origin, bearing=adjusted_angle_deg)
    trajectory.append((new_point.latitude, new_point.longitude))

# === 地図上に軌跡を描画 ===
lats, lons = zip(*trajectory)

# === 経度補正（180中心のPlateCarree用）===
lons_corrected = []
for lon in lons:
    if lon > 180:
        lon -= 360  # 例えば 200 → -160 に補正（東経200度 = 西経160度）
    lons_corrected.append(lon)

# === 地図描画 ===
lats, lons = zip(*trajectory)
fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))  # 太平洋中心
ax.set_extent([125, 290, -40, 55], crs=ccrs.PlateCarree())

# 陸地・海岸線・国境
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')

# 緯線5度・経線10度おき
gl = ax.gridlines(draw_labels=True, xlocs=np.arange(-180, 181, 10), ylocs=np.arange(-90, 91, 5))
gl.right_labels = False
gl.top_labels = False

# === 各ステップの点（黒）===
ax.scatter(lons_corrected, lats, color='black', s=1, transform=ccrs.PlateCarree(), label="Step Points")

# === 24ステップごとの赤点 ===
daily_indices = list(range(0, len(lats), 24))
ax.scatter([lons_corrected[i] for i in daily_indices], [lats[i] for i in daily_indices],
           color='red', s=10, transform=ccrs.PlateCarree(), label="Every 24h")

# === 軌跡の線と始点・終点 ===
#ax.plot(lons_corrected, lats, color='blue', linewidth=1.5, transform=ccrs.PlateCarree())
ax.plot(lons_corrected[0], lats[0], marker='*', color='green', markersize=8, label='Start', transform=ccrs.PlateCarree())
ax.plot(lons_corrected[-1], lats[-1], marker='*', color='red', markersize=8, label='End', transform=ccrs.PlateCarree())

# === 日付の初期値とステップ幅 ===
start_datetime = datetime(2024, 5, 1, 0, 0)
step_hours = 1  # 1時間ごと

# === 出発点の日時 ===
start_date_str = (start_datetime + timedelta(hours=0)).strftime('%Y/%#m/%#d')
ax.text(lons_corrected[0] + 2, lats[0], start_date_str,
        fontsize=9, color='black', transform=ccrs.PlateCarree())

# === 終点の日時 ===
end_date_str = (start_datetime + timedelta(hours=len(lons_corrected)-1)).strftime('%Y/%#m/%#d')
ax.text(lons_corrected[-1] + 2, lats[-1], end_date_str,
        fontsize=9, color='black', transform=ccrs.PlateCarree())

plt.title("Stejneger's petrels Migration Simulation (576 hourly steps)", fontsize=14)
plt.legend()
plt.show()


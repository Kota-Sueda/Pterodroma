import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# === ERA5風・表層流データの読み込み ===
wind_ds = xr.open_dataset("D:/Pterodroma/marine_data/ERA5_wind/ERA5_wind_1deg_xy.nc")
current_ds = xr.open_dataset("D:/Pterodroma/marine_data/surface_current/surface_current_2020_combined_with_xy.nc")

# === 時間軸整備：mmdd形式のキーを使って辞書化 ===
def make_mmdd_dict(ds, time_var='valid_time'):
    return {pd.Timestamp(t).strftime('%m%d'): i for i, t in enumerate(ds[time_var].values)}

wind_time_map = make_mmdd_dict(wind_ds)
current_time_map = make_mmdd_dict(current_ds)

# === シミュレーション条件 ===
T = 600  # 600ステップ（25日×24時間）
dt = 1   # 1時間
#start = np.array([261.0, -34.0])  # ヒメシロハラ
start = np.array([175.0, -30.0])  # クビワオオシロハラ
goal = np.array([165.0, 35.0])    # 
#goal = np.array([160.0, 28.0])
Vmax = 1

initial_energy = 100
C_basal = 1
C_fly = 3
C_float = 0.5
C_forage = 4
C_gain = 16.8

mode_probs = np.array([6, 2, 2]) / 10

# === シミュレーション本体 ===
num_runs = 10
trajectories = []
energy_tracks = []

# === 緯度経度ベースに変更 ===
lon_w = wind_ds['longitude'].values
lat_w = wind_ds['latitude'].values
lon_c = current_ds['longitude'].values
lat_c = current_ds['latitude'].values

for run in range(num_runs):
    energy = initial_energy
    r = start.copy()
    path = [r.copy()]
    e_track = [energy]
    total_distance = np.linalg.norm(goal - start) * 3.8
    delta_r_avg = total_distance / T

    for t in range(T):
        date = datetime(2020, 5, 1) + timedelta(hours=t)
        mmdd = date.strftime('%m%d')
        iw = wind_time_map.get(mmdd, 0)
        ic = current_time_map.get(mmdd, 0)
        
        ix_w = np.argmin(np.abs(lon_w - r[0]))
        iy_w = np.argmin(np.abs(lat_w - r[1]))
        ix_c = np.argmin(np.abs(lon_c - r[0]))
        iy_c = np.argmin(np.abs(lat_c - r[1]))

        # wind_vec = np.array([wind_ds['u10'][iw, iy_w, ix_w].values,
        #                      wind_ds['v10'][iw, iy_w, ix_w].values]) 
        # current_vec = np.array([current_ds['uo'][ic, iy_c, ix_c].values,
        #                         current_ds['vo'][ic, iy_c, ix_c].values])
        
        # === 緯度（degree）を取得（現在位置 r = [lon, lat]）===
        lat_here = r[1]
        # === ERA5の風速（m/s）を取得 ===
        u10 = wind_ds['u10'][iw, iy_w, ix_w].values  # zonal (東西)
        v10 = wind_ds['v10'][iw, iy_w, ix_w].values  # meridional (南北)
        # === 表層流（m/s）を取得 ===
        uc = current_ds['uo'][ic, iy_c, ix_c].values
        vc = current_ds['vo'][ic, iy_c, ix_c].values
        # === m/s → °/h に変換 ===
        cos_lat = np.cos(np.deg2rad(lat_here)) if abs(lat_here) < 89.9 else 1e-6  # 安全対策
        # wind
        dx_wind = (u10 * 3600) / (111320 * cos_lat)  # 経度方向
        dy_wind = (v10 * 3600) / 111320              # 緯度方向
        wind_vec = np.array([dx_wind, dy_wind])      # [dλ/dt, dφ/dt] in °/h
        # current
        dx_curr = (uc * 3600) / (111320 * cos_lat)
        dy_curr = (vc * 3600) / 111320
        current_vec = np.array([dx_curr, dy_curr])

        if energy < 20:
            mode = np.random.choice(["float", "forage"])
        else:
            mode = np.random.choice(["fly", "float", "forage"], p=mode_probs)

        if mode == "fly":
            d_goal = goal - r
            dist_to_goal = np.linalg.norm(d_goal)
            direction = d_goal / dist_to_goal if dist_to_goal > 0 else np.zeros(2)
            #print(f"direction{direction}")
            ideal_dist = total_distance * t / 600
            #print(f"ideal_dist {ideal_dist}")
            current_dist = np.linalg.norm(goal - r)
            #print(f"current_dist {current_dist}")
            lag_dist = max(current_dist - (total_distance - ideal_dist), 0.01)
            intention = lag_dist 

            direction_noise = np.random.normal(0, 0.1, size=2)
            effort_factor = max(energy / initial_energy, 0.1)
            Vself = (Vmax * intention * direction + direction_noise) * effort_factor

            # 正規化の安全チェック
            if (np.any(np.isnan(direction)) or np.any(np.isnan(wind_vec)) or
                np.linalg.norm(direction) < 1e-8 or np.linalg.norm(wind_vec) < 1e-8):
                cos_theta = 0.0
            else:
                cos_theta = np.dot(direction, wind_vec) / (np.linalg.norm(direction) * np.linalg.norm(wind_vec))

            theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
            angle = np.abs(theta) % (2 * np.pi)
            wind_coef = 1.0 if angle < np.pi/4 or angle > 7*np.pi/4 else (0.5 if np.pi/4 <= angle <= 3*np.pi/4 or 5*np.pi/4 <= angle <= 7*np.pi/4 else 0.2)

            Vwind = wind_coef * wind_vec
            Vtotal = ((1 * Vself + 3 * Vwind) / 4) * 8
            energy -= (C_basal + C_fly)

        elif mode == "float":
            # 欠損チェック：current_vec に NaN があればゼロベクトルに置換
            if np.any(np.isnan(current_vec)):
                current_vec = np.array([0.0, 0.0])

            Vtotal = 0.3 * wind_vec + 0.5 * current_vec
            energy -= (C_basal + C_float)

        elif mode == "forage":
            Vtotal = np.random.normal(0, 1.0, size=2)
            energy += (C_gain - C_forage - C_basal)

        r = r + Vtotal * dt 
        energy = max(0, energy)
        path.append(r.copy())
        e_track.append(energy)
        print(f"r {r}")

        if np.any(np.isnan(r)):
            print(f"Step {t}: NaN detected in position r")
            break

    trajectories.append(np.array(path))
    energy_tracks.append(e_track)

# # === プロット1: 移動軌跡 ===
# plt.figure(figsize=(10, 6))
# for path in trajectories:
#     plt.plot(path[:, 0], path[:, 1], '-o', markersize=2)
# plt.scatter([start[0]], [start[1]], color='green', label='Start')
# plt.scatter([goal[0]], [goal[1]], color='red', label='Goal')
# plt.title("Migration Trajectories")
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")
# plt.legend()
# plt.grid(True)
# plt.axis("equal")
# plt.show()

# === 地図描画の設定 ===
fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))  # 太平洋中心

# fig = plt.figure(figsize=(12, 8))
# ax = plt.axes(projection=ccrs.Mollweide(central_longitude=180))
#ax.set_global()  # Robinsonは全地球図法なので通常 set_global() を使います
ax.set_extent([125, 290, -70, 60], crs=ccrs.PlateCarree())

# 陸地・海岸線・国境
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')

# グリッド線
gl = ax.gridlines(draw_labels=True, xlocs=np.arange(-180, 181, 10), ylocs=np.arange(-90, 91, 5))
gl.right_labels = False
gl.top_labels = False

# === 移動軌跡のプロット ===
for path in trajectories:
    ax.plot(path[:, 0], path[:, 1], '-o', transform=ccrs.PlateCarree(), markersize=2)

# === 開始点と終了点のプロット ===
ax.scatter(start[0], start[1], color='green', s=50, label='Start', transform=ccrs.PlateCarree())
ax.scatter(goal[0], goal[1], color='red', s=50, label='Goal', transform=ccrs.PlateCarree())

# タイトルと凡例
plt.title("Migration Simulation Trajectories of Pterodromas", fontsize=14)
plt.legend()
plt.show()

# === プロット2: エネルギー推移 ===
plt.figure(figsize=(10, 4))
for track in energy_tracks:
    plt.plot(track)
plt.title("Energy Transition Over Time")
plt.xlabel("Time Step")
plt.ylabel("Energy")
plt.grid(True)
plt.show()

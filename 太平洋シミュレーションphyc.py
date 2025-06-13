import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# === ERA5風・表層流・植物プランクトンデータの読み込み ===
wind_ds = xr.open_dataset("D:/Pterodroma/marine_data/ERA5_wind/ERA5_wind_1deg_xy.nc")
current_ds = xr.open_dataset("D:/Pterodroma/marine_data/surface_current/surface_current_2020_combined_with_xy.nc")
phyto_ds = xr.open_dataset("D:/Pterodroma/marine_data/phytoplankton/phyto_combined.nc")

# === 時間軸整備：mmdd形式のキーを使って辞書化 ===
def make_mmdd_dict(ds, time_var='valid_time'):
    if time_var not in ds:
        raise KeyError(f"Time variable '{time_var}' not found in dataset")
    return {pd.Timestamp(t).strftime('%m%d'): i for i, t in enumerate(ds[time_var].values)}

# 各データセットに対応する時間変数で辞書を作成
wind_time_map = make_mmdd_dict(wind_ds, time_var='valid_time')
current_time_map = make_mmdd_dict(current_ds, time_var='valid_time')
phyto_time_map = make_mmdd_dict(phyto_ds, time_var='time')  # 植物プランクトンは 'time'

# === シミュレーション条件 ===
T = 600  # 600ステップ（25日×24時間）
dt = 1   # 1時間
#start = np.array([261.0, -34.0])  # 出発点
goal = np.array([180.0, -34.0])  # 出発点
start = np.array([160.0, 28.0])    # 越夏地
Vmax = 1

initial_energy = 100
C_basal = 1
C_fly = 3
C_float = 0.5
C_forage = 4
C_gain = 20

mode_probs = np.array([6, 2, 2]) / 10

# === 緯度経度の取得 ===
lon_w = wind_ds['longitude'].values
lat_w = wind_ds['latitude'].values
lon_c = current_ds['longitude'].values
lat_c = current_ds['latitude'].values
lon_p = phyto_ds['longitude'].values
lat_p = phyto_ds['latitude'].values

# === シミュレーション実行準備 ===
num_runs = 3
trajectories = []
energy_tracks = []

for run in range(num_runs):
    energy = initial_energy
    r = start.copy()
    path = [r.copy()]
    e_track = [energy]
    total_distance = np.linalg.norm(goal - start) * 3.5
    delta_r_avg = total_distance / T

    for t in range(T):
        date = datetime(2020, 5, 1) + timedelta(hours=t)
        mmdd = date.strftime('%m%d')
        iw = wind_time_map.get(mmdd, 0)
        ic = current_time_map.get(mmdd, 0)
        ip = phyto_time_map.get(mmdd, 0)
        
        ix_w = np.argmin(np.abs(lon_w - r[0]))
        iy_w = np.argmin(np.abs(lat_w - r[1]))
        ix_c = np.argmin(np.abs(lon_c - r[0]))
        iy_c = np.argmin(np.abs(lat_c - r[1]))
        ix_p = np.argmin(np.abs(lon_p - r[0]))
        iy_p = np.argmin(np.abs(lat_p - r[1]))

        # インデックスが領域内かを確認
        if not (1 <= ix_p < len(lon_p) - 1 and 1 <= iy_p < len(lat_p) - 1):
            print(f"Step {t}: Index out of bounds for phyc at ix_p={ix_p}, iy_p={iy_p}")
            break

        # === 座標・緯度準備 ===
        lat_here = r[1]

        # === ERA5の風速（m/s）を取得 ===
        u10 = wind_ds['u10'][iw, iy_w, ix_w].values  # zonal (東西)
        v10 = wind_ds['v10'][iw, iy_w, ix_w].values  # meridional (南北)

        # === 表層流（m/s）を取得 ===
        uc = current_ds['uo'][ic, iy_c, ix_c].values
        vc = current_ds['vo'][ic, iy_c, ix_c].values

        # === m/s → 度/h に変換（球面座標系） ===
        cos_lat = np.cos(np.deg2rad(lat_here)) if abs(lat_here) < 89.9 else 1e-6
        dx_wind = (u10 * 3600) / (111320 * cos_lat)
        dy_wind = (v10 * 3600) / 111320
        wind_vec = np.array([dx_wind, dy_wind])

        dx_curr = (uc * 3600) / (111320 * cos_lat)
        dy_curr = (vc * 3600) / 111320
        current_vec = np.array([dx_curr, dy_curr])

        # === 植物プランクトン濃度 phyc を取得 ===
        phyc_val = phyto_ds['phyc'][ip, 0, iy_p, ix_p].values  # 濃度 [mg/m^3 など]

        # === 植物プランクトンの勾配ベクトルを近似（中心差分） ===
        phyc_east  = phyto_ds['phyc'][ip, 0, iy_p, ix_p + 1].values
        phyc_west  = phyto_ds['phyc'][ip, 0, iy_p, ix_p - 1].values
        phyc_north = phyto_ds['phyc'][ip, 0, iy_p - 1, ix_p].values
        phyc_south = phyto_ds['phyc'][ip, 0, iy_p + 1, ix_p].values

        # 経度・緯度方向の差（deg → m 換算可能）
        dx_deg = lon_p[ix_p + 1] - lon_p[ix_p - 1]  # 度単位
        dy_deg = lat_p[iy_p - 1] - lat_p[iy_p + 1]  # 度単位（南北方向は逆）

        # 単純な度あたりの傾きベクトル（後でスケーリング可能）
        grad_x = (phyc_east - phyc_west) / dx_deg
        grad_y = (phyc_north - phyc_south) / dy_deg
        grad_phyc_vec = np.array([grad_x, grad_y])  # degあたりの濃度変化

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

            direction_noise = np.random.normal(0, 0.3, size=2)
            effort_factor = max(energy / initial_energy, 0.1)
            Vself = (Vmax * intention * direction + direction_noise) * effort_factor

            # --- 風と飛行方向のなす角から angle を取得 ---
            if np.linalg.norm(direction) < 1e-8 or np.linalg.norm(wind_vec) < 1e-8:
                wind_coef = 0.5
                C_fly_adj = C_fly  # デフォルト
            else:
                cos_theta = np.dot(direction, wind_vec) / (np.linalg.norm(direction) * np.linalg.norm(wind_vec))
                angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # 0〜π

                # --- 風向きに応じて係数を決定 ---
                if angle < np.pi / 4:
                    wind_coef = 1.0   # 追い風
                    C_fly_adj = C_fly * 0.8  # 消費軽減
                elif angle < 3 * np.pi / 4:
                    wind_coef = 0.5   # 横風
                    C_fly_adj = C_fly  # 通常
                else:
                    wind_coef = 0.2   # 向かい風
                    C_fly_adj = C_fly * 1.3  # 消費増加

            # --- 速度・エネルギー計算 ---
            Vwind = wind_coef * wind_vec
            Vtotal = ((1 * Vself + 4 * Vwind) / 5) * 6
            energy -= (C_basal + C_fly_adj)

        elif mode == "float":
            if np.any(np.isnan(current_vec)):
                current_vec = np.array([0.0, 0.0])

            Vtotal = 0.3 * wind_vec + 0.5 * current_vec
            energy -= (C_basal + C_float)
        
        elif mode == "forage":
            # 勾配ベクトルに沿って、少しノイズを加えて移動
            noise = np.random.normal(0, 0.2, size=2)
            forage_vec = grad_phyc_vec + noise
            if np.linalg.norm(forage_vec) > 0:
                forage_vec = forage_vec / np.linalg.norm(forage_vec)  # 正規化

            Vtotal = forage_vec * 0.1  # 速度スケーリング（必要に応じて調整）

            # 範囲を [0, 1] にクリップする一次関係
            scaling = np.clip(phyc_val/1.7, 0.95, 3.0)
            C_gain_adj = C_gain * scaling

            energy += (C_gain_adj - C_forage - C_basal)

        r = r + Vtotal * dt 
        energy = max(0, min(energy, 130))
        path.append(r.copy())
        e_track.append(energy)
        print(f"r {r}")

        if np.any(np.isnan(r)):
            print(f"Step {t}: NaN detected in position r")
            break

        if energy <= 0:
            print(f"Step {t}: Energy depleted. Ending run {run}.")
            break

    trajectories.append(np.array(path))
    energy_tracks.append(e_track)

# 軌跡を保存
#np.save('trajectories_4.npy', trajectories, allow_pickle=True)

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

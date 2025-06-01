import numpy as np
import xarray as xr
from datetime import timedelta
import matplotlib.pyplot as plt

# === ファイルパス ===
wind_path = r"D:\Pterodroma\marine_data\ERA5_wind\ERA5_wind_1deg_xy.nc"
current_path = r"D:\Pterodroma\marine_data\surface_current\surface_current_2020.nc"

# === データ読み込み ===
ds_wind = xr.open_dataset(wind_path)
ds_current = xr.open_dataset(current_path)

# 元のデータは 25日分（valid_timeが25個）であることを前提
uo_list = []
vo_list = []
time_list = []

# 1日ずつ取り出して、それを24時間にコピー
for day_idx in range(len(ds_current["valid_time"])):
    u_day = ds_current['uo'].isel(valid_time=day_idx)
    v_day = ds_current['vo'].isel(valid_time=day_idx)
    base_time = ds_current['valid_time'].values[day_idx]

    for h in range(24):  # 各24時間に展開
        uo_list.append(u_day)
        vo_list.append(v_day)
        time_list.append(base_time + np.timedelta64(h, 'h'))

# 時間方向に連結
uo_expanded = xr.concat(uo_list, dim='valid_time')
vo_expanded = xr.concat(vo_list, dim='valid_time')

# 時間座標を設定
uo_expanded['valid_time'] = time_list
vo_expanded['valid_time'] = time_list

# 拡張されたデータセットを構築
ds_current_expanded = xr.Dataset(
    {
        'uo': uo_expanded,
        'vo': vo_expanded
    },
    coords={
        'valid_time': ('valid_time', time_list),
        'latitude': ds_current['latitude'],
        'longitude': ds_current['longitude'],
        'x': ds_current['x'],
        'y': ds_current['y']
    }
)

# 保存（必要であれば有効化）
ds_current_expanded.to_netcdf(r"D:\Pterodroma\marine_data\surface_current\current_2020_expanded.nc")

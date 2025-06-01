import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

# ファイル読み込み
ds = xr.open_dataset(r"D:\Pterodroma\marine_data\surface_current\uv_2024-05-01.nc")

# データ抽出
u = ds['uo'].isel(time=0, depth=0)
v = ds['vo'].isel(time=0, depth=0)
lat = ds.latitude
lon = ds.longitude

# 流速の大きさ
speed = np.sqrt(u**2 + v**2)

# === 勾配計算（緯度・経度方向） ===
# 経度と緯度の間隔（メートルではなく度）
dy, dx = np.gradient(speed.values, lat.values, lon.values)

# 勾配ベクトルのノルム（大きさ）
grad_speed = np.sqrt(dx**2 + dy**2)

# === 描画準備 ===
fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.Mercator(central_longitude=180))

# 描画範囲（データに合わせて）
lat_min = float(lat.min())
lat_max = float(lat.max())
lon_min = float(lon.min())
lon_max = float(lon.max())
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

# 地形情報
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.coastlines(resolution='50m')

# # 緯線・経線
# gl = ax.gridlines(draw_labels=True,
#                   xlocs=np.arange(round(lon_min/10)*10, round(lon_max/10)*10 + 1, 10),
#                   ylocs=np.arange(round(lat_min/10)*10, round(lat_max/10)*10 + 1, 10),
#                   linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
# gl.top_labels = False
# gl.right_labels = False

# 経度を -180〜180 に変換する関数
def wrap_longitude(lons):
    return [(lon - 360 if lon > 180 else lon) for lon in lons]

# gridlines 設定
wrapped_xlocs = wrap_longitude(np.arange(120, 301, 10))  # 120E〜290E
gl = ax.gridlines(draw_labels=True,
                  xlocs=wrapped_xlocs,
                  ylocs=np.arange(round(lat_min/10)*10, round(lat_max/10)*10 + 1, 10),
                  linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
gl.top_labels = False
gl.right_labels = False


# === 勾配ヒートマップ描画 ===
lon2d, lat2d = np.meshgrid(lon, lat)
cmap_plot = ax.pcolormesh(lon2d, lat2d, grad_speed, cmap='plasma', vmin=0.0, vmax=3.0,
                          shading='auto', transform=ccrs.PlateCarree())

# カラーバー
cb = plt.colorbar(cmap_plot, orientation='vertical', pad=0.02, aspect=30)
cb.set_label('Gradient of Speed (1/s)')

# タイトル
plt.title('Gradient of Surface Ocean Currents - 2024-05-01', fontsize=14)

plt.tight_layout()
plt.show()

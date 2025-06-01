import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

# ファイル読み込み
ds = xr.open_dataset(r"D:\Pterodroma\marine_data\surface_current\uv_2024-05-01.nc")

# インデックスで指定（depth=0番目）
u = ds['uo'].isel(time=0, depth=0)
v = ds['vo'].isel(time=0, depth=0)

lat = ds.latitude
lon = ds.longitude

# グリッドを間引く
step = 20
u_skip = u.values[::step, ::step]
v_skip = v.values[::step, ::step]
Lat, Lon = np.meshgrid(lat.values[::step], lon.values[::step], indexing='ij')

# 流速の大きさ
speed = np.sqrt(u_skip**2 + v_skip**2)

# 描画設定
fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.Mercator(central_longitude=180))

# === ここが変更点 ===
# データの緯度経度範囲に合わせて表示
lat_min = float(lat.min())
lat_max = float(lat.max())
lon_min = float(lon.min())
lon_max = float(lon.max())
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
# ===================

# 陸地の描画（解像度50m）
ax.add_feature(cfeature.LAND, zorder=0, facecolor='lightgray')
ax.coastlines(resolution='50m')

# 緯線・経線を10度ごとに
gl = ax.gridlines(draw_labels=True,
                  xlocs=np.arange(round(lon_min/10)*10, round(lon_max/10)*10 + 1, 10),
                  ylocs=np.arange(round(lat_min/10)*10, round(lat_max/10)*10 + 1, 10),
                  linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
gl.top_labels = False
gl.right_labels = False

# 矢印の描画（色で強さ、長さは一定）
q = ax.quiver(Lon, Lat,
              u_skip, v_skip,
              speed,
              cmap='viridis',
              scale=5, scale_units='inches', transform=ccrs.PlateCarree())

# カラーバー
cb = plt.colorbar(q, orientation='vertical', pad=0.02, aspect=30)
cb.set_label('Speed (m/s)')

# # === 等値線を追加 ===
# contour = ax.contour(
#     Lon, Lat, speed,
#     levels=np.linspace(np.nanmin(speed), np.nanmax(speed), 10),  # 等値線の数や範囲は適宜
#     colors='k',  # 黒色線
#     linewidths=0.7,
#     transform=ccrs.PlateCarree()
# )

# # 等値線ラベル
# ax.clabel(contour, fontsize=8, fmt="%.2f")

# タイトル
plt.title('Surface Ocean Currents - 2024-05-01', fontsize=14)

plt.tight_layout()
plt.show()

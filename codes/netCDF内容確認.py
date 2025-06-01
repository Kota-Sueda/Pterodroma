# データ構造の確認 ==========================================================================
import xarray as xr

# ファイルパスを指定
file_path = r"D:\Pterodroma\marine_data\surface_current\surface_current_2020_combined_with_xy.nc"

# データを読み込む
ds = xr.open_dataset(file_path)

# 変数や構造を確認
print(ds)
print(ds["valid_time"])
print(len(["valid_time"]))


# 表層流を矢印で描画 =========================================================================
import matplotlib.pyplot as plt

# 一番最初の時間ステップを選択
u = ds['uo'].isel(time=0)
v = ds['vo'].isel(time=0)

# 座標取得（通常は 'longitude', 'latitude' だが要確認）
lon = ds['longitude']
lat = ds['latitude']

# 描画（サブサンプリングして見やすく）
skip = (slice(None, None, 5), slice(None, None, 5))  # 間引き
plt.figure(figsize=(12, 8))
plt.quiver(lon[skip[1]], lat[skip[0]], u.values[skip], v.values[skip], scale=2, scale_units='xy')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Surface Current Vectors")
plt.grid()
plt.show()

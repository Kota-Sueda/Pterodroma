import os
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from datetime import datetime, timedelta

# === フォルダ設定 ===
folder_path = r"D:\Pterodroma\marine_data\SST_forecast"
save_dir = r"D:\Pterodroma\marine_data\SST_image"
os.makedirs(save_dir, exist_ok=True)

# === 実行日と深さ設定 ===
start_date = datetime.now().date()
depth_target = 0.494025  # 表層のSST（m）

# === 可視化処理 ===
for i in range(9):
    date = start_date + timedelta(days=i)
    file_path = f"{folder_path}\\CMEMS_SST_{date}.nc"

    try:
        ds = xr.open_dataset(file_path)
        sst = ds['thetao'].sel(depth=depth_target, method='nearest')
        lat = ds['latitude']
        lon = ds['longitude']

        # データ1日分抽出
        sst_day = sst.isel(time=0)

        # 緯度マスク
        lat_mask = (lat >= 20) & (lat <= 55)
        # 経度マスクと修正（180度超え → -180未満に変換）
        sst_east = sst_day.sel(latitude=lat[lat_mask], longitude=slice(125, 180))
        sst_west = sst_day.sel(latitude=lat[lat_mask], longitude=slice(180, 200))
        sst_west = sst_west.assign_coords(longitude=sst_west.longitude - 360)

        # 結合して経度でソート
        sst_combined = xr.concat([sst_east, sst_west], dim='longitude')
        sst_combined = sst_combined.sortby('longitude') 

        # === 図の作成 ===
        fig = plt.figure(figsize=(10, 6))
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))  # 太平洋中心
        ax.set_extent([125, 200, 20, 55], crs=ccrs.PlateCarree()) 

        ax.add_feature(cfeature.LAND, facecolor='white', zorder=10)

        # 海岸線と国境
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linestyle=':')

        # 経線緯線（経線10度おき、緯線5度おき）
        gl = ax.gridlines(draw_labels=True, xlocs=np.arange(120, 201, 10), ylocs=np.arange(20, 56, 5))
        gl.right_labels = False
        gl.top_labels = False

        # 海面水温塗り分け（pcolormesh）
        im = ax.pcolormesh(
            sst_combined.longitude,
            sst_combined.latitude,
            sst_combined.values,
            transform=ccrs.PlateCarree(),
            cmap='jet',
            vmin=0, vmax=35
        )

        # === 等温線の追加（18度）===
        contour = ax.contour(
            sst_combined.longitude,
            sst_combined.latitude,
            sst_combined.values,
            levels=[18],
            colors='black',
            linewidths=0.5,
            transform=ccrs.PlateCarree()
        )
        ax.clabel(contour, fmt='%d°C', inline=True, fontsize=9)

        # カラーバーとタイトル
        cbar = plt.colorbar(im, orientation='vertical', pad=0.05, aspect=25)
        cbar.set_label('SST (°C)')
        plt.title(f'Sea Surface Temperature on {date}', fontsize=14)

        # 保存
        plt.savefig(os.path.join(save_dir, f"SST_plot_day{i}_{date}.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Day {i} ({date}) - Plotted and saved.")

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error on {date}: {e}")

import os
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from datetime import datetime, timedelta

# === フォルダ設定 ===
data_dir = r"D:\Pterodroma\marine_data\phytoplankton"
save_dir = r"D:\Pterodroma\marine_data\phytoplankton_images"
os.makedirs(save_dir, exist_ok=True)

# === 描画対象の日付リスト ===
start_date = datetime(2024, 5, 1)
end_date = datetime(2024, 5, 25)
n_days = (end_date - start_date).days + 1

for i in range(n_days):
    date = start_date + timedelta(days=i)
    date_str = date.strftime('%Y-%m-%d')
    file_path = os.path.join(data_dir, f"phyto_{date_str}.nc")

    try:
        ds = xr.open_dataset(file_path)
        phyc = ds['phyc'].isel(depth=0, time=0)  # 表層 & 最初の時刻
        lat = ds['latitude']
        lon = ds['longitude']

        # === 描画 ===
        fig = plt.figure(figsize=(10, 5))
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
        ax.set_extent([125, 290, -60, 65], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.COASTLINE)
        ax.gridlines(draw_labels=True)

        im = ax.pcolormesh(
            lon, lat, phyc,
            transform=ccrs.PlateCarree(),
            cmap='YlGn',  # 植物っぽいカラーマップ
            shading='auto'
        )

        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label('Phytoplankton Carbon (mg C m⁻³)')

        plt.title(f"Phytoplankton Biomass (PHYC) on {date_str}", fontsize=14)

        # 保存
        out_path = os.path.join(save_dir, f"phyc_{date_str}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_path}")

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error on {date_str}: {e}")

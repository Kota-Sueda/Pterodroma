import os
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from datetime import datetime, timedelta

# === フォルダ設定 ===
phyc_dir = r"D:\Pterodroma\marine_data\phyc_forecast"
sst_dir = r"D:\Pterodroma\marine_data\SST_forecast"
save_dir = r"D:\Pterodroma\marine_data\phyc_image"
os.makedirs(save_dir, exist_ok=True)

# === 実行日と深さ設定 ===
start_date = datetime.now().date()
depth_target = 0.4940253794193268  # 表層深度

# === 可視化ループ ===
for i in range(9):
    date = start_date + timedelta(days=i)
    date_str = date.strftime("%Y-%m-%d")
    phyc_file = os.path.join(phyc_dir, f"CMEMS_PHYC_{date}.nc")
    sst_file = os.path.join(sst_dir, f"CMEMS_SST_{date}.nc")

    try:
        # === PHYC データ読み込み ===
        ds_phyc = xr.open_dataset(phyc_file)
        phyc = ds_phyc['phyc'].sel(depth=depth_target, method='nearest').isel(time=0)
        lat = ds_phyc['latitude']
        lon = ds_phyc['longitude']

        # 緯度・経度マスク
        lat_mask = (lat >= 20) & (lat <= 55)
        phyc_east = phyc.sel(latitude=lat[lat_mask], longitude=slice(125, 180))
        phyc_west = phyc.sel(latitude=lat[lat_mask], longitude=slice(180, 200))
        phyc_west = phyc_west.assign_coords(longitude=phyc_west.longitude - 360)
        phyc_comb = xr.concat([phyc_east, phyc_west], dim='longitude').sortby('longitude')

        # === 図作成 ===
        fig = plt.figure(figsize=(10, 6))
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
        ax.set_extent([125, 200, 20, 55], crs=ccrs.PlateCarree())

        # 地図要素
        ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=10)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linestyle=':')

        # 緯線・経線
        gl = ax.gridlines(draw_labels=True, xlocs=np.arange(120, 201, 10), ylocs=np.arange(20, 56, 5))
        gl.right_labels = False
        gl.top_labels = False

        # === PHYCプロット ===
        im = ax.pcolormesh(
            phyc_comb.longitude,
            phyc_comb.latitude,
            phyc_comb.values,
            transform=ccrs.PlateCarree(),
            cmap='jet',
            vmin=0, vmax=12
        )

        # === SST 等温線（18℃）の追加 ===
        try:
            ds_sst = xr.open_dataset(sst_file)
            sst = ds_sst['thetao'].sel(depth=depth_target, method='nearest').isel(time=0)
            lat_sst = ds_sst['latitude']

            lat_mask_sst = (lat_sst >= 20) & (lat_sst <= 55)
            sst_east = sst.sel(latitude=lat_sst[lat_mask_sst], longitude=slice(125, 180))
            sst_west = sst.sel(latitude=lat_sst[lat_mask_sst], longitude=slice(180, 200))
            sst_west = sst_west.assign_coords(longitude=sst_west.longitude - 360)
            sst_comb = xr.concat([sst_east, sst_west], dim='longitude').sortby('longitude')

            contour = ax.contour(
                sst_comb.longitude,
                sst_comb.latitude,
                sst_comb.values,
                levels=[18],
                colors='white',
                linewidths=1.0,
                transform=ccrs.PlateCarree()
            )
            ax.clabel(contour, fmt='%d°C', fontsize=9)
        except Exception as e:
            print(f"⚠️ SST loading failed for {date_str}: {e}")

        # === カラーバーとタイトル ===
        cbar = plt.colorbar(im, orientation='vertical', pad=0.05, aspect=25)
        cbar.set_label('Phytoplankton [mmol/m³]')
        plt.title(f'Phytoplankton Biomass with 18°C SST — {date_str}', fontsize=14)

        # 保存
        out_path = os.path.join(save_dir, f"PHYC_plot_day{i}_{date}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Day {i} ({date_str}) - Saved to {out_path}")

    except FileNotFoundError:
        print(f"❌ PHYC file not found: {phyc_file}")
    except Exception as e:
        print(f"❌ Error on {date_str}: {e}")

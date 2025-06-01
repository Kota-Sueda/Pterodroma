import os
from datetime import datetime
import copernicusmarine

# === 保存先フォルダ ===
output_dir = r"D:\Pterodroma\marine_data\surface_wind"
os.makedirs(output_dir, exist_ok=True)

# === 日付範囲 ===
start_date = datetime(2024, 5, 1)
end_date = datetime(2024, 5, 25)

# === 出力ファイル名 ===
output_filename = "wind_L4_20240501_20240525.nc"

# === サブセットダウンロード ===
copernicusmarine.subset(
    dataset_id="cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H",   
    variables=["eastward_wind", "northward_wind"],  # 必要な風向風速成分
    minimum_longitude=125,
    maximum_longitude=290,
    minimum_latitude=-60,
    maximum_latitude=65,
    start_datetime=start_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
    end_datetime=end_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
    output_filename=output_filename,
    output_directory=output_dir
)

print(f"✅ ダウンロード完了: {os.path.join(output_dir, output_filename)}")

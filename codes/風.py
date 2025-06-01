import os
from datetime import datetime
import copernicusmarine

# === 出力ディレクトリを設定 ===
output_dir = r"D:\Pterodroma\marine_data\surface_wind"
os.makedirs(output_dir, exist_ok=True)

# === 日付設定 ===
start_date = datetime(2024, 5, 1)
end_date = datetime(2024, 5, 25)

# === ファイル名 ===
output_filename = f"wind_metopc_ascat_desc_20240501_20240525.nc"

# === データ取得 ===
copernicusmarine.subset(
    dataset_id="cmems_obs-wind_glo_phy_nrt_l3-metopc-ascat-des-0.125deg_P1D-i",
    variables=["eastward_wind", "northward_wind"],
    minimum_longitude=125,
    maximum_longitude=290,
    minimum_latitude=-60,
    maximum_latitude=65,
    start_datetime=start_date.strftime("%Y-%m-%d"),
    end_datetime=end_date.strftime("%Y-%m-%d"),
    output_filename=output_filename,
    output_directory=output_dir
)

print(f"Download complete: {os.path.join(output_dir, output_filename)}")

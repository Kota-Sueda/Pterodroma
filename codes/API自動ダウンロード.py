import copernicusmarine
from datetime import datetime, timedelta, timezone
import os

# === 現在の日付（日本時間）を取得 ===
jst = timezone(timedelta(hours=9))  # JST (UTC+9)
today = datetime.now(jst).date()

# === 出力ディレクトリ ===
output_dir = r"D:\Pterodroma\marine_data\phyc_forecast"

# === 0.49mのSSTを8日分ダウンロード ===
for i in range(9):
    date = today + timedelta(days=i)
    output_filename = f"CMEMS_PHYC_{date}.nc"

    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_bgc-pft_anfc_0.25deg_P1D-m",
        variables=["phyc"],
        minimum_longitude=125,
        maximum_longitude=200,
        minimum_latitude=20,
        maximum_latitude=55,
        start_datetime=date.strftime("%Y-%m-%d"),
        end_datetime=date.strftime("%Y-%m-%d"),
        minimum_depth=0.4940253794193268,
        maximum_depth=0.4940253794193268,
        output_filename=output_filename,
        output_directory=output_dir
    )

    print(f"Day {i+1} ({date}) - Download complete: {output_filename}")

# === ファイル削除処理 ===
for filename in os.listdir(output_dir):
    if filename.startswith("CMEMS_SST_") and filename.endswith(".nc"):
        filepath = os.path.join(output_dir, filename)

        try:
            date_str = filename.replace("CMEMS_SST_", "").replace(".nc", "")
            file_date = datetime.strptime(date_str, "%Y-%m-%d").date()

            # 7日より前のものを削除
            if (today - file_date).days > 7:
                os.remove(filepath)
                print(f"Deleted old file: {filename}")

        except ValueError:
            continue  # 日付変換に失敗したら無視

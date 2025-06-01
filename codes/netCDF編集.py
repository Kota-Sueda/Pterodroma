# import xarray as xr
# import numpy as np

# # 元データ読み込み
# ds = xr.open_dataset(r"D:\Pterodroma\marine_data\ERA5_wind\bc6b4b29501198b35d8398d74d73992f.nc")

# # 緯度経度の整数値のみ抽出（0.25度刻み→1度刻みに）
# lat_selected = ds.latitude[np.isin(ds.latitude, np.arange(-60, 61, 1))]
# lon_selected = ds.longitude[np.isin(ds.longitude, np.arange(125, 291, 1))]

# # 選択した緯度経度でサブセット
# ds_subset = ds.sel(latitude=lat_selected, longitude=lon_selected)

# # 緯度経度をxy座標に変換（単純変換）
# ds_subset = ds_subset.assign_coords({
#     "x": ("longitude", ds_subset.longitude.data),
#     "y": ("latitude", ds_subset.latitude.data)
# })

# output_path = r"D:\Pterodroma\marine_data\ERA5_wind\ERA5_wind_1deg_xy.nc"
# ds_subset.to_netcdf(output_path)

# import matplotlib.pyplot as plt

# # 5月1日00:00のu風を描画
# ds_subset.u10.sel(valid_time="2020-05-01T00:00").plot()
# plt.title("u10 wind at 2020-05-01 00:00 (1° grid)")
# plt.show()



# import os
# import xarray as xr
# import numpy as np
# import traceback

# # === フォルダパス ===
# input_folder = "D:/Pterodroma/marine_data/surface_current"
# output_folder = "D:/Pterodroma/marine_data/surface_current/trimmed"
# os.makedirs(output_folder, exist_ok=True)

# # === ファイル一覧取得 ===
# file_list = sorted([
#     f for f in os.listdir(input_folder)
#     if f.endswith(".nc") and "uv" in f
# ])

# print(f"🔍 見つかったファイル数: {len(file_list)}")

# # === 整数格子点を定義 ===
# lat_targets = np.arange(-60, 61, 1)
# lon_targets = np.arange(125, 291, 1)

# for file_name in file_list:
#     input_path = os.path.join(input_folder, file_name)
#     output_path = os.path.join(output_folder, file_name.replace(".nc", "_trimmed.nc"))

#     print(f"\n📂 処理中: {file_name}")

#     try:
#         # === データ読み込み ===
#         ds = xr.open_dataset(input_path)

#         # === depth 次元を削除（最表層）===
#         if "depth" in ds.dims:
#             ds = ds.isel(depth=0).drop_vars("depth")

#         # === 緯度・経度を整数格子点にリサンプリング（nearest）===
#         ds = ds.sel(
#             latitude=lat_targets,
#             longitude=lon_targets,
#             method="nearest"
#         )

#         # === 保存 ===
#         ds.to_netcdf(output_path, mode="w", engine="netcdf4")
#         print(f"✅ 保存成功: {output_path}")

#     except Exception as e:
#         print(f"❌ エラー発生: {file_name}")
#         traceback.print_exc()
#         continue

# import os
# import xarray as xr
# import traceback

# # === 入力・出力パス ===
# input_folder = "D:/Pterodroma/marine_data/surface_current/trimmed"
# output_path = "D:/Pterodroma/marine_data/surface_current/surface_current_2020_combined.nc"

# # === .ncファイル一覧を取得（ソート済）===
# file_list = sorted([
#     f for f in os.listdir(input_folder)
#     if f.endswith("_trimmed.nc")
# ])

# print(f"🔍 結合対象ファイル数: {len(file_list)}")

# # === 各データセットを格納するリスト ===
# dataset_list = []

# for file_name in file_list:
#     file_path = os.path.join(input_folder, file_name)
#     print(f"📂 読み込み中: {file_name}")
    
#     try:
#         ds = xr.open_dataset(file_path)
#         dataset_list.append(ds)
#     except Exception as e:
#         print(f"❌ 読み込み失敗: {file_name}")
#         traceback.print_exc()
#         continue

# # === 時間軸（valid_time）に沿って結合 ===
# try:
#     print("\n🔧 データ結合中...")
#     combined = xr.concat(dataset_list, dim="time")  # ← dim="valid_time" ではなく time の場合もあるので要確認

#     # === ディメンション名が "time" なら "valid_time" に変更して統一 ===
#     if "time" in combined.dims and "valid_time" not in combined.dims:
#         combined = combined.rename({"time": "valid_time"})

#     # === 結合結果を保存 ===
#     combined.to_netcdf(output_path, mode="w", engine="netcdf4")
#     print(f"✅ 結合＆保存成功: {output_path}")

# except Exception as e:
#     print("❌ 結合または保存時にエラーが発生しました")
#     traceback.print_exc()

import xarray as xr

# === ファイルパス ===
file_path = "D:/Pterodroma/marine_data/surface_current/surface_current_2020_combined.nc"

# === データ読み込み（with ブロックで自動クローズ）===
with xr.open_dataset(file_path) as ds:
    ds = ds.assign_coords({
        "x": ("longitude", ds.longitude.data),
        "y": ("latitude", ds.latitude.data)
    })

    # 別名で保存（←いったん安全に書き出す）
    temp_path = file_path.replace(".nc", "_with_xy.nc")
    ds.to_netcdf(temp_path, mode="w", engine="netcdf4")

print(f"✅ 保存完了: {temp_path}")

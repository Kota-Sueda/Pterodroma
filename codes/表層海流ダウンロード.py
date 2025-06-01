# from copernicusmarine import subset
# from datetime import datetime, timedelta
# import os

# # 保存先フォルダ
# output_dir = r"D:\Pterodroma\marine_data\phytoplankton"
# os.makedirs(output_dir, exist_ok=True)

# # 期間
# start_date = datetime(2024, 5, 1)
# end_date = datetime(2024, 5, 25)

# # 日ごとに処理
# current_date = start_date
# while current_date <= end_date:
#     date_str = current_date.strftime('%Y-%m-%d')
#     output_path = os.path.join(output_dir, f"phyto_{date_str}.nc")
#     print(f"Downloading: {date_str}")
#     try:
#         subset(
#             dataset_id="cmems_mod_glo_bgc-pft_anfc_0.25deg_P1D-m",
#             variables=["phyc"],
#             start_datetime=date_str + "T00:00:00",
#             end_datetime=date_str + "T23:59:59",
#             minimum_longitude=125,
#             maximum_longitude=290,
#             minimum_latitude=-60,
#             maximum_latitude=65,
#             minimum_depth=0.4940253794193268,
#             maximum_depth=0.4940253794193268,
#             output_filename=output_path,
#         )
#     except Exception as e:
#         print(f"Failed on {date_str}: {e}")

#     current_date += timedelta(days=1)


from copernicusmarine import subset
from datetime import datetime, timedelta
import os

# 保存先フォルダ
output_dir = r"D:\Pterodroma\marine_data\surface_current\surface_current_2020"
os.makedirs(output_dir, exist_ok=True)

# 期間
start_date = datetime(2021, 5, 1)
end_date = datetime(2021, 5, 25)

# 日ごとに処理
current_date = start_date
while current_date <= end_date:
    date_str = current_date.strftime('%Y-%m-%d')
    output_path = os.path.join(output_dir, f"uv_{date_str}.nc")
    print(f"Downloading: {date_str}")
    
    try:
        subset(
            dataset_id="cmems_mod_glo_phy_anfc_merged-uv_PT1H-i",
            variables=["uo", "vo"],
            start_datetime=date_str + "T00:00:00",
            end_datetime=date_str + "T23:59:59",
            minimum_longitude=125,
            maximum_longitude=290,
            minimum_latitude=-60,
            maximum_latitude=60,
            minimum_depth=0.49402499198913574,
            maximum_depth=0.49402499198913574,
            output_filename=output_path,
        )
    except Exception as e:
        print(f"Failed on {date_str}: {e}")
    
    current_date += timedelta(days=1)

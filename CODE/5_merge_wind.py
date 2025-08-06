import pandas as pd
import os
from glob import glob

base_dir = "firemask_output"
white_img_path = os.path.join(base_dir, "white.png")

def rounded(x):
    return round(float(x), 5)

firemask_df = pd.read_csv("firemask_modis.csv")
firemask_df["date"] = pd.to_datetime(firemask_df["date"])

for col in ["lat_min", "lat_max", "lon_min", "lon_max"]:
    firemask_df[col] = firemask_df[col].apply(rounded)

wind_folder = "wind_data"
wind_files = glob(os.path.join(wind_folder, "grid_*.csv"))

wind_all = []
for file in wind_files:
    date_str = os.path.basename(file).split("_")[1][:8]
    wind_df = pd.read_csv(file)
    wind_df["date"] = pd.to_datetime(date_str, format="%Y%m%d")
    for col in ["lat_min", "lat_max", "lon_min", "lon_max"]:
        wind_df[col] = wind_df[col].apply(rounded)
    wind_all.append(wind_df)

wind_df = pd.concat(wind_all, ignore_index=True)

# Merge(Left Join)
merged_df = pd.merge(
    firemask_df,
    wind_df,
    on=["date", "lat_min", "lat_max", "lon_min", "lon_max"],
    how="left",
    suffixes=('', '_wind')
)

# Merge 후에도 firemask 경로는 손대지 않음!
def verify_image_path_with_log(path):
    if isinstance(path, str) and os.path.exists(path):
        return path
    else:
        if not isinstance(path, str) or path != white_img_path:
            print(f"[경고] 존재하지 않는 이미지 경로 → white.png로 대체: {path}")
        return white_img_path

merged_df["today_firemask"] = merged_df["today_firemask"].apply(verify_image_path_with_log)
merged_df["prev_firemask"] = merged_df["prev_firemask"].apply(verify_image_path_with_log)

# wind 데이터 없는 행은 그냥 두어도 됨(혹은 dropna, fillna 등 선택 가능)
merged_df.to_csv("firemask_modis_wind.csv", index=False)
print("저장 완료: firemask_modis_wind.csv")

import pandas as pd
import os

base_dir = "firemask_output"
white_img_path = os.path.join(base_dir, "white.png")

fire_df = pd.read_csv("firemask_metadata_filtering.csv")
modis_df = pd.read_csv("MODIS_NDVI_EVI_15kmgrid_2019_2023.csv")

# 형식 통일
fire_df["date"] = pd.to_datetime(fire_df["date"]).dt.strftime("%Y-%m-%d")
modis_df["date"] = pd.to_datetime(modis_df["date"]).dt.strftime("%Y-%m-%d")

fire_df["lat_min"] = fire_df["lat_min"].round(5)
fire_df["lon_min"] = fire_df["lon_min"].round(5)
modis_df["lat_min"] = modis_df["lat_min"].round(5)
modis_df["lon_min"] = modis_df["lon_min"].round(5)

# 병합
modis_df = modis_df.rename(columns={
    "sur_refl_b01": "surf_reflectance_red",
    "sur_refl_b02": "surf_reflectance_nir",
    "sur_refl_b03": "surf_reflectance_blue"
})

modis_df = modis_df[[
    "date", "lat_min", "lon_min", "NDVI", "EVI",
    "surf_reflectance_red", "surf_reflectance_nir", "surf_reflectance_blue"
]]

merged_df = pd.merge(
    fire_df,
    modis_df,
    on=["date", "lat_min", "lon_min"],
    how="left"
)

def verify_image_path_with_log(path):
    if os.path.exists(path):
        return path
    else:
        print(f"[경고] 존재하지 않는 이미지 경로 → white.png로 대체: {path}")
        return white_img_path

merged_df["today_firemask"] = merged_df["today_firemask"].apply(verify_image_path_with_log)
merged_df["prev_firemask"] = merged_df["prev_firemask"].apply(verify_image_path_with_log)


merged_df.to_csv("firemask_modis.csv", index=False, encoding="utf-8-sig")
print("저장 완료: firemask_modis.csv")

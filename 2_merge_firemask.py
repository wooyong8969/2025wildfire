import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

base_dir = "firemask_output"
white_img_path = os.path.join(base_dir, "white.png")

def create_white_image(path, img_size=150):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        blank = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
        plt.imsave(path, blank)
        print(f"흰 이미지 생성: {path}")

create_white_image(white_img_path)

date_list = [
    '2019-04-04', '2019-04-05', '2019-04-06',
    '2020-03-19', '2020-03-20', '2020-03-21', '2020-03-22', '2020-03-23', '2020-03-24',
    '2020-04-24', '2020-04-25', '2020-04-26', '2020-04-27',
    '2020-05-01', '2020-05-02', '2020-05-03',
    '2021-02-18', '2021-02-21', '2021-02-22',
    '2022-02-25', '2022-02-26', '2022-02-27', '2022-02-28', '2022-03-01',
    '2022-03-04', '2022-03-05', '2022-03-06', '2022-03-07', '2022-03-08', '2022-03-09',
    '2022-03-10', '2022-03-11', '2022-03-12', '2022-03-13', '2022-03-14',
    '2022-04-05', '2022-04-06', '2022-04-07',
    '2022-04-09', '2022-04-10', '2022-04-11', '2022-04-12', '2022-04-13',
    '2022-04-22', '2022-04-23', '2022-04-24',
    '2022-05-28', '2022-05-29', '2022-05-30', '2022-05-31', '2022-06-01', '2022-06-02', '2022-06-03',
    '2023-02-28', '2023-03-01', '2023-03-02', '2023-03-03', '2023-03-04',
    '2023-03-08', '2023-03-09', '2023-03-10', '2023-03-11', '2023-03-12',
    '2023-03-16', '2023-03-17', '2023-03-18', '2023-03-19', '2023-03-20',
    '2023-03-26', '2023-03-27', '2023-03-28', '2023-03-29', '2023-03-30', '2023-03-31',
    '2023-04-01', '2023-04-02', '2023-04-03', '2023-04-04', '2023-04-05',
    '2023-04-11', '2023-04-12'
]

# 1. 메타데이터 불러오기
frames = []
for date_str in date_list:
    meta_path = os.path.join(base_dir, f"firemask_metadata_{date_str}.csv")
    if os.path.exists(meta_path):
        df = pd.read_csv(meta_path)
        df["date"] = date_str
        df["today_firemask"] = df["filename"].apply(lambda fname: os.path.join(base_dir, fname))
        frames.append(df)
    else:
        print(f"[경고] 메타데이터 파일 없음: {meta_path}")

if not frames:
    raise Exception("메타데이터 파일을 하나도 불러오지 못했습니다.")

all_df = pd.concat(frames, ignore_index=True)
all_df["date"] = pd.to_datetime(all_df["date"])
all_df.sort_values(by=["date", "lat_min", "lon_min"], inplace=True)
all_df = all_df.reset_index(drop=True)

# 좌표 오차 방지를 위해 소수점 5자리 고정
def rounded(x):
    return round(float(x), 5)
all_df["lat_min"] = all_df["lat_min"].apply(rounded)
all_df["lon_min"] = all_df["lon_min"].apply(rounded)

# 날짜 문자열 포맷 통일
all_df["date"] = all_df["date"].dt.strftime("%Y-%m-%d")

# 2. 오늘 없는(0 fire) 그리드: 전날에 있는데 오늘은 없는 경우
additional_rows = []
datetime_list = [datetime.strptime(d, "%Y-%m-%d") for d in date_list]
for idx in range(1, len(datetime_list)):
    prev_day = datetime_list[idx - 1]
    curr_day = datetime_list[idx]
    prev_str = prev_day.strftime("%Y-%m-%d")
    curr_str = curr_day.strftime("%Y-%m-%d")
    df_prev = all_df[all_df["date"] == prev_str]
    df_curr = all_df[all_df["date"] == curr_str]
    prev_keys = set(zip(df_prev["lat_min"], df_prev["lon_min"]))
    curr_keys = set(zip(df_curr["lat_min"], df_curr["lon_min"]))
    missing_keys = prev_keys - curr_keys
    for lat, lon in missing_keys:
        prev_row = df_prev[(df_prev["lat_min"] == lat) & (df_prev["lon_min"] == lon)].iloc[0]
        new_row = {
            "date": curr_str,
            "lat_min": lat,
            "lon_min": lon,
            "lat_max": prev_row["lat_max"],
            "lon_max": prev_row["lon_max"],
            "today_point_count": 0,
            "prev_fire_point": prev_row["point_count"],
            "today_frp_mean": 0.0,
            "prev_frp_mean": prev_row["frp_mean"],
            "today_firemask": white_img_path,
            "prev_firemask": prev_row["today_firemask"]
        }
        additional_rows.append(new_row)

# 3. 정확히 전날 기준 prev 정보 매핑
def get_prev_info(row):
    prev_date = (datetime.strptime(row["date"], "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    lat = rounded(row["lat_min"])
    lon = rounded(row["lon_min"])
    prev = all_df[(all_df["date"] == prev_date) &
                  (all_df["lat_min"] == lat) &
                  (all_df["lon_min"] == lon)]
    if len(prev) > 0:
        return prev.iloc[0]["point_count"], prev.iloc[0]["frp_mean"], prev.iloc[0]["today_firemask"]
    else:
        return 0, 0.0, white_img_path

prevs = all_df.apply(get_prev_info, axis=1, result_type="expand")
all_df["prev_fire_point"] = prevs[0]
all_df["prev_frp_mean"] = prevs[1]
all_df["prev_firemask"] = prevs[2]

# point_count → today_point_count, frp_mean → today_frp_mean 이름 통일
all_df = all_df.rename(columns={"point_count": "today_point_count", "frp_mean": "today_frp_mean"})

# 4. 추가행 붙이기
if additional_rows:
    additional_df = pd.DataFrame(additional_rows)
    final_df = pd.concat([all_df, additional_df], ignore_index=True, sort=False)
else:
    final_df = all_df

# 5. 컬럼 순서 정리
final_df = final_df[[
    "date", "lat_min", "lon_min", "lon_max", "lat_max",
    "prev_fire_point", "today_point_count",
    "prev_frp_mean", "today_frp_mean",
    "prev_firemask", "today_firemask"
]]
final_df = final_df.sort_values(by=["date", "lat_min", "lon_min"]).reset_index(drop=True)

# 6. 경로 실제 파일로 체크 (실제로 없는 파일은 white로)
def verify_image_path(path):
    return path if isinstance(path, str) and os.path.exists(path) else white_img_path

final_df["today_firemask"] = final_df["today_firemask"].apply(verify_image_path)
final_df["prev_firemask"] = final_df["prev_firemask"].apply(verify_image_path)

final_df.to_csv("firemask_metadata_all.csv", index=False, encoding="utf-8-sig")
print("저장 완료: firemask_metadata_all.csv")

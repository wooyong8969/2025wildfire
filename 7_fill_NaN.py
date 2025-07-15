# import os
# import pandas as pd
# import numpy as np
# from tqdm import tqdm

# main_csv_path = 'firemask_modis_wind_weather.csv'          # 결측치가 있는 메인 CSV
# evi_split_dir = 'EVI_split_data'                           # EVI_split_data 압축 해제 폴더 경로
# data_split_dir = 'dates_split'                              # data_split 압축 해제 폴더 경로

# main_df = pd.read_csv(main_csv_path)

# evi_split_dates = {}
# for file in os.listdir(evi_split_dir):
#     if not file.endswith('.csv'): continue
#     date_str = file.replace('EVI_', '').replace('.csv', '').replace('_', '-')
#     evi_split_dates[date_str] = os.path.join(evi_split_dir, file)

# data_split_dates = {}
# for file in os.listdir(data_split_dir):
#     if not file.endswith('.csv'): continue
#     date_str = file.replace('date_', '').replace('.csv', '')
#     data_split_dates[date_str] = os.path.join(data_split_dir, file)

# final_split_files = {}
# all_dates = set(list(evi_split_dates.keys()) + list(data_split_dates.keys()))
# for date in all_dates:
#     if date in evi_split_dates:
#         final_split_files[date] = ('EVI', evi_split_dates[date])
#     elif date in data_split_dates:
#         final_split_files[date] = ('DATA', data_split_dates[date])

# na_mask = main_df['NDVI'].isna() | main_df['EVI'].isna()
# main_df_filled = main_df.copy()

# for idx in tqdm(main_df[na_mask].index):
#     row = main_df.loc[idx]
#     date = str(row['date'])

#     center_lat = (row['lat_min'] + row['lat_max']) / 2
#     center_lon = (row['lon_min'] + row['lon_max']) / 2

#     if date in final_split_files:
#         src, split_path = final_split_files[date]
#         split_df = pd.read_csv(split_path)
#         if src == 'EVI':
#             lat_col, lon_col = 'latitude', 'longitude'
#         else:
#             lat_col, lon_col = 'G', 'F'

#         dists = np.sqrt(
#             (split_df[lat_col] - center_lat) ** 2 +
#             (split_df[lon_col] - center_lon) ** 2
#         )
#         min_idx = dists.idxmin()
#         if pd.isna(row['NDVI']):
#             main_df_filled.at[idx, 'NDVI'] = split_df.loc[min_idx, 'NDVI']
#         if pd.isna(row['EVI']):
#             main_df_filled.at[idx, 'EVI'] = split_df.loc[min_idx, 'EVI']

# main_df_filled.to_csv('firemask_modis_wind_weather_filled_priorityEVI.csv', index=False)
# print('저장 완료: firemask_modis_wind_weather_end.csv')



import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta

main_csv_path = 'firemask_modis_wind_weather.csv'
evi_split_dir = 'EVI_split_data'
data_split_dir = 'dates_split'

main_df = pd.read_csv(main_csv_path)

# 폴더 내 날짜별 파일 목록 만들기
def get_date_dict(split_dir, type_):
    out = {}
    for file in os.listdir(split_dir):
        if not file.endswith('.csv'): continue
        if type_ == 'EVI':
            # EVI_2019_04_04.csv → 2019-04-04
            date_str = file.replace('EVI_', '').replace('.csv', '').replace('_', '-')
        else:
            # date_2023-02-28.csv → 2023-02-28
            date_str = file.replace('date_', '').replace('.csv', '')
        out[date_str] = os.path.join(split_dir, file)
    return out

evi_split_dates = get_date_dict(evi_split_dir, 'EVI')
data_split_dates = get_date_dict(data_split_dir, 'DATA')

def to_dt(s):
    return datetime.strptime(s, "%Y-%m-%d")

evi_dates_dt = {to_dt(k): v for k, v in evi_split_dates.items()}
data_dates_dt = {to_dt(k): v for k, v in data_split_dates.items()}

# 결측치가 있는 행만 처리
na_mask = main_df['NDVI'].isna() | main_df['EVI'].isna()
main_df_filled = main_df.copy()

for idx in tqdm(main_df[na_mask].index):
    row = main_df.loc[idx]
    date_str = str(row['date'])
    # 중심좌표 계산 (그리드 중심)
    center_lat = (row['lat_min'] + row['lat_max']) / 2
    center_lon = (row['lon_min'] + row['lon_max']) / 2

    try:
        target_dt = to_dt(date_str)
    except Exception:
        continue

    # 1순위: 같은 날짜 split 파일 (EVI_split_data > data_split)
    src, split_path = None, None
    if target_dt in evi_dates_dt:
        src, split_path = 'EVI', evi_dates_dt[target_dt]
    elif target_dt in data_dates_dt:
        src, split_path = 'DATA', data_dates_dt[target_dt]
    else:
        # 2순위: ±2일 이내의 split 파일 탐색 (날짜 차이 가장 작은 쪽, 동점이면 EVI_split_data 우선)
        # 후보 목록
        candidates = []
        for offset in range(1, 3):
            for sign in [-1, 1]:
                dt = target_dt + timedelta(days=offset * sign)
                if dt in evi_dates_dt:
                    candidates.append( ('EVI', abs((dt - target_dt).days), evi_dates_dt[dt], dt) )
                if dt in data_dates_dt:
                    candidates.append( ('DATA', abs((dt - target_dt).days), data_dates_dt[dt], dt) )
        if candidates:
            # 날짜 차이가 가장 작은 후보만 남김, 여러 개면 EVI_split_data 우선
            min_diff = min([c[1] for c in candidates])
            candidates_min = [c for c in candidates if c[1]==min_diff]
            # EVI_split_data 우선
            evi_first = [c for c in candidates_min if c[0]=='EVI']
            if evi_first:
                src, _, split_path, _ = evi_first[0]
            else:
                src, _, split_path, _ = candidates_min[0]

    if split_path is not None:
        split_df = pd.read_csv(split_path)
        if src == 'EVI':
            lat_col, lon_col = 'latitude', 'longitude'
        else:
            lat_col, lon_col = 'G', 'F'
        # 최근접 값 찾기 (유클리드 거리)
        dists = np.sqrt(
            (split_df[lat_col] - center_lat) ** 2 +
            (split_df[lon_col] - center_lon) ** 2
        )
        min_idx = dists.idxmin()
        if pd.isna(row['NDVI']):
            main_df_filled.at[idx, 'NDVI'] = split_df.loc[min_idx, 'NDVI']
        if pd.isna(row['EVI']):
            main_df_filled.at[idx, 'EVI'] = split_df.loc[min_idx, 'EVI']
    # split 파일이 없으면 결측 유지


main_df_filled.to_csv('firemask_modis_wind_weather_end_2day.csv', index=False)
print('저장 완료: firemask_modis_wind_weather_end_2day.csv')


main_df.loc[main_df['NDVI'].isna() | main_df['EVI'].isna(), ['today_firemask', 'prev_firemask']]

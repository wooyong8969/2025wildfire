import pandas as pd
import os

# CSV 파일명, white.png 경로
csv_path = "firemask_modis_wind_weather_end_2day.csv"
base_dir = "firemask_output"
white_img_path = os.path.join(base_dir, "white.png")

# 데이터 불러오기
df = pd.read_csv(csv_path)

# today_firemask가 white.png인 비율
total = len(df)
white_cnt = (df["today_firemask"] == white_img_path).sum()
ratio = (white_cnt / total) * 100

print(f"전체 {total}건 중, today_firemask == white.png : {white_cnt}건")
print(f"비율: {ratio:.2f}%")

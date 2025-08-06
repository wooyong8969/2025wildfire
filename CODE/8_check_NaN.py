import pandas as pd

original_path = 'firemask_modis_wind_weather.csv'
filled_path = 'firemask_modis_wind_weather_end_2day.csv'

df_ori = pd.read_csv(original_path)
df_new = pd.read_csv(filled_path)

ori_na = ((df_ori['NDVI'].isna()) | (df_ori['EVI'].isna())).sum()
new_na = ((df_new['NDVI'].isna()) | (df_new['EVI'].isna())).sum()

print(f'원본 {ori_na}')
print(f'보간 후 {new_na}')
print(f'-> {ori_na-new_na}')


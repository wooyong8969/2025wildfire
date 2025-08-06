import pandas as pd
from geopy.distance import geodesic

firemask_path = "firemask_modis_wind.csv"
station_info_path = "KMA_ASOS_observation_location_info.csv"
weather_data_path = "all_filtered_weather_data.csv"

fire_df = pd.read_csv(firemask_path)
weather_df = pd.read_csv(weather_data_path)
station_df = pd.read_csv(station_info_path)

station_df = station_df.rename(columns={
    "STN(ID)": "stn_id",
    "LAT(degree)": "lat",
    "LON(degee)": "lon",
    "HT(m)": "altitude"
})
station_df["stn_id"] = station_df["stn_id"].astype(int)
station_df = station_df[["stn_id", "lat", "lon", "altitude"]]

fire_df["lat_center"] = (fire_df["lat_min"] + fire_df["lat_max"]) / 2
fire_df["lon_center"] = (fire_df["lon_min"] + fire_df["lon_max"]) / 2

def find_nearest_station(lat, lon, station_df):
    distances = station_df.apply(
        lambda row: geodesic((lat, lon), (row["lat"], row["lon"])).km,
        axis=1
    )
    min_idx = distances.idxmin()
    return station_df.loc[min_idx, "stn_id"]

fire_df["stn_id"] = fire_df.apply(
    lambda row: find_nearest_station(row["lat_center"], row["lon_center"], station_df),
    axis=1
)

weather_df = weather_df.rename(columns={"DATE": "date", "STN": "stn_id"})
weather_df["date"] = pd.to_datetime(weather_df["date"]).dt.strftime("%Y-%m-%d")
fire_df["date"] = pd.to_datetime(fire_df["date"]).dt.strftime("%Y-%m-%d")

merged = pd.merge(fire_df, weather_df, on=["date", "stn_id"], how="left")

merged = pd.merge(merged, station_df, on="stn_id", how="left")

drop_cols = ["lat", "lon", "lat_center", "lon_center", "YYMMDD"]
merged = merged.drop(columns=[col for col in drop_cols if col in merged.columns])

merged.to_csv("firemask_modis_wind_weather.csv", index=False, encoding="utf-8-sig")
print("저장 완료: firemask_modis_wind_weather.csv")

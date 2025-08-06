import pandas as pd

df = pd.read_csv("firemask_metadata_all.csv")

filtered_df = df[df["prev_fire_point"] > 0].copy()

filtered_df.to_csv("firemask_metadata_filtering.csv", index=False, encoding="utf-8-sig")
print("저장 완료: firemask_metadata_spread_only.csv")

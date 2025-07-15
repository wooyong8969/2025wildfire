import geopandas as gpd
import numpy as np
from shapely.geometry import box
import os
import matplotlib.pyplot as plt
import pandas as pd

class FireData:
    def __init__(self, shapefile_path, output_dir):
        self.shapefile_path = shapefile_path
        self.output_dir = output_dir
        self.lat_min = 33
        self.lat_max = 38
        self.lon_min = 125
        self.lon_max = 130
        self.km_per_grid = 15       # 한 칸 크기 (km)
        self.img_size = 150         # 이미지 한 변의 픽셀수
        self.block = 4              # 칠할 블록 반경
        self.gdf = gpd.read_file(shapefile_path)
        os.makedirs(output_dir, exist_ok=True)

    def get_grid_gdf(self):
        lat_step = self.km_per_grid / 111
        lon_step = self.km_per_grid / 88.8
        lat_points = np.arange(self.lat_min, self.lat_max, lat_step)
        lon_points = np.arange(self.lon_min, self.lon_max, lon_step)
        grid_list = []
        for lon in lon_points:
            for lat in lat_points:
                grid_poly = box(lon, lat, lon + lon_step, lat + lat_step)
                grid_list.append({'geometry': grid_poly, 'lon_min': lon, 'lat_min': lat,
                                  'lon_max': lon + lon_step, 'lat_max': lat + lat_step})
        return gpd.GeoDataFrame(grid_list, crs="EPSG:4326")

    def process_date(self, acq_date):
        # 날짜 필터링
        df = self.gdf.copy()
        df['ACQ_DATE'] = df['ACQ_DATE'].astype(str)
        fire_points = df[df['ACQ_DATE'] == acq_date]
        if fire_points.empty:
            print(f"[{acq_date}] 해당 날짜 데이터 없음")
            return

        grid_gdf = self.get_grid_gdf()
        points_gdf = fire_points.set_geometry('geometry')
        joined = gpd.sjoin(points_gdf, grid_gdf, how='left', predicate='within')
        grid_counts = joined.groupby(['lon_min', 'lat_min', 'lon_max', 'lat_max']).size().reset_index(name='count')
        selected_grids = grid_counts[grid_counts['count'] >= 1]

        save_rows = []
        for idx, row in selected_grids.iterrows():
            lon0, lat0, lon1, lat1 = row['lon_min'], row['lat_min'], row['lon_max'], row['lat_max']
            box_poly = box(lon0, lat0, lon1, lat1)
            points_in_box = fire_points[fire_points.within(box_poly)]

            W, H = self.img_size, self.img_size
            img = np.ones((H, W, 3), dtype=np.uint8) * 255

            if not points_in_box.empty:
                x_idx = ((points_in_box.geometry.x - lon0) / (lon1 - lon0) * (W - 1)).astype(int)
                y_idx = ((lat1 - points_in_box.geometry.y) / (lat1 - lat0) * (H - 1)).astype(int)

                block = self.block
                for x, y in zip(x_idx, y_idx):
                    x0 = max(0, x - block)
                    x1 = min(W, x + block + 1)
                    y0 = max(0, y - block)
                    y1 = min(H, y + block + 1)
                    img[y0:y1, x0:x1] = [255, 60, 0]

            fname = f"grid_{acq_date}_lon{lon0:.5f}_lat{lat0:.5f}.png"
            out_path = os.path.join(self.output_dir, fname)
            plt.imsave(out_path, img)

            frp_vals = pd.to_numeric(points_in_box['FRP'], errors='coerce')
            save_rows.append({
                'filename': fname,                # 이미지 파일명
                'lon_min': lon0,                  # 칸 서쪽 경계 경도
                'lat_min': lat0,                  # 칸 남쪽 경계 위도
                'lon_max': lon1,                  # 칸 동쪽 경계 경도
                'lat_max': lat1,                  # 칸 북쪽 경계 위도
                'point_count': len(points_in_box), # 칸 내 산불 점 개수
                'acq_date': acq_date,             # 관측 날짜
                'frp_mean': frp_vals.mean(),      # 복사화력 평균
            })
            print(f"[{acq_date}] 저장 완료: {out_path}")
        csv_path = os.path.join(self.output_dir, f"firemask_metadata_{acq_date}.csv")
        pd.DataFrame(save_rows).to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"[{acq_date}] CSV 저장 완료: {csv_path}")

    def process_dates(self, date_list):
        for acq_date in date_list:
            self.process_date(acq_date)



shapefile = r"korea_firms/fire_archive_SV-C2_629794.shp"
output_dir = "firemask_output"

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

mask_gen = FireData(shapefile, output_dir)
mask_gen.process_dates(date_list)
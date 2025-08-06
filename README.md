# 🔥 산불 확산 예측 탐구 (Wildfire Spread Prediction)

> 본 프로젝트는 **충북과학고등학교**의 **「2025 제 2회 융합탐구 321체」**에 참여한  
> **코팅"좌" 팀**이 수행한 **교내 융합탐구 프로젝트**입니다.



## 📌 탐구 개요

산불은 예측이 어렵고 빠르게 확산되어 생명과 재산에 큰 피해를 줍니다. 본 탐구는  
위성 영상과 기상 데이터를 바탕으로 **AI 모델이 산불 확산 위치를 예측**할 수 있도록  
데이터셋을 구축하고, segmentation 모델을 학습 및 평가하는 것을 목표로 합니다.


## 📦 1. 데이터셋 구축 과정 (Dataset Construction)

### 🔨 단계별 처리 흐름

| 단계 | 내용 | 관련 코드 |
|------|------|-----------|
| 1 | NASA FIRMS 위성 데이터를 기반으로, 15km x 15km 격자 단위 산불 마스크(`today_firemask`) 이미지 생성 | `1_make_only_firedata.py` |
| 2 | 날짜별 메타데이터 통합 및 전날 마스크(`prev_firemask`) 매핑 | `2_merge_firemask.py` |
| 3 | 산불이 전날 있었던 경우만 필터링 → “산불 확산” 케이스로 제한 | `3_filtering_firemask.py` |
| 4 | MODIS 기반 NDVI, EVI, 반사율(red/nir/blue) 데이터 병합 | `4_merge_modis.py` |
| 5 | 풍향/풍속 정보(기상청 wind 위성자료) 병합 | `5_merge_wind.py` |
| 6 | 기상청 관측소 기온/강수/습도 및 고도 정보 병합 | `6_merge_weather.py` |
| 7 | NDVI/EVI 결측치 보간 처리 (±2일 이내 인접 격자 기반) | `7_fill_NaN.py` |

최종 데이터는 `firemask_modis_wind_weather_end_2day.csv`로 저장됩니다.

### 🗂️ 최종 데이터 구성 (CSV Columns)

| 열 이름 | 설명 |
|---------|------|
| `date` | 해당 일자 |
| `lat_min`, `lat_max` | 격자의 위도 범위 |
| `lon_min`, `lon_max` | 격자의 경도 범위 |
| `prev_fire_point` | 전날 화재 감지 점 수 |
| `today_point_count` | 당일 화재 감지 점 수 |
| `prev_frp_mean` | 전날 복사화력(FRP) 평균 |
| `today_frp_mean` | 당일 복사화력 평균 |
| `prev_firemask` | 전날 화재 마스크 이미지 경로 |
| `today_firemask` | 당일 화재 마스크 이미지 경로 (예측 대상) |
| `NDVI`, `EVI` | 식생 지수 |
| `surf_reflectance_red`, `nir`, `blue` | 표면 반사율 |
| `wind_dir`, `wind_speed` | 풍향, 풍속 |
| `stn_id` | 기상 관측소 ID |
| `TA_AVG`, `TA_MAX`, `TA_MIN` | 평균, 최고, 최저 기온 |
| `HM_AVG` | 평균 습도 |
| `RN_DAY` | 일일 강수량 |
| `altitude` | 고도 (관측소 기준) |

> 🔎 `prev_firemask`, `today_firemask`는 실제 이미지 파일 경로이며, 모델 학습 시 직접 불러오는 구조입니다.



## 🧠 2. 모델 학습 및 예측 (Model Training & Prediction)

### 🧩 모델 구조

- 입력:
  - 이미지: `prev_firemask` (1×150×150)
  - 수치 데이터: 총 15개 feature
- 아키텍처:
  - CNN encoder + MLP encoder → U-Net style decoder
- 출력:
  - 예측 마스크 `today_firemask` (binary segmentation)

모델 구조는 `MultiModalUNet`으로 정의되어 있으며, CNN과 MLP의 출력 특성을 병합하여 복원하는 구조입니다.

### 📚 학습 방법

- 손실 함수: `BCEWithLogitsLoss`
- Optimizer: `Adam`
- Epochs: 20
- Batch size: 8


→ 학습 완료 후 `firemask_model.pth`로 모델 저장

### 🔍 예측 및 시각화

파일: `11_predict_data.py`

- 특정 행 선택하여 예측 마스크 생성
- 실제 마스크와 비교하여 IoU, Dice Score 산출
- 시각화 출력 (입력 / 예측 / 정답 마스크 비교)



## 🖼️ 예측 결과 예시 (Visualization Examples)

아래는 모델이 전날 화재 마스크와 기상 정보를 입력받아, 당일 산불 마스크를 예측한 결과입니다.



## 📂 폴더 구조

```
2025WILDFIRE/
├── CODE/
│   ├── 1_make_only_firedata.py
│   ├── 2_merge_firemask.py
│   ├── 3_filtering_firemask.py
│   ├── 4_merge_modis.py
│   ├── 5_merge_wind.py
│   ├── 6_merge_weather.py
│   ├── 7_fill_NaN.py
│   ├── 8_check_NaN.py
│   ├── 9_check_white.py
│   ├── 10_firemask_segmentation.py   ← 모델 학습 코드
│   ├── 11_predict_data.py            ← 예측 및 시각화
│   └── firemask_segmentation.py      ← (10번 코드와 동일)
│
├── firemask_output/                 ← 생성된 마스크 이미지
│
├── firemask_model.pth               ← 학습된 모델 가중치
└── firemask_modis_wind_weather_end_2day.csv ← 최종 데이터셋

```



## 📚 참고 자료

- NASA FIRMS: https://firms.modaps.eosdis.nasa.gov/
- MODIS Vegetation Indices: https://noaa-cdr-ndvi-pds.s3.amazonaws.com/index.html#data/
- 기상청 날씨 및 관측소 데이터 https://data.kma.go.kr/data/grnd/selectAsosRltmList.do
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from firemask_segmentation import MultiModalUNet

# === 1. 설정 ===
image_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((150, 150)),
    transforms.ToTensor()
])

def load_image(path):
    img = Image.open(path)
    return image_transform(img).squeeze().numpy()  # (150, 150)

def load_mask(path):
    mask = Image.open(path)
    mask = image_transform(mask).squeeze().numpy()
    return (mask > 0.5).astype(float)

# === 2. 예측 함수 ===
def predict_single_row(row, model):
    image = Image.open(row["prev_firemask"])
    image_tensor = image_transform(image).unsqueeze(0)  # (1, 1, 150, 150)

    numeric_cols = [
        'prev_fire_point', 'prev_frp_mean',
        'NDVI', 'EVI',
        'surf_reflectance_red', 'surf_reflectance_nir', 'surf_reflectance_blue',
        'wind_dir', 'wind_speed',
        'TA_AVG', 'TA_MAX', 'TA_MIN',
        'HM_AVG', 'RN_DAY', 'altitude'
    ]
    numeric_vals = torch.tensor([float(row[col]) for col in numeric_cols], dtype=torch.float32).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(image_tensor, numeric_vals)
        prob_mask = torch.sigmoid(output).squeeze().numpy()
        binary_mask = (prob_mask > 0.5).astype(float)

    return binary_mask, prob_mask

# === 3. 평가 함수 ===
def compute_iou(pred, target):
    intersection = (pred * target).sum()
    union = ((pred + target) > 0).sum()
    return intersection / union if union > 0 else 1.0

def compute_dice(pred, target):
    intersection = (pred * target).sum()
    return (2 * intersection) / (pred.sum() + target.sum()) if (pred.sum() + target.sum()) > 0 else 1.0

# === 4. 실행 ===
if __name__ == "__main__":
    # 1. CSV 로드
    df = pd.read_csv("firemask_modis_wind_weather_end_2day.csv")

    # 2. 예측할 이미지 이름
    targets = [
        "grid_2022-03-04_lon129.22297_lat37.05405.png",
        "grid_2022-03-05_lon128.88514_lat37.45946.png",
        "grid_2022-03-04_lon129.22297_lat36.91892.png"

    ]

    # 3. 해당 행 필터링
    matched = df[df["prev_firemask"].apply(lambda x: any(t in x for t in targets))]

    if matched.empty:
        print("대상 이미지가 포함된 행을 찾지 못했습니다.")
        exit()

    # 4. 모델 로드
    model = MultiModalUNet(num_numeric_features=15)
    model.load_state_dict(torch.load("firemask_model.pth", map_location=torch.device('cpu')))

    # 5. 예측 및 비교
    for idx, row in matched.iterrows():
        pred_mask, prob_mask = predict_single_row(row, model)
        true_mask = load_mask(row["today_firemask"])
        prev_img = load_image(row["prev_firemask"])

        # 지표 계산
        iou = compute_iou(pred_mask, true_mask)
        dice = compute_dice(pred_mask, true_mask)

        # 시각화
        plt.figure(figsize=(16, 4))

        plt.subplot(1, 3, 1)
        plt.title("Prev Firemask (input)")
        plt.imshow(prev_img, cmap='gray')

        plt.subplot(1, 3, 2)
        plt.title("Predicted Mask")
        plt.imshow(pred_mask, cmap='gray')

        plt.subplot(1, 3, 3)
        plt.title("True Mask")
        plt.imshow(true_mask, cmap='gray')

        plt.suptitle(f"File: {row['prev_firemask'].split('/')[-1]}")
        plt.tight_layout()
        plt.show()

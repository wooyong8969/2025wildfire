# firemask_segmentation.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np

# -----------------------------
# 1. Dataset 정의
# -----------------------------
class FiremaskDataset(Dataset):
    def __init__(self, csv_path, image_size=(150, 150)):
        self.data = pd.read_csv(csv_path)

        # 결측치 제거
        self.numeric_cols = [
            'prev_fire_point', 'prev_frp_mean',
            'NDVI', 'EVI',
            'surf_reflectance_red', 'surf_reflectance_nir', 'surf_reflectance_blue',
            'wind_dir', 'wind_speed',
            'TA_AVG', 'TA_MAX', 'TA_MIN',
            'HM_AVG', 'RN_DAY', 'altitude'
        ]
        self.data = self.data.dropna(subset=self.numeric_cols + ['prev_firemask', 'today_firemask']).reset_index(drop=True)

        self.image_size = image_size
        self.image_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

        self.mask_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Load prev_firemask
        prev_mask_path = row['prev_firemask']
        prev_mask = Image.open(prev_mask_path)
        prev_mask = self.image_transform(prev_mask)

        # Load today_firemask (target)
        today_mask_path = row['today_firemask']
        today_mask = Image.open(today_mask_path)
        today_mask = self.mask_transform(today_mask)
        today_mask = (today_mask > 0.5).float()  # 255 -> 1, binary

        # Numeric features
        numeric_values = [float(row[col]) for col in self.numeric_cols]
        numeric = torch.tensor(numeric_values, dtype=torch.float32)

        return prev_mask, numeric, today_mask

# -----------------------------
# 2. 모델 정의 (CNN + MLP + Decoder)
# -----------------------------
class MultiModalUNet(nn.Module):
    def __init__(self, num_numeric_features):
        super().__init__()

        # CNN encoder for image (input: 1x150x150)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 75x75
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 37x37
        )

        # MLP for numeric input
        self.mlp = nn.Sequential(
            nn.Linear(num_numeric_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Project numeric feature to image shape and concat
        self.numeric_to_image = nn.Linear(32, 32 * 37 * 37)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),  # 37 -> 74
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 2, stride=2, output_padding=1),  # 74 -> 149
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1)  # keep at 149x149
        )

    def forward(self, image, numeric):
        x_img = self.encoder(image)  # (B, 32, 37, 37)
        x_num = self.mlp(numeric)    # (B, 32)
        x_num_proj = self.numeric_to_image(x_num).view(-1, 32, 37, 37)

        x_combined = torch.cat([x_img, x_num_proj], dim=1)  # (B, 64, 37, 37)
        out = self.decoder(x_combined)  # (B, 1, 149, 149)

        # 최종 출력 크기 보정
        out = F.interpolate(out, size=(150, 150), mode='bilinear', align_corners=False)
        return out

# -----------------------------
# 3. 학습 루프
# -----------------------------
def train_model(csv_path, num_epochs=10, batch_size=8, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = FiremaskDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MultiModalUNet(num_numeric_features=15).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for img, numeric, mask in dataloader:
            img, numeric, mask = img.to(device), numeric.to(device), mask.to(device)

            optimizer.zero_grad()
            output = model(img, numeric)
            loss = criterion(output, mask)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

    # 모델 저장
    torch.save(model.state_dict(), "firemask_model.pth")
    print("\n 모델 학습 및 저장 완료")

# -----------------------------
# 4. 실행
# -----------------------------
if __name__ == "__main__":
    train_model("firemask_modis_wind_weather_end_2day.csv", num_epochs=20)

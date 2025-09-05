import pygame
import random
import math
import torch
from torchvision import transforms
from PIL import Image
from firemask_segmentation import MultiModalUNet

# --- 설정 ---
TILE_SIZE = 4
GRID_W, GRID_H = 150, 150
MAP_WIDTH = TILE_SIZE * GRID_W
PANEL_WIDTH = 300
WINDOW_W = MAP_WIDTH + PANEL_WIDTH
WINDOW_H = TILE_SIZE * GRID_H

BRUSH_RADIUS = 7  # 그리드 단위 반경
MODEL_PATH = "GAME1/firemask_model.pth"
MAX_ROUNDS = 1

# 색상
COLOR_BG = (255, 255, 255)
COLOR_FIRE = (255, 50, 50, 230)
COLOR_PREDICT = (60, 255, 100, 100)
COLOR_AI = (255, 215, 0, 100)

pygame.init()
screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
pygame.display.set_caption("Wildfire Prediction - IoU Scoring")
clock = pygame.time.Clock()
font = pygame.font.Font("C:/Windows/Fonts/malgun.ttf", 16)

# --- 상태 ---
fire_grid = [[0 for _ in range(GRID_W)] for _ in range(GRID_H)]
predict_grid = [[0 for _ in range(GRID_W)] for _ in range(GRID_H)]
ai_bin = [[0 for _ in range(GRID_W)] for _ in range(GRID_H)]
show_results = False
mouse_down = False
erase_mode = False

# 라운드 관리
current_round = 1
round_scores = []
total_intersection = 0
total_union = 0
game_over = False

# 15개 수치
numeric_feature_names = [
    '이전 화재 지점 수', '이전 FRP 평균',
    'NDVI', 'EVI',
    '적색 반사율', '근적외선 반사율', '청색 반사율',
    '풍향', '풍속',
    '평균기온', '최고기온', '최저기온',
    '평균습도', '일강수량', '고도'
]
numeric_feature_values = [0.0] * 15

# 모델 준비
device = torch.device("cpu")
model = MultiModalUNet(num_numeric_features=15)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

image_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((150, 150)),
    transforms.ToTensor()
])


def random_features():
    global numeric_feature_values
    numeric_feature_values = [round(random.uniform(0, 1), 3) for _ in range(15)]


def generate_clusters(target_pixels):
    filled = 0
    visited = set()
    while filled < target_pixels:
        sx, sy = random.randint(0, GRID_W - 1), random.randint(0, GRID_H - 1)
        cluster_target = random.randint(200, 800)
        queue = [(sx, sy)]
        while queue and filled < target_pixels and cluster_target > 0:
            x, y = queue.pop(0)
            if (x, y) in visited:
                continue
            visited.add((x, y))
            fire_grid[y][x] = 1
            filled += 1
            cluster_target -= 1
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (1, 1), (-1, 1), (1, -1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_W and 0 <= ny < GRID_H and (nx, ny) not in visited:
                    if random.random() < 0.5:
                        queue.append((nx, ny))


def grid_to_image(grid):
    img = Image.new("L", (GRID_W, GRID_H))
    for y in range(GRID_H):
        for x in range(GRID_W):
            val = 255 if grid[y][x] == 1 else 0
            img.putpixel((x, y), val)
    return img


def run_ai_prediction():
    global ai_bin
    prev_img = grid_to_image(fire_grid)
    img_tensor = image_transform(prev_img).unsqueeze(0)
    numeric_tensor = torch.tensor(numeric_feature_values, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = model(img_tensor, numeric_tensor)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    ai_bin = (prob > 0.5).astype(int).tolist()


def reset_game():
    global fire_grid, predict_grid, ai_bin, show_results
    fire_grid = [[0 for _ in range(GRID_W)] for _ in range(GRID_H)]
    predict_grid = [[0 for _ in range(GRID_W)] for _ in range(GRID_H)]
    ai_bin = [[0 for _ in range(GRID_W)] for _ in range(GRID_H)]
    show_results = False
    random_features()
    target_pixels = random.randint(int(GRID_W * GRID_H * 0.4), int(GRID_W * GRID_H * 0.7))
    generate_clusters(target_pixels)
    run_ai_prediction()


def draw_map():
    for y in range(GRID_H):
        for x in range(GRID_W):
            rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            pygame.draw.rect(screen, COLOR_BG, rect)

            if fire_grid[y][x] == 1:
                surf = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
                surf.fill(COLOR_FIRE)
                screen.blit(surf, rect.topleft)

            if predict_grid[y][x] == 1:
                surf = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
                surf.fill(COLOR_PREDICT)
                screen.blit(surf, rect.topleft)

            if show_results and ai_bin[y][x] == 1:
                surf = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
                surf.fill(COLOR_AI)
                screen.blit(surf, rect.topleft)


def draw_panel():
    panel_x = MAP_WIDTH + 10
    y_offset = 10

    screen.blit(font.render(f"Round: {current_round}/{MAX_ROUNDS}", True, (0, 0, 0)),
                (panel_x, y_offset))
    y_offset += 20

    if show_results:
        screen.blit(font.render(f"Round Score: {round_scores[-1]:.5f}%", True, (0, 0, 0)),
                    (panel_x, y_offset))
        y_offset += 20

    if game_over:
        final_score = (total_intersection / total_union * 100) if total_union > 0 else 0
        screen.blit(font.render(f"Final IoU Score: {final_score:.5f}%", True, (0, 0, 0)),
                    (panel_x, y_offset))
        y_offset += 30

    for name, val in zip(numeric_feature_names, numeric_feature_values):
        screen.blit(font.render(f"{name}: {val}", True, (0, 0, 0)), (panel_x, y_offset))
        y_offset += 18

    y_offset += 10
    instructions = [
        "그리기: 마우스 좌클릭 후 드래그",
        "지우기: 마우스 우클릭 후 드래그",
        "제출: Enter",
        "다음 라운드: L",
        "리셋: R"
    ]
    for line in instructions:
        screen.blit(font.render(line, True, (0, 0, 0)), (panel_x, y_offset))
        y_offset += 18


def add_prediction(cx, cy):
    for dy in range(-BRUSH_RADIUS, BRUSH_RADIUS + 1):
        for dx in range(-BRUSH_RADIUS, BRUSH_RADIUS + 1):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                if math.sqrt(dx ** 2 + dy ** 2) <= BRUSH_RADIUS:
                    predict_grid[ny][nx] = 1


def erase_prediction(cx, cy):
    for dy in range(-BRUSH_RADIUS, BRUSH_RADIUS + 1):
        for dx in range(-BRUSH_RADIUS, BRUSH_RADIUS + 1):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
                if math.sqrt(dx ** 2 + dy ** 2) <= BRUSH_RADIUS:
                    predict_grid[ny][nx] = 0


# 초기화
reset_game()

# 메인 루프
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button in (1, 3):
                mouse_down = True
                erase_mode = (event.button == 3)
                mx, my = event.pos
                if mx < MAP_WIDTH:
                    gx, gy = mx // TILE_SIZE, my // TILE_SIZE
                    if erase_mode:
                        erase_prediction(gx, gy)
                    else:
                        add_prediction(gx, gy)

        elif event.type == pygame.MOUSEBUTTONUP and event.button in (1, 3):
            mouse_down = False

        elif event.type == pygame.MOUSEMOTION and mouse_down:
            mx, my = event.pos
            if mx < MAP_WIDTH:
                gx, gy = mx // TILE_SIZE, my // TILE_SIZE
                if erase_mode:
                    erase_prediction(gx, gy)
                else:
                    add_prediction(gx, gy)

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                current_round = 1
                round_scores.clear()
                total_intersection = 0
                total_union = 0
                game_over = False
                reset_game()

            elif event.key == pygame.K_RETURN and not show_results and not game_over:
                # IoU 계산
                intersection = sum(1 for y in range(GRID_H) for x in range(GRID_W)
                                   if predict_grid[y][x] == 1 and ai_bin[y][x] == 1)
                union = sum(1 for y in range(GRID_H) for x in range(GRID_W)
                            if predict_grid[y][x] == 1 or ai_bin[y][x] == 1)

                round_score = (intersection / union * 100) if union > 0 else 0
                round_scores.append(round_score)

                total_intersection += intersection
                total_union += union

                show_results = True

            elif event.key == pygame.K_l and show_results and not game_over:
                if current_round < MAX_ROUNDS:
                    current_round += 1
                    reset_game()
                else:
                    game_over = True

    screen.fill(COLOR_BG)
    draw_map()
    draw_panel()

    pygame.display.flip()
    clock.tick(60)

pygame.quit()

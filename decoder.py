#!/usr/bin/env python3
"""
Microform Code Decoder — v2 prototype
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Pipeline:
  1. Load image → grayscale → threshold
  2. Find 4 corner markers (2×2 black squares) → 4 correspondence points
  3. Compute homography → warp to normalized grid
  4. Sample each data cell → extract 18-bit value via PCA + blob analysis
  5. Pack bits → RS decode → user data

Usage:
  python decoder.py microform_demo.png

Dependencies:
  pip install opencv-python numpy
"""

import sys
import math
import numpy as np
import cv2

# ── Layout (в ячейках) — должно совпадать с generator.py ─────────────────
MARKER_SIZE = 2
BORDER_WIDTH = 1
PADDING = 1
OFF = MARKER_SIZE + BORDER_WIDTH + PADDING  # = 4


# ── 1. Детекция маркеров ─────────────────────────────────────────────────
def _find_markers(gray: np.ndarray):
    """
    Найти 4 угловых маркера (2×2 чёрных квадрата).
    
    Стратегия: erosion убирает мелкие элементы данных,
    остаются только крупные блоки (маркеры). Ищем 4 квадрата.
    
    Returns:
        list[(cx, cy)] — 4 центра маркеров в порядке TL, TR, BL, BR
        или None, если не найдены.
    """
    h, w = gray.shape

    # Otsu threshold — инвертируем: чёрное → 255 (foreground)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Erosion: убираем данные (маленькие эллипсы/точки),
    # оставляем только солидные маркеры 18×18
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Оценка ожидаемого размера маркера
    # После 2× erosion'ов: 18×18 → ~14×14 → area ~170–225
    marker_area_min = 80
    marker_area_max = (w * h) * 0.04  # не больше 4% картинки

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < marker_area_min or area > marker_area_max:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = max(bw, bh) / max(min(bw, bh), 1)
        if aspect > 1.4:
            continue
        cx = float(x + bw // 2)
        cy = float(y + bh // 2)
        candidates.append((cx, cy, area))

    if len(candidates) < 4:
        return None

    # Сортируем по площади (сначала крупные)
    candidates.sort(key=lambda c: -c[2])
    top = candidates[:min(12, len(candidates))]

    # Распределяем по 4 квадрантам изображения
    mid_x, mid_y = w / 2.0, h / 2.0

    quads = {'tl': None, 'tr': None, 'bl': None, 'br': None}
    for cx, cy, area in top:
        if cx < mid_x and cy < mid_y:
            key = 'tl'
        elif cx >= mid_x and cy < mid_y:
            key = 'tr'
        elif cx < mid_x and cy >= mid_y:
            key = 'bl'
        else:
            key = 'br'
        if quads[key] is None or area > quads[key][2]:
            quads[key] = (cx, cy, area)

    if not all(quads.values()):
        return None

    return [(quads['tl'][0], quads['tl'][1]),
            (quads['tr'][0], quads['tr'][1]),
            (quads['bl'][0], quads['bl'][1]),
            (quads['br'][0], quads['br'][1])]


# ── 2. Перспективная коррекция → exact cell grid ───────────────────────
def _warp_and_locate(gray: np.ndarray, markers: list,
                     grid_h: int, grid_w: int,
                     cell_px_out: int = 32) -> tuple:
    """
    Выровнять перспективу. Маркеры маппятся на известные клеточные
    координаты → data-область оказывается в строго известном месте.
    
    Returns:
        (warped_full, data_x, data_y, cell_px_out)
    """
    # Координаты центров маркеров в клетках (из generator.py):
    # TL center = (PADDING + MARKER_SIZE/2, PADDING + MARKER_SIZE/2) = (2, 2)
    # TR center = (grid_w + 2*OFF - PADDING - MARKER_SIZE/2, 2) = (grid_w + 6, 2)
    # BL center = (2, grid_h + 6)
    # BR center = (grid_w + 6, grid_h + 6)
    
    src_pts = np.array(markers, dtype=np.float32)
    
    # Маркеры в пикселях выходного изображения (без margin)
    dst_pts = np.array([
        [ 2 * cell_px_out,  2 * cell_px_out],
        [(grid_w + 6) * cell_px_out,  2 * cell_px_out],
        [ 2 * cell_px_out, (grid_h + 6) * cell_px_out],
        [(grid_w + 6) * cell_px_out, (grid_h + 6) * cell_px_out],
    ], dtype=np.float32)

    # Полный размер кода в ячейках
    full_cells_w = grid_w + 2 * OFF
    full_cells_h = grid_h + 2 * OFF
    full_px_w = full_cells_w * cell_px_out
    full_px_h = full_cells_h * cell_px_out

    # Добавляем margin со всех сторон
    margin = cell_px_out * 2
    dst_margin = dst_pts + margin  # сдвигаем все точки

    H, _ = cv2.findHomography(src_pts, dst_margin)
    if H is None:
        return None, 0, 0, 0

    out_w = int(full_px_w + 2 * margin)
    out_h = int(full_px_h + 2 * margin)
    warped = cv2.warpPerspective(gray, H, (out_w, out_h),
                                 flags=cv2.INTER_NEAREST)

    # Data-область: в ячейках от (OFF, OFF) до (OFF+grid_w, OFF+grid_h)
    # В пикселях: margin + OFF * cell_px_out
    data_x = int(margin + OFF * cell_px_out)
    data_y = int(margin + OFF * cell_px_out)
    
    return warped, data_x, data_y, cell_px_out


# ── 3. Чтение ячейки: 18 бит через PCA + моменты ──────────────────────
def _cell_pca_bits(cell: np.ndarray) -> int:
    """
    Анализ ячейки 9×9 (или любого размера) → 18-битное значение.
    
    Использует PCA тёмных пикселей для определения параметров эллипса
    и анализ вторичного скопления для микро-точки.
    """
    h, w = cell.shape
    total_px = h * w

    # ── Бинаризация внутри ячейки ──────────────────────────────────────
    # Используем Otsu только на этой ячейке
    _, binary = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Координаты тёмных пикселей (с весами по интенсивности)
    dark_mask = binary > 0
    dark_count = np.sum(dark_mask)

    if dark_count < 2:
        # Ячейка пуста — возвращаем 0
        return 0

    ys, xs = np.where(dark_mask)

    if dark_count < 3:
        # Слишком мало пикселей для PCA — просто угол 0
        # Вычисляем только dot и scale
        xm, ym = xs.mean(), ys.mean()
        cx, cy = w / 2.0, h / 2.0
        dot_dx = int(np.clip(np.round((xm - cx) / (w * 0.3)), -3, 3) + 3)
        dot_dy = int(np.clip(np.round((ym - cy) / (h * 0.3)), -3, 3) + 3)
        coverage = dark_count / total_px
        rad = min(3, int(coverage * 6))
        scale = min(3, int(coverage * 5))
        return (0 << 12) | (scale << 10) | (0 << 8) | (dot_dx << 5) | (dot_dy << 2) | rad

    # ── PCA для основного эллипса ──────────────────────────────────────
    # Используем интенсивность как вес
    weights = 255.0 - cell[dark_mask].astype(np.float32) + 1.0
    mean = np.array([xs.mean(), ys.mean()])

    # Ковариационная матрица (взвешенная)
    diffs = np.column_stack([xs, ys]) - mean
    cov = np.dot((diffs * weights[:, np.newaxis]).T, diffs) / weights.sum()

    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Первая компонента (главная ось)
    if eigenvalues[1] > 0:
        angle_rad = math.atan2(eigenvectors[1, 1], eigenvectors[0, 1])
        angle_deg = math.degrees(angle_rad) % 180
        angle_v = int(np.clip(round(angle_deg / 179.0 * 63), 0, 63))

        # Вытянутость: sqrt(eigenvalue_max / eigenvalue_min)
        elong_raw = math.sqrt(eigenvalues[1] / max(eigenvalues[0], 0.1))
        elong_v = int(np.clip(round((elong_raw - 1.0) / 1.5 * 3), 0, 3))
    else:
        angle_v = 0
        elong_v = 0

    # ── Масштаб: доля тёмных пикселей ─────────────────────────────────
    coverage = dark_count / total_px
    scale_v = int(np.clip(round((coverage - 0.1) / 0.7 * 3), 0, 3))

    # ── Микро-точка: центр масс → отклонение от центра ячейки ────────
    cx, cy = w / 2.0, h / 2.0
    dx_px = mean[0] - cx
    dy_px = mean[1] - cy

    # Нормализуем в диапазон 0..7
    range_px = w * 0.35
    dot_dx = int(np.clip(round(dx_px / range_px * 3.5 + 3), 0, 7))
    dot_dy = int(np.clip(round(dy_px / range_px * 3.5 + 3), 0, 7))

    # Радиус точки: среднее расстояние тёмных пикселей от центра масс
    if dark_count > 3:
        dists = np.sqrt((xs - mean[0]) ** 2 + (ys - mean[1]) ** 2)
        mean_radius = np.average(dists, weights=weights)
        rad = int(np.clip(round((mean_radius - 1.0) / 3.0 * 3), 0, 3))
    else:
        rad = 0

    # ── Собираем 18 бит ──────────────────────────────────────────────
    value = (angle_v << 12) | (scale_v << 10) | (elong_v << 8) \
            | (dot_dx << 5) | (dot_dy << 2) | rad
    return value


# ── 4. Главная функция декодера ────────────────────────────────────────
def decode_image(image_path: str,
                 grid_h: int = 150, grid_w: int = 150) -> bytes | None:
    """
    Декодировать Microform Code из изображения.
    
    Returns:
        bytes — раскодированные пользовательские данные
        None — если декодирование не удалось
    """
    from codec import decode_bits, BITS_PER_CELL

    # ── Загрузить ──────────────────────────────────────────────────────
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[ERROR] Не могу загрузить: {image_path}")
        return None

    h, w = img.shape
    print(f"[Decode] Загружено: {w}x{h} пикселей")

    # ── Детекция маркеров ────────────────────────────────────────────
    markers = _find_markers(img)
    if markers is None:
        print("[ERROR] Не найдены 4 угловых маркера")
        return None

    print(f"[Decode] Маркеры найдены:")
    names = ["TL", "TR", "BL", "BR"]
    for name, pt in zip(names, markers):
        print(f"  {name}: ({pt[0]:.0f}, {pt[1]:.0f})")

    # ── Перспективная коррекция ──────────────────────────────────────
    result = _warp_and_locate(img, markers, grid_h, grid_w)
    if result[0] is None:
        print("[ERROR] Ошибка гомографии")
        return None
    
    warped, dx, dy, cell_px_out = result
    wh, ww = warped.shape
    print(f"[Decode] Warp: {ww}x{wh} px, data offset=({dx},{dy})")
    print(f"[Decode] Ячейка в warp: {cell_px_out} px")

    # ── Читаем каждую ячейку ──────────────────────────────────────────
    total_cells = grid_h * grid_w
    bits = []
    errors = 0
    cell_sz = cell_px_out

    for row in range(grid_h):
        for col in range(grid_w):
            y1 = int(dy + row * cell_sz)
            x1 = int(dx + col * cell_sz)
            y2 = int(y1 + cell_sz)
            x2 = int(x1 + cell_sz)

            # Безопасные границы
            y1 = max(0, y1); x1 = max(0, x1)
            y2 = min(wh, y2); x2 = min(ww, x2)

            cell_img = warped[y1:y2, x1:x2]
            if cell_img.size == 0:
                bits.extend([0] * 18)
                errors += 1
                continue

            value = _cell_pca_bits(cell_img)

            for b in range(18):
                bits.append((value >> (17 - b)) & 1)

    print(f"[Decode] Прочитано {len(bits)} бит, {errors} ошибок экстракции")

    # ── RS-декодирование ──────────────────────────────────────────────
    result = decode_bits(bits)
    if result:
        data_str = result.decode('utf-8', errors='replace')
        print(f"[Decode] OK: {len(result)} bytes decoded")
        print(f"[Decode] Data: {data_str[:200]}")
        return result
    else:
        print("[Decode] FAIL: RS decoding returned nothing")
        return None


# ── CLI ──────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Microform Code Decoder v2")
    parser.add_argument("image", help="путь к изображению")
    parser.add_argument("--grid", type=int, default=150,
                        help="размер сетки (default: 150)")
    args = parser.parse_args()

    result = decode_image(args.image, args.grid, args.grid)
    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    main()

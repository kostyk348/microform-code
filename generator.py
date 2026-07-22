#!/usr/bin/env python3
"""
Microform Code Generator — Iteration 1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Генерирует паттерн из микро-эллипсов для печати.
Каждый эллипс кодирует 10 бит: угол(6) + размер(2) + вытянутость(2).

  • Угол:      0-179°  (64 шага, ~2.8° точность)  → 6 бит
  • Размер:    4 уровня (35-75% ячейки)           → 2 бита
  • Вытянутость: 4 уровня (1.0-2.5×)               → 2 бита
                        ──────────────────────────
                         10 бит/эллипс

Использование:
  python generator.py --data "Hello, world!" -o test.png
  python generator.py --grid 100 --dpi 600 -o code.png
"""

import argparse
import hashlib
import struct
import numpy as np
from PIL import Image

# ── Константы ─────────────────────────────────────────────────────────────
BITS_PER_CELL = 10
ANGLE_BITS = 6
SIZE_BITS = 2
ELONG_BITS = 2

# Служебные элементы (в ячейках)
MARKER_SIZE = 3        # угловые маркеры (ячеек)
BORDER_WIDTH = 2       # калибровочная рамка (ячеек)
PADDING = 1            # отступ по краям (ячеек) для белого поля
# ──────────────────────────────────────────────────────────────────────────────


# ── Битовые операции ────────────────────────────────────────────────────────
def _bytes_to_bits(data: bytes) -> list[int]:
    bits = []
    for b in data:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    return bits


def _pack_cell_value(user_bits: list[int], offset: int) -> int:
    """Достать BITS_PER_CELL бит начиная с offset (с паддингом нулями)."""
    val = 0
    for i in range(BITS_PER_CELL):
        pos = offset + i
        if pos < len(user_bits):
            val = (val << 1) | user_bits[pos]
        else:
            val <<= 1
    return val


# ── Маппинг: значение → параметры эллипса ─────────────────────────────────
def _angle_from_val(val: int) -> float:
    """6-bit → угол в градусах (0-179)."""
    return (val / 63.0) * 179.0


def _scale_from_val(val: int) -> float:
    """2-bit → масштаб (0.35-0.75 от ячейки)."""
    return 0.35 + val * 0.133


def _elong_from_val(val: int) -> float:
    """2-bit → вытянутость (1.0-2.5)."""
    return 1.0 + val * 0.5


# ── Обратный маппинг (для декодера) ──────────────────────────────────────
def val_from_angle(deg: float) -> int:
    return min(63, round(deg / 179.0 * 63))


def val_from_scale(scale: float) -> int:
    return max(0, min(3, round((scale - 0.35) / 0.133)))


def val_from_elong(elong: float) -> int:
    return max(0, min(3, round((elong - 1.0) / 0.5)))


# ── 2D-генератор эллипса ─────────────────────────────────────────────────
def _make_ellipse_mask(size_px: int, angle_deg: float,
                       scale: float, elong: float) -> np.ndarray:
    """
    Создать маску эллипса размером size_px × size_px.
    Возвращает uint8: 0 = фон, 255 = эллипс.
    """
    # Используем расстояние до границы эллипса с антиалиасингом
    half = size_px / 2.0
    yy, xx = np.ogrid[-half:half, -half:half]

    cos_a = np.cos(np.radians(angle_deg))
    sin_a = np.sin(np.radians(angle_deg))

    # Поворот координат
    x_rot = xx * cos_a + yy * sin_a
    y_rot = -xx * sin_a + yy * cos_a

    # Полуоси
    a = half * scale
    b = a / elong

    # Расстояние до границы (нормализованное)
    dist = (x_rot / a) ** 2 + (y_rot / b) ** 2

    # Плавный край (~0.5 px для антиалиасинга)
    alpha = np.clip(1.0 - (dist - 0.85) / 0.3, 0.0, 1.0)
    return (alpha * 255).astype(np.uint8)


# ── Генератор паттерна ───────────────────────────────────────────────────
def generate(grid_h: int, grid_w: int, cell_size_px: int,
             data: bytes | None = None) -> np.ndarray:
    """
    Сгенерировать Microform Code как numpy-массив (uint8, 0-255).

    Args:
        grid_h: число ячеек данных по вертикали
        grid_w: число ячеек данных по горизонтали
        cell_size_px: размер ячейки в пикселях
        data: байты для кодирования (None = случайные)

    Returns:
        np.ndarray (H, W), uint8: 0=чёрный, 255=белый
    """
    off = MARKER_SIZE + BORDER_WIDTH + PADDING
    total_h = grid_h + 2 * off
    total_w = grid_w + 2 * off

    # ── Подготовка данных ───────────────────────────────────────────────
    bits_per_data = grid_h * grid_w * BITS_PER_CELL
    if data is None:
        # Детерминированные псевдослучайные данные
        seed = struct.pack(">II", grid_h, grid_w)
        rng = np.random.default_rng(
            int.from_bytes(hashlib.sha256(seed).digest()[:8], "big")
        )
        data_bytes = rng.bytes(bits_per_data // 8 + 1)
        data_bits = _bytes_to_bits(data_bytes)
    else:
        data_bits = _bytes_to_bits(data)
        while len(data_bits) < bits_per_data:
            data_bits.append(0)

    data_bits = data_bits[:bits_per_data]

    # ── Создаём полотно ─────────────────────────────────────────────────
    canvas = np.ones((total_h * cell_size_px, total_w * cell_size_px),
                     dtype=np.uint8) * 255

    def _cell_slice(cx: int, cy: int):
        """(cx, cy) → slice по y, slice по x для доступа к ячейке."""
        y1 = cy * cell_size_px
        x1 = cx * cell_size_px
        return (slice(y1, y1 + cell_size_px),
                slice(x1, x1 + cell_size_px))

    def _draw_cell(cx: int, cy: int, value: int):
        """Нарисовать эллипс в ячейке (cx, cy) по value (0-1023)."""
        angle_v = (value >> (SIZE_BITS + ELONG_BITS)) & ((1 << ANGLE_BITS) - 1)
        size_v = (value >> ELONG_BITS) & ((1 << SIZE_BITS) - 1)
        elong_v = value & ((1 << ELONG_BITS) - 1)

        angle = _angle_from_val(angle_v)
        scale = _scale_from_val(size_v)
        elong = _elong_from_val(elong_v)

        mask = _make_ellipse_mask(cell_size_px, angle, scale, elong)
        sy, sx = _cell_slice(cx, cy)
        # Multiply: black on white (255 - mask_dark)
        canvas[sy, sx] = np.minimum(canvas[sy, sx],
                                    255 - mask).astype(np.uint8)

    # ── 1. Белый фон уже есть ───────────────────────────────────────────

    # ── 2. Угловые маркеры (чёрные квадраты 3×3) ────────────────────────
    corners = [
        (PADDING, PADDING),
        (total_w - MARKER_SIZE - PADDING, PADDING),
        (PADDING, total_h - MARKER_SIZE - PADDING),
        (total_w - MARKER_SIZE - PADDING, total_h - MARKER_SIZE - PADDING),
    ]
    for cx, cy in corners:
        for dy in range(MARKER_SIZE):
            for dx in range(MARKER_SIZE):
                sy, sx = _cell_slice(cx + dx, cy + dy)
                canvas[sy, sx] = 0

    # ── 3. Калибровочная рамка (шахматка) ──────────────────────────────
    b_start = MARKER_SIZE + PADDING
    b_end_w = total_w - MARKER_SIZE - PADDING
    b_end_h = total_h - MARKER_SIZE - PADDING

    # Top + Bottom
    for cx in range(b_start, b_end_w):
        for dy in range(BORDER_WIDTH):
            val = 0 if ((cx + dy) % 2 == 0) else 255
            # Top
            sy, sx = _cell_slice(cx, b_start + dy)
            canvas[sy, sx] = val
            # Bottom
            sy, sx = _cell_slice(cx, b_end_h - BORDER_WIDTH + dy)
            canvas[sy, sx] = val

    # Left + Right (пропускаем углы)
    for cy in range(b_start + BORDER_WIDTH, b_end_h - BORDER_WIDTH):
        for dx in range(BORDER_WIDTH):
            val = 0 if ((cy + dx) % 2 == 0) else 255
            # Left
            sy, sx = _cell_slice(b_start + dx, cy)
            canvas[sy, sx] = val
            # Right
            sy, sx = _cell_slice(b_end_w - BORDER_WIDTH + dx, cy)
            canvas[sy, sx] = val

    # ── 4. Данные: эллипсы в каждой ячейке ─────────────────────────────
    data_idx = 0
    for row in range(grid_h):
        for col in range(grid_w):
            cy = off + row
            cx = off + col
            value = _pack_cell_value(data_bits, data_idx)
            _draw_cell(cx, cy, value)
            data_idx += BITS_PER_CELL

    return canvas


# ── CLI ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Microform Code Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python generator.py -o hello.png --data "Hello, world!"
  python generator.py --grid 150 --dpi 600 -o dense.png
  python generator.py --grid 100 --no-border -o raw_ellipses.png
        """
    )
    parser.add_argument("--grid", type=int, default=150,
                        help="ячеек по каждой стороне (default: 150)")
    parser.add_argument("--grid-h", type=int, default=None,
                        help="ячеек по вертикали (переопределяет --grid)")
    parser.add_argument("--grid-w", type=int, default=None,
                        help="ячеек по горизонтали (переопределяет --grid)")
    parser.add_argument("--cell-size", type=float, default=200.0,
                        help="размер ячейки в микронах (default: 200)")
    parser.add_argument("--dpi", type=int, default=1200,
                        help="DPI вывода (default: 1200)")
    parser.add_argument("--data", type=str, default=None,
                        help="строка для кодирования (иначе случайные данные)")
    parser.add_argument("--seed", type=int, default=0,
                        help="сид для случайных данных (default: 0)")
    parser.add_argument("-o", "--output", type=str, default="microform_code.png",
                        help="выходной PNG (default: microform_code.png)")
    parser.add_argument("--no-border", action="store_true",
                        help="без калибровочной рамки (только эллипсы)")
    parser.add_argument("--info", action="store_true",
                        help="вывести информацию о паттерне")

    args = parser.parse_args()

    grid_h = args.grid_h or args.grid
    grid_w = args.grid_w or args.grid

    # микрометры → пиксели
    microns_per_inch = 25400
    cell_size_px = max(4, round(args.cell_size * args.dpi / microns_per_inch))

    # Подготовка данных
    data_bytes = None
    if args.data:
        data_bytes = args.data.encode("utf-8")
        bits_needed = grid_h * grid_w * BITS_PER_CELL
        bytes_needed = (bits_needed + 7) // 8
        if len(data_bytes) > bytes_needed:
            print(f"⚠  Данные ({len(data_bytes)} B) больше чем влезает "
                  f"({bytes_needed} B). Будут обрезаны.")
            data_bytes = data_bytes[:bytes_needed]

    canvas = generate(grid_h, grid_w, cell_size_px, data=data_bytes)

    # Размер отпечатка
    total_off = MARKER_SIZE + BORDER_WIDTH + PADDING
    total_cells_h = grid_h + 2 * total_off
    total_cells_w = grid_w + 2 * total_off
    mm_h = total_cells_h * args.cell_size / 1000
    mm_w = total_cells_w * args.cell_size / 1000

    raw_bits = grid_h * grid_w * BITS_PER_CELL
    raw_kb = raw_bits / 8 / 1024

    if args.info or not args.no_border:
        print(f"[Microform Code]")
        print(f"  Сетка:            {grid_h}x{grid_w} = {grid_h * grid_w:,} ячеек")
        print(f"  Размер печати:    {mm_w:.1f}x{mm_h:.1f} мм")
        print(f"  Ячейка:           {args.cell_size:.0f} мкм ({cell_size_px} px при {args.dpi} DPI)")
        print(f"  Емкость raw:      {raw_bits:,} бит = {raw_kb:.1f} KB")
        print(f"  После ECC (~30%): {raw_kb * 0.7:.1f} KB (оценка)")
        print(f"  x к QR-40 (3 KB): {raw_kb * 0.7 / 3:.1f}x")
        if args.data:
            print(f"  Данные:           \"{args.data[:50]}\"")
            print(f"  Размер данных:    {len(data_bytes)} B")

    img = Image.fromarray(canvas, mode="L")
    img.save(args.output, compress_level=0)
    print(f"OK: {args.output} ({canvas.shape[1]}x{canvas.shape[0]} px)")


if __name__ == "__main__":
    main()

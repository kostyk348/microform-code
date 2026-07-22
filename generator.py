#!/usr/bin/env python3
"""
Microform Code Generator — Iteration 2 (dual-element + ECC)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Каждая ячейка: главный эллипс + микро-точка = 18 бит.
Данные защищены Reed-Solomon ECC.

Использование:
  python generator.py --data "Hello, world!" -o test.png
  python generator.py --grid 150 --dpi 1200 -o dense.png
  python generator.py --demo -o demo.png      # демо-картинка
"""

import argparse
import numpy as np
from PIL import Image
import codec

# ── Константы ─────────────────────────────────────────────────────────────
MARKER_SIZE = 2        # угловые маркеры (ячеек)
BORDER_WIDTH = 1       # калибровочная рамка (ячеек)
PADDING = 1            # отступ по краям
# ──────────────────────────────────────────────────────────────────────────────


# ── 2D-генератор эллипса ─────────────────────────────────────────────────
def _make_ellipse_mask(size_px: int, angle_deg: float,
                       scale: float, elong: float) -> np.ndarray:
    """Маска эллипса size_px × size_px, uint8: 0=фон, 255=эллипс."""
    half = size_px / 2.0
    yy, xx = np.ogrid[-half:half, -half:half]

    cos_a = np.cos(np.radians(angle_deg))
    sin_a = np.sin(np.radians(angle_deg))
    x_rot = xx * cos_a + yy * sin_a
    y_rot = -xx * sin_a + yy * cos_a

    a = half * scale
    b = a / elong

    dist = (x_rot / a) ** 2 + (y_rot / b) ** 2
    alpha = np.clip(1.0 - (dist - 0.90) / 0.2, 0.0, 1.0)
    return (alpha * 255).astype(np.uint8)


def _make_dot_mask(size_px: int, dx: float, dy: float,
                   radius: float) -> np.ndarray:
    """Маска микро-точки size_px × size_px в позиции (dx, dy) от центра."""
    half = size_px / 2.0
    yy, xx = np.ogrid[-half:half, -half:half]

    x_centered = xx - dx
    y_centered = yy - dy
    dist = np.sqrt(x_centered ** 2 + y_centered ** 2)
    alpha = np.clip(1.0 - (dist - radius + 0.5) / 1.0, 0.0, 1.0)
    return (alpha * 255).astype(np.uint8)


# ── Генератор ────────────────────────────────────────────────────────────
def generate(grid_h: int, grid_w: int, cell_size_px: int,
             data: bytes | None = None) -> np.ndarray:
    """
    Сгенерировать Microform Code (dual-element).

    Returns:
        np.ndarray (H, W), uint8: 0=черный, 255=белый
    """
    off = MARKER_SIZE + BORDER_WIDTH + PADDING
    total_h = grid_h + 2 * off
    total_w = grid_w + 2 * off

    # ── Подготовка данных ───────────────────────────────────────────────
    n_cells = grid_h * grid_w
    total_bits = n_cells * codec.BITS_PER_CELL

    if data is None:
        # Псевдослучайная нагрузка
        rng = np.random.default_rng(42)
        data = rng.bytes(total_bits // 8 + 1)

    encoded_bits = codec.encode_bits(data, total_bits)

    # ── Полотно ─────────────────────────────────────────────────────────
    canvas = np.ones((total_h * cell_size_px, total_w * cell_size_px),
                     dtype=np.uint8) * 255

    def _cell_rect(cx: int, cy: int):
        """(cx, cy) → slice_y, slice_x."""
        y1 = cy * cell_size_px
        x1 = cx * cell_size_px
        return (slice(y1, y1 + cell_size_px),
                slice(x1, x1 + cell_size_px))

    def _draw_cell(cx: int, cy: int, value: int):
        """Нарисовать dual-element в ячейке (cx, cy)."""
        params = codec.cell_value_to_params(value)
        cs = cell_size_px

        # Главный эллипс
        angle = codec.angle_v_to_deg(params['angle_v'])
        scale = codec.scale_v_to_frac(params['scale_v'])
        elong = codec.elong_v_to_ratio(params['elong_v'])
        mask = _make_ellipse_mask(cs, angle, scale, elong)

        # Микро-точка
        dx = codec.dot_dx_to_px(params['dot_dx'], cs)
        dy = codec.dot_dy_to_px(params['dot_dy'], cs)
        rad = codec.dot_rad_to_px(params['dot_rad'])
        dot_mask = _make_dot_mask(cs, dx, dy, rad)

        # Комбинируем: эллипс + точка (max = either dark wins)
        combined = np.maximum(mask, dot_mask)

        sy, sx = _cell_rect(cx, cy)
        canvas[sy, sx] = np.minimum(canvas[sy, sx],
                                    255 - combined).astype(np.uint8)

    # ── 1. Угловые маркеры (черные квадраты 3x3) ────────────────────────
    corners = [
        (PADDING, PADDING),
        (total_w - MARKER_SIZE - PADDING, PADDING),
        (PADDING, total_h - MARKER_SIZE - PADDING),
        (total_w - MARKER_SIZE - PADDING, total_h - MARKER_SIZE - PADDING),
    ]
    for cxo, cyo in corners:
        for dy in range(MARKER_SIZE):
            for dx in range(MARKER_SIZE):
                sy, sx = _cell_rect(cxo + dx, cyo + dy)
                canvas[sy, sx] = 0

    # ── 2. Калибровочная рамка (шахматка) ──────────────────────────────
    b_start = MARKER_SIZE + PADDING
    b_end_w = total_w - MARKER_SIZE - PADDING
    b_end_h = total_h - MARKER_SIZE - PADDING

    for cx in range(b_start, b_end_w):
        for dy in range(BORDER_WIDTH):
            val = 0 if ((cx + dy) % 2 == 0) else 255
            sy, sx = _cell_rect(cx, b_start + dy)
            canvas[sy, sx] = val
            sy, sx = _cell_rect(cx, b_end_h - BORDER_WIDTH + dy)
            canvas[sy, sx] = val

    for cy in range(b_start + BORDER_WIDTH, b_end_h - BORDER_WIDTH):
        for dx in range(BORDER_WIDTH):
            val = 0 if ((cy + dx) % 2 == 0) else 255
            sy, sx = _cell_rect(b_start + dx, cy)
            canvas[sy, sx] = val
            sy, sx = _cell_rect(b_end_w - BORDER_WIDTH + dx, cy)
            canvas[sy, sx] = val

    # ── 3. Данные: dual-element ─────────────────────────────────────────
    data_idx = 0
    for row in range(grid_h):
        for col in range(grid_w):
            cy = off + row
            cx = off + col

            # Собираем 18 бит → cell value
            val = 0
            for b in range(codec.BITS_PER_CELL):
                val = (val << 1) | encoded_bits[data_idx]
                data_idx += 1

            _draw_cell(cx, cy, val)

    return canvas


# ── CLI ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Microform Code Generator v2 (dual-element + ECC)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python generator.py -o test.png --data "Hello world"
  python generator.py -o dense.png --grid 150 --dpi 1200
  python generator.py -o demo.png --demo
        """)

    parser.add_argument("--grid", type=int, default=150,
                        help="ячеек по стороне (default: 150)")
    parser.add_argument("--grid-h", type=int, default=None)
    parser.add_argument("--grid-w", type=int, default=None)
    parser.add_argument("--cell-size", type=float, default=200.0,
                        help="ячейка в микронах (default: 200)")
    parser.add_argument("--dpi", type=int, default=1200,
                        help="DPI вывода (default: 1200)")
    parser.add_argument("--data", type=str, default=None,
                        help="строка для кодирования")
    parser.add_argument("-o", "--output", type=str,
                        default="microform_code.png")
    parser.add_argument("--demo", action="store_true",
                        help="демо-картинка с тестовыми данными")

    args = parser.parse_args()

    if args.demo:
        args.data = ("КРУТОЙ МИКРОФОРМ КОД v2\n"
                     "18 бит/ячейку + RS-ECC\n"
                     "Печатай и сканируй!")

    grid_h = args.grid_h or args.grid
    grid_w = args.grid_w or args.grid

    microns_per_inch = 25400
    cell_size_px = max(4, round(args.cell_size * args.dpi / microns_per_inch))

    # Подготовка данных
    data_bytes = None
    if args.data:
        data_bytes = args.data.encode("utf-8")

    canvas = generate(grid_h, grid_w, cell_size_px, data=data_bytes)

    # ── Инфо ────────────────────────────────────────────────────────────
    total_off = MARKER_SIZE + BORDER_WIDTH + PADDING
    total_cells_h = grid_h + 2 * total_off
    total_cells_w = grid_w + 2 * total_off
    mm_h = total_cells_h * args.cell_size / 1000
    mm_w = total_cells_w * args.cell_size / 1000

    info = codec.capacity_info(grid_h * grid_w)
    qr_x = info['data_kb'] / 3.0 if info['data_kb'] else 0

    print(f"[Microform Code v2]")
    print(f"  Сетка:            {grid_h}x{grid_w} = {grid_h*grid_w:,} ячеек")
    print(f"  Размер печати:    {mm_w:.1f}x{mm_h:.1f} мм")
    print(f"  Ячейка:           {args.cell_size:.0f} мкм ({cell_size_px} px @ {args.dpi} DPI)")
    print(f"  Бит/ячейку:       {codec.BITS_PER_CELL} (эллипс 10 + точка 8)")
    print(f"  ECC:              RS(255,{255-codec.RS_NSYM}) [{codec.RS_NSYM} parity/block]")
    print(f"  --")
    print(f"  Raw:              {info['raw_kb']:.1f} KB")
    print(f"  После RS:         {info['data_kb']:.1f} KB")
    print(f"  Оverhead RS:      {info['ecc_overhead_pct']:.1f}%")
    print(f"  x QR-40:          {qr_x:.1f}x")

    if args.data:
        print(f"  Данные:           \"{args.data[:60]}\"")

    img = Image.fromarray(canvas, mode="L")
    img.save(args.output, compress_level=0)
    print(f"  OK: {args.output} ({canvas.shape[1]}x{canvas.shape[0]} px)")


if __name__ == "__main__":
    main()

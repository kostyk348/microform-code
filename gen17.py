#!/usr/bin/env python3
"""Generate Microform Code image using 17-bit/cell encoding."""
import numpy as np
from PIL import Image
import codec_17bit as cc
from generator import _make_ellipse_mask, _make_dot_mask, MARKER_SIZE, BORDER_WIDTH, PADDING
from codec import angle_v_to_deg, scale_v_to_frac, elong_v_to_ratio
from codec import dot_dx_to_px, dot_dy_to_px, dot_rad_to_px


def generate(grid_h, grid_w, cell_size_px, data=None):
    off = MARKER_SIZE + BORDER_WIDTH + PADDING
    total_h = grid_h + 2 * off
    total_w = grid_w + 2 * off
    n_cells = grid_h * grid_w

    if data is None:
        data = b"Microform Code 17-bit/cell demo at 300um."
    codes = cc.encode(data, n_cells)

    canvas = np.ones((total_h * cell_size_px, total_w * cell_size_px), dtype=np.uint8) * 255

    def _cell_rect(cx, cy):
        y1 = cy * cell_size_px
        x1 = cx * cell_size_px
        return (slice(y1, y1 + cell_size_px), slice(x1, x1 + cell_size_px))

    def _draw_cell(cx, cy, code):
        p = cc.render_params(code)
        cs = cell_size_px
        em = _make_ellipse_mask(cs, angle_v_to_deg(p["av"]), scale_v_to_frac(p["sv"]), elong_v_to_ratio(p["ev"]))
        dm = _make_dot_mask(cs, dot_dx_to_px(p["dx"], cs), dot_dy_to_px(p["dy"], cs), dot_rad_to_px(p["r"]))
        combined = np.maximum(em, dm)
        sy, sx = _cell_rect(cx, cy)
        canvas[sy, sx] = np.minimum(canvas[sy, sx], 255 - combined).astype(np.uint8)

    # 1. Corner markers (black 2x2)
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

    # 2. Border (checkerboard)
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

    # 3. Data cells
    idx = 0
    for row in range(grid_h):
        for col in range(grid_w):
            _draw_cell(off + col, off + row, codes[idx])
            idx += 1

    return canvas


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=92)
    parser.add_argument("--cell-size", type=float, default=300.0)
    parser.add_argument("--dpi", type=int, default=1200)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("-o", default="mcf_17bit.png")
    args = parser.parse_args()

    cell_px = max(4, round(args.cell_size * args.dpi / 25400))
    data = args.data.encode() if args.data else None
    if not data:
        data = b"Microform Code 17-bit/cell demo - camera-decodable at 300um."
    
    canvas = generate(args.grid, args.grid, cell_px, data=data)
    mm = (args.grid + 8) * args.cell_size / 1000
    info = cc.capacity_info(args.grid * args.grid)
    
    print(f"[17-bit MCF] {args.grid}x{args.grid} = {args.grid*args.grid:,} cells")
    print(f"  Size: {mm:.1f}x{mm:.1f} mm, {cell_px} px/cell @ {args.dpi} DPI")
    print(f"  Capacity: {info['raw_kb']:.1f} KB raw, {info['data_kb']:.1f} KB payload")
    print(f"  Data: {len(data)} bytes")

    Image.fromarray(canvas, mode="L").save(args.o, compress_level=0)
    print(f"  Saved: {args.o} ({canvas.shape[1]}x{canvas.shape[0]} px)")

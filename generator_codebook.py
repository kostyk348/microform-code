#!/usr/bin/env python3
"""
Microform Code Generator — Codebook-based (no ambiguity)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Uses the codebook to generate ONLY distinct visual patterns,
eliminating decoding ambiguity.
"""
import argparse
import numpy as np
from PIL import Image
import codebook_codec as cc
from generator import _make_ellipse_mask, _make_dot_mask
from codec import angle_v_to_deg, scale_v_to_frac, elong_v_to_ratio
from codec import dot_dx_to_px, dot_dy_to_px, dot_rad_to_px

# ── Layout constants (same as original) ──────────────────────────────────
MARKER_SIZE = 2
BORDER_WIDTH = 1
PADDING = 1


def generate(grid_h: int, grid_w: int, cell_size_px: int,
             data: bytes | None = None) -> np.ndarray:
    """
    Generate Microform Code using codebook-based encoding.

    Returns:
        np.ndarray (H, W), uint8: 0=black, 255=white
    """
    off = MARKER_SIZE + BORDER_WIDTH + PADDING
    total_h = grid_h + 2 * off
    total_w = grid_w + 2 * off

    # ── Data → cell codes ────────────────────────────────────────────────
    n_cells = grid_h * grid_w

    if data is None:
        rng = np.random.default_rng(42)
        data = rng.bytes(n_cells)

    codes = cc.encode_to_codes(data, n_cells)

    # ── Canvas ────────────────────────────────────────────────────────────
    canvas = np.ones((total_h * cell_size_px, total_w * cell_size_px),
                     dtype=np.uint8) * 255

    def _cell_rect(cx, cy):
        y1 = cy * cell_size_px
        x1 = cx * cell_size_px
        return (slice(y1, y1 + cell_size_px), slice(x1, x1 + cell_size_px))

    def _draw_cell(cx, cy, code):
        params = cc.code_to_render_params(code)
        cs = cell_size_px

        # Render ellipse
        angle = angle_v_to_deg(params['av'])
        scale = scale_v_to_frac(params['sv'])
        elong = elong_v_to_ratio(params['ev'])
        mask = _make_ellipse_mask(cs, angle, scale, elong)

        # Render dot
        dx = dot_dx_to_px(params['dx'], cs)
        dy = dot_dy_to_px(params['dy'], cs)
        rad = dot_rad_to_px(params['r'])
        dot_mask = _make_dot_mask(cs, dx, dy, rad)

        combined = np.maximum(mask, dot_mask)
        sy, sx = _cell_rect(cx, cy)
        canvas[sy, sx] = np.minimum(canvas[sy, sx], 255 - combined).astype(np.uint8)

    # ── 1. Corner markers ─────────────────────────────────────────────────
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

    # ── 2. Calibration border (checkerboard) ────────────────────────────
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

    # ── 3. Data cells ─────────────────────────────────────────────────────
    idx = 0
    for row in range(grid_h):
        for col in range(grid_w):
            cy = off + row
            cx = off + col
            _draw_cell(cx, cy, codes[idx])
            idx += 1

    return canvas


# ── CLI ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Microform Code Generator (codebook)")
    parser.add_argument("--grid", type=int, default=92,
                        help="cells per side (default: 92)")
    parser.add_argument("--cell-size", type=float, default=300.0,
                        help="cell size in microns (default: 300)")
    parser.add_argument("--dpi", type=int, default=1200,
                        help="output DPI (default: 1200)")
    parser.add_argument("--data", type=str, default=None,
                        help="string to encode")
    parser.add_argument("-o", "--output", type=str, default="microform_codebook.png")

    args = parser.parse_args()
    microns_per_inch = 25400
    cell_size_px = max(4, round(args.cell_size * args.dpi / microns_per_inch))

    data_bytes = None
    if args.data:
        data_bytes = args.data.encode("utf-8")
    elif args.data is None:
        data_bytes = "Microform Code v2 - codebook-based. 17.5 bits/cell, RS ECC, camera-decodable.".encode()

    canvas = generate(args.grid, args.grid, cell_size_px, data=data_bytes)

    n_cells = args.grid * args.grid
    info = cc.capacity_info(n_cells)
    total_off = MARKER_SIZE + BORDER_WIDTH + PADDING
    mm_size = (args.grid + 2 * total_off) * args.cell_size / 1000

    print(f"[Microform Code — Codebook]")
    print(f"  Grid:            {args.grid}x{args.grid} = {n_cells:,} cells")
    print(f"  Print size:      {mm_size:.1f}x{mm_size:.1f} mm")
    print(f"  Cell:            {args.cell_size:.0f} um ({cell_size_px} px @ {args.dpi} DPI)")
    print(f"  Bits/cell:       {info['bits_per_cell']:.2f}")
    print(f"  Raw capacity:    {info['raw_kb']:.1f} KB")
    print(f"  Data capacity:   {info['data_kb']:.1f} KB")
    print(f"  RS:              RS(255,223) [{info['rs_blocks']} blocks]")

    if data_bytes:
        print(f"  Data:            \"{data_bytes[:60].decode('utf-8', errors='replace')}\"")
        print(f"  Data size:       {len(data_bytes)} bytes")

    img = Image.fromarray(canvas, mode="L")
    img.save(args.output, compress_level=0)
    print(f"  OK: {args.output} ({canvas.shape[1]}x{canvas.shape[0]} px)")


if __name__ == "__main__":
    main()

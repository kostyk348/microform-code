#!/usr/bin/env python3
"""
Microform Code Decoder — Codebook-based
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Reads cell pixels, looks up codes in codebook, reconstructs data.
"""
import cv2, numpy as np, sys
from decoder import _find_markers, OFF
import codebook_codec as cc

CELL_PX = 14
GRID = 92


def decode_image(img_path: str) -> bytes | None:
    """Decode a Microform Code image and return the original data."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"ERROR: Cannot read {img_path}", file=sys.stderr)
        return None

    h, w = img.shape
    print(f"  Image: {w}x{h} px", flush=True)

    # ── 1. Find markers ──────────────────────────────────────────────────
    markers = _find_markers(img)
    if len(markers) != 4:
        print(f"ERROR: Found {len(markers)} markers, expected 4", file=sys.stderr)
        return None
    print(f"  Markers: {markers}", flush=True)

    # ── 2. Homography ────────────────────────────────────────────────────
    src_pts = np.array(markers, dtype=np.float32)
    dst_pts = np.array([[2, 2],
                        [GRID + 6, 2],
                        [2, GRID + 6],
                        [GRID + 6, GRID + 6]], dtype=np.float32)
    H, _ = cv2.findHomography(src_pts, dst_pts)
    if H is None:
        print("ERROR: Homography failed", file=sys.stderr)
        return None
    H_inv = np.linalg.inv(H)

    # ── 3. Read cells ────────────────────────────────────────────────────
    codes = []
    no_match = 0
    miss_positions = []

    for row in range(GRID):
        for col in range(GRID):
            cell_center = (OFF + col + 0.5, OFF + row + 0.5)
            pt = cv2.perspectiveTransform(
                np.array([[[cell_center[0], cell_center[1]]]], dtype=np.float32),
                H_inv
            )[0, 0]

            x1 = int(round(pt[0] - CELL_PX / 2))
            y1 = int(round(pt[1] - CELL_PX / 2))

            if x1 < 0 or y1 < 0 or x1 + CELL_PX > w or y1 + CELL_PX > h:
                codes.append(0)
                no_match += 1
                continue

            cell = img[y1:y1 + CELL_PX, x1:x1 + CELL_PX]

            code = cc.pixels_to_code(cell)
            if code < 0:
                codes.append(0)
                no_match += 1
                miss_positions.append((row, col))
            else:
                codes.append(code)

    print(f"  Cells read: {len(codes)}", flush=True)
    if no_match > 0:
        print(f"  No match: {no_match} cells", flush=True)
        if miss_positions:
            print(f"  First 5 misses: {miss_positions[:5]}", flush=True)

    # ── 4. Decode ────────────────────────────────────────────────────────
    try:
        data = cc.decode_from_codes(codes)
        print(f"  Decoded: {len(data)} bytes", flush=True)
        return data
    except Exception as e:
        print(f"  DECODE ERROR: {e}", file=sys.stderr)
        return None


# ── CLI ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "mcf_codebook.png"
    print(f"Decoding: {path}", flush=True)
    result = decode_image(path)
    if result is not None:
        try:
            text = result.decode("utf-8", errors="replace")
            print(f"\nRESULT: {text}")
        except:
            print(f"\nRESULT: {result.hex()}")
        print(f"\n{'='*40}")
        print(f"SUCCESS: {len(result)} bytes decoded")
    else:
        print(f"\nFAILED to decode", file=sys.stderr)
        sys.exit(1)

#!/usr/bin/env python3
"""Decode Microform Code 17-bit/cell image."""
import cv2, numpy as np, sys
from decoder import _find_markers, OFF
import codec_17bit as cc

CELL_PX = 14
GRID = 92


def decode(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"ERROR: Cannot read {path}", file=sys.stderr)
        return None

    img_h, img_w = img.shape
    print(f"  Image: {img_w}x{img_h} px", flush=True)

    # Find markers
    markers = _find_markers(img)
    if len(markers) != 4:
        print(f"ERROR: Found {len(markers)} markers", file=sys.stderr)
        return None
    print(f"  Markers: {markers}", flush=True)

    # Homography
    src_pts = np.array(markers, dtype=np.float32)
    dst_pts = np.array([[2, 2], [GRID + 6, 2], [2, GRID + 6], [GRID + 6, GRID + 6]], dtype=np.float32)
    H, _ = cv2.findHomography(src_pts, dst_pts)
    if H is None:
        print("ERROR: Homography failed", file=sys.stderr)
        return None
    H_inv = np.linalg.inv(H)

    # Read cells
    codes = []
    no_match = 0
    for row in range(GRID):
        for col in range(GRID):
            cell_center = (OFF + col + 0.5, OFF + row + 0.5)
            pt = cv2.perspectiveTransform(
                np.array([[[cell_center[0], cell_center[1]]]], dtype=np.float32), H_inv
            )[0, 0]
            x1 = int(round(pt[0] - CELL_PX / 2))
            y1 = int(round(pt[1] - CELL_PX / 2))
            if x1 < 0 or y1 < 0 or x1 + CELL_PX > img_w or y1 + CELL_PX > img_h:
                codes.append(0)
                no_match += 1
                continue
            cell = img[y1:y1 + CELL_PX, x1:x1 + CELL_PX]
            code = cc.pixels_to_17bit(cell)
            if code < 0:
                codes.append(0)
                no_match += 1
            else:
                codes.append(code)

    print(f"  Cells: {len(codes)} read, {no_match} no match", flush=True)

    # Decode
    try:
        data = cc.decode(codes)
        return data
    except Exception as e:
        print(f"  DECODE ERROR: {e}", file=sys.stderr)
        return None


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "mcf_17bit.png"
    print(f"Decoding: {path}", flush=True)
    result = decode(path)
    if result is not None:
        text = result.decode("utf-8", errors="replace")
        print(f"\nRESULT: {text}")
        print(f"\nSUCCESS: {len(result)} bytes")
    else:
        print("FAILED", file=sys.stderr)
        sys.exit(1)

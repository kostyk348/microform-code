"""Test RS error correction on 17-bit/cell codec."""
import cv2, numpy as np, random
from decoder import _find_markers, OFF
import codec_17bit as cc

CELL_PX = 14
GRID = 92

img = cv2.imread("mcf_17bit.png", cv2.IMREAD_GRAYSCALE)
img_h, img_w = img.shape
markers = _find_markers(img)
src_pts = np.array(markers, dtype=np.float32)
dst_pts = np.array([[2,2],[GRID+6,2],[2,GRID+6],[GRID+6,GRID+6]], dtype=np.float32)
H_inv = np.linalg.inv(cv2.findHomography(src_pts, dst_pts)[0])

codes = []
for row in range(GRID):
    for col in range(GRID):
        pt = cv2.perspectiveTransform(
            np.array([[[OFF+col+0.5, OFF+row+0.5]]], dtype=np.float32), H_inv
        )[0,0]
        x1, y1 = int(round(pt[0]-CELL_PX/2)), int(round(pt[1]-CELL_PX/2))
        cell = img[y1:y1+CELL_PX, x1:x1+CELL_PX]
        code = cc.pixels_to_17bit(cell)
        codes.append(code if code >= 0 else 0)

ground_truth = cc.decode(codes)
print(f"Ground truth ({len(ground_truth)}b): {ground_truth[:50]}")
print()

for n_err in [0, 10, 50, 100, 200, 500, 1000, 2000]:
    corrupted = codes.copy()
    for _ in range(n_err):
        i = random.randint(0, len(corrupted)-1)
        corrupted[i] = random.randint(0, 131071)
    try:
        data = cc.decode(corrupted)
        ok = data == ground_truth
        pct = 100 * n_err / len(codes)
        print(f"  {n_err:5d} errors ({pct:.1f}%) -> {'OK' if ok else 'FAIL'}", flush=True)
    except Exception as e:
        print(f"  {n_err:5d} errors -> CRASH: {e}", flush=True)

"""Test RS error correction by injecting random errors into codes."""
import cv2, numpy as np, pickle, random, sys
from decoder import _find_markers, OFF
import codebook_codec as cc

# Use the proper decoder to get ground truth codes
from decoder_codebook import decode_image

# Get ground truth
truth = decode_image("mcf_codebook.png")
print(f"Ground truth ({len(truth)} bytes): {truth[:50]}")
print()

# Re-read codes from image for error injection
img = cv2.imread("mcf_codebook.png", cv2.IMREAD_GRAYSCALE)
img_h, img_w = img.shape
markers = _find_markers(img)
src_pts = np.array(markers, dtype=np.float32)
GRID = 92
dst_pts = np.array([[2,2],[GRID+6,2],[2,GRID+6],[GRID+6,GRID+6]], dtype=np.float32)
H_inv = np.linalg.inv(cv2.findHomography(src_pts, dst_pts)[0])
CELL_PX = 14

codes = []
for row in range(GRID):
    for col in range(GRID):
        pt = cv2.perspectiveTransform(
            np.array([[[OFF+col+0.5, OFF+row+0.5]]], dtype=np.float32), H_inv
        )[0,0]
        x1, y1 = int(round(pt[0]-CELL_PX/2)), int(round(pt[1]-CELL_PX/2))
        cell = img[y1:y1+CELL_PX, x1:x1+CELL_PX]
        code = cc.pixels_to_code(cell)
        codes.append(code)

print("Testing RS(255,223) with injected errors...")
print(f"Total cells: {len(codes)}, codespace: {cc.codebook_size()}")
print()

for n_errors in [0, 10, 50, 100, 200, 500, 1000, 2000]:
    corrupted = codes.copy()
    for _ in range(n_errors):
        i = random.randint(0, len(corrupted)-1)
        corrupted[i] = random.randint(0, cc.codebook_size()-1)
    
    try:
        data = cc.decode_from_codes(corrupted)
        ok = data == truth
        status = "OK" if ok else "CORRUPTED"
        preview = data[:40].decode("utf-8", errors="replace") if data else "(empty)"
    except Exception as e:
        status = f"CRASH: {e}"
        preview = ""
    
    print(f"  Errors: {n_errors:5d} / {len(codes)} ({100*n_errors/len(codes):.1f}%) -> {status}")
    if status != "OK" and preview:
        try: print(f"    {preview}")
        except: print(f"    (binary data, {len(data)} bytes)")

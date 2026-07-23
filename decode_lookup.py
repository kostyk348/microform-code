"""Decode using exact pixel lookup. Measure BER on distinct templates only."""
import cv2, numpy as np, time
from decoder import _find_markers, OFF
from codec import encode_bits, angle_v_to_deg, scale_v_to_frac, elong_v_to_ratio
from codec import dot_dx_to_px, dot_dy_to_px, dot_rad_to_px
from generator import _make_ellipse_mask, _make_dot_mask

CELL_PX = 14
GRID = 92

# Build distinct mapping: pixel_hash -> list of values
print("Building distinct template map...", flush=True)
distinct_map = {}  # bytes_hash -> list of values
all_values = []

for av in range(64):
    for sv in range(4):
        for ev in range(4):
            em = _make_ellipse_mask(CELL_PX, angle_v_to_deg(av),
                                    scale_v_to_frac(sv), elong_v_to_ratio(ev))
            for dx in range(8):
                for dy in range(8):
                    for r in range(4):
                        dm = _make_dot_mask(CELL_PX, dot_dx_to_px(dx, CELL_PX),
                                            dot_dy_to_px(dy, CELL_PX), dot_rad_to_px(r))
                        t = 255 - np.maximum(em, dm)
                        h = t.tobytes()
                        val = (av << 12) | (sv << 10) | (ev << 8) | (dx << 5) | (dy << 2) | r
                        if h not in distinct_map:
                            distinct_map[h] = []
                        distinct_map[h].append(val)
                        all_values.append(val)

print(f"Distinct patterns: {len(distinct_map)} of {len(all_values)}")
print(f"Effective bits: {np.log2(len(distinct_map)):.1f}")

# Load image
img = cv2.imread("mcf_300um.png", cv2.IMREAD_GRAYSCALE)
img_h, img_w = img.shape
markers = _find_markers(img)
src_pts = np.array(markers, dtype=np.float32)
dst_pts = np.array([[2,2],[GRID+6,2],[2,GRID+6],[GRID+6,GRID+6]], dtype=np.float32)
H_inv = np.linalg.inv(cv2.findHomography(src_pts, dst_pts)[0])

# Encode known data
data = b"MCF v2 -- 18bit/cell + RS -- TEST 300um"
orig = encode_bits(data, GRID*GRID*18)

N = 500
correct = 0
total_bits = 0
wrong_bits = 0
wrong_any = 0
unique_ok = 0
unique_total = 0
ambig_ok = 0
ambig_total = 0
no_match = 0

for idx in range(N):
    row = idx // 23
    col = idx % 23

    orig_val = 0
    for b in range(18):
        orig_val = (orig_val << 1) | orig[(row*GRID+col)*18 + b]

    cell_center = (OFF + col + 0.5, OFF + row + 0.5)
    pt = cv2.perspectiveTransform(np.array([[[cell_center[0], cell_center[1]]]],
                                  dtype=np.float32), H_inv)[0,0]
    x1 = int(round(pt[0] - CELL_PX/2))
    y1 = int(round(pt[1] - CELL_PX/2))
    if x1 < 0 or y1 < 0 or x1+CELL_PX > img_w or y1+CELL_PX > img_h:
        print(f"  Cell[{row},{col}]: OOB")
        continue

    cell_pixels = img[y1:y1+CELL_PX, x1:x1+CELL_PX].flatten()
    cell_hash = cell_pixels.tobytes()

    if cell_hash in distinct_map:
        candidates = distinct_map[cell_hash]
        best_val = candidates[0]
        is_unique = len(candidates) == 1
    else:
        best_val = -1
        no_match += 1

    if is_unique:
        unique_total += 1
        if best_val == orig_val:
            unique_ok += 1
    else:
        ambig_total += 1
        if best_val == orig_val:
            ambig_ok += 1

    if best_val == orig_val:
        correct += 1
    else:
        wrong_any += 1
        if best_val >= 0:
            diff = best_val ^ orig_val
            wrong_bits += bin(diff).count("1")
        else:
            wrong_bits += 18

    total_bits += 18

ber = 100 * wrong_bits / total_bits
print(f"\n=== Lookup decode results (N={N}) ===")
print(f"Correct: {correct}/{N} ({100*correct/N:.1f}%)")
print(f"BER: {ber:.2f}%")
print(f"No exact match: {no_match}/{N}")
print(f"Unique templates: {unique_ok}/{unique_total} correct ({100*unique_ok/max(unique_total,1):.1f}%)")
print(f"Ambiguous templates: {ambig_ok}/{ambig_total} correct ({100*ambig_ok/max(ambig_total,1):.1f}%)")

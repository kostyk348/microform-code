"""Analyze decoding error patterns for 300um code."""
import cv2, numpy as np, time
from decoder import _find_markers, OFF
from codec import encode_bits
from codec import angle_v_to_deg, scale_v_to_frac, elong_v_to_ratio
from codec import dot_dx_to_px, dot_dy_to_px, dot_rad_to_px
from generator import _make_ellipse_mask, _make_dot_mask

GRID = 92
data = b'MCF v2' * 20
orig = encode_bits(data, GRID*GRID*18)

img = cv2.imread('mcf_300um.png', cv2.IMREAD_GRAYSCALE)
h, w = img.shape
markers = _find_markers(img)
src_pts = np.array(markers, dtype=np.float32)
dst_pts = np.array([[2,2],[GRID+6,2],[2,GRID+6],[GRID+6,GRID+6]], dtype=np.float32)
H_inv = np.linalg.inv(cv2.findHomography(src_pts, dst_pts)[0])
CELL_PX = 14

# Render templates
print('Rendering TEMPLATES...', flush=True)
t0 = time.time()
templates = []
values = []

for av in range(64):
    for sv in range(4):
        for ev in range(4):
            ellipse_mask = _make_ellipse_mask(CELL_PX, angle_v_to_deg(av),
                                              scale_v_to_frac(sv), elong_v_to_ratio(ev))
            for dx in range(8):
                for dy in range(8):
                    for r in range(4):
                        dot_mask = _make_dot_mask(CELL_PX, dot_dx_to_px(dx, CELL_PX),
                                                  dot_dy_to_px(dy, CELL_PX), dot_rad_to_px(r))
                        combined = np.maximum(ellipse_mask, dot_mask)
                        img_t = 255 - combined
                        templates.append(img_t.flatten())
                        val = (av << 12) | (sv << 10) | (ev << 8) | (dx << 5) | (dy << 2) | r
                        values.append(val)

tarr = np.array(templates, dtype=np.float32)
t_mean = tarr.mean(axis=1, keepdims=True)
t_std = tarr.std(axis=1, keepdims=True)
t_norm = (tarr - t_mean) / np.maximum(t_std, 0.001)
print(f'  {tarr.shape}, {time.time()-t0:.1f}s', flush=True)

# Test on cells
N = min(500, GRID*GRID)
correct_cells = 0
total_bits = 0
wrong_bits = 0
wrong_any = 0

angle_err = 0
scale_err = 0
elong_err = 0
dot_dx_err = 0
dot_dy_err = 0
dot_r_err = 0

angle_diffs = {}
top5_accuracy = 0

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
    
    if x1 < 0 or y1 < 0 or x1+CELL_PX > w or y1+CELL_PX > h:
        continue
    
    cell = img[y1:y1+CELL_PX, x1:x1+CELL_PX].flatten().astype(float)
    cell_n = (cell - cell.mean()) / max(cell.std(), 0.001)
    
    corrs = np.dot(t_norm, cell_n) / 195
    order = np.argsort(corrs)[::-1]
    best_idx = order[0]
    best_val = values[best_idx]
    
    # Top-5: does the correct value appear in top 5?
    in_top5 = orig_val in [values[order[i]] for i in range(min(5, len(order)))]
    if in_top5:
        top5_accuracy += 1
    
    # Decompose
    orig_av = (orig_val >> 12) & 0x3F
    orig_sv = (orig_val >> 10) & 0x3
    orig_ev = (orig_val >> 8) & 0x3
    orig_dx = (orig_val >> 5) & 0x7
    orig_dy = (orig_val >> 2) & 0x7
    orig_r = orig_val & 0x3
    
    best_av = (best_val >> 12) & 0x3F
    best_sv = (best_val >> 10) & 0x3
    best_ev = (best_val >> 8) & 0x3
    best_dx = (best_val >> 5) & 0x7
    best_dy = (best_val >> 2) & 0x7
    best_r = best_val & 0x3
    
    if best_val == orig_val:
        correct_cells += 1
    else:
        wrong_any += 1
        if best_av != orig_av: angle_err += 1
        if best_sv != orig_sv: scale_err += 1
        if best_ev != orig_ev: elong_err += 1
        if best_dx != orig_dx: dot_dx_err += 1
        if best_dy != orig_dy: dot_dy_err += 1
        if best_r != orig_r: dot_r_err += 1
        
        if best_av != orig_av:
            d = abs(best_av - orig_av)
            d = min(d, 64 - d)
            angle_diffs[d] = angle_diffs.get(d, 0) + 1
        
        diff = best_val ^ orig_val
        wrong_bits += bin(diff).count('1')
    total_bits += 18

print(f"\n=== Results (N={N}) ===")
print(f"Correct cells: {correct_cells}/{wrong_any+correct_cells} "
      f"({100*correct_cells/(wrong_any+correct_cells):.1f}%)")
print(f"BER: {100*wrong_bits/total_bits:.2f}%")
print(f"Top-5 accuracy: {top5_accuracy}/{N} ({100*top5_accuracy/N:.1f}%)")

print(f"\n=== Error distribution (among {wrong_any} wrong cells) ===")
print(f"  Angle wrong:  {angle_err} ({100*angle_err/max(wrong_any,1):.0f}%)")
print(f"  Scale wrong:  {scale_err} ({100*scale_err/max(wrong_any,1):.0f}%)")
print(f"  Elong wrong:  {elong_err} ({100*elong_err/max(wrong_any,1):.0f}%)")
print(f"  Dot-dx wrong: {dot_dx_err} ({100*dot_dx_err/max(wrong_any,1):.0f}%)")
print(f"  Dot-dy wrong: {dot_dy_err} ({100*dot_dy_err/max(wrong_any,1):.0f}%)")
print(f"  Dot-r wrong:  {dot_r_err} ({100*dot_r_err/max(wrong_any,1):.0f}%)")

print(f"\n=== Angle error distances ===")
for d in sorted(angle_diffs.keys()):
    print(f"  diff={d}: {angle_diffs[d]}")

# Correlation analysis: correct vs wrong
corr_correct = []
corr_wrong = []
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
    if x1 < 0 or y1 < 0: continue
    cell = img[y1:y1+CELL_PX, x1:x1+CELL_PX].flatten().astype(float)
    cell_n = (cell - cell.mean()) / max(cell.std(), 0.001)
    corrs = np.dot(t_norm, cell_n) / 195
    best_idx = np.argmax(corrs)
    best_val = values[best_idx]
    if best_val == orig_val:
        corr_correct.append(corrs.max())
    else:
        corr_wrong.append(corrs.max())

if corr_correct:
    corr_correct = np.array(corr_correct)
    print(f"\n=== Correlation stats ===")
    print(f"  Correct: mean={corr_correct.mean():.4f}, min={corr_correct.min():.4f}, "
          f"max={corr_correct.max():.4f}")
if corr_wrong:
    corr_wrong = np.array(corr_wrong)
    print(f"  Wrong:   mean={corr_wrong.mean():.4f}, min={corr_wrong.min():.4f}, "
          f"max={corr_wrong.max():.4f}")

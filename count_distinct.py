"""Count how many distinct templates exist at 14px/cell."""
import numpy as np, time, sys
from codec import angle_v_to_deg, scale_v_to_frac, elong_v_to_ratio
from codec import dot_dx_to_px, dot_dy_to_px, dot_rad_to_px
from generator import _make_ellipse_mask, _make_dot_mask

CELL_PX = 14

# Phase 1: Just ellipse
print("Phase 1: distinct ellipse masks...", flush=True)
seen_ellipse = {}
total = 0
for av in range(64):
    for sv in range(4):
        for ev in range(4):
            total += 1
            mask = _make_ellipse_mask(CELL_PX, angle_v_to_deg(av), scale_v_to_frac(sv), elong_v_to_ratio(ev))
            h = hash(mask.tobytes())
            if h not in seen_ellipse:
                seen_ellipse[h] = (av, sv, ev)

print(f"  Total combinations: {total}")
print(f"  Distinct masks: {len(seen_ellipse)}")

# Group by sv, ev
groups = {}
for h, (av, sv, ev) in seen_ellipse.items():
    key = (sv, ev)
    if key not in groups:
        groups[key] = []
    groups[key].append(av)

for key in sorted(groups.keys()):
    angs = sorted(groups[key])
    sv, ev = key
    sample = ", ".join(str(a) for a in angs[:5])
    extra = f" ... ({len(angs)} total)" if len(angs) > 5 else ""
    print(f"  sv={sv} ev={ev}: {sample}{extra}")

# Phase 2: Full cell (ellipse + dot)
print("\nPhase 2: distinct full templates...", flush=True)
seen_full = set()
dup_count = 0
total_full = 0

for av in range(64):
    for sv in range(4):
        for ev in range(4):
            ellipse_mask = _make_ellipse_mask(CELL_PX, angle_v_to_deg(av),
                                              scale_v_to_frac(sv), elong_v_to_ratio(ev))
            for dx in range(8):
                for dy in range(8):
                    for r in range(4):
                        total_full += 1
                        dot_mask = _make_dot_mask(CELL_PX, dot_dx_to_px(dx, CELL_PX),
                                                  dot_dy_to_px(dy, CELL_PX), dot_rad_to_px(r))
                        combined = np.maximum(ellipse_mask, dot_mask)
                        img_t = 255 - combined
                        h = hash(img_t.tobytes())
                        if h in seen_full:
                            dup_count += 1
                        else:
                            seen_full.add(h)
                    if total_full % 100000 == 0:
                        print(f"  {total_full}/{262144} - {dup_count} dups, {len(seen_full)} distinct", flush=True)

print(f"\nTotal full templates: {total_full}")
print(f"Duplicates: {dup_count}")
print(f"Distinct: {len(seen_full)}")
print(f"Effective bits: {np.log2(len(seen_full)):.1f}")

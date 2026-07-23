"""Build codebook: map distinct pixel patterns → canonical codes."""
import pickle, sys, time
import numpy as np
from codec import angle_v_to_deg, scale_v_to_frac, elong_v_to_ratio
from codec import dot_dx_to_px, dot_dy_to_px, dot_rad_to_px
from generator import _make_ellipse_mask, _make_dot_mask

CELL_PX = 14

def build_codebook():
    """Build mappings for distinct visual patterns.

    Returns:
        codebook: list of dicts {code, value, av, sv, ev, dx, dy, r, template_bytes}
        reverse_map: dict {template_bytes: code}
    """
    print("Building codebook...", flush=True)
    t0 = time.time()

    # Step 1: Group all 262k param combos by visual hash
    hash_to_values = {}  # bytes_hash -> list of 18-bit values
    
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
                            t = 255 - np.maximum(em, dm)  # 0=ink, 255=bg
                            h = t.tobytes()
                            val = (av << 12) | (sv << 10) | (ev << 8) | \
                                  (dx << 5) | (dy << 2) | r
                            if h not in hash_to_values:
                                hash_to_values[h] = []
                            hash_to_values[h].append(val)

    print(f"  Distinct patterns: {len(hash_to_values)} of 262144", flush=True)

    # Step 2: Assign canonical code to each distinct pattern
    # Canonical value = the SMALLEST value (sorted by value, not by hash iteration order)
    codebook = []
    reverse_map = {}
    
    for h, vals in hash_to_values.items():
        canonical_val = min(vals)  # smallest 18-bit value = canonical
        code = len(codebook)
        av = (canonical_val >> 12) & 0x3F
        sv = (canonical_val >> 10) & 0x3
        ev = (canonical_val >> 8) & 0x3
        dx = (canonical_val >> 5) & 0x7
        dy = (canonical_val >> 2) & 0x7
        r = canonical_val & 0x3
        
        codebook.append({
            'code': code,
            'value': canonical_val,
            'av': av, 'sv': sv, 'ev': ev,
            'dx': dx, 'dy': dy, 'r': r,
        })
        reverse_map[h] = code
    
    elapsed = time.time() - t0
    n = len(codebook)
    effective_bits = np.log2(n)
    print(f"  Done in {elapsed:.1f}s: {n} entries, {effective_bits:.2f} bits/cell", flush=True)
    
    return codebook, reverse_map


def save_codebook(codebook, reverse_map, path="codebook.pkl"):
    """Save codebook to disk."""
    data = {
        'cell_px': CELL_PX,
        'codebook': codebook,
        'reverse_map': reverse_map,
        'n_distinct': len(codebook),
        'effective_bits': np.log2(len(codebook)),
    }
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    size_mb = sys.getsizeof(pickle.dumps(data, pickle.HIGHEST_PROTOCOL)) / 1024 / 1024
    print(f"  Saved to {path}: {size_mb:.1f} MB", flush=True)


def load_codebook(path="codebook.pkl"):
    """Load codebook from disk."""
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    cb, rm = build_codebook()
    save_codebook(cb, rm)

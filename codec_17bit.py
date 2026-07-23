#!/usr/bin/env python3
"""
Microform Code — 17-bit/cell Canonical Codec
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

17 bits per cell → 131,072 distinct visual patterns (subset of 188,819).
No ambiguity: each pattern maps to exactly one 17-bit value.
Clean RS error isolation: one cell error = 17 bits = ≤3 RS bytes.

Capacity: 8464 cells × 17 = 143,888 bits = 16,741 bytes raw (after RS ≈ 14,646 bytes payload)
"""
import struct, pickle, hashlib
import numpy as np
from reedsolo import RSCodec

CELL_PX = 14
BITS_PER_CELL = 17
N_PATTERNS = 2**17  # 131,072

RS_NSYM = 32
RS_N = 255
RS_K = RS_N - RS_NSYM  # 223

CODEBOOK_PATH = "codebook.pkl"

# ── Load codebook and build 17-bit mapping ──────────────────────────────

def _build_17bit_maps():
    """Build forward and reverse maps for 17-bit encoding.
    
    Returns:
        forward: list of 131,072 rendering param dicts (index = 17-bit value)
        reverse: dict {pixel_hash: 17-bit_value}
    """
    data = pickle.load(open(CODEBOOK_PATH, "rb"))
    cb = data["codebook"]
    rev_orig = data["reverse_map"]
    
    # Take first N_PATTERNS entries
    forward = cb[:N_PATTERNS]
    
    # Build reverse: for each selected entry, find its pixel hash and map to 17-bit value
    reverse = {}
    from codec import angle_v_to_deg, scale_v_to_frac, elong_v_to_ratio
    from codec import dot_dx_to_px, dot_dy_to_px, dot_rad_to_px
    from generator import _make_ellipse_mask, _make_dot_mask
    
    for idx, entry in enumerate(forward):
        # Render pixels
        av = entry["av"]
        sv = entry["sv"]
        ev = entry["ev"]
        dx = entry["dx"]
        dy = entry["dy"]
        r = entry["r"]
        
        em = _make_ellipse_mask(CELL_PX, angle_v_to_deg(av),
                                scale_v_to_frac(sv), elong_v_to_ratio(ev))
        dm = _make_dot_mask(CELL_PX, dot_dx_to_px(dx, CELL_PX),
                            dot_dy_to_px(dy, CELL_PX), dot_rad_to_px(r))
        pixels = (255 - np.maximum(em, dm)).astype(np.uint8)
        reverse[pixels.tobytes()] = idx
    
    return forward, reverse


# Cache the maps
_forward_17 = None
_reverse_17 = None

def _get_maps():
    global _forward_17, _reverse_17
    if _forward_17 is None:
        print("Building 17-bit codec maps...", flush=True)
        _forward_17, _reverse_17 = _build_17bit_maps()
        print(f"  {len(_forward_17)} patterns, {len(_reverse_17)} reverse entries", flush=True)
    return _forward_17, _reverse_17


def render_params(code_17bit: int) -> dict:
    """Get rendering parameters for a 17-bit code.
    
    Returns dict with: av, sv, ev, dx, dy, r
    """
    forward, _ = _get_maps()
    if code_17bit >= len(forward):
        code_17bit = 0
    entry = forward[code_17bit]
    return {
        "av": entry["av"], "sv": entry["sv"], "ev": entry["ev"],
        "dx": entry["dx"], "dy": entry["dy"], "r": entry["r"],
    }


def pixels_to_17bit(pixels: np.ndarray) -> int:
    """Look up cell pixels → 17-bit code.
    Returns -1 if not found."""
    _, reverse = _get_maps()
    h = pixels.astype(np.uint8).tobytes()
    return reverse.get(h, -1)


# ── RS encode/decode (operates on raw bitstream) ────────────────────────

def _align_to_rs(total_bits: int) -> int:
    total_bytes = total_bits // 8
    if total_bytes < RS_N:
        return total_bytes
    return (total_bytes // RS_N) * RS_N


def encode_bits(user_data: bytes, total_bits: int) -> list[int]:
    """Encode user data with RS ECC → flat bit array (MSB first)."""
    total_bytes = _align_to_rs(total_bits)
    n_blocks = total_bytes // RS_N
    data_bytes = n_blocks * RS_K

    payload = struct.pack(">I", len(user_data)) + user_data
    if len(payload) > data_bytes:
        payload = payload[:data_bytes]

    # Pad with PRNG
    pad_len = data_bytes - len(payload)
    if pad_len > 0:
        seed = int.from_bytes(
            hashlib.sha256(struct.pack(">II", total_bytes, len(payload))).digest()[:4], "big"
        )
        pad_rng = np.random.default_rng(seed)
        padded = payload + pad_rng.bytes(pad_len)
    else:
        padded = payload

    # RS encode
    rs = RSCodec(RS_NSYM)
    all_bytes = b""
    for i in range(n_blocks):
        block = padded[i * RS_K:(i + 1) * RS_K]
        all_bytes += rs.encode(block)

    # Fill remaining
    if len(all_bytes) < total_bytes:
        seed = int.from_bytes(
            hashlib.sha256(struct.pack(">II", total_bytes, len(payload))).digest()[:8], "big"
        )
        fill_rng = np.random.default_rng(seed)
        all_bytes += fill_rng.bytes(total_bytes - len(all_bytes))

    # → flat bits MSB first
    bits = []
    for b in all_bytes:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    while len(bits) < total_bits:
        bits.append(0)
    return bits[:total_bits]


def decode_bits(bits: list[int]) -> bytes:
    """RS-decode bit array → user data."""
    total_bits = len(bits) - (len(bits) % 8)
    if total_bits < 8:
        return b""
    total_bytes = total_bits // 8
    aligned = _align_to_rs(total_bits)
    if total_bytes > aligned:
        total_bytes = aligned

    # bits → bytes
    raw = bytearray()
    for i in range(0, total_bytes * 8, 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i + j]
        raw.append(byte)

    # RS decode
    n_blocks = len(raw) // RS_N
    rs = RSCodec(RS_NSYM)
    decoded = bytearray()
    for i in range(n_blocks):
        block = raw[i * RS_N:(i + 1) * RS_N]
        try:
            d, _, _ = rs.decode(block)
            decoded.extend(d)
        except Exception:
            decoded.extend(block[:RS_K])

    decoded = bytes(decoded)
    if len(decoded) < 4:
        return decoded
    data_len = struct.unpack(">I", decoded[:4])[0]
    if not (0 < data_len < len(decoded) - 4):
        return decoded[4:]
    return decoded[4:4 + data_len]


# ── Bit packing: 17 bits ↔ cell values ──────────────────────────────────

def bits_to_cell_values(bits: list[int]) -> list[int]:
    """Split bit array into 17-bit cell values."""
    n_cells = (len(bits) + BITS_PER_CELL - 1) // BITS_PER_CELL
    values = []
    for i in range(n_cells):
        start = i * BITS_PER_CELL
        val = 0
        for b in range(BITS_PER_CELL):
            if start + b < len(bits):
                val = (val << 1) | bits[start + b]
        values.append(val)
    return values


def cell_values_to_bits(values: list[int], total_bits: int) -> list[int]:
    """Convert 17-bit cell values back to bit array."""
    bits = []
    for val in values:
        for b in range(BITS_PER_CELL - 1, -1, -1):
            bits.append((val >> b) & 1)
    return bits[:total_bits]


# ── Convenience ─────────────────────────────────────────────────────────

def encode(user_data: bytes, n_cells: int) -> list[int]:
    """Encode data into 17-bit cell values."""
    total_bits = n_cells * BITS_PER_CELL
    bits = encode_bits(user_data, total_bits)
    return bits_to_cell_values(bits)


def decode(cell_values: list[int]) -> bytes:
    """Decode 17-bit cell values back to user data."""
    total_bits = len(cell_values) * BITS_PER_CELL
    bits = cell_values_to_bits(cell_values, total_bits)
    return decode_bits(bits)


# ── Capacity ────────────────────────────────────────────────────────────

def capacity_info(n_cells: int) -> dict:
    raw_bits = n_cells * BITS_PER_CELL
    aligned_bytes = _align_to_rs(raw_bits)
    n_blocks = aligned_bytes // RS_N
    data_bytes = n_blocks * RS_K - 4
    return dict(
        cells=n_cells, bits_per_cell=BITS_PER_CELL,
        raw_bits=raw_bits, raw_kb=raw_bits / 8 / 1024,
        data_bytes=max(0, data_bytes),
        data_kb=max(0, data_bytes / 1024),
        rs_nsym=RS_NSYM, rs_blocks=n_blocks,
    )


if __name__ == "__main__":
    test_data = b"Microform Code v2 - 17-bit canonical encoding with RS ECC."
    n_cells = 92 * 92
    print(f"Encoding {len(test_data)} bytes into {n_cells} cells...")
    
    # Encode
    vals = encode(test_data, n_cells)
    print(f"  {len(vals)} values, first 5: {vals[:5]}")
    
    # Decode
    result = decode(vals)
    print(f"Decoded: {len(result)} bytes, match={result == test_data}")
    print(f"  Data: {result}")
    
    info = capacity_info(n_cells)
    print(f"\nCapacity: {info['raw_kb']:.1f} KB raw, {info['data_kb']:.1f} KB payload")

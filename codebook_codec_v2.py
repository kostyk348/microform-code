#!/usr/bin/env python3
"""
Microform Code — Canonical 18-bit Codec
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Uses the codebook to canonicalize 18-bit cell values:
- Encoder maps any 18-bit value → canonical visual pattern
- Decoder reads visual pattern → canonical 18-bit value

Preserves 18-bit/cell encoding. Each cell error = 18 bits = ≤3 RS bytes.
"""
import struct, pickle, hashlib
import numpy as np
from reedsolo import RSCodec

CELL_PX = 14
BITS_PER_CELL = 18

# RS(255,223) parameters
RS_NSYM = 32
RS_N = 255
RS_K = RS_N - RS_NSYM  # 223

CODEBOOK_PATH = "codebook.pkl"

# Cache for codebook data
_cb_cache = None

def _get_codebook():
    global _cb_cache
    if _cb_cache is None:
        _cb_cache = pickle.load(open(CODEBOOK_PATH, "rb"))
    return _cb_cache


def _render_pixels(value_18bit):
    """Render the pixel pattern for an 18-bit value."""
    from codec import angle_v_to_deg, scale_v_to_frac, elong_v_to_ratio
    from codec import dot_dx_to_px, dot_dy_to_px, dot_rad_to_px
    from generator import _make_ellipse_mask, _make_dot_mask
    
    av = (value_18bit >> 12) & 0x3F
    sv = (value_18bit >> 10) & 0x3
    ev = (value_18bit >> 8) & 0x3
    dx = (value_18bit >> 5) & 0x7
    dy = (value_18bit >> 2) & 0x7
    r = value_18bit & 0x3
    
    em = _make_ellipse_mask(CELL_PX, angle_v_to_deg(av),
                            scale_v_to_frac(sv), elong_v_to_ratio(ev))
    dm = _make_dot_mask(CELL_PX, dot_dx_to_px(dx, CELL_PX),
                        dot_dy_to_px(dy, CELL_PX), dot_rad_to_px(r))
    return (255 - np.maximum(em, dm)).astype(np.uint8)


def canonical_cell_value(value_18bit):
    """Canonicalize: map any 18-bit value → canonical (min in equivalence class)."""
    data = _get_codebook()
    rev = data["reverse_map"]
    pixels = _render_pixels(value_18bit)
    code = rev.get(pixels.tobytes(), -1)
    if code < 0:
        return value_18bit
    return data["codebook"][code]["value"]


def pixels_to_18bit(pixels):
    """Read 14×14 cell pixels (uint8, 0=ink 255=bg) → canonical 18-bit value.
    Returns -1 if not found."""
    data = _get_codebook()
    code = data["reverse_map"].get(pixels.astype(np.uint8).tobytes(), -1)
    if code < 0:
        return -1
    return data["codebook"][code]["value"]


# ── RS encode/decode ────────────────────────────────────────────────────

def _align_total_bytes(capacity_bits):
    total_bytes = capacity_bits // 8
    if total_bytes < RS_N:
        return total_bytes
    return (total_bytes // RS_N) * RS_N


def encode_data_to_bits(user_data, total_bits):
    """Encode user data to RS-protected bit array of exactly total_bits."""
    total_bytes = _align_total_bytes(total_bits)
    n_blocks = total_bytes // RS_N
    data_bytes = n_blocks * RS_K

    payload = struct.pack(">I", len(user_data)) + user_data
    if len(payload) > data_bytes:
        payload = payload[:data_bytes]

    pad_len = data_bytes - len(payload)
    if pad_len > 0:
        seed = int.from_bytes(
            hashlib.sha256(struct.pack(">II", total_bytes, len(payload))).digest()[:4], "big"
        )
        pad_rng = np.random.default_rng(seed)
        padded = payload + pad_rng.bytes(pad_len)
    else:
        padded = payload

    rs = RSCodec(RS_NSYM)
    all_bytes = b""
    for i in range(n_blocks):
        block = padded[i * RS_K:(i + 1) * RS_K]
        all_bytes += rs.encode(block)

    if len(all_bytes) < total_bytes:
        seed = int.from_bytes(
            hashlib.sha256(struct.pack(">II", total_bytes, len(payload))).digest()[:8], "big"
        )
        fill_rng = np.random.default_rng(seed)
        all_bytes += fill_rng.bytes(total_bytes - len(all_bytes))

    bits = []
    for b in all_bytes:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    while len(bits) < total_bits:
        bits.append(0)
    return bits[:total_bits]


def decode_bits_to_data(bits):
    """RS-decode bit array back to user data."""
    total_bits = len(bits) - (len(bits) % 8)
    if total_bits < 8:
        return b""
    total_bytes = total_bits // 8
    aligned = _align_total_bytes(total_bits)
    if total_bytes > aligned:
        total_bytes = aligned

    raw = bytearray()
    for i in range(0, total_bytes * 8, 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i + j]
        raw.append(byte)

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


# ── Cell-level operations ────────────────────────────────────────────────

def bits_to_cell_values(bits):
    """Split bit array into 18-bit canonicalized cell values."""
    n_cells = (len(bits) + BITS_PER_CELL - 1) // BITS_PER_CELL
    values = []
    for i in range(n_cells):
        start = i * BITS_PER_CELL
        val = 0
        for b in range(BITS_PER_CELL):
            if start + b < len(bits):
                val = (val << 1) | bits[start + b]
        values.append(canonical_cell_value(val))
    return values


def cell_values_to_bits(values, total_bits):
    """Convert 18-bit cell values back to bit array."""
    bits = []
    for val in values:
        for b in range(BITS_PER_CELL - 1, -1, -1):
            bits.append((val >> b) & 1)
    return bits[:total_bits]


# ── Convenience ─────────────────────────────────────────────────────────

def encode(user_data, n_cells):
    """Encode data into canonical 18-bit cell values."""
    total_bits = n_cells * BITS_PER_CELL
    bits = encode_data_to_bits(user_data, total_bits)
    return bits_to_cell_values(bits)


def decode(cell_values, total_bits=None):
    """Decode cell values back to user data."""
    if total_bits is None:
        total_bits = len(cell_values) * BITS_PER_CELL
    bits = cell_values_to_bits(cell_values, total_bits)
    return decode_bits_to_data(bits)


# ── Capacity ────────────────────────────────────────────────────────────

def capacity_info(n_cells):
    raw_bits = n_cells * BITS_PER_CELL
    aligned_bytes = _align_total_bytes(raw_bits)
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
    test_data = b"Microform Code v2 - 18-bit canonical encoding with RS ECC."
    n_cells = 92 * 92
    print(f"Encoding {len(test_data)} bytes into {n_cells} cells...")
    vals = encode(test_data, n_cells)
    print(f"  {len(vals)} values, first 5: {vals[:5]}")

    result = decode(vals)
    print(f"Decoded: {len(result)} bytes")
    print(f"  Match: {result == test_data}")
    print(f"  Data: {result}")

    info = capacity_info(n_cells)
    print(f"\nCapacity: {info['raw_kb']:.1f} KB raw, {info['data_kb']:.1f} KB payload")

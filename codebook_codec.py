#!/usr/bin/env python3
"""
Microform Code — Codebook-based Codec
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Uses the distinct-visual-pattern codebook for unambiguous encoding/decoding.

Encoding chain:
  user data → RS(255,223) → base-188819 digits → codebook params → cell rendering

Decoding chain (reverse):
  cell pixels → reverse_map lookup → base-188819 digits → bigint → RS decode → data
"""
import struct, pickle, hashlib
import numpy as np
from reedsolo import RSCodec

CELL_PX = 14

# RS(255,223) parameters
RS_NSYM = 32
RS_N = 255
RS_K = RS_N - RS_NSYM  # 223

# Codebook path
CODEBOOK_PATH = "codebook.pkl"

# ── Codebook loading ─────────────────────────────────────────────────────

_codebook_cache = None
_reverse_cache = None

def _load_codebook():
    global _codebook_cache, _reverse_cache
    if _codebook_cache is None:
        data = pickle.load(open(CODEBOOK_PATH, "rb"))
        _codebook_cache = data["codebook"]
        _reverse_cache = data["reverse_map"]
    return _codebook_cache, _reverse_cache


def codebook_size() -> int:
    """Number of distinct visual patterns (≈188,819)."""
    cb, _ = _load_codebook()
    return len(cb)


def effective_bits_per_cell() -> float:
    return np.log2(codebook_size())


# ── Data → cell codes ────────────────────────────────────────────────────

def encode_to_codes(user_data: bytes, n_cells: int,
                    rs_nsym: int = RS_NSYM) -> list[int]:
    """Encode user data into a list of cell codes (0..codebook_size-1).

    Args:
        user_data: raw bytes to encode
        n_cells: number of cells to fill
        rs_nsym: RS parity bytes per block (32 → RS(255,223))

    Returns:
        list of `n_cells` integer codes
    """
    cb, _ = _load_codebook()
    n_codes = len(cb)

    # 1. Compute total payload capacity in bytes
    #    Each cell stores ~log2(n_codes) bits → total bits → total bytes
    import math
    log2_n = np.log2(n_codes)
    total_bits = int(math.ceil(n_cells * log2_n))
    total_bytes = total_bits // 8

    # Align to RS block size
    n_blocks = max(1, total_bytes // RS_N)
    aligned_bytes = n_blocks * RS_N
    data_bytes = n_blocks * RS_K

    # 2. Build payload with length prefix
    payload = struct.pack(">I", len(user_data)) + user_data
    if len(payload) > data_bytes:
        payload = payload[:data_bytes]

    # 3. Pad with seeded PRNG (not zeros!)
    pad_seed = int.from_bytes(
        hashlib.sha256(struct.pack(">II", total_bytes, len(payload))).digest()[:4], "big"
    )
    pad_rng = np.random.default_rng(pad_seed)
    padded = payload + pad_rng.bytes(data_bytes - len(payload))

    # 4. RS encode
    rs = RSCodec(rs_nsym)
    all_bytes = b""
    for i in range(n_blocks):
        block = padded[i * RS_K:(i + 1) * RS_K]
        all_bytes += rs.encode(block)

    # 5. Pad to aligned_bytes (if not enough RS blocks)
    if len(all_bytes) < aligned_bytes:
        fill_seed = int.from_bytes(
            hashlib.sha256(struct.pack(">II", aligned_bytes, len(payload))).digest()[:8], "big"
        )
        fill_rng = np.random.default_rng(fill_seed)
        all_bytes += fill_rng.bytes(aligned_bytes - len(all_bytes))

    # 6. Convert bytes → bigint → base-n_codes digits
    bigint = int.from_bytes(all_bytes, "big")
    codes = []
    for _ in range(n_cells):
        codes.append(bigint % n_codes)
        bigint //= n_codes
    codes.reverse()  # most significant first → cell[0]

    return codes


def codes_from_bytes(byte_data: bytes, n_cells: int) -> list[int]:
    """Convert raw bytes (post-RS) to cell codes.

    Used when the caller wants full control over RS encoding.
    """
    cb, _ = _load_codebook()
    n_codes = len(cb)
    bigint = int.from_bytes(byte_data, "big")
    codes = []
    for _ in range(n_cells):
        codes.append(bigint % n_codes)
        bigint //= n_codes
    codes.reverse()
    return codes


# ── Cell codes → data ────────────────────────────────────────────────────

def decode_from_codes(codes: list[int], rs_nsym: int = RS_NSYM) -> bytes:
    """Decode a list of cell codes back to user data.

    Args:
        codes: list of integer codes (len should match encode side)
        rs_nsym: same as encoding

    Returns:
        decoded user data bytes
    """
    cb, _ = _load_codebook()
    n_codes = len(cb)

    # 1. Convert codes → bigint
    bigint = 0
    for c in codes:
        bigint = bigint * n_codes + c

    # 2. Convert to bytes
    # Compute aligned byte count the same way as encode_to_codes
    total_bits = int(np.ceil(len(codes) * np.log2(n_codes)))
    total_bytes = total_bits // 8
    n_blocks = max(1, total_bytes // RS_N)
    aligned_bytes = n_blocks * RS_N

    # Safety: truncate if bigint exceeds capacity (corrupted data)
    max_val = (1 << (aligned_bytes * 8)) - 1
    if bigint > max_val:
        bigint = bigint & max_val

    raw_bytes = bigint.to_bytes(aligned_bytes, "big")

    # 3. RS decode
    rs = RSCodec(rs_nsym)
    decoded = bytearray()
    for i in range(n_blocks):
        block = raw_bytes[i * RS_N:(i + 1) * RS_N]
        try:
            data, _, _ = rs.decode(block)
            decoded.extend(data)
        except Exception:
            decoded.extend(block[:RS_K])

    decoded = bytes(decoded)

    # 4. Read length prefix
    if len(decoded) < 4:
        return decoded
    data_len = struct.unpack(">I", decoded[:4])[0]
    if not (0 < data_len < len(decoded) - 4):
        return decoded[4:]
    return decoded[4:4 + data_len]


def codes_to_bytes(codes: list[int], total_bytes: int) -> bytes:
    """Convert cell codes back to raw bytes (before RS decode)."""
    cb, _ = _load_codebook()
    n_codes = len(cb)
    bigint = 0
    for c in codes:
        bigint = bigint * n_codes + c
    return bigint.to_bytes(total_bytes, "big")


# ── Code ↔ rendering params ──────────────────────────────────────────────

def code_to_render_params(code: int) -> dict:
    """Get rendering parameters for a code.

    Returns dict with: av, sv, ev, dx, dy, r, value (18-bit canonical), code
    """
    cb, _ = _load_codebook()
    return cb[code]


def pixels_to_code(cell_pixels: np.ndarray) -> int:
    """Look up a 14×14 cell image and return its codebook code.

    Args:
        cell_pixels: 14×14 numpy array (0=ink, 255=bg)

    Returns:
        codebook index (0..n_distinct-1), or -1 if not found
    """
    _, rev = _load_codebook()
    h = cell_pixels.astype(np.uint8).tobytes()
    return rev.get(h, -1)


# ── Capacity estimate ────────────────────────────────────────────────────

def capacity_info(n_cells: int, rs_nsym: int = RS_NSYM) -> dict:
    """Estimate data capacity for n_cells."""
    import math
    n_codes = codebook_size()
    raw_bits = int(math.ceil(n_cells * np.log2(n_codes)))
    aligned_bytes = max(1, (raw_bits // 8) // RS_N) * RS_N
    n_blocks = aligned_bytes // RS_N
    data_bytes = n_blocks * RS_K - 4  # minus length prefix

    return dict(
        cells=n_cells,
        bits_per_cell=effective_bits_per_cell(),
        raw_bits=raw_bits,
        raw_kb=raw_bits / 8 / 1024,
        data_bytes=max(0, data_bytes),
        data_kb=max(0, data_bytes / 1024),
        rs_nsym=rs_nsym,
        rs_blocks=n_blocks,
    )


# ── Standalone test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    test_data = b"Hello, Microform Code! This is a test of the codebook encoding system."
    n_cells = 92 * 92  # 8464
    print(f"Encoding {len(test_data)} bytes into {n_cells} cells...")
    codes = encode_to_codes(test_data, n_cells)
    print(f"  Generated {len(codes)} codes")
    print(f"  First 5 codes: {codes[:5]}")

    decoded = decode_from_codes(codes)
    print(f"Decoded: {len(decoded)} bytes")
    print(f"  Match: {decoded == test_data}")
    print(f"  Data: {decoded}")

#!/usr/bin/env python3
"""
Microform Code — Codec: data encoding + Reed-Solomon ECC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Схема:
  user data → [length prefix 4B] → RS(255,223) ECC → PRNG шум в остаток
  → битовый массив

Весь паттерн заполняется равномерно: сперва RS-блоки (выглядят как шум),
потом PRNG-шум. Визуально — однородная текстура.
"""

import hashlib
import struct
import numpy as np
from reedsolo import RSCodec

# ── Cell encoding ────────────────────────────────────────────────────────
# Dual-element: 18 bits per cell
#   bits 17..8 (10 bits): main ellipse
#     bits 17..12: angle     (6 bits, 0-63 → 0-179°)
#     bits 11..10: scale     (2 bits, 0-3 → 0.50-0.90)
#     bits 9..8:  elong     (2 bits, 0-3 → 1.0-2.5)
#   bits 7..0 (8 bits): micro-dot
#     bits 7..5:  dot_dx    (3 bits, 0-7 → 7 x-позиций)
#     bits 4..2:  dot_dy    (3 bits, 0-7 → 7 y-позиций)
#     bits 1..0:  dot_rad   (2 bits, 0-3 → 4 радиуса)

BITS_PER_CELL = 18

# ── Параметры ECC ────────────────────────────────────────────────────────
RS_NSYM = 32        # parity bytes per RS block
RS_N = 255          # total bytes per RS block
RS_K = RS_N - RS_NSYM  # 223 data bytes per block
RS_BLOCK_BYTES = 255  # must be ≤ 255 for GF(256)


def _align_total_bytes(capacity_bits: int) -> int:
    """Округлить ёмкость вниз до 255-байтных блоков.
    Если ёмкость меньше одного блока — используем что есть."""
    total_bytes = capacity_bits // 8
    if total_bytes < RS_BLOCK_BYTES:
        return total_bytes
    aligned = (total_bytes // RS_BLOCK_BYTES) * RS_BLOCK_BYTES
    return max(RS_BLOCK_BYTES, aligned)


# ── Маппинг: cell value → параметры ────────────────────────────────────
def cell_value_to_params(value: int) -> dict:
    """18-bit cell value → параметры для рендеринга."""
    angle_v    = (value >> 12) & 0x3F
    scale_v    = (value >> 10) & 0x3
    elong_v    = (value >> 8)  & 0x3
    dot_dx     = (value >> 5)  & 0x7
    dot_dy     = (value >> 2)  & 0x7
    dot_rad    = value & 0x3
    return dict(angle_v=angle_v, scale_v=scale_v, elong_v=elong_v,
                dot_dx=dot_dx, dot_dy=dot_dy, dot_rad=dot_rad)


# ── Маппинг: параметры → физические значения ───────────────────────────
def angle_v_to_deg(v: int) -> float:
    return (v / 63.0) * 179.0

def scale_v_to_frac(v: int) -> float:
    return 0.50 + v * 0.133  # 0.50, 0.63, 0.77, 0.90

def elong_v_to_ratio(v: int) -> float:
    return 1.0 + v * 0.5  # 1.0, 1.5, 2.0, 2.5

def dot_dx_to_px(v: int, cell_size_px: int) -> float:
    margin = cell_size_px * 0.10
    span = cell_size_px * 0.5 - margin
    return -span + v * (2 * span / 7.0)

def dot_dy_to_px(v: int, cell_size_px: int) -> float:
    return dot_dx_to_px(v, cell_size_px)

def dot_rad_to_px(v: int) -> float:
    return 1.5 + v * 0.8  # 1.5, 2.3, 3.1, 3.9


# ── RS ECC ──────────────────────────────────────────────────────────────
def encode_bits(user_data: bytes, total_bits: int,
                rs_nsym: int = RS_NSYM) -> list[int]:
    """
    user_data → RS-закодированный битовый массив ровно total_bits.

    - Добавляется 4-байтовый length prefix
    - RS-кодирование до целых блоков
    - Остаток — PRNG-шум (визуально равномерный орнамент)
    """
    total_bytes = _align_total_bytes(total_bits)
    rs_k = RS_N - rs_nsym

    if len(user_data) == 0:
        user_data = b'\x00'

    # 1. Length prefix + данные
    payload = struct.pack(">I", len(user_data)) + user_data
    n_blocks = total_bytes // RS_N
    max_data = n_blocks * rs_k

    if len(payload) > max_data:
        payload = payload[:max_data]

    # 2. Pad с PRNG-шумом (не нулями!)
    pad_len = n_blocks * rs_k - len(payload)
    if pad_len > 0:
        pad_seed = int.from_bytes(
            hashlib.sha256(struct.pack(">II", total_bytes, len(payload))).digest()[:4],
            "big"
        )
        pad_rng = np.random.default_rng(pad_seed)
        padded = payload + pad_rng.bytes(pad_len)
    else:
        padded = payload

    rs = RSCodec(rs_nsym)
    all_bytes = b''
    for i in range(n_blocks):
        block = padded[i * rs_k:(i + 1) * rs_k]
        all_bytes += rs.encode(block)

    # 3. Если после RS-блоков ещё осталось место — PRNG-шум
    if len(all_bytes) < total_bytes:
        seed = int.from_bytes(
            hashlib.sha256(struct.pack(">II", total_bytes, len(payload))).digest()[:8],
            "big"
        )
        rng = np.random.default_rng(seed)
        all_bytes += rng.bytes(total_bytes - len(all_bytes))

    # → flat bits MSB first
    bits = []
    for b in all_bytes:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)

    # Добиваем ровно до total_bits (на случай если aligned < raw)
    while len(bits) < total_bits:
        bits.append(0)

    return bits[:total_bits]


def decode_bits(bits: list[int], rs_nsym: int = RS_NSYM) -> bytes:
    """Битовый массив → RS-декодированные данные (с length prefix)."""
    total_bits = len(bits) - (len(bits) % 8)
    if total_bits < 8:
        return b''

    total_bytes = total_bits // 8
    aligned = _align_total_bytes(total_bits)
    if total_bytes > aligned:
        total_bytes = aligned

    # bits → bytes
    raw = bytearray()
    for i in range(0, total_bytes * 8, 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i + j]
        raw.append(byte)
    raw = bytes(raw)

    # RS-декодирование
    n_blocks = len(raw) // RS_N
    rs = RSCodec(rs_nsym)
    rs_k = RS_N - rs_nsym

    decoded = bytearray()
    for i in range(n_blocks):
        block = raw[i * RS_N:(i + 1) * RS_N]
        try:
            data, _, _ = rs.decode(block)
            decoded.extend(data)
        except Exception:
            decoded.extend(block[:rs_k])

    decoded = bytes(decoded)

    # Читаем length prefix
    if len(decoded) < 4:
        return decoded
    data_len = struct.unpack(">I", decoded[:4])[0]
    if not (0 < data_len < len(decoded) - 4):
        return decoded[4:]
    return decoded[4:4 + data_len]


# ── Статистика ёмкости ─────────────────────────────────────────────────
def capacity_info(grid_cells: int, rs_nsym: int = RS_NSYM):
    """Информация о ёмкости: сколько данных влезает."""
    raw_bits = grid_cells * BITS_PER_CELL
    aligned_bytes = _align_total_bytes(raw_bits)
    n_blocks = aligned_bytes // RS_N
    rs_k = RS_N - rs_nsym
    data_bytes = n_blocks * rs_k - 4  # 4B length prefix
    qr_x = max(0, data_bytes) / 3072  # QR-40 ≈ 3KB

    return dict(
        cells=grid_cells,
        bits_per_cell=BITS_PER_CELL,
        raw_bits=raw_bits,
        used_bits=aligned_bytes * 8,
        raw_kb=raw_bits / 8 / 1024,
        data_bytes=max(0, data_bytes),
        data_kb=max(0, data_bytes / 1024),
        rs_nsym=rs_nsym,
        rs_blocks=n_blocks,
        ecc_overhead_pct=(1 - max(0, data_bytes) / raw_bits * 8) * 100 if raw_bits else 0,
        qr_x=qr_x,
    )

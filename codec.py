#!/usr/bin/env python3
"""
Microform Code — Codec: data encoding + Reed-Solomon ECC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Схема кодирования:
  user data → RS ECC → bit array → cell values → параметры форм
"""

from reedsolo import RSCodec

# ── Cell encoding ────────────────────────────────────────────────────────
# Dual-element: 18 bits per cell
#   bits 17..8 (10 bits): main ellipse
#     bits 17..12: angle     (6 bits, 0-63 → 0-179°)
#     bits 11..10: scale     (2 bits, 0-3 → 0.30-0.65)
#     bits 9..8:  elong     (2 bits, 0-3 → 1.0-2.5)
#   bits 7..0 (8 bits): micro-dot
#     bits 7..5:  dot_dx    (3 bits, 0-7 → 7 x-позиций)
#     bits 4..2:  dot_dy    (3 bits, 0-7 → 7 y-позиций)
#     bits 1..0:  dot_rad   (2 bits, 0-3 → 4 радиуса)

BITS_PER_CELL = 18

# ── Параметры по умолчанию ──────────────────────────────────────────────
RS_NSYM = 32        # parity bytes per RS block
RS_N = 255          # total bytes per RS block
RS_K = RS_N - RS_NSYM  # 223 data bytes per block


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


def params_to_value(angle_v: int, scale_v: int, elong_v: int,
                    dot_dx: int, dot_dy: int, dot_rad: int) -> int:
    """Параметры → 18-bit cell value."""
    return ((angle_v & 0x3F) << 12) | ((scale_v & 3) << 10) | \
           ((elong_v & 3) << 8) | ((dot_dx & 7) << 5) | \
           ((dot_dy & 7) << 2) | (dot_rad & 3)


# ── Маппинг: параметры → физические значения ───────────────────────────
def angle_v_to_deg(v: int) -> float:
    return (v / 63.0) * 179.0

def scale_v_to_frac(v: int) -> float:
    return 0.40 + v * 0.133  # 0.40, 0.53, 0.67, 0.80

def elong_v_to_ratio(v: int) -> float:
    return 1.0 + v * 0.5  # 1.0, 1.5, 2.0, 2.5

def dot_dx_to_px(v: int, cell_size_px: int) -> float:
    """dx 0-7 → x-offset от центра ячейки (в пикселях)."""
    margin = cell_size_px * 0.15
    span = cell_size_px * 0.5 - margin
    return -span + v * (2 * span / 7.0)

def dot_dy_to_px(v: int, cell_size_px: int) -> float:
    return dot_dx_to_px(v, cell_size_px)  # symmetric

def dot_rad_to_px(v: int) -> float:
    return 1.5 + v * 0.5  # 1.5, 2.0, 2.5, 3.0


# ── RS ECC ──────────────────────────────────────────────────────────────
def encode_bits(user_data: bytes, total_bits: int,
                rs_nsym: int = RS_NSYM) -> list[int]:
    """
    user_data → RS-закодированный битовый массив ровно total_bits.
    Если user_data не влезает — обрезается.
    Если влезает с запасом — добивается нулями.
    """
    total_bytes = (total_bits + 7) // 8
    rs_k = 255 - rs_nsym

    # Сколько блоков нужно
    if len(user_data) == 0:
        user_data = b'\x00'

    # Считаем, сколько данных влезет
    n_blocks = (total_bytes + 254) // 255  # ceil division
    max_data_bytes = n_blocks * rs_k

    if len(user_data) > max_data_bytes:
        user_data = user_data[:max_data_bytes]
        print(f"  ⚠  Данные обрезаны до {max_data_bytes} B")

    # Паддинг до целого числа блоков
    pad_len = n_blocks * rs_k - len(user_data)
    padded = user_data + b'\x00' * pad_len

    # RS-кодирование
    rs = RSCodec(rs_nsym)
    all_encoded = b''
    for i in range(n_blocks):
        block = padded[i * rs_k:(i + 1) * rs_k]
        encoded = rs.encode(block)  # → 255 bytes
        all_encoded += encoded

    # Обрезаем/добиваем до ровно total_bytes
    if len(all_encoded) > total_bytes:
        all_encoded = all_encoded[:total_bytes]
    elif len(all_encoded) < total_bytes:
        all_encoded += b'\x00' * (total_bytes - len(all_encoded))

    # → flat bits MSB first
    bits = []
    for b in all_encoded:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    return bits[:total_bits]


def decode_bits(bits: list[int], rs_nsym: int = RS_NSYM) -> bytes:
    """Битовый массив → RS-декодированные данные."""
    total_bits = len(bits) - (len(bits) % 8)
    if total_bits < 8:
        return b''

    # bits → bytes
    raw_bytes = bytearray()
    for i in range(0, total_bits, 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i + j]
        raw_bytes.append(byte)

    total_bytes = len(raw_bytes)
    n_blocks = (total_bytes + 254) // 255
    rs = RSCodec(rs_nsym)

    decoded = bytearray()
    for i in range(n_blocks):
        block = raw_bytes[i * 255:(i + 1) * 255]
        if len(block) < 255:
            block = block + b'\x00' * (255 - len(block))
        try:
            data, _, _ = rs.decode(block)
            decoded.extend(data)
        except Exception as e:
            print(f"  ⚠  RS-ошибка в блоке {i}: {e}")
            decoded.extend(block[:rs_k])

    return bytes(decoded)


# ── Статистика ──────────────────────────────────────────────────────────
def capacity_info(grid_cells: int, rs_nsym: int = RS_NSYM):
    """Вернуть словарь с информацией о ёмкости."""
    raw_bits = grid_cells * BITS_PER_CELL
    raw_bytes = raw_bits // 8
    rs_k = 255 - rs_nsym
    n_blocks = (raw_bytes + 254) // 255
    data_bytes = n_blocks * rs_k
    return dict(
        cells=grid_cells,
        bits_per_cell=BITS_PER_CELL,
        raw_bits=raw_bits,
        raw_kb=raw_bits / 8 / 1024,
        data_bytes=data_bytes,
        data_kb=data_bytes / 1024,
        rs_nsym=rs_nsym,
        rs_blocks=n_blocks,
        ecc_overhead_pct=(1 - data_bytes / raw_bytes) * 100 if raw_bytes else 0,
    )

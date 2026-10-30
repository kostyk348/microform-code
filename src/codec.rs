//! Microform Code — the 17-bit codec: framing, Reed–Solomon ECC, bit packing.
//!
//! Bitstream layout (identical to the reference prototype):
//!
//! ```text
//!   user data → [4-byte big-endian length prefix] → split into RS(255,223)
//!   blocks → parity appended → leftover capacity filled with deterministic
//!   PRNG noise → MSB-first bitstream → 17 bits per cell (row-major)
//! ```
//!
//! One RS block is exactly 255 bytes = 2040 bits = **120 cells**, so a
//! cell error (17 bits ≈ up to 3 bytes) stays inside one block and RS
//! isolation is clean.

use crate::layout::BITS_PER_CELL;
use reed_solomon::{Decoder, Encoder};

pub const RS_NSYM: usize = 32;
pub const RS_N: usize = 255;
pub const RS_K: usize = RS_N - RS_NSYM; // 223

/// Cells per RS block at the canonical 17-bit depth: 2040 / 17 = 120.
pub const CELLS_PER_BLOCK: usize = RS_N * 8 / BITS_PER_CELL as usize;

/// Byte span of one cell error at the given bit depth (worst case).
pub fn cell_byte_span(bits_per_cell: u32) -> usize {
    bits_per_cell.div_ceil(8) as usize
}

/// Cells per RS block at the given depth (exact for RS-aligned depths).
pub fn cells_per_block(bits_per_cell: u32) -> usize {
    (RS_N * 8) / bits_per_cell as usize
}

/// Guaranteed-correctable unknown-position cell errors per block.
/// A byte error costs 2 Reed–Solomon syndrome symbols (2e + f ≤ 32),
/// so the budget is RS_NSYM/2 = 16 byte errors, one cell spanning
/// ≤ span bytes → ⌊16 / span⌋ cells (5 at the canonical 17-bit depth).
pub fn guaranteed_errors_per_block(bits_per_cell: u32) -> usize {
    (RS_NSYM / 2) / cell_byte_span(bits_per_cell)
}

/// Guaranteed-recoverable known-missing (erased) cells per block.
/// An erasure costs 1 syndrome symbol per byte: ⌊32 / span⌋ cells
/// (10 at the canonical 17-bit depth) — twice the error budget.
pub fn guaranteed_erasures_per_block(bits_per_cell: u32) -> usize {
    RS_NSYM / cell_byte_span(bits_per_cell)
}

// ── Deterministic PRNG (SplitMix64) for padding noise ───────────────────

pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        SplitMix64(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn fill_bytes(&mut self, out: &mut [u8]) {
        for chunk in out.chunks_mut(8) {
            let v = self.next_u64().to_le_bytes();
            chunk.copy_from_slice(&v[..chunk.len()]);
        }
    }
}

fn pad_seed(total_bytes: usize, payload_len: usize) -> u64 {
    // FNV-1a over the two framing numbers — stable across versions.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in (total_bytes as u64).to_le_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3_8365);
    }
    for b in (payload_len as u64).to_le_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3_8365);
    }
    h
}

// ── Capacity ────────────────────────────────────────────────────────────

/// Round a bit capacity down to whole RS blocks (min one block).
pub fn align_to_blocks(total_bits: usize) -> usize {
    let total_bytes = total_bits / 8;
    if total_bytes < RS_N {
        return total_bytes;
    }
    (total_bytes / RS_N) * RS_N
}

/// Payload capacity in bytes for a grid of `n_cells` cells.
pub fn payload_capacity(n_cells: usize) -> usize {
    let aligned = align_to_blocks(n_cells * BITS_PER_CELL as usize);
    (aligned / RS_N) * RS_K - 4
}

/// Human-readable capacity breakdown.
#[derive(Debug, Clone, Copy)]
pub struct Capacity {
    pub cells: usize,
    pub bits_per_cell: usize,
    pub raw_bits: usize,
    pub raw_bytes: usize,
    pub aligned_bytes: usize,
    pub rs_blocks: usize,
    pub payload_bytes: usize,
}

pub fn capacity_info(n_cells: usize, bits_per_cell: u32) -> Capacity {
    let bpc = bits_per_cell as usize;
    let raw_bits = n_cells * bpc;
    let raw_bytes = raw_bits / 8;
    let aligned_bytes = align_to_blocks(raw_bits);
    if aligned_bytes < RS_N {
        // One partial block: whatever bytes remain after parity + prefix.
        let payload = (aligned_bytes + RS_N).min(RS_N); // below: raw < RS_N
        let _ = payload;
        let data = aligned_bytes.saturating_sub(RS_NSYM + 4);
        return Capacity {
            cells: n_cells,
            bits_per_cell: bpc,
            raw_bits,
            raw_bytes,
            aligned_bytes,
            rs_blocks: usize::from(aligned_bytes > RS_NSYM + 4),
            payload_bytes: data,
        };
    }
    let rs_blocks = aligned_bytes / RS_N;
    Capacity {
        cells: n_cells,
        bits_per_cell: bpc,
        raw_bits,
        raw_bytes,
        aligned_bytes,
        rs_blocks,
        payload_bytes: rs_blocks * RS_K - 4,
    }
}

// ── Bit packing: bitstream ↔ 17-bit cell values ─────────────────────────

pub fn bits_to_cell_values(bits: &[u8], bpc: usize) -> Vec<u32> {
    let n_cells = bits.len().div_ceil(bpc);
    let mut values = Vec::with_capacity(n_cells);
    for i in 0..n_cells {
        let start = i * bpc;
        let mut val: u32 = 0;
        for b in 0..bpc {
            if start + b < bits.len() {
                val = (val << 1) | bits[start + b] as u32;
            }
        }
        values.push(val);
    }
    values
}

pub fn cell_values_to_bits(values: &[u32], total_bits: usize, bpc: usize) -> Vec<u8> {
    let mut bits = Vec::with_capacity(values.len() * bpc);
    for &val in values {
        for b in (0..bpc).rev() {
            bits.push(((val >> b) & 1) as u8);
        }
    }
    bits.truncate(total_bits);
    bits
}

// ── RS framing ──────────────────────────────────────────────────────────

/// Encode user data into a full bitstream of exactly `total_bits` bits.
pub fn encode_bits(user_data: &[u8], total_bits: usize) -> Vec<u8> {
    let total_bytes = align_to_blocks(total_bits);
    let enc = Encoder::new(RS_NSYM);
    let mut all_bytes = Vec::with_capacity(total_bytes);

    if total_bytes < RS_N {
        // Single partial block: everything after parity is payload+padding.
        let data_bytes = total_bytes.saturating_sub(RS_NSYM);
        let mut payload = Vec::with_capacity(4 + user_data.len());
        payload.extend_from_slice(&(user_data.len() as u32).to_be_bytes());
        payload.extend_from_slice(user_data);
        payload.truncate(data_bytes);
        let mut padded = vec![0u8; data_bytes];
        padded[..payload.len()].copy_from_slice(&payload);
        SplitMix64::new(pad_seed(total_bytes, payload.len()))
            .fill_bytes(&mut padded[payload.len()..]);
        let buf = enc.encode(&padded);
        all_bytes.extend_from_slice(buf.data());
        all_bytes.extend_from_slice(buf.ecc());
    } else {
        let n_blocks = total_bytes / RS_N;
        let data_bytes = n_blocks * RS_K;

        let mut payload = Vec::with_capacity(4 + user_data.len());
        payload.extend_from_slice(&(user_data.len() as u32).to_be_bytes());
        payload.extend_from_slice(user_data);
        payload.truncate(data_bytes);
        let mut padded = vec![0u8; data_bytes];
        padded[..payload.len()].copy_from_slice(&payload);
        SplitMix64::new(pad_seed(total_bytes, payload.len()))
            .fill_bytes(&mut padded[payload.len()..]);

        for block in padded.chunks(RS_K) {
            let buf = enc.encode(block);
            all_bytes.extend_from_slice(buf.data());
            all_bytes.extend_from_slice(buf.ecc());
        }

        if all_bytes.len() < total_bytes {
            let mut fill = vec![0u8; total_bytes - all_bytes.len()];
            SplitMix64::new(pad_seed(total_bytes, payload.len()).rotate_left(1))
                .fill_bytes(&mut fill);
            all_bytes.extend_from_slice(&fill);
        }
    }

    // → flat bits, MSB first, zero-padded to total_bits
    let mut bits = vec![0u8; total_bits];
    for (i, &b) in all_bytes.iter().enumerate() {
        for j in 0..8 {
            let idx = i * 8 + j;
            if idx < total_bits {
                bits[idx] = (b >> (7 - j)) & 1;
            }
        }
    }
    bits
}

/// Decode cell values where `None` marks a *known-missing* cell.
///
/// Missing cells become Reed–Solomon erasures: their byte positions are
/// known (17 consecutive bits spanning ≤ 3 bytes), so the decoder spends
/// the 32-symbol budget twice as efficiently — up to 10 dead cells per
/// 120-cell block are recovered, versus 5 arbitrary errors.
pub fn decode_with_erasures(values: &[Option<u32>], bpc: u32) -> Result<Vec<u8>, DecodeError> {
    let bpc = bpc as usize;
    let total_bits = values.len() * bpc;
    if total_bits < 8 {
        return Err(DecodeError::TooShort);
    }
    let mut total_bytes = total_bits / 8;
    let aligned = align_to_blocks(total_bits);
    total_bytes = total_bytes.min(aligned);

    let mut raw = vec![0u8; total_bytes];
    let cpb = cells_per_block(bpc as u32);
    let mut block_erasures: Vec<Vec<u8>> = vec![Vec::new(); total_bytes.div_ceil(RS_N)];
    for (i, &v) in values.iter().enumerate() {
        let bit0 = i * bpc;
        for b in 0..bpc {
            let idx = bit0 + b;
            if idx >= total_bits {
                break;
            }
            let byte = idx / 8;
            if byte >= total_bytes {
                break;
            }
            let bit = 7 - (idx % 8);
            let val: u32 = v.unwrap_or(0); // placeholder bits, erased below
            raw[byte] |= (((val >> (bpc - 1 - b)) & 1) as u8) << bit;
        }
        if v.is_none() && bit0 < total_bytes * 8 {
            let block = i / cpb;
            let bit_in_block = (i % cpb) * bpc;
            let first = bit_in_block / 8;
            let last = (bit_in_block + bpc - 1) / 8;
            for byte in first..=last {
                if byte < RS_N {
                    let pos = byte as u8;
                    if !block_erasures[block].contains(&pos) {
                        block_erasures[block].push(pos);
                    }
                }
            }
        }
    }

    let dec = Decoder::new(RS_NSYM);
    let mut decoded = Vec::with_capacity(total_bytes);
    for (bi, block) in raw.chunks(RS_N).enumerate() {
        if block.len() > RS_NSYM {
            let erase = block_erasures.get(bi).map(Vec::as_slice).unwrap_or(&[]);
            let res = if erase.is_empty() {
                dec.correct(block, None)
            } else {
                dec.correct(block, Some(erase))
            };
            match res {
                Ok(received) => decoded.extend_from_slice(received.data()),
                Err(_) => decoded.extend_from_slice(&block[..block.len() - RS_NSYM]), // degraded
            }
        } else {
            decoded.extend_from_slice(block);
        }
    }

    if decoded.len() < 4 {
        return Err(DecodeError::TooShort);
    }
    let data_len = u32::from_be_bytes([decoded[0], decoded[1], decoded[2], decoded[3]]) as usize;
    if data_len == 0 || data_len > decoded.len() - 4 {
        return Err(DecodeError::BadLength(data_len));
    }
    Ok(decoded[4..4 + data_len].to_vec())
}

/// Decode a bitstream back to user data (all cells trusted).
///
/// Blocks that fail RS correction are used as-is (degraded bytes pass
/// through), matching the reference decoder. A wrong or unsafe length
/// prefix is an error, not garbage.
pub fn decode_bits(bits: &[u8]) -> Result<Vec<u8>, DecodeError> {
    let total_bits = bits.len() - (bits.len() % 8);
    if total_bits < 8 {
        return Err(DecodeError::TooShort);
    }
    let mut total_bytes = total_bits / 8;
    let aligned = align_to_blocks(total_bits);
    total_bytes = total_bytes.min(aligned);

    let mut raw = vec![0u8; total_bytes];
    for i in 0..total_bytes {
        let mut byte = 0u8;
        for j in 0..8 {
            byte = (byte << 1) | bits[i * 8 + j];
        }
        raw[i] = byte;
    }

    let dec = Decoder::new(RS_NSYM);
    let mut decoded = Vec::with_capacity(total_bytes);
    for block in raw.chunks(RS_N) {
        if block.len() > RS_NSYM {
            match dec.correct(block, None) {
                Ok(received) => decoded.extend_from_slice(received.data()),
                Err(_) => decoded.extend_from_slice(&block[..block.len() - RS_NSYM]), // degraded
            }
        } else {
            decoded.extend_from_slice(block);
        }
    }

    if decoded.len() < 4 {
        return Err(DecodeError::TooShort);
    }
    let data_len = u32::from_be_bytes([decoded[0], decoded[1], decoded[2], decoded[3]]) as usize;
    if data_len == 0 || data_len > decoded.len() - 4 {
        return Err(DecodeError::BadLength(data_len));
    }
    Ok(decoded[4..4 + data_len].to_vec())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    TooShort,
    BadLength(usize),
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodeError::TooShort => write!(f, "bitstream too short to contain a length prefix"),
            DecodeError::BadLength(n) => {
                write!(f, "length prefix {n} is out of frame bounds")
            }
        }
    }
}

impl std::error::Error for DecodeError {}

// ── Convenience: bytes ↔ cell values ────────────────────────────────────

pub fn encode(data: &[u8], n_cells: usize, bits_per_cell: u32) -> Vec<u32> {
    bits_to_cell_values(
        &encode_bits(data, n_cells * bits_per_cell as usize),
        bits_per_cell as usize,
    )
}

pub fn decode(cell_values: &[u32], bits_per_cell: u32) -> Result<Vec<u8>, DecodeError> {
    let total_bits = cell_values.len() * bits_per_cell as usize;
    let bits = cell_values_to_bits(cell_values, total_bits, bits_per_cell as usize);
    decode_bits(&bits)
}

// ── Cell-level error injection (for tests and the demo) ─────────────────

/// Corrupt `n_errors` cells chosen by a deterministic PRNG.
pub fn corrupt_cells(values: &mut [u32], n_errors: usize, bits_per_cell: u32, seed: u64) {
    let mut rng = SplitMix64::new(seed);
    let mut flipped = 0;
    let mut guard = 0;
    while flipped < n_errors && guard < n_errors * 100 {
        guard += 1;
        let idx = (rng.next_u64() % values.len() as u64) as usize;
        let bits = 1 + (rng.next_u64() % bits_per_cell as u64) as u32;
        values[idx] ^= 1 << (bits - 1);
        flipped += 1;
    }
}

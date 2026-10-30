//! Codec tests: framing, RS error/erasure guarantees, capacity math.

use microform::codec::{
    self, capacity_info, decode, decode_bits, decode_with_erasures, encode, encode_bits,
    guaranteed_erasures_per_block, guaranteed_errors_per_block, DecodeError,
};
use microform::layout::BITS_PER_CELL;

const PAYLOAD: &[u8] = b"Microform Code codec test - RS(255,223), 17 bits per cell, \
paper is the storage medium and the printer is the write head.";

#[test]
fn cell_value_roundtrip() {
    let values = encode(PAYLOAD, 24 * 24, BITS_PER_CELL);
    assert_eq!(values.len(), 576);
    let back = decode(&values, BITS_PER_CELL).unwrap();
    assert_eq!(back, PAYLOAD);
}

#[test]
fn rs_corrects_guaranteed_unknown_position_errors() {
    // 5 worst-case cell errors per 120-cell block, several seeds.
    for seed in [1u64, 0xDEAD, 0xBEEF] {
        let mut values = encode(PAYLOAD, 24 * 24, BITS_PER_CELL);
        let blocks = values.len() / codec::CELLS_PER_BLOCK;
        assert_eq!(guaranteed_errors_per_block(BITS_PER_CELL), 5);
        for b in 0..blocks {
            for k in 0..5usize {
                let idx = b * codec::CELLS_PER_BLOCK + (seed as usize + k * 7) % 120;
                values[idx] ^= 1 << ((seed + k as u64) % 17);
            }
        }
        let decoded = decode(&values, BITS_PER_CELL).unwrap();
        assert_eq!(decoded, PAYLOAD, "seed {seed:#x}");
    }
}

#[test]
fn rs_recovers_guaranteed_erasures() {
    // 10 known-missing cells per block — twice the error budget.
    assert_eq!(guaranteed_erasures_per_block(BITS_PER_CELL), 10);
    let values = encode(PAYLOAD, 24 * 24, BITS_PER_CELL);
    let mut erased: Vec<Option<u32>> = values.iter().map(|&v| Some(v)).collect();
    let blocks = values.len() / codec::CELLS_PER_BLOCK;
    for b in 0..blocks {
        for k in 0..10usize {
            let idx = b * codec::CELLS_PER_BLOCK + (k * 11 + 3) % 120;
            erased[idx] = None;
        }
    }
    assert_eq!(
        decode_with_erasures(&erased, BITS_PER_CELL).unwrap(),
        PAYLOAD
    );
}

#[test]
fn beyond_guarantee_is_detected_not_guessed() {
    // Way beyond the correction budget: either an error or wrong data —
    // but never a silent Ok with a fabricated length prefix.
    let mut values = encode(PAYLOAD, 24 * 24, BITS_PER_CELL);
    for v in values.iter_mut() {
        *v ^= 0x1_FFFF;
    }
    if let Ok(d) = decode(&values, BITS_PER_CELL) {
        assert_ne!(d, PAYLOAD);
    }
}

#[test]
fn framing_rejects_impossible_length() {
    let bits = vec![1u8; 17 * 8];
    assert_eq!(
        decode_bits(&bits),
        Err(DecodeError::BadLength(u32::MAX as usize))
    );
    assert_eq!(decode_bits(&[1u8; 4]), Err(DecodeError::TooShort));
}

#[test]
fn capacity_math_matches_the_spec() {
    // 92×92 = 8464 cells × 17 bits = 17,986 bytes → 70 RS blocks.
    let cap = capacity_info(8464, BITS_PER_CELL);
    assert_eq!(cap.rs_blocks, 70);
    assert_eq!(cap.aligned_bytes, 70 * 255);
    assert_eq!(cap.payload_bytes, 70 * 223 - 4);
    assert_eq!(cap.payload_bytes, 15_606);

    // Below one full block: a single partial block carries what remains
    // after parity and the length prefix.
    let small = capacity_info(48, BITS_PER_CELL);
    assert_eq!(small.rs_blocks, 1);
    assert_eq!(small.aligned_bytes, 102);
    assert_eq!(small.payload_bytes, 102 - 32 - 4);
    // And a partial block really encodes + decodes (payload must fit!):
    let tiny = b"tiny payload";
    let v = encode(tiny, 48, BITS_PER_CELL);
    assert_eq!(decode(&v, BITS_PER_CELL).unwrap(), tiny);
}

#[test]
fn encode_is_deterministic() {
    let a = encode_bits(PAYLOAD, 576 * 17);
    let b = encode_bits(PAYLOAD, 576 * 17);
    assert_eq!(a, b, "padding PRNG must be deterministic");
}

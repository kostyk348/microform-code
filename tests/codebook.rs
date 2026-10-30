//! Codebook tests: determinism, counts, bijection.

use microform::codebook::{self, full_stats, get_for_scale, N_DISTINCT_EXPECTED, N_PATTERNS};

#[test]
fn full_space_stats_are_deterministic() {
    let stats = full_stats();
    assert_eq!(stats.param_combos, 262_144);
    // The Rust reference rendering (pixel-center convention) yields this
    // exact count; the build is a fixed loop, so the number is stable.
    assert_eq!(stats.n_distinct, N_DISTINCT_EXPECTED);
    assert_eq!(stats.n_used, N_PATTERNS);
}

#[test]
fn canonical_book_has_bijective_lookup() {
    let cb = codebook::get();
    assert_eq!(cb.len(), N_PATTERNS as usize);
    // Spot-check the bijection: tile(code) → code_for must be identity.
    for code in [0u32, 1, 2, 1000, 65_535, 131_071] {
        let tile = cb.tile(code);
        assert_eq!(cb.code_for(&tile), Some(code), "bijection broken at {code}");
    }
    // Unknown tile (solid gray) must not decode to anything.
    assert_eq!(cb.code_for(&[127u8; 196]), None);
}

#[test]
fn two_scale_books_agree_on_the_same_canonical_codes() {
    // Determinism: building the same scale twice must give the same
    // usable-code list and the same tile→index mapping.
    let a = get_for_scale(10);
    let b = get_for_scale(10);
    assert_eq!(a.codes, b.codes);
    assert_eq!(a.bits, b.bits);
    // And the canonical book at 14 px must use every one of its codes:
    let canonical = codebook::get();
    let s14 = get_for_scale(14);
    assert_eq!(s14.codes.len(), N_PATTERNS as usize);
    assert_eq!(s14.bits, 17);
    assert_eq!(s14.codes.len(), canonical.len());
}

#[test]
fn smaller_scales_use_fewer_bits_but_stay_rs_aligned() {
    // At 7 px patterns merge, so the usable set shrinks and the cell bit
    // depth steps down; it must always divide one RS block (2040 bits).
    let s7 = get_for_scale(7);
    assert!(
        s7.codes.len() < 131_072,
        "7px book must lose codes to merging"
    );
    assert!(
        (2040 % s7.bits) == 0,
        "bit depth {} must divide 2040",
        s7.bits
    );
    assert!((s7.bits as f64) <= (s7.codes.len() as f64).log2());
}

//! Microform Code — the perceptual audit.
//!
//! The codebook guarantees that usable tiles are *byte-distinct*; this
//! module measures how *visually* close two usable tiles can get at a
//! given print scale. The decoder matches tiles exactly, so a scan can
//! only turn one tile into another if the pair differs by a handful of
//! gray levels on a few pixels — such "fragile pairs" are what the
//! audit counts. Every other scan deviation lands on no tile at all
//! and becomes an erasure, which RS(255,223) recovers by design.
//!
//! Two complementary passes:
//!
//! * **global net** — usable tiles are bucketed by a mean-threshold
//!   hash over an area-resampled grid; every bucket is compared
//!   pairwise (mean absolute difference and L∞), catching any pair
//!   that is close in coarse ink layout, wherever it sits in the
//!   parameter space;
//! * **neighbour pass** (canonical scale) — every usable code is
//!   compared to its one-step parameter neighbours; rendering is smooth
//!   in every parameter, so those are the visually closest pairs by
//!   construction, and per-parameter margins quantify each step.

use std::collections::{HashMap, HashSet};

use crate::codebook::{render_tile_at, Codebook, Params, ScaleCodebook};
use crate::layout as L;

/// Result of auditing one scale.
#[derive(Debug, Clone)]
pub struct ScaleAudit {
    pub cell_px: u32,
    pub usable: usize,
    pub bits: u32,
    /// Smallest mean-absolute-difference between two usable tiles,
    /// on the 0..255 gray scale (`INFINITY` when nothing was compared).
    pub min_mae: f32,
    /// Canonical codes of the closest pair found by the global net.
    pub closest: Option<(u32, u32)>,
    /// Number of audited pairs with L∞ ≤ 2 gray levels (flip-risk).
    pub linf_le_2: usize,
    /// Number of audited pairs with L∞ ≤ 4 gray levels.
    pub linf_le_4: usize,
    /// Number of audited pairs with L∞ ≤ 8 gray levels.
    pub linf_le_8: usize,
}

impl ScaleAudit {
    /// `true` when no two usable tiles are within scanner-quantization
    /// reach of each other at this scale.
    pub fn pass(&self) -> bool {
        self.linf_le_2 == 0
    }
}

/// Minimum one-step MAE per parameter, in the order
/// `[angle, scale, stretch, dot dx, dot dy, dot radius]`.
/// `INFINITY` marks a parameter with no usable neighbour pair at all.
pub type ParamMargins = [f32; 6];

pub const PARAM_NAMES: [&str; 6] = [
    "angle",
    "scale",
    "stretch",
    "dot dx",
    "dot dy",
    "dot radius",
];

/// Audit one scale codebook with the global hash-bucket net.
pub fn audit_scale(book: &ScaleCodebook) -> ScaleAudit {
    let s = book.cell_px;
    let mut audit = ScaleAudit {
        cell_px: s,
        usable: book.codes.len(),
        bits: book.bits,
        min_mae: f32::INFINITY,
        closest: None,
        linf_le_2: 0,
        linf_le_4: 0,
        linf_le_8: 0,
    };

    // Bucket usable tiles by their coarse appearance.
    let mut buckets: HashMap<u64, Vec<(u32, &[u8])>> = HashMap::new();
    for (code, tile) in book.iter_usable() {
        buckets
            .entry(tile_hash(tile, s))
            .or_default()
            .push((code, tile));
    }

    let mut seen: HashSet<(u32, u32)> = HashSet::new();
    for group in buckets.values() {
        if group.len() < 2 {
            continue;
        }
        for i in 0..group.len() {
            for j in (i + 1)..group.len() {
                consider(&mut audit, group[i], group[j], &mut seen);
            }
        }
    }
    audit
}

/// Per-parameter one-step margins at the canonical 14 px scale.
///
/// For every usable code each of the six parameters is stepped once in
/// each direction; if the neighbour tile is itself usable, the pair
/// contributes to that parameter's minimum MAE.
pub fn neighbour_margins(canonical: &Codebook) -> ParamMargins {
    let s = L::CELL_PX;
    let book = crate::codebook::get_for_scale(s);

    // Membership test for "usable at this scale": FNV prefilter, exact
    // lookup afterwards. A hash collision only costs an extra lookup.
    let usable_hashes: HashSet<u64> = book.iter_usable().map(|(_, t)| fnv64(t)).collect();

    let mut margins = [f32::INFINITY; 6];
    for (code, tile_c) in book.iter_usable() {
        let p = canonical.render_params(code);
        for (d, margin) in margins.iter_mut().enumerate() {
            for delta in [-1i32, 1] {
                let Some(q) = step_params(p, d, delta) else {
                    continue;
                };
                let tq = render_tile_at(q, s);
                if !usable_hashes.contains(&fnv64(&tq)) {
                    continue;
                }
                // The neighbour combo may render to the *same* tile (that
                // combo was deduplicated away at the canonical build, so
                // the code stays usable). Such a self-pair is not a decode
                // risk: the decoder can only ever produce usable codes.
                let Some(idx) = book.index_for(&tq) else {
                    continue;
                };
                if book.codes[idx as usize] == code {
                    continue;
                }
                let (mae, _) = tile_diff(tile_c, &tq);
                if mae < *margin {
                    *margin = mae;
                }
            }
        }
    }
    margins
}

/// Human-readable description of how two parameter tuples differ.
pub fn describe_delta(canonical: &Codebook, a: u32, b: u32) -> String {
    let pa = canonical.render_params(a);
    let pb = canonical.render_params(b);
    let (va, vb) = (params_arr(pa), params_arr(pb));
    let parts: Vec<String> = (0..6)
        .filter(|&i| va[i] != vb[i])
        .map(|i| format!("{} {:+}", PARAM_NAMES[i], vb[i] as i64 - va[i] as i64))
        .collect();
    if parts.is_empty() {
        "identical params".to_string()
    } else {
        parts.join(", ")
    }
}

// ── internals ───────────────────────────────────────────────────────────

type TileRef<'a> = (u32, &'a [u8]);

/// Record one pair: global statistics once per unordered pair (the
/// `seen` set deduplicates across buckets and passes).
fn consider(audit: &mut ScaleAudit, a: TileRef, b: TileRef, seen: &mut HashSet<(u32, u32)>) {
    let pair = (a.0.min(b.0), a.0.max(b.0));
    let (mae, linf) = tile_diff(a.1, b.1);

    if seen.insert(pair) {
        if mae < audit.min_mae || (mae == audit.min_mae && audit.closest.is_some_and(|c| pair < c))
        {
            audit.min_mae = mae;
            audit.closest = Some(pair);
        }
        if linf <= 2 {
            audit.linf_le_2 += 1;
        }
        if linf <= 4 {
            audit.linf_le_4 += 1;
        }
        if linf <= 8 {
            audit.linf_le_8 += 1;
        }
    }
}

/// `(mean absolute difference, L∞)` of two equal-length tiles.
fn tile_diff(a: &[u8], b: &[u8]) -> (f32, u32) {
    let mut sum = 0u64;
    let mut mx = 0u32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (i32::from(*x) - i32::from(*y)).unsigned_abs();
        sum += d as u64;
        mx = mx.max(d);
    }
    (sum as f32 / a.len() as f32, mx)
}

/// Mean-threshold hash over a `g×g` area-resampled grid (`g ≤ 8`,
/// so the hash fits a `u64`). Bit set = block darker than mid-gray.
fn tile_hash(tile: &[u8], s: u32) -> u64 {
    let g = s.min(8);
    let ranges: Vec<(u32, u32)> = (0..g).map(|i| (i * s / g, (i + 1) * s / g)).collect();
    let mut bits = 0u64;
    for (gy, &(y0, y1)) in ranges.iter().enumerate() {
        for (gx, &(x0, x1)) in ranges.iter().enumerate() {
            let mut sum = 0u32;
            for y in y0..y1 {
                for x in x0..x1 {
                    sum += u32::from(tile[(y * s + x) as usize]);
                }
            }
            let area = ((y1 - y0) * (x1 - x0)) as f32;
            if sum as f32 / area < 128.0 {
                bits |= 1u64 << (gy * g as usize + gx);
            }
        }
    }
    bits
}

fn fnv64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn params_arr(p: Params) -> [u32; 6] {
    [p.0, p.1, p.2, p.3, p.4, p.5]
}

fn step_params(p: Params, dim: usize, delta: i32) -> Option<Params> {
    const LIMITS: [u32; 6] = [
        L::ANGLE_VALUES,
        L::SCALE_VALUES,
        L::ELONG_VALUES,
        L::DOT_DX_VALUES,
        L::DOT_DY_VALUES,
        L::DOT_RAD_VALUES,
    ];
    let mut v = params_arr(p);
    let nv = v[dim] as i64 + i64::from(delta);
    if nv < 0 || nv >= i64::from(LIMITS[dim]) {
        return None;
    }
    v[dim] = nv as u32;
    Some((v[0], v[1], v[2], v[3], v[4], v[5]))
}

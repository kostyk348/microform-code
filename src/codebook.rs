//! Microform Code — the codebook.
//!
//! Of the 262,144 possible parameter combinations only a subset renders
//! to *distinct* pixel patterns at the canonical 14×14 tile (the rest
//! collapse: e.g. a round ellipse looks the same under all 64 angles).
//! The 17-bit codec uses the first 131,072 = 2¹⁷ distinct patterns in
//! canonical order, giving a strict bijection:
//!
//! ```text
//!     17-bit code  →  unique pixel tile  →  (lookup)  17-bit code
//! ```
//!
//! The build is fully deterministic: patterns are enumerated in a fixed
//! nested-loop order (`av, sv, ev, dx, dy, r`) and each distinct tile
//! keeps its first-seen — which is also its smallest — parameter tuple
//! as the canonical representative. The same version of this module
//! always produces the same codebook. There is no cache file: a release
//! build reconstructs the whole codebook in well under a second.

use std::collections::HashMap;

use crate::layout as L;
use crate::patterns::pattern_pixels;

/// Tile size in bytes: `CELL_PX × CELL_PX`.
pub const TILE_BYTES: usize = (L::CELL_PX * L::CELL_PX) as usize;

/// Codes actually used: 2¹⁷ = 131,072.
pub const N_PATTERNS: u32 = 1 << 17;

/// Distinct patterns over the full parameter space (Rust reference
/// rendering, deterministic build loop — asserted by tests).
pub const N_DISTINCT_EXPECTED: u32 = 184_154;

/// Rendering parameters of one cell pattern.
pub type Params = (u32, u32, u32, u32, u32, u32); // av, sv, ev, dx, dy, r

/// Bijection between 17-bit codes and canonical pixel tiles.
pub struct Codebook {
    forward: Vec<Params>,
    reverse: HashMap<[u8; TILE_BYTES], u32>,
}

impl Codebook {
    /// 17-bit code → rendering parameters.
    pub fn render_params(&self, code: u32) -> Params {
        let code = if (code as usize) < self.forward.len() {
            code
        } else {
            0
        };
        self.forward[code as usize]
    }

    /// Canonical tile pixels → 17-bit code, or `None` if unknown.
    pub fn code_for(&self, tile: &[u8]) -> Option<u32> {
        if tile.len() != TILE_BYTES {
            return None;
        }
        let mut key = [0u8; TILE_BYTES];
        key.copy_from_slice(tile);
        self.reverse.get(&key).copied()
    }

    /// 17-bit code → canonical pixel tile.
    pub fn tile(&self, code: u32) -> [u8; TILE_BYTES] {
        let (av, sv, ev, dx, dy, r) = self.render_params(code);
        let t = pattern_pixels(av, sv, ev, dx, dy, r);
        let mut out = [0u8; TILE_BYTES];
        out.copy_from_slice(&t);
        out
    }

    /// Number of codes (always 2¹⁷).
    pub fn len(&self) -> usize {
        self.forward.len()
    }

    pub fn is_empty(&self) -> bool {
        self.forward.is_empty()
    }

    /// Stable fingerprint of the mapping — for determinism tests.
    /// Two codebooks built in the same process must have equal fingerprints.
    pub fn fingerprint(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &(av, sv, ev, dx, dy, r) in &self.forward {
            for byte in [av as u8, sv as u8, ev as u8, dx as u8, dy as u8, r as u8] {
                h ^= byte as u64;
                h = h.wrapping_mul(0x0100_0000_01b3_8365);
            }
        }
        h
    }

    /// Build the 17-bit codebook (internal single-pass constructor).
    fn build() -> Self {
        let (mut forward, tiles) = build_full();

        let mut reverse: HashMap<[u8; TILE_BYTES], u32> =
            HashMap::with_capacity(N_PATTERNS as usize);
        for (code, key) in tiles.iter().enumerate().take(N_PATTERNS as usize) {
            reverse.insert(*key, code as u32);
        }

        forward.truncate(N_PATTERNS as usize);
        Codebook { forward, reverse }
    }
}

/// Enumerate the whole parameter space in canonical order and collect
/// every *distinct* tile once: `(canonical params, tile)` per pattern.
/// First-seen order == smallest-value order, so index = canonical code.
fn build_full() -> (Vec<Params>, Vec<[u8; TILE_BYTES]>) {
    let mut forward: Vec<Params> = Vec::with_capacity(N_DISTINCT_EXPECTED as usize);
    let mut tiles: Vec<[u8; TILE_BYTES]> = Vec::with_capacity(N_DISTINCT_EXPECTED as usize);
    let mut first_seen: std::collections::HashSet<[u8; TILE_BYTES]> =
        std::collections::HashSet::with_capacity(N_DISTINCT_EXPECTED as usize);

    for av in 0..L::ANGLE_VALUES {
        for sv in 0..L::SCALE_VALUES {
            for ev in 0..L::ELONG_VALUES {
                for dx in 0..L::DOT_DX_VALUES {
                    for dy in 0..L::DOT_DY_VALUES {
                        for r in 0..L::DOT_RAD_VALUES {
                            let tile = pattern_pixels(av, sv, ev, dx, dy, r);
                            let mut key = [0u8; TILE_BYTES];
                            key.copy_from_slice(&tile);
                            if first_seen.insert(key) {
                                // value increments in loop order, so the
                                // first-seen value is the smallest one:
                                forward.push((av, sv, ev, dx, dy, r));
                                tiles.push(key);
                            }
                        }
                    }
                }
            }
        }
    }
    (forward, tiles)
}

use std::sync::OnceLock;

static CODEBOOK: OnceLock<Codebook> = OnceLock::new();

/// Process-wide codebook singleton (built once, reused everywhere).
pub fn get() -> &'static Codebook {
    CODEBOOK.get_or_init(Codebook::build)
}

/// Statistics about the full parameter space (from a full build).
#[derive(Debug, Clone, Copy)]
pub struct CodebookStats {
    pub param_combos: u32,
    pub n_distinct: u32,
    pub n_used: u32,
    pub bits_per_cell_effective: f64,
}

/// Build the *full* codebook (all distinct patterns) and return counts.
/// Used by tests and `microform codebook` to report honest numbers.
pub fn full_stats() -> CodebookStats {
    let (forward, _) = build_full();
    CodebookStats {
        param_combos: L::PARAM_COMBOS,
        n_distinct: forward.len() as u32,
        n_used: N_PATTERNS,
        bits_per_cell_effective: (forward.len() as f64).log2(),
    }
}

// ── Per-scale codebooks (scale-independent decoding) ────────────────────

/// Bit depths that divide one RS block exactly (2040 bits = 255 bytes),
/// keeping every cell inside a single block for clean error isolation.
pub const VALID_BIT_DEPTHS: [u32; 5] = [17, 15, 12, 10, 8];

/// A codebook adapted to an integer cell size `cell_px`.
///
/// The canonical 14 px book guarantees 131,072 *distinct* tiles, but at
/// other scales patterns merge. For each scale this book keeps only the
/// codes whose tile stays unique among all rendered tiles at that scale
/// ("usable" codes) and assigns indices over them. The cell bit depth is
/// the smallest RS-aligned depth ≥ log2 of the usable count, so encoder
/// and decoder derive the identical table from the detected pitch alone.
pub struct ScaleCodebook {
    pub cell_px: u32,
    /// Usable canonical codes, in canonical order.
    pub codes: Vec<u32>,
    /// Cell bit depth used with this book (divides 2040).
    pub bits: u32,
    tile_bytes: usize,
    reverse: HashMap<Vec<u8>, u32>,
}

impl ScaleCodebook {
    pub fn new(cell_px: u32) -> Self {
        assert!(
            (4..=32).contains(&cell_px),
            "cell_px out of supported 4..=32 range"
        );
        let canonical = get();
        let tb = (cell_px * cell_px) as usize;

        // Render every canonical code at this scale and count collisions.
        let mut counts: HashMap<Vec<u8>, u32> = HashMap::with_capacity(N_PATTERNS as usize);
        let mut tiles: Vec<Vec<u8>> = Vec::with_capacity(N_PATTERNS as usize);
        for code in 0..N_PATTERNS {
            let tile = render_tile_at(canonical.render_params(code), cell_px);
            counts
                .entry(tile.clone())
                .and_modify(|c| *c += 1)
                .or_insert(1);
            tiles.push(tile);
        }

        let mut codes = Vec::new();
        let mut reverse = HashMap::with_capacity(N_PATTERNS as usize);
        for (code, tile) in tiles.into_iter().enumerate() {
            if counts[&tile] == 1 {
                reverse.insert(tile, codes.len() as u32);
                codes.push(code as u32);
            }
        }

        // Cell bit depth: the largest RS-aligned depth whose value range
        // fits the usable set (2^bits ≤ len) — encoder indices stay in range.
        let raw_bits = (codes.len() as f64).log2().floor() as u32;
        let bits = VALID_BIT_DEPTHS
            .iter()
            .copied()
            .find(|&d| d <= raw_bits)
            .unwrap_or(8);

        ScaleCodebook {
            cell_px,
            codes,
            bits,
            tile_bytes: tb,
            reverse,
        }
    }

    pub fn tile_bytes(&self) -> usize {
        self.tile_bytes
    }

    /// Tile pixels → index into [`Self::codes`], or `None` if unknown.
    pub fn index_for(&self, tile: &[u8]) -> Option<u32> {
        if tile.len() != self.tile_bytes {
            return None;
        }
        self.reverse.get(tile).copied()
    }

    /// Canonical code for a cell index.
    pub fn code(&self, index: u32) -> u32 {
        self.codes[index as usize]
    }
}

fn render_tile_at(params: Params, cell_px: u32) -> Vec<u8> {
    let (av, sv, ev, dx, dy, r) = params;
    let em = crate::patterns::ellipse_mask(
        cell_px,
        L::angle_v_to_deg(av),
        L::scale_v_to_frac(sv),
        L::elong_v_to_ratio(ev),
    );
    let dm = crate::patterns::dot_mask(
        cell_px,
        L::dot_dx_to_px(dx, cell_px as f64),
        L::dot_dy_to_px(dy, cell_px as f64),
        L::dot_rad_to_px(r),
    );
    em.iter()
        .zip(dm.iter())
        .map(|(e, d)| 255 - e.max(d))
        .collect()
}

use std::sync::{Arc, Mutex};

static SCALE_BOOKS: Mutex<Option<HashMap<u32, Arc<ScaleCodebook>>>> = Mutex::new(None);

/// Process-wide cache of scale codebooks.
pub fn get_for_scale(cell_px: u32) -> Arc<ScaleCodebook> {
    let mut guard = SCALE_BOOKS.lock().unwrap();
    let map = guard.get_or_insert_with(HashMap::new);
    map.entry(cell_px)
        .or_insert_with(|| Arc::new(ScaleCodebook::new(cell_px)))
        .clone()
}

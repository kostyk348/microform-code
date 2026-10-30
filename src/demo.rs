//! Microform Code — self-demonstration: honest checks on loopback.
//!
//! Every check is deterministic (seeded PRNGs), so the same numbers
//! appear in CI, in the README and on your machine. The checks also
//! draw the line between what v1 does and does not survive: pristine
//! images decode at any print scale, *known-missing* cells (a stamp, a
//! smudge) are recovered as Reed–Solomon erasures — but random noise
//! and camera blur need perceptual matching (roadmap), and the demo
//! deliberately does not claim otherwise.

use crate::codec::{self, SplitMix64};
use crate::decode::decode_image;
use crate::imgproc::add_noise;
use crate::layout as L;
use crate::pngio::{write_gray, GrayImage};
use crate::render::{generate, RenderSpec};

pub const PAYLOAD: &[u8] = b"Microform Code v1.0.0: 17 bits/cell, codebook 131072, RS(255,223). \
Paper is the storage medium; the printer is the write head.";

#[derive(Debug)]
pub struct Check {
    pub name: String,
    pub passed: bool,
    pub detail: String,
}

fn spec(grid: u32, cell_px: u32) -> RenderSpec {
    RenderSpec {
        grid_w: grid,
        grid_h: grid,
        cell_px,
    }
}

/// Paint the given data cells over with paper color (physical damage).
fn damage_cells(img: &mut GrayImage, _grid: u32, cell_px: u32, cells: &[(u32, u32)]) {
    let cs = cell_px as usize;
    for &(row, col) in cells {
        let x0 = (L::OFF + col) as usize * cs;
        let y0 = (L::OFF + row) as usize * cs;
        for y in y0..y0 + cs {
            for x in x0..x0 + cs {
                img.set(x, y, 255);
            }
        }
    }
}

/// Run all demo checks; `out_png` optionally saves a 92×92 artifact.
pub fn run(out_png: Option<&std::path::Path>) -> Vec<Check> {
    let mut checks = Vec::new();
    let grid = 24u32;
    let n_cells = (grid * grid) as usize;
    let per_block = codec::CELLS_PER_BLOCK; // 120
    let blocks = n_cells / per_block; // 4 full blocks (576 cells)

    // 1. Codec roundtrip on cell values (canonical 17-bit depth).
    let values = codec::encode(PAYLOAD, n_cells, crate::layout::BITS_PER_CELL);
    let ok = codec::decode(&values, crate::layout::BITS_PER_CELL)
        .map(|d| d == PAYLOAD)
        .unwrap_or(false);
    checks.push(Check {
        name: "codec roundtrip (24×24 cells)".into(),
        passed: ok,
        detail: format!("{} bytes", PAYLOAD.len()),
    });

    // 2. RS correction of unknown-position cell errors (guaranteed limit).
    let mut damaged = values.clone();
    let mut rng = SplitMix64::new(0xDECAFBAD);
    for b in 0..blocks {
        let lo = b * per_block;
        for _ in 0..codec::guaranteed_errors_per_block(crate::layout::BITS_PER_CELL) {
            let idx = lo + (rng.next_u64() % per_block as u64) as usize;
            damaged[idx] ^= 1 << (rng.next_u64() % 17);
        }
    }
    let ok = codec::decode(&damaged, crate::layout::BITS_PER_CELL)
        .map(|d| d == PAYLOAD)
        .unwrap_or(false);
    checks.push(Check {
        name: format!(
            "RS errors: {} cells per {}-cell block, positions unknown",
            codec::guaranteed_errors_per_block(crate::layout::BITS_PER_CELL),
            per_block
        ),
        passed: ok,
        detail: format!("{} blocks damaged", blocks),
    });

    // 3. RS erasures: known-missing cells, twice the budget.
    let mut erased: Vec<Option<u32>> = values.iter().map(|&v| Some(v)).collect();
    let mut per_block_erased = 0usize;
    for b in 0..blocks {
        for _ in 0..codec::guaranteed_erasures_per_block(crate::layout::BITS_PER_CELL) {
            let idx = b * per_block + (rng.next_u64() % per_block as u64) as usize;
            if erased[idx].is_some() {
                erased[idx] = None;
                per_block_erased += 1;
            }
        }
    }
    let ok = codec::decode_with_erasures(&erased, crate::layout::BITS_PER_CELL)
        .map(|d| d == PAYLOAD)
        .unwrap_or(false);
    checks.push(Check {
        name: format!(
            "RS erasures: {} cells per block, positions known",
            codec::guaranteed_erasures_per_block(crate::layout::BITS_PER_CELL)
        ),
        passed: ok,
        detail: format!("{} cells erased", per_block_erased),
    });

    // 4-6. End-to-end decode at three print scales, grid auto-detected.
    for cell_px in [L::CELL_PX, 7, 28] {
        let img = generate(spec(grid, cell_px), PAYLOAD);
        let name = format!("render → decode, {} px/cell (auto grid)", cell_px);
        checks.push(match decode_image(&img, None) {
            Ok(d) if d.data == PAYLOAD => Check {
                name,
                passed: true,
                detail: format!(
                    "grid {}×{} detected, hit-rate {:.1}%",
                    d.info.grid_w,
                    d.info.grid_h,
                    d.info.hit_rate * 100.0
                ),
            },
            Ok(d) => Check {
                name,
                passed: false,
                detail: format!("decoded {} bytes ≠ payload", d.data.len()),
            },
            Err(e) => Check {
                name,
                passed: false,
                detail: e.to_string(),
            },
        });
    }

    // 7. Physical damage: 8 cells painted over per block (stain/stamp),
    //    recovered as erasures because their positions are known-missing.
    let mut img = generate(spec(grid, L::CELL_PX), PAYLOAD);
    let mut damaged_cells = Vec::new();
    for b in 0..blocks {
        for _ in 0..8 {
            let idx = b * per_block + (rng.next_u64() % per_block as u64) as usize;
            damaged_cells.push(((idx / grid as usize) as u32, (idx % grid as usize) as u32));
        }
    }
    damage_cells(&mut img, grid, L::CELL_PX, &damaged_cells);
    checks.push(match decode_image(&img, None) {
        Ok(d) if d.data == PAYLOAD => Check {
            name: "physical damage: 8 cells per block painted over".into(),
            passed: true,
            detail: format!(
                "{} cells destroyed, {} read as erasures",
                damaged_cells.len(),
                d.info.misses
            ),
        },
        Ok(d) => Check {
            name: "physical damage: 8 cells per block painted over".into(),
            passed: false,
            detail: format!("decoded {} bytes ≠ payload", d.data.len()),
        },
        Err(e) => Check {
            name: "physical damage: 8 cells per block painted over".into(),
            passed: false,
            detail: e.to_string(),
        },
    });

    // 8. Honest boundary: uncorrectable random noise (needs perceptual
    //    matching — roadmap). We only verify the decoder *reports* it
    //    instead of returning garbage.
    let noisy = add_noise(&generate(spec(grid, L::CELL_PX), PAYLOAD), 30.0, 7);
    let honest = match decode_image(&noisy, None) {
        Ok(d) => d.data == PAYLOAD, // recovering would be fine too
        Err(_) => true,             // honest failure is the expected outcome
    };
    checks.push(Check {
        name: "honest boundary: heavy noise is reported, not guessed".into(),
        passed: honest,
        detail: "σ=30 noise → decode fails cleanly (perceptual matching = roadmap)".into(),
    });

    // Artifact for the docs.
    if let Some(path) = out_png {
        let img = generate(spec(92, L::CELL_PX), PAYLOAD);
        let _ = write_gray(path, &img);
    }

    checks
}

/// Capacity table for the three recommended print densities.
/// `(label, grid, size mm, raw bytes, payload bytes)`
pub fn capacity_rows() -> Vec<(String, u32, f64, usize, usize)> {
    let presets = [
        ("office paper 1200 dpi", 300u32, 92u32),
        ("office paper, max", 200, 150),
        ("photo paper", 120, 250),
    ];
    presets
        .iter()
        .map(|&(label, um, grid)| {
            let bits = crate::codebook::get_for_scale(grid_cell_px(um)).bits;
            let cap = codec::capacity_info((grid * grid) as usize, bits);
            let mm = (grid as f64 + 8.0) * um as f64 / 1000.0;
            (
                label.to_string(),
                grid,
                mm,
                cap.aligned_bytes,
                cap.payload_bytes,
            )
        })
        .collect()
}

/// Integer px/cell for a print density at 1200 dpi rasterization.
fn grid_cell_px(um: u32) -> u32 {
    ((um as f64 * 1200.0 / 25400.0).round() as u32).max(4)
}

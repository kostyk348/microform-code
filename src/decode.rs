//! Microform Code — the decoder: image → bytes.
//!
//! Scale-independent: the cell pitch is measured from the markers, every
//! cell quad is resampled to the canonical 14×14 tile and looked up in
//! the codebook. The grid size is estimated from marker geometry and
//! validated by the codebook hit-rate (candidates around the estimate
//! are tried; an explicit grid can be forced).

use crate::codebook;
use crate::codec;
use crate::homography::{apply, from_points};
use crate::layout as L;
use crate::locate::{cellspace_marker_points, locate, marker_points};
use crate::pngio::GrayImage;

#[derive(Debug, Clone)]
pub struct DecodeInfo {
    pub grid_w: u32,
    pub grid_h: u32,
    pub pitch_px: f64,
    /// Integer cell size the decoder matched against (px/cell).
    pub cell_px: u32,
    /// Cell bit depth of the matched scale codebook.
    pub bits_per_cell: u32,
    pub cells: usize,
    pub misses: usize,
    pub hit_rate: f64,
}

#[derive(Debug)]
pub enum DecodeFailure {
    NoMarkers,
    PoorMatch {
        best_hit_rate: f64,
        grid: (u32, u32),
    },
    HomographyFailed,
    Codec(codec::DecodeError),
}

impl std::fmt::Display for DecodeFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodeFailure::NoMarkers => write!(f, "corner markers not found"),
            DecodeFailure::HomographyFailed => write!(f, "homography failed"),
            DecodeFailure::PoorMatch {
                best_hit_rate,
                grid,
            } => write!(
                f,
                "cells do not match the codebook (best hit-rate {:.1}% at grid {}×{}; \
                 try --grid if the size is known)",
                best_hit_rate * 100.0,
                grid.0,
                grid.1
            ),
            DecodeFailure::Codec(e) => write!(f, "codec: {e}"),
        }
    }
}

impl std::error::Error for DecodeFailure {}

impl From<codec::DecodeError> for DecodeFailure {
    fn from(e: codec::DecodeError) -> Self {
        DecodeFailure::Codec(e)
    }
}

pub struct Decoded {
    pub data: Vec<u8>,
    pub info: DecodeInfo,
}

/// Decode a grayscale image of a Microform Code.
///
/// `grid`: forced `Some((w, h))` skips estimation (still validated by
/// hit-rate reporting).
pub fn decode_image(img: &GrayImage, grid: Option<(u32, u32)>) -> Result<Decoded, DecodeFailure> {
    let loc = locate(img).ok_or(DecodeFailure::NoMarkers)?;
    let markers = marker_points(&loc);

    let candidates: Vec<(u32, u32)> = match grid {
        Some((w, h)) => vec![(w, h)],
        None => {
            let mut v = Vec::new();
            for dj in -1i32..=1 {
                for di in -1i32..=1 {
                    let w = (loc.grid_w as i32 + di).max(4) as u32;
                    let h = (loc.grid_h as i32 + dj).max(4) as u32;
                    v.push((w, h));
                }
            }
            v
        }
    };

    // Integer render scale closest to the measured pitch.
    let cell = loc.pitch_px.round().clamp(4.0, 32.0) as u32;
    let book = codebook::get_for_scale(cell);
    let n = book.tile_bytes();
    type Best = Option<(usize, u32, u32, Vec<Option<u32>>, f64)>; // hits, w, h, values, rate
    let mut best: Best = None;

    for (gw, gh) in candidates {
        let cs_pts = cellspace_marker_points(gw, gh);
        let Some(h) = from_points(&cs_pts, &markers) else {
            return Err(DecodeFailure::HomographyFailed);
        };

        let n_cells = (gw * gh) as usize;
        let mut values: Vec<Option<u32>> = Vec::with_capacity(n_cells);
        let mut misses = 0usize;

        for row in 0..gh as i64 {
            for col in 0..gw as i64 {
                let x0 = (L::OFF as i64 + col) as f64;
                let y0 = (L::OFF as i64 + row) as f64;
                let c00 = apply(&h, x0, y0);
                let c10 = apply(&h, x0 + 1.0, y0);
                let c01 = apply(&h, x0, y0 + 1.0);
                let c11 = apply(&h, x0 + 1.0, y0 + 1.0);

                let mut tile = vec![0u8; n];
                // Sample at integer positions: pixel centers sit on the
                // integer grid of the continuous image plane.
                for ty in 0..cell as usize {
                    let v = ty as f64 / cell as f64;
                    for tx in 0..cell as usize {
                        let u = tx as f64 / cell as f64;
                        let (px, py) = bilerp(c00, c10, c01, c11, u, v);
                        tile[ty * cell as usize + tx] = sample_bilinear(img, px, py);
                    }
                }
                match book.index_for(&tile) {
                    Some(index) => values.push(Some(index)),
                    None => {
                        values.push(None);
                        misses += 1;
                    }
                }
            }
        }

        let hits = n_cells - misses;
        let rate = hits as f64 / n_cells as f64;
        if best.as_ref().is_none_or(|b| rate > b.4) {
            best = Some((hits, gw, gh, values, rate));
        }
        if misses == 0 {
            break; // cannot do better
        }
    }

    let (hits, gw, gh, values, rate) = best.ok_or(DecodeFailure::HomographyFailed)?;
    let misses = values.len() - hits;
    let info = DecodeInfo {
        grid_w: gw,
        grid_h: gh,
        pitch_px: loc.pitch_px,
        cell_px: cell,
        bits_per_cell: book.bits,
        cells: values.len(),
        misses,
        hit_rate: rate,
    };

    if rate < 0.5 {
        return Err(DecodeFailure::PoorMatch {
            best_hit_rate: rate,
            grid: (gw, gh),
        });
    }

    let data = codec::decode_with_erasures(&values, book.bits)?;
    Ok(Decoded { data, info })
}

#[allow(clippy::needless_range_loop)]
fn bilerp(
    c00: (f64, f64),
    c10: (f64, f64),
    c01: (f64, f64),
    c11: (f64, f64),
    u: f64,
    v: f64,
) -> (f64, f64) {
    let (x00, y00) = c00;
    let (x10, y10) = c10;
    let (x01, y01) = c01;
    let (x11, y11) = c11;
    let x = (1.0 - u) * (1.0 - v) * x00 + u * (1.0 - v) * x10 + (1.0 - u) * v * x01 + u * v * x11;
    let y = (1.0 - u) * (1.0 - v) * y00 + u * (1.0 - v) * y10 + (1.0 - u) * v * y01 + u * v * y11;
    (x, y)
}

fn sample_bilinear(img: &GrayImage, x: f64, y: f64) -> u8 {
    if x < 0.0 || y < 0.0 || x >= img.w as f64 - 1.0 || y >= img.h as f64 - 1.0 {
        return 255; // paper
    }
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let fx = x - x0 as f64;
    let fy = y - y0 as f64;
    let p00 = img.get(x0, y0) as f64;
    let p10 = img.get(x0 + 1, y0) as f64;
    let p01 = img.get(x0, y0 + 1) as f64;
    let p11 = img.get(x0 + 1, y0 + 1) as f64;
    let v = p00 * (1.0 - fx) * (1.0 - fy)
        + p10 * fx * (1.0 - fy)
        + p01 * (1.0 - fx) * fy
        + p11 * fx * fy;
    v.round().clamp(0.0, 255.0) as u8
}

/// Convenience: decode from a PNG file on disk.
pub fn decode_png(
    path: &std::path::Path,
    grid: Option<(u32, u32)>,
) -> Result<Decoded, DecodeFailure> {
    let img = crate::pngio::read_gray(path).map_err(|_| DecodeFailure::NoMarkers)?;
    decode_image(&img, grid)
}

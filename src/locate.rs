//! Microform Code — locating the code: corner markers, cell pitch, grid size.
//!
//! Pipeline (port of the reference `decoder.py`, dependency-free):
//! Otsu → erode ×2 → connected components → filter by area/aspect →
//! one largest candidate per quadrant → 4 marker centers. The same
//! components give the cell pitch: an eroded 2×2-cell marker has side
//! `2p − 4`, so `p = (side + 4) / 2`.

use crate::imgproc::{binarize_ink, connected_components, erode3x3, Component};
use crate::layout as L;
use crate::pngio::GrayImage;

pub const MARKER_AREA_MIN: u64 = 40;
pub const MARKER_AREA_MAX_FRAC: f64 = 0.04;
pub const MARKER_ASPECT_MAX: f64 = 1.4;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Marker {
    pub cx: f64,
    pub cy: f64,
    /// Eroded bounding-box side in px (both sides equal for a square marker).
    pub side_px: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct Located {
    /// Marker centers, order: TL, TR, BL, BR.
    pub markers: [Marker; 4],
    /// Estimated cell pitch in px.
    pub pitch_px: f64,
    /// Estimated grid width/height in cells.
    pub grid_w: u32,
    pub grid_h: u32,
}

/// Find the four corner markers and estimate pitch/grid size.
pub fn locate(img: &GrayImage) -> Option<Located> {
    let bin = binarize_ink(img);
    let eroded = erode3x3(&erode3x3(&bin));
    let comps = connected_components(&eroded);

    let max_area = (img.w as f64 * img.h as f64 * MARKER_AREA_MAX_FRAC) as u64;
    let mut candidates: Vec<&Component> = comps
        .iter()
        .filter(|c| {
            c.pixels >= MARKER_AREA_MIN
                && c.pixels <= max_area
                && c.bbox_w() >= 2
                && c.bbox_h() >= 2
        })
        .filter(|c| {
            let aspect =
                c.bbox_w().max(c.bbox_h()) as f64 / c.bbox_w().min(c.bbox_h()).max(1) as f64;
            aspect <= MARKER_ASPECT_MAX
        })
        .collect();
    if candidates.len() < 4 {
        return None;
    }
    candidates.sort_by_key(|c| std::cmp::Reverse(c.pixels));

    // One largest candidate per quadrant, like the reference decoder.
    let mid_x = img.w as f64 / 2.0;
    let mid_y = img.h as f64 / 2.0;
    let mut best: [Option<&Component>; 4] = [None; 4]; // tl, tr, bl, br
    for &c in candidates.iter().take(12) {
        let (cx, cy) = c.center();
        let q = match (cx < mid_x, cy < mid_y) {
            (true, true) => 0,
            (false, true) => 1,
            (true, false) => 2,
            (false, false) => 3,
        };
        if best[q].is_none_or(|b| c.pixels > b.pixels) {
            best[q] = Some(c);
        }
    }
    if best.iter().any(|b| b.is_none()) {
        return None;
    }

    let markers: [Marker; 4] = std::array::from_fn(|i| {
        let c = best[i].unwrap();
        let (cx, cy) = c.center();
        Marker {
            cx,
            cy,
            // two erosions shrink a solid square 2 px per side
            side_px: (c.bbox_w() + c.bbox_h()) as f64 / 2.0 + 4.0,
        }
    });

    // Cell pitch from marker size (2×2 cells) — average over 4 markers.
    let pitch = markers.iter().map(|m| m.side_px / 2.0).sum::<f64>() / 4.0;
    if !(4.0..=200.0).contains(&pitch) {
        return None;
    }

    // Grid size from marker-center distances: centers span grid + 4 cells.
    let dist_x = markers[1].cx - markers[0].cx;
    let dist_y = markers[2].cy - markers[0].cy;
    if dist_x <= 0.0 || dist_y <= 0.0 {
        return None;
    }
    let grid_w = ((dist_x / pitch).round() as i64 - 4).max(4) as u32;
    let grid_h = ((dist_y / pitch).round() as i64 - 4).max(4) as u32;

    Some(Located {
        markers,
        pitch_px: pitch,
        grid_w,
        grid_h,
    })
}

/// Marker centers as `[(x, y); 4]` (TL, TR, BL, BR).
pub fn marker_points(loc: &Located) -> [(f64, f64); 4] {
    [
        (loc.markers[0].cx, loc.markers[0].cy),
        (loc.markers[1].cx, loc.markers[1].cy),
        (loc.markers[2].cx, loc.markers[2].cy),
        (loc.markers[3].cx, loc.markers[3].cy),
    ]
}

/// Marker centers in cell-space for a given grid (the renderer contract).
pub fn cellspace_marker_points(grid_w: u32, grid_h: u32) -> [(f64, f64); 4] {
    let w = grid_w + 2 * L::OFF;
    let h = grid_h + 2 * L::OFF;
    let m = L::MARKER_SIZE as f64 / 2.0; // 1.0
    [
        (L::PADDING as f64 + m, L::PADDING as f64 + m), // (1+1, 1+1) = (2, 2)
        (w as f64 - L::PADDING as f64 - m, L::PADDING as f64 + m),
        (L::PADDING as f64 + m, h as f64 - L::PADDING as f64 - m),
        (
            w as f64 - L::PADDING as f64 - m,
            h as f64 - L::PADDING as f64 - m,
        ),
    ]
}

//! Microform Code — visual patterns: masks, rendering, canonical tiles.
//!
//! A cell pattern is fully described by six integer parameters
//! (see [`crate::layout`]). [`pattern_pixels`] renders them to the
//! canonical 14×14 grayscale tile: 255 = paper, 0 = ink. Two parameter
//! sets that render to identical pixels are visually the same pattern —
//! the codebook keeps one canonical representative for each.

use crate::layout as L;

/// Ellipse mask: 0 = background, 255 = ink.
pub fn ellipse_mask(size_px: u32, angle_deg: f64, scale: f64, elong: f64) -> Vec<u8> {
    let n = size_px as usize;
    let half = size_px as f64 / 2.0;
    let (cos_a, sin_a) = (angle_deg.to_radians().cos(), angle_deg.to_radians().sin());
    let a = half * scale;
    let b = a / elong;

    let mut out = vec![0u8; n * n];
    for y in 0..n {
        let yy = y as f64 + 0.5 - half;
        for x in 0..n {
            let xx = x as f64 + 0.5 - half;
            let x_rot = xx * cos_a + yy * sin_a;
            let y_rot = -xx * sin_a + yy * cos_a;
            let dist = (x_rot / a).powi(2) + (y_rot / b).powi(2);
            let alpha = (1.0 - (dist - 0.90) / 0.2).clamp(0.0, 1.0);
            out[y * n + x] = (alpha * 255.0) as u8;
        }
    }
    out
}

/// Micro-dot mask centered at `(dx, dy)` with the given radius.
pub fn dot_mask(size_px: u32, dx: f64, dy: f64, radius: f64) -> Vec<u8> {
    let n = size_px as usize;
    let half = size_px as f64 / 2.0;

    let mut out = vec![0u8; n * n];
    for y in 0..n {
        let yy = y as f64 + 0.5 - half;
        for x in 0..n {
            let xx = x as f64 + 0.5 - half;
            let dist = ((xx - dx).powi(2) + (yy - dy).powi(2)).sqrt();
            let alpha = (1.0 - (dist - radius + 0.5) / 1.0).clamp(0.0, 1.0);
            out[y * n + x] = (alpha * 255.0) as u8;
        }
    }
    out
}

/// Render one pattern to a canonical grayscale tile (paper = 255).
pub fn pattern_pixels(av: u32, sv: u32, ev: u32, dx: u32, dy: u32, r: u32) -> Vec<u8> {
    let cs = L::CELL_PX as f64;
    let em = ellipse_mask(
        L::CELL_PX,
        L::angle_v_to_deg(av),
        L::scale_v_to_frac(sv),
        L::elong_v_to_ratio(ev),
    );
    let dm = dot_mask(
        L::CELL_PX,
        L::dot_dx_to_px(dx, cs),
        L::dot_dy_to_px(dy, cs),
        L::dot_rad_to_px(r),
    );
    em.iter()
        .zip(dm.iter())
        .map(|(e, d)| 255 - (*e).max(*d))
        .collect()
}

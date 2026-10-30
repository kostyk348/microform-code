//! Microform Code — geometry, layout and parameter space.
//!
//! The code is a canvas measured in *cells*. Every cell is a square that
//! carries one 17-bit pattern (an ellipse + a micro-dot). The canvas is
//! framed by corner markers, a checkerboard border and padding:
//!
//! ```text
//!   ┌──────────────────────────────┐
//!   │ ▪ P P ...            P P ▪   │   ▪ = marker (2×2 cells)
//!   │ P ▦ ▦ border ▦ ...      ▦ P  │   ▦ = checkerboard border
//!   │ P ▦ [ data grid W×H ] ▦ P    │
//!   │ P ▦ ▦ border ▦ ...      ▦ P  │
//!   │ ▪ P P ...            P P ▪   │
//!   └──────────────────────────────┘
//! ```
//!
//! Marker centers sit at cell coordinates `(2, 2)`, `(W+6, 2)`,
//! `(2, H+6)`, `(W+6, H+6)` — the decoder relies on this contract.

/// Corner marker size, in cells.
pub const MARKER_SIZE: u32 = 2;
/// Checkerboard calibration frame width, in cells.
pub const BORDER_WIDTH: u32 = 1;
/// Quiet zone between frame and edge, in cells.
pub const PADDING: u32 = 1;
/// Data-area offset from the canvas edge, in cells.
pub const OFF: u32 = MARKER_SIZE + BORDER_WIDTH + PADDING; // 4

/// Canonical rendering resolution of one cell, in pixels.
///
/// The codebook is built for exactly this size; the decoder resamples
/// every cell window to `CELL_PX × CELL_PX` before the lookup.
pub const CELL_PX: u32 = 14;

// ── Cell parameter space: 10 bits ellipse + 7 bits dot = 17 bits/cell ───

pub const ANGLE_VALUES: u32 = 64; // 6 bits, 0..179°
pub const SCALE_VALUES: u32 = 4; // 2 bits, 0.50..0.90
pub const ELONG_VALUES: u32 = 4; // 2 bits, 1.0..2.5
pub const DOT_DX_VALUES: u32 = 8; // 3 bits
pub const DOT_DY_VALUES: u32 = 8; // 3 bits
pub const DOT_RAD_VALUES: u32 = 4; // 2 bits

/// All parameter combinations: 64·4·4·8·8·4 = 262,144.
pub const PARAM_COMBOS: u32 =
    ANGLE_VALUES * SCALE_VALUES * ELONG_VALUES * DOT_DX_VALUES * DOT_DY_VALUES * DOT_RAD_VALUES;

/// Bits carried by one cell.
pub const BITS_PER_CELL: u32 = 17;

// ── Parameter value → physical units ────────────────────────────────────

pub fn angle_v_to_deg(v: u32) -> f64 {
    (v as f64 / 63.0) * 179.0
}

pub fn scale_v_to_frac(v: u32) -> f64 {
    0.50 + v as f64 * 0.133 // 0.50, 0.63, 0.77, 0.90
}

pub fn elong_v_to_ratio(v: u32) -> f64 {
    1.0 + v as f64 * 0.5 // 1.0, 1.5, 2.0, 2.5
}

pub fn dot_dx_to_px(v: u32, cell_size_px: f64) -> f64 {
    let margin = cell_size_px * 0.10;
    let span = cell_size_px * 0.5 - margin;
    -span + v as f64 * (2.0 * span / 7.0)
}

pub fn dot_dy_to_px(v: u32, cell_size_px: f64) -> f64 {
    dot_dx_to_px(v, cell_size_px)
}

pub fn dot_rad_to_px(v: u32) -> f64 {
    1.5 + v as f64 * 0.8 // 1.5, 2.3, 3.1, 3.9
}

/// Total canvas size in cells, including the frame: `(width, height)`.
pub fn canvas_cells(grid_w: u32, grid_h: u32) -> (u32, u32) {
    (grid_w + 2 * OFF, grid_h + 2 * OFF)
}

/// Split a 17-bit cell value into `(av, sv, ev, dx, dy, r)`.
pub fn params_from_value(value: u32) -> (u32, u32, u32, u32, u32, u32) {
    (
        (value >> 12) & 0x3F,
        (value >> 10) & 0x3,
        (value >> 8) & 0x3,
        (value >> 5) & 0x7,
        (value >> 2) & 0x7,
        value & 0x3,
    )
}

/// Pack `(av, sv, ev, dx, dy, r)` into a 17-bit cell value.
pub fn value_from_params(av: u32, sv: u32, ev: u32, dx: u32, dy: u32, r: u32) -> u32 {
    (av << 12) | (sv << 10) | (ev << 8) | (dx << 5) | (dy << 2) | r
}

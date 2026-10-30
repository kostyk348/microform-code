//! Microform Code — image generation: canvas, markers, border, data cells.

use crate::codebook;
use crate::codec;
use crate::layout as L;
use crate::patterns::{dot_mask, ellipse_mask};
use crate::pngio::GrayImage;

/// Rendering parameters for one code image.
#[derive(Debug, Clone, Copy)]
pub struct RenderSpec {
    /// Data grid width in cells.
    pub grid_w: u32,
    /// Data grid height in cells.
    pub grid_h: u32,
    /// Cell size in output pixels (integer ≥ 4).
    pub cell_px: u32,
}

impl Default for RenderSpec {
    fn default() -> Self {
        RenderSpec {
            grid_w: 92,
            grid_h: 92,
            cell_px: L::CELL_PX,
        }
    }
}

/// Encode `data` and render the full code canvas.
///
/// The cell bit depth comes from the scale codebook of `spec.cell_px`:
/// at the canonical 14 px/cell it is 17 bits, at other scales the depth
/// adapts so that every used pattern stays visually unique.
pub fn generate(spec: RenderSpec, data: &[u8]) -> GrayImage {
    let book = codebook::get_for_scale(spec.cell_px);
    let n_cells = (spec.grid_w * spec.grid_h) as usize;
    let indices = codec::encode(data, n_cells, book.bits);

    let (cells_w, cells_h) = L::canvas_cells(spec.grid_w, spec.grid_h);
    let cs = spec.cell_px as usize;
    let mut canvas = GrayImage::new(cells_w as usize * cs, cells_h as usize * cs, 255);

    // 1. Corner markers: solid black 2×2 cells.
    let corners = [
        (L::PADDING, L::PADDING),
        (cells_w - L::MARKER_SIZE - L::PADDING, L::PADDING),
        (L::PADDING, cells_h - L::MARKER_SIZE - L::PADDING),
        (
            cells_w - L::MARKER_SIZE - L::PADDING,
            cells_h - L::MARKER_SIZE - L::PADDING,
        ),
    ];
    for (cx0, cy0) in corners {
        for dy in 0..L::MARKER_SIZE {
            for dx in 0..L::MARKER_SIZE {
                fill_cell(&mut canvas, (cx0 + dx) as usize, (cy0 + dy) as usize, cs, 0);
            }
        }
    }

    // 2. Checkerboard border frame.
    let b_start = (L::MARKER_SIZE + L::PADDING) as usize;
    let b_end_w = cells_w as usize - L::MARKER_SIZE as usize - L::PADDING as usize;
    let b_end_h = cells_h as usize - L::MARKER_SIZE as usize - L::PADDING as usize;
    let bw = L::BORDER_WIDTH as usize;
    for cx in b_start..b_end_w {
        for k in 0..bw {
            let v = if (cx + k) % 2 == 0 { 0 } else { 255 };
            fill_cell(&mut canvas, cx, b_start + k, cs, v);
            fill_cell(&mut canvas, cx, b_end_h - bw + k, cs, v);
        }
    }
    for cy in (b_start + bw)..(b_end_h - bw) {
        for k in 0..bw {
            let v = if (cy + k) % 2 == 0 { 0 } else { 255 };
            fill_cell(&mut canvas, b_start + k, cy, cs, v);
            fill_cell(&mut canvas, b_end_w - bw + k, cy, cs, v);
        }
    }

    // 3. Data cells: ellipse + micro-dot from the codebook.
    let cb = codebook::get();
    let csf = cs as f64;
    let mut idx = 0usize;
    for row in 0..spec.grid_h as usize {
        for col in 0..spec.grid_w as usize {
            let (av, sv, ev, dx, dy, r) = cb.render_params(book.code(indices[idx]));
            idx += 1;
            let em = ellipse_mask(
                spec.cell_px,
                L::angle_v_to_deg(av),
                L::scale_v_to_frac(sv),
                L::elong_v_to_ratio(ev),
            );
            let dm = dot_mask(
                spec.cell_px,
                L::dot_dx_to_px(dx, csf),
                L::dot_dy_to_px(dy, csf),
                L::dot_rad_to_px(r),
            );
            let x0 = (L::OFF as usize + col) * cs;
            let y0 = (L::OFF as usize + row) * cs;
            for y in 0..cs {
                for x in 0..cs {
                    let ink = em[y * cs + x].max(dm[y * cs + x]);
                    canvas.set(x0 + x, y0 + y, 255 - ink);
                }
            }
        }
    }

    canvas
}

fn fill_cell(canvas: &mut GrayImage, cx: usize, cy: usize, cs: usize, v: u8) {
    for y in cy * cs..(cy + 1) * cs {
        for x in cx * cs..(cx + 1) * cs {
            canvas.set(x, y, v);
        }
    }
}

/// Physical size of the code in millimetres, given a cell size in µm.
pub fn physical_mm(spec: RenderSpec, cell_um: f64) -> f64 {
    let (cw, ch) = L::canvas_cells(spec.grid_w, spec.grid_h);
    (cw.max(ch) as f64 * cell_um) / 1000.0
}

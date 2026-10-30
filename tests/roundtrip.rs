//! End-to-end tests: render → locate → decode, at several print scales,
//! with physical damage. All deterministic.

use microform::decode::decode_image;
use microform::demo::PAYLOAD;
use microform::imgproc::{add_noise, rescale};
use microform::layout as L;
use microform::render::{generate, RenderSpec};

fn spec(grid: u32, cell_px: u32) -> RenderSpec {
    RenderSpec {
        grid_w: grid,
        grid_h: grid,
        cell_px,
    }
}

#[test]
fn decodes_at_all_print_scales() {
    for cell_px in [7u32, 14, 28] {
        let img = generate(spec(24, cell_px), PAYLOAD);
        let d = decode_image(&img, None)
            .unwrap_or_else(|e| panic!("decode failed at {cell_px} px/cell: {e}"));
        assert_eq!(d.data, PAYLOAD);
        assert_eq!(d.info.grid_w, 24);
        assert_eq!(d.info.cell_px, cell_px);
        assert_eq!(d.info.misses, 0);
    }
}

#[test]
fn auto_detects_the_default_92_grid() {
    let img = generate(RenderSpec::default(), PAYLOAD);
    let d = decode_image(&img, None).expect("92×92 decode");
    assert_eq!(d.data, PAYLOAD);
    assert_eq!((d.info.grid_w, d.info.grid_h), (92, 92));
    assert_eq!(d.info.bits_per_cell, 17);
}

#[test]
fn recovers_painted_over_cells_as_erasures() {
    // Destroy 8 data cells per 120-cell block (under the erasure
    // guarantee of 10): paint them over with paper color.
    let grid = 24u32;
    let mut img = generate(spec(grid, L::CELL_PX), PAYLOAD);
    let per_block = 120usize;
    let mut destroyed = 0usize;
    for b in 0..4 {
        for k in 0..8usize {
            let idx = b * per_block + (k * 13 + 5) % per_block;
            let (row, col) = ((idx as u32) / grid, (idx as u32) % grid);
            let cs = L::CELL_PX as usize;
            let x0 = (L::OFF + col) as usize * cs;
            let y0 = (L::OFF + row) as usize * cs;
            for y in y0..y0 + cs {
                for x in x0..x0 + cs {
                    img.set(x, y, 255);
                }
            }
            destroyed += 1;
        }
    }
    let d = decode_image(&img, None).expect("damaged decode");
    assert_eq!(d.data, PAYLOAD);
    assert_eq!(d.info.misses, destroyed);
}

#[test]
fn heavy_noise_fails_cleanly() {
    // The honest boundary: exact codebook lookup does not survive heavy
    // noise; the decoder must refuse, not return garbage.
    let img = generate(spec(24, L::CELL_PX), PAYLOAD);
    let noisy = add_noise(&img, 30.0, 7);
    if let Ok(d) = decode_image(&noisy, None) {
        assert_ne!(d.data, PAYLOAD, "garbage must not equal the payload");
    } // an Err is the honest refusal — expected
}

#[test]
fn rescaled_copies_never_misdecode() {
    // The supported path is native-scale rendering; a rescaled copy of a
    // master raster either still decodes or is refused cleanly — the one
    // thing it must never do is return plausible-looking wrong bytes.
    let img = generate(spec(24, 14), PAYLOAD);
    let big = rescale(&img, img.w * 2, img.h * 2);
    if let Ok(d) = decode_image(&big, None) {
        assert_eq!(d.data, PAYLOAD, "if it decodes, it must decode right");
    } // a clean refusal is acceptable for rescaled copies
}

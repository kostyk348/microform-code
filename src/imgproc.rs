//! Microform Code — tiny image-processing toolkit (cv2 replacements).
//!
//! Only what the decoder needs: Otsu thresholding, binary erosion and
//! connected components for marker detection. All functions are
//! deterministic and dependency-free.

use crate::pngio::GrayImage;

/// Otsu threshold. Returns the threshold value `t`; ink is `px <= t`.
pub fn otsu_threshold(img: &GrayImage) -> u8 {
    let mut hist = [0u64; 256];
    for &v in &img.data {
        hist[v as usize] += 1;
    }
    let total = img.data.len() as f64;
    let mut sum_all = 0u64;
    for (v, &c) in hist.iter().enumerate() {
        sum_all += (v as u64) * c;
    }

    let (mut sum_b, mut w_b) = (0u64, 0u64);
    let (mut best_t, mut best_var) = (0u8, -1.0f64);
    for t in 0..256u32 {
        w_b += hist[t as usize];
        if w_b == 0 {
            continue;
        }
        let w_f = total - w_b as f64;
        if w_f == 0.0 {
            break;
        }
        sum_b += (t as u64) * hist[t as usize];
        let m_b = sum_b as f64 / w_b as f64;
        let m_f = (sum_all - sum_b) as f64 / w_f;
        let var = w_b as f64 * w_f * (m_b - m_f) * (m_b - m_f);
        if var > best_var {
            best_var = var;
            best_t = t as u8;
        }
    }
    best_t
}

/// Binarize: ink (dark) = 255, paper = 0.
pub fn binarize_ink(img: &GrayImage) -> GrayImage {
    let t = otsu_threshold(img);
    GrayImage {
        w: img.w,
        h: img.h,
        data: img
            .data
            .iter()
            .map(|&v| if v <= t { 255 } else { 0 })
            .collect(),
    }
}

/// Binary erosion with a 3×3 structuring element; out-of-bounds counts
/// as background (255 stays only if the whole 3×3 neighborhood is ink).
pub fn erode3x3(bin: &GrayImage) -> GrayImage {
    let mut out = GrayImage::new(bin.w, bin.h, 0);
    for y in 0..bin.h {
        for x in 0..bin.w {
            let mut all_ink = true;
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                    let ink = nx >= 0
                        && ny >= 0
                        && (nx as usize) < bin.w
                        && (ny as usize) < bin.h
                        && bin.get(nx as usize, ny as usize) != 0;
                    if !ink {
                        all_ink = false;
                    }
                }
            }
            if all_ink {
                out.set(x, y, 255);
            }
        }
    }
    out
}

/// Connected component of a binary image, 8-connectivity.
#[derive(Debug, Clone)]
pub struct Component {
    pub pixels: u64,
    pub min_x: usize,
    pub min_y: usize,
    pub max_x: usize,
    pub max_y: usize,
}

impl Component {
    pub fn bbox_w(&self) -> usize {
        self.max_x - self.min_x + 1
    }
    pub fn bbox_h(&self) -> usize {
        self.max_y - self.min_y + 1
    }
    pub fn center(&self) -> (f64, f64) {
        // Continuous center: pixels [min..max] span [min, max+1).
        (
            self.min_x as f64 + self.bbox_w() as f64 / 2.0,
            self.min_y as f64 + self.bbox_h() as f64 / 2.0,
        )
    }
}

/// Label connected components (8-connectivity) of a binary image.
pub fn connected_components(bin: &GrayImage) -> Vec<Component> {
    let mut visited = vec![false; bin.w * bin.h];
    let mut out = Vec::new();
    let mut stack: Vec<(usize, usize)> = Vec::new();

    for y0 in 0..bin.h {
        for x0 in 0..bin.w {
            if visited[y0 * bin.w + x0] || bin.get(x0, y0) == 0 {
                continue;
            }
            stack.clear();
            stack.push((x0, y0));
            visited[y0 * bin.w + x0] = true;
            let mut comp = Component {
                pixels: 0,
                min_x: usize::MAX,
                min_y: usize::MAX,
                max_x: 0,
                max_y: 0,
            };
            while let Some((x, y)) = stack.pop() {
                comp.pixels += 1;
                comp.min_x = comp.min_x.min(x);
                comp.min_y = comp.min_y.min(y);
                comp.max_x = comp.max_x.max(x);
                comp.max_y = comp.max_y.max(y);
                for dy in -1i64..=1 {
                    for dx in -1i64..=1 {
                        let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                        if nx < 0 || ny < 0 || nx as usize >= bin.w || ny as usize >= bin.h {
                            continue;
                        }
                        let (nx, ny) = (nx as usize, ny as usize);
                        if !visited[ny * bin.w + nx] && bin.get(nx, ny) != 0 {
                            visited[ny * bin.w + nx] = true;
                            stack.push((nx, ny));
                        }
                    }
                }
            }
            out.push(comp);
        }
    }
    out
}

/// Box blur with the given radius (separable), for scan simulation.
pub fn box_blur(img: &GrayImage, radius: usize) -> GrayImage {
    if radius == 0 {
        return img.clone();
    }
    let w = img.w;
    let h = img.h;
    let mut tmp = vec![0u32; w * h];

    // horizontal
    for y in 0..h {
        let mut acc: u32 = 0;
        let row = y * w;
        for x in 0..w {
            acc += img.data[row + x] as u32;
            if x > 2 * radius {
                acc -= img.data[row + x - (2 * radius + 1)] as u32;
            }
            if x >= radius {
                let lo = x.saturating_sub(2 * radius);
                tmp[row + lo] = acc / ((2 * radius + 1) as u32);
            }
        }
        for x in 0..radius.min(w) {
            tmp[row + x] = tmp[row + radius.min(w - 1)];
        }
    }

    let mut out = GrayImage::new(w, h, 0);
    // vertical
    for x in 0..w {
        let mut acc: u32 = 0;
        for y in 0..h {
            acc += tmp[y * w + x];
            if y > 2 * radius {
                acc -= tmp[(y - (2 * radius + 1)) * w + x];
            }
            if y >= radius {
                let lo = y.saturating_sub(2 * radius);
                out.data[lo * w + x] = (acc / ((2 * radius + 1) as u32)) as u8;
            }
        }
        for y in 0..radius.min(h) {
            out.data[y * w + x] = out.data[radius.min(h - 1) * w + x];
        }
    }
    out
}

/// Deterministic gaussian-ish noise (Box–Muller over SplitMix64).
pub fn add_noise(img: &GrayImage, sigma: f64, seed: u64) -> GrayImage {
    let mut out = img.clone();
    let mut s: u64 = seed;
    let mut next = move || {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    for v in out.data.iter_mut() {
        let u1 = ((next() >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
        let u2 = ((next() >> 11) as f64) / (1u64 << 53) as f64;
        let g = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
        *v = ((*v as f64) + sigma * g).round().clamp(0.0, 255.0) as u8;
    }
    out
}

/// Nearest-neighbor rescale (scan simulation: DPI mismatch).
pub fn rescale(img: &GrayImage, new_w: usize, new_h: usize) -> GrayImage {
    let mut out = GrayImage::new(new_w, new_h, 0);
    for y in 0..new_h {
        let sy = (y as u64 * img.h as u64 / new_h as u64) as usize;
        for x in 0..new_w {
            let sx = (x as u64 * img.w as u64 / new_w as u64) as usize;
            out.set(x, y, img.get(sx, sy));
        }
    }
    out
}

//! Microform Code — planar homography from four point pairs.
//!
//! Solves the exact 8×8 DLT system (h₈ = 1) with Gaussian elimination
//! and partial pivoting. Enough for the four corner markers; no external
//! linear-algebra dependency.

/// Apply a 3×3 homography (row-major, h₈ last) to a point.
pub fn apply(h: &[f64; 9], x: f64, y: f64) -> (f64, f64) {
    let d = h[6] * x + h[7] * y + h[8];
    (
        (h[0] * x + h[1] * y + h[2]) / d,
        (h[3] * x + h[4] * y + h[5]) / d,
    )
}

/// Homography `src → dst` from exactly four point pairs.
pub fn from_points(src: &[(f64, f64); 4], dst: &[(f64, f64); 4]) -> Option<[f64; 9]> {
    // u = (h0 x + h1 y + h2) / (h6 x + h7 y + 1)
    // v = (h3 x + h4 y + h5) / (h6 x + h7 y + 1)
    // 8 unknowns: h0..h7
    let mut a = [[0.0f64; 8]; 8];
    let mut b = [0.0f64; 8];
    for i in 0..4 {
        let (x, y) = src[i];
        let (u, v) = dst[i];
        a[2 * i] = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y];
        b[2 * i] = u;
        a[2 * i + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y];
        b[2 * i + 1] = v;
    }
    gauss_solve(&mut a, &mut b).map(|h| {
        let mut out = [0.0f64; 9];
        out[..8].copy_from_slice(&h);
        out[8] = 1.0;
        out
    })
}

/// Gaussian elimination with partial pivoting; solves `a x = b`.
fn gauss_solve(a: &mut [[f64; 8]; 8], b: &mut [f64; 8]) -> Option<[f64; 8]> {
    let n = 8;
    for col in 0..n {
        // pivot
        let mut piv = col;
        for row in col + 1..n {
            if a[row][col].abs() > a[piv][col].abs() {
                piv = row;
            }
        }
        if a[piv][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);

        let inv = 1.0 / a[col][col];
        for row in col + 1..n {
            let f = a[row][col] * inv;
            if f == 0.0 {
                continue;
            }
            let pivot_row = a[col];
            for (k, ak) in a[row][col..].iter_mut().enumerate() {
                *ak -= f * pivot_row[col + k];
            }
            b[row] -= f * b[col];
        }
    }

    let mut x = [0.0f64; 8];
    for row in (0..n).rev() {
        let mut s = b[row];
        for k in row + 1..n {
            s -= a[row][k] * x[k];
        }
        x[row] = s / a[row][row];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity() {
        let pts = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0)];
        let h = from_points(&pts, &pts).unwrap();
        let (u, v) = apply(&h, 3.0, 7.0);
        assert!((u - 3.0).abs() < 1e-9 && (v - 7.0).abs() < 1e-9);
    }

    #[test]
    fn scale_and_translate() {
        let src = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0)];
        let dst = [(5.0, 7.0), (25.0, 7.0), (5.0, 27.0), (25.0, 27.0)];
        let h = from_points(&src, &dst).unwrap();
        let (u, v) = apply(&h, 4.0, 3.0);
        assert!((u - 13.0).abs() < 1e-9 && (v - 13.0).abs() < 1e-9);
    }

    #[test]
    fn projective() {
        let src = [(0.0, 0.0), (100.0, 0.0), (0.0, 100.0), (100.0, 100.0)];
        let dst = [(10.0, 10.0), (200.0, 20.0), (30.0, 190.0), (220.0, 210.0)];
        let h = from_points(&src, &dst).unwrap();
        for (s, d) in src.iter().zip(dst.iter()) {
            let p = apply(&h, s.0, s.1);
            assert!((p.0 - d.0).abs() < 1e-6 && (p.1 - d.1).abs() < 1e-6);
        }
    }
}

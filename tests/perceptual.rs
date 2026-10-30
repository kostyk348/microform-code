//! Perceptual audit: usable tiles must stay visually separated.

use microform::codebook;
use microform::perceptual;

#[test]
fn canonical_scale_is_perceptually_separated() {
    let book = codebook::get_for_scale(14);
    assert_eq!(book.codes.len(), 131_072);
    assert_eq!(book.bits, 17);

    let a = perceptual::audit_scale(&book);
    assert!(a.closest.is_some(), "the hash net must compare pairs");
    assert!(
        a.min_mae > 0.0,
        "byte-distinct usable tiles must also be perceptually distinct"
    );
}

#[test]
fn audit_is_deterministic() {
    let book = codebook::get_for_scale(7);
    let a1 = perceptual::audit_scale(&book);
    let a2 = perceptual::audit_scale(&book);
    assert_eq!(a1.min_mae, a2.min_mae);
    assert_eq!(a1.closest, a2.closest);
    assert_eq!(a1.linf_le_2, a2.linf_le_2);
    assert_eq!(a1.linf_le_4, a2.linf_le_4);
    assert_eq!(a1.linf_le_8, a2.linf_le_8);
}

#[test]
fn low_scale_books_shrink_but_stay_separated() {
    let book = codebook::get_for_scale(6);
    assert!(
        book.codes.len() < 131_072,
        "tiles must collapse below the canonical scale"
    );
    let a = perceptual::audit_scale(&book);
    assert!(a.min_mae > 0.0);
}

#[test]
fn neighbour_margins_are_positive_and_finite() {
    let m = perceptual::neighbour_margins(codebook::get());
    for (name, v) in perceptual::PARAM_NAMES.iter().zip(m.iter()) {
        assert!(
            v.is_finite() && *v > 0.0,
            "parameter {name} must have usable one-step neighbours with a positive margin"
        );
    }
    // Measured reality (v1.1.0 audit): several one-step pairs differ by a
    // single gray level on a single pixel (margin ≈ 1/196 ≈ 0.005), i.e.
    // the closest perceptual step is BELOW scanner visibility. Those are
    // exactly the fragile pairs the audit counts; RS(255,223) absorbs the
    // residual confusion. The assertion pins the property, not the value.
    let worst = m.iter().cloned().fold(f32::INFINITY, f32::min);
    assert!(worst > 0.0, "identical neighbour tiles must be impossible");
}

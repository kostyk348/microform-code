//! Microform Code — a dense 2D code on paper.
//!
//! Each cell of the grid carries 17 bits: a rotated/scaled/stretched
//! ellipse (10 bits) plus an offset micro-dot (7 bits). A deterministic
//! codebook keeps exactly the 131,072 parameter combinations that render
//! to *distinct* 14×14 pixel tiles, so decoding is an exact lookup, and
//! Reed–Solomon(255,223) protects the bitstream (one RS block = exactly
//! 120 cells). At 300 µm/cell a 92×92 code on a 30×30 mm sticker holds
//! ~15 KB — about ×5 a QR-40 of the same size.
//!
//! # Quick start
//!
//! ```no_run
//! use microform::{render, pngio, decode};
//!
//! let img = render::generate(render::RenderSpec::default(), b"Hello, paper!");
//! pngio::write_gray(std::path::Path::new("code.png"), &img).unwrap();
//! let decoded = decode::decode_png(std::path::Path::new("code.png"), None).unwrap();
//! assert_eq!(decoded.data, b"Hello, paper!");
//! ```
//!
//! The CLI mirrors the library: `microform encode`, `decode`, `capacity`,
//! `codebook`, `perceptual`, `demo`.

pub mod codebook;
pub mod codec;
pub mod decode;
pub mod demo;
pub mod homography;
pub mod imgproc;

/// JNI bridge for the Android app — compiled only with `--features jni`.
#[cfg(feature = "jni")]
pub mod jni;

pub mod layout;
pub mod locate;
pub mod patterns;
pub mod perceptual;
pub mod pngio;
pub mod render;

pub use codebook::{get as codebook, get_for_scale, Codebook, ScaleCodebook};
pub use codec::{capacity_info, decode as decode_cells, encode as encode_cells, Capacity};
pub use decode::{decode_image, decode_png, DecodeFailure, Decoded};
pub use perceptual::{audit_scale, neighbour_margins, ScaleAudit};
pub use pngio::GrayImage;
pub use render::{generate, RenderSpec};

/// Crate version, as reported by the CLI.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

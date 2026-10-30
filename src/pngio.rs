//! Microform Code — minimal grayscale PNG I/O (8-bit, single channel).
//!
//! Both file-based and in-memory helpers are provided; the in-memory
//! pair (`encode_gray` / `decode_gray`) is what the JNI bridge uses on
//! Android.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

/// Read a PNG as an 8-bit grayscale image. Color inputs are converted
/// with the ITU-R BT.601 luma formula.
pub fn read_gray(path: &Path) -> Result<GrayImage, String> {
    let file = File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    decode_reader(file)
}

/// Decode a PNG from memory as an 8-bit grayscale image (same color
/// conversions as [`read_gray`]).
pub fn decode_gray(bytes: &[u8]) -> Result<GrayImage, String> {
    decode_reader(std::io::Cursor::new(bytes))
}

fn decode_reader<R: std::io::Read>(reader_src: R) -> Result<GrayImage, String> {
    let decoder = png::Decoder::new(reader_src);
    let mut reader = decoder
        .read_info()
        .map_err(|e| format!("png header: {e}"))?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| format!("png frame: {e}"))?;

    let w = info.width as usize;
    let h = info.height as usize;
    let data = &buf[..info.buffer_size()];

    let gray: Vec<u8> = match info.color_type {
        png::ColorType::Grayscale => data.to_vec(),
        png::ColorType::GrayscaleAlpha => data.chunks(2).map(|px| px[0]).collect(),
        png::ColorType::Rgb => data
            .chunks(3)
            .map(|px| (0.299 * px[0] as f32 + 0.587 * px[1] as f32 + 0.114 * px[2] as f32) as u8)
            .collect(),
        png::ColorType::Rgba => data
            .chunks(4)
            .map(|px| (0.299 * px[0] as f32 + 0.587 * px[1] as f32 + 0.114 * px[2] as f32) as u8)
            .collect(),
        other => return Err(format!("unsupported color type: {other:?}")),
    };

    Ok(GrayImage { w, h, data: gray })
}

/// Write an 8-bit grayscale PNG.
pub fn write_gray(path: &Path, img: &GrayImage) -> Result<(), String> {
    let file = File::create(path).map_err(|e| format!("cannot create {}: {e}", path.display()))?;
    let mut w = BufWriter::new(file);
    let mut enc = png::Encoder::new(&mut w, img.w as u32, img.h as u32);
    enc.set_color(png::ColorType::Grayscale);
    enc.set_depth(png::BitDepth::Eight);
    enc.set_compression(png::Compression::Fast);
    let mut writer = enc.write_header().map_err(|e| format!("png header: {e}"))?;
    writer
        .write_image_data(&img.data)
        .map_err(|e| format!("png data: {e}"))?;
    writer.finish().map_err(|e| format!("png finish: {e}"))?;
    Ok(())
}

/// Encode an 8-bit grayscale image as a PNG, in memory.
pub fn encode_gray(img: &GrayImage) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    let mut enc = png::Encoder::new(&mut out, img.w as u32, img.h as u32);
    enc.set_color(png::ColorType::Grayscale);
    enc.set_depth(png::BitDepth::Eight);
    enc.set_compression(png::Compression::Fast);
    let mut writer = enc.write_header().map_err(|e| format!("png header: {e}"))?;
    writer
        .write_image_data(&img.data)
        .map_err(|e| format!("png data: {e}"))?;
    writer.finish().map_err(|e| format!("png finish: {e}"))?;
    Ok(out)
}

/// An 8-bit grayscale image, row-major.
#[derive(Clone)]
pub struct GrayImage {
    pub w: usize,
    pub h: usize,
    pub data: Vec<u8>,
}

impl GrayImage {
    pub fn new(w: usize, h: usize, fill: u8) -> Self {
        GrayImage {
            w,
            h,
            data: vec![fill; w * h],
        }
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> u8 {
        self.data[y * self.w + x]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, v: u8) {
        self.data[y * self.w + x] = v;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn png_memory_roundtrip() {
        let mut img = GrayImage::new(37, 23, 0);
        for y in 0..23usize {
            for x in 0..37usize {
                img.set(x, y, ((x * 7 + y * 13) % 256) as u8);
            }
        }
        let png = encode_gray(&img).unwrap();
        let back = decode_gray(&png).unwrap();
        assert_eq!((back.w, back.h), (37, 23));
        assert_eq!(back.data, img.data);
    }

    #[test]
    fn code_roundtrip_through_memory_png() {
        // The path the Android app takes: generate → PNG bytes → decode.
        let img = crate::render::generate(
            crate::render::RenderSpec {
                grid_w: 12,
                grid_h: 12,
                cell_px: 14,
            },
            b"in-memory",
        );
        let png = encode_gray(&img).unwrap();
        let back = decode_gray(&png).unwrap();
        let decoded = crate::decode::decode_image(&back, Some((12, 12))).unwrap();
        assert_eq!(decoded.data, b"in-memory");
    }
}

//! Microform Code — minimal grayscale PNG I/O (8-bit, single channel).

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

/// Read a PNG as an 8-bit grayscale image. Color inputs are converted
/// with the ITU-R BT.601 luma formula.
pub fn read_gray(path: &Path) -> Result<GrayImage, String> {
    let file = File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let decoder = png::Decoder::new(file);
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

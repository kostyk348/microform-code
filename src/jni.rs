//! JNI bridge for the Android app (`--features jni`, built with cargo-ndk).
//!
//! Exposes the same Rust core to Kotlin (`MicroformLib.kt`): render bytes
//! to a grayscale PNG, decode a PNG, and decode a raw grayscale camera
//! frame. `encodePng` raises a Java exception on failure; the decode
//! functions return `null` *without* raising — a frame with no code is a
//! normal outcome on a camera thread, and a pending exception would abort
//! the next JNI call on that thread.
//!
//! The functions are only compiled when the `jni` feature is enabled —
//! desktop builds (bin + cdylib without a JVM) stay dependency-free.

use jni::objects::{JByteArray, JClass, JString};
use jni::JNIEnv;

const ERR_CLASS: &str = "java/lang/IllegalArgumentException";

fn throw(env: &mut JNIEnv, msg: &str) {
    // Best effort: if raising fails there is nothing sensible left to do;
    // the Kotlin side treats a null result as an error either way.
    let _ = env.throw_new(ERR_CLASS, msg);
}

#[no_mangle]
pub extern "system" fn Java_com_kostyk348_microform_MicroformLib_version<'local>(
    env: JNIEnv<'local>,
    _class: JClass<'local>,
) -> JString<'local> {
    env.new_string(crate::VERSION)
        .unwrap_or_else(|_| JString::default())
}

/// `byte[] encodePng(byte[] data) -> byte[]` — render the default 92×92
/// code as a grayscale PNG. Rejects input that would not fit (the codec
/// would otherwise truncate silently).
#[no_mangle]
pub extern "system" fn Java_com_kostyk348_microform_MicroformLib_encodePng<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    data: JByteArray<'local>,
) -> JByteArray<'local> {
    match encode_png(&mut env, &data) {
        Ok(v) => v,
        Err(msg) => {
            throw(&mut env, &msg);
            JByteArray::default() // null
        }
    }
}

fn encode_png<'local>(
    env: &mut JNIEnv<'local>,
    data: &JByteArray<'local>,
) -> Result<JByteArray<'local>, String> {
    let bytes = env
        .convert_byte_array(data)
        .map_err(|e| format!("cannot read input: {e}"))?;

    let spec = crate::render::RenderSpec::default();
    let book = crate::codebook::get_for_scale(spec.cell_px);
    let n_cells = (spec.grid_w * spec.grid_h) as usize;
    let cap = crate::codec::capacity_info(n_cells, book.bits).payload_bytes;
    if bytes.len() > cap {
        return Err(format!(
            "input is {} bytes; the default 92×92 code holds {} bytes of payload (see `microform capacity`)",
            bytes.len(),
            cap
        ));
    }

    let img = crate::render::generate(spec, &bytes);
    let png = crate::pngio::encode_gray(&img)?;
    env.byte_array_from_slice(&png)
        .map_err(|e| format!("cannot create result array: {e}"))
}

/// `byte[] decodePngBytes(byte[] png) -> byte[]` — decode a grayscale
/// PNG of a code, returns the payload bytes; null when no code was found
/// (no exception is raised).
#[no_mangle]
pub extern "system" fn Java_com_kostyk348_microform_MicroformLib_decodePngBytes<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    png: JByteArray<'local>,
) -> JByteArray<'local> {
    // A decode miss is a normal outcome → plain null, no pending exception.
    decode_png_bytes(&mut env, &png).unwrap_or_default()
}

fn decode_png_bytes<'local>(
    env: &mut JNIEnv<'local>,
    png: &JByteArray<'local>,
) -> Result<JByteArray<'local>, String> {
    let bytes = env
        .convert_byte_array(png)
        .map_err(|e| format!("cannot read input: {e}"))?;
    let img = crate::pngio::decode_gray(&bytes)?;
    let decoded = crate::decode::decode_image(&img, None).map_err(|e| format!("decode: {e}"))?;
    env.byte_array_from_slice(&decoded.data)
        .map_err(|e| format!("cannot create result array: {e}"))
}

/// `byte[] decodeGray(byte[] gray, int width, int height) -> byte[]` —
/// decode a raw 8-bit grayscale camera frame (row-major, tightly packed;
/// the Kotlin analyzer deals with strides and rotation). Returns the
/// payload bytes, or plain null when nothing was found (no exception).
#[no_mangle]
pub extern "system" fn Java_com_kostyk348_microform_MicroformLib_decodeGray<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    gray: JByteArray<'local>,
    width: i32,
    height: i32,
) -> JByteArray<'local> {
    // A decode miss is a normal outcome → plain null, no pending exception.
    decode_gray_frame(&mut env, &gray, width, height).unwrap_or_default()
}

fn decode_gray_frame<'local>(
    env: &mut JNIEnv<'local>,
    gray: &JByteArray<'local>,
    width: i32,
    height: i32,
) -> Result<JByteArray<'local>, String> {
    if width <= 0 || height <= 0 {
        return Err(format!("bad frame size {width}×{height}"));
    }
    let (w, h) = (width as usize, height as usize);
    let buf = env
        .convert_byte_array(gray)
        .map_err(|e| format!("cannot read frame: {e}"))?;
    if buf.len() < w * h {
        return Err(format!(
            "frame buffer is {} bytes, expected {} for {w}×{h}",
            buf.len(),
            w * h
        ));
    }
    let img = crate::pngio::GrayImage {
        w,
        h,
        data: buf[..w * h].to_vec(),
    };
    let decoded = crate::decode::decode_image(&img, None).map_err(|e| format!("decode: {e}"))?;
    env.byte_array_from_slice(&decoded.data)
        .map_err(|e| format!("cannot create result array: {e}"))
}

package com.kostyk348.microform

/**
 * Kotlin facade over the Rust core (`libmicroform.so`, built with
 * cargo-ndk from the crate in the repository root — see `src/jni.rs`).
 *
 * Decode functions return `null` when nothing was found. That is a normal
 * outcome for a camera frame, not an error, so no Java exception is raised
 * (a pending JNI exception would abort the next native call on the thread).
 * `encodePng` throws [IllegalArgumentException] when the input does not
 * fit into the default 92×92 code.
 */
object MicroformLib {
    init {
        System.loadLibrary("microform")
    }

    /** Version of the Rust core (crate version). */
    external fun version(): String

    /**
     * Render [data] (UTF-8, at most 15 606 bytes) as the default 92×92
     * code, returned as a grayscale PNG.
     */
    external fun encodePng(data: ByteArray): ByteArray?

    /** Decode a grayscale PNG of a code; null when no code was found. */
    external fun decodePngBytes(png: ByteArray): ByteArray?

    /**
     * Decode a raw packed 8-bit grayscale frame (width × height,
     * row-major, no stride padding; the analyzer deals with strides and
     * rotation before calling); null when no code was found.
     */
    external fun decodeGray(gray: ByteArray, width: Int, height: Int): ByteArray?
}

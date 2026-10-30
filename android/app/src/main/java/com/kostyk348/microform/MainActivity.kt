package com.kostyk348.microform

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Size
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import java.io.File
import java.util.concurrent.Executors

/**
 * Minimal front end for the Microform Code: render text as a printable
 * code (PNG), or read a printed code live from the camera. All of the
 * heavy lifting happens in the Rust core via JNI — this activity only
 * moves bytes around.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var editData: EditText
    private lateinit var status: TextView
    private lateinit var codeImage: ImageView
    private lateinit var previewView: PreviewView
    private lateinit var btnEncode: Button
    private lateinit var btnShare: Button
    private lateinit var btnScan: Button

    private val analyzerExecutor = Executors.newSingleThreadExecutor()
    private var cameraProvider: ProcessCameraProvider? = null

    @Volatile
    private var scanning = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        editData = findViewById(R.id.edit_data)
        status = findViewById(R.id.status)
        codeImage = findViewById(R.id.code_image)
        previewView = findViewById(R.id.preview)
        btnEncode = findViewById(R.id.btn_encode)
        btnShare = findViewById(R.id.btn_share)
        btnScan = findViewById(R.id.btn_scan)

        status.text = getString(R.string.status_idle, MicroformLib.version())

        btnEncode.setOnClickListener { encode() }
        btnShare.setOnClickListener { sharePng() }
        btnScan.setOnClickListener { toggleScan() }
    }

    // ── encode ──────────────────────────────────────────────────────────

    private fun encode() {
        val bytes = editData.text.toString().toByteArray(Charsets.UTF_8)
        if (bytes.isEmpty()) {
            Toast.makeText(this, R.string.empty_input, Toast.LENGTH_SHORT).show()
            return
        }
        status.setText(R.string.encoding)
        btnEncode.isEnabled = false
        Thread {
            val png = try {
                MicroformLib.encodePng(bytes)
            } catch (e: Throwable) {
                runOnUiThread { Toast.makeText(this, e.message ?: "encode failed", Toast.LENGTH_LONG).show() }
                null
            }
            runOnUiThread {
                btnEncode.isEnabled = true
                if (png == null) {
                    status.setText(R.string.encode_failed)
                } else {
                    val bmp = BitmapFactory.decodeByteArray(png, 0, png.size)
                    codeImage.setImageBitmap(bmp)
                    codeImage.visibility = View.VISIBLE
                    btnShare.isEnabled = true
                    val where = savePng(png)
                    status.text = getString(
                        R.string.encoded_ok,
                        bmp.width, bmp.height, png.size / 1024,
                        where?.let { getString(R.string.saved_to, it) } ?: ""
                    )
                }
            }
        }.start()
    }

    private fun savePng(png: ByteArray): String? = try {
        val dir = File(filesDir, "codes").apply { mkdirs() }
        File(dir, "microform.png").apply { writeBytes(png) }.absolutePath
    } catch (e: Exception) {
        null
    }

    private fun sharePng() {
        val f = File(File(filesDir, "codes"), "microform.png")
        if (!f.exists()) return
        val uri = FileProvider.getUriForFile(this, "$packageName.files", f)
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "image/png"
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        startActivity(Intent.createChooser(intent, getString(R.string.share_title)))
    }

    // ── scan ────────────────────────────────────────────────────────────

    private fun toggleScan() {
        if (scanning) {
            stopScan()
            return
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), REQ_CAMERA)
            return
        }
        startScan()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray,
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQ_CAMERA) {
            if (grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) {
                startScan()
            } else {
                status.setText(R.string.no_camera_permission)
            }
        }
    }

    private fun startScan() {
        scanning = true
        btnScan.setText(R.string.stop_scan)
        previewView.visibility = View.VISIBLE
        status.setText(R.string.scan_hint)

        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            cameraProvider = provider

            val preview = Preview.Builder()
                .build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }

            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(
                    ResolutionSelector.Builder()
                        .setResolutionStrategy(
                            ResolutionStrategy(
                                Size(1600, 1200),
                                ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER,
                            )
                        )
                        .build()
                )
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { it.setAnalyzer(analyzerExecutor) { frame -> analyzeFrame(frame) } }

            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun stopScan() {
        scanning = false
        btnScan.setText(R.string.scan)
        cameraProvider?.unbindAll()
        previewView.visibility = View.GONE
        status.setText(R.string.scan_stopped)
    }

    private fun analyzeFrame(frame: ImageProxy) {
        if (!scanning) {
            frame.close()
            return
        }
        val w = frame.width
        val h = frame.height
        val packed = yPlanePacked(frame)
        val rotation = frame.imageInfo.rotationDegrees
        frame.close()
        if (packed == null) return

        // Upright grayscale for the decoder; dimensions swap at 90°/270°.
        var ew = w
        var eh = h
        val gray = when (rotation) {
            90 -> {
                ew = h; eh = w
                rotate90(packed, w, h)
            }
            180 -> rotate180(packed, w, h)
            270 -> {
                ew = h; eh = w
                rotate90(rotate180(packed, w, h), w, h)
            }
            else -> packed
        }

        val data = MicroformLib.decodeGray(gray, ew, eh)
        if (data != null && scanning) {
            runOnUiThread { onDecoded(data) }
        }
    }

    /** Copy the Y plane into a tightly packed w×h buffer, or null. */
    private fun yPlanePacked(frame: ImageProxy): ByteArray? {
        val w = frame.width
        val h = frame.height
        val plane = frame.planes[0]
        if (plane.pixelStride != 1) return null
        val buf = plane.buffer
        val rowStride = plane.rowStride
        val packed = ByteArray(w * h)
        for (y in 0 until h) {
            buf.position(y * rowStride)
            if (buf.remaining() < w) return null
            buf.get(packed, y * w, w)
        }
        return packed
    }

    private fun rotate90(src: ByteArray, w: Int, h: Int): ByteArray {
        val dst = ByteArray(src.size)
        for (y in 0 until h) {
            for (x in 0 until w) {
                dst[x * h + (h - 1 - y)] = src[y * w + x]
            }
        }
        return dst
    }

    private fun rotate180(src: ByteArray, w: Int, h: Int): ByteArray {
        val dst = ByteArray(src.size)
        for (i in src.indices) {
            dst[i] = src[src.size - 1 - i]
        }
        return dst
    }

    private fun onDecoded(data: ByteArray) {
        stopScan()
        val text = runCatching { String(data, Charsets.UTF_8) }.getOrNull()
        val shown = if (text != null && text.isNotEmpty()) {
            if (text.length > 400) text.take(400) + "…" else text
        } else {
            getString(R.string.binary_payload, data.size)
        }
        status.text = getString(R.string.decoded_ok, data.size, shown)
    }

    override fun onDestroy() {
        super.onDestroy()
        analyzerExecutor.shutdown()
    }

    private companion object {
        const val REQ_CAMERA = 41
    }
}

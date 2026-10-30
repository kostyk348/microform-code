# Limitations — the honest list

What Microform Code v1.0.0 does, and what it does not do yet.

## Decoding is an exact hash lookup — by design

The codebook guarantees that every *used* tile renders to unique pixels
at its scale. Decoding compares pixels byte-for-byte. Consequences:

* ✅ A pristine raster (what the printer was fed) decodes perfectly at
  any supported scale, with auto-detected grid.
* ❌ Random noise, blur, uneven lighting, paper grain, or a photograph
  break the hash. There is no nearest-neighbor fallback in v1. A real
  camera pipeline needs perceptual matching (e.g. descriptors robust to
  blur/lighting) — that is the single biggest roadmap item, and the demo
  shows the decoder refusing rather than guessing.
* ✅ What *is* covered: cells destroyed by a stamp, a smudge or a blob
  become erasures with known byte positions and recover up to
  ⌊32 / span⌋ ≈ 10 cells per 120-cell block — twice the 5-error budget.

## Scale is a format parameter, not a free parameter

The code tables are derived deterministically from the integer cell
size. A code printed at 14 px/cell decodes when the raster is read back
at 14 px/cell; a photograph of the same print (whose cells are, say,
23.7 px) needs rescaling to the print lattice *before* v1's exact
lookup works — and after rescaling, see the point above. Supported
scales today: 4..32 px/cell (integer).

## Error isolation depends on the bit depth

One RS block is 2040 bits. The cell depth (17/15/12/10 bits — the
largest depth whose value range fits the usable pattern set and which
divides 2040) keeps every cell inside one block. At 17 bits a cell error
spans ≤ 3 bytes → 5 guaranteed corrections per block; at 15 bits ≤ 2
bytes → 8. Numbers per scale are printed by `microform capacity` and
verified by tests.

## No interlacing, masking or registration marks beyond the frame

* There is no QR-style data masking: the PRNG padding makes the texture
  visually uniform, but no effort is made to avoid accidental large
  blank/dark regions (the deterministic noise makes them improbable,
  not impossible).
* Perspective tolerance is bounded by marker detection: the four
  2×2-cell markers survive mild rotation and skew (eroded blobs must
  stay ≥ 40 px and roughly square), but strong perspective or partial
  occlusion of a marker breaks location.
* Minimum useful grid: one RS block of payload — about 120 cells at 17
  bits (e.g. 12×10); below that, capacity is 0 bytes and `encode`
  refuses.

## Performance

* First decode at a new scale builds that scale's codebook (~0.3–1 s);
  it is cached per process.
* A full 92×92 decode is ~1 s including the scale book; encode is
  ~20 ms after the book is built.

## Not done yet (roadmap)

- [ ] Perceptual matching for camera images (the big one)
- [ ] Data masking / region interleaving for burst-error robustness
- [ ] Larger marker alphabet (orientation + scale-encoded fiducials)
- [ ] Color layering (2 channels ≈ ×2 capacity, needs its own codebook)
- [ ] Real print-and-photograph roundtrip validation

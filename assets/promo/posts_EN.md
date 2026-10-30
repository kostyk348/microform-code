# Launch posts for Microform Code (EN)

## Hacker News (Show HN)

**Show HN: Microform Code — a paper 2D code with 17 bits per cell (×5 QR capacity)**

I turned my Python prototype into a complete Rust crate. Microform Code
encodes 17 bits into every grid cell — a rotated/scaled/stretched ellipse
(10 bits) plus an offset micro-dot (7 bits) — and protects the bitstream
with RS(255,223). At 300 µm per cell a 30×30 mm sticker holds 15.2 KB,
against ~3 KB for a QR-40 of the same size.

The part I care most about: decoding is an *exact pixel-tile lookup*
against a deterministic codebook. Of 262,144 parameter combinations,
184,154 render to visually distinct tiles at the canonical scale; the
codec uses exactly 2¹⁷ of them. No heuristics, no thresholds — a cell is
either a known tile or a *known-missing* one, and missing cells become
Reed–Solomon erasures (10 per block recovered, versus 5 unknown-position
errors). One RS block is exactly 120 cells, so cell damage stays isolated.

The honest boundary, stated in the README and enforced by the demo:
pristine rasters decode at any print scale (the decoder measures the cell
pitch from corner markers and adapts the code table per scale), but noise,
blur and photographs break the exact hash — camera decoding needs
perceptual matching, which is roadmap, not a claim. `microform demo` runs
8 deterministic checks in ~2 s, including "heavy noise is reported, not
guessed".

Two dependencies (`png`, `reed-solomon`); Otsu, erosion, connected
components and the homography solver are written in the crate. 24 tests,
Apache-2.0.

https://github.com/kostyk348/microform-code

## Reddit r/rust (text post)

**microform 1.0.0 — a 2D paper code at 17 bits/cell, as a two-dependency Rust crate**

I replaced my Python prototype (cv2 + numpy + reedsolo, decode locked to
one scale) with a from-scratch Rust implementation. Everything except PNG
I/O and Reed–Solomon is hand-written: Otsu thresholding, binary erosion,
connected components for marker detection, an exact 4-point homography
solver, a scale-aware codebook, and a CLI with no clap.

Fun engineering bits:
- deterministic codebook build (~0.4 s, no cache file) — same bytes on
  every machine, asserted by tests;
- scale-aware decoding: the pitch measured from markers selects a code
  book of tiles that stay unique at *that* scale; the cell bit depth
  adapts (17→15→12 bits, always a divisor of a 2040-bit RS block);
- RS erasure decoding for "known-missing" cells: a stamp or smudge over
  10 cells per block still decodes;
- partial RS blocks so tiny grids are useful at all.

Numbers: 92×92 cells at 300 µm → 15.2 KB payload in 30×30 mm (×5 QR-40);
150×150 at 200 µm → 35.9 KB. The limitation is stated plainly: exact
lookup survives pristine rasters, not cameras — that's the roadmap.

https://github.com/kostyk348/microform-code

## habr (черновик вступления)

Бумага — недооценённый носитель: она не требует питания, читается
через десять лет и не боится форматов. Microform Code — плотный
двумерный код для печати: 17 бит на ячейку (эллипс + микро-точка),
детерминированный кодбук на 2¹⁷ визуально уникальных тайлов и
RS(255,223) со стёртиями. Стикер 30×30 мм хранит 15 КБ — впятеро больше
QR того же размера. Версия 1.0.0 — полный Rust-крейт с двумя
зависимостями вместо питоновского прототипа: точный lookup вместо
эвристик, масштабо-осведомлённое декодирование и честный демо-стенд,
который показывает и границы подхода.

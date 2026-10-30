//! Microform Code — CLI (hand-rolled arg parsing, zero extra deps).
//!
//!   microform encode [--data STR | --file PATH] [-o OUT.png]
//!                    [--grid N] [--cell-size UM] [--dpi N]
//!   microform decode IMG.png [--grid N] [-o OUT.bin] [--text]
//!   microform capacity [--grid N]
//!   microform codebook
//!   microform perceptual [--scales 7,14,28]
//!   microform demo [-o OUT.png]

use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use microform::codebook;
use microform::codec;
use microform::decode::{decode_image, DecodeFailure};
use microform::demo;
use microform::perceptual;
use microform::pngio::{read_gray, write_gray, GrayImage};
use microform::render::{generate, physical_mm, RenderSpec};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = args.first() else {
        print_usage();
        return ExitCode::from(2);
    };
    let flags = parse_flags(&args[1..]);
    match cmd.as_str() {
        "encode" => cmd_encode(&flags),
        "decode" => cmd_decode(&flags),
        "capacity" => cmd_capacity(&flags),
        "codebook" => cmd_codebook(),
        "perceptual" => cmd_perceptual(&flags),
        "demo" => cmd_demo(&flags),
        "--help" | "-h" | "help" => {
            print_usage();
            ExitCode::SUCCESS
        }
        "--version" | "-V" => {
            println!("microform {}", microform::VERSION);
            ExitCode::SUCCESS
        }
        other => {
            eprintln!("unknown command: {other}\n");
            print_usage();
            ExitCode::from(2)
        }
    }
}

type Flags = Vec<(String, String)>;

fn parse_flags(args: &[String]) -> Flags {
    let mut out = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if let Some(name) = a.strip_prefix("--").or_else(|| {
            // short options: -o VALUE
            let s = a.strip_prefix('-')?;
            (s.len() == 1 && s.chars().next().is_some_and(|c| c.is_ascii_alphabetic())).then_some(s)
        }) {
            // --name value | --name=value | bare flag
            if let Some((n, v)) = name.split_once('=') {
                out.push((n.to_string(), v.to_string()));
            } else if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                out.push((name.to_string(), args[i + 1].clone()));
                i += 1;
            } else {
                out.push((name.to_string(), String::new()));
            }
        } else {
            out.push((String::new(), a.clone())); // positional
        }
        i += 1;
    }
    out
}

fn flag<'a>(flags: &'a [(String, String)], name: &str) -> Option<&'a str> {
    flags
        .iter()
        .find(|(n, _)| n == name)
        .map(|(_, v)| v.as_str())
}

fn flag_num<T: std::str::FromStr>(flags: &[(String, String)], name: &str, default: T) -> T {
    flag(flags, name)
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn print_usage() {
    println!(
        "microform {} — a dense 2D code on paper (17 bits/cell, RS(255,223))

USAGE:
    microform encode [--data STR | --file PATH] [-o OUT.png]
                     [--grid N] [--cell-size UM] [--dpi N]
    microform decode IMG.png [--grid N] [-o OUT.bin] [--text]
    microform capacity [--grid N]
    microform codebook
    microform perceptual [--scales 7,14,28]
    microform demo [-o OUT.png]

Defaults: grid 92, cell 300 µm, 1200 dpi → 14 px/cell, ~15.2 KB payload.",
        microform::VERSION
    );
}

fn cell_px_from(flags: &Flags) -> u32 {
    let um: f64 = flag_num(flags, "cell-size", 300.0);
    let dpi: f64 = flag_num(flags, "dpi", 1200.0);
    ((um * dpi / 25400.0).round() as u32).max(4)
}

fn cmd_encode(flags: &Flags) -> ExitCode {
    let data: Vec<u8> = if let Some(f) = flag(flags, "file") {
        match std::fs::read(f) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("error: cannot read --file {f}: {e}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        flag(flags, "data")
            .unwrap_or("Microform Code 17-bit/cell demo at 300 µm.")
            .as_bytes()
            .to_vec()
    };

    let grid: u32 = flag_num(flags, "grid", 92);
    let spec = RenderSpec {
        grid_w: grid,
        grid_h: grid,
        cell_px: cell_px_from(flags),
    };

    let book = microform::codebook::get_for_scale(spec.cell_px);
    let cap = codec::capacity_info((grid * grid) as usize, book.bits);
    if data.len() > cap.payload_bytes {
        eprintln!(
            "error: {} bytes exceed the {}-byte payload of a {}×{} grid \
             (use a bigger --grid, see `microform capacity`)",
            data.len(),
            cap.payload_bytes,
            grid,
            grid
        );
        return ExitCode::FAILURE;
    }

    let t0 = Instant::now();
    let img = generate(spec, &data);
    let out = Path::new(flag(flags, "o").unwrap_or("microform.png"));
    if let Err(e) = write_gray(out, &img) {
        eprintln!("error: {e}");
        return ExitCode::FAILURE;
    }

    println!(
        "[microform] {}×{} cells, {:.1}×{:.1} mm at {} µm/cell, {} px/cell",
        grid,
        grid,
        physical_mm(spec, flag_num(flags, "cell-size", 300.0)),
        physical_mm(spec, flag_num(flags, "cell-size", 300.0)),
        flag_num::<f64>(flags, "cell-size", 300.0),
        spec.cell_px
    );
    println!(
        "  capacity: {:.1} KB raw, {:.1} KB payload ({} RS blocks) | data: {} bytes",
        cap.aligned_bytes as f64 / 1024.0,
        cap.payload_bytes as f64 / 1024.0,
        cap.rs_blocks,
        data.len()
    );
    println!(
        "  saved {} ({}×{} px) in {:.1?}",
        out.display(),
        img.w,
        img.h,
        t0.elapsed()
    );
    ExitCode::SUCCESS
}

fn cmd_decode(flags: &Flags) -> ExitCode {
    let Some(path) = flags.iter().find(|(n, _)| n.is_empty()).map(|(_, v)| v) else {
        eprintln!("error: decode needs an image path");
        return ExitCode::from(2);
    };
    let img: GrayImage = match read_gray(Path::new(path)) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let grid = flag(flags, "grid").and_then(|v| v.parse::<u32>().ok().map(|g| (g, g)));

    let t0 = Instant::now();
    match decode_image(&img, grid) {
        Ok(d) => {
            println!(
                "[microform] {}×{} px, pitch {:.1} px/cell → grid {}×{}, {} cells, {} missed",
                img.w,
                img.h,
                d.info.pitch_px,
                d.info.grid_w,
                d.info.grid_h,
                d.info.cells,
                d.info.misses
            );
            if flag(flags, "o").is_some() {
                let out = Path::new(flag(flags, "o").unwrap());
                if let Err(e) = std::fs::write(out, &d.data) {
                    eprintln!("error: cannot write {out:?}: {e}");
                    return ExitCode::FAILURE;
                }
                println!("  payload: {} bytes → {}", d.data.len(), out.display());
            } else if flag(flags, "text").is_some() {
                println!("  payload: {} bytes", d.data.len());
                println!("{}", String::from_utf8_lossy(&d.data));
            } else {
                match std::str::from_utf8(&d.data) {
                    Ok(s) => {
                        println!("  payload: {} bytes (utf-8)", d.data.len());
                        println!("{s}");
                    }
                    Err(_) => {
                        println!(
                            "  payload: {} bytes (binary; use -o OUT.bin to save)",
                            d.data.len()
                        );
                    }
                }
            }
            println!("  decoded in {:.1?}", t0.elapsed());
            ExitCode::SUCCESS
        }
        Err(e) => {
            if let DecodeFailure::Codec(_) = e {
                println!(
                    "  image {}×{} px: cells located but RS/frame failed: {e}",
                    img.w, img.h
                );
            } else {
                println!("  {e}");
            }
            ExitCode::FAILURE
        }
    }
}

fn cmd_capacity(flags: &Flags) -> ExitCode {
    println!("cell, µm | grid | size, mm | raw KB | payload KB | RS blocks");
    println!("---|---|---|---|---|---");
    let rows = demo::capacity_rows();
    let custom: Option<u32> = flag(flags, "grid").and_then(|v| v.parse().ok());
    if let Some(g) = custom {
        let cap = codec::capacity_info((g * g) as usize, microform::layout::BITS_PER_CELL);
        println!(
            "custom | {g} | — | {:.1} | {:.1} | {}",
            cap.aligned_bytes as f64 / 1024.0,
            cap.payload_bytes as f64 / 1024.0,
            cap.rs_blocks
        );
    }
    for (label, grid, mm, raw, payload) in rows {
        println!(
            "{label} | {grid}×{grid} | {mm:.1} | {:.1} | {:.1} | —",
            raw as f64 / 1024.0,
            payload as f64 / 1024.0,
        );
    }
    ExitCode::SUCCESS
}

fn cmd_codebook() -> ExitCode {
    let t0 = Instant::now();
    let stats = codebook::full_stats();
    let build = t0.elapsed();
    let cb = codebook::get();
    println!(
        "codebook: {} distinct patterns of {} parameter combos ({:.2} bits/cell effective)",
        stats.n_distinct, stats.param_combos, stats.bits_per_cell_effective
    );
    println!(
        "  17-bit codec uses the first {} patterns ({} entries in the live map)",
        stats.n_used,
        cb.len()
    );
    println!(
        "  full build: {:.2?} (deterministic, fingerprint {:016x})",
        build,
        cb.fingerprint()
    );
    ExitCode::SUCCESS
}

fn cmd_perceptual(flags: &Flags) -> ExitCode {
    let scales: Vec<u32> = flag(flags, "scales")
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![7, 14, 28]);

    println!("perceptual audit — how close can two usable tiles look\n");
    let canonical = codebook::get();
    let t0 = Instant::now();

    let fmt = |x: f32| {
        if x.is_finite() {
            format!("{x:.3}")
        } else {
            "—".to_string()
        }
    };

    for s in scales {
        let s = s.clamp(4, 32);
        let book = codebook::get_for_scale(s);
        let a = perceptual::audit_scale(&book);
        println!(
            "scale {} px/cell | usable {} codes | {} bits/cell",
            a.cell_px, a.usable, a.bits
        );
        match a.closest {
            Some((ca, cb)) => println!(
                "  closest pair: MAE {:.3}/255 — codes {ca} ↔ {cb} ({})",
                a.min_mae,
                perceptual::describe_delta(canonical, ca, cb)
            ),
            None => println!("  closest pair: none found by the hash net"),
        }
        println!(
            "  fragile pairs: L∞≤2: {}, L∞≤4: {}, L∞≤8: {}",
            a.linf_le_2, a.linf_le_4, a.linf_le_8
        );
        println!(
            "  verdict: {}",
            if a.pass() {
                "PASS — no fragile pairs; scan noise degrades to erasures, not wrong reads"
            } else {
                "WARN — fragile pairs exist; RS(255,223) absorbs the residual confusion"
            }
        );
        println!();
    }

    let margins = perceptual::neighbour_margins(canonical);
    println!(
        "one-step param margins at the canonical {} px (min MAE per one-step neighbour):",
        microform::layout::CELL_PX
    );
    for (i, name) in perceptual::PARAM_NAMES.iter().enumerate() {
        println!("  {name:<11} {}", fmt(margins[i]));
    }

    println!("\naudit took {:.1?}", t0.elapsed());
    ExitCode::SUCCESS
}

fn cmd_demo(flags: &Flags) -> ExitCode {
    println!(
        "microform {} demo — honest checks, loopback\n",
        microform::VERSION
    );
    let t0 = Instant::now();
    let out = flag(flags, "o").map(Path::new);
    let checks = demo::run(out);
    let mut failed = 0;
    for c in &checks {
        println!(
            "  [{}] {} — {}",
            if c.passed { "PASS" } else { "FAIL" },
            c.name,
            c.detail
        );
        if !c.passed {
            failed += 1;
        }
    }
    println!(
        "\nитог: проверок {}, провалено {} ({:.1?})",
        checks.len(),
        failed,
        t0.elapsed()
    );
    if failed == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

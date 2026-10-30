//! CLI integration tests: the binary is the user-facing contract.

use std::process::Command;

fn run(args: &[&str]) -> (String, String, bool) {
    let out = Command::new(env!("CARGO_BIN_EXE_microform"))
        .args(args)
        .output()
        .expect("spawn microform");
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        out.status.success(),
    )
}

#[test]
fn encode_decode_roundtrip_via_files() {
    let dir = std::env::temp_dir().join(format!("mcf-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let png = dir.join("code.png");

    let (out, err, ok) = run(&[
        "encode",
        "--data",
        "CLI roundtrip: 17 bits/cell on paper.",
        "-o",
        png.to_str().unwrap(),
        "--grid",
        "24",
    ]);
    assert!(ok, "encode failed: {err}");
    assert!(png.exists(), "no file: {out}");

    let (out, _err, ok) = run(&["decode", png.to_str().unwrap(), "--text"]);
    assert!(ok);
    assert!(
        out.contains("CLI roundtrip: 17 bits/cell on paper."),
        "stdout: {out}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn capacity_and_codebook_and_demo() {
    let (out, _, ok) = run(&["capacity"]);
    assert!(ok);
    assert!(out.contains("15.2"), "92×92 payload must appear: {out}");

    let (_, _, ok) = run(&["codebook"]);
    assert!(ok);

    let (out, _, ok) = run(&["demo"]);
    assert!(ok, "demo must pass");
    assert!(
        out.contains("провалено 0") || out.contains("failed: 0"),
        "{out}"
    );
}

#[test]
fn oversize_payload_is_rejected_with_advice() {
    let big = "x".repeat(20_000);
    let (_, err, ok) = run(&["encode", "--data", &big, "-o", "/tmp/mcf-never.png"]);
    assert!(!ok, "20 KB must not fit a 92×92 grid");
    assert!(err.contains("exceed"), "stderr: {err}");
}

#[test]
fn version_and_usage() {
    let (out, _, ok) = run(&["--version"]);
    assert!(ok && out.contains("microform 1."), "{out}");
    let (_, _, ok) = run(&[]);
    assert!(!ok, "no args → usage + exit 2");
}

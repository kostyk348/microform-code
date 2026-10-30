# Contributing

Thanks for looking at Microform Code!

## The rules of the house

1. **Determinism is a feature.** Any change to pattern rendering,
   the codebook enumeration order, the padding PRNG or the framing
   changes the format and every printed code. Such changes need a
   semver-major bump and a note in CHANGELOG.md.
2. **Honest claims only.** If a test does not back a number, the number
   does not go into the README. The demo exists to show boundaries, not
   to hide them.
3. **Dependencies are costly.** The crate has two (`png`,
   `reed-solomon`). A new dependency needs a strong justification.

## Workflow

- Branch from `master`, conventional commits (English), one logical
  change per PR.
- `cargo fmt`, `cargo clippy --all-targets -- -D warnings` and
  `cargo test --release` must pass locally before the PR; CI enforces
  the same.
- New features need tests. Decode-path bugs need a test that fails
  without the fix.

## Where to help

See the roadmap at the bottom of [LIMITATIONS.md](LIMITATIONS.md).
The camera/perceptual-matching item is the one that matters most.

## Reporting bugs

Include: the exact command, the image (if decodable/attachable), the
`microform --version`, and the full output. For decode failures with a
known grid, add the output of the same command with `--grid`.

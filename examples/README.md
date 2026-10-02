# examples/

Everything here is **opt-in**: nothing in this folder runs during `cargo build`,
`cargo test` or `cargo bench`.

## Profiling driver

- `prof.rs` — reproduces the `bench/run.py` scenarios **without Python**, so the
  release binary can be sampled with `sample(1)` (macOS) or `perf` (Linux)
  without the PyO3 boundary in the profile. It reuses the crate's test-only
  data generator (`src/gen.rs`) through a small shim, runs the requested
  scenario, and prints a deterministic checksum of the retained ordering so
  runs at different thread counts can be compared byte for byte.

```bash
cargo build --release --example prof
./target/release/examples/prof fwd1600k150 3    # scenario, repetitions
```

Known scenarios: `fwd800`, `fwd1600k150`, `fwd1600k300`, `fwd1600k600`,
`bwd200`, `bwd500i`, `bwd500d`.

The measurement harnesses that drive the README figures live in
[`bench/`](../bench/README.md) (Python, through the binding) and in
[`scripts/`](../scripts) (figure generators).

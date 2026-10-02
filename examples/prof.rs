//! Profiling driver: reproduces the `bench/run.py` scenarios without Python,
//! so the release binary can be sampled with `sample(1)` / `perf`.
//!
//! Usage: `cargo build --release --example prof && ./target/release/examples/prof fwd1600k150 3`
#![allow(dead_code)]
// `src/gen.rs` is included verbatim; its `generate` intentionally takes the
// whole generator parameter set.
#![allow(clippy::too_many_arguments)]

// Shim so that the verbatim `gen.rs` (which uses `crate::matrix`) compiles.
mod matrix {
    pub use sifters::matrix::DataMatrix;
}

#[path = "../src/gen.rs"]
mod gen;

use gen::{generate, GenKind};
use sifters::greedy::{backward, forward_multiseed, AlgoConfig, EvalMode, RefineCfg};
use sifters::op::{Dataset, Repr};
use std::time::Instant;

fn main() {
    let name = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "fwd1600k150".into());
    let reps: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    // (n, m, direction, k, eval, kmax/kmin)
    let (n, m, forward, k, eval) = match name.as_str() {
        "fwd800" => (500, 800, true, 50, EvalMode::Auto),
        "fwd1600k150" => (500, 1600, true, 150, EvalMode::Auto),
        "fwd1600k300" => (500, 1600, true, 300, EvalMode::Auto),
        "fwd1600k600" => (500, 1600, true, 600, EvalMode::Auto),
        "bwd200" => (600, 200, false, 1, EvalMode::Auto),
        "bwd500i" => (600, 500, false, 200, EvalMode::Inverse),
        "bwd500d" => (600, 500, false, 200, EvalMode::Direct),
        other => panic!("unknown scenario {other}"),
    };

    let mut dm = generate(GenKind::Blocks, n, m, 0.3, 0.0, 8, 4, 1);
    dm.standardize(true);

    let batch: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);

    let cfg = AlgoConfig {
        tol: 1e-10,
        max_iters_cold: 400,
        max_iters_warm: 60,
        max_exact: 0,
        batch,
        num_low: 4,
        eval,
        inverse_max_m: 3000,
    };

    for r in 0..reps {
        let mut ds = Dataset::new(dm.clone(), Repr::Auto, 4096 * 1024 * 1024, 64);
        let t0 = Instant::now();
        let path = if forward {
            let mut noop = |_: &sifters::greedy::StepRecord| true;
            forward_multiseed(&mut ds, k, &cfg, 0, 1, None, &mut noop)
        } else {
            let mut noop = |_: &sifters::greedy::StepRecord| true;
            backward(&mut ds, k, &cfg, None::<RefineCfg>, false, &mut noop)
        };
        // Deterministic checksum of the retained ordering + final value, so that
        // runs at different thread counts can be compared byte for byte.
        let order = path.order();
        let cks: u64 = order
            .iter()
            .enumerate()
            .fold(1469598103934665603u64, |h, (i, &o)| {
                (h ^ (o as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ i as u64)
                    .wrapping_mul(1099511628211)
            });
        println!(
            "{name} rep{r} wall={:.3}s engine={:.3}s steps={} cks={cks:016x}",
            t0.elapsed().as_secs_f64(),
            path.seconds,
            path.steps.len()
        );
    }
}

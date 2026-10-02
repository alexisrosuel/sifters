# Benchmarks `sifters`

Python benchmark harnesses for the engine, via the **native PyO3 binding**
(`import sifters`). No subprocess, no binary: the former command-line interface
has been removed, these scripts replace it.

## Files

| File | Role |
|---|---|
| [`scenarios.py`](scenarios.py) | scenario catalogue (reproduces exactly the old `bench/*.sh`) |
| [`run.py`](run.py) | runs the scenarios, writes JSON + CSV into `bench/out/<label>/` |
| [`compare.py`](compare.py) | compares two runs (A/B "ref vs now") |
| [`exact.py`](exact.py) | benchmarks the exact mode: proof time, evaluations, certified gap |
| [`_paths.py`](_paths.py) | makes `scripts/` importable and re-exports `make_data` |
| [`final_report.md`](final_report.md) | historical measurements (CLI binary, before removal) |

The data generator lives in
[`scripts/compare_baseline.py`](../scripts/compare_baseline.py) (`make_data`) and is
shared with [`scripts/bench_numpy.py`](../scripts/bench_numpy.py).

## Usage

```bash
cd <repository root>
maturin develop --release          # builds the sifters extension

python3 bench/run.py --all                       # all scenarios, label "run"
python3 bench/run.py --scenario fwd400 bwd200    # selection
python3 bench/run.py --all --repeat 3 --threads 1
python3 bench/run.py --all --curve-csv           # + CSV K / lambda_min / subset
```

Outputs in `bench/out/<label>/`: one `<scenario>.json` per scenario,
`summary.csv`, and `<scenario>.csv` with `--curve-csv`.

### A/B (two revisions)

```bash
git checkout <rev_ref> && maturin develop --release
python3 bench/run.py --all --label ref
git checkout <rev_now> && maturin develop --release
python3 bench/run.py --all --label now
python3 bench/compare.py bench/out/ref bench/out/now --markdown
```

Like the old `bench/ab.sh`, `compare.py` checks that the recovered subsets are
identical (`OK`) or not (`DIVERGENT`).

## Scenarios

The data use the `blocks` generator with `rho=0.3`, `rho_out=0.0`,
`blocks=8`, `seed=1` (old CLI defaults) and the **exact path**
(`prefilter=False`, `forward_top=0`, `max_exact=0`).

The datasets are now drawn by the numpy `Generator` (PCG64) via
`compare_baseline.make_data`: statistical structures identical to the binary's
old xorshift64\* generator, but **not bit for bit**. The numbers of evaluated
candidates (`exacts`) may therefore differ slightly from
`final_report.md`, which remains a historical document.

| Scenario | N | M | Direction | Bounds | Options |
|---|---|---|---|---|---|
| `fwd400` | 500 | 400 | forward | kmax=50 | |
| `fwd800` | 500 | 800 | forward | kmax=50 | |
| `fwd3200` | 500 | 3200 | forward | kmax=50 | |
| `fwd1600k150` | 500 | 1600 | forward | kmax=150 | |
| `bwd200` | 600 | 200 | backward | kmin=1 | |
| `bwd500d` | 600 | 500 | backward | kmin=200 | `eval="direct"` |
| `bwd500i` | 600 | 500 | backward | kmin=200 | `eval="inverse"` |
| `bwd500c` | 600 | 500 | backward | kmin=200 | `eval="direct"`, `iters_warm=300` |
| `bwd500def` | 600 | 500 | backward | kmin=200 | |

## Exact mode

`bench/exact.py` sweeps `M`, `K` and the data structure for
`sifters.select(..., exact=True)`:

```bash
python3 bench/exact.py                       # writes bench/out/exact.json
python3 bench/exact.py --time-budget 60 --ms 20 40 60 80 --ks 4 6 8
python3 bench/exact.py --figure              # + docs/img/fig_exact.png (matplotlib)
```

It reports, per cell, the greedy baseline time, the exact search time, the
number of `lambda_min` evaluations, whether optimality was **proven** within the
budget, the certified gap otherwise, and the `lambda_min` gain over the greedy
incumbent. The exact search is seeded by `forward_seeds=8`.

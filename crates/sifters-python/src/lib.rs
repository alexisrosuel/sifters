//! Extension native `sifters._sifters` (PyO3).
//!
//! This module is the low-level layer of the Python package: a single entry
//! point, [`run`], which takes the data matrix as a `float64` buffer, runs the
//! greedy walk of the `sifters` crate and returns a [`Selection`] object holding
//! the complete nested family.
//!
//! All the computation runs **without the GIL** ([`Python::detach`]): other
//! Python threads can work during a selection, and the `rayon` pool uses every
//! core.  The optional `progress` callback is called **with** the GIL, once per
//! step, and may interrupt the walk; POSIX signals (`Ctrl-C`) are checked at the
//! same moment.
//!
//! The `sifters` package (see `python/sifters/__init__.py`) adds the Python
//! ergonomics on top of this layer: list/DataFrame conversion, automatic choice
//! of the walk direction, type stubs and documentation.

// PyO3 generates `unsafe` code (FFI): `forbid(unsafe_code)` is therefore not
// possible here. No `unsafe` is written by hand in this crate.
//
// Accepted numeric idioms (see `src/lib.rs` for the rationale).
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_debug_implementations)]

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::Instant;

use pyo3::buffer::PyBuffer;
use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyModule;

use sifters::greedy::{self, AlgoConfig, Direction, EvalMode, PathResult, RefineCfg, StepRecord};
use sifters::matrix::DataMatrix;
use sifters::op::{Dataset, Repr};

create_exception!(
    _sifters,
    SiftersError,
    PyRuntimeError,
    "Error raised by the sifters numerical engine."
);

/// Version of the native module (taken from the crate manifest, so it always
/// matches the wheel built by maturin).
const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Width of the forward prefilter enabled by `prefilter=true`: only the 16 best
/// candidates (secular score) are evaluated exactly. `forward_top` selects
/// another width.
const PREFILTER_TOP: usize = 16;

// ---------------------------------------------------------------------------
// Resultats exposes a Python
// ---------------------------------------------------------------------------

/// One step of the greedy walk (removal, addition or local swap).
///
/// Indices are given in the numbering of the input matrix columns, before any
/// removal of degenerate columns.
#[pyclass(frozen, module = "sifters._sifters", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct Step {
    /// Size of the set before the step.
    #[pyo3(get)]
    pub k_before: usize,
    /// Size of the set after the step.
    #[pyo3(get)]
    pub k_after: usize,
    /// Variable ajoutee, retiree ou sortante (indice d'origine).
    #[pyo3(get)]
    pub index: usize,
    /// `lambda_min` of the new set (warm Ritz value).
    #[pyo3(get)]
    pub lambda_min: f64,
    /// Recalcul a froid strict de `lambda_min` (`nan` si non demande).
    #[pyo3(get)]
    pub lambda_verified: f64,
    /// Upper bound of the retained candidate.
    #[pyo3(get)]
    pub upper_bound: f64,
    /// Rayleigh bound alone, for the retained candidate.
    #[pyo3(get)]
    pub rayleigh_bound: f64,
    /// Candidats examines exactement.
    #[pyo3(get)]
    pub exact_evals: usize,
    /// Candidates available at this step.
    #[pyo3(get)]
    pub candidates: usize,
    /// True when the greedy optimality of this step is certified.
    #[pyo3(get)]
    pub certified: bool,
    /// Lanczos residual of the retained value.
    #[pyo3(get)]
    pub residual: f64,
    /// Lanczos iterations accumulated over the step.
    #[pyo3(get)]
    pub lanczos_iters: usize,
}

impl Step {
    fn from_record(rec: &StepRecord, keep_map: &[usize]) -> Self {
        Self {
            k_before: rec.k_before,
            k_after: rec.k_after,
            index: map_index(rec.changed_orig, keep_map),
            lambda_min: rec.lambda,
            lambda_verified: rec.lambda_verified,
            upper_bound: rec.upper,
            rayleigh_bound: rec.rayleigh,
            exact_evals: rec.exact_evals,
            candidates: rec.candidates,
            certified: rec.certified,
            residual: rec.residual,
            lanczos_iters: rec.iters,
        }
    }
}

#[pymethods]
impl Step {
    fn __repr__(&self) -> String {
        let kind = match self.k_after.cmp(&self.k_before) {
            std::cmp::Ordering::Less => "remove",
            std::cmp::Ordering::Greater => "add",
            std::cmp::Ordering::Equal => "swap",
        };
        format!(
            "Step({kind} index={}, K {} -> {}, lambda_min={:.9}, certified={})",
            self.index, self.k_before, self.k_after, self.lambda_min, self.certified
        )
    }
}

/// Complete result of a selection: nested family + curve + diagnostics.
///
/// Indices (`subset`, `order`, `initial_subset`, `steps[i].index`) always refer
/// always refer to the columns of the input matrix: the degenerate columns
/// dropped by standardization are listed in :attr:`dropped`.
#[pyclass(frozen, module = "sifters._sifters", skip_from_py_object)]
#[derive(Debug)]
pub struct Selection {
    path: PathResult,
    keep_map: Vec<usize>,
    steps: Vec<Py<Step>>,
    k: usize,
    subset: Vec<usize>,
    lambda_min: f64,
    lambda_residual: f64,
    n: usize,
    m_total: usize,
    dropped: Vec<usize>,
    seconds: f64,
    path_seconds: f64,
    load_seconds: f64,
    representation: &'static str,
    eval: &'static str,
    proved_optimal: bool,
    gap_certified: f64,
    exact_evals: usize,
}

#[pymethods]
impl Selection {
    /// Direction of the walk: `"backward"` or `"forward"`.
    #[getter]
    fn direction(&self) -> &'static str {
        match self.path.direction {
            Direction::Backward => "backward",
            Direction::Forward => "forward",
        }
    }

    /// Size of the retained set (:attr:`subset`).
    #[getter]
    fn k(&self) -> usize {
        self.k
    }

    /// Retained variables (original indices, increasing).
    #[getter]
    fn subset(&self) -> Vec<usize> {
        self.subset.clone()
    }

    /// `lambda_min` of the retained subset (cold-revalidated when `verify`).
    #[getter]
    fn lambda_min(&self) -> f64 {
        self.lambda_min
    }

    /// Residu de Lanczos associe a :attr:`lambda_min`.
    #[getter]
    fn lambda_residual(&self) -> f64 {
        self.lambda_residual
    }

    /// `{K: lambda_min}` curve of the whole family, with increasing keys.
    #[getter]
    fn curve(&self) -> BTreeMap<usize, f64> {
        self.path.curve().into_iter().collect()
    }

    /// Duree totale de l'appel (s), normalisation et verification incluses.
    #[getter]
    fn seconds(&self) -> f64 {
        self.seconds
    }

    /// Duration of the greedy walk alone (s).
    #[getter]
    fn path_seconds(&self) -> f64 {
        self.path_seconds
    }

    /// Duration of the load + standardization + correlation construction.
    #[getter]
    fn load_seconds(&self) -> f64 {
        self.load_seconds
    }

    /// Order of the removed (backward) or added (forward) variables, as original indices.
    #[getter]
    fn order(&self) -> Vec<usize> {
        self.path
            .order()
            .iter()
            .map(|&i| map_index(i, &self.keep_map))
            .collect()
    }

    /// Starting subset (original indices).
    #[getter]
    fn initial_subset(&self) -> Vec<usize> {
        self.path
            .initial_subset
            .iter()
            .map(|&i| map_index(i, &self.keep_map))
            .collect()
    }

    /// Number of observations (rows of the input matrix).
    #[getter]
    fn n_observations(&self) -> usize {
        self.n
    }

    /// Number of variables of the input matrix.
    #[getter]
    fn m_total(&self) -> usize {
        self.m_total
    }

    /// Number of actually usable variables (non-degenerate columns).
    #[getter]
    fn m_usable(&self) -> usize {
        self.keep_map.len()
    }

    /// Columns dropped for zero variance (original indices).
    #[getter]
    fn dropped(&self) -> Vec<usize> {
        self.dropped.clone()
    }

    /// Steps of the walk.
    #[getter]
    fn steps<'py>(&self, py: Python<'py>) -> Vec<Py<Step>> {
        self.steps.iter().map(|s| s.clone_ref(py)).collect()
    }

    /// Steps whose greedy optimality is certified.
    #[getter]
    fn certified_steps(&self) -> usize {
        self.path.certified_steps()
    }

    /// Total number of candidates evaluated exactly.
    #[getter]
    fn total_exact_evals(&self) -> usize {
        self.path.total_exact()
    }

    /// Iterations de Lanczos cumulees.
    #[getter]
    fn total_lanczos_iters(&self) -> usize {
        self.path.total_iters()
    }

    /// True when the exact search proved that :attr:`subset` is optimal.
    #[getter]
    fn proved_optimal(&self) -> bool {
        self.proved_optimal
    }

    /// Certified relative optimality gap of :attr:`subset`.
    ///
    /// `0.0` when :attr:`proved_optimal`; `inf` when no exact search was run.
    #[getter]
    fn gap_certified(&self) -> f64 {
        self.gap_certified
    }

    /// Number of `lambda_min` evaluations spent by the exact search.
    #[getter]
    fn exact_evals(&self) -> usize {
        self.exact_evals
    }

    /// Correlation representation in use (`"packed"` or `"implicit"`).
    #[getter]
    fn representation(&self) -> &'static str {
        self.representation
    }

    /// Methode d'evaluation reellement employee (`"direct"` ou `"inverse"`).
    #[getter]
    fn eval(&self) -> &'static str {
        self.eval
    }

    /// Subset of size `k` in the nested family, or `None`.
    ///
    /// Valid sizes range from the end of the walk to its beginning (typically
    /// `kmin..=m_usable` in backward mode, `1..=kmax` in forward mode).
    fn subset_at(&self, k: usize) -> Option<Vec<usize>> {
        self.path.subset_at(k).map(|s| {
            s.into_iter()
                .map(|i| map_index(i, &self.keep_map))
                .collect()
        })
    }

    /// `lambda_min` associated with size `k` of the walk, or `None`.
    fn lambda_at(&self, k: usize) -> Option<f64> {
        self.path.lambda_at(k)
    }

    /// Raw value (without cold revalidation) for size `k`.
    fn lambda_raw_at(&self, k: usize) -> Option<f64> {
        self.path.lambda_raw_at(k)
    }

    /// Same content as a nested dict (JSON serializable).
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let d = pyo3::types::PyDict::new(py);
        d.set_item("direction", self.direction())?;
        d.set_item("k", self.k)?;
        d.set_item("subset", self.subset.clone())?;
        d.set_item("lambda_min", self.lambda_min)?;
        d.set_item("lambda_residual", self.lambda_residual)?;
        d.set_item("curve", self.curve())?;
        d.set_item("order", self.order())?;
        d.set_item("initial_subset", self.initial_subset())?;
        d.set_item("n_observations", self.n)?;
        d.set_item("m_total", self.m_total)?;
        d.set_item("m_usable", self.m_usable())?;
        d.set_item("dropped", self.dropped.clone())?;
        d.set_item("seconds", self.seconds)?;
        d.set_item("path_seconds", self.path_seconds)?;
        d.set_item("load_seconds", self.load_seconds)?;
        d.set_item("representation", self.representation)?;
        d.set_item("eval", self.eval)?;
        d.set_item("certified_steps", self.certified_steps())?;
        d.set_item("total_exact_evals", self.total_exact_evals())?;
        d.set_item("total_lanczos_iters", self.total_lanczos_iters())?;
        d.set_item("proved_optimal", self.proved_optimal)?;
        d.set_item("gap_certified", self.gap_certified)?;
        d.set_item("exact_evals", self.exact_evals)?;
        let steps = pyo3::types::PyList::empty(py);
        for s in &self.steps {
            let b = s.bind(py);
            let sd = pyo3::types::PyDict::new(py);
            sd.set_item("k_before", b.getattr("k_before")?)?;
            sd.set_item("k_after", b.getattr("k_after")?)?;
            sd.set_item("index", b.getattr("index")?)?;
            sd.set_item("lambda_min", b.getattr("lambda_min")?)?;
            sd.set_item("lambda_verified", b.getattr("lambda_verified")?)?;
            sd.set_item("upper_bound", b.getattr("upper_bound")?)?;
            sd.set_item("certified", b.getattr("certified")?)?;
            sd.set_item("exact_evals", b.getattr("exact_evals")?)?;
            steps.append(sd)?;
        }
        d.set_item("steps", steps)?;
        Ok(d)
    }

    fn __len__(&self) -> usize {
        self.k
    }

    fn __repr__(&self) -> String {
        format!(
            "Selection(direction='{}', k={}, lambda_min={:.9}, n={}, m={}, seconds={:.4})",
            self.direction(),
            self.k,
            self.lambda_min,
            self.n,
            self.m_total,
            self.seconds
        )
    }
}

// ---------------------------------------------------------------------------
// Entree principale
// ---------------------------------------------------------------------------

/// Run an E-optimal selection on a 2D `float64` buffer.
///
/// In forward selection (`direction="forward"`), the prefilter is **never**
/// enabled automatically: by default (`prefilter=false`) *every* candidate is
/// evaluated exactly, which realizes the true E-optimal greedy walk. Setting it
/// to `true` evaluates only the `forward_top` (16 by default) best candidates of
/// the secular score: faster on large `M`, but `lambda_min` may lose a few
/// percent (the greedy walk becomes heuristic).
///
/// `forward_seeds > 1` tries several starting variables in forward selection:
/// the historical heuristic first, then the best of the one-step lookahead
/// ranking (`lambda_min` at `k = 2`); the family with the largest final
/// `lambda_min` is retained. See `greedy::forward_multiseed`.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    x,
    *,
    shape = None,
    direction = "backward",
    k = None,
    kmin = None,
    kmax = None,
    verify = true,
    center = true,
    low_rank = 4,
    tol = 1e-10,
    iters_warm = 60,
    iters_cold = 400,
    max_exact = 0,
    batch = 8,
    prefilter = false,
    forward_top = 0,
    forward_first = None,
    forward_seeds = 1,
    exact = false,
    exact_time_s = 30.0,
    exact_max_evals = 0,
    swap_passes = 0,
    swap_top = 8,
    swap_from = None,
    eval = "auto",
    representation = "auto",
    mem_budget_mb = 4096,
    block_rows = 64,
    threads = None,
    progress = None,
))]
fn run(
    py: Python<'_>,
    x: &Bound<'_, PyAny>,
    shape: Option<(usize, usize)>,
    direction: &str,
    k: Option<usize>,
    kmin: Option<usize>,
    kmax: Option<usize>,
    verify: bool,
    center: bool,
    low_rank: usize,
    tol: f64,
    iters_warm: usize,
    iters_cold: usize,
    max_exact: usize,
    batch: usize,
    prefilter: bool,
    forward_top: usize,
    forward_first: Option<usize>,
    forward_seeds: usize,
    exact: bool,
    exact_time_s: f64,
    exact_max_evals: usize,
    swap_passes: usize,
    swap_top: usize,
    swap_from: Option<usize>,
    eval: &str,
    representation: &str,
    mem_budget_mb: usize,
    block_rows: usize,
    threads: Option<usize>,
    progress: Option<Py<PyAny>>,
) -> PyResult<Selection> {
    let t_all = Instant::now();
    let direction = parse_direction(direction)?;
    let eval_mode = parse_eval(eval)?;
    let want_repr = parse_repr(representation)?;
    if let Some(cb) = progress.as_ref() {
        if !cb.bind(py).is_callable() {
            return Err(PyTypeError::new_err("progress must be callable or None"));
        }
    }
    if block_rows == 0 {
        return Err(PyValueError::new_err("block_rows must be >= 1"));
    }
    if tol <= 0.0 || !tol.is_finite() {
        return Err(PyValueError::new_err("tol must be a real number > 0"));
    }

    // ---- data (GIL held: reading the Python buffer) ----
    let t_load = Instant::now();
    let mut dm = matrix_from_python(py, x, shape)?;
    let n = dm.rows;
    let m_total = dm.cols;
    let std = dm.standardize(center);
    let m = dm.cols;
    if m == 0 {
        return Err(SiftersError::new_err("no usable column (zero variance)"));
    }
    let dropped = std.dropped.clone();
    let keep_map: Vec<usize> = (0..m_total).filter(|j| !dropped.contains(j)).collect();

    // ---- walk bounds ----
    let (k_min, k_max) = match direction {
        Direction::Backward => {
            let kk = k.unwrap_or_else(|| kmin.unwrap_or(1));
            if kk < 1 || kk > m {
                return Err(PyValueError::new_err(format!(
                    "k/kmin must be in 1..={m} (usable columns), got {kk}"
                )));
            }
            (kk, m)
        }
        Direction::Forward => {
            let kk = k.unwrap_or_else(|| kmax.unwrap_or(m));
            if kk < 1 || kk > m {
                return Err(PyValueError::new_err(format!(
                    "k/kmax must be in 1..={m} (usable columns), got {kk}"
                )));
            }
            (1, kk)
        }
    };
    if let Some(f) = forward_first {
        if f >= m {
            return Err(PyValueError::new_err(format!(
                "forward_first must be in 0..{m}, got {f}"
            )));
        }
    }

    // Forward prefilter: never enabled automatically. `forward_top` (when
    // non-zero) sets the width; otherwise `prefilter` chooses between top-16 and
    // exhaustive evaluation of every candidate (exact greedy).
    let forward_top_eff = if forward_top > 0 {
        forward_top
    } else if prefilter {
        PREFILTER_TOP
    } else {
        0
    };

    let alg = AlgoConfig {
        tol,
        max_iters_cold: iters_cold,
        max_iters_warm: iters_warm,
        max_exact,
        batch: batch.max(1),
        num_low: low_rank.max(1),
        eval: eval_mode,
        inverse_max_m: 3000,
    };
    let refine = if swap_passes > 0 {
        Some(RefineCfg {
            passes: swap_passes,
            top: swap_top.max(1),
            from_k: swap_from.unwrap_or(k_min),
        })
    } else {
        None
    };
    let budget = mem_budget_mb.saturating_mul(1024 * 1024);
    let load_seconds = t_load.elapsed().as_secs_f64();

    // ---- pool de threads dedie (jamais `build_global`, deja initialise) ----
    let pool = match threads {
        Some(0) | None => None,
        Some(t) => Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .map_err(|e| SiftersError::new_err(format!("rayon pool: {e}")))?,
        ),
    };

    let cb_error: Mutex<Option<PyErr>> = Mutex::new(None);

    // ---- computation without the GIL ----
    let outcome = py.detach(|| -> PyResult<Outcome> {
        let mut on_step = |rec: &StepRecord| -> bool {
            let res = Python::attach(|py| -> PyResult<bool> {
                // Ctrl-C (and any other signal) is still handled during the computation.
                py.check_signals()?;
                let Some(cb) = progress.as_ref() else {
                    return Ok(true);
                };
                let step = Py::new(py, Step::from_record(rec, &keep_map))?;
                let out = cb.bind(py).call1((step,))?;
                // `None` (purely informative callback) keeps going; only an
                // explicitly falsy return interrupts the walk.
                if out.is_none() {
                    Ok(true)
                } else {
                    out.is_truthy()
                }
            });
            match res {
                Ok(keep_going) => keep_going,
                Err(e) => {
                    if let Ok(mut slot) = cb_error.lock() {
                        *slot = Some(e);
                    }
                    false
                }
            }
        };

        let compute = || -> PyResult<Outcome> {
            let mut ds = Dataset::new(dm, want_repr, budget, block_rows);
            let path = match direction {
                Direction::Backward => {
                    greedy::backward(&mut ds, k_min, &alg, refine, verify, &mut on_step)
                }
                Direction::Forward => greedy::forward_multiseed(
                    &mut ds,
                    k_max,
                    &alg,
                    forward_top_eff,
                    forward_seeds,
                    forward_first,
                    &mut on_step,
                ),
            };

            let final_k = path
                .steps
                .last()
                .map(|s| s.k_after)
                .unwrap_or(path.initial_k)
                .min(ds.m())
                .max(1);
            let mut subset = path.subset_at(final_k).unwrap_or_default();

            // Strict revalidation of the retained subset: a single cold Lanczos
            // pass, tight tolerance, no warm start.
            let (mut lambda_min, mut lambda_residual) = if verify && !subset.is_empty() {
                strict_lambda(&ds, &subset, final_k, &alg, budget, block_rows)
            } else {
                (path.lambda_at(final_k).unwrap_or(f64::NAN), f64::NAN)
            };

            // ---- recherche exacte optionnelle (branch and bound) ----
            //
            // The greedy walk provides the incumbent: it is what makes the proof
            // cheap. `exact_at` works on the physical positions of `ds`, so the
            // incumbent is converted and the result
            // converted back to original indices.
            let mut proved_optimal = false;
            let mut gap_certified = f64::INFINITY;
            let mut exact_evals = 0usize;
            if exact {
                let positions: Vec<usize> = subset
                    .iter()
                    .filter_map(|&orig| ds.active.iter().position(|&x| x == orig))
                    .collect();
                let ebudget = sifters::exact::ExactBudget {
                    time_budget_s: exact_time_s,
                    max_evals: exact_max_evals,
                };
                let incumbent = if positions.len() == final_k {
                    Some(&positions)
                } else {
                    None
                };
                let out = sifters::exact::exact_at(
                    &ds.z,
                    final_k,
                    &alg,
                    ebudget,
                    incumbent.map(|v| v.as_slice()),
                );
                exact_evals = out.evaluations;
                proved_optimal = out.proved_optimal;
                gap_certified = out.gap_certified;
                if out.subset.len() == final_k {
                    let mut found: Vec<usize> = out.subset.iter().map(|&p| ds.orig(p)).collect();
                    found.sort_unstable();
                    let (v, r) = strict_lambda(&ds, &found, final_k, &alg, budget, block_rows);
                    subset = found;
                    lambda_min = v;
                    lambda_residual = r;
                }
            }

            // Methode d'evaluation reellement employee.
            let eval_used = match eval_mode {
                EvalMode::Direct => "direct",
                EvalMode::Inverse => "inverse",
                EvalMode::Auto => {
                    if direction == Direction::Backward
                        && ds.is_packed()
                        && ds.m() <= alg.inverse_max_m
                    {
                        "inverse"
                    } else {
                        "direct"
                    }
                }
            };

            Ok(Outcome {
                packed: ds.is_packed(),
                eval: eval_used,
                path,
                final_k,
                subset,
                lambda_min,
                lambda_residual,
                proved_optimal,
                gap_certified,
                exact_evals,
            })
        };

        match &pool {
            Some(p) => p.install(compute),
            None => compute(),
        }
    });
    drop(pool);

    if let Some(e) = cb_error.lock().ok().and_then(|mut s| s.take()) {
        return Err(e);
    }
    let outcome = outcome?;

    // ---- conversion en objets Python ----
    let steps = outcome
        .path
        .steps
        .iter()
        .map(|rec| Py::new(py, Step::from_record(rec, &keep_map)))
        .collect::<PyResult<Vec<_>>>()?;

    Ok(Selection {
        k: outcome.final_k,
        subset: outcome
            .subset
            .iter()
            .map(|&i| map_index(i, &keep_map))
            .collect(),
        lambda_min: outcome.lambda_min,
        lambda_residual: outcome.lambda_residual,
        n,
        m_total,
        dropped,
        seconds: t_all.elapsed().as_secs_f64(),
        path_seconds: outcome.path.seconds,
        load_seconds,
        representation: if outcome.packed { "packed" } else { "implicit" },
        eval: outcome.eval,
        path: outcome.path,
        keep_map,
        steps,
        proved_optimal: outcome.proved_optimal,
        gap_certified: outcome.gap_certified,
        exact_evals: outcome.exact_evals,
    })
}

// ---------------------------------------------------------------------------
// Aides internes
// ---------------------------------------------------------------------------

struct Outcome {
    path: PathResult,
    final_k: usize,
    subset: Vec<usize>,
    lambda_min: f64,
    lambda_residual: f64,
    packed: bool,
    eval: &'static str,
    proved_optimal: bool,
    gap_certified: f64,
    exact_evals: usize,
}

/// Strict `lambda_min` (cold Lanczos, tolerance 1e-13) of the subset `set`.
///
/// `set` gives original column indices; `ds.z` is in its current *physical*
/// order (the greedy walk permutes columns), so the selected columns are located
/// through `ds.active` and gathered explicitly into the head of a fresh dataset.
/// Swapping them into place is not an option: `Dataset::new` resets `active` to
/// the identity, so the permutation would have to be tracked by hand.
fn strict_lambda(
    ds: &Dataset,
    set: &[usize],
    k: usize,
    alg: &AlgoConfig,
    budget: usize,
    block_rows: usize,
) -> (f64, f64) {
    let strict = AlgoConfig {
        tol: 1e-13,
        max_iters_cold: 4000,
        max_iters_warm: 400,
        ..*alg
    };
    let units = |ev: &sifters::lanczos::LanczosOutcome| (ev.value, ev.residual);
    let m = ds.m();
    let rows = ds.z.rows;
    if set.len() != k || k == 0 || m == 0 {
        return (f64::NAN, f64::NAN);
    }
    let mut data: Vec<f64> = Vec::with_capacity(rows * m);
    let mut used = vec![false; m];
    let mut placed = 0usize;
    for &orig in set {
        if let Some(p) = ds.active.iter().position(|&x| x == orig) {
            data.extend_from_slice(ds.z.col(p));
            used[p] = true;
            placed += 1;
        }
    }
    if placed != k {
        return (f64::NAN, f64::NAN);
    }
    for p in 0..m {
        if !used[p] {
            data.extend_from_slice(ds.z.col(p));
        }
    }
    let repr = if ds.is_packed() {
        Repr::Packed
    } else {
        Repr::Implicit
    };
    let check = Dataset::new(
        DataMatrix::from_col_major(rows, m, data),
        repr,
        budget,
        block_rows,
    );
    units(&greedy::initial_eigenpair_public(&check, k, &strict))
}

#[inline]
fn map_index(i: usize, keep_map: &[usize]) -> usize {
    keep_map.get(i).copied().unwrap_or(i)
}

fn parse_direction(s: &str) -> PyResult<Direction> {
    match s {
        "backward" => Ok(Direction::Backward),
        "forward" => Ok(Direction::Forward),
        other => Err(PyValueError::new_err(format!(
            "direction must be 'backward' or 'forward', got '{other}'"
        ))),
    }
}

fn parse_eval(s: &str) -> PyResult<EvalMode> {
    match s {
        "auto" => Ok(EvalMode::Auto),
        "direct" => Ok(EvalMode::Direct),
        "inverse" => Ok(EvalMode::Inverse),
        other => Err(PyValueError::new_err(format!(
            "eval must be 'auto', 'direct' or 'inverse', got '{other}'"
        ))),
    }
}

fn parse_repr(s: &str) -> PyResult<Repr> {
    match s {
        "auto" => Ok(Repr::Auto),
        "packed" => Ok(Repr::Packed),
        "implicit" => Ok(Repr::Implicit),
        other => Err(PyValueError::new_err(format!(
            "repr must be 'auto', 'packed' or 'implicit', got '{other}'"
        ))),
    }
}

/// Read `x` as a `(observations, variables)` matrix of `float64`.
///
/// Accepts any object exposing the buffer protocol (`numpy.ndarray`,
/// `memoryview`, `array.array`, ...); the Python package handles lists and
/// DataFrames upstream.  Non-finite values are rejected.
fn matrix_from_python(
    py: Python<'_>,
    x: &Bound<'_, PyAny>,
    shape: Option<(usize, usize)>,
) -> PyResult<DataMatrix> {
    let buf = PyBuffer::<f64>::get(x).map_err(|_| {
        PyTypeError::new_err(
            "X must expose a float64 buffer (numpy.ndarray, memoryview, \
             array.array...); use sifters.as_matrix(X) to convert lists \
             and DataFrames",
        )
    })?;
    let dims = buf.dimensions();
    let (n, m) = match shape {
        // 1D buffer + explicit shape (row-major): numpy-free path.
        Some((n, m)) => {
            if dims != 1 {
                return Err(PyValueError::new_err(
                    "shape ne peut accompagner qu'un tampon 1D",
                ));
            }
            if buf.item_count() != n * m {
                return Err(PyValueError::new_err(format!(
                    "shape {n}x{m} = {} elements, but the buffer holds {}",
                    n * m,
                    buf.item_count()
                )));
            }
            (n, m)
        }
        None => {
            if dims != 2 {
                return Err(PyValueError::new_err(format!(
                    "X must be 2D (observations, variables), got {dims}"
                )));
            }
            let s = buf.shape();
            (s[0], s[1])
        }
    };
    if n == 0 || m == 0 {
        return Err(PyValueError::new_err(format!("X is empty (shape {n}x{m})")));
    }
    let mut flat = vec![0.0f64; n * m];
    buf.copy_to_slice(py, &mut flat)?;
    if let Some(pos) = flat.iter().position(|v| !v.is_finite()) {
        return Err(SiftersError::new_err(format!(
            "X contains a non-finite value (nan or inf) at row {}, column {}",
            pos / m,
            pos % m
        )));
    }
    Ok(DataMatrix::from_row_major(n, m, &flat))
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

/// Compilation profile of the extension: `"release"` or `"debug"`.
///
/// Timings are meaningless on a `debug` build (measured on the reference
/// machine: about 14x slower than `--release`), and the most natural local
/// install command, `maturin develop` without `--release`, silently produces
/// one.  `python/sifters/__init__.py` warns at import time when this reports
/// `"debug"`, so an accidental debug install cannot quietly invalidate a
/// benchmark.
#[pyfunction]
fn build_profile() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}

#[pymodule]
fn _sifters(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", VERSION)?;
    m.add_function(wrap_pyfunction!(run, m)?)?;
    m.add_function(wrap_pyfunction!(build_profile, m)?)?;
    m.add_class::<Selection>()?;
    m.add_class::<Step>()?;
    m.add("SiftersError", m.py().get_type::<SiftersError>())?;
    Ok(())
}

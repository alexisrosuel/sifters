//! Greedy selection algorithms.
//!
//! # Problem
//!
//! Maximize `lambda_min(R_S)`, the smallest eigenvalue of the correlation
//! matrix of a subset `S` of `K` variables. NP-hard; we build a
//! **nested** family of subsets by certified backward elimination.
//!
//! # Certified cascade
//!
//! Let `R = R_S` of size `k` and `(lambda_l, u_l)_{l<=p}` its `p` smallest
//! eigenpairs. Removing variable `i` gives `R_{-i}`, whose eigenvalues
//! are (Cauchy) interlaced: `lambda_l(R) <= lambda_l(R_{-i}) <= lambda_{l+1}(R)`.
//!
//! ## Bound 1: Rayleigh (always valid, O(1))
//!
//! The vector `u_1` with its component `i` removed, renormalized, is a trial
//! vector for `R_{-i}`; its Rayleigh quotient is, with `y = R u_1` and
//! `rq = u_1^T R u_1`:
//!
//! ```text
//! rho_i = ( rq - 2 u_1(i) y(i) + u_1(i)^2 ) / ( 1 - u_1(i)^2 )  >=  lambda_min(R_{-i})
//! ```
//!
//! This bound is `O(1)` per candidate but **loose** (it reaches `lambda_1 + m_i`,
//! where `m_i = u_1(i)^2` is the eigenvector mass): it does not allow pruning.
//!
//! ## Bound 2: `p`-dimensional spectral Temple (tight)
//!
//! The eigenvalues of `R_{-i}` are exactly the roots of the secular
//! equation (KKT conditions of `min x^T R x` under `x_i = 0`):
//!
//! ```text
//! S_i(mu) = sum_l (u_l(i))^2 / (lambda_l - mu) = 0,    root in (lambda_1, lambda_2)
//! ```
//!
//! Separating the first `p` terms and lower-bounding the tail by Cauchy-Schwarz,
//! with `m_i = u_1(i)^2`, `B_i = sum_{l=2..p} u_l(i)^2/(lambda_l - lambda_1)`,
//! `T_i = 1 - sum_{l<=p} u_l(i)^2` and
//! `Sig_i = (1 - lambda_1) - sum_{l=2..p} u_l(i)^2 (lambda_l - lambda_1)`
//! (exact identities `sum_l u_l(i)^2 = 1` and `sum_l u_l(i)^2 lambda_l = R_ii = 1`),
//! we obtain
//!
//! ```text
//! G_i = B_i + T_i^2 / Sig_i        (tail lower bound)
//! ub_i = lambda_1 + m_i / G_i      >= lambda_min(R_{-i})
//! ```
//!
//! much tighter than `rho_i` (it reduces to Rayleigh for `p = 1`). We take
//! `min(rho_i, ub_i)`. The Lanczos residuals are subtracted from `T_i` and added to
//! `Sig_i` (conservative margin).
//!
//! ## Selection
//!
//! Candidates are sorted by increasing bound and evaluated exactly (warm Lanczos)
//! until the best value reached exceeds the bound of the next candidate:
//! **no unevaluated candidate can then do better**, so the step is
//! certified optimal (up to tolerance and margin).

use crate::inverse::{InverseSym, NegInvSubOp};
use crate::lanczos::{smallest_eigenpair, smallest_eigenpair_constrained, LanczosOutcome};
use crate::num::dot;
use crate::op::SymOp;
use crate::op::{z_dot_all, z_loading, BorderedHeadOp, Dataset, GatheredZOp, SubOp};
use crate::packed::PackedSym;
use rayon::prelude::*;
use std::time::Instant;

/// Direction of traversal of the nested family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    /// Backward elimination: `M -> k_min`.
    Backward,
    /// Forward selection: `1 -> k_max`.
    Forward,
}

/// Method for evaluating `lambda_min` of a candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvalMode {
    /// Automatic detection (inverse if the correlation is positive definite and
    /// `M` reasonable, otherwise direct matvec).
    Auto,
    /// Lanczos on `R` directly.
    Direct,
    /// Lanczos on the maintained inverse: `lambda_min(R_{-i}) = -1 / theta_min(-R_{-i}^{-1})`.
    /// The relative gaps in the lower spectrum are strongly amplified there, hence
    /// convergence in a few iterations instead of several tens.
    Inverse,
}

/// Numerical parameters.
#[derive(Clone, Copy, Debug)]
pub struct AlgoConfig {
    /// Relative Lanczos tolerance.
    pub tol: f64,
    /// Max iterations for a cold start.
    pub max_iters_cold: usize,
    /// Max iterations for a warm start.
    pub max_iters_warm: usize,
    /// Max number of candidates evaluated exactly per step (0 = unlimited).
    pub max_exact: usize,
    /// Size of the parallel batches.
    pub batch: usize,
    /// Number `p` of eigenpairs used for the Temple bound.
    pub num_low: usize,
    /// Method for evaluating candidates.
    pub eval: EvalMode,
    /// Threshold on `M` for inverse evaluation in `Auto` mode.
    pub inverse_max_m: usize,
}

impl Default for AlgoConfig {
    fn default() -> Self {
        Self {
            tol: 1e-10,
            max_iters_cold: 400,
            max_iters_warm: 60,
            max_exact: 0,
            batch: 8,
            num_low: 4,
            eval: EvalMode::Auto,
            inverse_max_m: 3000,
        }
    }
}

/// Trace of one step.
#[derive(Clone, Debug)]
pub struct StepRecord {
    /// Size before the step.
    pub k_before: usize,
    /// Size after the step.
    pub k_after: usize,
    /// Original index of the variable entering/leaving.
    pub changed_orig: usize,
    /// `lambda_min` of the new set (warm Ritz value).
    pub lambda: f64,
    /// `lambda_min` recomputed cold strictly (`NaN` if not requested).
    pub lambda_verified: f64,
    /// Bound (Temple/Rayleigh) of the selected candidate.
    pub upper: f64,
    /// Rayleigh bound alone of the selected candidate.
    pub rayleigh: f64,
    /// Candidates examined exactly.
    pub exact_evals: usize,
    /// Total candidates.
    pub candidates: usize,
    /// True if the greedy optimality is certified.
    pub certified: bool,
    /// Lanczos residual of the selected candidate.
    pub residual: f64,
    /// Lanczos iterations accumulated over the step.
    pub iters: usize,
}

/// Full result of a greedy path.
#[derive(Clone, Debug)]
pub struct PathResult {
    /// Direction of the path.
    pub direction: Direction,
    /// Initial size.
    pub initial_k: usize,
    /// Initial `lambda_min` (`1` for a singleton, `NaN` if empty).
    pub initial_lambda: f64,
    /// Residual of the initial computation.
    pub initial_residual: f64,
    /// Successive steps.
    pub steps: Vec<StepRecord>,
    /// Duration of the path (s).
    pub seconds: f64,
    /// Starting subset (`1` index in forward selection, `initial_k` indices
    /// in backward elimination).
    pub initial_subset: Vec<usize>,
}

impl PathResult {
    /// Total number of candidates evaluated exactly.
    pub fn total_exact(&self) -> usize {
        self.steps.iter().map(|s| s.exact_evals).sum()
    }

    /// Number of certified steps.
    pub fn certified_steps(&self) -> usize {
        self.steps.iter().filter(|s| s.certified).count()
    }

    /// Lanczos iterations accumulated.
    pub fn total_iters(&self) -> usize {
        self.steps.iter().map(|s| s.iters).sum()
    }

    /// Sequence of `(K, lambda_min)`.
    pub fn curve(&self) -> Vec<(usize, f64)> {
        let mut out = Vec::with_capacity(self.steps.len() + 1);
        out.push((self.initial_k, self.initial_lambda));
        for s in &self.steps {
            out.push((
                s.k_after,
                if s.lambda_verified.is_nan() {
                    s.lambda
                } else {
                    s.lambda_verified
                },
            ));
        }
        out
    }

    /// Order of the variables removed (backward) or added (forward).
    ///
    /// In forward selection, the subset of size `K` is
    /// `initial_subset ++ order[..K-initial_k]`; in backward elimination it is
    /// `initial_subset` minus `order[..initial_k-K]`.
    pub fn order(&self) -> Vec<usize> {
        self.steps.iter().map(|s| s.changed_orig).collect()
    }

    /// Value of `lambda_min` for a visited size `K`.
    pub fn lambda_at(&self, k: usize) -> Option<f64> {
        if k == self.initial_k {
            return Some(self.initial_lambda);
        }
        self.steps.iter().find(|s| s.k_after == k).map(|s| {
            if s.lambda_verified.is_nan() {
                s.lambda
            } else {
                s.lambda_verified
            }
        })
    }

    /// Raw value (warm Ritz) of the path, without revalidation.
    pub fn lambda_raw_at(&self, k: usize) -> Option<f64> {
        if k == self.initial_k {
            return Some(self.initial_lambda);
        }
        self.steps.iter().find(|s| s.k_after == k).map(|s| s.lambda)
    }

    /// Subset of variables (original indices, sorted) of size `K`.
    pub fn subset_at(&self, k: usize) -> Option<Vec<usize>> {
        match self.direction {
            Direction::Backward => {
                let k_end = self
                    .steps
                    .last()
                    .map(|s| s.k_after)
                    .unwrap_or(self.initial_k);
                if k > self.initial_k || k < k_end {
                    return None;
                }
                let mut set: Vec<usize> = (0..self.initial_k).collect();
                for s in self.steps.iter().take(self.initial_k - k) {
                    if let Some(p) = set.iter().position(|&x| x == s.changed_orig) {
                        set.swap_remove(p);
                    }
                }
                set.sort_unstable();
                Some(set)
            }
            Direction::Forward => {
                if k < self.initial_k || k > self.initial_k + self.steps.len() {
                    return None;
                }
                let mut set: Vec<usize> = self.initial_subset.clone();
                set.extend(
                    self.steps
                        .iter()
                        .take(k - self.initial_k)
                        .map(|s| s.changed_orig),
                );
                set.sort_unstable();
                set.dedup();
                if set.len() == k {
                    Some(set)
                } else {
                    None
                }
            }
        }
    }
}

/// The `p` smallest eigenpairs of a set of size `k`.
#[derive(Clone, Debug)]
pub struct LowSpectrum {
    /// Size of the set.
    pub dim: usize,
    /// Increasing eigenvalues.
    pub values: Vec<f64>,
    /// Associated eigenvectors (norm 1, length `dim`).
    pub vectors: Vec<Vec<f64>>,
    /// Residuals `||R u - lambda u||`.
    pub residuals: Vec<f64>,
}

impl LowSpectrum {
    /// Singleton.
    pub fn singleton() -> Self {
        Self {
            dim: 1,
            values: vec![1.0],
            vectors: vec![vec![1.0]],
            residuals: vec![0.0],
        }
    }

    /// Number of available eigenpairs.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// True if no eigenpair is available.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Smallest eigenvalue.
    pub fn lambda_min(&self) -> f64 {
        self.values.first().copied().unwrap_or(f64::NAN)
    }

    /// Maximal residual.
    pub fn max_residual(&self) -> f64 {
        self.residuals.iter().cloned().fold(0.0f64, f64::max)
    }

    /// Upper bound of `lambda_min(R_{-i})` (see the module docs).
    ///
    /// `ray` is the Rayleigh bound, always valid; we return the minimum of the
    /// two bounds.
    pub fn upper_bound(&self, i: usize, ray: f64) -> f64 {
        let p = self.values.len();
        if p == 0 {
            return f64::INFINITY;
        }
        let lam1 = self.values[0];
        let u1i2 = self.vectors[0][i] * self.vectors[0][i];
        let mut b = 0.0f64;
        let mut mass = u1i2;
        let mut smass = 0.0f64;
        for j in 1..p {
            let u2 = self.vectors[j][i] * self.vectors[j][i];
            let d = (self.values[j] - lam1).max(1e-300);
            b += u2 / d;
            smass += u2 * d;
            mass += u2;
        }
        // Conservative margin: we underestimate the tail mass and overestimate
        // the tail norm, which can only decrease `G_i` and therefore increase the
        // bound (thus remain valid).
        let slack = 2.0 * p as f64 * self.max_residual();
        let t = (1.0 - mass - slack).max(0.0);
        let gap_p = if p > 1 {
            (self.values[p - 1] - lam1).max(0.0)
        } else {
            0.0
        };
        let sig = (1.0 - lam1) - smass - slack * (1.0 + gap_p);
        // The tail term `t^2/sig` is a 0/0 numerical case when the tail mass
        // is zero (case `p = k`): we then drop it, and fall back to Rayleigh
        // if the ratio is not reliable.
        let refined = if t <= 1e-12 {
            lam1 + u1i2 / b.max(1e-300)
        } else if sig > 1e-12 {
            lam1 + u1i2 / (b + t * t / sig)
        } else {
            f64::INFINITY
        };
        refined.min(ray).max(lam1)
    }
}

/// Eigenpair of the smallest eigenvalue of a set of `k` variables
/// (leading block), with a cold start.
pub fn initial_eigenpair_public(ds: &Dataset, k: usize, cfg: &AlgoConfig) -> LanczosOutcome {
    if k == 0 {
        return LanczosOutcome {
            value: f64::NAN,
            vector: Vec::new(),
            residual: 0.0,
            iters: 0,
            converged: true,
        };
    }
    if k == 1 {
        return LanczosOutcome {
            value: 1.0,
            vector: vec![1.0],
            residual: 0.0,
            iters: 1,
            converged: true,
        };
    }
    let mut op = SubOp::head(ds, k);
    smallest_eigenpair(&mut op, None, cfg.tol, cfg.max_iters_cold)
}

/// Eigenpair of the smallest eigenvalue of a set of `k` variables,
/// exploiting the inverse if it is available.
pub fn initial_eigenpair_inv(
    ds: &Dataset,
    k: usize,
    cfg: &AlgoConfig,
    inv: Option<&InverseSym>,
) -> LanczosOutcome {
    if k <= 1 {
        return initial_eigenpair_public(ds, k, cfg);
    }
    match inv {
        Some(iv) => {
            let mut op = NegInvSubOp::new(iv, k, None);
            let o = smallest_eigenpair(&mut op, None, cfg.tol, cfg.max_iters_cold);
            LanczosOutcome {
                value: to_lambda(o.value, true),
                ..o
            }
        }
        None => initial_eigenpair_public(ds, k, cfg),
    }
}

/// Computes the `p` smallest eigenpairs of the current set.
///
/// `known` reuses the eigenpair obtained by the exact evaluation of the winner of
/// the previous step; `seeds[j]` serves as a warm start for eigenpair `j`.
pub fn low_spectrum(
    ds: &Dataset,
    k: usize,
    p: usize,
    seeds: &[Vec<f64>],
    known: Option<&LanczosOutcome>,
    cfg: &AlgoConfig,
    cold: bool,
) -> LowSpectrum {
    low_spectrum_inv(ds, k, p, seeds, known, cfg, cold, None)
}

/// Largest block size for which the whole spectrum is computed densely.
///
/// The dense solver is `O(k^3)`; beyond a few hundred variables its memory
/// footprint (`k^2` f64) and its cost per step stop competing with the
/// iterative path, which only pays for the eigenpairs it actually needs.
pub const DENSE_SPECTRUM_MAX_K: usize = 1024;

/// Whole low spectrum of the leading `k x k` block of a packed correlation, by
/// dense symmetric eigendecomposition.
///
/// Returns the `k` eigenpairs sorted by increasing eigenvalue, with the *exact*
/// Lanczos residual `||R u_l - lambda_l u_l||` recomputed on the packed operator
/// (the conservative margin of the cascade bounds consumes it).
fn dense_spectrum(pk: &PackedSym, k: usize) -> LowSpectrum {
    let mut a = vec![0.0f64; k * k];
    for i in 0..k {
        for j in 0..=i {
            let v = pk.get(i, j);
            a[i * k + j] = v;
            a[j * k + i] = v;
        }
    }
    let (values, vecs) = crate::dense::eigen_sym_ql(&a, k);
    // The `k` residuals are independent: each is an `O(k^2)` matvec on the packed
    // operator, dispatched in parallel (`collect` preserves the order).
    let pairs: Vec<(Vec<f64>, f64)> = (0..k)
        .into_par_iter()
        .map(|l| {
            let u: Vec<f64> = (0..k).map(|e| vecs[e * k + l]).collect();
            let mut y = vec![0.0f64; k];
            pk.matvec_head(k, &u, &mut y);
            let mut r = 0.0f64;
            for i in 0..k {
                let t = y[i] - values[l] * u[i];
                r += t * t;
            }
            (u, r.sqrt())
        })
        .collect();
    let (vectors, residuals): (Vec<Vec<f64>>, Vec<f64>) = pairs.into_iter().unzip();
    LowSpectrum {
        dim: k,
        values,
        vectors,
        residuals,
    }
}

/// Same, optionally exploiting the inverse `W = R^{-1}` (see [`EvalMode`]).
#[allow(clippy::too_many_arguments)]
pub fn low_spectrum_inv(
    ds: &Dataset,
    k: usize,
    p: usize,
    seeds: &[Vec<f64>],
    known: Option<&LanczosOutcome>,
    cfg: &AlgoConfig,
    cold: bool,
    inv: Option<&InverseSym>,
) -> LowSpectrum {
    if k == 0 {
        return LowSpectrum {
            dim: 0,
            values: Vec::new(),
            vectors: Vec::new(),
            residuals: Vec::new(),
        };
    }
    if k == 1 {
        return LowSpectrum::singleton();
    }
    let p = p.clamp(1, k);
    // Whole-spectrum fast path: with the correlation materialized, a dense
    // symmetric eigensolver on the leading `k x k` block replaces `k`
    // constrained Lanczos runs (both `O(k^3)`, but the dense constant is ~10x
    // smaller and there is no deflation projection). The residuals are then
    // computed explicitly on the packed operator, so the cascade bounds stay
    // valid and become essentially exact (`p = k`).
    if p == k && k <= DENSE_SPECTRUM_MAX_K {
        if let Some(pk) = ds.packed.as_ref() {
            return dense_spectrum(pk, k);
        }
    }
    let iters = if cold {
        cfg.max_iters_cold
    } else {
        cfg.max_iters_warm
    };
    let mut spec = LowSpectrum {
        dim: k,
        values: Vec::new(),
        vectors: Vec::new(),
        residuals: Vec::new(),
    };
    if let Some(kn) = known {
        if kn.vector.len() == k {
            spec.values.push(kn.value);
            spec.vectors.push(kn.vector.clone());
            spec.residuals.push(kn.residual);
        }
    }
    while spec.values.len() < p {
        let j = spec.values.len();
        let seed = if j < seeds.len() && seeds[j].len() == k {
            seeds[j].clone()
        } else {
            crate::lanczos::default_seed(k, (k as u64) << 8 | j as u64)
        };
        // Search restricted to the complement of the vectors already found: the
        // smallest Ritz value then converges to the (j+1)-th eigenvalue.
        let mut out = match inv {
            Some(inv) => {
                let mut op = NegInvSubOp::new(inv, k, None);
                let o = smallest_eigenpair_constrained(
                    &mut op,
                    Some(seed.as_slice()),
                    cfg.tol,
                    iters,
                    &spec.vectors,
                );
                LanczosOutcome {
                    value: to_lambda(o.value, true),
                    ..o
                }
            }
            None => {
                let mut op = SubOp::head(ds, k);
                smallest_eigenpair_constrained(
                    &mut op,
                    Some(seed.as_slice()),
                    cfg.tol,
                    iters,
                    &spec.vectors,
                )
            }
        };
        let mut u = out.vector;
        let nu = crate::num::norm2(&u);
        let value;
        if nu > 1e-12 {
            for x in u.iter_mut() {
                *x /= nu;
            }
            let mut raw = SubOp::head(ds, k);
            let mut au = vec![0.0; k];
            raw.mul(&u, &mut au);
            let rq = dot(&u, &au);
            value = if nu > 0.999 { out.value } else { rq };
            out.value = value;
            crate::num::axpy(-value, &u, &mut au);
            let residual = crate::num::norm2(&au);
            spec.values.push(value);
            spec.vectors.push(u);
            spec.residuals.push(residual);
        } else {
            break;
        }
    }
    spec
}

/// Root of the secular equation truncated to the first `p` eigenpairs,
/// in the interval `(lambda_1, lambda_2)`:
///
/// ```text
/// S_i(mu) = sum_{l<=p} u_l(i)^2 / (lambda_l - mu) = 0
/// ```
///
/// It is an upper bound of `lambda_min(R_{-i})` (the tail terms are positive on
/// the interval, so the truncated root is above the exact root) and
/// above all the vector `x = (R - mu I)^{-1} e_i` is the exact eigenvector of
/// `R_{-i}`: it is an almost perfect warm start for Lanczos.
///
/// Returns `None` when the component `u_1(i)` is numerically zero: in that
/// case `lambda_min(R_{-i}) = lambda_1` and the eigenvector is `u_1` restricted.
pub fn secular_root(spec: &LowSpectrum, i: usize) -> Option<f64> {
    let p = spec.values.len();
    if p == 0 {
        return None;
    }
    let lam1 = spec.values[0];
    let u1i2 = spec.vectors[0][i] * spec.vectors[0][i];
    if u1i2 <= 1e-14 {
        return None;
    }
    let hi = if p > 1 { spec.values[1] } else { lam1 + 1.0 };
    let f = |mu: f64| -> f64 {
        spec.values
            .iter()
            .zip(spec.vectors.iter())
            .map(|(l, u)| {
                let t = u[i];
                t * t / (l - mu)
            })
            .sum()
    };
    let mut a = lam1 + 1e-13 * (1.0 + lam1.abs());
    let mut b = hi - 1e-13 * (1.0 + hi.abs());
    if !(a < b) {
        return None;
    }
    if f(a) > 0.0 {
        return None;
    }
    for _ in 0..80 {
        let m = 0.5 * (a + b);
        if !m.is_finite() {
            return None;
        }
        if f(m) < 0.0 {
            a = m;
        } else {
            b = m;
        }
    }
    Some(0.5 * (a + b))
}

/// Trial vector `x = (R - mu I)^{-1} e_i` truncated to the first `p` eigenpairs.
pub fn trial_vector(spec: &LowSpectrum, i: usize, mu: f64) -> Vec<f64> {
    let mut x = vec![0.0; spec.dim];
    for (l, u) in spec.values.iter().zip(spec.vectors.iter()) {
        let d = l - mu;
        if d.abs() > 1e-300 {
            let w = u[i] / d;
            if w != 0.0 {
                crate::num::axpy(w, u, &mut x);
            }
        }
    }
    x
}

/// Rayleigh bound for removing the `i`-th element, from `y = R u`.
///
/// Valid for **any** unit vector `u`.
#[inline]
pub fn rayleigh_upper(rq: f64, ui: f64, yi: f64) -> f64 {
    let v2 = 1.0 - ui * ui;
    if v2 <= 1e-14 {
        return f64::INFINITY;
    }
    (rq - 2.0 * ui * yi + ui * ui) / v2
}

fn head_rayleigh(ds: &Dataset, k: usize, u: &[f64]) -> (f64, Vec<f64>) {
    let mut op = SubOp::head(ds, k);
    let mut y = vec![0.0; k];
    op.mul(u, &mut y);
    let rq = dot(u, &y);
    (rq, y)
}

/// Restriction to `{0..k}\{i}` then permutation resulting from `swap(i, k-1)`.
fn restrict_and_permute(v: &[f64], i: usize, k: usize) -> Vec<f64> {
    let mut w = Vec::with_capacity(k.saturating_sub(1));
    w.extend_from_slice(&v[..i]);
    w.extend_from_slice(&v[i + 1..k]);
    if i + 1 < k {
        let last = w.pop().expect("non-empty vector");
        w.insert(i, last);
    }
    w
}

fn permute_after_swap(v: &mut Vec<f64>, i: usize, k: usize) {
    if i + 1 < k {
        let last = v.pop().expect("non-empty vector");
        v.insert(i, last);
    }
}

/// Operator of a candidate: direct (`R_{-i}`) or inverse (`-R_{-i}^{-1}`).
enum DelOp<'a> {
    Direct(SubOp<'a>),
    Inverse(NegInvSubOp<'a>),
}

impl SymOp for DelOp<'_> {
    fn n(&self) -> usize {
        match self {
            DelOp::Direct(o) => o.n(),
            DelOp::Inverse(o) => o.n(),
        }
    }
    fn mul(&mut self, x: &[f64], y: &mut [f64]) {
        match self {
            DelOp::Direct(o) => o.mul(x, y),
            DelOp::Inverse(o) => o.mul(x, y),
        }
    }
}

/// Converts an eigenvalue of the operator used into `lambda_min` of `R`.
#[inline]
fn to_lambda(theta: f64, inv: bool) -> f64 {
    if inv {
        if theta.abs() < 1e-300 {
            f64::INFINITY
        } else {
            -1.0 / theta
        }
    } else {
        theta
    }
}

#[derive(Debug)]
struct Eval {
    pos: usize,
    upper: f64,
    out: LanczosOutcome,
}

fn eval_deletion(
    ds: &Dataset,
    k: usize,
    seed: &[f64],
    i: usize,
    upper: f64,
    cfg: &AlgoConfig,
    inv: Option<&InverseSym>,
) -> Eval {
    let mut out = match inv {
        Some(iv) => {
            let mut op = DelOp::Inverse(NegInvSubOp::new(iv, k, Some(i)));
            let o = smallest_eigenpair(&mut op, Some(seed), cfg.tol, cfg.max_iters_warm);
            LanczosOutcome {
                value: to_lambda(o.value, true),
                ..o
            }
        }
        None => {
            let mut op = DelOp::Direct(SubOp::deleted(ds, k, i));
            smallest_eigenpair(&mut op, Some(seed), cfg.tol, cfg.max_iters_warm)
        }
    };
    out.value = out.value.max(f64::NEG_INFINITY);
    Eval { pos: i, upper, out }
}

/// Evaluation seed for removing variable `i`: exact eigenvector of
/// `R_{-i}` approximated by `(R - mu I)^{-1} e_i` (`mu` = truncated secular root),
/// with a fallback to `u_1` restricted.
///
/// The trial vector has dimension `k` (it carries component `i`): we
/// **restrict** it to `{0..k}\{i}` so that it is a vector of the operator of
/// dimension `k-1`. Without this restriction, the solver silently rejected the
/// seed (incompatible length) and restarted from a random vector, which
/// multiplied the number of Lanczos iterations per candidate by ~30.
fn deletion_seed(spec: &LowSpectrum, i: usize, k: usize) -> Vec<f64> {
    let u = &spec.vectors[0];
    let restrict = |v: &[f64]| {
        let mut s = Vec::with_capacity(k - 1);
        s.extend_from_slice(&v[..i]);
        s.extend_from_slice(&v[i + 1..k]);
        s
    };
    match secular_root(spec, i) {
        Some(mu) => {
            let x = trial_vector(spec, i, mu);
            let n = crate::num::norm2(&x);
            if n > 1e-8 {
                restrict(&x)
            } else {
                restrict(u)
            }
        }
        None => restrict(u),
    }
}

/// A backward elimination step: chooses the variable to remove and returns the
/// eigenpair of the new set.
fn step_eliminate(
    ds: &Dataset,
    k: usize,
    spec: &LowSpectrum,
    cfg: &AlgoConfig,
    inv: Option<&InverseSym>,
) -> (usize, StepRecord, LanczosOutcome) {
    let u = &spec.vectors[0];
    let (rq, y) = head_rayleigh(ds, k, u);
    let mut cands: Vec<(f64, f64, usize)> = (0..k)
        .into_par_iter()
        .map(|i| {
            let ray = rayleigh_upper(rq, u[i], y[i]);
            let ub = spec.upper_bound(i, ray);
            (ub, ray, i)
        })
        .collect();
    // We start with the LARGEST bounds: these are the only ones that can
    // still beat the best already-evaluated candidate; as soon as the best exact
    // value exceeds the largest remaining bound, the step is certified.
    cands.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Less));

    let mut best: Option<Eval> = None;
    let mut evaluated = 0usize;
    let mut iters = 0usize;
    let mut certified = false;
    let margin = cfg.tol * 50.0 * (1.0 + rq.abs());
    let mut idx = 0usize;
    while idx < k {
        if let Some(b) = &best {
            if b.out.value >= cands[idx].0 - margin {
                certified = true;
                break;
            }
        }
        if cfg.max_exact > 0 && evaluated >= cfg.max_exact {
            break;
        }
        let mut bs = cfg.batch.max(1);
        if cfg.max_exact > 0 {
            bs = bs.min(cfg.max_exact - evaluated);
        }
        if bs == 0 {
            break;
        }
        let end = (idx + bs).min(k);
        // Same as forward: the deflation seed (`secular_root`: 80 bisections) is
        // built inside the parallel task rather than serially before the batch.
        let batch: Vec<Eval> = cands[idx..end]
            .par_iter()
            .map(|&(ub, _ray, i)| {
                let seed = deletion_seed(spec, i, k);
                eval_deletion(ds, k, &seed, i, ub, cfg, inv)
            })
            .collect();
        for e in batch {
            evaluated += 1;
            iters += e.out.iters;
            // Ties (up to tolerance): we keep the first encountered, hence the
            // best score, to avoid random numerical differences
            // deciding between indistinguishable candidates.
            let better = match &best {
                None => true,
                Some(b) => e.out.value > b.out.value + cfg.tol * 10.0 * (1.0 + b.out.value.abs()),
            };
            if better {
                best = Some(e);
            }
        }
        idx = end;
    }
    if idx >= k {
        certified = true;
    }
    let best = best.expect("at least one candidate evaluated");
    let chosen = best.pos;
    let mut vector = best.out.vector.clone();
    if chosen + 1 < k {
        permute_after_swap(&mut vector, chosen, k);
    }
    let value = best.out.value;
    let residual = best.out.residual;
    let out = LanczosOutcome {
        value,
        vector,
        residual,
        iters: best.out.iters,
        converged: best.out.converged,
    };
    let rec = StepRecord {
        k_before: k,
        k_after: k - 1,
        changed_orig: ds.orig(chosen),
        lambda: value,
        lambda_verified: f64::NAN,
        upper: best.upper,
        rayleigh: rayleigh_upper(rq, u[chosen], y[chosen]),
        exact_evals: evaluated,
        candidates: k,
        certified,
        residual,
        iters,
    };
    (chosen, rec, out)
}

/// Parameters of the local improvement by swaps.
#[derive(Clone, Copy, Debug)]
pub struct RefineCfg {
    /// Number of passes.
    pub passes: usize,
    /// Leaving/entering considered per pass.
    pub top: usize,
    /// Apply the swaps only to sizes `k <= from_k`.
    pub from_k: usize,
}

/// Certified backward elimination from `M` down to `k_min`.
///
/// `progress` is called after each step and returns `true` to continue the
/// path, `false` to stop it cleanly (the partial result is
/// returned, with the steps already performed).
pub fn backward(
    ds: &mut Dataset,
    k_min: usize,
    cfg: &AlgoConfig,
    refine: Option<RefineCfg>,
    verify: bool,
    progress: &mut dyn FnMut(&StepRecord) -> bool,
) -> PathResult {
    let t0 = Instant::now();
    let m = ds.m();
    assert!(k_min >= 1 && k_min <= m, "invalid k_min");
    // Maintained inverse: evaluates candidates via -R_{-i}^{-1} (fast convergence).
    let want_inv = match cfg.eval {
        EvalMode::Direct => false,
        EvalMode::Inverse => true,
        EvalMode::Auto => ds.is_packed() && m <= cfg.inverse_max_m,
    };
    let mut inv = if want_inv {
        ds.packed
            .as_ref()
            .and_then(|p| InverseSym::from_packed(p, m))
    } else {
        None
    };
    // The lower spectrum (hence the cascade bounds) is computed in direct mode:
    // the bounds are then identical to `--eval direct`, only the exact computation of
    // candidates exploits the inverse.
    let init = low_spectrum(ds, m, cfg.num_low, &[], None, cfg, true);
    let initial_lambda = init.lambda_min();
    let initial_residual = init.max_residual();
    let mut spec = init;
    let mut k = m;
    let mut steps = Vec::with_capacity(m - k_min);
    while k > k_min {
        let (chosen, rec, winner) = step_eliminate(ds, k, &spec, cfg, inv.as_ref());
        ds.swap(chosen, k - 1);
        if let Some(iv) = inv.as_mut() {
            iv.swap_leading(chosen, k - 1, k);
            iv.downdate_last(k);
        }
        let mut seeds: Vec<Vec<f64>> = spec
            .vectors
            .iter()
            .map(|v| restrict_and_permute(v, chosen, k))
            .collect();
        if !seeds.is_empty() {
            seeds[0] = winner.vector.clone();
        }
        k -= 1;
        let keep_going = progress(&rec);
        steps.push(rec);
        if !keep_going {
            break;
        }
        if k >= k_min && k > 1 {
            if verify {
                let strict = AlgoConfig {
                    tol: 1e-13,
                    max_iters_cold: 4000,
                    max_iters_warm: 400,
                    ..*cfg
                };
                let ev = initial_eigenpair_inv(ds, k, &strict, inv.as_ref());
                if let Some(last) = steps.last_mut() {
                    last.lambda_verified = ev.value;
                }
            }
            spec = low_spectrum(ds, k, cfg.num_low, &seeds, Some(&winner), cfg, false);
        }
        if let Some(r) = refine {
            if k >= 2 && k <= r.from_k && r.passes > 0 {
                let mut u = spec.vectors[0].clone();
                // `refine_swaps` does not drive the stop: we capture its flag via
                // a local closure, then stop the cascade if requested.
                let mut stop = false;
                let (nl, _imp) = {
                    let mut relay = |st: &StepRecord| {
                        if !progress(st) {
                            stop = true;
                        }
                    };
                    refine_swaps(
                        ds,
                        k,
                        spec.lambda_min(),
                        &mut u,
                        cfg,
                        r.passes,
                        r.top,
                        &mut relay,
                    )
                };
                if let Some(last) = steps.last_mut() {
                    last.lambda = nl;
                }
                if stop {
                    break;
                }
                spec = low_spectrum(ds, k, cfg.num_low, &[], None, cfg, false);
                if !spec.vectors.is_empty() {
                    spec.vectors[0] = u;
                    spec.values[0] = nl;
                }
            }
        }
    }
    PathResult {
        direction: Direction::Backward,
        initial_k: m,
        initial_lambda,
        initial_residual,
        steps,
        seconds: t0.elapsed().as_secs_f64(),
        initial_subset: (0..m).collect(),
    }
}

/// Exact evaluation of adding variable `j` (KKT trial vector as seed).
///
/// If the correlation is materialized, we evaluate the bordered matrix directly
/// (`O(k^2)` per matvec) rather than via `Z` (`O(N k)`): for large `N` it is the
/// limiting factor of the exact forward greedy.
fn eval_addition(
    ds: &Dataset,
    k: usize,
    seed: &[f64],
    j: usize,
    upper: f64,
    cfg: &AlgoConfig,
) -> Eval {
    let out = match ds.packed.as_ref() {
        Some(p) => {
            let mut op = BorderedHeadOp::new(p, k, j);
            smallest_eigenpair(&mut op, Some(seed), cfg.tol, cfg.max_iters_warm)
        }
        None => {
            let mut op = GatheredZOp::build(&ds.z, k, None, Some(j));
            smallest_eigenpair(&mut op, Some(seed), cfg.tol, cfg.max_iters_warm)
        }
    };
    Eval { pos: j, upper, out }
}

/// Root of the secular equation of the bordered matrix
/// `[[R_S, c], [c^T, 1]]`, i.e. `1 - mu = sum_l g_l^2 / (lambda_l - mu)` with
/// `g_l = u_l^T c`, searched below `lambda_1`.  It is an upper bound of the new
/// `lambda_min` when the sum is truncated to the first `p` eigenpairs.
fn addition_secular(values: &[f64], g: &[f64]) -> f64 {
    let lam1 = values[0];
    let f = |mu: f64| -> f64 {
        1.0 - mu
            - values
                .iter()
                .zip(g.iter())
                .map(|(l, gi)| gi * gi / (l - mu))
                .sum::<f64>()
    };
    let mut lo = -1.0 - lam1.abs() - g.iter().map(|x| x * x).sum::<f64>();
    let mut hi = lam1 - 1e-13 * (1.0 + lam1.abs());
    for _ in 0..10 {
        if f(lo) > 0.0 {
            break;
        }
        lo *= 2.0;
    }
    if !(lo < hi) || f(lo) <= 0.0 {
        return lam1;
    }
    for _ in 0..80 {
        let m = 0.5 * (lo + hi);
        if f(m) > 0.0 {
            lo = m;
        } else {
            hi = m;
        }
    }
    0.5 * (lo + hi)
}

/// Evaluation seed for adding variable `j`: KKT trial vector
/// `[y ; 1]` built from the bordered secular root, with a fallback to `[u_1 ; 1]`
/// (never catastrophic) if that root is not usable.
///
/// `mu_hint = Some(mu)` reuses a root already computed by the caller (the
/// certified bound of the same candidate), avoiding a second 80-step bisection.
fn addition_seed(
    spec: &LowSpectrum,
    g: &[f64],
    m: usize,
    j: usize,
    mu_hint: Option<f64>,
) -> Vec<f64> {
    let p = spec.values.len();
    let k = spec.dim;
    let gl: Vec<f64> = (0..p).map(|l| g[l * m + j]).collect();
    let mu = match mu_hint {
        Some(mu) => mu,
        None => addition_secular(&spec.values, &gl),
    };
    let usable = mu < spec.values[0] - 1e-9 * (1.0 + spec.values[0].abs());
    let mut seed = vec![0.0; k];
    if usable {
        for l in 0..p {
            let d = spec.values[l] - mu;
            if d.abs() > 1e-300 {
                // y = -(R_S - mu I)^{-1} c  (bordered system: see docs)
                let wv = -gl[l] / d;
                if wv != 0.0 {
                    crate::num::axpy(wv, &spec.vectors[l], &mut seed);
                }
            }
        }
        // component on e_j: the exact trial vector satisfies x = [y ; 1], we
        // normalize to avoid blow-ups.
        let ny = crate::num::norm2(&seed);
        if !(ny.is_finite()) || ny > 1e6 {
            seed = spec.vectors[0].clone();
        }
    } else {
        seed = spec.vectors[0].clone();
    }
    seed.push(1.0);
    seed
}

/// Forward selection: starts from a singleton and greedily adds the variable that
/// maximizes `lambda_min`.
///
/// Candidate ranking uses the secular equation of the bordered matrix:
/// the first-order loss is `sum_l g_l^2 / (lambda_l - lambda_1)` with
/// `g_l = u_l . c_j` (`c_j` = correlations of `j` with `S`), instead of only
/// `|g_1|`: eigen-directions close to `lambda_1` cost more.
///
/// `top = 0` (or `top >= number of candidates`) disables the pre-filter: the step
/// is then **certified optimal**. The truncated secular root upper-bounds
/// `lambda_min` of the bordered matrix; by sorting by DECREASING bound and
/// evaluating exactly in batches, we stop as soon as the best achieved value
/// exceeds the largest remaining bound -- without evaluating the `M-k` candidates.
/// This is the default mode; a finite `top` (typically 16) is an explicit choice
/// of the caller, which restricts the evaluation to the `top` best loss scores
/// and therefore trades optimality for time.
///
/// Each candidate is evaluated by warm Lanczos with the KKT trial vector
/// `[y ; 1]` as seed, `y = -sum_l g_l/(lambda_l - mu) u_l`.
///
/// `progress` returns `true` to continue, `false` to stop cleanly.
pub fn forward(
    ds: &mut Dataset,
    k_max: usize,
    cfg: &AlgoConfig,
    top: usize,
    first: Option<usize>,
    progress: &mut dyn FnMut(&StepRecord) -> bool,
) -> PathResult {
    let mut trace = Vec::new();
    forward_traced(ds, k_max, cfg, top, first, progress, &mut trace)
}

/// Identical to [`forward`], but records the sequence of physical swaps
/// performed. Since swaps are involutions, replaying them backwards restores
/// exactly the input state: this is what allows chaining several starts
/// without rebuilding the correlation matrix (`O(N M^2)`).
fn forward_traced(
    ds: &mut Dataset,
    k_max: usize,
    cfg: &AlgoConfig,
    top: usize,
    first: Option<usize>,
    progress: &mut dyn FnMut(&StepRecord) -> bool,
    trace: &mut Vec<(usize, usize)>,
) -> PathResult {
    let t0 = Instant::now();
    let m = ds.m();
    assert!(k_max >= 1 && k_max <= m, "invalid k_max");
    let f = first.unwrap_or_else(|| best_first_feature(ds)).min(m - 1);
    ds.swap(0, f);
    trace.push((0, f));
    let first_orig = ds.orig(0);
    let mut k = 1usize;
    let mut spec = LowSpectrum::singleton();
    let mut steps = Vec::with_capacity(k_max.saturating_sub(1));
    let mut carry: Option<LanczosOutcome>;
    while k < k_max {
        let cands: Vec<usize> = (k..m).collect();
        if cands.is_empty() {
            break;
        }
        // `g_{l,j} = u_l^T Z_S^T z_j = (Z_S u_l) . z_j`: a single parallel sweep
        // over the `p` eigenpairs, each computing its loadings `w_l` then the `M`
        // dot products. The former version ran the `p` loadings serially and then
        // `p` barriers of `z_dot_all`; the `w_l` buffers are no longer retained
        // (and the redundant pass on `w[0]`, whose result was unused, is gone).
        let p = spec.values.len();
        let mut g = vec![0.0f64; p * m];
        g.par_chunks_mut(m)
            .zip(spec.vectors.par_iter())
            .for_each(|(gl, u)| {
                let mut wl = vec![0.0; ds.z.rows];
                z_loading(&ds.z, k, u, &mut wl);
                for (j, gj) in gl.iter_mut().enumerate() {
                    *gj = dot(ds.z.col(j), &wl);
                }
            });
        let rq = spec.lambda_min();
        // First-order score (estimated secular loss): it is used to sort the
        // pre-filter and to break ties.
        let lam1 = spec.values[0];
        let mut scored: Vec<(f64, usize)> = cands
            .par_iter()
            .map(|&j| {
                let mut loss = 0.0;
                for l in 0..p {
                    let d = (spec.values[l] - lam1).max(1e-12);
                    let v = g[l * m + j];
                    loss += v * v / d;
                }
                (loss, j)
            })
            .collect();
        // The pre-filter is applied only if the caller explicitly requested it
        // (finite `top`): `top = 0` achieves the true E-optimal greedy, but by
        // certification (see below) instead of an exhaustive evaluation.
        let exact = top == 0 || top >= scored.len();

        // `work` = `(bound_or_score, loss, j)`.
        //  * exact mode   : `bound` is a certified upper bound of the new
        //    `lambda_min`, and the sort is by DECREASING bound;
        //  * filter mode  : `bound` = `loss` (the historical score) and the sort is
        //    by increasing loss over the first `top`.
        let work: Vec<(f64, f64, usize)> = if exact {
            // The secular root truncated to the first `p` eigenpairs upper-bounds
            // `lambda_min` of the bordered matrix `[[R_S, c],[c^T, 1]]` (the tail
            // terms are positive below `lambda_1`). Sorting by decreasing bound then
            // evaluating exactly allows stopping as soon as the best achieved
            // value exceeds the largest remaining bound: the choice is then
            // that of the exact greedy, without evaluating the `M-k` candidates.
            let mut bounded: Vec<(f64, f64, usize)> = scored
                .par_iter()
                .map(|&(loss, j)| {
                    let gl: Vec<f64> = (0..p).map(|l| g[l * m + j]).collect();
                    (addition_secular(&spec.values, &gl), loss, j)
                })
                .collect();
            bounded.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Less));
            bounded
        } else {
            scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(top);
            scored
                .into_iter()
                .map(|(loss, j)| (loss, loss, j))
                .collect()
        };

        let margin = cfg.tol * 50.0 * (1.0 + lam1.abs());
        let mut best: Option<Eval> = None;
        let mut best_loss = f64::INFINITY;
        let mut iters = 0usize;
        let mut evaluated = 0usize;
        let mut certified = false;
        let mut idx = 0usize;
        while idx < work.len() {
            if exact {
                if let Some(b) = &best {
                    if b.out.value >= work[idx].0 - margin {
                        certified = true;
                        break;
                    }
                }
            }
            if cfg.max_exact > 0 && evaluated >= cfg.max_exact {
                break;
            }
            let mut bs = cfg.batch.max(1);
            if cfg.max_exact > 0 {
                bs = bs.min(cfg.max_exact - evaluated);
            }
            if bs == 0 {
                break;
            }
            let end = (idx + bs).min(work.len());
            // The KKT seed is built *inside* the parallel task: it runs an
            // 80-step bisection (`addition_secular`), which has no business being
            // serially prepended to the batch. In exact mode the same root has
            // already been computed for the bound, so it is reused as-is.
            let batch: Vec<Eval> = work[idx..end]
                .par_iter()
                .map(|&(ub, _, j)| {
                    let hint = if exact { Some(ub) } else { None };
                    let seed = addition_seed(&spec, &g, m, j, hint);
                    eval_addition(ds, k, &seed, j, ub, cfg)
                })
                .collect();
            for (e, &(_, loss, _)) in batch.into_iter().zip(work[idx..end].iter()) {
                evaluated += 1;
                iters += e.out.iters;
                let better = match &best {
                    None => true,
                    Some(b) => {
                        let tie = cfg.tol * 10.0 * (1.0 + b.out.value.abs());
                        if e.out.value > b.out.value + tie {
                            true
                        } else if e.out.value >= b.out.value - tie {
                            // Ties up to tolerance: we keep the smallest
                            // loss score, as the increasing sort did.
                            loss < best_loss
                        } else {
                            false
                        }
                    }
                };
                if better {
                    best = Some(e);
                    best_loss = loss;
                }
            }
            idx = end;
        }
        if exact && idx >= work.len() {
            certified = true;
        }
        let best = best.expect("at least one candidate evaluated");
        let chosen = best.pos;
        let value = best.out.value;
        let residual = best.out.residual;
        let upper = best_loss;
        let vector = best.out.vector.clone();
        ds.swap(k, chosen);
        trace.push((k, chosen));
        let rec = StepRecord {
            k_before: k,
            k_after: k + 1,
            changed_orig: ds.orig(k),
            lambda: value,
            lambda_verified: f64::NAN,
            upper: rq,
            rayleigh: upper,
            exact_evals: evaluated,
            candidates: cands.len(),
            certified,
            residual,
            iters,
        };
        k += 1;
        let keep_going = progress(&rec);
        steps.push(rec);
        if !keep_going {
            break;
        }
        // spectrum of the new set (warm starts: vectors extended by a 0)
        if k < k_max {
            let seeds: Vec<Vec<f64>> = spec
                .vectors
                .iter()
                .map(|v| {
                    let mut e = v.clone();
                    e.push(0.0);
                    e
                })
                .collect();
            carry = Some(LanczosOutcome {
                value,
                vector,
                residual,
                iters,
                converged: best.out.converged,
            });
            // In certified exact mode, we request the **full spectrum** (`p = k`):
            // the truncated secular root then becomes *exact* (cf.
            // `forward_secular_bound_quality`: gap ~1e-15), the certification
            // therefore stops at the first batch, and the cost of the spectrum (k warm
            // Lanczos) is far lower than the evaluations it saves. The
            // pre-filter, however, remains set by `--low-rank`.
            let p_spec = if top == 0 { k } else { cfg.num_low };
            spec = low_spectrum(ds, k, p_spec, &seeds, carry.as_ref(), cfg, false);
        }
    }
    PathResult {
        direction: Direction::Forward,
        initial_k: 1,
        initial_lambda: 1.0,
        initial_residual: 0.0,
        steps,
        seconds: t0.elapsed().as_secs_f64(),
        initial_subset: vec![first_orig],
    }
}

/// Ranking of the starting variables of the forward greedy, from best to worst.
///
/// One-step lookahead score: starting from `j`, the best `lambda_min`
/// reachable at `k = 2` is `1 - min_{i != j} |r_ij|` (closed form for a
/// 2x2 correlation matrix: `lambda_min([[1, r], [r, 1]]) = 1 - |r|`). It
/// therefore measures exactly what the first step yields, whereas
/// [`best_first_feature`] only looks at the maximal correlation.
///
/// In implicit representation (correlation not materialized) the exact computation
/// would cost `O(N M^2)`: we fall back to the distance to the centroid, `O(N M)`, which
/// generalizes the heuristic of the same name.
pub fn forward_seed_ranking(ds: &Dataset) -> Vec<usize> {
    let m = ds.m();
    let mut score = vec![0.0f64; m];
    if let Some(p) = ds.packed.as_ref() {
        let raw = p.raw();
        for i in 0..m {
            let base = crate::packed::row_offset(i);
            let mut mn = f64::INFINITY;
            for &v in &raw[base..base + i] {
                let a = v.abs();
                if a < mn {
                    mn = a;
                }
            }
            for j in (i + 1)..m {
                let a = p.get(j, i).abs();
                if a < mn {
                    mn = a;
                }
            }
            score[i] = if mn.is_finite() { 1.0 - mn } else { 1.0 };
        }
    } else {
        let rows = ds.z.rows;
        let mut centroid = vec![0.0; rows];
        for j in 0..m {
            crate::num::axpy(1.0, ds.z.col(j), &mut centroid);
        }
        let inv = 1.0 / m as f64;
        for v in centroid.iter_mut() {
            *v *= inv;
        }
        for j in 0..m {
            let mut d = 0.0f64;
            for (a, b) in ds.z.col(j).iter().zip(centroid.iter()) {
                let t = a - b;
                d += t * t;
            }
            score[j] = d;
        }
    }
    let mut idx: Vec<usize> = (0..m).collect();
    idx.sort_by(|&a, &b| {
        score[b]
            .partial_cmp(&score[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx
}

/// Multi-start forward greedy: several initial variables are tried one
/// after another, and the retained family is the one whose final `lambda_min` is
/// the largest (for equal size).
///
/// The first start is that of [`best_first_feature`] (historical
/// heuristic), then come the best variables of the ranking
/// [`forward_seed_ranking`], without duplicates. This construction guarantees two
/// things: `seeds = 1` reproduces exactly [`forward`], and the set of
/// tried starts **grows** with `seeds` -- increasing the number of starts
/// can therefore never degrade the final `lambda_min`. `seeds = M` tries all
/// starts.
///
/// A forced `first` disables the multi-start. The cost is multiplied by the
/// number of tried starts.
///
/// `progress` is called for each step of each start: the sequence of
/// `k_after` therefore restarts from 1 at each seed. Returning `false` stops everything.
pub fn forward_multiseed(
    ds: &mut Dataset,
    k_max: usize,
    cfg: &AlgoConfig,
    top: usize,
    seeds: usize,
    first: Option<usize>,
    progress: &mut dyn FnMut(&StepRecord) -> bool,
) -> PathResult {
    let m = ds.m();
    assert!(k_max >= 1 && k_max <= m, "invalid k_max");
    let n = if first.is_some() {
        1
    } else {
        seeds.clamp(1, m)
    };
    if n == 1 {
        return forward(ds, k_max, cfg, top, first, progress);
    }
    // Historical start first, then the lookahead ranking: the set of
    // starts is nested, so the result is monotone in `seeds`.
    let mut cands: Vec<usize> = vec![best_first_feature(ds)];
    for &s in forward_seed_ranking(ds).iter() {
        if cands.len() >= n {
            break;
        }
        if !cands.contains(&s) {
            cands.push(s);
        }
    }
    let t0 = Instant::now();
    let mut best: Option<PathResult> = None;
    for &seed in cands.iter() {
        let mut trace = Vec::with_capacity(k_max + 1);
        let mut stop = false;
        let path = {
            let mut relay = |rec: &StepRecord| {
                if progress(rec) {
                    true
                } else {
                    stop = true;
                    false
                }
            };
            forward_traced(ds, k_max, cfg, top, Some(seed), &mut relay, &mut trace)
        };
        // Restoration of the input state: the swaps are involutions,
        // replayed backwards.
        for &(a, b) in trace.iter().rev() {
            ds.swap(a, b);
        }
        let better = match &best {
            None => true,
            Some(b) => {
                let ka = path
                    .steps
                    .last()
                    .map(|s| s.k_after)
                    .unwrap_or(path.initial_k);
                let kb = b.steps.last().map(|s| s.k_after).unwrap_or(b.initial_k);
                let la = path.lambda_at(ka).unwrap_or(f64::NEG_INFINITY);
                let lb = b.lambda_at(kb).unwrap_or(f64::NEG_INFINITY);
                let tie = cfg.tol * 10.0 * (1.0 + lb.abs());
                ka > kb || (ka == kb && la > lb + tie)
            }
        };
        if better {
            best = Some(path);
        }
        if stop {
            break;
        }
    }
    let mut win = best.expect("at least one seed evaluated");
    win.seconds = t0.elapsed().as_secs_f64();
    win
}

/// First variable: the one whose maximal correlation with the others is the
/// smallest (Gershgorin-type heuristic). In implicit representation, we fall back
/// to the column farthest from the centroid (`O(NM)`).
pub fn best_first_feature(ds: &Dataset) -> usize {
    let m = ds.m();
    if m <= 1 {
        return 0;
    }
    if let Some(p) = ds.packed.as_ref() {
        if m <= 20_000 {
            let raw = p.raw();
            let mut best = 0usize;
            let mut best_score = f64::INFINITY;
            for i in 0..m {
                let base = crate::packed::row_offset(i);
                let mut mx = 0.0f64;
                for &v in &raw[base..base + i] {
                    let a = v.abs();
                    if a > mx {
                        mx = a;
                    }
                }
                for j in (i + 1)..m {
                    let a = p.get(j, i).abs();
                    if a > mx {
                        mx = a;
                    }
                }
                if mx < best_score {
                    best_score = mx;
                    best = i;
                }
            }
            return best;
        }
        return 0;
    }
    let rows = ds.z.rows;
    let mut centroid = vec![0.0; rows];
    for j in 0..m {
        crate::num::axpy(1.0, ds.z.col(j), &mut centroid);
    }
    let inv = 1.0 / m as f64;
    for v in centroid.iter_mut() {
        *v *= inv;
    }
    let mut best = 0usize;
    let mut best_d = -1.0f64;
    for j in 0..m {
        let mut d = 0.0f64;
        for (a, b) in ds.z.col(j).iter().zip(centroid.iter()) {
            let t = a - b;
            d += t * t;
        }
        if d > best_d {
            best_d = d;
            best = j;
        }
    }
    best
}

/// Local improvement by 1-for-1 swaps (constant size `K`).
pub fn refine_swaps(
    ds: &mut Dataset,
    k: usize,
    lam: f64,
    u: &mut Vec<f64>,
    cfg: &AlgoConfig,
    passes: usize,
    top: usize,
    progress: &mut dyn FnMut(&StepRecord),
) -> (f64, usize) {
    if k < 2 || k >= ds.m() {
        return (lam, 0);
    }
    let mut cur = lam;
    let mut improvements = 0usize;
    for _pass in 0..passes {
        let (rq, ys) = head_rayleigh(ds, k, u);
        let mut outs: Vec<(f64, usize)> = (0..k)
            .map(|i| (rayleigh_upper(rq, u[i], ys[i]), i))
            .collect();
        outs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut w = vec![0.0; ds.z.rows];
        z_loading(&ds.z, k, u, &mut w);
        let mut g = vec![0.0; ds.m()];
        z_dot_all(&ds.z, &w, &mut g);
        let mut ins: Vec<(f64, usize)> = (k..ds.m()).map(|j| (g[j].abs(), j)).collect();
        ins.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let nt = top.max(1);
        let mut pairs: Vec<(f64, f64, usize, usize)> = Vec::new();
        for &(rho_i, i) in outs.iter().take(nt) {
            for &(gj, j) in ins.iter().take(nt) {
                pairs.push((rho_i - gj, rho_i, i, j));
            }
        }
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        pairs.truncate(cfg.batch.max(1));
        let evals: Vec<(usize, usize, f64, LanczosOutcome)> = pairs
            .par_iter()
            .map(|&(_, rho_i, i, j)| {
                let mut op = GatheredZOp::build(&ds.z, k, Some(i), Some(j));
                let mut seed = Vec::with_capacity(k);
                seed.extend_from_slice(&u[..i]);
                seed.extend_from_slice(&u[i + 1..]);
                seed.push(0.0);
                let out = smallest_eigenpair(&mut op, Some(&seed), cfg.tol, cfg.max_iters_warm);
                (i, j, rho_i, out)
            })
            .collect();
        let mut bi = 0usize;
        let mut bj = 0usize;
        let mut bup = f64::NAN;
        let mut bval = cur;
        let mut bvec: Vec<f64> = Vec::new();
        let mut bres = 0.0f64;
        let mut bit = 0usize;
        for (i, j, rho_i, out) in evals {
            bit += out.iters;
            if out.value > bval {
                bval = out.value;
                bup = rho_i;
                bvec = out.vector;
                bres = out.residual;
                bi = i;
                bj = j;
            }
        }
        if bvec.is_empty() || bval <= cur {
            break;
        }
        let orig_out = ds.orig(bi);
        ds.swap(bi, bj);
        permute_after_swap(&mut bvec, bi, k);
        *u = bvec;
        progress(&StepRecord {
            k_before: k,
            k_after: k,
            changed_orig: orig_out,
            lambda: bval,
            lambda_verified: f64::NAN,
            upper: bup,
            rayleigh: bup,
            exact_evals: pairs.len(),
            candidates: outs.len() * ins.len(),
            certified: false,
            residual: bres,
            iters: bit,
        });
        cur = bval;
        improvements += 1;
    }
    (cur, improvements)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gen::{generate, GenKind};
    use crate::jacobi::eigen_sym_sorted;
    use crate::op::{Dataset, Repr};

    fn dense_sub(ds: &Dataset, idx: &[usize]) -> Vec<f64> {
        let d = idx.len();
        let mut a = vec![0.0; d * d];
        for (x, &ix) in idx.iter().enumerate() {
            for (y, &iy) in idx.iter().enumerate() {
                let v = match ds.packed.as_ref() {
                    Some(p) => p.get(ix, iy),
                    None => dot(ds.z.col(ix), ds.z.col(iy)),
                };
                a[x * d + y] = v;
            }
        }
        a
    }

    /// The Rayleigh bound must be valid for ANY unit vector.
    #[test]
    fn rayleigh_upper_is_valid_for_any_unit_vector() {
        let mut dm = generate(GenKind::Blocks, 200, 24, 0.5, 0.1, 3, 1, 77);
        dm.standardize(true);
        let ds = Dataset::new(dm, Repr::Packed, usize::MAX, 8);
        let k = 24;
        let cfg = AlgoConfig {
            tol: 1e-13,
            max_iters_cold: 1000,
            ..Default::default()
        };
        let spec = low_spectrum(&ds, k, 1, &[], None, &cfg, true);
        let mut rng = crate::gen::Rng::new(2024);
        let mut vectors: Vec<Vec<f64>> = vec![spec.vectors[0].clone()];
        for _ in 0..20 {
            let mut v: Vec<f64> = (0..k).map(|_| rng.normal()).collect();
            let n = crate::num::norm2(&v);
            for x in v.iter_mut() {
                *x /= n;
            }
            vectors.push(v);
        }
        for u in &vectors {
            let mut y = vec![0.0; k];
            let mut op = SubOp::head(&ds, k);
            op.mul(u, &mut y);
            let rq = dot(u, &y);
            for i in 0..k {
                let rho = rayleigh_upper(rq, u[i], y[i]);
                let idx: Vec<usize> = (0..k).filter(|&j| j != i).collect();
                let mut a = dense_sub(&ds, &idx);
                let (vals, _) = eigen_sym_sorted(&mut a, k - 1);
                assert!(
                    vals[0] <= rho + 1e-10,
                    "i={i} lam_min={} rho={rho}",
                    vals[0]
                );
            }
        }
    }

    /// The Temple bound must dominate `lambda_min(R_{-i})` and be tighter.
    #[test]
    fn temple_bound_is_valid_and_tighter() {
        for seed in 0..5u64 {
            let mut dm = generate(GenKind::Blocks, 120, 22, 0.6, 0.05, 3, 1, seed);
            dm.standardize(true);
            let ds = Dataset::new(dm, Repr::Packed, usize::MAX, 8);
            let k = 22;
            let cfg = AlgoConfig {
                tol: 1e-13,
                max_iters_cold: 2000,
                ..Default::default()
            };
            let spec = low_spectrum(&ds, k, 8, &[], None, &cfg, true);
            let (rq, y) = head_rayleigh(&ds, k, &spec.vectors[0]);
            let mut sum_ray = 0.0f64;
            let mut sum_tem = 0.0f64;
            for i in 0..k {
                let ray = rayleigh_upper(rq, spec.vectors[0][i], y[i]);
                let ub = spec.upper_bound(i, ray);
                let idx: Vec<usize> = (0..k).filter(|&j| j != i).collect();
                let mut a = dense_sub(&ds, &idx);
                let (vals, _) = eigen_sym_sorted(&mut a, k - 1);
                assert!(
                    ub >= vals[0] - 1e-8,
                    "seed={seed} i={i}: ub={ub} exact={}",
                    vals[0]
                );
                sum_ray += ray - vals[0];
                sum_tem += ub - vals[0];
            }
            assert!(sum_tem < sum_ray, "the Temple bound must be tighter");
            eprintln!(
                "seed={seed} mean gap Rayleigh={:.5} Temple={:.5}",
                sum_ray / k as f64,
                sum_tem / k as f64
            );
        }
    }

    /// The greedy choice must match brute force (small M).
    #[test]
    fn backward_matches_brute_force_greedy() {
        for seed in 0..4u64 {
            let mut dm = generate(GenKind::Blocks, 120, 12, 0.6, 0.05, 3, 1, seed);
            dm.standardize(true);
            let mut ds = Dataset::new(dm, Repr::Packed, usize::MAX, 8);
            let cfg = AlgoConfig {
                tol: 1e-13,
                max_iters_cold: 2000,
                max_iters_warm: 200,
                ..Default::default()
            };
            let m = ds.m();
            let path = backward(&mut ds, 1, &cfg, None, false, &mut |_| true);
            let mut ds2 = Dataset::new(ds.z.clone(), Repr::Packed, usize::MAX, 8);
            let mut k = m;
            let mut step = 0;
            while k > 1 {
                let mut best_i = 0usize;
                let mut best_v = f64::NEG_INFINITY;
                for i in 0..k {
                    let idx: Vec<usize> = (0..k).filter(|&j| j != i).collect();
                    let mut a = dense_sub(&ds2, &idx);
                    let (vals, _) = eigen_sym_sorted(&mut a, k - 1);
                    if vals[0] > best_v {
                        best_v = vals[0];
                        best_i = i;
                    }
                }
                let got = &path.steps[step];
                assert!(
                    (got.lambda - best_v).abs() < 1e-7,
                    "seed={seed} k={k}: greedy {} vs brute force {}",
                    got.lambda,
                    best_v
                );
                ds2.swap(best_i, k - 1);
                k -= 1;
                step += 1;
            }
        }
    }

    /// Inverse evaluation must give the same value as direct evaluation
    /// (and the same eigenpair), up to the solver tolerance.
    #[test]
    fn inverse_eval_matches_direct_eval() {
        let mut dm = generate(GenKind::Blocks, 300, 24, 0.6, 0.05, 3, 1, 5);
        dm.standardize(true);
        let ds = Dataset::new(dm, Repr::Packed, usize::MAX, 8);
        let k = 24;
        let cfg = AlgoConfig {
            tol: 1e-12,
            max_iters_cold: 4000,
            max_iters_warm: 400,
            ..Default::default()
        };
        let inv = InverseSym::from_packed(ds.packed.as_ref().unwrap(), k).expect("PD");
        let direct = initial_eigenpair_public(&ds, k, &cfg);
        let via_inv = initial_eigenpair_inv(&ds, k, &cfg, Some(&inv));
        assert!(
            (direct.value - via_inv.value).abs() < 1e-9,
            "{} vs {}",
            direct.value,
            via_inv.value
        );
        let spec = low_spectrum(&ds, k, 2, &[], Some(&direct), &cfg, true);
        for i in [0usize, 1, 7, 13, 23] {
            let seed = deletion_seed(&spec, i, k);
            let d = eval_deletion(&ds, k, &seed, i, f64::INFINITY, &cfg, None);
            let v = eval_deletion(&ds, k, &seed, i, f64::INFINITY, &cfg, Some(&inv));
            assert!(
                (d.out.value - v.out.value).abs() < 1e-8,
                "i={i}: direct {} inverse {}",
                d.out.value,
                v.out.value
            );
        }
    }

    /// The implicit path must give the same results as the packed path.
    #[test]
    fn backward_implicit_matches_packed() {
        let mut dm = generate(GenKind::Equi, 150, 30, 0.4, 0.0, 1, 1, 3);
        dm.standardize(true);
        let cfg = AlgoConfig {
            tol: 1e-12,
            max_iters_cold: 2000,
            max_iters_warm: 200,
            ..Default::default()
        };
        let mut dsp = Dataset::new(dm.clone(), Repr::Packed, usize::MAX, 8);
        let mut dsi = Dataset::new(dm, Repr::Implicit, 0, 8);
        let rp = backward(&mut dsp, 5, &cfg, None, false, &mut |_| true);
        let ri = backward(&mut dsi, 5, &cfg, None, false, &mut |_| true);
        for (a, b) in rp.steps.iter().zip(ri.steps.iter()) {
            assert!(
                (a.lambda - b.lambda).abs() < 1e-8,
                "{} vs {}",
                a.lambda,
                b.lambda
            );
        }
    }

    /// Quality of the Temple bound used to certify backward elimination:
    /// the gap to the true `lambda_min(R_{-i})` directly drives the number of
    /// candidates evaluated exactly, and decreases as `1/p` -- it would take `p = k`
    /// (full spectrum) to make it zero.
    #[test]
    fn deletion_bound_quality() {
        let mut dm = generate(GenKind::Blocks, 600, 60, 0.5, 0.05, 4, 1, 3);
        dm.standardize(true);
        let ds = Dataset::new(dm, Repr::Packed, usize::MAX, 16);
        let cfg = AlgoConfig {
            tol: 1e-13,
            max_iters_cold: 4000,
            max_iters_warm: 400,
            ..Default::default()
        };
        let k = 60usize;
        let mut prev_mean = f64::INFINITY;
        for p in [4usize, 8, 16] {
            let spec = low_spectrum(&ds, k, p, &[], None, &cfg, true);
            let (rq, y) = head_rayleigh(&ds, k, &spec.vectors[0]);
            let mut gaps = Vec::new();
            let mut exacts = Vec::new();
            for i in 0..k {
                let idx: Vec<usize> = (0..k).filter(|&j| j != i).collect();
                let mut a = dense_sub(&ds, &idx);
                let (vals, _) = eigen_sym_sorted(&mut a, k - 1);
                let ray = rayleigh_upper(rq, spec.vectors[0][i], y[i]);
                let ub = spec.upper_bound(i, ray);
                assert!(
                    ub >= vals[0] - 1e-8,
                    "invalid bound i={i}: {ub} < {}",
                    vals[0]
                );
                gaps.push(ub - vals[0]);
                exacts.push(vals[0]);
            }
            exacts.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let spread = exacts[exacts.len() - 1] - exacts[0];
            let mean = gaps.iter().sum::<f64>() / gaps.len() as f64;
            assert!(mean < prev_mean, "the gap must decrease with p");
            prev_mean = mean;
            eprintln!(
                "p={p:<3}: mean gap={mean:.3e} ({:.1}% of the spread)",
                100.0 * mean / spread
            );
        }
    }

    /// The deletion secular seed must have dimension `k-1` (a vector of
    /// the operator `R_{-i}`): this is what makes it usable by Lanczos. As long
    /// as it kept component `i`, the solver rejected it (length `k`) and
    /// restarted from a random vector, multiplying the iterations by ~30.
    #[test]
    fn deletion_seed_is_restricted_and_effective() {
        let mut dm = generate(GenKind::Blocks, 600, 120, 0.3, 0.0, 8, 1, 4);
        dm.standardize(true);
        let ds = Dataset::new(dm, Repr::Packed, usize::MAX, 16);
        let cfg = AlgoConfig {
            tol: 1e-13,
            max_iters_cold: 3000,
            max_iters_warm: 400,
            ..Default::default()
        };
        let k = 80usize;
        let spec = low_spectrum(&ds, k, 4, &[], None, &cfg, true);
        let inv = InverseSym::from_packed(ds.packed.as_ref().unwrap(), k).expect("PD");
        let mut seeded = 0usize;
        let mut blind = 0usize;
        for i in [3usize, 17, 41, 62] {
            let seed = deletion_seed(&spec, i, k);
            assert_eq!(seed.len(), k - 1, "the seed must live in R^(k-1)");
            assert!(crate::num::norm2(&seed) > 0.0);
            seeded += eval_deletion(&ds, k, &seed, i, f64::INFINITY, &cfg, Some(&inv))
                .out
                .iters;
            let rnd = crate::lanczos::default_seed(k - 1, i as u64);
            blind += eval_deletion(&ds, k, &rnd, i, f64::INFINITY, &cfg, Some(&inv))
                .out
                .iters;
        }
        assert!(
            seeded < blind,
            "the secular seed must reduce the iterations: {seeded} vs {blind}"
        );
    }

    /// The forward greedy must match brute force, including in certified
    /// mode (the pre-filter is disabled: `top = 0`).
    #[test]
    fn forward_matches_brute_force_greedy() {
        for seed in 0..3u64 {
            let mut dm = generate(GenKind::Blocks, 120, 11, 0.6, 0.05, 3, 1, seed);
            dm.standardize(true);
            let mut ds = Dataset::new(dm, Repr::Packed, usize::MAX, 8);
            let cfg = AlgoConfig {
                tol: 1e-13,
                max_iters_cold: 2000,
                max_iters_warm: 400,
                ..Default::default()
            };
            let m = ds.m();
            let path = forward(&mut ds, m, &cfg, 0, None, &mut |_| true);
            assert_eq!(path.steps.len(), m - 1);
            for st in &path.steps {
                assert!(st.certified, "each step of exact mode must be certified");
            }
            // Replays the exact greedy in PHYSICAL coordinates: `forward` has
            // brought the chosen variables to the front, so the current set is
            // `0..k` and the candidates are `k..m`.
            let ds2 = Dataset::new(ds.z.clone(), Repr::Packed, usize::MAX, 8);
            for step in 0..(m - 1) {
                let k = step + 1;
                let mut best_v = f64::NEG_INFINITY;
                for j in k..m {
                    let mut idx: Vec<usize> = (0..k).collect();
                    idx.push(j);
                    let mut a = dense_sub(&ds2, &idx);
                    let (vals, _) = eigen_sym_sorted(&mut a, k + 1);
                    if vals[0] > best_v {
                        best_v = vals[0];
                    }
                }
                let got = &path.steps[step];
                assert!(
                    (got.lambda - best_v).abs() < 1e-8,
                    "seed={seed} k={}: greedy {} vs brute force {}",
                    k,
                    got.lambda,
                    best_v
                );
            }
        }
    }

    /// The starting ranking must follow exactly the one-step lookahead
    /// `1 - min_{i != j} |r_ij|` (lambda_min of `R_{{j, i}}` at `k = 2`).
    #[test]
    fn forward_seed_ranking_matches_one_step_lookahead() {
        for seed in 0..3u64 {
            let mut dm = generate(GenKind::Blocks, 120, 13, 0.6, 0.05, 3, 1, seed);
            dm.standardize(true);
            let ds = Dataset::new(dm, Repr::Packed, usize::MAX, 8);
            let m = ds.m();
            let mut expected = Vec::with_capacity(m);
            for j in 0..m {
                let mut best = f64::NEG_INFINITY;
                for i in 0..m {
                    if i == j {
                        continue;
                    }
                    let mut a = dense_sub(&ds, &[i, j]);
                    let (vals, _) = eigen_sym_sorted(&mut a, 2);
                    if vals[0] > best {
                        best = vals[0];
                    }
                }
                expected.push(best);
            }
            let ranking = forward_seed_ranking(&ds);
            assert_eq!(ranking.len(), m);
            for w in ranking.windows(2) {
                assert!(
                    expected[w[0]] >= expected[w[1]] - 1e-12,
                    "seed={seed}: {} (score {}) must precede {} (score {})",
                    w[0],
                    expected[w[0]],
                    w[1],
                    expected[w[1]]
                );
            }
        }
    }

    /// `seeds = M` must achieve the best of the `M` possible starts, and
    /// leave the dataset exactly in its input state.
    #[test]
    fn forward_multiseed_reaches_best_single_seed() {
        for seed in 0..3u64 {
            let mut dm = generate(GenKind::Blocks, 120, 11, 0.6, 0.05, 3, 1, seed);
            dm.standardize(true);
            let cfg = AlgoConfig {
                tol: 1e-13,
                max_iters_cold: 2000,
                max_iters_warm: 400,
                ..Default::default()
            };
            let m = 11;
            // best of the starts tried one by one
            let best_single = (0..m)
                .map(|f| {
                    let mut ds = Dataset::new(dm.clone(), Repr::Packed, usize::MAX, 8);
                    let p = forward(&mut ds, m, &cfg, 0, Some(f), &mut |_| true);
                    p.lambda_at(m).unwrap()
                })
                .fold(f64::NEG_INFINITY, f64::max);

            let mut ds = Dataset::new(dm.clone(), Repr::Packed, usize::MAX, 8);
            let path = forward_multiseed(&mut ds, m, &cfg, 0, m, None, &mut |_| true);
            let got = path.lambda_at(m).unwrap();
            assert!(
                (got - best_single).abs() < 1e-9,
                "seed={seed}: multi-start {got} vs best start {best_single}"
            );
            // the state must be restored (no residual permutation)
            assert_eq!(
                ds.active,
                (0..m).collect::<Vec<_>>(),
                "dataset state not restored"
            );
            // and a subsequent path must give exactly the standard greedy
            let again = forward(&mut ds, m, &cfg, 0, None, &mut |_| true);
            let mut ref_ds = Dataset::new(dm, Repr::Packed, usize::MAX, 8);
            let reference = forward(&mut ref_ds, m, &cfg, 0, None, &mut |_| true);
            assert_eq!(
                again.order(),
                reference.order(),
                "seed={seed}: corrupted state"
            );
        }
    }

    /// The set of starts is nested: increasing `seeds` cannot
    /// degrade the final `lambda_min`, and `seeds = 1` = historical greedy.
    #[test]
    fn forward_multiseed_is_monotone_in_seeds() {
        for seed in 0..3u64 {
            let mut dm = generate(GenKind::Blocks, 120, 12, 0.6, 0.05, 3, 1, seed);
            dm.standardize(true);
            let cfg = AlgoConfig {
                tol: 1e-13,
                max_iters_cold: 2000,
                max_iters_warm: 400,
                ..Default::default()
            };
            let m = 12;
            let reference = {
                let mut ds = Dataset::new(dm.clone(), Repr::Packed, usize::MAX, 8);
                forward(&mut ds, m, &cfg, 0, None, &mut |_| true)
                    .lambda_at(m)
                    .unwrap()
            };
            let mut previous = f64::NEG_INFINITY;
            for n in 1..=m {
                let mut ds = Dataset::new(dm.clone(), Repr::Packed, usize::MAX, 8);
                let v = forward_multiseed(&mut ds, m, &cfg, 0, n, None, &mut |_| true)
                    .lambda_at(m)
                    .unwrap();
                if n == 1 {
                    assert!(
                        (v - reference).abs() < 1e-12,
                        "seed={seed}: seeds=1 must be the greedy"
                    );
                }
                assert!(
                    v >= previous - 1e-12,
                    "seed={seed}: seeds={n} ({v}) < seeds={} ({previous})",
                    n - 1
                );
                previous = v;
            }
        }
    }

    /// Diagnoses the quality of the secular bound used to certify
    /// forward selection: mean gap to the true `lambda_min` of `R_{S cup {j}}`.
    #[test]
    fn forward_secular_bound_quality() {
        let mut dm = generate(GenKind::Blocks, 500, 140, 0.3, 0.0, 8, 1, 4);
        dm.standardize(true);
        let ds = Dataset::new(dm, Repr::Packed, usize::MAX, 16);
        let pk = ds.packed.as_ref().expect("packed");
        // Two regimes: very strict cold lower spectrum, and production settings
        // (warm start, `max_iters_warm`).
        let cfgs = [
            (
                AlgoConfig {
                    tol: 1e-13,
                    max_iters_cold: 3000,
                    max_iters_warm: 400,
                    ..Default::default()
                },
                true,
            ),
            (AlgoConfig::default(), false),
        ];
        for (cfg, cold) in cfgs {
            for k in [10usize, 30, 60] {
                for p in [4usize, k] {
                    let spec = low_spectrum(&ds, k, p, &[], None, &cfg, cold);
                    assert_eq!(spec.len(), p.min(k));
                    let mut g = vec![0.0f64; spec.len()];
                    let mut worst = 0.0f64;
                    let mut sum = 0.0f64;
                    let mut n = 0usize;
                    for j in k..ds.m() {
                        for l in 0..spec.len() {
                            g[l] = (0..k).map(|i| pk.get(i, j) * spec.vectors[l][i]).sum();
                        }
                        let ub = addition_secular(&spec.values, &g);
                        let d = k + 1;
                        let mut a = vec![0.0; d * d];
                        for x in 0..k {
                            for y in 0..k {
                                a[x * d + y] = pk.get(x, y);
                            }
                            a[x * d + k] = pk.get(x, j);
                            a[k * d + x] = pk.get(x, j);
                        }
                        a[k * d + k] = 1.0;
                        let (vals, _) = eigen_sym_sorted(&mut a, d);
                        assert!(
                            ub >= vals[0] - 1e-8,
                            "invalid bound (k={k}, p={p}, cold={cold}): {ub} < {}",
                            vals[0]
                        );
                        worst = worst.max(ub - vals[0]);
                        sum += ub - vals[0];
                        n += 1;
                    }
                    eprintln!(
                        "k={k:<3} p={p:<3} cold={cold:<5}: mean bound-exact gap={:.3e} max={:.3e}",
                        sum / n as f64,
                        worst
                    );
                }
            }
        }
    }

    /// Whole-spectrum fast path (`p = k`, materialized correlation): the returned
    /// basis must be orthonormal, increasingly sorted, equal to the dense
    /// reference, and have machine-precision residuals (`p = k` is what makes the
    /// secular bound exact, and the cascade consumes `max_residual()`).
    #[test]
    fn dense_whole_spectrum_is_exact_and_orthonormal() {
        let mut dm = generate(GenKind::Blocks, 200, 50, 0.4, 0.05, 4, 1, 11);
        dm.standardize(true);
        let ds = Dataset::new(dm, Repr::Packed, usize::MAX, 16);
        let pk = ds.packed.as_ref().expect("packed");
        let cfg = AlgoConfig::default();
        for k in [2usize, 5, 17, 50] {
            let spec = low_spectrum(&ds, k, k, &[], None, &cfg, true);
            assert_eq!(spec.len(), k);
            for w in spec.values.windows(2) {
                assert!(w[0] <= w[1] + 1e-12, "k={k}: not increasing");
            }
            for l in 0..k {
                let nrm = crate::num::norm2(&spec.vectors[l]);
                assert!((nrm - 1.0).abs() < 1e-12, "k={k} l={l} norm={nrm}");
                for m in (l + 1)..k {
                    let dp: f64 = (0..k)
                        .map(|e| spec.vectors[l][e] * spec.vectors[m][e])
                        .sum();
                    assert!(dp.abs() < 1e-9, "k={k} <{l},{m}>={dp}");
                }
            }
            // dense reference
            let mut a = vec![0.0f64; k * k];
            for i in 0..k {
                for j in 0..=i {
                    let v = pk.get(i, j);
                    a[i * k + j] = v;
                    a[j * k + i] = v;
                }
            }
            let (vals, _) = eigen_sym_sorted(&mut a, k);
            for l in 0..k {
                assert!(
                    (spec.values[l] - vals[l]).abs() < 1e-9,
                    "k={k} l={l}: {} vs {}",
                    spec.values[l],
                    vals[l]
                );
            }
            assert!(
                spec.max_residual() < 1e-9,
                "k={k}: residual {} too large for the dense path",
                spec.max_residual()
            );
        }
    }
}

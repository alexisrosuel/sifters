//! Exact maximization of `lambda_min` over subsets of a **fixed size** `k`.
//!
//! The greedy family built by [`crate::greedy`] is myopic: "certified" only means
//! that each *step* is optimal. This module closes the gap for a target size by
//! exploring the space of `k`-subsets with a branch and bound, and returns either
//! the proven optimum or a certified optimality gap if the budget runs out.
//!
//! # Why the bound is free
//!
//! `lambda_min` is **monotonically non-increasing** when the set grows: `R_S` is a
//! principal submatrix of `R_T` for `S subset T`, and Cauchy interlacing gives
//! `lambda_min(R_S) >= lambda_min(R_T)`. Therefore the value of a partial set `S`
//! upper-bounds the value of *every* completion of `S`, and can be used to prune
//! without any extra computation: the search only ever evaluates `f(S + {j})`,
//! which is exactly the value it needs to sort the children.
//!
//! The search is a depth-first traversal with an explicit stack (bounded by
//! `O(m * k)` nodes), children explored in decreasing bound order. A multiset of
//! the bounds of the open nodes provides the upper bound `UB` of the certified
//! gap at any point: `best` (a real subset) is the lower bound, `max(UB, best)` is
//! the upper bound, and the search stops as soon as `UB <= best + margin`.
//!
//! # Numerical contract
//!
//! A Ritz value bounds `lambda_min` from above, never from below, so pruning is
//! always conservative: the search can never discard a subtree that contains a
//! strictly better subset. The reported `lambda_min` is the Ritz value of the
//! best subset found; callers that need the certified value should re-validate it
//! with a strict cold Lanczos (the Python binding does this when `verify=True`).

use rayon::prelude::*;
use std::collections::BTreeMap;
use std::time::Instant;

use crate::greedy::AlgoConfig;
use crate::lanczos::smallest_eigenpair;
use crate::matrix::DataMatrix;
use crate::op::GatheredZOp;

/// Search budget for [`exact_at`].
#[derive(Clone, Copy, Debug)]
pub struct ExactBudget {
    /// Wall-clock budget in seconds (`<= 0.0` = unlimited).
    pub time_budget_s: f64,
    /// Maximum number of `lambda_min` evaluations (`0` = unlimited).
    pub max_evals: usize,
}

impl Default for ExactBudget {
    fn default() -> Self {
        Self {
            time_budget_s: 0.0,
            max_evals: 0,
        }
    }
}

impl ExactBudget {
    /// True if the budget is unlimited.
    pub fn unlimited(&self) -> bool {
        self.time_budget_s <= 0.0 && self.max_evals == 0
    }
}

/// Outcome of an exact (or budget-limited) search.
#[derive(Clone, Debug)]
pub struct ExactOutcome {
    /// Best subset found, as original column indices, sorted.
    pub subset: Vec<usize>,
    /// `lambda_min` of that subset (best Ritz value seen).
    pub lambda_min: f64,
    /// True when the search finished: the optimum is proven.
    pub proved_optimal: bool,
    /// Certified **relative** optimality gap `(UB - LB) / LB`, `0.0` when proven.
    pub gap_certified: f64,
    /// Number of `lambda_min` evaluations performed.
    pub evaluations: usize,
    /// Number of internal nodes expanded.
    pub expanded: usize,
    /// Wall-clock seconds spent in the search.
    pub seconds: f64,
}

/// Open node of the search: a partial subset plus the range of candidate columns
/// still allowed (columns are visited in increasing order, so every `k`-subset is
/// generated exactly once).
struct Node {
    sel: Vec<u32>,
    start: usize,
    bound: f64,
    seed: Vec<f64>,
}

/// Order-preserving `f64 -> u64` map, used to key the bound multiset.
#[inline]
fn ord_key(v: f64) -> u64 {
    let b = v.to_bits();
    if b >> 63 == 1 {
        !b
    } else {
        b | (1u64 << 63)
    }
}

/// Inverse of [`ord_key`].
#[inline]
fn unord_key(kk: u64) -> f64 {
    let b = if kk >> 63 == 1 {
        kk & !(1u64 << 63)
    } else {
        !kk
    };
    f64::from_bits(b)
}

/// `lambda_min` of the Gram matrix of the columns in `sel`, plus its eigenvector.
///
/// Sizes 0 and 1 are exact and free: standardized columns have unit norm, so a
/// single variable gives `lambda_min = 1`.
fn eval_subset(
    z: &DataMatrix,
    sel: &[u32],
    seed: Option<&[f64]>,
    cfg: &AlgoConfig,
) -> (f64, Vec<f64>) {
    if sel.len() <= 1 {
        return (1.0, vec![1.0; sel.len().max(1)]);
    }
    let cols: Vec<&[f64]> = sel.iter().map(|&j| z.col(j as usize)).collect();
    let mut op = GatheredZOp::new(cols, z.rows);
    let iters = if seed.is_some() {
        cfg.max_iters_warm
    } else {
        cfg.max_iters_cold
    };
    let out = smallest_eigenpair(&mut op, seed, cfg.tol, iters);
    (out.value, out.vector)
}

/// Strictly re-evaluates a subset with a cold, longer Lanczos.
fn strict_value(z: &DataMatrix, sel: &[u32], cfg: &AlgoConfig) -> f64 {
    if sel.len() <= 1 {
        return 1.0;
    }
    let strict = AlgoConfig {
        tol: 1e-13,
        max_iters_cold: 4000,
        max_iters_warm: 400,
        ..*cfg
    };
    eval_subset(z, sel, None, &strict).0
}

/// Maximizes `lambda_min` over all subsets of size `k`.
///
/// `incumbent` (optional) is a warm start of size `k`: the search keeps it if it
/// is not beaten. A good incumbent is what makes the bound effective -- the greedy
/// family of [`crate::greedy`] is the intended provider, and the proof then costs
/// a small fraction of the exhaustive enumeration.
///
/// Returns the best subset found; `proved_optimal` is true when the whole space
/// was either explored or pruned, otherwise `gap_certified` bounds the remaining
/// relative improvement.
pub fn exact_at(
    z: &DataMatrix,
    k: usize,
    cfg: &AlgoConfig,
    budget: ExactBudget,
    incumbent: Option<&[usize]>,
) -> ExactOutcome {
    let t0 = Instant::now();
    let m = z.cols;
    assert!(k >= 1 && k <= m, "k invalide");
    let deadline = if budget.time_budget_s > 0.0 {
        Some(t0 + std::time::Duration::from_secs_f64(budget.time_budget_s))
    } else {
        None
    };
    let mut evaluations = 0usize;
    let mut expanded = 0usize;

    // ---- incumbent ----
    let mut best = f64::NEG_INFINITY;
    let mut best_set: Vec<u32> = Vec::new();
    if let Some(inc) = incumbent {
        if inc.len() == k {
            let sel: Vec<u32> = inc.iter().map(|&j| j as u32).collect();
            best = eval_subset(z, &sel, None, cfg).0;
            best_set = sel;
            evaluations += 1;
        }
    }

    if k == m {
        // The full set is the only subset of its size.
        let sel: Vec<u32> = (0..m as u32).collect();
        let strict = strict_value(z, &sel, cfg);
        return ExactOutcome {
            subset: (0..m).collect(),
            lambda_min: strict,
            proved_optimal: true,
            gap_certified: 0.0,
            evaluations,
            expanded,
            seconds: t0.elapsed().as_secs_f64(),
        };
    }

    // ---- depth-first branch and bound ----
    let margin = cfg.tol * 50.0 * (1.0 + best.abs().min(1.0));
    let mut stack: Vec<Node> = vec![Node {
        sel: Vec::new(),
        start: 0,
        bound: 1.0,
        seed: Vec::new(),
    }];
    let mut open_bounds: BTreeMap<u64, usize> = BTreeMap::new();
    open_bounds.insert(ord_key(1.0), 1);
    let mut proved = false;
    let mut ub;

    loop {
        // Certified upper bound = largest bound among open nodes.
        ub = open_bounds
            .last_key_value()
            .map(|(&kk, _)| unord_key(kk))
            .unwrap_or(f64::NEG_INFINITY);
        if stack.is_empty() || ub <= best + margin {
            proved = true;
            break;
        }
        if let Some(d) = deadline {
            if Instant::now() >= d {
                break;
            }
        }
        let node = stack.pop().expect("pile non vide");
        if let Some(c) = open_bounds.get_mut(&ord_key(node.bound)) {
            *c -= 1;
            if *c == 0 {
                open_bounds.remove(&ord_key(node.bound));
            }
        }
        if node.bound <= best + margin {
            continue;
        }
        if node.sel.len() == k {
            // Complete subset: its bound is exact.
            if node.bound > best {
                best = node.bound;
                best_set = node.sel.clone();
            }
            continue;
        }
        // Budget check before paying for a whole batch of children.
        let n_children = m.saturating_sub(node.start);
        if budget.max_evals > 0 && evaluations + n_children > budget.max_evals {
            *open_bounds.entry(ord_key(node.bound)).or_insert(0) += 1;
            stack.push(node);
            break;
        }
        let mut base_seed = node.seed.clone();
        if base_seed.len() < node.sel.len() {
            base_seed = vec![0.0; node.sel.len()];
        }
        let children: Vec<(f64, u32, Vec<f64>)> = (node.start..m)
            .into_par_iter()
            .map(|j| {
                let mut child = node.sel.clone();
                child.push(j as u32);
                let mut seed = base_seed.clone();
                seed.push(0.0);
                let (v, vec) = eval_subset(z, &child, Some(&seed), cfg);
                (v, j as u32, vec)
            })
            .collect();
        evaluations += children.len();
        expanded += 1;

        let mut kept: Vec<(f64, u32, Vec<f64>)> = children
            .into_iter()
            .filter(|&(v, _, _)| v > best + margin)
            .collect();
        kept.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        for (v, j, vec) in kept.into_iter().rev() {
            let mut sel = node.sel.clone();
            sel.push(j);
            *open_bounds.entry(ord_key(v)).or_insert(0) += 1;
            stack.push(Node {
                sel,
                start: (j as usize) + 1,
                bound: v,
                seed: vec,
            });
        }
    }

    // ---- report ----
    let ub_now = if proved {
        best
    } else {
        open_bounds
            .last_key_value()
            .map(|(&kk, _)| unord_key(kk))
            .unwrap_or(best)
            .max(best)
    };
    let lb = if best_set.is_empty() {
        f64::NAN
    } else {
        strict_value(z, &best_set, cfg)
    };
    let mut subset: Vec<usize> = best_set.iter().map(|&j| j as usize).collect();
    subset.sort_unstable();
    let gap = if proved {
        0.0
    } else if lb.is_finite() && lb > 0.0 {
        ((ub_now - lb) / lb).max(0.0)
    } else {
        f64::INFINITY
    };
    ExactOutcome {
        subset,
        lambda_min: lb,
        proved_optimal: proved,
        gap_certified: gap,
        evaluations,
        expanded,
        seconds: t0.elapsed().as_secs_f64(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gen::{generate, GenKind};
    use crate::jacobi::eigen_sym_sorted;

    fn data(seed: u64, m: usize) -> DataMatrix {
        let mut dm = generate(GenKind::Blocks, 200, m, 0.6, 0.05, 3, 1, seed);
        dm.standardize(true);
        dm
    }

    fn lmin_exact(z: &DataMatrix, sel: &[usize]) -> f64 {
        let d = sel.len();
        let mut a = vec![0.0; d * d];
        for (x, &ix) in sel.iter().enumerate() {
            for (y, &iy) in sel.iter().enumerate() {
                a[x * d + y] = crate::num::dot(z.col(ix), z.col(iy));
            }
        }
        let (vals, _) = eigen_sym_sorted(&mut a, d);
        vals[0]
    }

    fn brute(z: &DataMatrix, k: usize) -> f64 {
        let m = z.cols;
        let mut idx = vec![0usize; k];
        for (i, v) in idx.iter_mut().enumerate() {
            *v = i;
        }
        let mut best = f64::NEG_INFINITY;
        loop {
            let v = lmin_exact(z, &idx);
            if v > best {
                best = v;
            }
            // next combination in lexicographic order
            let mut i = k;
            while i > 0 {
                i -= 1;
                if idx[i] != i + m - k {
                    idx[i] += 1;
                    for j in (i + 1)..k {
                        idx[j] = idx[j - 1] + 1;
                    }
                    break;
                }
                if i == 0 {
                    return best;
                }
            }
        }
    }

    /// The exact search must return the brute-force optimum, for every structure.
    #[test]
    fn exact_matches_brute_force() {
        let cfg = AlgoConfig {
            tol: 1e-12,
            max_iters_cold: 2000,
            max_iters_warm: 400,
            ..Default::default()
        };
        for kind in [GenKind::Blocks, GenKind::Ar] {
            for seed in 0..3u64 {
                let mut dm = generate(kind, 150, 14, 0.7, 0.05, 3, 1, seed);
                dm.standardize(true);
                for k in [1usize, 2, 3, 5, 8, 13, 14] {
                    let outcome = exact_at(&dm, k, &cfg, ExactBudget::default(), None);
                    assert!(outcome.proved_optimal, "kind={kind:?} k={k}: non prouve");
                    assert_eq!(outcome.gap_certified, 0.0);
                    assert_eq!(outcome.subset.len(), k);
                    let got = lmin_exact(&dm, &outcome.subset);
                    let want = brute(&dm, k);
                    assert!(
                        (got - want).abs() < 1e-7,
                        "kind={kind:?} seed={seed} k={k}: exact {got} vs force brute {want}"
                    );
                }
            }
        }
    }

    /// A warm incumbent must not change the result, only the cost.
    #[test]
    fn exact_incumbent_only_speeds_up() {
        let cfg = AlgoConfig {
            tol: 1e-12,
            max_iters_cold: 2000,
            max_iters_warm: 400,
            ..Default::default()
        };
        let dm = data(7, 16);
        let k = 6;
        let cold = exact_at(&dm, k, &cfg, ExactBudget::default(), None);
        // greedy-ish incumbent: the k variables least correlated with the rest
        let mut order: Vec<usize> = (0..dm.cols).collect();
        order.sort_by(|&i, &j| {
            let mi = (0..dm.cols)
                .filter(|&x| x != i)
                .map(|x| crate::num::dot(dm.col(i), dm.col(x)).abs())
                .fold(0.0f64, f64::max);
            let mj = (0..dm.cols)
                .filter(|&x| x != j)
                .map(|x| crate::num::dot(dm.col(j), dm.col(x)).abs())
                .fold(0.0f64, f64::max);
            mi.partial_cmp(&mj).unwrap()
        });
        let inc: Vec<usize> = order[..k].to_vec();
        let warm = exact_at(&dm, k, &cfg, ExactBudget::default(), Some(&inc));
        assert!((cold.lambda_min - warm.lambda_min).abs() < 1e-9);
        assert!(warm.evaluations <= cold.evaluations);
    }

    /// When the budget runs out, the gap must be a valid bound on the remaining
    /// improvement, and the returned subset must be a real one.
    #[test]
    fn exact_budget_reports_valid_gap() {
        let cfg = AlgoConfig {
            tol: 1e-12,
            max_iters_cold: 2000,
            max_iters_warm: 400,
            ..Default::default()
        };
        let dm = data(3, 18);
        let k = 6;
        let full = exact_at(&dm, k, &cfg, ExactBudget::default(), None);
        assert!(full.proved_optimal);
        let limited = exact_at(
            &dm,
            k,
            &cfg,
            ExactBudget {
                time_budget_s: 0.0,
                max_evals: 200,
            },
            None,
        );
        assert!(!limited.proved_optimal);
        assert_eq!(limited.subset.len(), k);
        assert!(
            limited.evaluations <= 200,
            "budget depasse: {}",
            limited.evaluations
        );
        assert!(limited.gap_certified.is_finite() && limited.gap_certified >= 0.0);
        let lb = lmin_exact(&dm, &limited.subset);
        let ub = lb * (1.0 + limited.gap_certified) + 1e-9;
        assert!(
            full.lambda_min <= ub,
            "the optimum {} must stay below the bound {}",
            full.lambda_min,
            ub
        );
    }
}

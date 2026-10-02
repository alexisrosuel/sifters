//! Lanczos (full reorthogonalization) for the **smallest** eigenvalue
//! of a symmetric operator, with warm start.
//!
//! This is the building block that makes the greedy selection fast:
//!  * between two steps, the sought eigenvector varies very little, so a
//!    warm start converges in a few iterations;
//!  * at each iteration, the smallest eigenvalue of the tridiagonal `T_j`
//!    is obtained by bisection/Sturm in `O(50 j)`, which gives an almost
//!    free convergence test (without re-diagonalization `O(j^3)`).

use crate::jacobi::tridiag_smallest;
use crate::num::{axpy, dot, norm2};
pub use crate::op::SymOp;

/// Result of a Lanczos call.
#[derive(Clone, Debug)]
pub struct LanczosOutcome {
    /// Approximate eigenvalue (Ritz value).
    pub value: f64,
    /// Associated eigenvector (norm 1).
    pub vector: Vec<f64>,
    /// `||A v - value * v||` evaluated exactly on the final vector.
    pub residual: f64,
    /// Number of iterations performed.
    pub iters: usize,
    /// True if the convergence criterion is reached.
    pub converged: bool,
}

/// Deterministic seed if the caller does not provide one.
pub fn default_seed(n: usize, tag: u64) -> Vec<f64> {
    let mut s = tag.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 * (1.0 / (1u64 << 53) as f64) - 0.5
        })
        .collect()
}

/// Smallest eigenvalue (and eigenvector) of `op`.
///
/// * `seed`: start vector (typically the previous eigenvector);
/// * `tol` : relative tolerance on the stabilization of the Ritz value;
/// * `max_iters`: maximum number of Lanczos iterations.
pub fn smallest_eigenpair(
    op: &mut dyn SymOp,
    seed: Option<&[f64]>,
    tol: f64,
    max_iters: usize,
) -> LanczosOutcome {
    smallest_eigenpair_constrained(op, seed, tol, max_iters, &[])
}

/// Same, but restricting the search space to the orthogonal complement of
/// `constraints` (deflation by explicit orthogonalization).
///
/// Each new basis vector is reorthogonalized against `constraints`, which
/// avoids the catastrophic amplification of the spurious component when the
/// Lanczos coefficient `beta` becomes small (unlike the breakdown by projected
/// operator `(I-P)A`, whose trial vector keeps the component and gets
/// amplified by `alpha/beta`).
pub fn smallest_eigenpair_constrained(
    op: &mut dyn SymOp,
    seed: Option<&[f64]>,
    tol: f64,
    max_iters: usize,
    constraints: &[Vec<f64>],
) -> LanczosOutcome {
    let n = op.n();
    assert!(n > 0, "zero-size operator");
    if n == 1 {
        let mut y = [0.0];
        op.mul(&[1.0], &mut y);
        return LanczosOutcome {
            value: y[0],
            vector: vec![1.0],
            residual: 0.0,
            iters: 1,
            converged: true,
        };
    }

    let maxm = max_iters.clamp(1, n);
    let mut v: Vec<f64> = match seed {
        Some(s) if s.len() == n && norm2(s) > 0.0 => s.to_vec(),
        _ => default_seed(n, n as u64),
    };
    let nv = norm2(&v);
    for x in v.iter_mut() {
        *x /= nv;
    }
    if !constraints.is_empty() {
        for _ in 0..2 {
            for c in constraints {
                let d = dot(c, &v);
                axpy(-d, c, &mut v);
            }
        }
        let nv = norm2(&v);
        if nv > 0.0 {
            for x in v.iter_mut() {
                *x /= nv;
            }
        }
    }

    let mut basis: Vec<f64> = Vec::with_capacity(maxm * n);
    let mut alphas: Vec<f64> = Vec::with_capacity(maxm);
    let mut betas: Vec<f64> = Vec::with_capacity(maxm);
    let mut w = vec![0.0; n];
    let mut prev = vec![0.0; n];

    let mut iters = 0usize;
    let mut converged = false;
    let mut beta_prev = 0.0f64;

    for j in 0..maxm {
        basis.extend_from_slice(&v);
        op.mul(&v, &mut w);
        let alpha = dot(&v, &w);
        axpy(-alpha, &v, &mut w);
        if j > 0 {
            axpy(-beta_prev, &prev, &mut w);
        }
        // Full reorthogonalization (two passes: numerical stability) then
        // projection onto the complement of the deflation constraints.
        //
        // The second pass runs only if the first one made the norm of `w` drop
        // (a sign of catastrophic cancellation, hence of a loss of
        // orthogonality): classical criterion, threshold `1/sqrt(2)`. Warm
        // iterations, where `w` loses little, thus save a full pass over the basis.
        let nrm_before = norm2(&w);
        for pass in 0..2 {
            for k in 0..=j {
                let vk = &basis[k * n..(k + 1) * n];
                let c = dot(vk, &w);
                axpy(-c, vk, &mut w);
            }
            for c in constraints {
                let d = dot(c, &w);
                if d != 0.0 {
                    axpy(-d, c, &mut w);
                }
            }
            if pass == 0 && (j == 0 || norm2(&w) > std::f64::consts::FRAC_1_SQRT_2 * nrm_before) {
                break;
            }
        }
        alphas.push(alpha);
        if j > 0 {
            betas.push(beta_prev);
        }
        iters = j + 1;
        beta_prev = norm2(&w);
        // Detection of an invariant subspace: below this relative threshold,
        // `w` is nothing but rounding noise and adding it as a basis vector
        // would make spurious eigenvalues appear.
        if beta_prev <= 1e-9 * (1.0 + alpha.abs()) {
            converged = true;
            break;
        }
        // `theta` is used only for the convergence test (the final value is
        // recomputed below with 60 bisection steps): 40 steps are enough to
        // make it more accurate than the tolerance. The bisection only needs the
        // predicate "at least one eigenvalue below `mu`", hence the early
        // exit of the Sturm sequence.
        let m = alphas.len();
        let tb = &betas[..m.saturating_sub(1)];
        let theta = tridiag_smallest(&alphas, tb, 40);
        // Stopping criterion **on the Ritz residual** `||A u - theta u|| = beta_j |y_j|`
        // (`y_j` = last component of the tridiagonal eigenvector). This is the
        // measure that guarantees the quality of the *value*: a simple test of
        // stabilization of `theta` can trigger on a plateau while the
        // residual is still large, which overestimates `lambda_min` and corrupts the
        // certification of the greedy step.
        let y = crate::jacobi::tridiag_smallest_eigenvector(&alphas, tb, theta, 3);
        if beta_prev * y[m - 1].abs() <= tol * (1.0 + theta.abs()) {
            converged = true;
            break;
        }
        prev.copy_from_slice(&v);
        for i in 0..n {
            v[i] = w[i] / beta_prev;
        }
    }

    // Ritz vector: smallest eigenvalue of the tridiagonal by bisection
    // (Sturm), then eigenvector by inverse iteration: `O(m)` instead of the
    // Jacobi diagonalization `O(m^3)` done on each call.
    let m = alphas.len();
    let tb = &betas[..m.saturating_sub(1)];
    let value = tridiag_smallest(&alphas, tb, 60);
    let y = crate::jacobi::tridiag_smallest_eigenvector(&alphas, tb, value, 6);
    let mut u = vec![0.0; n];
    for k in 0..m {
        axpy(y[k], &basis[k * n..(k + 1) * n], &mut u);
    }
    let nu = norm2(&u);
    if nu > 0.0 {
        for x in u.iter_mut() {
            *x /= nu;
        }
    }

    // exact residual
    let mut au = vec![0.0; n];
    op.mul(&u, &mut au);
    axpy(-value, &u, &mut au);
    let residual = norm2(&au);

    LanczosOutcome {
        value,
        vector: u,
        residual,
        iters,
        converged,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gen::{generate, GenKind};
    use crate::jacobi::eigen_sym_sorted;
    use crate::op::{Dataset, Repr, SubOp};

    #[test]
    fn lanczos_matches_jacobi_on_random_correlation() {
        let mut dm = generate(GenKind::Blocks, 400, 40, 0.6, 0.05, 5, 1, 99);
        dm.standardize(true);
        let ds = Dataset::new(dm, Repr::Packed, usize::MAX, 16);
        let k = 40;
        let mut op = SubOp::head(&ds, k);
        let out = smallest_eigenpair(&mut op, None, 1e-12, 400);
        // Jacobi reference
        let mut dense = vec![0.0; k * k];
        for i in 0..k {
            for j in 0..k {
                dense[i * k + j] = ds.packed.as_ref().unwrap().get(i, j);
            }
        }
        let (vals, _) = eigen_sym_sorted(&mut dense, k);
        assert!(
            (out.value - vals[0]).abs() < 1e-9,
            "lanczos {} jacobi {}",
            out.value,
            vals[0]
        );
        // The Ritz vector residual is typically much larger than the error
        // on the eigenvalue (which converges quadratically): it is an indicator of
        // vector quality, not of the value. The bounds of the cascade remain
        // rigorous whatever this residual.
        assert!(out.residual < 1e-6, "residual {}", out.residual);
        assert!((out.value - vals[0]).abs() < 1e-9);
    }

    #[test]
    fn warm_start_is_cheaper_than_cold() {
        let mut dm = generate(GenKind::Equi, 300, 60, 0.2, 0.0, 1, 1, 5);
        dm.standardize(true);
        let ds = Dataset::new(dm, Repr::Packed, usize::MAX, 16);
        let k = 60;
        let mut op = SubOp::head(&ds, k);
        let cold = smallest_eigenpair(&mut op, None, 1e-12, 500);
        let mut op2 = SubOp::head(&ds, k);
        let warm = smallest_eigenpair(&mut op2, Some(&cold.vector), 1e-12, 500);
        assert!(warm.iters <= cold.iters);
        assert!((warm.value - cold.value).abs() < 1e-10);
    }
}

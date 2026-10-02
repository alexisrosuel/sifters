//! Small dense spectral solvers used by Lanczos and by the tests:
//! * `eigen_sym`: cyclic Jacobi (very robust, `O(n^3)`, `n` small);
//! * `tridiag_smallest`: smallest eigenvalue of a symmetric tridiagonal
//!   matrix by bisection + Sturm sequence (`O(iters * n)`), used at each
//!   Lanczos iteration for an almost free convergence test.

/// Eigendecomposition of a dense symmetric `n x n` matrix stored row-major.
///
/// Returns `(eigenvalues, eigenvectors as columns)`; `a` is destroyed.
pub fn eigen_sym(a: &mut [f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    if n == 0 {
        return (Vec::new(), v);
    }
    if n == 1 {
        return (vec![a[0]], v);
    }
    for _sweep in 0..100 {
        let mut off = 0.0;
        for p in 0..n - 1 {
            for q in p + 1..n {
                off += a[p * n + q] * a[p * n + q];
            }
        }
        let diag: f64 = (0..n).map(|i| a[i * n + i] * a[i * n + i]).sum();
        if off <= 1e-30 * diag.max(1e-300) {
            break;
        }
        for p in 0..n - 1 {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let theta = (aqq - app) / (2.0 * apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    -1.0 / (-theta + (1.0 + theta * theta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                // rotation A <- J^T A J
                for k in 0..n {
                    let akp = a[k * n + p];
                    let akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p * n + k];
                    let aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                // V <- V J: after k rotations, A_k = V^T A V with V = J_1...J_k,
                // so A = V A_k V^T and the eigenvectors of A are the columns
                // of V (update on columns p and q, like A <- A J).
                for k in 0..n {
                    let vkp = v[k * n + p];
                    let vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let vals: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    (vals, v)
}

/// Eigenvalues sorted in increasing order, associated vectors (columns).
pub fn eigen_sym_sorted(a: &mut [f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let (vals, vecs) = eigen_sym(a, n);
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| {
        vals[i]
            .partial_cmp(&vals[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let sv: Vec<f64> = idx.iter().map(|&i| vals[i]).collect();
    // Convention: `vecs[e * n + k]` = element `e` of eigenvector `k` (columns).
    let mut svec = vec![0.0; n * n];
    for (newc, &oldc) in idx.iter().enumerate() {
        for e in 0..n {
            svec[e * n + newc] = vecs[e * n + oldc];
        }
    }
    (sv, svec)
}

/// Number of eigenvalues of `T` (symmetric tridiagonal) strictly
/// less than `mu`, via the Sturm sequence (`O(n)`).
pub fn sturm_count(alphas: &[f64], betas: &[f64], mu: f64) -> usize {
    let n = alphas.len();
    if n == 0 {
        return 0;
    }
    let mut d = alphas[0] - mu;
    let mut cnt = if d < 0.0 { 1 } else { 0 };
    for i in 1..n {
        if d == 0.0 {
            // infinitesimal perturbation: avoids division by zero
            d = f64::MIN_POSITIVE * (1.0 + alphas[i].abs());
        }
        d = (alphas[i] - mu) - betas[i - 1] * betas[i - 1] / d;
        if d < 0.0 {
            cnt += 1;
        }
    }
    cnt
}

/// True if there exists **at least one** eigenvalue of `T` less than `mu`
/// (Sturm sequence, `O(n)` but with early exit).
///
/// The Sturm counter is increasing: as soon as a pivot becomes negative, the
/// answer is settled and the rest of the recurrence can be abandoned. The
/// bisection of [`tridiag_smallest`], called at each Lanczos iteration, needs
/// only this predicate.
#[inline]
pub fn sturm_has_negative(alphas: &[f64], betas: &[f64], mu: f64) -> bool {
    let n = alphas.len();
    if n == 0 {
        return false;
    }
    let mut d = alphas[0] - mu;
    if d < 0.0 {
        return true;
    }
    for i in 1..n {
        if d == 0.0 {
            d = f64::MIN_POSITIVE * (1.0 + alphas[i].abs());
        }
        d = (alphas[i] - mu) - betas[i - 1] * betas[i - 1] / d;
        if d < 0.0 {
            return true;
        }
    }
    false
}

/// Smallest eigenvalue of a symmetric tridiagonal (bisection).
///
/// `alphas`: diagonal (`n`), `betas[j]`: coupling between `j` and `j+1` (`n-1`).
pub fn tridiag_smallest(alphas: &[f64], betas: &[f64], iters: usize) -> f64 {
    let n = alphas.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return alphas[0];
    }
    // Gershgorin bound on the tridiagonal (much tighter than on R).
    let mut radius = 0.0f64;
    for i in 0..n {
        let mut r = alphas[i].abs();
        if i > 0 {
            r += betas[i - 1].abs();
        }
        if i + 1 < n {
            r += betas[i].abs();
        }
        radius = radius.max(r);
    }
    if !radius.is_finite() {
        return alphas[0];
    }
    let (mut lo, mut hi) = (-radius - 1.0, radius + 1.0);
    for _ in 0..iters {
        let mid = 0.5 * (lo + hi);
        if sturm_has_negative(alphas, betas, mid) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    hi
}

/// Rebuilds the dense matrix from a tridiagonal (for Jacobi).
pub fn tridiag_to_dense(alphas: &[f64], betas: &[f64]) -> Vec<f64> {
    let n = alphas.len();
    let mut t = vec![0.0; n * n];
    for i in 0..n {
        t[i * n + i] = alphas[i];
        if i + 1 < n {
            t[i * n + i + 1] = betas[i];
            t[(i + 1) * n + i] = betas[i];
        }
    }
    t
}

/// Eigenvector associated with the **smallest** eigenvalue of a tridiagonal,
/// by inverse iteration (`theta` must be an approximation of that value,
/// typically the result of [`tridiag_smallest`]).
///
/// Cost `O(iters * n)`: replaces a Jacobi diagonalization `O(n^3)` in the
/// warm loop of Lanczos, where `n` (number of iterations) can reach several
/// hundreds.
pub fn tridiag_smallest_eigenvector(
    alphas: &[f64],
    betas: &[f64],
    theta: f64,
    iters: usize,
) -> Vec<f64> {
    let n = alphas.len();
    if n == 0 {
        return Vec::new();
    }
    let start = vec![1.0f64 / (n as f64).sqrt(); n];
    tridiag_inverse_iterate(alphas, betas, theta, &start, iters)
}

/// `iters` inverse iterations on `(T - theta I) z = y` starting from `start`,
/// with LU factorization (`O(n)`) done only once. Returns normalized `z`.
///
/// Used to maintain the eigenpair of the Lanczos tridiagonal from one
/// iteration to the next: starting from the previous eigenvector extended, two or
/// three iterations suffice, against ~50 Sturm sweeps for a complete
/// bisection.
pub fn tridiag_inverse_iterate(
    alphas: &[f64],
    betas: &[f64],
    theta: f64,
    start: &[f64],
    iters: usize,
) -> Vec<f64> {
    let n = alphas.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let mut l = vec![0.0f64; n];
    let mut z = vec![0.0f64; n];
    let mut y = vec![0.0f64; n];
    y.copy_from_slice(&start[..n]);
    let ny = crate::num::norm2(&y);
    if ny > 0.0 {
        for v in y.iter_mut() {
            *v /= ny;
        }
    } else {
        for v in y.iter_mut() {
            *v = 1.0 / (n as f64).sqrt();
        }
    }
    l[0] = alphas[0] - theta;
    for i in 1..n {
        if l[i - 1].abs() < 1e-300 {
            l[i - 1] = 1e-300;
        }
        // The tridiagonal overflows the length of the provided betas (`n-1`).
        let b = betas[i - 1];
        l[i] = (alphas[i] - theta) - b * b / l[i - 1];
    }
    if l[n - 1].abs() < 1e-300 {
        l[n - 1] = 1e-300;
    }
    for _ in 0..iters.max(1) {
        z[0] = y[0] / l[0];
        for i in 1..n {
            z[i] = (y[i] - betas[i - 1] * z[i - 1]) / l[i];
        }
        for i in (0..n - 1).rev() {
            z[i] -= betas[i] * z[i + 1] / l[i];
        }
        let nz = crate::num::norm2(&z);
        if !(nz > 0.0) || !nz.is_finite() {
            break;
        }
        for i in 0..n {
            y[i] = z[i] / nz;
        }
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jacobi_matches_known_spectrum() {
        // 3x3 symmetric matrix with a known spectrum
        let a0 = [2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0];
        let mut a = a0;
        let (v, _) = eigen_sym(&mut a, 3);
        let mut v = v;
        v.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let expect = [2.0 - 2f64.sqrt(), 2.0, 2.0 + 2f64.sqrt()];
        for i in 0..3 {
            assert!((v[i] - expect[i]).abs() < 1e-12, "{:?} vs {:?}", v, expect);
        }
    }

    #[test]
    fn sorted_eigenvectors_are_consistent() {
        let n = 5;
        let a0 = [
            4.0, 1.0, 0.5, 0.2, 0.0, 1.0, 3.0, 1.0, 0.5, 0.2, 0.5, 1.0, 5.0, 1.0, 0.5, 0.2, 0.5,
            1.0, 6.0, 1.0, 0.0, 0.2, 0.5, 1.0, 7.0,
        ];
        let mut a = a0;
        let (vals, vecs) = eigen_sym_sorted(&mut a, n);
        for k in 0..n {
            let v: Vec<f64> = (0..n).map(|e| vecs[e * n + k]).collect();
            for i in 0..n {
                let av: f64 = (0..n).map(|j| a0[i * n + j] * v[j]).sum();
                assert!((av - vals[k] * v[i]).abs() < 1e-10, "k={k} i={i}");
            }
        }
    }

    #[test]
    fn tridiag_bisection_matches_jacobi() {
        let alphas = [0.5, -1.0, 2.0, 0.25, -0.5, 1.5];
        let betas = [0.3, -0.7, 0.2, 0.9, -0.1];
        let t = tridiag_to_dense(&alphas, &betas);
        let mut t2 = t.clone();
        let (v, _) = eigen_sym(&mut t2, alphas.len());
        let exact = v.iter().cloned().fold(f64::INFINITY, f64::min);
        let bis = tridiag_smallest(&alphas, &betas, 200);
        assert!((bis - exact).abs() < 1e-10, "bis={bis} exact={exact}");
    }

    #[test]
    fn sturm_count_is_correct() {
        let alphas = [1.0, 1.0, 1.0];
        let betas = [1.0, 1.0];
        // spectrum: 1-sqrt2, 1, 1+sqrt2
        assert_eq!(sturm_count(&alphas, &betas, -10.0), 0);
        assert_eq!(sturm_count(&alphas, &betas, 0.0), 1);
        assert_eq!(sturm_count(&alphas, &betas, 0.5), 1);
        assert_eq!(sturm_count(&alphas, &betas, 1.5), 2);
        assert_eq!(sturm_count(&alphas, &betas, 10.0), 3);
    }
}

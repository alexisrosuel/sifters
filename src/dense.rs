//! Dense symmetric eigendecomposition, used when the **whole** low spectrum of a
//! materialized block is needed (certified forward selection, `p = k`).
//!
//! Householder reduction to tridiagonal form (`tred2`) followed by implicit QL
//! with shifts (`tqli`), after EISPACK / *Numerical Recipes*. Cost `O(n^3)` with
//! a small constant, deterministic, orthonormal eigenvectors by construction.
//! This replaces `n` constrained Lanczos runs in the whole-spectrum case: those
//! pay the same `O(n^3)` for the matvecs, plus another `O(n^3)` for the deflation
//! projections, plus the tridiagonal bisections.

/// Eigendecomposition of a dense symmetric `n x n` matrix stored row-major.
///
/// Returns `(eigenvalues sorted increasingly, eigenvectors as columns)`:
/// `values[l]` and `vecs[e * n + l]` = element `e` of the associated
/// eigenvector (norm 1). Only the lower triangle of `a` is significant, and `a`
/// is left untouched.
pub fn eigen_sym_ql(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    if n == 1 {
        return (vec![a[0]], vec![1.0]);
    }

    // The historical routines are 1-based; we keep their exact indexing on
    // arrays of size `n + 1` (index 0 unused) to minimize transcription risk.
    let w = n + 1;
    let mut t = vec![0.0f64; w * w];
    for i in 1..=n {
        for j in 1..=n {
            t[i * w + j] = a[(i - 1) * n + (j - 1)];
        }
    }
    let mut d = vec![0.0f64; w];
    let mut e = vec![0.0f64; w];

    // ---- tred2: Householder reduction to tridiagonal form ----
    for i in (2..=n).rev() {
        let l = i - 1;
        let mut h = 0.0f64;
        let mut scale = 0.0f64;
        if l > 1 {
            for k in 1..=l {
                scale += t[i * w + k].abs();
            }
            if scale == 0.0 {
                e[i] = t[i * w + l];
            } else {
                for k in 1..=l {
                    let v = t[i * w + k] / scale;
                    t[i * w + k] = v;
                    h += v * v;
                }
                let f0 = t[i * w + l];
                let g0 = if f0 >= 0.0 { -h.sqrt() } else { h.sqrt() };
                e[i] = scale * g0;
                h -= f0 * g0;
                t[i * w + l] = f0 - g0;
                let mut f = 0.0f64;
                for j in 1..=l {
                    t[j * w + i] = t[i * w + j] / h;
                    let mut g = 0.0f64;
                    for k in 1..=j {
                        g += t[j * w + k] * t[i * w + k];
                    }
                    for k in (j + 1)..=l {
                        g += t[k * w + j] * t[i * w + k];
                    }
                    e[j] = g / h;
                    f += e[j] * t[i * w + j];
                }
                let hh = f / (h + h);
                for j in 1..=l {
                    let fj = t[i * w + j];
                    let gj = e[j] - hh * fj;
                    e[j] = gj;
                    for k in 1..=j {
                        t[j * w + k] -= fj * e[k] + gj * t[i * w + k];
                    }
                }
            }
        } else {
            e[i] = t[i * w + l];
        }
        d[i] = h;
    }
    d[1] = 0.0;
    e[1] = 0.0;

    // Accumulate the transformations: the columns of `t` become the
    // eigenvector basis that `tqli` will rotate.
    for i in 1..=n {
        let l = i - 1;
        if d[i] != 0.0 {
            for j in 1..=l {
                let mut g = 0.0f64;
                for k in 1..=l {
                    g += t[i * w + k] * t[k * w + j];
                }
                for k in 1..=l {
                    t[k * w + j] -= g * t[k * w + i];
                }
            }
        }
        d[i] = t[i * w + i];
        t[i * w + i] = 1.0;
        for j in 1..=l {
            t[j * w + i] = 0.0;
            t[i * w + j] = 0.0;
        }
    }

    // ---- tqli: implicit QL with shifts ----
    for i in 2..=n {
        e[i - 1] = e[i];
    }
    e[n] = 0.0;
    for l in 1..=n {
        for _iter in 0..60 {
            // First index `m` from which `e` is negligible.
            let mut m = l;
            while m < n {
                let dd = d[m].abs() + d[m + 1].abs();
                if e[m].abs() + dd == dd {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let r0 = g.hypot(1.0);
            g = d[m] - d[l] + e[l] / (g + if g >= 0.0 { r0 } else { -r0 });
            let mut s = 1.0f64;
            let mut c = 1.0f64;
            let mut p = 0.0f64;
            let mut i = m;
            let mut zero_break = false;
            while i > l {
                i -= 1;
                let f = s * e[i];
                let b = c * e[i];
                let r = f.hypot(g);
                e[i + 1] = r;
                if r == 0.0 {
                    // Recover from an underflow.
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    zero_break = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                let r2 = (d[i] - g) * s + 2.0 * c * b;
                p = s * r2;
                d[i + 1] = g + p;
                g = c * r2 - b;
                // Rotate the eigenvector columns `i` and `i + 1`.
                for k in 1..=n {
                    let f2 = t[k * w + i + 1];
                    t[k * w + i + 1] = s * t[k * w + i] + c * f2;
                    t[k * w + i] = c * t[k * w + i] - s * f2;
                }
            }
            if zero_break {
                continue;
            }
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }

    // ---- sort increasingly and flatten to the 0-based output layout ----
    let mut idx: Vec<usize> = (1..=n).collect();
    idx.sort_by(|&i, &j| d[i].partial_cmp(&d[j]).unwrap_or(std::cmp::Ordering::Equal));
    let values: Vec<f64> = idx.iter().map(|&i| d[i]).collect();
    // `vecs[e * n + l]` = element `e` of eigenvector `l`.
    let mut vecs = vec![0.0f64; n * n];
    for (l, &col) in idx.iter().enumerate() {
        for k in 1..=n {
            vecs[(k - 1) * n + l] = t[k * w + col];
        }
    }
    (values, vecs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gen::{generate, GenKind};

    /// The dense solver must agree with the historical Jacobi, every pair must
    /// satisfy `A v = lambda v` to machine precision, and the vectors must be
    /// orthonormal (the cascade bounds rely on it).
    #[test]
    fn ql_matches_jacobi_and_residuals() {
        for n in 1..=10usize {
            for seed in 0..4u64 {
                let mut a = vec![0.0f64; n * n];
                let mut rng = crate::gen::Rng::new(seed * 31 + n as u64);
                for i in 0..n {
                    for j in 0..=i {
                        let v = if i == j {
                            1.0 + 2.0 * rng.uniform()
                        } else {
                            0.3 * (rng.uniform() - 0.5)
                        };
                        a[i * n + j] = v;
                        a[j * n + i] = v;
                    }
                }
                let a0 = a.clone();
                let mut b = a.clone();
                let (vals_j, _) = crate::jacobi::eigen_sym(&mut b, n);
                let mut vals_j = vals_j;
                vals_j.sort_by(|x, y| x.partial_cmp(y).unwrap());
                let (vals, vecs) = eigen_sym_ql(&a, n);
                assert_eq!(vals.len(), n);
                for l in 0..n {
                    assert!(
                        (vals[l] - vals_j[l]).abs() < 1e-10,
                        "n={n} seed={seed} l={l}: ql {} vs jacobi {}",
                        vals[l],
                        vals_j[l]
                    );
                    for i in 0..n {
                        let av: f64 = (0..n).map(|j| a0[i * n + j] * vecs[j * n + l]).sum();
                        assert!(
                            (av - vals[l] * vecs[i * n + l]).abs() < 1e-10,
                            "n={n} seed={seed} l={l} i={i}"
                        );
                    }
                    let nrm: f64 = (0..n).map(|e| vecs[e * n + l] * vecs[e * n + l]).sum();
                    assert!((nrm - 1.0).abs() < 1e-12, "n={n} l={l} norm={nrm}");
                }
                // orthonormality of the whole basis
                for l in 0..n {
                    for m in (l + 1)..n {
                        let dp: f64 = (0..n).map(|e| vecs[e * n + l] * vecs[e * n + m]).sum();
                        assert!(dp.abs() < 1e-10, "n={n} <{l},{m}>={dp}");
                    }
                }
                for w in vals.windows(2) {
                    assert!(w[0] <= w[1] + 1e-14);
                }
            }
        }
    }

    /// The solver must be usable on a real correlation block (diagonal exactly 1,
    /// eigenvalues in `[0, k]`).
    #[test]
    fn ql_on_correlation_block() {
        let mut dm = generate(GenKind::Blocks, 200, 40, 0.4, 0.05, 4, 1, 7);
        dm.standardize(true);
        let ds = crate::op::Dataset::new(dm, crate::op::Repr::Packed, usize::MAX, 16);
        let pk = ds.packed.as_ref().expect("packed");
        for k in [2usize, 7, 40] {
            let mut a = vec![0.0f64; k * k];
            for i in 0..k {
                for j in 0..=i {
                    let v = pk.get(i, j);
                    a[i * k + j] = v;
                    a[j * k + i] = v;
                }
            }
            let (vals, _) = eigen_sym_ql(&a, k);
            assert!(vals[0] >= -1e-9, "k={k} lambda_min={}", vals[0]);
            assert!(vals[k - 1] <= k as f64 + 1e-9);
        }
    }
}

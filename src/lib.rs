//! `sifters` - spectral feature selection in Rust: variable subset selection by
//! maximization of the smallest eigenvalue of the correlation matrix
//! (E-optimal / min-eigenvalue criterion).
//!
//! The crate is entirely `safe`: `#![forbid(unsafe_code)]`.
#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]
// Deliberate numeric idioms: kernels index several arrays in lockstep
// (`needless_range_loop`), take many scalar arguments (`too_many_arguments`),
// and use `!(x > 0.0)` on purpose because it also catches NaN, unlike
// `x <= 0.0` (`neg_cmp_op_on_partial_ord`).
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::neg_cmp_op_on_partial_ord)]

/// Dense symmetric eigendecomposition (whole-spectrum fast path).
pub mod dense;
/// Exact branch and bound over subsets of a fixed size.
pub mod exact;
/// Synthetic data generators: used only by the crate's unit
/// tests (real data comes through the Python binding).
#[cfg(test)]
pub mod gen;
pub mod greedy;
pub mod inverse;
pub mod jacobi;
pub mod lanczos;
pub mod matrix;
pub mod op;
pub mod packed;

/// Small shared numerical utility.
pub mod num {
    /// 4-way unrolled dot product (automatic vectorization).
    #[inline]
    pub fn dot(a: &[f64], b: &[f64]) -> f64 {
        debug_assert_eq!(a.len(), b.len());
        let n = a.len();
        let nb = n / 4;
        let (mut a0, mut a1, mut a2, mut a3) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for i in 0..nb {
            let o = 4 * i;
            a0 += a[o] * b[o];
            a1 += a[o + 1] * b[o + 1];
            a2 += a[o + 2] * b[o + 2];
            a3 += a[o + 3] * b[o + 3];
        }
        let mut s = (a0 + a1) + (a2 + a3);
        for i in (4 * nb)..n {
            s += a[i] * b[i];
        }
        s
    }

    /// `y <- y + alpha * x`
    #[inline]
    pub fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
        debug_assert_eq!(x.len(), y.len());
        if alpha == 0.0 {
            return;
        }
        for (yi, &xi) in y.iter_mut().zip(x.iter()) {
            *yi += alpha * xi;
        }
    }

    /// 4-way unrolled Euclidean norm.
    #[inline]
    pub fn norm2(x: &[f64]) -> f64 {
        dot(x, x).max(0.0).sqrt()
    }

    /// Sum of squares.
    #[inline]
    pub fn sumsq(x: &[f64]) -> f64 {
        dot(x, x)
    }
}

//! `minvp` — selection de sous-ensembles de variables par maximisation de la plus
//! petite valeur propre de la matrice de correlation (critere E-optimal).
//!
//! Le crate est integralement `safe` : `#![forbid(unsafe_code)]`.
#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod cli;
pub mod gen;
pub mod greedy;
pub mod inverse;
pub mod io;
pub mod jacobi;
pub mod lanczos;
pub mod matrix;
pub mod op;
pub mod packed;
pub mod report;

/// Petit utilitaire numerique partage.
pub mod num {
    /// Produit scalaire 4-way unrolled (vectorisation automatique).
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

    /// Norme euclidienne 4-way unrolled.
    #[inline]
    pub fn norm2(x: &[f64]) -> f64 {
        dot(x, x).max(0.0).sqrt()
    }

    /// Somme des carres.
    #[inline]
    pub fn sumsq(x: &[f64]) -> f64 {
        dot(x, x)
    }
}

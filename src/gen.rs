//! Reproducible synthetic data generators (and a minimal dependency-free RNG).

use crate::matrix::DataMatrix;
use std::f64::consts::PI;

/// xorshift64*: fast, deterministic, sufficient for noise generation.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    /// Creates an RNG from a seed (the null state is corrected).
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform on [0, 1).
    #[inline]
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Standard normal distribution (Box-Muller).
    #[inline]
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

/// Families of synthetic matrices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GenKind {
    /// i.i.d. Gaussian columns (correlation ~ 0).
    Iid,
    /// Equi-correlated: `r_ij = rho` off the diagonal.
    Equi,
    /// First-order autoregressive: `r_ij = rho^|i-j|` (Toeplitz).
    Ar,
    /// Block structure: `rho_in` within the block, `rho_out` between blocks.
    Blocks,
    /// Factor model: `rank` latent factors + noise.
    Factor,
}

/// Generates a matrix `n x m` (observations x variables) according to `kind`.
pub fn generate(
    kind: GenKind,
    n: usize,
    m: usize,
    rho: f64,
    rho_out: f64,
    blocks: usize,
    rank: usize,
    seed: u64,
) -> DataMatrix {
    let mut rng = Rng::new(seed);
    let mut data = vec![0.0; n * m]; // column-major
    match kind {
        GenKind::Iid => {
            for v in data.iter_mut() {
                *v = rng.normal();
            }
        }
        GenKind::Equi => {
            // x_ij = sqrt(rho) g_i + sqrt(1-rho) e_ij
            let a = rho.max(0.0).sqrt();
            let b = (1.0 - rho).max(0.0).sqrt();
            for i in 0..n {
                let g = rng.normal();
                for j in 0..m {
                    data[j * n + i] = a * g + b * rng.normal();
                }
            }
        }
        GenKind::Ar => {
            // AR(1) filter applied along the VARIABLES axis:
            // x_j = rho x_{j-1} + sqrt(1-rho^2) e_j  =>  corr(x_i, x_j) = rho^|i-j|.
            let s = (1.0 - rho * rho).max(0.0).sqrt();
            for i in 0..n {
                let mut x = rng.normal();
                data[i] = x;
                for j in 1..m {
                    x = rho * x + s * rng.normal();
                    data[j * n + i] = x;
                }
            }
        }
        GenKind::Blocks => {
            let b = blocks.max(1);
            let a = rho.max(0.0).sqrt();
            let c = rho_out.max(0.0).sqrt();
            let s = (1.0 - rho).max(0.0).sqrt();
            for i in 0..n {
                let common = rng.normal();
                let mut blockfac = vec![0.0; b];
                for f in blockfac.iter_mut() {
                    *f = rng.normal();
                }
                for j in 0..m {
                    let blk = (j * b) / m.max(1);
                    let g = c * common + a * blockfac[blk.min(b - 1)];
                    data[j * n + i] = g + s * rng.normal();
                }
            }
        }
        GenKind::Factor => {
            let r = rank.max(1);
            let mut loadings = vec![0.0; m * r];
            for v in loadings.iter_mut() {
                *v = rng.normal();
            }
            for i in 0..n {
                let mut f = vec![0.0; r];
                for v in f.iter_mut() {
                    *v = rng.normal();
                }
                for j in 0..m {
                    let mut s = 0.0;
                    for k in 0..r {
                        s += loadings[j * r + k] * f[k];
                    }
                    data[j * n + i] = s + rng.normal();
                }
            }
        }
    }
    DataMatrix::from_col_major(n, m, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ar1_has_toeplitz_correlation() {
        let n = 20_000;
        let m = 4;
        let rho = 0.7;
        let mut dm = generate(GenKind::Ar, n, m, rho, 0.0, 1, 1, 42);
        dm.standardize(true);
        for j in 0..m {
            for k in 0..m {
                let r = crate::num::dot(dm.col(j), dm.col(k));
                let expect = rho.powi((j as i32 - k as i32).abs());
                assert!((r - expect).abs() < 0.02, "r({j},{k})={r} expect {expect}");
            }
        }
    }
}

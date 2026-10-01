//! Inverse de la matrice de corrélation, maintenue **incrémentalement**.
//!
//! # Pourquoi
//!
//! La plus petite valeur propre `lambda_min(A)` d'une matrice est la plus **grande**
//! valeur propre de `A^{-1}`, égale à `1/lambda_min(A)`. Or l'inversion amplifie
//! considérablement les écarts relatifs du bas du spectre : si
//! `{0.0050, 0.0057, 0.0061, ...}` est un amas proche de 0, l'inverse donne
//! `{200, 175, 164, ...}` dont les écarts relatifs (~12 %) sont exploitables par
//! Lanczos en quelques itérations, là où il en fallait 60 à 200 sur la matrice
//! d'origine. C'est l'effet « shift-invert » sans factorisation par candidat.
//!
//! # Mise à jour
//!
//! Soit `W = R^{-1}` et `A = R_{-i}` le bloc privé de l'indice `i`. En partitionnant
//! `R = [[A, b],[b^T, R_ii]]`, l'identité d'inverse par blocs donne
//!
//! ```text
//! W = [[ A^{-1} + (A^{-1}b)(A^{-1}b)^T/s ,  -A^{-1}b/s ],
//!      [ -b^T A^{-1}/s                    ,  1/s       ]]   avec s = 1/W_ii
//! ```
//!
//! d'où la mise à jour **rang 1** (`O(k^2)` par suppression) :
//!
//! ```text
//! A^{-1} = W_{-i,-i} - (1/W_ii) w w^T ,     w = W_{.,i}|_{. != i}
//! ```
//!
//! La même identité appliquée à l'opérateur donne le produit matrice-vecteur de
//! `A^{-1}` **sans former A** : `y = W_{-i,-i} x - (w^T x) w / W_ii`.

use crate::num::dot;
use crate::packed::PackedSym;

/// Inverse (dense, symétrique) du bloc de tête d'une matrice de corrélation.
#[derive(Clone, Debug)]
pub struct InverseSym {
    m: usize,
    w: Vec<f64>,
}

impl InverseSym {
    /// Inverse du bloc de tête `m x m` de `p`, par factorisation de Cholesky.
    ///
    /// Retourne `None` si le bloc n'est pas défini positif (corrélation singulière,
    /// cas fréquent quand `M > N`).
    pub fn from_packed(p: &PackedSym, m: usize) -> Option<Self> {
        if m == 0 {
            return Some(Self { m: 0, w: Vec::new() });
        }
        let scale = (0..m).map(|i| p.get(i, i).abs()).fold(1.0f64, f64::max);
        let eps = 1e-13 * scale;
        // Cholesky R = L L^T (lignes-major, triangle inférieur)
        let mut l = vec![0.0f64; m * m];
        for i in 0..m {
            for j in 0..=i {
                let mut s = p.get(i, j);
                for k in 0..j {
                    s -= l[i * m + k] * l[j * m + k];
                }
                if i == j {
                    if !(s > eps) {
                        return None;
                    }
                    l[i * m + i] = s.sqrt();
                } else {
                    l[i * m + j] = s / l[j * m + j];
                }
            }
        }
        // Inverse de L (triangulaire inférieure)
        let mut li = vec![0.0f64; m * m];
        for i in 0..m {
            li[i * m + i] = 1.0 / l[i * m + i];
            for j in 0..i {
                let mut s = 0.0;
                for k in j..i {
                    s += l[i * m + k] * li[k * m + j];
                }
                li[i * m + j] = -s / l[i * m + i];
            }
        }
        // W = L^{-T} L^{-1}
        let mut w = vec![0.0f64; m * m];
        for i in 0..m {
            for j in 0..=i {
                let mut s = 0.0;
                for k in i..m {
                    s += li[k * m + i] * li[k * m + j];
                }
                w[i * m + j] = s;
                w[j * m + i] = s;
            }
        }
        Some(Self { m, w })
    }

    /// Dimension initiale (pas de stockage).
    pub fn dim(&self) -> usize {
        self.m
    }

    /// Accès `(i, j)`.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.w[i * self.m + j]
    }

    /// Échange les indices `i` et `j` dans le bloc de tête `k` (même permutation que
    /// `PackedSym::swap_leading`, appliquée à l'inverse).
    pub fn swap_leading(&mut self, i: usize, j: usize, k: usize) {
        if i == j {
            return;
        }
        let (i, j) = if i < j { (i, j) } else { (j, i) };
        for c in 0..k {
            if c != i && c != j {
                self.w.swap(c * self.m + i, c * self.m + j);
                self.w.swap(i * self.m + c, j * self.m + c);
            }
        }
        self.w.swap(i * self.m + i, j * self.m + j);
        // Le stockage est plein et symetrique : on recopie l'entree `(i, j)` sur sa
        // symetrique `(j, i)`.
        self.w[j * self.m + i] = self.w[i * self.m + j];
    }

    /// Supprime le dernier indice du bloc de tête `k` : le bloc devient `k-1`.
    ///
    /// `A^{-1} = W_{-i,-i} - (1/W_ii) w w^T` avec `i = k-1`.
    pub fn downdate_last(&mut self, k: usize) {
        if k <= 1 {
            return;
        }
        let m = self.m;
        let i = k - 1;
        let wii = self.w[i * m + i];
        if wii.abs() < 1e-300 {
            return;
        }
        let inv = 1.0 / wii;
        let col: Vec<f64> = (0..i).map(|j| self.w[j * m + i]).collect();
        for a in 0..i {
            let ca = col[a] * inv;
            for b in 0..=a {
                self.w[a * m + b] -= ca * col[b];
            }
            for b in 0..a {
                self.w[b * m + a] = self.w[a * m + b];
            }
        }
    }

    /// `y = W_{-del,-del} x - (w^T x) w / W_{del,del}` (longueur `k-1`), où
    /// `W` est le bloc de tête `k x k`. Sans `del`, `y = W x`.
    ///
    /// `y` reçoit l'**opposé** si `negate` est vrai (utile pour chercher la plus petite
    /// valeur propre de `-A^{-1}`, c'est-à-dire la plus grande de `A^{-1}`).
    ///
    /// La voie `del` évite toute branche dans la boucle `O(k^2)` : les produits
    /// scalaires sont scindés en deux segments contigus de part et d'autre de
    /// l'indice supprimé, ce qui les rend vectorisables.
    pub fn mul(&self, k: usize, del: Option<usize>, x: &[f64], y: &mut [f64], negate: bool) {
        let m = self.m;
        debug_assert!(k <= m);
        let sign = if negate { -1.0 } else { 1.0 };
        match del {
            None => {
                for i in 0..k {
                    let row = &self.w[i * m..i * m + k];
                    y[i] = sign * dot(row, x);
                }
            }
            Some(d) => {
                let xa = &x[..d];
                let xb = &x[d..k - 1];
                // y[0..k-1] = W_{-d,-d} x, deux segments contigus par ligne.
                for t in 0..d {
                    let row = &self.w[t * m..t * m + k];
                    y[t] = dot(&row[..d], xa) + dot(&row[d + 1..k], xb);
                }
                for t in d..k - 1 {
                    let row = &self.w[(t + 1) * m..(t + 1) * m + k];
                    y[t] = dot(&row[..d], xa) + dot(&row[d + 1..k], xb);
                }
                // correction rang 1 : (w^T x) w / W_dd
                let wdd = self.w[d * m + d];
                if wdd.abs() > 1e-300 {
                    let mut c = 0.0;
                    for (u, &xu) in xa.iter().enumerate() {
                        c += self.w[u * m + d] * xu;
                    }
                    for (u, &xu) in xb.iter().enumerate() {
                        c += self.w[(d + 1 + u) * m + d] * xu;
                    }
                    let f = c / wdd;
                    for t in 0..d {
                        y[t] -= f * self.w[t * m + d];
                    }
                    for t in d..k - 1 {
                        y[t] -= f * self.w[(t + 1) * m + d];
                    }
                }
                if negate {
                    for yi in y.iter_mut().take(k - 1) {
                        *yi = -*yi;
                    }
                }
            }
        }
    }
}

/// Opérateur `x -> -W_{-d,-d} x + (w^T x) w / W_dd` : l'opposé de l'inverse de la
/// sous-matrice privée de `d`. Sa plus **petite** valeur propre vaut
/// `-1/lambda_min(R_{-d})`.
#[derive(Debug)]
pub struct NegInvSubOp<'a> {
    inv: &'a InverseSym,
    k: usize,
    del: Option<usize>,
    dim: usize,
}

impl<'a> NegInvSubOp<'a> {
    /// Opérateur sur le bloc `0..k` privé de `del` (ou entier si `del = None`).
    pub fn new(inv: &'a InverseSym, k: usize, del: Option<usize>) -> Self {
        let dim = if del.is_some() { k.saturating_sub(1) } else { k };
        Self { inv, k, del, dim }
    }
}

impl crate::op::SymOp for NegInvSubOp<'_> {
    fn n(&self) -> usize {
        self.dim
    }
    fn mul(&mut self, x: &[f64], y: &mut [f64]) {
        self.inv.mul(self.k, self.del, x, y, true);
    }
}

/// Bornes de Gershgorin sur les valeurs propres de `W` (utilitaire de test).
pub fn row_abs_sum(inv: &InverseSym, k: usize, i: usize) -> f64 {
    (0..k).map(|j| inv.get(i, j).abs()).sum::<f64>() - inv.get(i, i).abs()
}

/// `y += alpha * row_i(W)` (utilitaire).
pub fn row_axpy(inv: &InverseSym, k: usize, i: usize, alpha: f64, y: &mut [f64]) {
    for j in 0..k {
        y[j] += alpha * inv.get(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gen::{generate, GenKind};
    use crate::packed::PackedSym;

    fn dataset(kind: GenKind, n: usize, m: usize, rho: f64, seed: u64) -> PackedSym {
        let mut dm = generate(kind, n, m, rho, 0.05, 4, 1, seed);
        dm.standardize(true);
        PackedSym::correlation(&dm, 16)
    }

    #[test]
    fn inverse_matches_explicit_solve() {
        let m = 30;
        let p = dataset(GenKind::Blocks, 200, m, 0.5, 7);
        let inv = InverseSym::from_packed(&p, m).expect("PD");
        // W R = I
        for i in 0..m {
            for j in 0..m {
                let mut s = 0.0;
                for k in 0..m {
                    s += inv.get(i, k) * p.get(k, j);
                }
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!((s - expect).abs() < 1e-10, "({i},{j}) {s}");
            }
        }
    }

    #[test]
    fn swap_leading_matches_permutation() {
        let m = 6;
        let p = dataset(GenKind::Blocks, 200, m, 0.5, 17);
        let mut inv = InverseSym::from_packed(&p, m).expect("PD");
        let before: Vec<f64> = (0..m * m).map(|t| inv.w[t]).collect();
        let (i, j) = (1usize, 4usize);
        inv.swap_leading(i, j, m);
        for a in 0..m {
            for b in 0..m {
                let pa = if a == i { j } else if a == j { i } else { a };
                let pb = if b == i { j } else if b == j { i } else { b };
                let expect = before[pa * m + pb];
                assert!(
                    (inv.get(a, b) - expect).abs() < 1e-14,
                    "({a},{b}) {} vs {expect}",
                    inv.get(a, b)
                );
            }
        }
    }

    #[test]
    fn downdate_last_only() {
        let m = 6;
        let p = dataset(GenKind::Blocks, 200, m, 0.5, 23);
        let inv0 = InverseSym::from_packed(&p, m).expect("PD");
        let before: Vec<f64> = (0..m * m).map(|t| inv0.w[t]).collect();
        let mut inv = inv0.clone();
        inv.downdate_last(m);
        let k = m - 1;
        for i in 0..k {
            for j in 0..k {
                let mut s = 0.0;
                for l in 0..k {
                    s += inv.get(i, l) * p.get(l, j);
                }
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!((s - expect).abs() < 1e-9, "({i},{j}) {s}");
            }
        }
        // la formule : W' = W - col col^T / W_ii
        for a in 0..k {
            for b in 0..k {
                let expect = before[a * m + b]
                    - before[a * m + k] * before[k * m + b] / before[k * m + k];
                assert!((inv.get(a, b) - expect).abs() < 1e-12, "({a},{b})");
            }
        }
    }

    #[test]
    fn downdate_error_growth() {
        for (m, steps, n) in [(60usize, 40usize, 200usize), (200, 150, 400)] {
            let p = dataset(GenKind::Blocks, n, m, 0.6, 31);
            let mut inv = InverseSym::from_packed(&p, m).expect("PD");
            let mut perm: Vec<usize> = (0..m).collect();
            let mut k = m;
            let mut rng = crate::gen::Rng::new(5);
            let mut worst = 0.0f64;
            while k > m - steps && k > 2 {
                let d = (rng.next_u64() as usize) % k;
                inv.swap_leading(d, k - 1, k);
                perm.swap(d, k - 1);
                inv.downdate_last(k);
                k -= 1;
                // erreur max de W R - I sur un echantillon
                for i in (0..k).step_by((k / 8).max(1)) {
                    for j in (0..k).step_by((k / 8).max(1)) {
                        let mut s = 0.0;
                        for l in 0..k {
                            s += inv.get(i, l) * p.get(perm[l], perm[j]);
                        }
                        let e = if i == j { 1.0 } else { 0.0 };
                        worst = worst.max((s - e).abs());
                    }
                }
            }
            eprintln!("m={m} steps={steps} -> erreur max W R - I = {worst:.3e} (k final {k})");
            assert!(worst < 1e-6, "derive trop forte : {worst:.3e}");
        }
    }

    #[test]
    fn downdate_matches_explicit_submatrix_inverse() {
        let m = 24;
        let p = dataset(GenKind::Blocks, 200, m, 0.5, 11);
        let mut inv = InverseSym::from_packed(&p, m).expect("PD");
        let mut perm: Vec<usize> = (0..m).collect();
        let mut k = m;
        // on retire les indices dans un ordre quelconque
        let mut rng = crate::gen::Rng::new(3);
        while k > 2 {
            let d = (rng.next_u64() as usize) % k;
            inv.swap_leading(d, k - 1, k);
            perm.swap(d, k - 1);
            inv.downdate_last(k);
            k -= 1;
            // vérifie W = (R_perm[0..k])^{-1}
            for i in 0..k {
                for j in 0..k {
                    let mut s = 0.0;
                    for l in 0..k {
                        s += inv.get(i, l) * p.get(perm[l], perm[j]);
                    }
                    let expect = if i == j { 1.0 } else { 0.0 };
                    assert!((s - expect).abs() < 1e-8, "k={k} ({i},{j}) {s}");
                }
            }
        }
    }

    #[test]
    fn mul_sub_matches_explicit_inverse() {
        let m = 18;
        let p = dataset(GenKind::Equi, 200, m, 0.4, 5);
        let inv = InverseSym::from_packed(&p, m).expect("PD");
        let k = m;
        let x: Vec<f64> = (0..k - 1).map(|i| (i as f64 * 0.3).sin()).collect();
        for d in 0..k {
            let mut y = vec![0.0; k - 1];
            inv.mul(k, Some(d), &x, &mut y, false);
            // référence : inverse explicite du bloc privé de d
            let idx: Vec<usize> = (0..k).filter(|&j| j != d).collect();
            let d2 = idx.len();
            let mut sub = vec![0.0; d2 * d2];
            for a in 0..d2 {
                for b in 0..d2 {
                    sub[a * d2 + b] = p.get(idx[a], idx[b]);
                }
            }
            // x = A y  =>  y attendu = A^{-1} x : on résout par Cholesky
            let mut l = vec![0.0; d2 * d2];
            for i in 0..d2 {
                for j in 0..=i {
                    let mut s = sub[i * d2 + j];
                    for t in 0..j {
                        s -= l[i * d2 + t] * l[j * d2 + t];
                    }
                    if i == j {
                        l[i * d2 + i] = s.sqrt();
                    } else {
                        l[i * d2 + j] = s / l[j * d2 + j];
                    }
                }
            }
            let mut z = x.clone();
            for i in 0..d2 {
                for j in 0..i {
                    z[i] -= l[i * d2 + j] * z[j];
                }
                z[i] /= l[i * d2 + i];
            }
            for i in (0..d2).rev() {
                for j in i + 1..d2 {
                    z[i] -= l[j * d2 + i] * z[j];
                }
                z[i] /= l[i * d2 + i];
            }
            for a in 0..d2 {
                assert!((y[a] - z[a]).abs() < 1e-9, "d={d} a={a}");
            }
        }
    }

    #[test]
    fn inverse_smallest_eigenvalue_matches_mul() {
        // la plus grande valeur propre de A^{-1} = 1/lambda_min(A)
        let m = 16;
        let p = dataset(GenKind::Blocks, 300, m, 0.6, 13);
        let inv = InverseSym::from_packed(&p, m).expect("PD");
        let k = m;
        let mut op = NegInvSubOp::new(&inv, k, None);
        let out = crate::lanczos::smallest_eigenpair(&mut op, None, 1e-13, 200);
        let lam_min = -1.0 / out.value;
        // référence par Jacobi
        let mut a = vec![0.0; k * k];
        for i in 0..k {
            for j in 0..k {
                a[i * k + j] = p.get(i, j);
            }
        }
        let (vals, _) = crate::jacobi::eigen_sym_sorted(&mut a, k);
        assert!((lam_min - vals[0]).abs() < 1e-9, "{lam_min} vs {}", vals[0]);
    }
}

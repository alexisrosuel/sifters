//! Operateurs lineaires symetriques et representations de la matrice de correlation.
//!
//! Deux representations sont disponibles, choisies automatiquement :
//!
//! * `Packed`  : la matrice de correlation est materialisee en triangle inferieur.
//!   Cout d'un produit matrice-vecteur `O(k^2)`, construction `O(N M^2)`.
//! * `Implicit`: on garde `Z` (`N x M`, colonnes normees) et on applique
//!   `R x = Z_S^T (Z_S x)`. Cout d'un matvec `O(N k)`, aucune materialisation.
//!   Optimal des que `2N < k`.

use crate::matrix::DataMatrix;
use crate::num::{axpy, dot};
use crate::packed::PackedSym;
use rayon::prelude::*;

/// Operateur symetrique de dimension `n` (matvec `y = A x`).
///
/// `&mut self` permet a chaque operateur d'embarquer ses buffers de travail sans
/// allocation dans la boucle chaude ; cela rend aussi l'objet `Send` pour rayon.
pub trait SymOp {
    /// Dimension de l'operateur.
    fn n(&self) -> usize;
    /// `y <- A x`.
    fn mul(&mut self, x: &[f64], y: &mut [f64]);
}

/// Representation de la matrice de correlation du jeu de donnees.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Repr {
    /// Detecte automatiquement (packed si le budget memoire le permet).
    Auto,
    /// Triangle inferieur materialise.
    Packed,
    /// Implicite via `Z`.
    Implicit,
}

/// Jeu de donnees + ordre physique courant des colonnes.
///
/// L'ordre physique est celui du bloc de tete : les `k` premieres positions sont
/// les variables actives. `active[pos]` donne l'indice d'origine.
#[derive(Debug)]
pub struct Dataset {
    /// Colonnes normees, `N x M`, colonnes-major.
    pub z: DataMatrix,
    /// Triangle inferieur de `Z^T Z` (optionnel).
    pub packed: Option<PackedSym>,
    /// Position physique -> indice de colonne d'origine.
    pub active: Vec<usize>,
}

impl Dataset {
    /// Construit le jeu de donnees, en materialisant la correlation si demande /
    /// si le budget memoire le permet.
    pub fn new(z: DataMatrix, want: Repr, mem_budget_bytes: usize, block_rows: usize) -> Self {
        let m = z.cols;
        let need = crate::packed::packed_len(m) * std::mem::size_of::<f64>();
        let materialize = match want {
            Repr::Packed => true,
            Repr::Implicit => false,
            Repr::Auto => need <= mem_budget_bytes,
        };
        let packed = if materialize { Some(PackedSym::correlation(&z, block_rows)) } else { None };
        let active = (0..m).collect();
        Self { z, packed, active }
    }

    /// Nombre de variables courantes.
    #[inline]
    pub fn m(&self) -> usize {
        self.z.cols
    }

    /// Echange les positions physiques `i` et `j` dans toutes les representations.
    pub fn swap(&mut self, i: usize, j: usize) {
        if i == j {
            return;
        }
        self.active.swap(i, j);
        // `PackedSym::swap_leading` exige `i < j` : on normalise l'ordre (l'echange
        // est symetrique, mais le stockage packed ne l'est pas).
        let (a, b) = if i < j { (i, j) } else { (j, i) };
        if let Some(p) = self.packed.as_mut() {
            p.swap_leading(a, b);
        }
        let n = self.z.rows;
        let (left, right) = self.z.data.split_at_mut(b * n);
        let ca = &mut left[a * n..a * n + n];
        let cb = &mut right[..n];
        ca.swap_with_slice(cb);
    }

    /// Indice de colonne d'origine de la position `pos`.
    #[inline]
    pub fn orig(&self, pos: usize) -> usize {
        self.active[pos]
    }

    /// Vrai si la correlation est materialisee.
    pub fn is_packed(&self) -> bool {
        self.packed.is_some()
    }
}

/// Vue "sous-matrice principale" : bloc de tete `0..k`, eventuellement prive de
/// l'indice `del` (suppression d'une variable candidate).
#[derive(Debug)]
pub struct SubOp<'a> {
    packed: Option<&'a PackedSym>,
    z: &'a DataMatrix,
    k: usize,
    del: Option<usize>,
    dim: usize,
    x_full: Vec<f64>,
    y_full: Vec<f64>,
    tmp: Vec<f64>,
}

impl<'a> SubOp<'a> {
    /// Operateur sur le bloc `0..k` (sans suppression).
    pub fn head(ds: &'a Dataset, k: usize) -> Self {
        Self::make(ds, k, None)
    }

    /// Operateur sur le bloc `0..k` prive de l'indice `del`.
    pub fn deleted(ds: &'a Dataset, k: usize, del: usize) -> Self {
        Self::make(ds, k, Some(del))
    }

    fn make(ds: &'a Dataset, k: usize, del: Option<usize>) -> Self {
        let dim = match del {
            Some(_) => k.saturating_sub(1),
            None => k,
        };
        Self {
            packed: ds.packed.as_ref(),
            z: &ds.z,
            k,
            del,
            dim,
            x_full: vec![0.0; k],
            y_full: vec![0.0; k],
            tmp: vec![0.0; ds.z.rows],
        }
    }

    /// Applique la sous-matrice au vecteur `x` de dimension `dim`.
    fn apply(&mut self, x: &[f64]) {
        let k = self.k;
        match self.del {
            None => self.x_full[..k].copy_from_slice(&x[..k]),
            Some(d) => {
                self.x_full[..k].fill(0.0);
                self.x_full[..d].copy_from_slice(&x[..d]);
                self.x_full[d + 1..k].copy_from_slice(&x[d..k - 1]);
            }
        }
        if let Some(p) = self.packed {
            p.matvec_head(k, &self.x_full, &mut self.y_full);
        } else {
            let rows = self.z.rows;
            self.tmp[..rows].fill(0.0);
            for c in 0..k {
                let xc = self.x_full[c];
                if xc != 0.0 {
                    axpy(xc, self.z.col(c), &mut self.tmp[..rows]);
                }
            }
            for c in 0..k {
                self.y_full[c] = dot(self.z.col(c), &self.tmp[..rows]);
            }
        }
    }
}

impl SymOp for SubOp<'_> {
    fn n(&self) -> usize {
        self.dim
    }

    fn mul(&mut self, x: &[f64], y: &mut [f64]) {
        self.apply(x);
        match self.del {
            None => y[..self.k].copy_from_slice(&self.y_full[..self.k]),
            Some(d) => {
                y[..d].copy_from_slice(&self.y_full[..d]);
                y[d..].copy_from_slice(&self.y_full[d + 1..self.k]);
            }
        }
    }
}

/// `out[0..rows] <- Z_{0..k} x` (chargements de la sous-matrice courante).
pub fn z_loading(z: &DataMatrix, k: usize, x: &[f64], out: &mut [f64]) {
    let rows = z.rows;
    out[..rows].fill(0.0);
    for c in 0..k {
        let xc = x[c];
        if xc != 0.0 {
            axpy(xc, z.col(c), &mut out[..rows]);
        }
    }
}

/// `out[j] <- z_j . w` pour toutes les colonnes, en parallele (O(N M)).
pub fn z_dot_all(z: &DataMatrix, w: &[f64], out: &mut [f64]) {
    let rows = z.rows;
    out.par_iter_mut()
        .zip(z.data.par_chunks(rows))
        .for_each(|(o, col)| *o = dot(col, w));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gen::{generate, GenKind};

    #[test]
    fn packed_and_implicit_matvec_agree() {
        let mut dm = generate(GenKind::Blocks, 120, 31, 0.5, 0.15, 3, 1, 17);
        dm.standardize(true);
        let ds_p = Dataset::new(dm.clone(), Repr::Packed, usize::MAX, 16);
        let ds_i = Dataset::new(dm.clone(), Repr::Implicit, 0, 16);
        assert!(ds_p.is_packed() && !ds_i.is_packed());
        let k = 31;
        let x: Vec<f64> = (0..k).map(|i| ((i * 13 % 7) as f64) - 3.0).collect();
        let mut yp = vec![0.0; k];
        let mut yi = vec![0.0; k];
        SubOp::head(&ds_p, k).mul(&x, &mut yp);
        SubOp::head(&ds_i, k).mul(&x, &mut yi);
        for i in 0..k {
            assert!((yp[i] - yi[i]).abs() < 1e-10, "i={i}");
        }
    }

    #[test]
    fn deleted_op_matches_explicit_submatrix() {
        let mut dm = generate(GenKind::Equi, 90, 17, 0.35, 0.0, 1, 1, 23);
        dm.standardize(true);
        let ds = Dataset::new(dm.clone(), Repr::Packed, usize::MAX, 16);
        let k = 17;
        for del in 0..k {
            let mut op = SubOp::deleted(&ds, k, del);
            let x: Vec<f64> = (0..k - 1).map(|i| (i as f64 * 0.37).sin()).collect();
            let mut y = vec![0.0; k - 1];
            op.mul(&x, &mut y);
            for a in 0..k - 1 {
                let ia = if a < del { a } else { a + 1 };
                let mut expect = 0.0;
                for b in 0..k - 1 {
                    let ib = if b < del { b } else { b + 1 };
                    expect += dm.col(ia).iter().zip(dm.col(ib)).map(|(p, q)| p * q).sum::<f64>() * x[b];
                }
                assert!((y[a] - expect).abs() < 1e-11, "del={del} a={a}");
            }
        }
    }

    /// Regression : `Dataset::swap(i, j)` doit etre valide quel que soit l'ordre
    /// des arguments. Le stockage packed n'est pas symetrique en `(i, j)`, donc
    /// passer `i > j` a `PackedSym::swap_leading` corrompt la matrice — ce que
    /// fait tout appel qui amene une variable situee *avant* la position cible
    /// (verification du sous-ensemble retenu, echanges locaux arbitraires...).
    #[test]
    fn dataset_swap_is_order_insensitive() {
        let mut dm = generate(GenKind::Blocks, 120, 17, 0.5, 0.1, 3, 1, 21);
        dm.standardize(true);
        let m = dm.cols;
        let mut ds = Dataset::new(dm.clone(), Repr::Packed, usize::MAX, 8);
        let mut perm: Vec<usize> = (0..m).collect();
        let mut rng = crate::gen::Rng::new(7);
        for t in 0..60 {
            let i = (rng.next_u64() as usize) % m;
            let j = (rng.next_u64() as usize) % m;
            // on alterne volontairement les deux ordres d'appel
            if t % 2 == 0 {
                ds.swap(i, j);
            } else {
                ds.swap(j, i);
            }
            perm.swap(i, j);
            assert_eq!(ds.active, perm, "permutation des indices d'origine");
            let p = ds.packed.as_ref().expect("representation packed");
            for a in 0..m {
                for b in 0..m {
                    let expect = crate::num::dot(dm.col(perm[a]), dm.col(perm[b]));
                    assert!(
                        (p.get(a, b) - expect).abs() < 1e-12,
                        "t={t} a={a} b={b} : {} vs {expect}",
                        p.get(a, b)
                    );
                }
            }
            // les colonnes de `z` suivent la meme permutation
            for a in 0..m {
                for r in 0..dm.rows {
                    assert!((ds.z.col(a)[r] - dm.col(perm[a])[r]).abs() < 1e-15);
                }
            }
        }
    }
}

/// Operateur de la matrice bordee `[[R_S, c],[c^T, 1]]` ou `S` est le bloc de tete
/// `0..k` et `c` la colonne `j` de la correlation (`j >= k`).
///
/// Quand `R` est materialisee, le matvec coute `O(k^2)` (donnees residentes en
/// cache) au lieu de `O(N k)` pour [`GatheredZOp`] : c'est la representation a
/// privilegier pour la selection avant des que `k` n'est plus petit devant `2N`.
#[derive(Debug)]
pub struct BorderedHeadOp<'a> {
    packed: &'a PackedSym,
    k: usize,
    col: Vec<f64>,
}

impl<'a> BorderedHeadOp<'a> {
    /// Construit l'operateur pour `S = 0..k` et le candidat `j`.
    pub fn new(packed: &'a PackedSym, k: usize, j: usize) -> Self {
        debug_assert!(j >= k && j < packed.dim());
        // `j >= k` : la colonne est stockee de facon contigue dans la ligne `j`.
        let base = crate::packed::row_offset(j);
        let col = packed.raw()[base..base + k].to_vec();
        Self { packed, k, col }
    }
}

impl SymOp for BorderedHeadOp<'_> {
    fn n(&self) -> usize {
        self.k + 1
    }

    fn mul(&mut self, x: &[f64], y: &mut [f64]) {
        let k = self.k;
        self.packed.matvec_head(k, &x[..k], &mut y[..k]);
        let xk = x[k];
        for i in 0..k {
            y[i] += self.col[i] * xk;
        }
        y[k] = dot(&self.col, &x[..k]) + xk;
    }
}

/// Operateur "colonne selectionnees" : sous-matrice de Gram `C^T C` ou `C` est une
/// liste arbitraire de colonnes de `Z`. Sert a la selection avant (ajout d'une
/// variable candidate) et aux echanges locaux, sans modifier l'ordre physique.
#[derive(Debug)]
pub struct GatheredZOp<'a> {
    cols: Vec<&'a [f64]>,
    rows: usize,
    tmp: Vec<f64>,
}

impl<'a> GatheredZOp<'a> {
    /// Construit l'operateur depuis une liste de colonnes (references).
    pub fn new(cols: Vec<&'a [f64]>, rows: usize) -> Self {
        Self { cols, rows, tmp: vec![0.0; rows] }
    }

    /// Construit l'operateur pour `S = {0..k}\{del}` eventuellement augmente de
    /// `extra` (position physique hors du bloc de tete).
    pub fn build(z: &'a DataMatrix, k: usize, del: Option<usize>, extra: Option<usize>) -> Self {
        let mut cols: Vec<&'a [f64]> = Vec::with_capacity(k + 1);
        for c in 0..k {
            if Some(c) != del {
                cols.push(z.col(c));
            }
        }
        if let Some(j) = extra {
            cols.push(z.col(j));
        }
        Self::new(cols, z.rows)
    }
}

impl SymOp for GatheredZOp<'_> {
    fn n(&self) -> usize {
        self.cols.len()
    }

    fn mul(&mut self, x: &[f64], y: &mut [f64]) {
        self.tmp[..self.rows].fill(0.0);
        for (c, col) in self.cols.iter().enumerate() {
            let xc = x[c];
            if xc != 0.0 {
                axpy(xc, col, &mut self.tmp[..self.rows]);
            }
        }
        for (c, col) in self.cols.iter().enumerate() {
            y[c] = dot(col, &self.tmp[..self.rows]);
        }
    }
}

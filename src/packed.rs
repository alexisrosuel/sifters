//! Matrice symetrique stockee en triangle inferieur "packed" (row-major).
//!
//! La ligne `i` contient `i+1` coefficients, a l'offset `i*(i+1)/2`.
//! La diagonale vaut 1 (matrice de correlation) mais reste stockee pour simplifier
//! les acces et permettre un usage generique.
//!
//! Proprietes cles exploitees par le solveur :
//!  * la sous-matrice active est **toujours le bloc de tete** `0..k` ;
//!  * supprimer une variable = une permutation de lignes/colonnes `i <-> k-1`
//!    restreinte au bloc de tete (cout `O(k)`, pas de recompaction `O(M^2)`).

use crate::matrix::DataMatrix;
use crate::num::{axpy, dot};
use rayon::prelude::*;

/// Offset du debut de la ligne `i` dans le buffer packed.
#[inline]
pub const fn row_offset(i: usize) -> usize {
    i * (i + 1) / 2
}

/// Nombre d'entrees du triangle inferieur d'une matrice `m x m`.
#[inline]
pub const fn packed_len(m: usize) -> usize {
    m * (m + 1) / 2
}

/// Matrice symetrique `m x m` en triangle inferieur.
#[derive(Clone, Debug)]
pub struct PackedSym {
    m: usize,
    data: Vec<f64>,
}

impl PackedSym {
    /// Matrice nulle de taille `m` (diagonale non initialisee a 1).
    pub fn zeros(m: usize) -> Self {
        Self { m, data: vec![0.0; packed_len(m)] }
    }

    /// Taille.
    #[inline]
    pub fn dim(&self) -> usize {
        self.m
    }

    /// Acces au buffer brut.
    #[inline]
    pub fn raw(&self) -> &[f64] {
        &self.data
    }

    /// Acces `(i, j)` symetrique.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        if i >= j {
            self.data[row_offset(i) + j]
        } else {
            self.data[row_offset(j) + i]
        }
    }

    /// Construit la matrice de correlation `Z^T Z` depuis les colonnes deja
    /// normalisees de `dm`, en parallele par blocs de lignes.
    ///
    /// `block_rows` controle la taille du bloc de lignes traite par tache : un bloc
    /// reste resident en cache pendant qu'on balaye les blocs de colonnes, ce qui
    /// reduit le trafic memoire d'un facteur ~`M / block_rows`.
    pub fn correlation(dm: &DataMatrix, block_rows: usize) -> Self {
        let m = dm.cols;
        let n = dm.rows;
        let mut data = vec![0.0f64; packed_len(m)];
        if m == 0 {
            return Self { m, data };
        }
        let br = block_rows.max(1).min(m);
        let cb = 64usize.min(m.max(1)); // bloc de colonnes (cache-friendly)
        let nblocks = m.div_ceil(br);

        // Decoupage du buffer packed en tranches disjointes, une par bloc de lignes.
        let mut chunks: Vec<(&mut [f64], usize, usize)> = Vec::with_capacity(nblocks);
        let mut rest: &mut [f64] = &mut data;
        for b in 0..nblocks {
            let r0 = b * br;
            let r1 = (r0 + br).min(m);
            let len = row_offset(r1) - row_offset(r0);
            let (head, tail) = rest.split_at_mut(len);
            chunks.push((head, r0, r1));
            rest = tail;
        }

        chunks.into_par_iter().for_each(|(buf, r0, r1)| {
            let base = row_offset(r0);
            // 1) blocs de colonnes entierement sous le bloc de lignes : `j < r0 <= i`.
            let mut j0 = 0usize;
            while j0 < r0 {
                let j1 = (j0 + cb).min(r0);
                for i in r0..r1 {
                    let zi = dm.col(i);
                    let row = &mut buf[row_offset(i) - base..row_offset(i) - base + i + 1];
                    for j in j0..j1 {
                        row[j] = dot(zi, dm.col(j));
                    }
                }
                j0 = j1;
            }
            // 2) region triangulaire du bloc diagonal.
            for i in r0..r1 {
                let zi = dm.col(i);
                let row = &mut buf[row_offset(i) - base..row_offset(i) - base + i + 1];
                for j in r0..=i {
                    row[j] = dot(zi, dm.col(j));
                }
            }
        });

        let _ = n;
        Self { m, data }
    }

    /// Produit matrice-vecteur sur le bloc de tete `k x k` : `y[0..k] = P[0..k,0..k] x[0..k]`.
    ///
    /// Un seul balayage sequentiel du triangle packed (bon pour le prefetcher),
    /// boucle interne vectorisable.
    pub fn matvec_head(&self, k: usize, x: &[f64], y: &mut [f64]) {
        debug_assert!(k <= self.m && x.len() >= k && y.len() >= k);
        y[..k].fill(0.0);
        for i in 0..k {
            let base = row_offset(i);
            let row = &self.data[base..base + i + 1];
            let xi = x[i];
            // diagonale + triangle inferieur
            let mut s = row[i] * xi;
            for (j, &v) in row[..i].iter().enumerate() {
                s += v * x[j];
                y[j] += v * xi;
            }
            y[i] += s;
        }
    }

    /// Permute les indices `i` et `j` (`i < j`) dans le bloc `0..k`, puis "oublie"
    /// la ligne/colonne `j` en decroissant `k` chez l'appelant.
    ///
    /// Aucune donnee n'est deplacee en dehors du bloc de tete : cout `O(k)`.
    pub fn swap_leading(&mut self, i: usize, j: usize) {
        if i == j {
            return;
        }
        // L'echange est symetrique, mais l'algorithme ci-dessous suppose `i < j`.
        if i > j {
            return self.swap_leading(j, i);
        }
        debug_assert!(i < j);
        // diagonale
        let (oi, oj) = (row_offset(i), row_offset(j));
        self.data.swap(oi + i, oj + j);
        // colonnes q < i : (i,q) <-> (j,q)
        for q in 0..i {
            self.data.swap(oi + q, oj + q);
        }
        // lignes p dans (i, j) : la paire (p,i) est stockee en (p,i) et la paire
        // (p,j) en (j,p) [car p < j] : on echange ces deux emplacements.
        for p in (i + 1)..j {
            self.data.swap(row_offset(p) + i, row_offset(j) + p);
        }
        // lignes p > j : les deux paires sont stockees dans la meme ligne.
        for p in (j + 1)..self.m {
            self.data.swap(row_offset(p) + i, row_offset(p) + j);
        }
    }

    /// Ajoute un scalaire a la diagonale sur le bloc de tete (regularisation).
    pub fn add_to_diagonal_head(&mut self, k: usize, delta: f64) {
        for i in 0..k {
            self.data[row_offset(i) + i] += delta;
        }
    }

    /// Produit scalaire d'une ligne (`i < k`) avec un vecteur de longueur `k`.
    pub fn row_dot_head(&self, k: usize, i: usize, x: &[f64]) -> f64 {
        debug_assert!(i < k && k <= self.m);
        let base = row_offset(i);
        let mut s = self.data[base + i] * x[i];
        s += dot(&self.data[base..base + i], &x[..i]);
        for (j, &xj) in x.iter().enumerate().skip(i + 1).take(k - i - 1) {
            s += self.data[row_offset(j) + i] * xj;
        }
        s
    }
}

/// Reconstruit une ligne complete (`k` coefficients) depuis la matrice packed.
pub fn row_into(p: &PackedSym, k: usize, i: usize, out: &mut [f64]) {
    debug_assert!(i < k && k <= p.m && out.len() >= k);
    let base = row_offset(i);
    for j in 0..i {
        out[j] = p.data[base + j];
    }
    out[i] = p.data[base + i];
    for j in (i + 1)..k {
        out[j] = p.data[row_offset(j) + i];
    }
}

/// `y += alpha * row_i(0..k)` (utile pour les mises a jour incrementales).
pub fn row_axpy(p: &PackedSym, k: usize, i: usize, alpha: f64, y: &mut [f64]) {
    debug_assert!(i < k && k <= p.m);
    let base = row_offset(i);
    axpy(alpha, &p.data[base..base + i], &mut y[..i]);
    y[i] += alpha * p.data[base + i];
    for j in (i + 1)..k {
        y[j] += alpha * p.data[row_offset(j) + i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gen::{generate, GenKind};

    fn naive_corr(dm: &DataMatrix) -> Vec<Vec<f64>> {
        let m = dm.cols;
        (0..m)
            .map(|i| (0..m).map(|j| crate::num::dot(dm.col(i), dm.col(j))).collect())
            .collect()
    }

    #[test]
    fn correlation_matches_naive() {
        let mut dm = generate(GenKind::Blocks, 200, 37, 0.6, 0.2, 4, 2, 7);
        dm.standardize(true);
        let p = PackedSym::correlation(&dm, 8);
        let naive = naive_corr(&dm);
        for i in 0..dm.cols {
            for j in 0..dm.cols {
                assert!((p.get(i, j) - naive[i][j]).abs() < 1e-12, "({i},{j})");
            }
        }
    }

    #[test]
    fn matvec_matches_naive() {
        let mut dm = generate(GenKind::Equi, 100, 23, 0.4, 0.0, 1, 1, 3);
        dm.standardize(true);
        let p = PackedSym::correlation(&dm, 16);
        let m = dm.cols;
        let x: Vec<f64> = (0..m).map(|i| ((i * 37 % 11) as f64) - 5.0).collect();
        let mut y = vec![0.0; m];
        p.matvec_head(m, &x, &mut y);
        let naive = naive_corr(&dm);
        for i in 0..m {
            let expect: f64 = (0..m).map(|j| naive[i][j] * x[j]).sum();
            assert!((y[i] - expect).abs() < 1e-11, "i={i} {} vs {}", y[i], expect);
        }
    }

    #[test]
    fn swap_leading_equals_rebuild() {
        let mut dm = generate(GenKind::Blocks, 150, 19, 0.5, 0.1, 3, 1, 11);
        dm.standardize(true);
        let mut p = PackedSym::correlation(&dm, 8);
        let m = dm.cols;
        // permutation appliquee manuellement a la matrice naive
        let mut perm: Vec<usize> = (0..m).collect();
        let mut k = m;
        let mut rng = crate::gen::Rng::new(5);
        while k > 1 {
            let i = (rng.next_u64() as usize) % k;
            let j = k - 1;
            p.swap_leading(i, j);
            perm.swap(i, j);
            k -= 1;
            // verifie tout le bloc de tete
            for a in 0..k {
                for b in 0..k {
                    let expect = crate::num::dot(dm.col(perm[a]), dm.col(perm[b]));
                    assert!((p.get(a, b) - expect).abs() < 1e-12, "k={k} a={a} b={b}");
                }
            }
        }
    }

    #[test]
    fn swap_arbitrary_pair_equals_rebuild() {
        let mut dm = generate(GenKind::Blocks, 150, 15, 0.5, 0.1, 3, 1, 11);
        dm.standardize(true);
        let m = dm.cols;
        let mut p = PackedSym::correlation(&dm, 8);
        let mut perm: Vec<usize> = (0..m).collect();
        let mut rng = crate::gen::Rng::new(99);
        for _ in 0..40 {
            let i = (rng.next_u64() as usize) % m;
            let j = (rng.next_u64() as usize) % m;
            p.swap_leading(i.min(j), i.max(j));
            perm.swap(i, j);
            for a in 0..m {
                for b in 0..m {
                    let expect = crate::num::dot(dm.col(perm[a]), dm.col(perm[b]));
                    assert!((p.get(a, b) - expect).abs() < 1e-12, "a={a} b={b}");
                }
            }
        }
    }

    /// `swap_leading` est symetrique du point de vue de l'appelant : `(i, j)` et
    /// `(j, i)` doivent donner le meme resultat (le stockage packed, lui, impose
    /// un ordre interne).
    #[test]
    fn swap_leading_accepts_reversed_arguments() {
        let mut dm = generate(GenKind::Blocks, 150, 15, 0.5, 0.1, 3, 1, 11);
        dm.standardize(true);
        let m = dm.cols;
        let mut p = PackedSym::correlation(&dm, 8);
        let mut perm: Vec<usize> = (0..m).collect();
        let mut rng = crate::gen::Rng::new(99);
        for t in 0..40 {
            let i = (rng.next_u64() as usize) % m;
            let j = (rng.next_u64() as usize) % m;
            if t % 2 == 0 {
                p.swap_leading(i, j);
            } else {
                p.swap_leading(j, i);
            }
            perm.swap(i, j);
            for a in 0..m {
                for b in 0..m {
                    let expect = crate::num::dot(dm.col(perm[a]), dm.col(perm[b]));
                    assert!(
                        (p.get(a, b) - expect).abs() < 1e-12,
                        "t={t} i={i} j={j} a={a} b={b}"
                    );
                }
            }
        }
    }

    #[test]
    fn row_into_and_row_dot_agree() {
        let mut dm = generate(GenKind::Ar, 80, 13, 0.6, 0.0, 1, 1, 2);
        dm.standardize(true);
        let p = PackedSym::correlation(&dm, 8);
        let k = 13;
        let x: Vec<f64> = (0..k).map(|i| (i as f64).sin()).collect();
        for i in 0..k {
            let mut row = vec![0.0; k];
            row_into(&p, k, i, &mut row);
            let direct: f64 = (0..k).map(|j| p.get(i, j) * x[j]).sum();
            assert!((p.row_dot_head(k, i, &x) - direct).abs() < 1e-12);
        }
    }
}

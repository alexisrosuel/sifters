//! Matrice de donnees dense `N x M` (N observations, M variables), stockee en
//! colonnes-major : chaque colonne (une variable) est contigue en memoire, ce qui
//! est le bon layout pour tous les produits scalaires du solveur.

use rayon::prelude::*;

/// Matrice de donnees `rows x cols` en colonnes-major.
#[derive(Clone, Debug)]
pub struct DataMatrix {
    /// Nombre d'observations (N).
    pub rows: usize,
    /// Nombre de variables (M).
    pub cols: usize,
    /// `data[col * rows + row]`.
    pub data: Vec<f64>,
}

impl DataMatrix {
    /// Construit une matrice depuis un buffer colonnes-major (aucune copie).
    pub fn from_col_major(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols, "taille de buffer incoherente");
        Self { rows, cols, data }
    }

    /// Construit une matrice depuis un buffer lignes-major (transposition).
    pub fn from_row_major(rows: usize, cols: usize, data: &[f64]) -> Self {
        assert_eq!(data.len(), rows * cols);
        let mut out = vec![0.0; rows * cols];
        for r in 0..rows {
            let src = &data[r * cols..(r + 1) * cols];
            for (c, &v) in src.iter().enumerate() {
                out[c * rows + r] = v;
            }
        }
        Self { rows, cols, data: out }
    }

    /// Vue sur la colonne `j`.
    #[inline]
    pub fn col(&self, j: usize) -> &[f64] {
        let o = j * self.rows;
        &self.data[o..o + self.rows]
    }

    /// Vue mutable sur la colonne `j`.
    #[inline]
    pub fn col_mut(&mut self, j: usize) -> &mut [f64] {
        let o = j * self.rows;
        &mut self.data[o..o + self.rows]
    }

    /// Toutes les colonnes, decoupees en tranches contigues.
    pub fn par_cols_mut(&mut self) -> impl IndexedParallelIterator<Item = &mut [f64]> {
        self.data.par_chunks_mut(self.rows)
    }

    /// Sommes et normes des colonnes (parallele).
    pub fn column_stats(&self) -> (Vec<f64>, Vec<f64>) {
        let rows = self.rows;
        let mut means = vec![0.0; self.cols];
        let mut sumsq = vec![0.0; self.cols];
        self.data
            .par_chunks(rows)
            .zip(means.par_iter_mut())
            .zip(sumsq.par_iter_mut())
            .for_each(|((c, m), q)| {
                let s: f64 = c.iter().sum();
                *m = s / rows as f64;
                let ss: f64 = c.iter().map(|v| (v - *m) * (v - *m)).sum();
                *q = ss;
            });
        (means, sumsq)
    }

    /// Centre (si `center`) puis normalise chaque colonne a la norme 1.
    ///
    /// Apres cet appel la matrice est `Z` telle que `R = Z^T Z` soit la matrice de
    /// correlation (si `center = true`) ou la matrice des cosinus (sinon).
    ///
    /// Retourne le masque des colonnes conservees (les colonnes de variance nulle
    /// sont eliminees, `false`), ainsi que la liste des indices d'origine elimines.
    pub fn standardize(&mut self, center: bool) -> StandardizeReport {
        let rows_f = self.rows as f64;
        let cols = self.cols;
        // 1) moyennes puis normes centrees, en parallele.
        let mut norms = vec![0.0f64; cols];
        self.data
            .par_chunks_mut(self.rows)
            .zip(norms.par_iter_mut())
            .for_each(|(c, nrm)| {
                if center {
                    let s: f64 = c.iter().sum();
                    let mean = s / rows_f;
                    for v in c.iter_mut() {
                        *v -= mean;
                    }
                }
                let ss: f64 = c.iter().map(|v| v * v).sum();
                let n = ss.sqrt();
                if n > 0.0 {
                    let inv = 1.0 / n;
                    for v in c.iter_mut() {
                        *v *= inv;
                    }
                }
                *nrm = n;
            });
        // 2) elimination des colonnes degenerees (recompaction des colonnes gardees).
        let threshold = (self.rows as f64).sqrt() * f64::EPSILON * 100.0;
        let mut dropped = Vec::new();
        for (j, &n) in norms.iter().enumerate() {
            if !(n > threshold) {
                dropped.push(j);
            }
        }
        if !dropped.is_empty() {
            let keep: Vec<usize> = (0..cols).filter(|j| norms[*j] > threshold).collect();
            let new_cols = keep.len();
            let mut new_data = vec![0.0; new_cols * self.rows];
            new_data
                .par_chunks_mut(self.rows)
                .zip(keep.par_iter())
                .for_each(|(dst, &j)| {
                    let o = j * self.rows;
                    dst.copy_from_slice(&self.data[o..o + self.rows]);
                });
            self.data = new_data;
            self.cols = new_cols;
        }
        StandardizeReport { dropped }
    }
}

/// Resultat de la standardisation.
#[derive(Clone, Debug)]
pub struct StandardizeReport {
    /// Indices (dans la numerotation d'origine) des colonnes supprimees.
    pub dropped: Vec<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standardize_gives_unit_norm() {
        let mut m = DataMatrix::from_row_major(3, 2, &[1.0, 2.0, 3.0, 10.0, 20.0, 30.0]);
        let rep = m.standardize(true);
        assert!(rep.dropped.is_empty());
        for j in 0..m.cols {
            let c = m.col(j);
            let n: f64 = c.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!((n - 1.0).abs() < 1e-14, "col {j} norm {n}");
            let s: f64 = c.iter().sum();
            assert!(s.abs() < 1e-14);
        }
    }

    #[test]
    fn drops_constant_column() {
        let mut m = DataMatrix::from_row_major(3, 2, &[1.0, 5.0, 1.0, 6.0, 1.0, 7.0]);
        let rep = m.standardize(true);
        assert_eq!(rep.dropped, vec![0]);
        assert_eq!(m.cols, 1);
    }
}

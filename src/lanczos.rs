//! Lanczos (reorthogonalisation complete) pour la **plus petite** valeur propre
//! d'un operateur symetrique, avec demarrage a chaud.
//!
//! C'est la brique qui rend la selection gloutonne rapide :
//!  * entre deux etapes, le vecteur propre cherche varie tres peu, donc un
//!    demarrage a chaud converge en quelques iterations ;
//!  * a chaque iteration, la plus petite valeur propre de la tridiagonale `T_j`
//!    est obtenue par bissection/Sturm en `O(50 j)`, ce qui donne un test de
//!    convergence quasi gratuit (sans re-diagonalisation `O(j^3)`).

use crate::jacobi::tridiag_smallest;
pub use crate::op::SymOp;
use crate::num::{axpy, dot, norm2};

/// Resultat d'un appel Lanczos.
#[derive(Clone, Debug)]
pub struct LanczosOutcome {
    /// Valeur propre approchee (valeur de Ritz).
    pub value: f64,
    /// Vecteur propre associe (norme 1).
    pub vector: Vec<f64>,
    /// `||A v - value * v||` evalue exactement sur le vecteur final.
    pub residual: f64,
    /// Nombre d'iterations effectuees.
    pub iters: usize,
    /// Vrai si le critere de convergence est atteint.
    pub converged: bool,
}

/// Graine deterministe si l'appelant n'en fournit pas.
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

/// Plus petite valeur propre (et vecteur propre) de `op`.
///
/// * `seed` : vecteur de depart (typiquement le vecteur propre precedent) ;
/// * `tol`  : tolerance relative sur la stabilisation de la valeur de Ritz ;
/// * `max_iters` : nombre maximal d'iterations de Lanczos.
pub fn smallest_eigenpair(
    op: &mut dyn SymOp,
    seed: Option<&[f64]>,
    tol: f64,
    max_iters: usize,
) -> LanczosOutcome {
    smallest_eigenpair_constrained(op, seed, tol, max_iters, &[])
}

/// Idem, mais en restreignant l'espace de recherche au supplementaire orthogonal de
/// `constraints` (deflation par orthogonalisation explicite).
///
/// Chaque nouveau vecteur de base est reorthogonalise contre `constraints`, ce qui
/// evite l'amplification catastrophique de la composante parasite lorsque le
/// coefficient de Lanczos `beta` devient petit (contrairement a l'eclatement par
/// operateur projete `(I-P)A`, dont le vecteur d'essai conserve la composante et se
/// fait amplifier par `alpha/beta`).
pub fn smallest_eigenpair_constrained(
    op: &mut dyn SymOp,
    seed: Option<&[f64]>,
    tol: f64,
    max_iters: usize,
    constraints: &[Vec<f64>],
) -> LanczosOutcome {
    let n = op.n();
    assert!(n > 0, "operateur de dimension nulle");
    if n == 1 {
        let mut y = [0.0];
        op.mul(&[1.0], &mut y);
        return LanczosOutcome { value: y[0], vector: vec![1.0], residual: 0.0, iters: 1, converged: true };
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
        // Reorthogonalisation complete (deux passes : stabilite numerique) puis
        // projection sur le supplementaire des contraintes de deflation.
        //
        // La seconde passe n'est exécutée que si la première a fait chuter la norme
        // de `w` (signe d'une annulation catastrophique, donc d'une perte
        // d'orthogonalité) : critère classique, seuil `1/sqrt(2)`. Les itérations
        // chaudes, où `w` perd peu, économisent ainsi une passe complète sur la base.
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
            if pass == 0
                && (j == 0 || norm2(&w) > std::f64::consts::FRAC_1_SQRT_2 * nrm_before)
            {
                break;
            }
        }
        alphas.push(alpha);
        if j > 0 {
            betas.push(beta_prev);
        }
        iters = j + 1;
        beta_prev = norm2(&w);
        // Detection de sous-espace invariant : en dessous de ce seuil relatif,
        // `w` n'est plus que du bruit d'arrondi et l'ajouter comme vecteur de base
        // ferait apparaitre des valeurs propres parasites.
        if beta_prev <= 1e-9 * (1.0 + alpha.abs()) {
            converged = true;
            break;
        }
        // `theta` sert uniquement au test de convergence (la valeur finale est
        // recalculee plus bas avec 60 pas de bissection) : 40 pas suffisent a le
        // rendre plus precis que la tolerance. La bissection n'a besoin que du
        // predicat « au moins une valeur propre sous `mu` », d'ou la sortie
        // anticipee de la suite de Sturm.
        let m = alphas.len();
        let tb = &betas[..m.saturating_sub(1)];
        let theta = tridiag_smallest(&alphas, tb, 40);
        // Critere d'arret **sur le residu de Ritz** `||A u - theta u|| = beta_j |y_j|`
        // (`y_j` = derniere composante du vecteur propre de la tridiagonale). C'est la
        // mesure qui garantit la qualite de la *valeur* : un simple test de
        // stabilisation de `theta` peut se declencher sur un palier alors que le
        // residu est encore grand, ce qui surestime `lambda_min` et fausse la
        // certification de l'etape gloutonne.
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

    // Vecteur de Ritz : plus petite valeur propre de la tridiagonale par bissection
    // (Sturm), puis vecteur propre par iteration inverse — `O(m)` au lieu de la
    // diagonalisation de Jacobi `O(m^3)` faite a chaque appel.
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

    // residu exact
    let mut au = vec![0.0; n];
    op.mul(&u, &mut au);
    axpy(-value, &u, &mut au);
    let residual = norm2(&au);

    LanczosOutcome { value, vector: u, residual, iters, converged }
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
        // reference Jacobi
        let mut dense = vec![0.0; k * k];
        for i in 0..k {
            for j in 0..k {
                dense[i * k + j] = ds.packed.as_ref().unwrap().get(i, j);
            }
        }
        let (vals, _) = eigen_sym_sorted(&mut dense, k);
        assert!((out.value - vals[0]).abs() < 1e-9, "lanczos {} jacobi {}", out.value, vals[0]);
        // Le residu du vecteur de Ritz est typiquement bien plus grand que l'erreur
        // sur la valeur propre (qui converge quadratiquement) : c'est un indicateur de
        // qualite du vecteur, pas de la valeur. Les bornes de la cascade restent
        // rigoureuses quel que soit ce residu.
        assert!(out.residual < 1e-6, "residu {}", out.residual);
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

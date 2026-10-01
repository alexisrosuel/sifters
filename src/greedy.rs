//! Algorithmes de selection gloutonne.
//!
//! # Probleme
//!
//! Maximiser `lambda_min(R_S)`, la plus petite valeur propre de la matrice de
//! correlation d'un sous-ensemble `S` de `K` variables. NP-difficile ; on construit
//! une famille **imbriquee** de sous-ensembles par elimination arriere certifiee.
//!
//! # Cascade certifiee
//!
//! Soit `R = R_S` de taille `k` et `(lambda_l, u_l)_{l<=p}` ses `p` plus petits
//! couples propres. Retirer la variable `i` donne `R_{-i}`, dont les valeurs propres
//! sont (Cauchy) entrelacees : `lambda_l(R) <= lambda_l(R_{-i}) <= lambda_{l+1}(R)`.
//!
//! ## Borne 1 : Rayleigh (toujours valide, O(1))
//!
//! Le vecteur `u_1` prive de sa composante `i`, renormalise, est un vecteur d'essai
//! pour `R_{-i}` ; son quotient de Rayleigh vaut, avec `y = R u_1` et
//! `rq = u_1^T R u_1` :
//!
//! ```text
//! rho_i = ( rq - 2 u_1(i) y(i) + u_1(i)^2 ) / ( 1 - u_1(i)^2 )  >=  lambda_min(R_{-i})
//! ```
//!
//! Cette borne est `O(1)` par candidat mais **lache** (elle atteint `lambda_1 + m_i`,
//! ou `m_i = u_1(i)^2` est la masse du vecteur propre) : elle ne permet pas d'elaguer.
//!
//! ## Borne 2 : Temple spectrale `p`-dimensionnelle (serree)
//!
//! Les valeurs propres de `R_{-i}` sont exactement les racines de l'equation
//! seculaire (conditions KKT de `min x^T R x` sous `x_i = 0`) :
//!
//! ```text
//! S_i(mu) = sum_l (u_l(i))^2 / (lambda_l - mu) = 0,    racine dans (lambda_1, lambda_2)
//! ```
//!
//! En separant les `p` premiers termes et en minorant la queue par Cauchy-Schwarz,
//! avec `m_i = u_1(i)^2`, `B_i = sum_{l=2..p} u_l(i)^2/(lambda_l - lambda_1)`,
//! `T_i = 1 - sum_{l<=p} u_l(i)^2` et
//! `Sig_i = (1 - lambda_1) - sum_{l=2..p} u_l(i)^2 (lambda_l - lambda_1)`
//! (identites exactes `sum_l u_l(i)^2 = 1` et `sum_l u_l(i)^2 lambda_l = R_ii = 1`),
//! on obtient
//!
//! ```text
//! G_i = B_i + T_i^2 / Sig_i        (minoration de la queue)
//! ub_i = lambda_1 + m_i / G_i      >= lambda_min(R_{-i})
//! ```
//!
//! bien plus serree que `rho_i` (elle se reduit a Rayleigh pour `p = 1`). On prend
//! `min(rho_i, ub_i)`. Les residus de Lanczos sont retranches de `T_i` et ajoutes a
//! `Sig_i` (marge conservative).
//!
//! ## Selection
//!
//! On trie les candidats par borne croissante et on les evalue exactement (Lanczos a
//! chaud) jusqu'a ce que la meilleure valeur atteinte depasse la borne du candidat
//! suivant : **aucun candidat non evalue ne peut alors faire mieux**, l'etape est
//! certifiee optimale (a la tolerance et a la marge pres).

use crate::inverse::{InverseSym, NegInvSubOp};
use crate::lanczos::{smallest_eigenpair, smallest_eigenpair_constrained, LanczosOutcome};
use crate::op::SymOp;
use crate::num::dot;
use crate::op::{z_dot_all, z_loading, BorderedHeadOp, Dataset, GatheredZOp, SubOp};
use rayon::prelude::*;
use std::time::Instant;

/// Sens de parcours de la famille imbriquee.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    /// Elimination arriere : `M -> k_min`.
    Backward,
    /// Selection avant : `1 -> k_max`.
    Forward,
}

/// Methode d'evaluation de `lambda_min` d'un candidat.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvalMode {
    /// Detection automatique (inverse si la correlation est definie positive et
    /// `M` raisonnable, sinon matvec direct).
    Auto,
    /// Lanczos sur `R` directement.
    Direct,
    /// Lanczos sur l'inverse maintenu : `lambda_min(R_{-i}) = -1 / theta_min(-R_{-i}^{-1})`.
    /// Les ecarts relatifs du bas du spectre y sont fortement amplifies, d'ou une
    /// convergence en quelques iterations au lieu de plusieurs dizaines.
    Inverse,
}

/// Parametres numeriques.
#[derive(Clone, Copy, Debug)]
pub struct AlgoConfig {
    /// Tolerance relative de Lanczos.
    pub tol: f64,
    /// Iterations max pour un demarrage a froid.
    pub max_iters_cold: usize,
    /// Iterations max pour un demarrage a chaud.
    pub max_iters_warm: usize,
    /// Nombre max de candidats evalues exactement par etape (0 = illimite).
    pub max_exact: usize,
    /// Taille des lots paralleles.
    pub batch: usize,
    /// Nombre `p` de couples propres utilises pour la borne de Temple.
    pub num_low: usize,
    /// Methode d'evaluation des candidats.
    pub eval: EvalMode,
    /// Seuil sur `M` pour l'evaluation par l'inverse en mode `Auto`.
    pub inverse_max_m: usize,
}

impl Default for AlgoConfig {
    fn default() -> Self {
        Self {
            tol: 1e-10,
            max_iters_cold: 400,
            max_iters_warm: 60,
            max_exact: 0,
            batch: 8,
            num_low: 4,
            eval: EvalMode::Auto,
            inverse_max_m: 3000,
        }
    }
}

/// Trace d'une etape.
#[derive(Clone, Debug)]
pub struct StepRecord {
    /// Taille avant l'etape.
    pub k_before: usize,
    /// Taille apres l'etape.
    pub k_after: usize,
    /// Indice d'origine de la variable entree/sortie.
    pub changed_orig: usize,
    /// `lambda_min` du nouvel ensemble (valeur de Ritz a chaud).
    pub lambda: f64,
    /// `lambda_min` recalcule a froid de facon stricte (`NaN` si non demande).
    pub lambda_verified: f64,
    /// Borne (Temple/Rayleigh) du candidat retenu.
    pub upper: f64,
    /// Borne de Rayleigh seule du candidat retenu.
    pub rayleigh: f64,
    /// Candidats examines exactement.
    pub exact_evals: usize,
    /// Candidats au total.
    pub candidates: usize,
    /// Vrai si l'optimalite gloutonne est certifiee.
    pub certified: bool,
    /// Residu de Lanczos du candidat retenu.
    pub residual: f64,
    /// Iterations Lanczos cumulees sur l'etape.
    pub iters: usize,
}

/// Resultat complet d'un parcours glouton.
#[derive(Clone, Debug)]
pub struct PathResult {
    /// Sens du parcours.
    pub direction: Direction,
    /// Taille initiale.
    pub initial_k: usize,
    /// `lambda_min` initial (`1` pour un singleton, `NaN` si vide).
    pub initial_lambda: f64,
    /// Residu du calcul initial.
    pub initial_residual: f64,
    /// Etapes successives.
    pub steps: Vec<StepRecord>,
    /// Duree du parcours (s).
    pub seconds: f64,
    /// Sous-ensemble de depart (`1` indice en selection avant, `initial_k` indices
    /// en elimination arriere).
    pub initial_subset: Vec<usize>,
}

impl PathResult {
    /// Nombre total de candidats evalues exactement.
    pub fn total_exact(&self) -> usize {
        self.steps.iter().map(|s| s.exact_evals).sum()
    }

    /// Nombre d'etapes certifiees.
    pub fn certified_steps(&self) -> usize {
        self.steps.iter().filter(|s| s.certified).count()
    }

    /// Iterations Lanczos cumulees.
    pub fn total_iters(&self) -> usize {
        self.steps.iter().map(|s| s.iters).sum()
    }

    /// Suite des `(K, lambda_min)`.
    pub fn curve(&self) -> Vec<(usize, f64)> {
        let mut out = Vec::with_capacity(self.steps.len() + 1);
        out.push((self.initial_k, self.initial_lambda));
        for s in &self.steps {
            out.push((
                s.k_after,
                if s.lambda_verified.is_nan() { s.lambda } else { s.lambda_verified },
            ));
        }
        out
    }

    /// Ordre des variables retirees (backward) ou ajoutees (forward).
    ///
    /// En selection avant, le sous-ensemble de taille `K` est
    /// `initial_subset ++ order[..K-initial_k]` ; en elimination arriere c'est
    /// `initial_subset` prive de `order[..initial_k-K]`.
    pub fn order(&self) -> Vec<usize> {
        self.steps.iter().map(|s| s.changed_orig).collect()
    }

    /// Valeur de `lambda_min` pour une taille `K` visitee.
    pub fn lambda_at(&self, k: usize) -> Option<f64> {
        if k == self.initial_k {
            return Some(self.initial_lambda);
        }
        self.steps
            .iter()
            .find(|s| s.k_after == k)
            .map(|s| if s.lambda_verified.is_nan() { s.lambda } else { s.lambda_verified })
    }

    /// Valeur brute (Ritz a chaud) du parcours, sans revalidation.
    pub fn lambda_raw_at(&self, k: usize) -> Option<f64> {
        if k == self.initial_k {
            return Some(self.initial_lambda);
        }
        self.steps.iter().find(|s| s.k_after == k).map(|s| s.lambda)
    }

    /// Sous-ensemble de variables (indices d'origine, tries) de taille `K`.
    pub fn subset_at(&self, k: usize) -> Option<Vec<usize>> {
        match self.direction {
            Direction::Backward => {
                let k_end = self.steps.last().map(|s| s.k_after).unwrap_or(self.initial_k);
                if k > self.initial_k || k < k_end {
                    return None;
                }
                let mut set: Vec<usize> = (0..self.initial_k).collect();
                for s in self.steps.iter().take(self.initial_k - k) {
                    if let Some(p) = set.iter().position(|&x| x == s.changed_orig) {
                        set.swap_remove(p);
                    }
                }
                set.sort_unstable();
                Some(set)
            }
            Direction::Forward => {
                if k < self.initial_k || k > self.initial_k + self.steps.len() {
                    return None;
                }
                let mut set: Vec<usize> = self.initial_subset.clone();
                set.extend(
                    self.steps.iter().take(k - self.initial_k).map(|s| s.changed_orig),
                );
                set.sort_unstable();
                set.dedup();
                if set.len() == k {
                    Some(set)
                } else {
                    None
                }
            }
        }
    }
}

/// Les `p` plus petits couples propres d'un ensemble de taille `k`.
#[derive(Clone, Debug)]
pub struct LowSpectrum {
    /// Taille de l'ensemble.
    pub dim: usize,
    /// Valeurs propres croissantes.
    pub values: Vec<f64>,
    /// Vecteurs propres associes (norme 1, longueur `dim`).
    pub vectors: Vec<Vec<f64>>,
    /// Residus `||R u - lambda u||`.
    pub residuals: Vec<f64>,
}

impl LowSpectrum {
    /// Singleton.
    pub fn singleton() -> Self {
        Self { dim: 1, values: vec![1.0], vectors: vec![vec![1.0]], residuals: vec![0.0] }
    }

    /// Nombre de couples disponibles.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Vrai si aucun couple n'est disponible.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Plus petite valeur propre.
    pub fn lambda_min(&self) -> f64 {
        self.values.first().copied().unwrap_or(f64::NAN)
    }

    /// Residu maximal.
    pub fn max_residual(&self) -> f64 {
        self.residuals.iter().cloned().fold(0.0f64, f64::max)
    }

    /// Borne superieure de `lambda_min(R_{-i})` (voir la doc du module).
    ///
    /// `ray` est la borne de Rayleigh, toujours valide ; on retourne le minimum des
    /// deux bornes.
    pub fn upper_bound(&self, i: usize, ray: f64) -> f64 {
        let p = self.values.len();
        if p == 0 {
            return f64::INFINITY;
        }
        let lam1 = self.values[0];
        let u1i2 = self.vectors[0][i] * self.vectors[0][i];
        let mut b = 0.0f64;
        let mut mass = u1i2;
        let mut smass = 0.0f64;
        for j in 1..p {
            let u2 = self.vectors[j][i] * self.vectors[j][i];
            let d = (self.values[j] - lam1).max(1e-300);
            b += u2 / d;
            smass += u2 * d;
            mass += u2;
        }
        // Marginal conservative : on sous-estime la masse de queue et on surestime
        // la norme de queue, ce qui ne peut que diminuer `G_i` et donc augmenter la
        // borne (donc rester valide).
        let slack = 2.0 * p as f64 * self.max_residual();
        let t = (1.0 - mass - slack).max(0.0);
        let gap_p = if p > 1 { (self.values[p - 1] - lam1).max(0.0) } else { 0.0 };
        let sig = (1.0 - lam1) - smass - slack * (1.0 + gap_p);
        // Le terme de queue `t^2/sig` est un 0/0 numerique lorsque la masse de queue
        // est nulle (cas `p = k`) : on le supprime alors, et on se rabat sur Rayleigh
        // si le rapport n'est pas fiable.
        let refined = if t <= 1e-12 {
            lam1 + u1i2 / b.max(1e-300)
        } else if sig > 1e-12 {
            lam1 + u1i2 / (b + t * t / sig)
        } else {
            f64::INFINITY
        };
        refined.min(ray).max(lam1)
    }
}

/// Couple propre de plus petite valeur propre d'un ensemble de `k` variables
/// (bloc de tete), avec demarrage a froid.
pub fn initial_eigenpair_public(ds: &Dataset, k: usize, cfg: &AlgoConfig) -> LanczosOutcome {
    if k == 0 {
        return LanczosOutcome {
            value: f64::NAN,
            vector: Vec::new(),
            residual: 0.0,
            iters: 0,
            converged: true,
        };
    }
    if k == 1 {
        return LanczosOutcome {
            value: 1.0,
            vector: vec![1.0],
            residual: 0.0,
            iters: 1,
            converged: true,
        };
    }
    let mut op = SubOp::head(ds, k);
    smallest_eigenpair(&mut op, None, cfg.tol, cfg.max_iters_cold)
}

/// Couple propre de plus petite valeur propre d'un ensemble de `k` variables,
/// en exploitant l'inverse s'il est disponible.
pub fn initial_eigenpair_inv(
    ds: &Dataset,
    k: usize,
    cfg: &AlgoConfig,
    inv: Option<&InverseSym>,
) -> LanczosOutcome {
    if k <= 1 {
        return initial_eigenpair_public(ds, k, cfg);
    }
    match inv {
        Some(iv) => {
            let mut op = NegInvSubOp::new(iv, k, None);
            let o = smallest_eigenpair(&mut op, None, cfg.tol, cfg.max_iters_cold);
            LanczosOutcome { value: to_lambda(o.value, true), ..o }
        }
        None => initial_eigenpair_public(ds, k, cfg),
    }
}

/// Calcule les `p` plus petits couples propres de l'ensemble courant.
///
/// `known` reutilise le couple propre obtenu par l'evaluation exacte du gagnant de
/// l'etape precedente ; `seeds[j]` sert de demarrage a chaud au couple `j`.
pub fn low_spectrum(
    ds: &Dataset,
    k: usize,
    p: usize,
    seeds: &[Vec<f64>],
    known: Option<&LanczosOutcome>,
    cfg: &AlgoConfig,
    cold: bool,
) -> LowSpectrum {
    low_spectrum_inv(ds, k, p, seeds, known, cfg, cold, None)
}

/// Idem, en exploitant eventuellement l'inverse `W = R^{-1}` (voir [`EvalMode`]).
#[allow(clippy::too_many_arguments)]
pub fn low_spectrum_inv(
    ds: &Dataset,
    k: usize,
    p: usize,
    seeds: &[Vec<f64>],
    known: Option<&LanczosOutcome>,
    cfg: &AlgoConfig,
    cold: bool,
    inv: Option<&InverseSym>,
) -> LowSpectrum {
    if k == 0 {
        return LowSpectrum { dim: 0, values: Vec::new(), vectors: Vec::new(), residuals: Vec::new() };
    }
    if k == 1 {
        return LowSpectrum::singleton();
    }
    let p = p.clamp(1, k);
    let iters = if cold { cfg.max_iters_cold } else { cfg.max_iters_warm };
    let mut spec =
        LowSpectrum { dim: k, values: Vec::new(), vectors: Vec::new(), residuals: Vec::new() };
    if let Some(kn) = known {
        if kn.vector.len() == k {
            spec.values.push(kn.value);
            spec.vectors.push(kn.vector.clone());
            spec.residuals.push(kn.residual);
        }
    }
    while spec.values.len() < p {
        let j = spec.values.len();
        let seed = if j < seeds.len() && seeds[j].len() == k {
            seeds[j].clone()
        } else {
            crate::lanczos::default_seed(k, (k as u64) << 8 | j as u64)
        };
        // Recherche restreinte au supplementaire des vecteurs deja trouves : la plus
        // petite valeur de Ritz converge alors vers la (j+1)-ieme valeur propre.
        let mut out = match inv {
            Some(inv) => {
                let mut op = NegInvSubOp::new(inv, k, None);
                let o = smallest_eigenpair_constrained(
                    &mut op,
                    Some(seed.as_slice()),
                    cfg.tol,
                    iters,
                    &spec.vectors,
                );
                LanczosOutcome { value: to_lambda(o.value, true), ..o }
            }
            None => {
                let mut op = SubOp::head(ds, k);
                smallest_eigenpair_constrained(
                    &mut op,
                    Some(seed.as_slice()),
                    cfg.tol,
                    iters,
                    &spec.vectors,
                )
            }
        };
        let mut u = out.vector;
        let nu = crate::num::norm2(&u);
        let value;
        if nu > 1e-12 {
            for x in u.iter_mut() {
                *x /= nu;
            }
            let mut raw = SubOp::head(ds, k);
            let mut au = vec![0.0; k];
            raw.mul(&u, &mut au);
            let rq = dot(&u, &au);
            value = if nu > 0.999 { out.value } else { rq };
            out.value = value;
            crate::num::axpy(-value, &u, &mut au);
            let residual = crate::num::norm2(&au);
            spec.values.push(value);
            spec.vectors.push(u);
            spec.residuals.push(residual);
        } else {
            break;
        }
    }
    spec
}

/// Racine de l'equation seculaire tronquee aux `p` premiers couples propres,
/// dans l'intervalle `(lambda_1, lambda_2)` :
///
/// ```text
/// S_i(mu) = sum_{l<=p} u_l(i)^2 / (lambda_l - mu) = 0
/// ```
///
/// C'est un majorant de `lambda_min(R_{-i})` (les termes de queue sont positifs sur
/// l'intervalle, donc la racine tronquee est au-dessus de la racine exacte) et
/// surtout le vecteur `x = (R - mu I)^{-1} e_i` est le vecteur propre exact de
/// `R_{-i}` : c'est un demarrage a chaud quasi parfait pour Lanczos.
///
/// Renvoie `None` lorsque la composante `u_1(i)` est numeriquement nulle : dans ce
/// cas `lambda_min(R_{-i}) = lambda_1` et le vecteur propre est `u_1` restreint.
pub fn secular_root(spec: &LowSpectrum, i: usize) -> Option<f64> {
    let p = spec.values.len();
    if p == 0 {
        return None;
    }
    let lam1 = spec.values[0];
    let u1i2 = spec.vectors[0][i] * spec.vectors[0][i];
    if u1i2 <= 1e-14 {
        return None;
    }
    let hi = if p > 1 { spec.values[1] } else { lam1 + 1.0 };
    let f = |mu: f64| -> f64 {
        spec.values
            .iter()
            .zip(spec.vectors.iter())
            .map(|(l, u)| {
                let t = u[i];
                t * t / (l - mu)
            })
            .sum()
    };
    let mut a = lam1 + 1e-13 * (1.0 + lam1.abs());
    let mut b = hi - 1e-13 * (1.0 + hi.abs());
    if !(a < b) {
        return None;
    }
    if f(a) > 0.0 {
        return None;
    }
    for _ in 0..80 {
        let m = 0.5 * (a + b);
        if !m.is_finite() {
            return None;
        }
        if f(m) < 0.0 {
            a = m;
        } else {
            b = m;
        }
    }
    Some(0.5 * (a + b))
}

/// Vecteur d'essai `x = (R - mu I)^{-1} e_i` tronque aux `p` premiers couples propres.
pub fn trial_vector(spec: &LowSpectrum, i: usize, mu: f64) -> Vec<f64> {
    let mut x = vec![0.0; spec.dim];
    for (l, u) in spec.values.iter().zip(spec.vectors.iter()) {
        let d = l - mu;
        if d.abs() > 1e-300 {
            let w = u[i] / d;
            if w != 0.0 {
                crate::num::axpy(w, u, &mut x);
            }
        }
    }
    x
}

/// Borne de Rayleigh pour le retrait du `i`-eme element, a partir de `y = R u`.
///
/// Valide pour **tout** vecteur unitaire `u`.
#[inline]
pub fn rayleigh_upper(rq: f64, ui: f64, yi: f64) -> f64 {
    let v2 = 1.0 - ui * ui;
    if v2 <= 1e-14 {
        return f64::INFINITY;
    }
    (rq - 2.0 * ui * yi + ui * ui) / v2
}

fn head_rayleigh(ds: &Dataset, k: usize, u: &[f64]) -> (f64, Vec<f64>) {
    let mut op = SubOp::head(ds, k);
    let mut y = vec![0.0; k];
    op.mul(u, &mut y);
    let rq = dot(u, &y);
    (rq, y)
}

/// Restriction a `{0..k}\{i}` puis permutation consequence du `swap(i, k-1)`.
fn restrict_and_permute(v: &[f64], i: usize, k: usize) -> Vec<f64> {
    let mut w = Vec::with_capacity(k.saturating_sub(1));
    w.extend_from_slice(&v[..i]);
    w.extend_from_slice(&v[i + 1..k]);
    if i + 1 < k {
        let last = w.pop().expect("vecteur non vide");
        w.insert(i, last);
    }
    w
}

fn permute_after_swap(v: &mut Vec<f64>, i: usize, k: usize) {
    if i + 1 < k {
        let last = v.pop().expect("vecteur non vide");
        v.insert(i, last);
    }
}

/// Operateur de tete : direct (`R_S`) ou inverse (`-R_S^{-1}`).
enum HeadOp<'a> {
    Direct(SubOp<'a>),
    Inverse(NegInvSubOp<'a>),
}

impl SymOp for HeadOp<'_> {
    fn n(&self) -> usize {
        match self {
            HeadOp::Direct(o) => o.n(),
            HeadOp::Inverse(o) => o.n(),
        }
    }
    fn mul(&mut self, x: &[f64], y: &mut [f64]) {
        match self {
            HeadOp::Direct(o) => o.mul(x, y),
            HeadOp::Inverse(o) => o.mul(x, y),
        }
    }
}

/// Operateur d'un candidat : direct (`R_{-i}`) ou inverse (`-R_{-i}^{-1}`).
enum DelOp<'a> {
    Direct(SubOp<'a>),
    Inverse(NegInvSubOp<'a>),
}

impl SymOp for DelOp<'_> {
    fn n(&self) -> usize {
        match self {
            DelOp::Direct(o) => o.n(),
            DelOp::Inverse(o) => o.n(),
        }
    }
    fn mul(&mut self, x: &[f64], y: &mut [f64]) {
        match self {
            DelOp::Direct(o) => o.mul(x, y),
            DelOp::Inverse(o) => o.mul(x, y),
        }
    }
}

/// Convertit une valeur propre de l'operateur utilise en `lambda_min` de `R`.
#[inline]
fn to_lambda(theta: f64, inv: bool) -> f64 {
    if inv {
        if theta.abs() < 1e-300 {
            f64::INFINITY
        } else {
            -1.0 / theta
        }
    } else {
        theta
    }
}

#[derive(Debug)]
struct Eval {
    pos: usize,
    upper: f64,
    out: LanczosOutcome,
}

fn eval_deletion(
    ds: &Dataset,
    k: usize,
    seed: &[f64],
    i: usize,
    upper: f64,
    cfg: &AlgoConfig,
    inv: Option<&InverseSym>,
) -> Eval {
    let mut out = match inv {
        Some(iv) => {
            let mut op = DelOp::Inverse(NegInvSubOp::new(iv, k, Some(i)));
            let o = smallest_eigenpair(&mut op, Some(seed), cfg.tol, cfg.max_iters_warm);
            LanczosOutcome { value: to_lambda(o.value, true), ..o }
        }
        None => {
            let mut op = DelOp::Direct(SubOp::deleted(ds, k, i));
            smallest_eigenpair(&mut op, Some(seed), cfg.tol, cfg.max_iters_warm)
        }
    };
    out.value = out.value.max(f64::NEG_INFINITY);
    Eval { pos: i, upper, out }
}

/// Graine d'evaluation pour le retrait de la variable `i` : vecteur propre exact de
/// `R_{-i}` approche par `(R - mu I)^{-1} e_i` (`mu` = racine seculaire tronquee),
/// avec repli sur `u_1` restreint.
///
/// Le vecteur d'essai est de dimension `k` (il porte la composante `i`) : on le
/// **restreint** a `{0..k}\{i}` pour qu'il soit un vecteur de l'operateur de
/// dimension `k-1`. Sans cette restriction, le solveur rejetait silencieusement la
/// graine (longueur incompatible) et repartait d'un vecteur aleatoire, ce qui
/// multipliait par ~30 le nombre d'iterations Lanczos par candidat.
fn deletion_seed(spec: &LowSpectrum, i: usize, k: usize) -> Vec<f64> {
    let u = &spec.vectors[0];
    let restrict = |v: &[f64]| {
        let mut s = Vec::with_capacity(k - 1);
        s.extend_from_slice(&v[..i]);
        s.extend_from_slice(&v[i + 1..k]);
        s
    };
    match secular_root(spec, i) {
        Some(mu) => {
            let x = trial_vector(spec, i, mu);
            let n = crate::num::norm2(&x);
            if n > 1e-8 {
                restrict(&x)
            } else {
                restrict(u)
            }
        }
        None => restrict(u),
    }
}

/// Une etape d'elimination : choisit la variable a retirer et renvoie le couple
/// propre du nouvel ensemble.
fn step_eliminate(
    ds: &Dataset,
    k: usize,
    spec: &LowSpectrum,
    cfg: &AlgoConfig,
    inv: Option<&InverseSym>,
) -> (usize, StepRecord, LanczosOutcome) {
    let u = &spec.vectors[0];
    let (rq, y) = head_rayleigh(ds, k, u);
    let mut cands: Vec<(f64, f64, usize)> = (0..k)
        .map(|i| {
            let ray = rayleigh_upper(rq, u[i], y[i]);
            let ub = spec.upper_bound(i, ray);
            (ub, ray, i)
        })
        .collect();
    // On commence par les bornes les PLUS GRANDES : ce sont les seules qui peuvent
    // encore battre le meilleur candidat deja evalue ; des que la meilleure valeur
    // exacte depasse la plus grande borne restante, l'etape est certifiee.
    cands.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Less));

    let mut best: Option<Eval> = None;
    let mut evaluated = 0usize;
    let mut iters = 0usize;
    let mut certified = false;
    let margin = cfg.tol * 50.0 * (1.0 + rq.abs());
    let mut idx = 0usize;
    while idx < k {
        if let Some(b) = &best {
            if b.out.value >= cands[idx].0 - margin {
                certified = true;
                break;
            }
        }
        if cfg.max_exact > 0 && evaluated >= cfg.max_exact {
            break;
        }
        let mut bs = cfg.batch.max(1);
        if cfg.max_exact > 0 {
            bs = bs.min(cfg.max_exact - evaluated);
        }
        if bs == 0 {
            break;
        }
        let end = (idx + bs).min(k);
        let seeds: Vec<Vec<f64>> =
            cands[idx..end].iter().map(|&(_, _, i)| deletion_seed(spec, i, k)).collect();
        let batch: Vec<Eval> = cands[idx..end]
            .par_iter()
            .zip(seeds.par_iter())
            .map(|(&(ub, _ray, i), seed)| eval_deletion(ds, k, seed, i, ub, cfg, inv))
            .collect();
        for e in batch {
            evaluated += 1;
            iters += e.out.iters;
            // Ex aequo (a la tolerance pres) : on garde le premier rencontre, donc le
            // meilleur score, pour eviter que des differences numeriques aleatoires
            // ne tranchent entre candidats indiscernables.
            let better = match &best {
                None => true,
                Some(b) => e.out.value > b.out.value + cfg.tol * 10.0 * (1.0 + b.out.value.abs()),
            };
            if better {
                best = Some(e);
            }
        }
        idx = end;
    }
    if idx >= k {
        certified = true;
    }
    let best = best.expect("au moins un candidat evalue");
    let chosen = best.pos;
    let mut vector = best.out.vector.clone();
    if chosen + 1 < k {
        permute_after_swap(&mut vector, chosen, k);
    }
    let value = best.out.value;
    let residual = best.out.residual;
    let out =
        LanczosOutcome { value, vector, residual, iters: best.out.iters, converged: best.out.converged };
    let rec = StepRecord {
        k_before: k,
        k_after: k - 1,
        changed_orig: ds.orig(chosen),
        lambda: value,
        lambda_verified: f64::NAN,
        upper: best.upper,
        rayleigh: rayleigh_upper(rq, u[chosen], y[chosen]),
        exact_evals: evaluated,
        candidates: k,
        certified,
        residual,
        iters,
    };
    (chosen, rec, out)
}

/// Parametres de l'amelioration locale par echanges.
#[derive(Clone, Copy, Debug)]
pub struct RefineCfg {
    /// Nombre de passes.
    pub passes: usize,
    /// Sortants/entrants consideres par passe.
    pub top: usize,
    /// N'appliquer les echanges qu'aux tailles `k <= from_k`.
    pub from_k: usize,
}

/// Elimination arriere certifiee de `M` jusqu'a `k_min`.
///
/// `progress` est appele apres chaque etape et renvoie `true` pour poursuivre le
/// parcours, `false` pour l'interrompre proprement (le resultat partiel est
/// retourne, avec les etapes deja effectuees).
pub fn backward(
    ds: &mut Dataset,
    k_min: usize,
    cfg: &AlgoConfig,
    refine: Option<RefineCfg>,
    verify: bool,
    progress: &mut dyn FnMut(&StepRecord) -> bool,
) -> PathResult {
    let t0 = Instant::now();
    let m = ds.m();
    assert!(k_min >= 1 && k_min <= m, "k_min invalide");
    // Inverse maintenu : evalue les candidats via -R_{-i}^{-1} (convergence rapide).
    let want_inv = match cfg.eval {
        EvalMode::Direct => false,
        EvalMode::Inverse => true,
        EvalMode::Auto => ds.is_packed() && m <= cfg.inverse_max_m,
    };
    let mut inv = if want_inv {
        ds.packed.as_ref().and_then(|p| InverseSym::from_packed(p, m))
    } else {
        None
    };
    // Le spectre bas (donc les bornes de la cascade) est calcule en mode direct :
    // les bornes sont alors identiques a `--eval direct`, seul le calcul exact des
    // candidats exploite l'inverse.
    let init = low_spectrum(ds, m, cfg.num_low, &[], None, cfg, true);
    let initial_lambda = init.lambda_min();
    let initial_residual = init.max_residual();
    let mut spec = init;
    let mut k = m;
    let mut steps = Vec::with_capacity(m - k_min);
    while k > k_min {
        let (chosen, rec, winner) = step_eliminate(ds, k, &spec, cfg, inv.as_ref());
        ds.swap(chosen, k - 1);
        if let Some(iv) = inv.as_mut() {
            iv.swap_leading(chosen, k - 1, k);
            iv.downdate_last(k);
        }
        let mut seeds: Vec<Vec<f64>> =
            spec.vectors.iter().map(|v| restrict_and_permute(v, chosen, k)).collect();
        if !seeds.is_empty() {
            seeds[0] = winner.vector.clone();
        }
        k -= 1;
        let keep_going = progress(&rec);
        steps.push(rec);
        if !keep_going {
            break;
        }
        if k >= k_min && k > 1 {
            if verify {
                let strict = AlgoConfig {
                    tol: 1e-13,
                    max_iters_cold: 4000,
                    max_iters_warm: 400,
                    ..*cfg
                };
                let ev = initial_eigenpair_inv(ds, k, &strict, inv.as_ref());
                if let Some(last) = steps.last_mut() {
                    last.lambda_verified = ev.value;
                }
            }
            spec = low_spectrum(ds, k, cfg.num_low, &seeds, Some(&winner), cfg, false);
        }
        if let Some(r) = refine {
            if k >= 2 && k <= r.from_k && r.passes > 0 {
                let mut u = spec.vectors[0].clone();
                // `refine_swaps` ne pilote pas l'arret : on capture son drapeau via
                // une fermeture locale, puis on interrompt la cascade si demande.
                let mut stop = false;
                let (nl, _imp) = {
                    let mut relay = |st: &StepRecord| {
                        if !progress(st) {
                            stop = true;
                        }
                    };
                    refine_swaps(ds, k, spec.lambda_min(), &mut u, cfg, r.passes, r.top, &mut relay)
                };
                if let Some(last) = steps.last_mut() {
                    last.lambda = nl;
                }
                if stop {
                    break;
                }
                spec = low_spectrum(ds, k, cfg.num_low, &[], None, cfg, false);
                if !spec.vectors.is_empty() {
                    spec.vectors[0] = u;
                    spec.values[0] = nl;
                }
            }
        }
    }
    PathResult {
        direction: Direction::Backward,
        initial_k: m,
        initial_lambda,
        initial_residual,
        steps,
        seconds: t0.elapsed().as_secs_f64(),
        initial_subset: (0..m).collect(),
    }
}

/// Evaluation exacte de l'ajout de la variable `j` (vecteur d'essai KKT en graine).
///
/// Si la correlation est materialisee, on evalue la matrice bordee directement
/// (`O(k^2)` par matvec) plutot que via `Z` (`O(N k)`) : a `N` grand c'est le
/// facteur limitant du glouton avant exact.
fn eval_addition(
    ds: &Dataset,
    k: usize,
    seed: &[f64],
    j: usize,
    upper: f64,
    cfg: &AlgoConfig,
) -> Eval {
    let out = match ds.packed.as_ref() {
        Some(p) => {
            let mut op = BorderedHeadOp::new(p, k, j);
            smallest_eigenpair(&mut op, Some(seed), cfg.tol, cfg.max_iters_warm)
        }
        None => {
            let mut op = GatheredZOp::build(&ds.z, k, None, Some(j));
            smallest_eigenpair(&mut op, Some(seed), cfg.tol, cfg.max_iters_warm)
        }
    };
    Eval { pos: j, upper, out }
}

/// Racine de l'equation seculaire de la matrice bordee
/// `[[R_S, c], [c^T, 1]]`, soit `1 - mu = sum_l g_l^2 / (lambda_l - mu)` avec
/// `g_l = u_l^T c`, cherchee sous `lambda_1`.  C'est un majorant de la nouvelle
/// `lambda_min` lorsque la somme est tronquee aux `p` premiers couples propres.
fn addition_secular(values: &[f64], g: &[f64]) -> f64 {
    let lam1 = values[0];
    let f = |mu: f64| -> f64 {
        1.0 - mu - values.iter().zip(g.iter()).map(|(l, gi)| gi * gi / (l - mu)).sum::<f64>()
    };
    let mut lo = -1.0 - lam1.abs() - g.iter().map(|x| x * x).sum::<f64>();
    let mut hi = lam1 - 1e-13 * (1.0 + lam1.abs());
    for _ in 0..10 {
        if f(lo) > 0.0 {
            break;
        }
        lo *= 2.0;
    }
    if !(lo < hi) || f(lo) <= 0.0 {
        return lam1;
    }
    for _ in 0..80 {
        let m = 0.5 * (lo + hi);
        if f(m) > 0.0 {
            lo = m;
        } else {
            hi = m;
        }
    }
    0.5 * (lo + hi)
}

/// Graine d'evaluation pour l'ajout de la variable `j` : vecteur d'essai KKT
/// `[y ; 1]` construit depuis la racine seculaire bordee, avec repli sur `[u_1 ; 1]`
/// (jamais catastrophique) si cette racine n'est pas exploitable.
fn addition_seed(spec: &LowSpectrum, g: &[f64], m: usize, j: usize) -> Vec<f64> {
    let p = spec.values.len();
    let k = spec.dim;
    let gl: Vec<f64> = (0..p).map(|l| g[l * m + j]).collect();
    let mu = addition_secular(&spec.values, &gl);
    let usable = mu < spec.values[0] - 1e-9 * (1.0 + spec.values[0].abs());
    let mut seed = vec![0.0; k];
    if usable {
        for l in 0..p {
            let d = spec.values[l] - mu;
            if d.abs() > 1e-300 {
                // y = -(R_S - mu I)^{-1} c  (systeme borde : voir doc)
                let wv = -gl[l] / d;
                if wv != 0.0 {
                    crate::num::axpy(wv, &spec.vectors[l], &mut seed);
                }
            }
        }
        // composante sur e_j : le vecteur d'essai exact verifie x = [y ; 1], on
        // normalise pour eviter les explosions.
        let ny = crate::num::norm2(&seed);
        if !(ny.is_finite()) || ny > 1e6 {
            seed = spec.vectors[0].clone();
        }
    } else {
        seed = spec.vectors[0].clone();
    }
    seed.push(1.0);
    seed
}

/// Selection avant : part d'un singleton et ajoute gloutonnement la variable qui
/// maximise `lambda_min`.
///
/// Le classement des candidats utilise l'equation seculaire de la matrice bordee :
/// la perte au premier ordre vaut `sum_l g_l^2 / (lambda_l - lambda_1)` avec
/// `g_l = u_l . c_j` (`c_j` = correlations de `j` avec `S`), au lieu du seul
/// `|g_1|` : les directions propres proches de `lambda_1` coutent plus cher.
///
/// `top = 0` (ou `top >= nombre de candidats`) desactive le pre-filtre : l'etape
/// est alors **certifiee optimale**. La racine seculaire tronquee majore
/// `lambda_min` de la matrice bordee ; en triant par borne DECROISSANTE et en
/// evaluant exactement par lots, on s'arrete des que la meilleure valeur realisee
/// depasse la plus grande borne restante — sans evaluer les `M-k` candidats.
/// C'est le mode par defaut ; un `top` fini (typiquement 16) est un choix explicite
/// de l'appelant, qui restreint l'evaluation aux `top` meilleurs du score de perte
/// et echange donc l'optimalite contre du temps.
///
/// Chaque candidat est evalue par Lanczos a chaud avec pour graine le vecteur
/// d'essai KKT `[y ; 1]`, `y = -sum_l g_l/(lambda_l - mu) u_l`.
///
/// `progress` renvoie `true` pour poursuivre, `false` pour s'arreter proprement.
pub fn forward(
    ds: &mut Dataset,
    k_max: usize,
    cfg: &AlgoConfig,
    top: usize,
    first: Option<usize>,
    progress: &mut dyn FnMut(&StepRecord) -> bool,
) -> PathResult {
    let t0 = Instant::now();
    let m = ds.m();
    assert!(k_max >= 1 && k_max <= m, "k_max invalide");
    let f = first.unwrap_or_else(|| best_first_feature(ds)).min(m - 1);
    ds.swap(0, f);
    let first_orig = ds.orig(0);
    let mut k = 1usize;
    let mut spec = LowSpectrum::singleton();
    let mut steps = Vec::with_capacity(k_max.saturating_sub(1));
    let mut carry: Option<LanczosOutcome>;
    while k < k_max {
        let cands: Vec<usize> = (k..m).collect();
        if cands.is_empty() {
            break;
        }
        // chargements w_l = Z_S u_l pour les p couples propres
        let p = spec.values.len();
        let mut w: Vec<Vec<f64>> = Vec::with_capacity(p);
        for l in 0..p {
            let mut wl = vec![0.0; ds.z.rows];
            z_loading(&ds.z, k, &spec.vectors[l], &mut wl);
            w.push(wl);
        }
        let mut c = vec![0.0; m];
        z_dot_all(&ds.z, &w[0], &mut c);
        let rq = spec.lambda_min();
        // g_{j,l} = u_l . c_j = w_l . z_j
        let mut g = vec![0.0f64; p * m];
        for l in 0..p {
            let mut gl = vec![0.0; m];
            z_dot_all(&ds.z, &w[l], &mut gl);
            g[l * m..(l + 1) * m].copy_from_slice(&gl);
        }
        // Score de premier ordre (perte seculaire estimee) : il sert au tri du
        // pre-filtre et a departager les ex aequo.
        let lam1 = spec.values[0];
        let mut scored: Vec<(f64, usize)> = cands
            .iter()
            .map(|&j| {
                let mut loss = 0.0;
                for l in 0..p {
                    let d = (spec.values[l] - lam1).max(1e-12);
                    let v = g[l * m + j];
                    loss += v * v / d;
                }
                (loss, j)
            })
            .collect();
        // Le pre-filtre n'est applique que si l'appelant l'a explicitement demande
        // (`top` fini) : `top = 0` realise le vrai glouton E-optimal, mais par
        // certification (voir plus bas) au lieu d'une evaluation exhaustive.
        let exact = top == 0 || top >= scored.len();

        // `work` = `(borne_or_score, perte, j)`.
        //  * mode exact   : `borne` est un majorant certifie de la nouvelle
        //    `lambda_min`, et le tri se fait par borne DECROISSANTE ;
        //  * mode filtre  : `borne` = `perte` (le score historique) et le tri se
        //    fait par perte croissante sur les `top` premiers.
        let work: Vec<(f64, f64, usize)> = if exact {
            // La racine seculaire tronquee aux `p` premiers couples propres majore
            // `lambda_min` de la matrice bordee `[[R_S, c],[c^T, 1]]` (les termes de
            // queue sont positifs sous `lambda_1`). Trier par borne decroissante puis
            // evaluer exactement permet de s'arreter des que la meilleure valeur
            // realisee depasse la plus grande borne restante : le choix est alors
            // celui du glouton exact, sans evaluer les `M-k` candidats.
            let mut bounded: Vec<(f64, f64, usize)> = scored
                .iter()
                .map(|&(loss, j)| {
                    let gl: Vec<f64> = (0..p).map(|l| g[l * m + j]).collect();
                    (addition_secular(&spec.values, &gl), loss, j)
                })
                .collect();
            bounded.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Less));
            bounded
        } else {
            scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(top);
            scored.into_iter().map(|(loss, j)| (loss, loss, j)).collect()
        };

        let margin = cfg.tol * 50.0 * (1.0 + lam1.abs());
        let mut best: Option<Eval> = None;
        let mut best_loss = f64::INFINITY;
        let mut iters = 0usize;
        let mut evaluated = 0usize;
        let mut certified = false;
        let mut idx = 0usize;
        while idx < work.len() {
            if exact {
                if let Some(b) = &best {
                    if b.out.value >= work[idx].0 - margin {
                        certified = true;
                        break;
                    }
                }
            }
            if cfg.max_exact > 0 && evaluated >= cfg.max_exact {
                break;
            }
            let mut bs = cfg.batch.max(1);
            if cfg.max_exact > 0 {
                bs = bs.min(cfg.max_exact - evaluated);
            }
            if bs == 0 {
                break;
            }
            let end = (idx + bs).min(work.len());
            let seeds: Vec<Vec<f64>> =
                work[idx..end].iter().map(|&(_, _, j)| addition_seed(&spec, &g, m, j)).collect();
            let batch: Vec<Eval> = work[idx..end]
                .par_iter()
                .zip(seeds.par_iter())
                .map(|(&(ub, _, j), seed)| eval_addition(ds, k, seed, j, ub, cfg))
                .collect();
            for (e, &(_, loss, _)) in batch.into_iter().zip(work[idx..end].iter()) {
                evaluated += 1;
                iters += e.out.iters;
                let better = match &best {
                    None => true,
                    Some(b) => {
                        let tie = cfg.tol * 10.0 * (1.0 + b.out.value.abs());
                        if e.out.value > b.out.value + tie {
                            true
                        } else if e.out.value >= b.out.value - tie {
                            // Ex aequo a la tolerance pres : on conserve le plus petit
                            // score de perte, comme le faisait le tri croissant.
                            loss < best_loss
                        } else {
                            false
                        }
                    }
                };
                if better {
                    best = Some(e);
                    best_loss = loss;
                }
            }
            idx = end;
        }
        if exact && idx >= work.len() {
            certified = true;
        }
        let best = best.expect("au moins un candidat evalue");
        let chosen = best.pos;
        let value = best.out.value;
        let residual = best.out.residual;
        let upper = best_loss;
        let vector = best.out.vector.clone();
        ds.swap(k, chosen);
        let rec = StepRecord {
            k_before: k,
            k_after: k + 1,
            changed_orig: ds.orig(k),
            lambda: value,
            lambda_verified: f64::NAN,
            upper: rq,
            rayleigh: upper,
            exact_evals: evaluated,
            candidates: cands.len(),
            certified,
            residual,
            iters,
        };
        k += 1;
        let keep_going = progress(&rec);
        steps.push(rec);
        if !keep_going {
            break;
        }
        // spectre du nouvel ensemble (demarrages a chaud : vecteurs etendus d'un 0)
        if k < k_max {
            let seeds: Vec<Vec<f64>> = spec
                .vectors
                .iter()
                .map(|v| {
                    let mut e = v.clone();
                    e.push(0.0);
                    e
                })
                .collect();
            carry = Some(LanczosOutcome {
                value,
                vector,
                residual,
                iters,
                converged: best.out.converged,
            });
            spec = low_spectrum(ds, k, cfg.num_low, &seeds, carry.as_ref(), cfg, false);
        }
    }
    PathResult {
        direction: Direction::Forward,
        initial_k: 1,
        initial_lambda: 1.0,
        initial_residual: 0.0,
        steps,
        seconds: t0.elapsed().as_secs_f64(),
        initial_subset: vec![first_orig],
    }
}

/// Premiere variable : celle dont la correlation maximale avec les autres est la plus
/// faible (heuristique de type Gershgorin). En representation implicite, on se rabat
/// sur la colonne la plus eloignee du centroide (`O(NM)`).
pub fn best_first_feature(ds: &Dataset) -> usize {
    let m = ds.m();
    if m <= 1 {
        return 0;
    }
    if let Some(p) = ds.packed.as_ref() {
        if m <= 20_000 {
            let raw = p.raw();
            let mut best = 0usize;
            let mut best_score = f64::INFINITY;
            for i in 0..m {
                let base = crate::packed::row_offset(i);
                let mut mx = 0.0f64;
                for &v in &raw[base..base + i] {
                    let a = v.abs();
                    if a > mx {
                        mx = a;
                    }
                }
                for j in (i + 1)..m {
                    let a = p.get(j, i).abs();
                    if a > mx {
                        mx = a;
                    }
                }
                if mx < best_score {
                    best_score = mx;
                    best = i;
                }
            }
            return best;
        }
        return 0;
    }
    let rows = ds.z.rows;
    let mut centroid = vec![0.0; rows];
    for j in 0..m {
        crate::num::axpy(1.0, ds.z.col(j), &mut centroid);
    }
    let inv = 1.0 / m as f64;
    for v in centroid.iter_mut() {
        *v *= inv;
    }
    let mut best = 0usize;
    let mut best_d = -1.0f64;
    for j in 0..m {
        let mut d = 0.0f64;
        for (a, b) in ds.z.col(j).iter().zip(centroid.iter()) {
            let t = a - b;
            d += t * t;
        }
        if d > best_d {
            best_d = d;
            best = j;
        }
    }
    best
}

/// Amelioration locale par echanges 1-contre-1 (taille `K` constante).
pub fn refine_swaps(
    ds: &mut Dataset,
    k: usize,
    lam: f64,
    u: &mut Vec<f64>,
    cfg: &AlgoConfig,
    passes: usize,
    top: usize,
    progress: &mut dyn FnMut(&StepRecord),
) -> (f64, usize) {
    if k < 2 || k >= ds.m() {
        return (lam, 0);
    }
    let mut cur = lam;
    let mut improvements = 0usize;
    for _pass in 0..passes {
        let (rq, ys) = head_rayleigh(ds, k, u);
        let mut outs: Vec<(f64, usize)> =
            (0..k).map(|i| (rayleigh_upper(rq, u[i], ys[i]), i)).collect();
        outs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut w = vec![0.0; ds.z.rows];
        z_loading(&ds.z, k, u, &mut w);
        let mut g = vec![0.0; ds.m()];
        z_dot_all(&ds.z, &w, &mut g);
        let mut ins: Vec<(f64, usize)> = (k..ds.m()).map(|j| (g[j].abs(), j)).collect();
        ins.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let nt = top.max(1);
        let mut pairs: Vec<(f64, f64, usize, usize)> = Vec::new();
        for &(rho_i, i) in outs.iter().take(nt) {
            for &(gj, j) in ins.iter().take(nt) {
                pairs.push((rho_i - gj, rho_i, i, j));
            }
        }
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        pairs.truncate(cfg.batch.max(1));
        let evals: Vec<(usize, usize, f64, LanczosOutcome)> = pairs
            .par_iter()
            .map(|&(_, rho_i, i, j)| {
                let mut op = GatheredZOp::build(&ds.z, k, Some(i), Some(j));
                let mut seed = Vec::with_capacity(k);
                seed.extend_from_slice(&u[..i]);
                seed.extend_from_slice(&u[i + 1..]);
                seed.push(0.0);
                let out = smallest_eigenpair(&mut op, Some(&seed), cfg.tol, cfg.max_iters_warm);
                (i, j, rho_i, out)
            })
            .collect();
        let mut bi = 0usize;
        let mut bj = 0usize;
        let mut bup = f64::NAN;
        let mut bval = cur;
        let mut bvec: Vec<f64> = Vec::new();
        let mut bres = 0.0f64;
        let mut bit = 0usize;
        for (i, j, rho_i, out) in evals {
            bit += out.iters;
            if out.value > bval {
                bval = out.value;
                bup = rho_i;
                bvec = out.vector;
                bres = out.residual;
                bi = i;
                bj = j;
            }
        }
        if bvec.is_empty() || bval <= cur {
            break;
        }
        let orig_out = ds.orig(bi);
        ds.swap(bi, bj);
        permute_after_swap(&mut bvec, bi, k);
        *u = bvec;
        progress(&StepRecord {
            k_before: k,
            k_after: k,
            changed_orig: orig_out,
            lambda: bval,
            lambda_verified: f64::NAN,
            upper: bup,
            rayleigh: bup,
            exact_evals: pairs.len(),
            candidates: outs.len() * ins.len(),
            certified: false,
            residual: bres,
            iters: bit,
        });
        cur = bval;
        improvements += 1;
    }
    (cur, improvements)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gen::{generate, GenKind};
    use crate::jacobi::eigen_sym_sorted;
    use crate::op::{Dataset, Repr};

    fn dense_sub(ds: &Dataset, idx: &[usize]) -> Vec<f64> {
        let d = idx.len();
        let mut a = vec![0.0; d * d];
        for (x, &ix) in idx.iter().enumerate() {
            for (y, &iy) in idx.iter().enumerate() {
                let v = match ds.packed.as_ref() {
                    Some(p) => p.get(ix, iy),
                    None => dot(ds.z.col(ix), ds.z.col(iy)),
                };
                a[x * d + y] = v;
            }
        }
        a
    }

    /// La borne de Rayleigh doit etre valide pour TOUT vecteur unitaire.
    #[test]
    fn rayleigh_upper_is_valid_for_any_unit_vector() {
        let mut dm = generate(GenKind::Blocks, 200, 24, 0.5, 0.1, 3, 1, 77);
        dm.standardize(true);
        let ds = Dataset::new(dm, Repr::Packed, usize::MAX, 8);
        let k = 24;
        let cfg = AlgoConfig { tol: 1e-13, max_iters_cold: 1000, ..Default::default() };
        let spec = low_spectrum(&ds, k, 1, &[], None, &cfg, true);
        let mut rng = crate::gen::Rng::new(2024);
        let mut vectors: Vec<Vec<f64>> = vec![spec.vectors[0].clone()];
        for _ in 0..20 {
            let mut v: Vec<f64> = (0..k).map(|_| rng.normal()).collect();
            let n = crate::num::norm2(&v);
            for x in v.iter_mut() {
                *x /= n;
            }
            vectors.push(v);
        }
        for u in &vectors {
            let mut y = vec![0.0; k];
            let mut op = SubOp::head(&ds, k);
            op.mul(u, &mut y);
            let rq = dot(u, &y);
            for i in 0..k {
                let rho = rayleigh_upper(rq, u[i], y[i]);
                let idx: Vec<usize> = (0..k).filter(|&j| j != i).collect();
                let mut a = dense_sub(&ds, &idx);
                let (vals, _) = eigen_sym_sorted(&mut a, k - 1);
                assert!(vals[0] <= rho + 1e-10, "i={i} lam_min={} rho={rho}", vals[0]);
            }
        }
    }

    /// La borne de Temple doit dominer `lambda_min(R_{-i})` et etre plus serree.
    #[test]
    fn temple_bound_is_valid_and_tighter() {
        for seed in 0..5u64 {
            let mut dm = generate(GenKind::Blocks, 120, 22, 0.6, 0.05, 3, 1, seed);
            dm.standardize(true);
            let ds = Dataset::new(dm, Repr::Packed, usize::MAX, 8);
            let k = 22;
            let cfg = AlgoConfig { tol: 1e-13, max_iters_cold: 2000, ..Default::default() };
            let spec = low_spectrum(&ds, k, 8, &[], None, &cfg, true);
            let (rq, y) = head_rayleigh(&ds, k, &spec.vectors[0]);
            let mut sum_ray = 0.0f64;
            let mut sum_tem = 0.0f64;
            for i in 0..k {
                let ray = rayleigh_upper(rq, spec.vectors[0][i], y[i]);
                let ub = spec.upper_bound(i, ray);
                let idx: Vec<usize> = (0..k).filter(|&j| j != i).collect();
                let mut a = dense_sub(&ds, &idx);
                let (vals, _) = eigen_sym_sorted(&mut a, k - 1);
                assert!(ub >= vals[0] - 1e-8, "seed={seed} i={i}: ub={ub} exact={}", vals[0]);
                sum_ray += ray - vals[0];
                sum_tem += ub - vals[0];
            }
            assert!(sum_tem < sum_ray, "la borne de Temple doit etre plus serree");
            eprintln!(
                "seed={seed} ecart moyen Rayleigh={:.5} Temple={:.5}",
                sum_ray / k as f64,
                sum_tem / k as f64
            );
        }
    }

    /// Le choix glouton doit coincider avec la force brute (petits M).
    #[test]
    fn backward_matches_brute_force_greedy() {
        for seed in 0..4u64 {
            let mut dm = generate(GenKind::Blocks, 120, 12, 0.6, 0.05, 3, 1, seed);
            dm.standardize(true);
            let mut ds = Dataset::new(dm, Repr::Packed, usize::MAX, 8);
            let cfg =
                AlgoConfig { tol: 1e-13, max_iters_cold: 2000, max_iters_warm: 200, ..Default::default() };
            let m = ds.m();
            let path = backward(&mut ds, 1, &cfg, None, false, &mut |_| true);
            let mut ds2 = Dataset::new(ds.z.clone(), Repr::Packed, usize::MAX, 8);
            let mut k = m;
            let mut step = 0;
            while k > 1 {
                let mut best_i = 0usize;
                let mut best_v = f64::NEG_INFINITY;
                for i in 0..k {
                    let idx: Vec<usize> = (0..k).filter(|&j| j != i).collect();
                    let mut a = dense_sub(&ds2, &idx);
                    let (vals, _) = eigen_sym_sorted(&mut a, k - 1);
                    if vals[0] > best_v {
                        best_v = vals[0];
                        best_i = i;
                    }
                }
                let got = &path.steps[step];
                assert!(
                    (got.lambda - best_v).abs() < 1e-7,
                    "seed={seed} k={k}: glouton {} vs brute force {}",
                    got.lambda,
                    best_v
                );
                ds2.swap(best_i, k - 1);
                k -= 1;
                step += 1;
            }
        }
    }

    /// L'evaluation par l'inverse doit donner la meme valeur que l'evaluation directe
    /// (et le meme couple propre), a la tolerance du solveur pres.
    #[test]
    fn inverse_eval_matches_direct_eval() {
        let mut dm = generate(GenKind::Blocks, 300, 24, 0.6, 0.05, 3, 1, 5);
        dm.standardize(true);
        let ds = Dataset::new(dm, Repr::Packed, usize::MAX, 8);
        let k = 24;
        let cfg = AlgoConfig { tol: 1e-12, max_iters_cold: 4000, max_iters_warm: 400, ..Default::default() };
        let inv = InverseSym::from_packed(ds.packed.as_ref().unwrap(), k).expect("PD");
        let direct = initial_eigenpair_public(&ds, k, &cfg);
        let via_inv = initial_eigenpair_inv(&ds, k, &cfg, Some(&inv));
        assert!(
            (direct.value - via_inv.value).abs() < 1e-9,
            "{} vs {}",
            direct.value,
            via_inv.value
        );
        let spec = low_spectrum(&ds, k, 2, &[], Some(&direct), &cfg, true);
        for i in [0usize, 1, 7, 13, 23] {
            let seed = deletion_seed(&spec, i, k);
            let d = eval_deletion(&ds, k, &seed, i, f64::INFINITY, &cfg, None);
            let v = eval_deletion(&ds, k, &seed, i, f64::INFINITY, &cfg, Some(&inv));
            assert!(
                (d.out.value - v.out.value).abs() < 1e-8,
                "i={i}: direct {} inverse {}",
                d.out.value,
                v.out.value
            );
        }
    }

    /// La voie implicite doit donner les memes resultats que la voie packed.
    #[test]
    fn backward_implicit_matches_packed() {
        let mut dm = generate(GenKind::Equi, 150, 30, 0.4, 0.0, 1, 1, 3);
        dm.standardize(true);
        let cfg = AlgoConfig { tol: 1e-12, max_iters_cold: 2000, max_iters_warm: 200, ..Default::default() };
        let mut dsp = Dataset::new(dm.clone(), Repr::Packed, usize::MAX, 8);
        let mut dsi = Dataset::new(dm, Repr::Implicit, 0, 8);
        let rp = backward(&mut dsp, 5, &cfg, None, false, &mut |_| true);
        let ri = backward(&mut dsi, 5, &cfg, None, false, &mut |_| true);
        for (a, b) in rp.steps.iter().zip(ri.steps.iter()) {
            assert!((a.lambda - b.lambda).abs() < 1e-8, "{} vs {}", a.lambda, b.lambda);
        }
    }

    /// La graine seculaire de suppression doit etre de dimension `k-1` (vecteur de
    /// l'operateur `R_{-i}`) : c'est ce qui la rend exploitable par Lanczos. Tant
    /// qu'elle gardait la composante `i`, le solveur la rejetait (longueur `k`) et
    /// repartait d'un vecteur aleatoire, multipliant les iterations par ~30.
    #[test]
    fn deletion_seed_is_restricted_and_effective() {
        let mut dm = generate(GenKind::Blocks, 600, 120, 0.3, 0.0, 8, 1, 4);
        dm.standardize(true);
        let ds = Dataset::new(dm, Repr::Packed, usize::MAX, 16);
        let cfg = AlgoConfig { tol: 1e-13, max_iters_cold: 3000, max_iters_warm: 400, ..Default::default() };
        let k = 80usize;
        let spec = low_spectrum(&ds, k, 4, &[], None, &cfg, true);
        let inv = InverseSym::from_packed(ds.packed.as_ref().unwrap(), k).expect("PD");
        let mut seeded = 0usize;
        let mut blind = 0usize;
        for i in [3usize, 17, 41, 62] {
            let seed = deletion_seed(&spec, i, k);
            assert_eq!(seed.len(), k - 1, "la graine doit vivre dans R^(k-1)");
            assert!(crate::num::norm2(&seed) > 0.0);
            seeded += eval_deletion(&ds, k, &seed, i, f64::INFINITY, &cfg, Some(&inv)).out.iters;
            let rnd = crate::lanczos::default_seed(k - 1, i as u64);
            blind += eval_deletion(&ds, k, &rnd, i, f64::INFINITY, &cfg, Some(&inv)).out.iters;
        }
        assert!(
            seeded < blind,
            "la graine seculaire doit reduire les iterations : {seeded} vs {blind}"
        );
    }

    /// Le glouton avant doit coincider avec la force brute, y compris en mode
    /// certifie (le pre-filtre est desactive : `top = 0`).
    #[test]
    fn forward_matches_brute_force_greedy() {
        for seed in 0..3u64 {
            let mut dm = generate(GenKind::Blocks, 120, 11, 0.6, 0.05, 3, 1, seed);
            dm.standardize(true);
            let mut ds = Dataset::new(dm, Repr::Packed, usize::MAX, 8);
            let cfg = AlgoConfig { tol: 1e-13, max_iters_cold: 2000, max_iters_warm: 400, ..Default::default() };
            let m = ds.m();
            let path = forward(&mut ds, m, &cfg, 0, None, &mut |_| true);
            assert_eq!(path.steps.len(), m - 1);
            for st in &path.steps {
                assert!(st.certified, "chaque etape du mode exact doit etre certifiee");
            }
            // Rejoue le glouton exact en coordonnees PHYSIQUES : `forward` a
            // ramene les variables choisies en tete, donc l'ensemble courant est
            // `0..k` et les candidats sont `k..m`.
            let ds2 = Dataset::new(ds.z.clone(), Repr::Packed, usize::MAX, 8);
            for step in 0..(m - 1) {
                let k = step + 1;
                let mut best_v = f64::NEG_INFINITY;
                for j in k..m {
                    let mut idx: Vec<usize> = (0..k).collect();
                    idx.push(j);
                    let mut a = dense_sub(&ds2, &idx);
                    let (vals, _) = eigen_sym_sorted(&mut a, k + 1);
                    if vals[0] > best_v {
                        best_v = vals[0];
                    }
                }
                let got = &path.steps[step];
                assert!(
                    (got.lambda - best_v).abs() < 1e-8,
                    "seed={seed} k={}: glouton {} vs force brute {}",
                    k,
                    got.lambda,
                    best_v
                );
            }
        }
    }

    /// Diagnostique la qualite de la borne seculaire utilisee pour certifier la
    /// selection avant : ecart moyen a la vraie `lambda_min` de `R_{S cup {j}}`.
    #[test]
    fn forward_secular_bound_quality() {
        let mut dm = generate(GenKind::Blocks, 500, 140, 0.3, 0.0, 8, 1, 4);
        dm.standardize(true);
        let ds = Dataset::new(dm, Repr::Packed, usize::MAX, 16);
        let cfg = AlgoConfig { tol: 1e-13, max_iters_cold: 3000, ..Default::default() };
        let p = ds.packed.as_ref().expect("packed");
        for k in [10usize, 30, 60] {
            let spec = low_spectrum(&ds, k, 4, &[], None, &cfg, true);
            let mut g = vec![0.0f64; spec.values.len()];
            let mut worst = 0.0f64;
            let mut sum = 0.0f64;
            let mut n = 0usize;
            for j in k..ds.m() {
                for l in 0..spec.values.len() {
                    let c: f64 = (0..k).map(|i| p.get(i, j) * spec.vectors[l][i]).sum();
                    g[l] = c;
                }
                let ub = addition_secular(&spec.values, &g);
                // exact : Jacobi sur la matrice bordee
                let d = k + 1;
                let mut a = vec![0.0; d * d];
                for x in 0..k {
                    for y in 0..k {
                        a[x * d + y] = p.get(x, y);
                    }
                    a[x * d + k] = p.get(x, j);
                    a[k * d + x] = p.get(x, j);
                }
                a[k * d + k] = 1.0;
                let (vals, _) = eigen_sym_sorted(&mut a, d);
                assert!(ub >= vals[0] - 1e-9, "borne invalide : {ub} < {}", vals[0]);
                worst = worst.max(ub - vals[0]);
                sum += ub - vals[0];
                n += 1;
            }
            eprintln!(
                "k={k}: ecart borne-exact moyen={:.3e} max={:.3e}",
                sum / n as f64,
                worst
            );
        }
    }
}

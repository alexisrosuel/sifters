//! Extension native `minvp._minvp` (PyO3).
//!
//! Ce module est la couche basse du paquet Python : une unique porte d'entree,
//! [`run`], qui prend la matrice de donnees sous forme de tampon `float64`,
//! execute le parcours glouton du crate `minvp` et renvoie un objet
//! [`Selection`] contenant la famille imbriquee complete.
//!
//! Tout le calcul est effectue **sans le GIL** ([`Python::detach`]) : les autres
//! threads Python peuvent travailler pendant une selection, et le pool `rayon`
//! utilise tous les cœurs.  Le callback `progress` optionnel est rappele **avec**
//! le GIL, une fois par etape, et peut interrompre le parcours ; les signaux
//! POSIX (`Ctrl-C`) sont verifies au meme moment.
//!
//! Le paquet `minvp` (voir `python/minvp/__init__.py`) ajoute au-dessus de cette
//! couche l'ergonomie Python : conversion des listes/DataFrames, choix
//! automatique du sens de parcours, stubs de typage et documentation.

// PyO3 genere du code `unsafe` (FFI) : `forbid(unsafe_code)` est donc impossible
// ici. Aucun `unsafe` n'est ecrit a la main dans ce crate.
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_debug_implementations)]

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::Instant;

use pyo3::buffer::PyBuffer;
use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyModule;

use minvp::greedy::{self, AlgoConfig, Direction, EvalMode, PathResult, RefineCfg, StepRecord};
use minvp::matrix::DataMatrix;
use minvp::op::{Dataset, Repr};

create_exception!(
    _minvp,
    MinvpError,
    PyRuntimeError,
    "Erreur remontee par le moteur numerique minvp."
);

/// Version du module natif (synchronisee avec le paquet Python).
const VERSION: &str = "0.2.0";

/// Largeur du pre-filtre avant active par `prefilter=true` : seuls les 16
/// meilleurs candidats (score seculaire) sont evalues exactement. `forward_top`
/// permet de choisir une autre largeur.
const PREFILTER_TOP: usize = 16;

// ---------------------------------------------------------------------------
// Resultats exposes a Python
// ---------------------------------------------------------------------------

/// Une etape du parcours glouton (retrait, ajout ou echange local).
///
/// Les indices sont donnes dans la numerotation des colonnes de la matrice
/// d'entree, avant suppression eventuelle des colonnes degenerees.
#[pyclass(frozen, module = "minvp._minvp", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct Step {
    /// Taille de l'ensemble avant l'etape.
    #[pyo3(get)]
    pub k_before: usize,
    /// Taille de l'ensemble apres l'etape.
    #[pyo3(get)]
    pub k_after: usize,
    /// Variable ajoutee, retiree ou sortante (indice d'origine).
    #[pyo3(get)]
    pub index: usize,
    /// `lambda_min` du nouvel ensemble (valeur de Ritz a chaud).
    #[pyo3(get)]
    pub lambda_min: f64,
    /// Recalcul a froid strict de `lambda_min` (`nan` si non demande).
    #[pyo3(get)]
    pub lambda_verified: f64,
    /// Borne superieure du candidat retenu.
    #[pyo3(get)]
    pub upper_bound: f64,
    /// Borne de Rayleigh seule du candidat retenu.
    #[pyo3(get)]
    pub rayleigh_bound: f64,
    /// Candidats examines exactement.
    #[pyo3(get)]
    pub exact_evals: usize,
    /// Candidats disponibles pour cette etape.
    #[pyo3(get)]
    pub candidates: usize,
    /// Vrai si l'optimalite gloutonne de l'etape est certifiee.
    #[pyo3(get)]
    pub certified: bool,
    /// Residu de Lanczos de la valeur retenue.
    #[pyo3(get)]
    pub residual: f64,
    /// Iterations de Lanczos cumulees sur l'etape.
    #[pyo3(get)]
    pub lanczos_iters: usize,
}

impl Step {
    fn from_record(rec: &StepRecord, keep_map: &[usize]) -> Self {
        Self {
            k_before: rec.k_before,
            k_after: rec.k_after,
            index: map_index(rec.changed_orig, keep_map),
            lambda_min: rec.lambda,
            lambda_verified: rec.lambda_verified,
            upper_bound: rec.upper,
            rayleigh_bound: rec.rayleigh,
            exact_evals: rec.exact_evals,
            candidates: rec.candidates,
            certified: rec.certified,
            residual: rec.residual,
            lanczos_iters: rec.iters,
        }
    }
}

#[pymethods]
impl Step {
    fn __repr__(&self) -> String {
        let kind = match self.k_after.cmp(&self.k_before) {
            std::cmp::Ordering::Less => "remove",
            std::cmp::Ordering::Greater => "add",
            std::cmp::Ordering::Equal => "swap",
        };
        format!(
            "Step({kind} index={}, K {} -> {}, lambda_min={:.9}, certified={})",
            self.index, self.k_before, self.k_after, self.lambda_min, self.certified
        )
    }
}

/// Resultat complet d'une selection : famille imbriquee + courbe + diagnostic.
///
/// Les indices (`subset`, `order`, `initial_subset`, `steps[i].index`) renvoient
/// toujours aux colonnes de la matrice d'entree : les colonnes degenerees
/// ecartees par la standardisation sont listees dans :attr:`dropped`.
#[pyclass(frozen, module = "minvp._minvp", skip_from_py_object)]
#[derive(Debug)]
pub struct Selection {
    path: PathResult,
    keep_map: Vec<usize>,
    steps: Vec<Py<Step>>,
    k: usize,
    subset: Vec<usize>,
    lambda_min: f64,
    lambda_residual: f64,
    n: usize,
    m_total: usize,
    dropped: Vec<usize>,
    seconds: f64,
    path_seconds: f64,
    load_seconds: f64,
    representation: &'static str,
    eval: &'static str,
}

#[pymethods]
impl Selection {
    /// Sens du parcours : `"backward"` ou `"forward"`.
    #[getter]
    fn direction(&self) -> &'static str {
        match self.path.direction {
            Direction::Backward => "backward",
            Direction::Forward => "forward",
        }
    }

    /// Taille de l'ensemble retenu (:attr:`subset`).
    #[getter]
    fn k(&self) -> usize {
        self.k
    }

    /// Variables retenues (indices d'origine, croissants).
    #[getter]
    fn subset(&self) -> Vec<usize> {
        self.subset.clone()
    }

    /// `lambda_min` du sous-ensemble retenu (revalide a froid si `verify`).
    #[getter]
    fn lambda_min(&self) -> f64 {
        self.lambda_min
    }

    /// Residu de Lanczos associe a :attr:`lambda_min`.
    #[getter]
    fn lambda_residual(&self) -> f64 {
        self.lambda_residual
    }

    /// Courbe `{K: lambda_min}` de toute la famille, cles croissantes.
    #[getter]
    fn curve(&self) -> BTreeMap<usize, f64> {
        self.path.curve().into_iter().collect()
    }

    /// Duree totale de l'appel (s), normalisation et verification incluses.
    #[getter]
    fn seconds(&self) -> f64 {
        self.seconds
    }

    /// Duree du seul parcours glouton (s).
    #[getter]
    fn path_seconds(&self) -> f64 {
        self.path_seconds
    }

    /// Duree de la lecture + standardisation + construction de la correlation.
    #[getter]
    fn load_seconds(&self) -> f64 {
        self.load_seconds
    }

    /// Ordre des variables retirees (arriere) ou ajoutees (avant), en indices d'origine.
    #[getter]
    fn order(&self) -> Vec<usize> {
        self.path.order().iter().map(|&i| map_index(i, &self.keep_map)).collect()
    }

    /// Sous-ensemble de depart (indices d'origine).
    #[getter]
    fn initial_subset(&self) -> Vec<usize> {
        self.path
            .initial_subset
            .iter()
            .map(|&i| map_index(i, &self.keep_map))
            .collect()
    }

    /// Nombre d'observations (lignes de la matrice d'entree).
    #[getter]
    fn n_observations(&self) -> usize {
        self.n
    }

    /// Nombre de variables de la matrice d'entree.
    #[getter]
    fn m_total(&self) -> usize {
        self.m_total
    }

    /// Nombre de variables effectivement utilisables (colonnes non degenerees).
    #[getter]
    fn m_usable(&self) -> usize {
        self.keep_map.len()
    }

    /// Colonnes ecartees car de variance nulle (indices d'origine).
    #[getter]
    fn dropped(&self) -> Vec<usize> {
        self.dropped.clone()
    }

    /// Etapes du parcours.
    #[getter]
    fn steps<'py>(&self, py: Python<'py>) -> Vec<Py<Step>> {
        self.steps.iter().map(|s| s.clone_ref(py)).collect()
    }

    /// Etapes dont l'optimalite gloutonne est certifiee.
    #[getter]
    fn certified_steps(&self) -> usize {
        self.path.certified_steps()
    }

    /// Nombre total de candidats evalues exactement.
    #[getter]
    fn total_exact_evals(&self) -> usize {
        self.path.total_exact()
    }

    /// Iterations de Lanczos cumulees.
    #[getter]
    fn total_lanczos_iters(&self) -> usize {
        self.path.total_iters()
    }

    /// Representation de la correlation utilisee (`"packed"` ou `"implicit"`).
    #[getter]
    fn representation(&self) -> &'static str {
        self.representation
    }

    /// Methode d'evaluation reellement employee (`"direct"` ou `"inverse"`).
    #[getter]
    fn eval(&self) -> &'static str {
        self.eval
    }

    /// Sous-ensemble de taille `k` dans la famille imbriquee, ou `None`.
    ///
    /// Les tailles valides vont de la fin du parcours a son debut (typiquement
    /// `kmin..=m_usable` en mode arriere, `1..=kmax` en mode avant).
    fn subset_at(&self, k: usize) -> Option<Vec<usize>> {
        self.path
            .subset_at(k)
            .map(|s| s.into_iter().map(|i| map_index(i, &self.keep_map)).collect())
    }

    /// `lambda_min` associe a la taille `k` du parcours, ou `None`.
    fn lambda_at(&self, k: usize) -> Option<f64> {
        self.path.lambda_at(k)
    }

    /// Valeur brute (sans revalidation a froid) pour la taille `k`.
    fn lambda_raw_at(&self, k: usize) -> Option<f64> {
        self.path.lambda_raw_at(k)
    }

    /// Meme contenu, sous forme de dictionnaire imbrique (serialisable JSON).
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let d = pyo3::types::PyDict::new(py);
        d.set_item("direction", self.direction())?;
        d.set_item("k", self.k)?;
        d.set_item("subset", self.subset.clone())?;
        d.set_item("lambda_min", self.lambda_min)?;
        d.set_item("lambda_residual", self.lambda_residual)?;
        d.set_item("curve", self.curve())?;
        d.set_item("order", self.order())?;
        d.set_item("initial_subset", self.initial_subset())?;
        d.set_item("n_observations", self.n)?;
        d.set_item("m_total", self.m_total)?;
        d.set_item("m_usable", self.m_usable())?;
        d.set_item("dropped", self.dropped.clone())?;
        d.set_item("seconds", self.seconds)?;
        d.set_item("path_seconds", self.path_seconds)?;
        d.set_item("load_seconds", self.load_seconds)?;
        d.set_item("representation", self.representation)?;
        d.set_item("eval", self.eval)?;
        d.set_item("certified_steps", self.certified_steps())?;
        d.set_item("total_exact_evals", self.total_exact_evals())?;
        d.set_item("total_lanczos_iters", self.total_lanczos_iters())?;
        let steps = pyo3::types::PyList::empty(py);
        for s in &self.steps {
            let b = s.bind(py);
            let sd = pyo3::types::PyDict::new(py);
            sd.set_item("k_before", b.getattr("k_before")?)?;
            sd.set_item("k_after", b.getattr("k_after")?)?;
            sd.set_item("index", b.getattr("index")?)?;
            sd.set_item("lambda_min", b.getattr("lambda_min")?)?;
            sd.set_item("lambda_verified", b.getattr("lambda_verified")?)?;
            sd.set_item("upper_bound", b.getattr("upper_bound")?)?;
            sd.set_item("certified", b.getattr("certified")?)?;
            sd.set_item("exact_evals", b.getattr("exact_evals")?)?;
            steps.append(sd)?;
        }
        d.set_item("steps", steps)?;
        Ok(d)
    }

    fn __len__(&self) -> usize {
        self.k
    }

    fn __repr__(&self) -> String {
        format!(
            "Selection(direction='{}', k={}, lambda_min={:.9}, n={}, m={}, seconds={:.4})",
            self.direction(),
            self.k,
            self.lambda_min,
            self.n,
            self.m_total,
            self.seconds
        )
    }
}

// ---------------------------------------------------------------------------
// Entree principale
// ---------------------------------------------------------------------------

/// Execute une selection E-optimale sur un tampon `float64` 2D.
///
/// En selection avant (`direction="forward"`), le pre-filtre n'est **jamais**
/// active automatiquement : par defaut (`prefilter=false`) *tous* les candidats
/// sont evalues exactement, ce qui realise le vrai glouton E-optimal. Le passer a
/// `true` n'evalue que les `forward_top` (16 par defaut) meilleurs candidats du
/// score seculaire : plus rapide sur les grands `M`, mais `lambda_min` peut
/// perdre quelques pour cent (le glouton devient heuristique).
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    x,
    *,
    shape = None,
    direction = "backward",
    k = None,
    kmin = None,
    kmax = None,
    verify = true,
    center = true,
    low_rank = 4,
    tol = 1e-10,
    iters_warm = 60,
    iters_cold = 400,
    max_exact = 0,
    batch = 8,
    prefilter = false,
    forward_top = 0,
    forward_first = None,
    swap_passes = 0,
    swap_top = 8,
    swap_from = None,
    eval = "auto",
    representation = "auto",
    mem_budget_mb = 4096,
    block_rows = 64,
    threads = None,
    progress = None,
))]
fn run(
    py: Python<'_>,
    x: &Bound<'_, PyAny>,
    shape: Option<(usize, usize)>,
    direction: &str,
    k: Option<usize>,
    kmin: Option<usize>,
    kmax: Option<usize>,
    verify: bool,
    center: bool,
    low_rank: usize,
    tol: f64,
    iters_warm: usize,
    iters_cold: usize,
    max_exact: usize,
    batch: usize,
    prefilter: bool,
    forward_top: usize,
    forward_first: Option<usize>,
    swap_passes: usize,
    swap_top: usize,
    swap_from: Option<usize>,
    eval: &str,
    representation: &str,
    mem_budget_mb: usize,
    block_rows: usize,
    threads: Option<usize>,
    progress: Option<Py<PyAny>>,
) -> PyResult<Selection> {
    let t_all = Instant::now();
    let direction = parse_direction(direction)?;
    let eval_mode = parse_eval(eval)?;
    let want_repr = parse_repr(representation)?;
    if let Some(cb) = progress.as_ref() {
        if !cb.bind(py).is_callable() {
            return Err(PyTypeError::new_err("progress doit etre appelable ou None"));
        }
    }
    if block_rows == 0 {
        return Err(PyValueError::new_err("block_rows doit etre >= 1"));
    }
    if tol <= 0.0 || !tol.is_finite() {
        return Err(PyValueError::new_err("tol doit etre un reel > 0"));
    }

    // ---- donnees (GIL conserve : lecture du tampon Python) ----
    let t_load = Instant::now();
    let mut dm = matrix_from_python(py, x, shape)?;
    let n = dm.rows;
    let m_total = dm.cols;
    let std = dm.standardize(center);
    let m = dm.cols;
    if m == 0 {
        return Err(MinvpError::new_err("aucune colonne exploitable (variance nulle)"));
    }
    let dropped = std.dropped.clone();
    let keep_map: Vec<usize> = (0..m_total).filter(|j| !dropped.contains(j)).collect();

    // ---- bornes du parcours ----
    let (k_min, k_max) = match direction {
        Direction::Backward => {
            let kk = k.unwrap_or_else(|| kmin.unwrap_or(1));
            if kk < 1 || kk > m {
                return Err(PyValueError::new_err(format!(
                    "k/kmin doit etre dans 1..={m} (colonnes utilisables), recu {kk}"
                )));
            }
            (kk, m)
        }
        Direction::Forward => {
            let kk = k.unwrap_or_else(|| kmax.unwrap_or(m));
            if kk < 1 || kk > m {
                return Err(PyValueError::new_err(format!(
                    "k/kmax doit etre dans 1..={m} (colonnes utilisables), recu {kk}"
                )));
            }
            (1, kk)
        }
    };
    if let Some(f) = forward_first {
        if f >= m {
            return Err(PyValueError::new_err(format!(
                "forward_first doit etre dans 0..{m}, recu {f}"
            )));
        }
    }

    // Pre-filtre avant : jamais active automatiquement. `forward_top` (s'il est
    // non nul) fixe la largeur ; sinon `prefilter` choisit entre top-16 et
    // evaluation exhaustive de tous les candidats (glouton exact).
    let forward_top_eff = if forward_top > 0 {
        forward_top
    } else if prefilter {
        PREFILTER_TOP
    } else {
        0
    };

    let alg = AlgoConfig {
        tol,
        max_iters_cold: iters_cold,
        max_iters_warm: iters_warm,
        max_exact,
        batch: batch.max(1),
        num_low: low_rank.max(1),
        eval: eval_mode,
        inverse_max_m: 3000,
    };
    let refine = if swap_passes > 0 {
        Some(RefineCfg { passes: swap_passes, top: swap_top.max(1), from_k: swap_from.unwrap_or(k_min) })
    } else {
        None
    };
    let budget = mem_budget_mb.saturating_mul(1024 * 1024);
    let load_seconds = t_load.elapsed().as_secs_f64();

    // ---- pool de threads dedie (jamais `build_global`, deja initialise) ----
    let pool = match threads {
        Some(0) | None => None,
        Some(t) => Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .map_err(|e| MinvpError::new_err(format!("pool rayon: {e}")))?,
        ),
    };

    let cb_error: Mutex<Option<PyErr>> = Mutex::new(None);

    // ---- calcul sans le GIL ----
    let outcome = py.detach(|| -> PyResult<Outcome> {
        let mut on_step = |rec: &StepRecord| -> bool {
            let res = Python::attach(|py| -> PyResult<bool> {
                // Ctrl-C (et tout autre signal) reste traite pendant le calcul.
                py.check_signals()?;
                let Some(cb) = progress.as_ref() else {
                    return Ok(true);
                };
                let step = Py::new(py, Step::from_record(rec, &keep_map))?;
                let out = cb.bind(py).call1((step,))?;
                // `None` (callback purement informatif) poursuit ; seul un
                // retour explicitement faux interrompt le parcours.
                if out.is_none() {
                    Ok(true)
                } else {
                    out.is_truthy()
                }
            });
            match res {
                Ok(keep_going) => keep_going,
                Err(e) => {
                    if let Ok(mut slot) = cb_error.lock() {
                        *slot = Some(e);
                    }
                    false
                }
            }
        };

        let compute = || -> PyResult<Outcome> {
            let mut ds = Dataset::new(dm, want_repr, budget, block_rows);
            let path = match direction {
                Direction::Backward => greedy::backward(
                    &mut ds,
                    k_min,
                    &alg,
                    refine,
                    verify,
                    &mut on_step,
                ),
                Direction::Forward => greedy::forward(
                    &mut ds,
                    k_max,
                    &alg,
                    forward_top_eff,
                    forward_first,
                    &mut on_step,
                ),
            };

            let final_k = path
                .steps
                .last()
                .map(|s| s.k_after)
                .unwrap_or(path.initial_k)
                .min(ds.m())
                .max(1);
            let subset = path.subset_at(final_k).unwrap_or_default();

            // Revalidation stricte du sous-ensemble retenu : une seule passe de
            // Lanczos a froid, tolerance serree, sans demarrage a chaud.
            let (lambda_min, lambda_residual) = if verify && !subset.is_empty() {
                strict_lambda(&ds, &subset, final_k, &alg, budget, block_rows)
            } else {
                (path.lambda_at(final_k).unwrap_or(f64::NAN), f64::NAN)
            };

            // Methode d'evaluation reellement employee.
            let eval_used = match eval_mode {
                EvalMode::Direct => "direct",
                EvalMode::Inverse => "inverse",
                EvalMode::Auto => {
                    if direction == Direction::Backward
                        && ds.is_packed()
                        && ds.m() <= alg.inverse_max_m
                    {
                        "inverse"
                    } else {
                        "direct"
                    }
                }
            };

            Ok(Outcome {
                packed: ds.is_packed(),
                eval: eval_used,
                path,
                final_k,
                subset,
                lambda_min,
                lambda_residual,
            })
        };

        match &pool {
            Some(p) => p.install(compute),
            None => compute(),
        }
    });
    drop(pool);

    if let Some(e) = cb_error.lock().ok().and_then(|mut s| s.take()) {
        return Err(e);
    }
    let outcome = outcome?;

    // ---- conversion en objets Python ----
    let steps = outcome
        .path
        .steps
        .iter()
        .map(|rec| Py::new(py, Step::from_record(rec, &keep_map)))
        .collect::<PyResult<Vec<_>>>()?;

    Ok(Selection {
        k: outcome.final_k,
        subset: outcome
            .subset
            .iter()
            .map(|&i| map_index(i, &keep_map))
            .collect(),
        lambda_min: outcome.lambda_min,
        lambda_residual: outcome.lambda_residual,
        n,
        m_total,
        dropped,
        seconds: t_all.elapsed().as_secs_f64(),
        path_seconds: outcome.path.seconds,
        load_seconds,
        representation: if outcome.packed { "packed" } else { "implicit" },
        eval: outcome.eval,
        path: outcome.path,
        keep_map,
        steps,
    })
}

// ---------------------------------------------------------------------------
// Aides internes
// ---------------------------------------------------------------------------

struct Outcome {
    path: PathResult,
    final_k: usize,
    subset: Vec<usize>,
    lambda_min: f64,
    lambda_residual: f64,
    packed: bool,
    eval: &'static str,
}

/// `lambda_min` stricte (Lanczos froid, tolerance 1e-13) du sous-ensemble `set`.
fn strict_lambda(
    ds: &Dataset,
    set: &[usize],
    k: usize,
    alg: &AlgoConfig,
    budget: usize,
    block_rows: usize,
) -> (f64, f64) {
    let repr = if ds.is_packed() { Repr::Packed } else { Repr::Implicit };
    let mut check = Dataset::new(ds.z.clone(), repr, budget, block_rows);
    // `ds.z` est dans l'ordre **physique** courant (le glouton a permute les
    // colonnes), alors que `set` donne des indices d'origine : on retrouve donc
    // chaque variable via `ds.active`, pas via l'ordre identite du clone.
    for (pos, &orig) in set.iter().enumerate() {
        if let Some(cur) = ds.active.iter().position(|&x| x == orig) {
            check.swap(pos, cur);
        }
    }
    let strict = AlgoConfig { tol: 1e-13, max_iters_cold: 4000, max_iters_warm: 400, ..*alg };
    let ev = greedy::initial_eigenpair_public(&check, k, &strict);
    (ev.value, ev.residual)
}

#[inline]
fn map_index(i: usize, keep_map: &[usize]) -> usize {
    keep_map.get(i).copied().unwrap_or(i)
}

fn parse_direction(s: &str) -> PyResult<Direction> {
    match s {
        "backward" => Ok(Direction::Backward),
        "forward" => Ok(Direction::Forward),
        other => Err(PyValueError::new_err(format!(
            "direction doit valoir 'backward' ou 'forward', recu '{other}'"
        ))),
    }
}

fn parse_eval(s: &str) -> PyResult<EvalMode> {
    match s {
        "auto" => Ok(EvalMode::Auto),
        "direct" => Ok(EvalMode::Direct),
        "inverse" => Ok(EvalMode::Inverse),
        other => Err(PyValueError::new_err(format!(
            "eval doit valoir 'auto', 'direct' ou 'inverse', recu '{other}'"
        ))),
    }
}

fn parse_repr(s: &str) -> PyResult<Repr> {
    match s {
        "auto" => Ok(Repr::Auto),
        "packed" => Ok(Repr::Packed),
        "implicit" => Ok(Repr::Implicit),
        other => Err(PyValueError::new_err(format!(
            "repr doit valoir 'auto', 'packed' ou 'implicit', recu '{other}'"
        ))),
    }
}

/// Lit `x` comme une matrice `(observations, variables)` de `float64`.
///
/// Accepte tout objet exposant le protocole tampon (`numpy.ndarray`,
/// `memoryview`, `array.array`, ...) ; le paquet Python s'occupe des listes et
/// DataFrames en amont.  Les valeurs non finies sont refusees.
fn matrix_from_python(
    py: Python<'_>,
    x: &Bound<'_, PyAny>,
    shape: Option<(usize, usize)>,
) -> PyResult<DataMatrix> {
    let buf = PyBuffer::<f64>::get(x).map_err(|_| {
        PyTypeError::new_err(
            "X doit exposer un tampon de float64 (numpy.ndarray, memoryview, \
             array.array...) ; utilisez minvp.as_matrix(X) pour convertir listes \
             et DataFrames",
        )
    })?;
    let dims = buf.dimensions();
    let (n, m) = match shape {
        // Tampon 1D + forme explicite (rows-major) : chemin sans numpy.
        Some((n, m)) => {
            if dims != 1 {
                return Err(PyValueError::new_err(
                    "shape ne peut accompagner qu'un tampon 1D",
                ));
            }
            if buf.item_count() != n * m {
                return Err(PyValueError::new_err(format!(
                    "shape {n}x{m} = {} elements, mais le tampon en contient {}",
                    n * m,
                    buf.item_count()
                )));
            }
            (n, m)
        }
        None => {
            if dims != 2 {
                return Err(PyValueError::new_err(format!(
                    "X doit avoir 2 dimensions (observations, variables), recu {dims}"
                )));
            }
            let s = buf.shape();
            (s[0], s[1])
        }
    };
    if n == 0 || m == 0 {
        return Err(PyValueError::new_err(format!("X est vide (forme {n}x{m})")));
    }
    let mut flat = vec![0.0f64; n * m];
    buf.copy_to_slice(py, &mut flat)?;
    if let Some(pos) = flat.iter().position(|v| !v.is_finite()) {
        return Err(MinvpError::new_err(format!(
            "X contient une valeur non finie (nan ou infini) en ligne {}, colonne {}",
            pos / m,
            pos % m
        )));
    }
    Ok(DataMatrix::from_row_major(n, m, &flat))
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

#[pymodule]
fn _minvp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", VERSION)?;
    m.add_function(wrap_pyfunction!(run, m)?)?;
    m.add_class::<Selection>()?;
    m.add_class::<Step>()?;
    m.add("MinvpError", m.py().get_type::<MinvpError>())?;
    Ok(())
}

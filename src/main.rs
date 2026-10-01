//! Point d'entree `minvp`.
#![forbid(unsafe_code)]

use minvp::cli::{self, Config};
use minvp::gen::{generate, GenKind};
use minvp::greedy::{self, AlgoConfig, Direction, PathResult, RefineCfg};
use minvp::io;
use minvp::matrix::DataMatrix;
use minvp::op::{Dataset, Repr};
use minvp::report;
use std::process::ExitCode;
use std::time::Instant;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cfg = match cli::parse(&args) {
        Ok(c) => c,
        Err(e) => {
            if e == "help" {
                print!("{}", cli::usage());
                return ExitCode::SUCCESS;
            }
            eprintln!("erreur: {e}\n");
            eprint!("{}", cli::usage());
            return ExitCode::from(2);
        }
    };
    match run(&cfg) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("erreur: {e}");
            ExitCode::from(1)
        }
    }
}

fn run(cfg: &Config) -> Result<(), String> {
    if let Some(t) = cfg.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build_global()
            .map_err(|e| format!("pool rayon: {e}"))?;
    }
    let t_load = Instant::now();

    // ---- donnees ----
    let (mut dm, m_original) = load_data(cfg)?;

    // ---- normalisation ----
    let t_norm = Instant::now();
    let rep = dm.standardize(cfg.center);
    let n = dm.rows;
    let m = dm.cols;
    if !cfg.quiet {
        eprintln!(
            "donnees: N={n}, M={m} (original {m_original}), normalisation {:.3}s",
            t_norm.elapsed().as_secs_f64()
        );
    }
    if !rep.dropped.is_empty() {
        eprintln!(
            "attention: {} colonne(s) de variance nulle ignoree(s) : {:?}",
            rep.dropped.len(),
            &rep.dropped[..rep.dropped.len().min(10)]
        );
    }
    let keep_map: Vec<usize> = {
        let dropped: std::collections::HashSet<usize> = rep.dropped.iter().copied().collect();
        (0..m_original).filter(|j| !dropped.contains(j)).collect()
    };
    if m == 0 {
        return Err("aucune colonne exploitable".into());
    }

    // ---- representation ----
    let t_repr = Instant::now();
    let budget = cfg.mem_budget_mb.saturating_mul(1024 * 1024);
    let mut ds = Dataset::new(dm, cfg.repr, budget, cfg.block_rows);
    if !cfg.quiet {
        eprintln!(
            "correlation: representation {}, construction {:.3}s",
            if ds.is_packed() { "packed" } else { "implicite" },
            t_repr.elapsed().as_secs_f64()
        );
    }

    let alg = AlgoConfig {
        tol: cfg.tol,
        max_iters_cold: cfg.max_iters_cold,
        max_iters_warm: cfg.max_iters_warm,
        max_exact: cfg.max_exact,
        batch: cfg.batch,
        num_low: cfg.num_low,
        eval: match cfg.eval.as_str() {
            "direct" => greedy::EvalMode::Direct,
            "inverse" => greedy::EvalMode::Inverse,
            _ => greedy::EvalMode::Auto,
        },
        inverse_max_m: 3000,
    };

    // ---- parcours ----
    let k_min = cfg.k_min.clamp(1, m);
    let k_max = if cfg.k_max == 0 { m } else { cfg.k_max.clamp(1, m) };
    let verbose = !cfg.quiet;
    let mut progress = |st: &greedy::StepRecord| -> bool {
        if !verbose {
            return true;
        }
        match st.k_after.cmp(&st.k_before) {
            std::cmp::Ordering::Less => {
                if st.k_after <= 50 || st.k_after % 50 == 0 {
                    eprintln!(
                        "  K={:<6} lambda_min={:.9}  retrait={:<7} evals={:<3} cert={:<5} res={:.1e}",
                        st.k_after,
                        st.lambda,
                        st.changed_orig,
                        st.exact_evals,
                        st.certified,
                        st.residual
                    );
                }
            }
            std::cmp::Ordering::Greater => {
                if st.k_after <= 50 || st.k_after % 50 == 0 {
                    eprintln!(
                        "  K={:<6} lambda_min={:.9}  ajout={:<7} evals={:<3} res={:.1e}",
                        st.k_after, st.lambda, st.changed_orig, st.exact_evals, st.residual
                    );
                }
            }
            std::cmp::Ordering::Equal => {
                eprintln!(
                    "  echange K={} : lambda_min={:.9} (sortant {}), evals={}",
                    st.k_after, st.lambda, st.changed_orig, st.exact_evals
                );
            }
        }
        true
    };

    let res: PathResult = match cfg.direction {
        Direction::Backward => {
            let refine = if cfg.swap_passes > 0 {
                Some(RefineCfg {
                    passes: cfg.swap_passes,
                    top: cfg.swap_top,
                    from_k: cfg.swap_from.unwrap_or(k_min),
                })
            } else {
                None
            };
            greedy::backward(&mut ds, k_min, &alg, refine, cfg.verify, &mut progress)
        }
        Direction::Forward => {
            greedy::forward(&mut ds, k_max, &alg, cfg.forward_top, cfg.forward_first, &mut progress)
        }
    };

    // ---- sorties ----
    // Le sous-ensemble retenu est revalide par un Lanczos froid tres strict : la
    // valeur affichee est alors une valeur de Ritz certifiee a la tolerance pres.
    let strict = AlgoConfig { tol: 1e-13, max_iters_cold: 4000, max_iters_warm: 400, ..alg };
    let mut subsets: Vec<(usize, Vec<usize>)> = Vec::new();
    for &k in &cfg.subset_k {
        match res.subset_at(k) {
            Some(set) => {
                let mapped: Vec<usize> =
                    set.iter().map(|&i| keep_map.get(i).copied().unwrap_or(i)).collect();
                // verification : on replace les variables choisies en tete puis on
                // recalcule lambda_min sans demarrage a chaud.
                let check = head_subset_dataset(&ds, &set, budget, cfg.block_rows);
                let ev = greedy::initial_eigenpair_public(&check, k, &strict);
                if !cfg.quiet {
                    eprintln!(
                        "verification K={k}: lambda_min={:.12} (residu {:.1e}, {} iterations)",
                        ev.value, ev.residual, ev.iters
                    );
                }
                subsets.push((k, mapped));
            }
            None => eprintln!("K={k} hors du parcours (rien a afficher)"),
        }
    }

    if let Some(p) = &cfg.out_csv {
        if p.as_os_str() == "-" {
            print!("{}", report::csv_string(&res));
        } else {
            report::write_csv(p, &res).map_err(|e| format!("ecriture {}: {e}", p.display()))?;
            if !cfg.quiet {
                eprintln!("CSV ecrit : {}", p.display());
            }
        }
    }
    let mut meta: Vec<(String, String)> = vec![
        ("n_observations".into(), n.to_string()),
        ("m_variables".into(), m.to_string()),
        ("center".into(), cfg.center.to_string()),
        ("repr".into(), if ds.is_packed() { "packed".into() } else { "implicit".to_string() }),
        ("tol".into(), format!("{:.1e}", cfg.tol)),
        ("max_exact".into(), cfg.max_exact.to_string()),
        ("load_seconds".into(), format!("{:.4}", t_load.elapsed().as_secs_f64())),
    ];
    if let Some(g) = cfg.gen {
        meta.push(("generator".into(), format!("{g:?}")));
        meta.push(("seed".into(), cfg.seed.to_string()));
        meta.push(("rho".into(), cfg.rho.to_string()));
    }
    if let Some(p) = &cfg.out_json {
        if p.as_os_str() == "-" {
            print!("{}", report::json_string(&res, &meta, &subsets));
        } else {
            report::write_json(p, &res, &meta, &subsets)
                .map_err(|e| format!("ecriture {}: {e}", p.display()))?;
            if !cfg.quiet {
                eprintln!("JSON ecrit : {}", p.display());
            }
        }
    }
    let json_to_stdout = cfg.out_json.as_ref().map(|p| p.as_os_str() == "-").unwrap_or(false);
    if !cfg.stdout_mode && !json_to_stdout {
        report::print_summary(&res, &subsets, !cfg.quiet || !subsets.is_empty());
    }
    Ok(())
}

fn load_data(cfg: &Config) -> Result<(DataMatrix, usize), String> {
    if cfg.stdin_f64 {
        let (r, c) = cfg.shape.ok_or("--stdin-f64 exige --shape N,M")?;
        let dm = io::read_stdin_f64(r, c).map_err(|e| format!("lecture stdin: {e}"))?;
        let m0 = dm.cols;
        return Ok((dm, m0));
    }
    if let Some(path) = &cfg.input {
        let dm = match cfg.binary {
            Some(fmt) => {
                let (r, c) = cfg.shape.ok_or("--binary exige --shape N,M")?;
                io::read_binary(path, fmt, r, c)
            }
            None => io::read_text(path),
        }
        .map_err(|e| format!("lecture {}: {e}", path.display()))?;
        let m0 = dm.cols;
        return Ok((dm, m0));
    }
    let kind = cfg.gen.unwrap_or(GenKind::Equi);
    if cfg.gen_n == 0 || cfg.gen_m == 0 {
        return Err("--gen-n et --gen-m doivent etre > 0".into());
    }
    let dm = generate(
        kind, cfg.gen_n, cfg.gen_m, cfg.rho, cfg.rho_out, cfg.blocks, cfg.rank, cfg.seed,
    );
    let m0 = dm.cols;
    Ok((dm, m0))
}

fn repr_of(ds: &Dataset) -> Repr {
    if ds.is_packed() {
        Repr::Packed
    } else {
        Repr::Implicit
    }
}

/// Force l'emploi de `Repr` (documentation / tests unitaires du binaire).
#[allow(dead_code)]
fn _repr_doc(r: Repr) -> &'static str {
    match r {
        Repr::Auto => "auto",
        Repr::Packed => "packed",
        Repr::Implicit => "implicit",
    }
}

/// Reconstruit un [`Dataset`] ou les variables `set` (indices d'origine) occupent
/// le bloc de tete, en conservant la permutation courante de `ds`.
///
/// Indispensable : apres un parcours, `ds.z` est permute, alors que
/// `Dataset::new` remet `active` a `0..m`. Sans cette reprise, la correspondance
/// position -> indice d'origine serait rompue et la revalidation porterait sur un
/// autre sous-ensemble.
fn head_subset_dataset(
    ds: &Dataset,
    set: &[usize],
    budget: usize,
    block_rows: usize,
) -> Dataset {
    let mut check = Dataset::new(ds.z.clone(), repr_of(ds), budget, block_rows);
    check.active.copy_from_slice(&ds.active);
    for (pos, &orig) in set.iter().enumerate() {
        if let Some(cur) = check.active.iter().position(|&x| x == orig) {
            check.swap(pos, cur);
        }
    }
    check
}

#[cfg(test)]
mod tests {
    use super::*;
    use minvp::gen::{generate, GenKind};
    use minvp::jacobi::eigen_sym_sorted;

    /// La revalidation doit porter sur le sous-ensemble demande, meme apres un
    /// parcours qui a permute `z` (regression : `active` etait remis a `0..m`).
    #[test]
    fn head_subset_dataset_validates_the_right_subset() {
        let mut dm = generate(GenKind::Blocks, 200, 25, 0.5, 0.1, 3, 1, 7);
        dm.standardize(true);
        let mut ds = Dataset::new(dm, Repr::Packed, usize::MAX, 8);
        // permutation quelconque, comme le ferait un parcours
        for (a, b) in [(0usize, 17usize), (4, 11), (2, 20), (1, 9)] {
            ds.swap(a, b);
        }
        let set = vec![3usize, 5, 9, 14];
        let check = head_subset_dataset(&ds, &set, usize::MAX, 8);

        let mut head = check.active[..set.len()].to_vec();
        head.sort_unstable();
        assert_eq!(head, set, "le bloc de tete doit etre exactement le sous-ensemble");

        let strict = AlgoConfig {
            tol: 1e-13,
            max_iters_cold: 4000,
            max_iters_warm: 400,
            ..Default::default()
        };
        let ev = greedy::initial_eigenpair_public(&check, set.len(), &strict);

        // reference independante : sous-matrice dense de la correlation, Jacobi.
        // `packed` est indexee par position physique, `set` par indice d'origine.
        let d = set.len();
        let packed = ds.packed.as_ref().expect("packed");
        let pos_of = |orig: usize| ds.active.iter().position(|&x| x == orig).expect("present");
        let mut a = vec![0.0f64; d * d];
        for (x, &ix) in set.iter().enumerate() {
            for (y, &iy) in set.iter().enumerate() {
                a[x * d + y] = packed.get(pos_of(ix), pos_of(iy));
            }
        }
        let (vals, _) = eigen_sym_sorted(&mut a, d);
        assert!(
            (ev.value - vals[0]).abs() < 1e-9,
            "lambda_min verifie {} vs dense {}",
            ev.value,
            vals[0]
        );
    }
}

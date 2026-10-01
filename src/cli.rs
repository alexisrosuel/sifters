//! Interface en ligne de commande (analyse manuelle, sans dependance).

use crate::gen::GenKind;
use crate::greedy::Direction;
use crate::io::BinaryFormat;
use crate::op::Repr;
use std::path::PathBuf;

/// Configuration complete d'une execution.
#[derive(Clone, Debug)]
pub struct Config {
    /// Fichier de donnees.
    pub input: Option<PathBuf>,
    /// Lecture de la matrice `f64` LE depuis stdin (binding Python).
    pub stdin_f64: bool,
    /// Format binaire eventuel.
    pub binary: Option<BinaryFormat>,
    /// Dimensions pour l'entree binaire.
    pub shape: Option<(usize, usize)>,
    /// Generateur synthetique.
    pub gen: Option<GenKind>,
    /// Observations du generateur.
    pub gen_n: usize,
    /// Variables du generateur.
    pub gen_m: usize,
    /// Correlation cible du generateur.
    pub rho: f64,
    /// Correlation inter-blocs.
    pub rho_out: f64,
    /// Nombre de blocs.
    pub blocks: usize,
    /// Rang du modele a facteurs.
    pub rank: usize,
    /// Graine.
    pub seed: u64,
    /// Centrage des colonnes.
    pub center: bool,
    /// Representation de la correlation.
    pub repr: Repr,
    /// Budget memoire pour materialiser la correlation (Mo).
    pub mem_budget_mb: usize,
    /// Taille des blocs de lignes pour la construction.
    pub block_rows: usize,
    /// Sens du parcours.
    pub direction: Direction,
    /// Taille minimale.
    pub k_min: usize,
    /// Taille maximale.
    pub k_max: usize,
    /// Tolerance Lanczos.
    pub tol: f64,
    /// Iterations max a froid.
    pub max_iters_cold: usize,
    /// Iterations max a chaud.
    pub max_iters_warm: usize,
    /// Candidats evalues exactement au maximum par etape (0 = illimite).
    pub max_exact: usize,
    /// Taille des lots paralleles.
    pub batch: usize,
    /// Nombre de couples propres pour la borne de Temple.
    pub num_low: usize,
    /// Methode d'evaluation : auto | direct | inverse.
    pub eval: String,
    /// Largeur du pre-filtre avant ; `0` = pas de pre-filtre (etape certifiee
    /// optimale, sans evaluation exhaustive). Aucune activation automatique selon
    /// la taille.
    pub forward_top: usize,
    /// Premiere variable imposee (selection avant).
    pub forward_first: Option<usize>,
    /// Passes d'echanges locaux.
    pub swap_passes: usize,
    /// Nombre de sortants/entrants consideres par passe d'echange.
    pub swap_top: usize,
    /// N'appliquer les echanges qu'aux tailles <= cette valeur.
    pub swap_from: Option<usize>,
    /// Sortie CSV.
    pub out_csv: Option<PathBuf>,
    /// Sortie JSON.
    pub out_json: Option<PathBuf>,
    /// Sous-ensembles a afficher.
    pub subset_k: Vec<usize>,
    /// Nombre de threads rayon.
    pub threads: Option<usize>,
    /// Revalide la courbe par un Lanczos froid strict a chaque etape.
    pub verify: bool,
    /// Mode silencieux.
    pub quiet: bool,
    /// Sortie JSON sur stdout (pas de resume console).
    pub stdout_mode: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            input: None,
            stdin_f64: false,
            binary: None,
            shape: None,
            gen: None,
            gen_n: 500,
            gen_m: 2000,
            rho: 0.3,
            rho_out: 0.0,
            blocks: 8,
            rank: 4,
            seed: 1,
            center: true,
            repr: Repr::Auto,
            mem_budget_mb: 4096,
            block_rows: 64,
            direction: Direction::Backward,
            k_min: 1,
            k_max: 0,
            tol: 1e-10,
            max_iters_cold: 400,
            max_iters_warm: 60,
            max_exact: 0,
            batch: 8,
            num_low: 4,
            eval: "auto".to_string(),
            forward_top: 0,
            forward_first: None,
            swap_passes: 0,
            swap_top: 8,
            swap_from: None,
            out_csv: None,
            out_json: None,
            subset_k: Vec::new(),
            verify: true,
            stdout_mode: false,
            threads: None,
            quiet: false,
        }
    }
}

/// Texte d'aide.
pub fn usage() -> String {
    r#"minvp — maximisation de la plus petite valeur propre d'une matrice de correlation

Selectionne, pour chaque taille K, un sous-ensemble de K variables maximisant
lambda_min(R_S) (critere E-optimal), par glouton certifie + Lanczos multicore.

USAGE:
  minvp [OPTIONS]

DONNEES
  --input <fichier>        matrice texte (lignes = observations, sep. blanc/virgule)
  --stdin-f64              lit la matrice (f64 LE, lignes-major) sur stdin
  --binary f32|f64         entree binaire brute (lignes-major), exige --shape
  --shape N,M              dimensions pour l'entree binaire
  --gen iid|equi|ar|blocks|factor     jeu synthetique
  --gen-n N                observations du jeu synthetique (def. 500)
  --gen-m M                variables du jeu synthetique (def. 2000)
  --rho F                  correlation cible (def. 0.3)
  --rho-out F              correlation inter-blocs (def. 0)
  --blocks B               nombre de blocs (def. 8)
  --rank R                 rang du modele a facteurs (def. 4)
  --seed U                 graine (def. 1)
  --no-center              ne pas centrer (cosinus au lieu de correlation)

ALGORITHME
  --dir backward|forward   sens du parcours (def. backward)
  --kmin K                 taille finale (def. 1)
  --kmax K                 taille finale en selection avant (def. M)
  --tol F                  tolerance Lanczos (def. 1e-10)
  --iters-cold N           iterations max a froid (def. 400)
  --iters-warm N           iterations max a chaud (def. 60)
  --max-exact N            candidats evalues exactement par etape (0 = illimite)
  --batch N                taille des lots paralleles (def. 8)
  --low-rank P             couples propres p utilises par la borne de Temple (def. 4).
                           Sans effet en selection avant certifiee (--forward-top 0),
                           qui utilise le spectre complet : la borne y est alors exacte.
  --eval auto|direct|inverse   evaluation des candidats par l'inverse maintenu (def. auto)
  --forward-top N          pre-filtre avant : n'evalue que les N meilleurs candidats
                           (0 = desactive -> chaque etape est certifiee optimale
                           sans evaluer tous les candidats ; def. 0)
  --prefilter              active le pre-filtre avant (largeur 16 ; --forward-top N
                           pour une autre largeur)
  --forward-first J        premiere variable imposee (selection avant)
  --swap-passes N          passes d'echanges locaux 1-1 (def. 0)
  --swap-top N             sortants/entrants consideres par passe (def. 8)
  --swap-from K            n'appliquer les echanges qu'aux tailles <= K (def. kmin)

PERFORMANCE
  --repr auto|packed|implicit   representation de la correlation (def. auto)
  --mem-budget MB          budget pour materialiser R (def. 4096)
  --block-rows N           bloc de lignes de la construction (def. 64)
  --threads N              threads rayon (def. tous)

SORTIES
  --no-verify              ne pas revalider la courbe (Lanczos froid strict)
  --out-csv <fichier>      courbe complete (K, lambda_min, variable, ...)
  --out-json <fichier>     resume + courbe
  --subset K[,K...]        affiche les variables selectionnees pour ces K
  --quiet                  n'affiche que les resultats demandes
  -h, --help               cette aide
"#
    .to_string()
}

fn parse_usize(v: &str, what: &str) -> Result<usize, String> {
    v.parse::<usize>().map_err(|_| format!("{what}: entier attendu, recu '{v}'"))
}

fn parse_f64(v: &str, what: &str) -> Result<f64, String> {
    v.parse::<f64>().map_err(|_| format!("{what}: reel attendu, recu '{v}'"))
}

/// Analyse les arguments (sans le nom du programme).
pub fn parse(args: &[String]) -> Result<Config, String> {
    let mut c = Config::default();
    let mut i = 0usize;
    while i < args.len() {
        let a = args[i].as_str();
        let mut next = |what: &str| -> Result<String, String> {
            i += 1;
            if i >= args.len() {
                return Err(format!("{what}: argument manquant"));
            }
            Ok(args[i].clone())
        };
        match a {
            "-h" | "--help" => return Err("help".to_string()),
            "--input" => c.input = Some(PathBuf::from(next("--input")?)),
            "--stdin-f64" => c.stdin_f64 = true,
            "--binary" => {
                let v = next("--binary")?;
                c.binary = Some(match v.as_str() {
                    "f64" | "double" => BinaryFormat::F64,
                    "f32" | "float" => BinaryFormat::F32,
                    _ => return Err(format!("--binary: format inconnu '{v}'")),
                });
            }
            "--shape" => {
                let v = next("--shape")?;
                let mut it = v.split(['x', 'X', ',']);
                let r = it.next().ok_or("--shape: format N,M")?;
                let col = it.next().ok_or("--shape: format N,M")?;
                c.shape = Some((parse_usize(r, "--shape N")?, parse_usize(col, "--shape M")?));
            }
            "--gen" => {
                let v = next("--gen")?;
                c.gen = Some(match v.as_str() {
                    "iid" | "random" => GenKind::Iid,
                    "equi" | "equicorrelation" => GenKind::Equi,
                    "ar" | "ar1" | "toeplitz" => GenKind::Ar,
                    "blocks" | "block" => GenKind::Blocks,
                    "factor" | "facteurs" => GenKind::Factor,
                    _ => return Err(format!("--gen: type inconnu '{v}'")),
                });
            }
            "--gen-n" => c.gen_n = parse_usize(&next(a)?, "--gen-n")?,
            "--gen-m" => c.gen_m = parse_usize(&next(a)?, "--gen-m")?,
            "--rho" => c.rho = parse_f64(&next(a)?, "--rho")?,
            "--rho-out" => c.rho_out = parse_f64(&next(a)?, "--rho-out")?,
            "--blocks" => c.blocks = parse_usize(&next(a)?, "--blocks")?,
            "--rank" => c.rank = parse_usize(&next(a)?, "--rank")?,
            "--seed" => c.seed = parse_usize(&next(a)?, "--seed")? as u64,
            "--no-center" => c.center = false,
            "--dir" => {
                let v = next("--dir")?;
                c.direction = match v.as_str() {
                    "backward" | "back" => Direction::Backward,
                    "forward" => Direction::Forward,
                    _ => return Err(format!("--dir: '{v}' (backward|forward)")),
                };
            }
            "--kmin" => c.k_min = parse_usize(&next(a)?, "--kmin")?,
            "--kmax" => c.k_max = parse_usize(&next(a)?, "--kmax")?,
            "--tol" => c.tol = parse_f64(&next(a)?, "--tol")?,
            "--iters-cold" => c.max_iters_cold = parse_usize(&next(a)?, "--iters-cold")?,
            "--iters-warm" => c.max_iters_warm = parse_usize(&next(a)?, "--iters-warm")?,
            "--max-exact" => c.max_exact = parse_usize(&next(a)?, "--max-exact")?,
            "--batch" => c.batch = parse_usize(&next(a)?, "--batch")?,
            "--low-rank" => c.num_low = parse_usize(&next(a)?, "--low-rank")?,
            "--forward-top" => c.forward_top = parse_usize(&next(a)?, "--forward-top")?,
            "--prefilter" => {
                if c.forward_top == 0 {
                    c.forward_top = 16;
                }
            }
            "--eval" => {
                let v = next("--eval")?;
                match v.as_str() {
                    "auto" | "direct" | "inverse" => c.eval = v,
                    _ => return Err(format!("--eval: '{v}' (auto|direct|inverse)")),
                }
            }
            "--forward-first" => c.forward_first = Some(parse_usize(&next(a)?, "--forward-first")?),
            "--swap-passes" => c.swap_passes = parse_usize(&next(a)?, "--swap-passes")?,
            "--swap-top" => c.swap_top = parse_usize(&next(a)?, "--swap-top")?,
            "--swap-from" => c.swap_from = Some(parse_usize(&next(a)?, "--swap-from")?),
            "--repr" => {
                let v = next("--repr")?;
                c.repr = match v.as_str() {
                    "auto" => Repr::Auto,
                    "packed" => Repr::Packed,
                    "implicit" => Repr::Implicit,
                    _ => return Err(format!("--repr: '{v}' (auto|packed|implicit)")),
                };
            }
            "--mem-budget" => c.mem_budget_mb = parse_usize(&next(a)?, "--mem-budget")?,
            "--block-rows" => c.block_rows = parse_usize(&next(a)?, "--block-rows")?,
            "--threads" => c.threads = Some(parse_usize(&next(a)?, "--threads")?),
            "--out-csv" => c.out_csv = Some(PathBuf::from(next("--out-csv")?)),
            "--out-json" => c.out_json = Some(PathBuf::from(next("--out-json")?)),
            "--subset" => {
                let v = next("--subset")?;
                for tok in v.split(',') {
                    c.subset_k.push(parse_usize(tok.trim(), "--subset")?);
                }
            }
            "--stdout" => c.stdout_mode = true,
            "--no-verify" => c.verify = false,
            "--quiet" => c.quiet = true,
            other => return Err(format!("option inconnue '{other}'")),
        }
        i += 1;
    }
    Ok(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_basic() {
        let args: Vec<String> = "--gen ar --rho 0.5 --gen-m 100 --kmin 5 --subset 5,10"
            .split(' ')
            .map(|s| s.to_string())
            .collect();
        let c = parse(&args).unwrap();
        assert_eq!(c.gen, Some(GenKind::Ar));
        assert!((c.rho - 0.5).abs() < 1e-12);
        assert_eq!(c.gen_m, 100);
        assert_eq!(c.k_min, 5);
        assert_eq!(c.subset_k, vec![5, 10]);
    }
}

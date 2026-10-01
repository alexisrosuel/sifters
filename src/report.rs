//! Rapports : sortie console, CSV et JSON (ecriture manuelle, sans serde).

use crate::greedy::{Direction, PathResult};
use std::fs;
use std::io;
use std::path::Path;

/// Ecrit la courbe complete au format CSV.
///
/// Colonnes : `k,lambda_min,variable_changee,borne_sup,exact_evals,candidats,certifie,residu,iterations`
pub fn write_csv(path: &Path, res: &PathResult) -> io::Result<()> {
    fs::write(path, csv_string(res))
}

/// Contenu CSV (sans ecriture).
pub fn csv_string(res: &PathResult) -> String {
    let mut s = String::with_capacity(64 * (res.steps.len() + 2));
    s.push_str("k,lambda_min_ritz,lambda_min_verifie,variable_changee,borne_sup,exact_evals,candidats,certifie,residu,iterations\n");
    s.push_str(&format!(
        "{},{:.17e},,,",
        res.initial_k, res.initial_lambda
    ));
    s.push_str(&format!(",,{:.3e},\n", res.initial_residual));
    for st in &res.steps {
        s.push_str(&format!(
            "{},{:.17e},{:.17e},{},{:.17e},{},{},{},{:.3e},{}\n",
            st.k_after,
            st.lambda,
            st.lambda_verified,
            st.changed_orig,
            st.upper,
            st.exact_evals,
            st.candidates,
            if st.certified { 1 } else { 0 },
            st.residual,
            st.iters
        ));
    }
    s
}

/// Ecrit un resume JSON.
pub fn write_json(
    path: &Path,
    res: &PathResult,
    meta: &[(String, String)],
    subsets: &[(usize, Vec<usize>)],
) -> io::Result<()> {
    fs::write(path, json_string(res, meta, subsets))
}

/// Contenu JSON (sans ecriture).
pub fn json_string(res: &PathResult, meta: &[(String, String)], subsets: &[(usize, Vec<usize>)]) -> String {
    let mut s = String::new();
    s.push_str("{\n");
    s.push_str("  \"meta\": {");
    for (i, (k, v)) in meta.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!("\n    {}: {}", json_str(k), json_val(v)));
    }
    if !meta.is_empty() {
        s.push('\n');
        s.push_str("  ");
    }
    s.push_str("},\n");
    let dir = match res.direction {
        Direction::Backward => "backward",
        Direction::Forward => "forward",
    };
    s.push_str(&format!("  \"direction\": \"{dir}\",\n"));
    s.push_str(&format!("  \"initial_k\": {},\n", res.initial_k));
    s.push_str(&format!("  \"initial_lambda\": {:.17e},\n", res.initial_lambda));
    s.push_str(&format!("  \"seconds\": {:.6},\n", res.seconds));
    s.push_str(&format!("  \"total_exact_evals\": {},\n", res.total_exact()));
    s.push_str(&format!("  \"total_lanczos_iters\": {},\n", res.total_iters()));
    s.push_str(&format!("  \"certified_steps\": {},\n", res.certified_steps()));
    let order: Vec<String> = res.order().iter().map(|v| v.to_string()).collect();
    s.push_str(&format!("  \"order\": [{}],\n", order.join(", ")));
    let init: Vec<String> = res.initial_subset.iter().map(|v| v.to_string()).collect();
    s.push_str(&format!("  \"initial_subset\": [{}],\n", init.join(", ")));
    s.push_str("  \"curve\": [");
    for (i, (k, l)) in res.curve().iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!("\n    [{k}, {l:.17e}]"));
    }
    s.push_str("\n  ],\n");
    s.push_str("  \"subsets\": {");
    for (i, (k, set)) in subsets.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        let list: Vec<String> = set.iter().map(|v| v.to_string()).collect();
        s.push_str(&format!("\n    \"{k}\": [{}]", list.join(", ")));
    }
    if !subsets.is_empty() {
        s.push('\n');
        s.push_str("  ");
    }
    s.push_str("}\n");
    s.push_str("}\n");
    s
}

fn json_str(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            _ => o.push(c),
        }
    }
    o.push('"');
    o
}

fn json_val(v: &str) -> String {
    if v.parse::<f64>().is_ok() && v != "inf" && v != "nan" {
        v.to_string()
    } else {
        json_str(v)
    }
}

/// Resume console.
pub fn print_summary(res: &PathResult, subset_rows: &[(usize, Vec<usize>)], verbose: bool) {
    let curve = res.curve();
    println!();
    println!("=== Parcours glouton ({:?}) ===", res.direction);
    println!("  duree              : {:.3} s", res.seconds);
    println!("  candidats exacts   : {}", res.total_exact());
    println!("  iterations Lanczos : {}", res.total_iters());
    println!(
        "  etapes certifiees  : {}/{}",
        res.certified_steps(),
        res.steps.len()
    );
    if verbose && curve.len() <= 60 {
        println!("  K : lambda_min");
        for (k, l) in &curve {
            println!("   {k:>5} : {l:.12}");
        }
    } else if verbose {
        println!("  echantillon de la courbe (K : lambda_min) :");
        let n = curve.len();
        for idx in [0, n / 4, n / 2, (3 * n) / 4, n - 1] {
            let (k, l) = curve[idx];
            println!("   {k:>5} : {l:.12}");
        }
    }
    for (k, set) in subset_rows {
        let head = if set.len() <= 24 {
            set.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ")
        } else {
            let mut v: Vec<String> = set.iter().take(24).map(|x| x.to_string()).collect();
            v.push(format!("... (+{})", set.len() - 24));
            v.join(", ")
        };
        let lam = res.lambda_at(*k).map(|l| format!("{l:.12}")).unwrap_or_else(|| "n/a".into());
        println!("  K={k} : lambda_min={lam} | variables = [{head}]");
    }
}

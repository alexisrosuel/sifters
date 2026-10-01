//! Entrees/sorties : lecture de matrices texte (blancs/CSV) et binaires brutes.

use crate::matrix::DataMatrix;
use std::fs;
use std::io;
use std::path::Path;

/// Format binaire d'entree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryFormat {
    /// `f64` little-endian, ordre des lignes (C order).
    F64,
    /// `f32` little-endian, ordre des lignes (C order).
    F32,
}

/// Lit une matrice texte : une ligne = une observation, colonnes separees par
/// espaces, tabulations, `;` ou `,`. Une eventuelle ligne d'en-tete non numerique
/// est ignoree.
pub fn read_text(path: &Path) -> io::Result<DataMatrix> {
    let content = fs::read_to_string(path)?;
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for (lineno, line) in content.lines().enumerate() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        let mut vals = Vec::new();
        let mut ok = true;
        for tok in t.split(|c: char| c == ',' || c == ';' || c.is_whitespace()).filter(|s| !s.is_empty()) {
            match tok.parse::<f64>() {
                Ok(v) => vals.push(v),
                Err(_) => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            if rows.is_empty() && lineno < 4 {
                continue; // en-tete
            }
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("valeur non numerique ligne {}", lineno + 1),
            ));
        }
        if !vals.is_empty() {
            if let Some(first) = rows.first() {
                if first.len() != vals.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("ligne {} : {} colonnes au lieu de {}", lineno + 1, vals.len(), first.len()),
                    ));
                }
            }
            rows.push(vals);
        }
    }
    if rows.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "fichier vide"));
    }
    let n = rows.len();
    let m = rows[0].len();
    let mut data = vec![0.0; n * m];
    for (i, r) in rows.iter().enumerate() {
        for (j, &v) in r.iter().enumerate() {
            data[j * n + i] = v;
        }
    }
    Ok(DataMatrix::from_col_major(n, m, data))
}

/// Lit une matrice `f64` little-endian lignes-major depuis l'entree standard
/// (utilise par le binding Python : transfert binaire direct, sans fichier temporaire).
pub fn read_stdin_f64(rows: usize, cols: usize) -> io::Result<DataMatrix> {
    use std::io::Read;
    let mut bytes = Vec::with_capacity(rows * cols * 8);
    std::io::stdin().read_to_end(&mut bytes)?;
    if bytes.len() != rows * cols * 8 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("stdin: {} octets recus, {} attendus", bytes.len(), rows * cols * 8),
        ));
    }
    let mut data = vec![0.0f64; rows * cols];
    for (k, v) in data.iter_mut().enumerate() {
        let mut b = [0u8; 8];
        b.copy_from_slice(&bytes[k * 8..k * 8 + 8]);
        *v = f64::from_le_bytes(b);
    }
    Ok(DataMatrix::from_row_major(rows, cols, &data))
}

/// Lit une matrice binaire brute (lignes-major) de dimensions `rows x cols`.
pub fn read_binary(path: &Path, fmt: BinaryFormat, rows: usize, cols: usize) -> io::Result<DataMatrix> {
    let bytes = fs::read(path)?;
    let elem = match fmt {
        BinaryFormat::F64 => 8,
        BinaryFormat::F32 => 4,
    };
    let expect = rows * cols * elem;
    if bytes.len() != expect {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "taille binaire {} octets, attendu {} ({}x{}x{})",
                bytes.len(),
                expect,
                rows,
                cols,
                elem
            ),
        ));
    }
    let mut buf = vec![0.0f64; rows * cols];
    match fmt {
        BinaryFormat::F64 => {
            for (k, v) in buf.iter_mut().enumerate() {
                let mut b = [0u8; 8];
                b.copy_from_slice(&bytes[k * 8..k * 8 + 8]);
                *v = f64::from_le_bytes(b);
            }
        }
        BinaryFormat::F32 => {
            for (k, v) in buf.iter_mut().enumerate() {
                let mut b = [0u8; 4];
                b.copy_from_slice(&bytes[k * 4..k * 4 + 4]);
                *v = f32::from_le_bytes(b) as f64;
            }
        }
    }
    Ok(DataMatrix::from_row_major(rows, cols, &buf))
}

//! Script de build de l'extension : sur macOS (et WebAssembly), l'editeur de
//! liens doit laisser les symboles de l'API Python non resolus
//! (`-undefined dynamic_lookup`) pour qu'un module d'extension se charge dans
//! l'interpreteur qui l'importe.
//!
//! Sur Linux et Windows cet appel ne fait rien.

fn main() {
    pyo3_build_config::add_extension_module_link_args();
}

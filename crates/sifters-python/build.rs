//! Extension build script: on macOS (and WebAssembly), the linker
//! must leave the Python API symbols unresolved
//! (`-undefined dynamic_lookup`) so that an extension module loads in
//! the interpreter that imports it.
//!
//! On Linux and Windows this call does nothing.

fn main() {
    pyo3_build_config::add_extension_module_link_args();
}

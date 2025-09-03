// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

fn main() {
    pyo3_build_config::add_extension_module_link_args();
    let config = pyo3_build_config::get();
    if let Some(dir) = &config.lib_dir {
        println!("cargo:rustc-link-search=native={}", dir);
    }
    if let Some(lib) = &config.lib_name {
        println!("cargo:rustc-link-lib={}", lib);
    }
}

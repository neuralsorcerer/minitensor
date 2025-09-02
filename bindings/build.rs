// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

fn main() {
    pyo3_build_config::add_extension_module_link_args();
    // Ensure linking against the Python shared library for tests
    println!("cargo:rustc-link-lib=python3.12");
    println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
}

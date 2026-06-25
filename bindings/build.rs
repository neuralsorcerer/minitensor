// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

fn main() {
    // Add platform-specific linker arguments only when building the Python
    // extension module. Regular Rust builds/tests must link libpython normally,
    // while extension wheels must not link libpython for manylinux compliance.
    #[cfg(feature = "extension-module")]
    pyo3_build_config::add_extension_module_link_args();
}

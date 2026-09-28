// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Record what a plugin and its host have to agree on.
//!
//! A plugin hands the host Rust values -- a `Box<dyn Plugin>`, `Arc<dyn
//! CustomOp>`s -- whose layout and vtables only match when both were built by
//! the same compiler, for the same target, against the same engine.
//! `plugins::PLUGIN_ABI` carries those three, each side's copy compiled into it,
//! so the loader can compare them before it touches anything the library made.

use std::process::Command;

fn main() {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    // `-vV` rather than `-V`: its commit hash tells apart two nightlies that
    // share a version number.
    let compiler = Command::new(&rustc)
        .arg("-vV")
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|text| {
            text.lines()
                .filter(|line| line.starts_with("rustc ") || line.starts_with("commit-hash:"))
                .collect::<Vec<_>>()
                .join(", ")
        })
        .filter(|described| !described.is_empty())
        .unwrap_or_else(|| "unknown rustc".to_string());
    let target = std::env::var("TARGET").unwrap_or_default();
    let version = std::env::var("CARGO_PKG_VERSION").unwrap_or_default();

    println!("cargo:rustc-env=MINITENSOR_PLUGIN_ABI=engine {version}; {compiler}; {target}");
    println!("cargo:rerun-if-env-changed=RUSTC");
    println!("cargo:rerun-if-changed=build.rs");
}

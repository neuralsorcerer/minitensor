// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

use engine::{
    device::Device,
    hardware::{cpu::CpuInfo, gpu::GpuDevice},
};

#[test]
fn test_device_enumeration() {
    let cpu = Device::cpu();
    assert_eq!(cpu.to_string(), "cpu");
    // GPU detection should not panic even if no GPU is present
    let _gpus = GpuDevice::detect_all();
}

#[test]
fn test_cpu_feature_detection() {
    let info = CpuInfo::detect();
    assert!(info.cores > 0);
}

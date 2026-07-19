// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use engine::device::Device;

#[test]
fn test_device_enumeration() {
    let cpu = Device::cpu();
    assert_eq!(cpu.to_string(), "cpu");
    // GPU detection should not panic even if no GPU is present
    #[cfg(feature = "hardware")]
    {
        let _gpus = engine::hardware::gpu::GpuDevice::detect_all();
    }
}

#[cfg(feature = "hardware")]
#[test]
fn test_cpu_feature_detection() {
    let info = engine::hardware::cpu::CpuInfo::detect();
    assert!(info.cores > 0);
}

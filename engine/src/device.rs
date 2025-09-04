// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Device types supported by minitensor
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceType {
    /// CPU device
    Cpu,
    /// CUDA GPU device
    Cuda,
    /// Metal GPU device (Apple Silicon)
    Metal,
    /// OpenCL device
    OpenCL,
}

/// Device specification with optional device ID
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Device {
    device_type: DeviceType,
    device_id: Option<usize>,
}

impl Device {
    /// Create a CPU device
    pub fn cpu() -> Self {
        Self {
            device_type: DeviceType::Cpu,
            device_id: None,
        }
    }

    /// Create a CUDA device with optional device ID
    pub fn cuda(device_id: Option<usize>) -> Self {
        Self {
            device_type: DeviceType::Cuda,
            device_id,
        }
    }

    /// Create a Metal device
    pub fn metal() -> Self {
        Self {
            device_type: DeviceType::Metal,
            device_id: None,
        }
    }

    /// Create an OpenCL device with optional device ID
    pub fn opencl(device_id: Option<usize>) -> Self {
        Self {
            device_type: DeviceType::OpenCL,
            device_id,
        }
    }

    /// Get the device type
    pub fn device_type(&self) -> DeviceType {
        self.device_type
    }

    /// Get the device ID
    pub fn device_id(&self) -> Option<usize> {
        self.device_id
    }

    /// Get the device ID (returns 0 for devices without explicit ID)
    pub fn id(&self) -> usize {
        self.device_id.unwrap_or(0)
    }

    /// Check if this is a CPU device
    pub fn is_cpu(&self) -> bool {
        matches!(self.device_type, DeviceType::Cpu)
    }

    /// Check if this is a GPU device
    pub fn is_gpu(&self) -> bool {
        matches!(
            self.device_type,
            DeviceType::Cuda | DeviceType::Metal | DeviceType::OpenCL
        )
    }

    /// Check if this is a CUDA device
    pub fn is_cuda(&self) -> bool {
        matches!(self.device_type, DeviceType::Cuda)
    }

    /// Check if this is a Metal device
    pub fn is_metal(&self) -> bool {
        matches!(self.device_type, DeviceType::Metal)
    }

    /// Check if this is an OpenCL device
    pub fn is_opencl(&self) -> bool {
        matches!(self.device_type, DeviceType::OpenCL)
    }

    /// Get device name as string
    pub fn name(&self) -> String {
        match (self.device_type, self.device_id) {
            (DeviceType::Cpu, _) => "cpu".to_string(),
            (DeviceType::Cuda, Some(id)) => format!("cuda:{}", id),
            (DeviceType::Cuda, None) => "cuda:0".to_string(),
            (DeviceType::Metal, _) => "metal".to_string(),
            (DeviceType::OpenCL, Some(id)) => format!("opencl:{}", id),
            (DeviceType::OpenCL, None) => "opencl:0".to_string(),
        }
    }

    /// Parse device from string
    pub fn from_str(device_str: &str) -> Result<Self, String> {
        match device_str.to_lowercase().as_str() {
            "cpu" => Ok(Self::cpu()),
            "cuda" => Ok(Self::cuda(Some(0))),
            "metal" => Ok(Self::metal()),
            "opencl" => Ok(Self::opencl(Some(0))),
            s if s.starts_with("cuda:") => {
                let id_str = &s[5..];
                let id = id_str
                    .parse::<usize>()
                    .map_err(|_| format!("Invalid CUDA device ID: {}", id_str))?;
                Ok(Self::cuda(Some(id)))
            }
            s if s.starts_with("opencl:") => {
                let id_str = &s[7..];
                let id = id_str
                    .parse::<usize>()
                    .map_err(|_| format!("Invalid OpenCL device ID: {}", id_str))?;
                Ok(Self::opencl(Some(id)))
            }
            _ => Err(format!("Unknown device: {}", device_str)),
        }
    }
}

impl Default for Device {
    fn default() -> Self {
        Self::cpu()
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl fmt::Display for DeviceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceType::Cpu => write!(f, "cpu"),
            DeviceType::Cuda => write!(f, "cuda"),
            DeviceType::Metal => write!(f, "metal"),
            DeviceType::OpenCL => write!(f, "opencl"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_creation() {
        let cpu = Device::cpu();
        assert!(cpu.is_cpu());
        assert!(!cpu.is_gpu());
        assert_eq!(cpu.name(), "cpu");

        let cuda = Device::cuda(Some(1));
        assert!(cuda.is_cuda());
        assert!(cuda.is_gpu());
        assert_eq!(cuda.name(), "cuda:1");
        assert_eq!(cuda.device_id(), Some(1));

        let metal = Device::metal();
        assert!(metal.is_metal());
        assert!(metal.is_gpu());
        assert_eq!(metal.name(), "metal");
    }

    #[test]
    fn test_device_parsing() {
        assert_eq!(Device::from_str("cpu").unwrap(), Device::cpu());
        assert_eq!(Device::from_str("cuda").unwrap(), Device::cuda(Some(0)));
        assert_eq!(Device::from_str("cuda:2").unwrap(), Device::cuda(Some(2)));
        assert_eq!(Device::from_str("metal").unwrap(), Device::metal());

        assert!(Device::from_str("invalid").is_err());
        assert!(Device::from_str("cuda:abc").is_err());
    }

    #[test]
    fn test_device_display() {
        assert_eq!(Device::cpu().to_string(), "cpu");
        assert_eq!(Device::cuda(Some(0)).to_string(), "cuda:0");
        assert_eq!(Device::metal().to_string(), "metal");
    }
}

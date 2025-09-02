// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    device::Device, memory::global_allocate, memory::global_deallocate, tensor::dtype::DataType,
};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Tensor data storage with reference counting
#[derive(Debug)]
pub struct TensorData {
    /// Raw data buffer
    buffer: TensorBuffer,
    /// Memory layout information
    layout: MemoryLayout,
    /// Reference count for memory management
    ref_count: AtomicUsize,
}

/// Buffer storage for tensor data
#[derive(Debug)]
enum TensorBuffer {
    /// Owned vector buffer (for CPU)
    Owned(Vec<u8>),
    /// Raw pointer buffer (for GPU or custom allocators)
    Raw {
        ptr: *mut u8,
        size: usize,
        device: Device,
    },
}

/// Memory layout specification
#[derive(Debug, Clone)]
pub struct MemoryLayout {
    /// Data type of elements
    pub dtype: DataType,
    /// Number of elements
    pub numel: usize,
    /// Whether the data is contiguous
    pub is_contiguous: bool,
    /// Device where data is stored
    pub device: Device,
}

impl TensorData {
    /// Create new tensor data with zeros on CPU
    pub fn zeros(numel: usize, dtype: DataType) -> Self {
        Self::zeros_on_device(numel, dtype, Device::cpu())
    }

    /// Create new tensor data with ones on CPU
    pub fn ones(numel: usize, dtype: DataType) -> Self {
        Self::ones_on_device(numel, dtype, Device::cpu())
    }

    /// Create new tensor data with ones on specified device
    pub fn ones_on_device(numel: usize, dtype: DataType, device: Device) -> Self {
        let mut data = Self::zeros_on_device(numel, dtype, device);

        // Fill with ones based on data type
        match dtype {
            DataType::Float32 => {
                if let Some(slice) = data.as_f32_slice_mut() {
                    slice.fill(1.0);
                }
            }
            DataType::Float64 => {
                if let Some(slice) = data.as_f64_slice_mut() {
                    slice.fill(1.0);
                }
            }
            DataType::Int32 => {
                if let Some(slice) = data.as_i32_slice_mut() {
                    slice.fill(1);
                }
            }
            DataType::Int64 => {
                if let Some(slice) = data.as_i64_slice_mut() {
                    slice.fill(1);
                }
            }
            DataType::Bool => {
                if let Some(slice) = data.as_bool_slice_mut() {
                    slice.fill(true);
                }
            }
        }

        data
    }

    /// Create new tensor data with zeros on specified device
    pub fn zeros_on_device(numel: usize, dtype: DataType, device: Device) -> Self {
        let size_bytes = numel * dtype.size_bytes();

        let buffer = if device.is_cpu() {
            // Use Vec for CPU
            TensorBuffer::Owned(vec![0u8; size_bytes])
        } else {
            // Use custom allocator for GPU
            match global_allocate(size_bytes, device) {
                Ok(ptr) => {
                    // Initialize memory to zero
                    unsafe {
                        std::ptr::write_bytes(ptr, 0, size_bytes);
                    }
                    TensorBuffer::Raw {
                        ptr,
                        size: size_bytes,
                        device,
                    }
                }
                Err(_) => {
                    // Fallback to CPU if GPU allocation fails
                    TensorBuffer::Owned(vec![0u8; size_bytes])
                }
            }
        };

        Self {
            buffer,
            layout: MemoryLayout {
                dtype,
                numel,
                is_contiguous: true,
                device,
            },
            ref_count: AtomicUsize::new(1),
        }
    }

    /// Create new tensor data from raw bytes on CPU
    pub fn from_bytes(buffer: Vec<u8>, dtype: DataType, numel: usize) -> Self {
        Self {
            buffer: TensorBuffer::Owned(buffer),
            layout: MemoryLayout {
                dtype,
                numel,
                is_contiguous: true,
                device: Device::cpu(),
            },
            ref_count: AtomicUsize::new(1),
        }
    }

    /// Create tensor data from a vector of typed values
    pub fn from_vec<T: Copy + 'static>(data: Vec<T>, dtype: DataType, device: Device) -> Self {
        let numel = data.len();
        let size_bytes = numel * std::mem::size_of::<T>();

        // Convert typed data to bytes
        let buffer = if device.is_cpu() {
            let bytes = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const u8, size_bytes).to_vec()
            };
            TensorBuffer::Owned(bytes)
        } else {
            // For GPU, allocate and copy
            match global_allocate(size_bytes, device) {
                Ok(ptr) => {
                    unsafe {
                        std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, ptr, size_bytes);
                    }
                    TensorBuffer::Raw {
                        ptr,
                        size: size_bytes,
                        device,
                    }
                }
                Err(_) => {
                    // Fallback to CPU
                    let bytes = unsafe {
                        std::slice::from_raw_parts(data.as_ptr() as *const u8, size_bytes).to_vec()
                    };
                    TensorBuffer::Owned(bytes)
                }
            }
        };

        Self {
            buffer,
            layout: MemoryLayout {
                dtype,
                numel,
                is_contiguous: true,
                device,
            },
            ref_count: AtomicUsize::new(1),
        }
    }

    /// Create tensor data from a vector of f32 values
    pub fn from_vec_f32(data: Vec<f32>, device: Device) -> Self {
        Self::from_vec(data, DataType::Float32, device)
    }

    /// Create tensor data from a vector of f64 values
    pub fn from_vec_f64(data: Vec<f64>, device: Device) -> Self {
        Self::from_vec(data, DataType::Float64, device)
    }

    /// Create tensor data from a vector of i32 values
    pub fn from_vec_i32(data: Vec<i32>, device: Device) -> Self {
        Self::from_vec(data, DataType::Int32, device)
    }

    /// Create tensor data from a vector of i64 values
    pub fn from_vec_i64(data: Vec<i64>, device: Device) -> Self {
        Self::from_vec(data, DataType::Int64, device)
    }

    /// Create tensor data from a vector of bool values
    pub fn from_vec_bool(data: Vec<bool>, device: Device) -> Self {
        Self::from_vec(data, DataType::Bool, device)
    }

    /// Create tensor data from raw pointer (for GPU or external memory)
    pub fn from_raw_ptr(
        ptr: *mut u8,
        size: usize,
        dtype: DataType,
        numel: usize,
        device: Device,
    ) -> Self {
        Self {
            buffer: TensorBuffer::Raw { ptr, size, device },
            layout: MemoryLayout {
                dtype,
                numel,
                is_contiguous: true,
                device,
            },
            ref_count: AtomicUsize::new(1),
        }
    }

    /// Get the raw buffer as a slice (CPU only)
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match &self.buffer {
            TensorBuffer::Owned(vec) => Some(vec.as_slice()),
            TensorBuffer::Raw { ptr, size, device } => {
                if device.is_cpu() {
                    Some(unsafe { std::slice::from_raw_parts(*ptr, *size) })
                } else {
                    None // GPU memory not directly accessible
                }
            }
        }
    }

    /// Get the raw buffer as a mutable slice (CPU only)
    pub fn as_bytes_mut(&mut self) -> Option<&mut [u8]> {
        match &mut self.buffer {
            TensorBuffer::Owned(vec) => Some(vec.as_mut_slice()),
            TensorBuffer::Raw { ptr, size, device } => {
                if device.is_cpu() {
                    Some(unsafe { std::slice::from_raw_parts_mut(*ptr, *size) })
                } else {
                    None // GPU memory not directly accessible
                }
            }
        }
    }

    /// Get the raw pointer (for GPU operations)
    pub fn as_ptr(&self) -> *const u8 {
        match &self.buffer {
            TensorBuffer::Owned(vec) => vec.as_ptr(),
            TensorBuffer::Raw { ptr, .. } => *ptr,
        }
    }

    /// Get the mutable raw pointer (for GPU operations)
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        match &mut self.buffer {
            TensorBuffer::Owned(vec) => vec.as_mut_ptr(),
            TensorBuffer::Raw { ptr, .. } => *ptr,
        }
    }

    /// Get the memory layout
    pub fn layout(&self) -> &MemoryLayout {
        &self.layout
    }

    /// Get the data type
    pub fn dtype(&self) -> DataType {
        self.layout.dtype
    }

    /// Get the number of elements
    pub fn numel(&self) -> usize {
        self.layout.numel
    }

    /// Get the number of elements (alias for numel)
    pub fn len(&self) -> usize {
        self.layout.numel
    }

    /// Get the size in bytes
    pub fn size_bytes(&self) -> usize {
        match &self.buffer {
            TensorBuffer::Owned(vec) => vec.len(),
            TensorBuffer::Raw { size, .. } => *size,
        }
    }

    /// Get the device where data is stored
    pub fn device(&self) -> Device {
        self.layout.device
    }

    /// Check if the data is contiguous
    pub fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous
    }

    /// Increment reference count
    pub fn inc_ref(&self) {
        self.ref_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement reference count and return new count
    pub fn dec_ref(&self) -> usize {
        self.ref_count.fetch_sub(1, Ordering::Relaxed) - 1
    }

    /// Get current reference count
    pub fn ref_count(&self) -> usize {
        self.ref_count.load(Ordering::Relaxed)
    }

    /// Create a copy of the tensor data
    pub fn clone_data(&self) -> Self {
        let new_buffer = match &self.buffer {
            TensorBuffer::Owned(vec) => TensorBuffer::Owned(vec.clone()),
            TensorBuffer::Raw { ptr, size, device } => {
                // For GPU memory, we need to allocate new memory and copy
                if device.is_cpu() {
                    let slice = unsafe { std::slice::from_raw_parts(*ptr, *size) };
                    TensorBuffer::Owned(slice.to_vec())
                } else {
                    // For GPU, allocate new memory and copy (simplified)
                    match global_allocate(*size, *device) {
                        Ok(new_ptr) => TensorBuffer::Raw {
                            ptr: new_ptr,
                            size: *size,
                            device: *device,
                        },
                        Err(_) => TensorBuffer::Owned(vec![0u8; *size]),
                    }
                }
            }
        };

        Self {
            buffer: new_buffer,
            layout: self.layout.clone(),
            ref_count: AtomicUsize::new(1),
        }
    }

    /// Get typed slice for f32 data (CPU only)
    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        if self.layout.dtype != DataType::Float32 || !self.layout.device.is_cpu() {
            return None;
        }

        let ptr = self.as_ptr() as *const f32;
        Some(unsafe { std::slice::from_raw_parts(ptr, self.layout.numel) })
    }

    /// Get mutable typed slice for f32 data (CPU only)
    pub fn as_f32_slice_mut(&mut self) -> Option<&mut [f32]> {
        if self.layout.dtype != DataType::Float32 || !self.layout.device.is_cpu() {
            return None;
        }

        let ptr = self.as_mut_ptr() as *mut f32;
        Some(unsafe { std::slice::from_raw_parts_mut(ptr, self.layout.numel) })
    }

    /// Get typed slice for f64 data (CPU only)
    pub fn as_f64_slice(&self) -> Option<&[f64]> {
        if self.layout.dtype != DataType::Float64 || !self.layout.device.is_cpu() {
            return None;
        }

        let ptr = self.as_ptr() as *const f64;
        Some(unsafe { std::slice::from_raw_parts(ptr, self.layout.numel) })
    }

    /// Get mutable typed slice for f64 data (CPU only)
    pub fn as_f64_slice_mut(&mut self) -> Option<&mut [f64]> {
        if self.layout.dtype != DataType::Float64 || !self.layout.device.is_cpu() {
            return None;
        }

        let ptr = self.as_mut_ptr() as *mut f64;
        Some(unsafe { std::slice::from_raw_parts_mut(ptr, self.layout.numel) })
    }

    /// Get typed slice for i32 data (CPU only)
    pub fn as_i32_slice(&self) -> Option<&[i32]> {
        if self.layout.dtype != DataType::Int32 || !self.layout.device.is_cpu() {
            return None;
        }

        let ptr = self.as_ptr() as *const i32;
        Some(unsafe { std::slice::from_raw_parts(ptr, self.layout.numel) })
    }

    /// Get mutable typed slice for i32 data (CPU only)
    pub fn as_i32_slice_mut(&mut self) -> Option<&mut [i32]> {
        if self.layout.dtype != DataType::Int32 || !self.layout.device.is_cpu() {
            return None;
        }

        let ptr = self.as_mut_ptr() as *mut i32;
        Some(unsafe { std::slice::from_raw_parts_mut(ptr, self.layout.numel) })
    }

    /// Get typed slice for i64 data (CPU only)
    pub fn as_i64_slice(&self) -> Option<&[i64]> {
        if self.layout.dtype != DataType::Int64 || !self.layout.device.is_cpu() {
            return None;
        }

        let ptr = self.as_ptr() as *const i64;
        Some(unsafe { std::slice::from_raw_parts(ptr, self.layout.numel) })
    }

    /// Get mutable typed slice for i64 data (CPU only)
    pub fn as_i64_slice_mut(&mut self) -> Option<&mut [i64]> {
        if self.layout.dtype != DataType::Int64 || !self.layout.device.is_cpu() {
            return None;
        }

        let ptr = self.as_mut_ptr() as *mut i64;
        Some(unsafe { std::slice::from_raw_parts_mut(ptr, self.layout.numel) })
    }

    /// Get typed slice for bool data (CPU only)
    pub fn as_bool_slice(&self) -> Option<&[bool]> {
        if self.layout.dtype != DataType::Bool || !self.layout.device.is_cpu() {
            return None;
        }

        let ptr = self.as_ptr() as *const bool;
        Some(unsafe { std::slice::from_raw_parts(ptr, self.layout.numel) })
    }

    /// Get mutable typed slice for bool data (CPU only)
    pub fn as_bool_slice_mut(&mut self) -> Option<&mut [bool]> {
        if self.layout.dtype != DataType::Bool || !self.layout.device.is_cpu() {
            return None;
        }

        let ptr = self.as_mut_ptr() as *mut bool;
        Some(unsafe { std::slice::from_raw_parts_mut(ptr, self.layout.numel) })
    }
}

impl Drop for TensorData {
    fn drop(&mut self) {
        // Only deallocate if this is the last reference
        if self.ref_count.load(Ordering::Relaxed) == 1 {
            if let TensorBuffer::Raw {
                ptr,
                size: _,
                device,
            } = &self.buffer
            {
                // Deallocate GPU memory
                let _ = global_deallocate(*ptr, *device);
            }
        }
    }
}

unsafe impl Send for TensorData {}
unsafe impl Sync for TensorData {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_data_creation() {
        let data = TensorData::zeros(10, DataType::Float32);
        assert_eq!(data.numel(), 10);
        assert_eq!(data.dtype(), DataType::Float32);
        assert_eq!(data.size_bytes(), 40); // 10 * 4 bytes
        assert!(data.is_contiguous());
        assert_eq!(data.ref_count(), 1);
    }

    #[test]
    fn test_typed_slices() {
        let mut data = TensorData::zeros(5, DataType::Float32);

        {
            let slice = data.as_f32_slice().unwrap();
            assert_eq!(slice.len(), 5);
            assert_eq!(slice, &[0.0; 5]);
        }

        {
            let slice_mut = data.as_f32_slice_mut().unwrap();
            slice_mut[0] = 1.0;
            slice_mut[1] = 2.0;
        }

        let slice = data.as_f32_slice().unwrap();
        assert_eq!(slice[0], 1.0);
        assert_eq!(slice[1], 2.0);
    }

    #[test]
    fn test_device_specific_data() {
        let cpu_data = TensorData::zeros_on_device(10, DataType::Float32, Device::cpu());
        assert_eq!(cpu_data.device(), Device::cpu());
        assert!(cpu_data.as_f32_slice().is_some());

        // Test GPU data creation (will fallback to CPU if GPU not available)
        let gpu_data = TensorData::zeros_on_device(10, DataType::Float32, Device::cuda(Some(0)));
        assert_eq!(gpu_data.numel(), 10);
        assert_eq!(gpu_data.dtype(), DataType::Float32);
    }

    #[test]
    fn test_reference_counting() {
        let data = TensorData::zeros(5, DataType::Float32);
        assert_eq!(data.ref_count(), 1);

        data.inc_ref();
        assert_eq!(data.ref_count(), 2);

        let new_count = data.dec_ref();
        assert_eq!(new_count, 1);
        assert_eq!(data.ref_count(), 1);
    }

    #[test]
    fn test_ones_creation() {
        let data = TensorData::ones(5, DataType::Float32);
        assert_eq!(data.numel(), 5);
        assert_eq!(data.dtype(), DataType::Float32);

        let slice = data.as_f32_slice().unwrap();
        assert_eq!(slice, &[1.0; 5]);
    }

    #[test]
    fn test_ones_different_types() {
        // Test f64
        let data_f64 = TensorData::ones(3, DataType::Float64);
        let slice_f64 = data_f64.as_f64_slice().unwrap();
        assert_eq!(slice_f64, &[1.0; 3]);

        // Test i32
        let data_i32 = TensorData::ones(3, DataType::Int32);
        let slice_i32 = data_i32.as_i32_slice().unwrap();
        assert_eq!(slice_i32, &[1; 3]);

        // Test bool
        let data_bool = TensorData::ones(3, DataType::Bool);
        let slice_bool = data_bool.as_bool_slice().unwrap();
        assert_eq!(slice_bool, &[true; 3]);
    }
}

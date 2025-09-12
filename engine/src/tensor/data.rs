// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    device::Device, memory::global_allocate, memory::global_deallocate, tensor::dtype::DataType,
};
use rayon::prelude::*;
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
    #[inline(always)]
    pub fn zeros(numel: usize, dtype: DataType) -> Self {
        Self::zeros_on_device(numel, dtype, Device::cpu())
    }

    /// Create new tensor data with ones on CPU
    #[inline(always)]
    pub fn ones(numel: usize, dtype: DataType) -> Self {
        Self::ones_on_device(numel, dtype, Device::cpu())
    }

    /// Create new tensor data with ones on specified device
    #[inline(always)]
    pub fn ones_on_device(numel: usize, dtype: DataType, device: Device) -> Self {
        if device.is_cpu() {
            let size_bytes = numel * dtype.size_bytes();
            let mut vec = Vec::with_capacity(size_bytes);
            unsafe {
                vec.set_len(size_bytes);
            }
            let mut data = Self {
                buffer: TensorBuffer::Owned(vec),
                layout: MemoryLayout {
                    dtype,
                    numel,
                    is_contiguous: true,
                    device,
                },
                ref_count: AtomicUsize::new(1),
            };

            // Fill with ones in parallel for large buffers
            match dtype {
                DataType::Float32 => {
                    if let Some(slice) = data.as_f32_slice_mut() {
                        Self::fill_slice(slice, 1.0);
                    }
                }
                DataType::Float64 => {
                    if let Some(slice) = data.as_f64_slice_mut() {
                        Self::fill_slice(slice, 1.0);
                    }
                }
                DataType::Int32 => {
                    if let Some(slice) = data.as_i32_slice_mut() {
                        Self::fill_slice(slice, 1);
                    }
                }
                DataType::Int64 => {
                    if let Some(slice) = data.as_i64_slice_mut() {
                        Self::fill_slice(slice, 1);
                    }
                }
                DataType::Bool => {
                    if let Some(slice) = data.as_bool_slice_mut() {
                        Self::fill_slice(slice, true);
                    }
                }
            }

            data
        } else {
            // For non-CPU devices, fall back to zero initialization
            // and attempt to fill if the allocation falls back to CPU.
            let mut data = Self::zeros_on_device(numel, dtype, device);
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
    }

    #[inline(always)]
    fn fill_slice<T: Copy + Send + Sync>(slice: &mut [T], value: T) {
        if slice.len() >= 1024 {
            slice.par_iter_mut().for_each(|x| *x = value);
        } else {
            slice.fill(value);
        }
    }

    /// Create new tensor data with zeros on specified device
    #[inline(always)]
    pub fn zeros_on_device(numel: usize, dtype: DataType, device: Device) -> Self {
        let size_bytes = numel * dtype.size_bytes();

        let buffer = if device.is_cpu() {
            // Use an uninitialized Vec and explicitly zero it. This avoids the
            // double-initialization that `vec![0u8; size_bytes]` may perform on
            // some platforms and lets the optimizer emit a single `memset`.
            let mut vec = Vec::with_capacity(size_bytes);
            unsafe {
                vec.set_len(size_bytes);
                std::ptr::write_bytes(vec.as_mut_ptr(), 0, size_bytes);
            }
            TensorBuffer::Owned(vec)
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
                    let mut vec = Vec::with_capacity(size_bytes);
                    unsafe {
                        vec.set_len(size_bytes);
                        std::ptr::write_bytes(vec.as_mut_ptr(), 0, size_bytes);
                    }
                    TensorBuffer::Owned(vec)
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

    /// Create new tensor data with uninitialized contents on specified device
    #[inline(always)]
    pub fn uninitialized_on_device(numel: usize, dtype: DataType, device: Device) -> Self {
        let size_bytes = numel * dtype.size_bytes();

        let buffer = if device.is_cpu() {
            // Allocate vector without initializing memory for maximum performance
            let mut vec = Vec::with_capacity(size_bytes);
            unsafe {
                vec.set_len(size_bytes);
            }
            TensorBuffer::Owned(vec)
        } else {
            match global_allocate(size_bytes, device) {
                Ok(ptr) => TensorBuffer::Raw {
                    ptr,
                    size: size_bytes,
                    device,
                },
                Err(_) => {
                    // Fallback to CPU allocation if GPU allocation fails
                    let mut vec = Vec::with_capacity(size_bytes);
                    unsafe {
                        vec.set_len(size_bytes);
                    }
                    TensorBuffer::Owned(vec)
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
    #[inline(always)]
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
    #[inline(always)]
    pub fn from_vec<T: Copy + 'static>(data: Vec<T>, dtype: DataType, device: Device) -> Self {
        let numel = data.len();
        let size_bytes = numel * std::mem::size_of::<T>();

        // Convert typed data to bytes
        let buffer = if device.is_cpu() {
            // For CPU memory we can often avoid an extra allocation by
            // reinterpreting the `Vec<T>` as a `Vec<u8>`. This is safe for all
            // `T` except `bool`, which has a special bit-level representation in
            // Rust. Handle that case explicitly.
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<bool>() {
                // SAFETY: We have confirmed T == bool, so this transmute is valid
                let bools: Vec<bool> = unsafe { std::mem::transmute(data) };
                let bytes: Vec<u8> = bools.into_iter().map(|b| b as u8).collect();
                TensorBuffer::Owned(bytes)
            } else {
                use std::mem::{ManuallyDrop, size_of};
                let mut data = ManuallyDrop::new(data);
                let ptr = data.as_mut_ptr() as *mut u8;
                let len = numel * size_of::<T>();
                let capacity = data.capacity() * size_of::<T>();
                unsafe { TensorBuffer::Owned(Vec::from_raw_parts(ptr, len, capacity)) }
            }
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
                    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<bool>() {
                        let bools: Vec<bool> = unsafe { std::mem::transmute(data) };
                        let bytes: Vec<u8> = bools.into_iter().map(|b| b as u8).collect();
                        TensorBuffer::Owned(bytes)
                    } else {
                        let bytes = unsafe {
                            std::slice::from_raw_parts(data.as_ptr() as *const u8, size_bytes)
                                .to_vec()
                        };
                        TensorBuffer::Owned(bytes)
                    }
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
    #[inline(always)]
    pub fn from_vec_f32(data: Vec<f32>, device: Device) -> Self {
        Self::from_vec(data, DataType::Float32, device)
    }

    /// Create tensor data from a vector of f64 values
    #[inline(always)]
    pub fn from_vec_f64(data: Vec<f64>, device: Device) -> Self {
        Self::from_vec(data, DataType::Float64, device)
    }

    /// Create tensor data from a vector of i32 values
    #[inline(always)]
    pub fn from_vec_i32(data: Vec<i32>, device: Device) -> Self {
        Self::from_vec(data, DataType::Int32, device)
    }

    /// Create tensor data from a vector of i64 values
    #[inline(always)]
    pub fn from_vec_i64(data: Vec<i64>, device: Device) -> Self {
        Self::from_vec(data, DataType::Int64, device)
    }

    /// Create tensor data from a vector of bool values
    #[inline(always)]
    pub fn from_vec_bool(data: Vec<bool>, device: Device) -> Self {
        Self::from_vec(data, DataType::Bool, device)
    }

    /// Create tensor data from raw pointer (for GPU or external memory)
    #[inline(always)]
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
    #[inline(always)]
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match &self.buffer {
            TensorBuffer::Owned(vec) => Some(vec.as_slice()),
            TensorBuffer::Raw { ptr, size, device } => {
                if device.is_cpu() {
                    if *size == 0 {
                        Some(&[])
                    } else {
                        Some(unsafe { std::slice::from_raw_parts(*ptr, *size) })
                    }
                } else {
                    None // GPU memory not directly accessible
                }
            }
        }
    }

    /// Get the raw buffer as a mutable slice (CPU only)
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> Option<&mut [u8]> {
        match &mut self.buffer {
            TensorBuffer::Owned(vec) => Some(vec.as_mut_slice()),
            TensorBuffer::Raw { ptr, size, device } => {
                if device.is_cpu() {
                    if *size == 0 {
                        Some(&mut [])
                    } else {
                        Some(unsafe { std::slice::from_raw_parts_mut(*ptr, *size) })
                    }
                } else {
                    None // GPU memory not directly accessible
                }
            }
        }
    }

    /// Get the raw pointer (for GPU operations)
    #[inline(always)]
    pub fn as_ptr(&self) -> *const u8 {
        match &self.buffer {
            TensorBuffer::Owned(vec) => vec.as_ptr(),
            TensorBuffer::Raw { ptr, .. } => *ptr,
        }
    }

    /// Get the mutable raw pointer (for GPU operations)
    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        match &mut self.buffer {
            TensorBuffer::Owned(vec) => vec.as_mut_ptr(),
            TensorBuffer::Raw { ptr, .. } => *ptr,
        }
    }

    /// Get the memory layout
    #[inline(always)]
    pub fn layout(&self) -> &MemoryLayout {
        &self.layout
    }

    /// Get the data type
    #[inline(always)]
    pub fn dtype(&self) -> DataType {
        self.layout.dtype
    }

    /// Get the number of elements
    #[inline(always)]
    pub fn numel(&self) -> usize {
        self.layout.numel
    }

    /// Get the number of elements (alias for numel)
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.layout.numel
    }

    /// Check if there are no elements
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.layout.numel == 0
    }

    /// Get the size in bytes
    #[inline(always)]
    pub fn size_bytes(&self) -> usize {
        match &self.buffer {
            TensorBuffer::Owned(vec) => vec.len(),
            TensorBuffer::Raw { size, .. } => *size,
        }
    }

    /// Get the device where data is stored
    #[inline(always)]
    pub fn device(&self) -> Device {
        self.layout.device
    }

    /// Check if the data is contiguous
    #[inline(always)]
    pub fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous
    }

    /// Increment reference count
    #[inline(always)]
    pub fn inc_ref(&self) {
        self.ref_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement reference count and return new count
    #[inline(always)]
    pub fn dec_ref(&self) -> usize {
        self.ref_count.fetch_sub(1, Ordering::Relaxed) - 1
    }

    /// Get current reference count
    #[inline(always)]
    pub fn ref_count(&self) -> usize {
        self.ref_count.load(Ordering::Relaxed)
    }

    /// Create a copy of the tensor data
    pub fn clone_data(&self) -> Self {
        let new_buffer = match &self.buffer {
            TensorBuffer::Owned(vec) => {
                let mut out = Vec::with_capacity(vec.len());
                unsafe {
                    out.set_len(vec.len());
                    std::ptr::copy_nonoverlapping(vec.as_ptr(), out.as_mut_ptr(), vec.len());
                }
                TensorBuffer::Owned(out)
            }
            TensorBuffer::Raw { ptr, size, device } => {
                if device.is_cpu() {
                    // Raw CPU pointer: copy into a Vec for safety
                    let mut out = Vec::with_capacity(*size);
                    unsafe {
                        out.set_len(*size);
                        std::ptr::copy_nonoverlapping(*ptr, out.as_mut_ptr(), *size);
                    }
                    TensorBuffer::Owned(out)
                } else {
                    // For GPU, allocate new memory and copy
                    match global_allocate(*size, *device) {
                        Ok(new_ptr) => {
                            unsafe {
                                std::ptr::copy_nonoverlapping(*ptr, new_ptr, *size);
                            }
                            TensorBuffer::Raw {
                                ptr: new_ptr,
                                size: *size,
                                device: *device,
                            }
                        }
                        Err(_) => {
                            // Fallback to CPU buffer to avoid allocation failure
                            let mut out = Vec::with_capacity(*size);
                            unsafe {
                                out.set_len(*size);
                                std::ptr::write_bytes(out.as_mut_ptr(), 0, *size);
                            }
                            TensorBuffer::Owned(out)
                        }
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
    #[inline(always)]
    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        if self.layout.dtype != DataType::Float32 || !self.layout.device.is_cpu() {
            return None;
        }

        let ptr = self.as_ptr() as *const f32;
        let ptr = if self.layout.numel == 0 {
            std::ptr::NonNull::<f32>::dangling().as_ptr()
        } else {
            ptr
        };
        Some(unsafe { std::slice::from_raw_parts(ptr, self.layout.numel) })
    }

    /// Get mutable typed slice for f32 data (CPU only)
    #[inline(always)]
    pub fn as_f32_slice_mut(&mut self) -> Option<&mut [f32]> {
        if self.layout.dtype != DataType::Float32 || !self.layout.device.is_cpu() {
            return None;
        }

        let ptr = self.as_mut_ptr() as *mut f32;
        let ptr = if self.layout.numel == 0 {
            std::ptr::NonNull::<f32>::dangling().as_ptr() as *mut f32
        } else {
            ptr
        };
        Some(unsafe { std::slice::from_raw_parts_mut(ptr, self.layout.numel) })
    }

    /// Get typed slice for f64 data (CPU only)
    #[inline(always)]
    pub fn as_f64_slice(&self) -> Option<&[f64]> {
        if self.layout.dtype != DataType::Float64 || !self.layout.device.is_cpu() {
            return None;
        }

        let ptr = self.as_ptr() as *const f64;
        let ptr = if self.layout.numel == 0 {
            std::ptr::NonNull::<f64>::dangling().as_ptr()
        } else {
            ptr
        };
        Some(unsafe { std::slice::from_raw_parts(ptr, self.layout.numel) })
    }

    /// Get mutable typed slice for f64 data (CPU only)
    #[inline(always)]
    pub fn as_f64_slice_mut(&mut self) -> Option<&mut [f64]> {
        if self.layout.dtype != DataType::Float64 || !self.layout.device.is_cpu() {
            return None;
        }

        let ptr = self.as_mut_ptr() as *mut f64;
        let ptr = if self.layout.numel == 0 {
            std::ptr::NonNull::<f64>::dangling().as_ptr() as *mut f64
        } else {
            ptr
        };
        Some(unsafe { std::slice::from_raw_parts_mut(ptr, self.layout.numel) })
    }

    /// Get typed slice for i32 data (CPU only)
    #[inline(always)]
    pub fn as_i32_slice(&self) -> Option<&[i32]> {
        if self.layout.dtype != DataType::Int32 || !self.layout.device.is_cpu() {
            return None;
        }

        let ptr = self.as_ptr() as *const i32;
        let ptr = if self.layout.numel == 0 {
            std::ptr::NonNull::<i32>::dangling().as_ptr()
        } else {
            ptr
        };
        Some(unsafe { std::slice::from_raw_parts(ptr, self.layout.numel) })
    }

    /// Get mutable typed slice for i32 data (CPU only)
    #[inline(always)]
    pub fn as_i32_slice_mut(&mut self) -> Option<&mut [i32]> {
        if self.layout.dtype != DataType::Int32 || !self.layout.device.is_cpu() {
            return None;
        }

        let ptr = self.as_mut_ptr() as *mut i32;
        let ptr = if self.layout.numel == 0 {
            std::ptr::NonNull::<i32>::dangling().as_ptr() as *mut i32
        } else {
            ptr
        };
        Some(unsafe { std::slice::from_raw_parts_mut(ptr, self.layout.numel) })
    }

    /// Get typed slice for i64 data (CPU only)
    #[inline(always)]
    pub fn as_i64_slice(&self) -> Option<&[i64]> {
        if self.layout.dtype != DataType::Int64 || !self.layout.device.is_cpu() {
            return None;
        }

        let ptr = self.as_ptr() as *const i64;
        let ptr = if self.layout.numel == 0 {
            std::ptr::NonNull::<i64>::dangling().as_ptr()
        } else {
            ptr
        };
        Some(unsafe { std::slice::from_raw_parts(ptr, self.layout.numel) })
    }

    /// Get mutable typed slice for i64 data (CPU only)
    #[inline(always)]
    pub fn as_i64_slice_mut(&mut self) -> Option<&mut [i64]> {
        if self.layout.dtype != DataType::Int64 || !self.layout.device.is_cpu() {
            return None;
        }

        let ptr = self.as_mut_ptr() as *mut i64;
        let ptr = if self.layout.numel == 0 {
            std::ptr::NonNull::<i64>::dangling().as_ptr() as *mut i64
        } else {
            ptr
        };
        Some(unsafe { std::slice::from_raw_parts_mut(ptr, self.layout.numel) })
    }

    /// Get typed slice for bool data (CPU only)
    #[inline(always)]
    pub fn as_bool_slice(&self) -> Option<&[bool]> {
        if self.layout.dtype != DataType::Bool || !self.layout.device.is_cpu() {
            return None;
        }

        let ptr = self.as_ptr() as *const bool;
        let ptr = if self.layout.numel == 0 {
            std::ptr::NonNull::<bool>::dangling().as_ptr()
        } else {
            ptr
        };
        Some(unsafe { std::slice::from_raw_parts(ptr, self.layout.numel) })
    }

    /// Get mutable typed slice for bool data (CPU only)
    #[inline(always)]
    pub fn as_bool_slice_mut(&mut self) -> Option<&mut [bool]> {
        if self.layout.dtype != DataType::Bool || !self.layout.device.is_cpu() {
            return None;
        }

        let ptr = self.as_mut_ptr() as *mut bool;
        let ptr = if self.layout.numel == 0 {
            std::ptr::NonNull::<bool>::dangling().as_ptr() as *mut bool
        } else {
            ptr
        };
        Some(unsafe { std::slice::from_raw_parts_mut(ptr, self.layout.numel) })
    }
}

impl Drop for TensorData {
    fn drop(&mut self) {
        // Only deallocate if this is the last reference
        if self.ref_count.load(Ordering::Relaxed) == 1 {
            if let TensorBuffer::Raw { ptr, size, device } = &self.buffer {
                // Deallocate GPU memory
                let _ = global_deallocate(*ptr, *size, *device);
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

    #[test]
    fn test_zeros_different_types() {
        // Test f64
        let data_f64 = TensorData::zeros(2, DataType::Float64);
        assert_eq!(data_f64.as_f64_slice().unwrap(), &[0.0; 2]);

        // Test i32
        let data_i32 = TensorData::zeros(2, DataType::Int32);
        assert_eq!(data_i32.as_i32_slice().unwrap(), &[0; 2]);

        // Test i64
        let data_i64 = TensorData::zeros(2, DataType::Int64);
        assert_eq!(data_i64.as_i64_slice().unwrap(), &[0; 2]);

        // Test bool
        let data_bool = TensorData::zeros(2, DataType::Bool);
        assert_eq!(data_bool.as_bool_slice().unwrap(), &[false; 2]);
    }

    #[test]
    fn test_from_vec_and_bytes_roundtrip() {
        let values = vec![1.0f32, 2.0, 3.0];
        let data = TensorData::from_vec_f32(values.clone(), Device::cpu());
        assert_eq!(data.numel(), 3);
        assert_eq!(data.dtype(), DataType::Float32);
        assert_eq!(data.as_f32_slice().unwrap(), values.as_slice());
        let bytes = data.as_bytes().unwrap();
        assert_eq!(bytes.len(), 3 * 4);
    }

    #[test]
    fn test_from_vec_various_types() {
        let data_i32 = TensorData::from_vec_i32(vec![1, -2, 3], Device::cpu());
        assert_eq!(data_i32.as_i32_slice().unwrap(), &[1, -2, 3]);

        let data_i64 = TensorData::from_vec_i64(vec![1, -2, 3], Device::cpu());
        assert_eq!(data_i64.as_i64_slice().unwrap(), &[1, -2, 3]);

        let data_bool = TensorData::from_vec_bool(vec![true, false, true], Device::cpu());
        assert_eq!(data_bool.as_bool_slice().unwrap(), &[true, false, true]);
    }

    #[test]
    fn test_clone_data_independence() {
        let original = TensorData::ones(3, DataType::Float32);
        let mut cloned = original.clone_data();

        // modify the clone and ensure original remains unchanged
        {
            let slice = cloned.as_f32_slice_mut().unwrap();
            slice[0] = 5.0;
        }

        assert_eq!(original.as_f32_slice().unwrap(), &[1.0; 3]);
        assert_eq!(cloned.as_f32_slice().unwrap()[0], 5.0);
    }
}

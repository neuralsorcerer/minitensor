// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    device::Device, memory::global_allocate, memory::global_deallocate, tensor::dtype::DataType,
};
use rayon::prelude::*;
use std::cell::UnsafeCell;

/// Tensor data storage.
///
/// Sharing is managed exclusively through `Arc<TensorData>`; the struct itself
/// owns its buffer and frees it on drop.
///
/// The buffer lives in an [`UnsafeCell`] because one mutation path is
/// deliberately allowed through a shared reference: in-place parameter
/// updates, which must stay visible through every `Arc` handle to the
/// parameter (see [`DataMut`]). All other access goes through ordinary
/// `&self`/`&mut self` methods.
pub struct TensorData {
    /// Raw data buffer
    buffer: UnsafeCell<TensorBuffer>,
    /// Memory layout information
    layout: MemoryLayout,
}

impl std::fmt::Debug for TensorData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = match self.buffer_ref() {
            TensorBuffer::Owned(_) => "owned",
            TensorBuffer::Raw { .. } => "raw",
        };
        f.debug_struct("TensorData")
            .field("buffer", &kind)
            .field("layout", &self.layout)
            .finish()
    }
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

/// Generates the typed slice accessors for one dtype:
/// - `$shared(&self)`: shared read view;
/// - `$exclusive(&mut self)`: exclusive write view;
/// - `$unchecked(&self)`: write view through a shared reference, used only by
///   [`DataMut`] for in-place parameter updates (see its safety contract).
///
/// Empty tensors hand out well-aligned dangling pointers so callers can treat
/// every dtype uniformly. All accessors return `None` for dtype mismatches or
/// non-CPU storage.
macro_rules! typed_slice_accessors {
    ($ty:ty, $variant:ident, $shared:ident, $exclusive:ident, $unchecked:ident) => {
        #[inline(always)]
        pub fn $shared(&self) -> Option<&[$ty]> {
            if self.layout.dtype != DataType::$variant || !self.layout.device.is_cpu() {
                return None;
            }
            let ptr = if self.layout.numel == 0 {
                std::ptr::NonNull::<$ty>::dangling().as_ptr()
            } else {
                self.as_ptr() as *const $ty
            };
            Some(unsafe { std::slice::from_raw_parts(ptr, self.layout.numel) })
        }

        #[inline(always)]
        pub fn $exclusive(&mut self) -> Option<&mut [$ty]> {
            if self.layout.dtype != DataType::$variant || !self.layout.device.is_cpu() {
                return None;
            }
            let ptr = if self.layout.numel == 0 {
                std::ptr::NonNull::<$ty>::dangling().as_ptr()
            } else {
                self.as_mut_ptr() as *mut $ty
            };
            Some(unsafe { std::slice::from_raw_parts_mut(ptr, self.layout.numel) })
        }

        /// # Safety
        /// See [`TensorData::data_ptr_shared`]: the returned slice must not
        /// overlap the lifetime of any other reference to this tensor's
        /// element data, and access must be externally synchronized.
        // `&self -> &mut` is the point of this accessor: it is the
        // UnsafeCell-backed interior-mutability path for in-place parameter
        // updates, with the aliasing contract stated above.
        #[allow(clippy::mut_from_ref)]
        #[inline(always)]
        pub(crate) unsafe fn $unchecked(&self) -> Option<&mut [$ty]> {
            if self.layout.dtype != DataType::$variant || !self.layout.device.is_cpu() {
                return None;
            }
            let ptr = if self.layout.numel == 0 {
                std::ptr::NonNull::<$ty>::dangling().as_ptr()
            } else {
                unsafe { self.data_ptr_shared() as *mut $ty }
            };
            Some(unsafe { std::slice::from_raw_parts_mut(ptr, self.layout.numel) })
        }
    };
}

impl TensorData {
    /// Shared view of the buffer.
    ///
    /// Sound because nothing ever mutates the `TensorBuffer` value itself
    /// (enum tag, `Vec` header, raw-pointer fields) through a shared
    /// reference — the shared-mutation path in [`DataMut`] only writes the
    /// *element bytes* the buffer points to.
    #[inline(always)]
    fn buffer_ref(&self) -> &TensorBuffer {
        unsafe { &*self.buffer.get() }
    }

    /// Pointer to the element bytes with write provenance, obtained through
    /// the `UnsafeCell` from a shared reference.
    ///
    /// # Safety
    /// Callers must guarantee that writes through the returned pointer do not
    /// overlap the lifetime of any other reference to this tensor's element
    /// data, and that access is externally synchronized (in practice: the
    /// Python GIL serializes optimizer steps, and gradient functions only
    /// read their saved operands before the step runs).
    #[inline(always)]
    unsafe fn data_ptr_shared(&self) -> *mut u8 {
        // A transient `&mut TensorBuffer` scoped to this expression is the
        // sanctioned way to derive a writable pointer from an `UnsafeCell`.
        match unsafe { &mut *self.buffer.get() } {
            TensorBuffer::Owned(vec) => vec.as_mut_ptr(),
            TensorBuffer::Raw { ptr, .. } => *ptr,
        }
    }

    #[inline(always)]
    fn validate_from_vec_type<T: 'static>(dtype: DataType) {
        let type_id = std::any::TypeId::of::<T>();
        let matches = match dtype {
            DataType::Float32 => type_id == std::any::TypeId::of::<f32>(),
            DataType::Float64 => type_id == std::any::TypeId::of::<f64>(),
            DataType::Int32 => type_id == std::any::TypeId::of::<i32>(),
            DataType::Int64 => type_id == std::any::TypeId::of::<i64>(),
            DataType::Bool => type_id == std::any::TypeId::of::<bool>(),
        };

        assert!(matches, "dtype/type mismatch in TensorData::from_vec");
    }

    /// Allocate a zero-initialized byte buffer (`alloc_zeroed`).
    #[inline(always)]
    fn owned_zeroed_buffer(size_bytes: usize) -> Vec<u8> {
        vec![0u8; size_bytes]
    }

    /// Allocate a CPU buffer for an operation output obtained through
    /// [`Self::uninitialized_on_device`].
    ///
    /// **This buffer is zero-initialized and callers rely on that.** A kernel
    /// that fetches it as `&mut [f32]`/`&mut [i64]`/… (via `as_*_slice_mut`)
    /// and does not write every element would otherwise expose uninitialized
    /// memory: `&mut T` must point to a *valid* `T`, and a float/integer read
    /// from uninitialized memory is not a valid value even though every bit
    /// pattern is representable. The zeroing makes any un-overwritten element a
    /// defined zero rather than undefined behavior, so this must not be
    /// weakened to a genuinely-uninitialized allocation while any such caller
    /// exists (activation, reduction, and several autograd kernels still take
    /// this path).
    ///
    /// The zero-cost alternative — writing each output element exactly once
    /// through `MaybeUninit`, with no `memset` — is implemented in
    /// [`crate::ops::map`] (`build_vec_with` and the `unary_map` /
    /// `binary_map` / `strided_gather` / `broadcast_binary_map` combinators
    /// built on it) and is what the element-wise arithmetic, `where`,
    /// `tril`/`triu`, and gather kernels now use; those no longer allocate
    /// through here at all. Migrating the remaining zero-then-overwrite kernels
    /// onto that path is the tracked follow-up in docs/architecture_review.md.
    ///
    /// Measured cost of the `memset` this path still pays (interleaved A/B,
    /// release): ~25–35% on the pure element-wise microbenchmark (bound by
    /// output write traffic), within noise on matmul-dominated training steps.
    #[inline(always)]
    fn owned_buffer_for_dtype(size_bytes: usize, _dtype: DataType) -> Vec<u8> {
        Self::owned_zeroed_buffer(size_bytes)
    }

    #[inline(always)]
    fn fill_with_ones(&mut self) {
        match self.layout.dtype {
            DataType::Float32 => {
                if let Some(slice) = self.as_f32_slice_mut() {
                    Self::fill_slice(slice, 1.0);
                }
            }
            DataType::Float64 => {
                if let Some(slice) = self.as_f64_slice_mut() {
                    Self::fill_slice(slice, 1.0);
                }
            }
            DataType::Int32 => {
                if let Some(slice) = self.as_i32_slice_mut() {
                    Self::fill_slice(slice, 1);
                }
            }
            DataType::Int64 => {
                if let Some(slice) = self.as_i64_slice_mut() {
                    Self::fill_slice(slice, 1);
                }
            }
            DataType::Bool => {
                if let Some(slice) = self.as_bool_slice_mut() {
                    Self::fill_slice(slice, true);
                }
            }
        }
    }

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
            // Build the filled buffer in one pass instead of zeroing and then
            // overwriting it.
            match dtype {
                DataType::Float32 => Self::from_vec(vec![1.0f32; numel], dtype, device),
                DataType::Float64 => Self::from_vec(vec![1.0f64; numel], dtype, device),
                DataType::Int32 => Self::from_vec(vec![1i32; numel], dtype, device),
                DataType::Int64 => Self::from_vec(vec![1i64; numel], dtype, device),
                DataType::Bool => Self::from_vec(vec![true; numel], dtype, device),
            }
        } else {
            // For non-CPU devices, fall back to zero initialization
            // and attempt to fill if the allocation falls back to CPU.
            let mut data = Self::zeros_on_device(numel, dtype, device);
            data.fill_with_ones();
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
        let size_bytes = numel
            .checked_mul(dtype.size_bytes())
            .expect("tensor size overflow");

        let buffer = if device.is_cpu() {
            TensorBuffer::Owned(Self::owned_zeroed_buffer(size_bytes))
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
                    TensorBuffer::Owned(Self::owned_zeroed_buffer(size_bytes))
                }
            }
        };
        let actual_device = match &buffer {
            TensorBuffer::Owned(_) => Device::cpu(),
            TensorBuffer::Raw { device, .. } => *device,
        };

        Self {
            buffer: UnsafeCell::new(buffer),
            layout: MemoryLayout {
                dtype,
                numel,
                is_contiguous: true,
                device: actual_device,
            },
        }
    }

    /// Create new tensor data for use as an operation output buffer.
    ///
    /// Historically this handed out genuinely uninitialized memory; CPU
    /// buffers are now zero-initialized via `alloc_zeroed` (see
    /// [`Self::owned_zeroed_buffer`]), which keeps the fast allocation path
    /// while making accidental reads defined. The name is kept for API
    /// stability; callers should still treat the contents as unspecified.
    #[inline(always)]
    pub fn uninitialized_on_device(numel: usize, dtype: DataType, device: Device) -> Self {
        let size_bytes = numel
            .checked_mul(dtype.size_bytes())
            .expect("tensor size overflow");

        let buffer = if device.is_cpu() {
            TensorBuffer::Owned(Self::owned_buffer_for_dtype(size_bytes, dtype))
        } else {
            match global_allocate(size_bytes, device) {
                Ok(ptr) => TensorBuffer::Raw {
                    ptr,
                    size: size_bytes,
                    device,
                },
                Err(_) => {
                    // Fallback to CPU allocation if GPU allocation fails
                    TensorBuffer::Owned(Self::owned_buffer_for_dtype(size_bytes, dtype))
                }
            }
        };
        let actual_device = match &buffer {
            TensorBuffer::Owned(_) => Device::cpu(),
            TensorBuffer::Raw { device, .. } => *device,
        };

        Self {
            buffer: UnsafeCell::new(buffer),
            layout: MemoryLayout {
                dtype,
                numel,
                is_contiguous: true,
                device: actual_device,
            },
        }
    }

    /// Create new tensor data from raw bytes on CPU
    #[inline(always)]
    pub fn from_bytes(buffer: Vec<u8>, dtype: DataType, numel: usize) -> Self {
        Self {
            buffer: UnsafeCell::new(TensorBuffer::Owned(buffer)),
            layout: MemoryLayout {
                dtype,
                numel,
                is_contiguous: true,
                device: Device::cpu(),
            },
        }
    }

    /// Create tensor data from a vector of typed values
    #[inline(always)]
    pub fn from_vec<T: Copy + 'static>(data: Vec<T>, dtype: DataType, device: Device) -> Self {
        let numel = data.len();
        let size_bytes = numel
            .checked_mul(std::mem::size_of::<T>())
            .expect("tensor size overflow");
        assert_eq!(
            std::mem::size_of::<T>(),
            dtype.size_bytes(),
            "dtype size mismatch in TensorData::from_vec"
        );
        Self::validate_from_vec_type::<T>(dtype);

        // Convert typed data to bytes
        let buffer = if device.is_cpu() {
            // For CPU memory we avoid an extra allocation by reinterpreting
            // the `Vec<T>` as a `Vec<u8>`. This is sound for every supported
            // element type, including `bool`: a `bool` is one byte whose only
            // valid representations (0x00/0x01) are also valid `u8`s. (Only
            // the reverse u8-to-bool direction would need a copy with
            // validation, and `from_vec` never goes that way.)
            {
                use std::mem::{ManuallyDrop, size_of};
                let mut data = ManuallyDrop::new(data);
                let ptr = data.as_mut_ptr() as *mut u8;
                let len = size_bytes;
                let capacity = data
                    .capacity()
                    .checked_mul(size_of::<T>())
                    .expect("tensor size overflow");
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
                    // Fallback to CPU. Byte-copying is sound for `bool` too:
                    // its valid representations (0x00/0x01) are valid `u8`s.
                    let bytes = unsafe {
                        std::slice::from_raw_parts(data.as_ptr() as *const u8, size_bytes).to_vec()
                    };
                    TensorBuffer::Owned(bytes)
                }
            }
        };
        let actual_device = match &buffer {
            TensorBuffer::Owned(_) => Device::cpu(),
            TensorBuffer::Raw { device, .. } => *device,
        };

        Self {
            buffer: UnsafeCell::new(buffer),
            layout: MemoryLayout {
                dtype,
                numel,
                is_contiguous: true,
                device: actual_device,
            },
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
            buffer: UnsafeCell::new(TensorBuffer::Raw { ptr, size, device }),
            layout: MemoryLayout {
                dtype,
                numel,
                is_contiguous: true,
                device,
            },
        }
    }

    /// Get the raw buffer as a slice (CPU only)
    #[inline(always)]
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self.buffer_ref() {
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
        match self.buffer.get_mut() {
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
        match self.buffer_ref() {
            TensorBuffer::Owned(vec) => vec.as_ptr(),
            TensorBuffer::Raw { ptr, .. } => *ptr,
        }
    }

    /// Get the mutable raw pointer (for GPU operations)
    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        match self.buffer.get_mut() {
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
        match self.buffer_ref() {
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

    /// Create a copy of the tensor data
    pub fn clone_data(&self) -> Self {
        let new_buffer = match self.buffer_ref() {
            TensorBuffer::Owned(vec) => TensorBuffer::Owned(vec.clone()),
            TensorBuffer::Raw { ptr, size, device } => {
                if device.is_cpu() {
                    // Raw CPU pointer: copy into a Vec for safety
                    let bytes = unsafe { std::slice::from_raw_parts(*ptr, *size) }.to_vec();
                    TensorBuffer::Owned(bytes)
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
                            // Fallback to a zeroed CPU buffer to avoid
                            // failing the clone outright.
                            TensorBuffer::Owned(Self::owned_zeroed_buffer(*size))
                        }
                    }
                }
            }
        };

        Self {
            buffer: UnsafeCell::new(new_buffer),
            layout: self.layout.clone(),
        }
    }

    typed_slice_accessors!(
        f32,
        Float32,
        as_f32_slice,
        as_f32_slice_mut,
        as_f32_slice_mut_unchecked
    );
    typed_slice_accessors!(
        f64,
        Float64,
        as_f64_slice,
        as_f64_slice_mut,
        as_f64_slice_mut_unchecked
    );
    typed_slice_accessors!(
        i32,
        Int32,
        as_i32_slice,
        as_i32_slice_mut,
        as_i32_slice_mut_unchecked
    );
    typed_slice_accessors!(
        i64,
        Int64,
        as_i64_slice,
        as_i64_slice_mut,
        as_i64_slice_mut_unchecked
    );
    typed_slice_accessors!(
        bool,
        Bool,
        as_bool_slice,
        as_bool_slice_mut,
        as_bool_slice_mut_unchecked
    );
}

/// Mutable access to a tensor's storage, produced by `Tensor::data_mut`.
///
/// The `Unique` variant is ordinary exclusive access (storage uniquely owned,
/// possibly after copy-on-write). The `Shared` variant is the one deliberate
/// exception in the crate: in-place updates of leaf parameters whose storage
/// is shared across `Arc` handles, so the update stays visible through every
/// handle (PyTorch in-place parameter semantics). Its safety contract is
/// documented on [`TensorData::data_ptr_shared`] and upheld by the callers:
/// optimizer steps run GIL-serialized, after backward has finished reading
/// saved operands.
pub enum DataMut<'a> {
    Unique(&'a mut TensorData),
    Shared(&'a TensorData),
}

macro_rules! data_mut_accessor {
    ($name:ident, $unchecked:ident, $ty:ty) => {
        /// Consumes the access token and returns a slice borrowing from the
        /// underlying tensor, so `t.data_mut().as_…_slice_mut()` keeps the
        /// slice alive for the caller's borrow of `t`.
        #[inline(always)]
        pub fn $name(self) -> Option<&'a mut [$ty]> {
            match self {
                DataMut::Unique(data) => data.$name(),
                // SAFETY: contract documented on `DataMut` and
                // `TensorData::data_ptr_shared`.
                DataMut::Shared(data) => unsafe { data.$unchecked() },
            }
        }
    };
}

impl<'a> DataMut<'a> {
    data_mut_accessor!(as_f32_slice_mut, as_f32_slice_mut_unchecked, f32);
    data_mut_accessor!(as_f64_slice_mut, as_f64_slice_mut_unchecked, f64);
    data_mut_accessor!(as_i32_slice_mut, as_i32_slice_mut_unchecked, i32);
    data_mut_accessor!(as_i64_slice_mut, as_i64_slice_mut_unchecked, i64);
    data_mut_accessor!(as_bool_slice_mut, as_bool_slice_mut_unchecked, bool);
}

impl Drop for TensorData {
    fn drop(&mut self) {
        // `TensorData` is shared via `Arc`, so `drop` runs exactly once, when
        // the last reference goes away. Owned buffers free themselves; raw
        // device buffers are returned to the allocator here.
        if let TensorBuffer::Raw { ptr, size, device } = self.buffer.get_mut() {
            let _ = global_deallocate(*ptr, *size, *device);
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
    fn test_arc_sharing() {
        use std::sync::Arc;

        let data = Arc::new(TensorData::zeros(5, DataType::Float32));
        let shared = Arc::clone(&data);
        assert_eq!(Arc::strong_count(&data), 2);
        drop(shared);
        assert_eq!(Arc::strong_count(&data), 1);
        assert_eq!(data.numel(), 5);
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
    fn test_uninitialized_bool_cpu_is_valid_false_initialized() {
        let data = TensorData::uninitialized_on_device(4, DataType::Bool, Device::cpu());
        assert_eq!(data.as_bool_slice().unwrap(), &[false; 4]);
    }

    #[test]
    #[should_panic(expected = "dtype size mismatch in TensorData::from_vec")]
    fn test_from_vec_dtype_size_mismatch_panics() {
        let _ = TensorData::from_vec(vec![1_i64, 2_i64], DataType::Int32, Device::cpu());
    }

    #[test]
    #[should_panic(expected = "dtype/type mismatch in TensorData::from_vec")]
    fn test_from_vec_dtype_type_mismatch_panics() {
        let _ = TensorData::from_vec(vec![2_u8, 0_u8], DataType::Bool, Device::cpu());
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

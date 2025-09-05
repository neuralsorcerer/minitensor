// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

pub mod stream;

use super::Backend;
use crate::{device::Device, error::Result};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use cudarc::nvrtc::Ptx;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub use stream::{CudaExecutionContext, CudaStreamPool, PooledCudaStream};

/// CUDA backend for GPU tensor operations
pub struct CudaBackend {
    device: Device,
    cuda_device: Arc<CudaDevice>,
    execution_context: Arc<Mutex<CudaExecutionContext>>,
    kernels: Arc<Mutex<HashMap<String, cudarc::driver::CudaFunction>>>,
}

impl CudaBackend {
    /// Get the CUDA device
    pub fn cuda_device(&self) -> &Arc<CudaDevice> {
        &self.cuda_device
    }

    /// Get the execution context
    pub fn execution_context(&self) -> &Arc<Mutex<CudaExecutionContext>> {
        &self.execution_context
    }

    /// Load and compile a CUDA kernel
    pub fn load_kernel(&self, name: &str, ptx_src: &str) -> Result<()> {
        let ptx = Ptx::from_src(ptx_src);
        self.cuda_device.load_ptx(ptx, name, &[name]).map_err(|e| {
            crate::error::MinitensorError::backend_error(
                "CUDA",
                format!("Failed to load CUDA kernel: {}", e),
            )
        })?;

        let function = self.cuda_device.get_func(name, name).map_err(|e| {
            crate::error::MinitensorError::backend_error(
                "CUDA",
                format!("Failed to get CUDA function: {}", e),
            )
        })?;

        let mut kernels = self.kernels.lock().unwrap();
        kernels.insert(name.to_string(), function);
        Ok(())
    }

    /// Get a loaded kernel function
    pub fn get_kernel(&self, name: &str) -> Option<cudarc::driver::CudaFunction> {
        let kernels = self.kernels.lock().unwrap();
        kernels.get(name).cloned()
    }

    /// Allocate device memory and return a CudaSlice
    pub fn allocate_slice<T>(&self, len: usize) -> Result<CudaSlice<T>>
    where
        T: cudarc::driver::DeviceRepr,
    {
        self.cuda_device.alloc_zeros::<T>(len).map_err(|e| {
            crate::error::MinitensorError::memory_error(format!("CUDA allocation failed: {}", e))
        })
    }

    /// Copy data from host to device
    pub fn copy_to_device<T>(&self, data: &[T]) -> Result<CudaSlice<T>>
    where
        T: cudarc::driver::DeviceRepr + Clone,
    {
        self.cuda_device.htod_copy(data.to_vec()).map_err(|e| {
            crate::error::MinitensorError::memory_error(format!(
                "Host to device copy failed: {}",
                e
            ))
        })
    }

    /// Copy data from device to host
    pub fn copy_from_device<T>(&self, device_data: &CudaSlice<T>) -> Result<Vec<T>>
    where
        T: cudarc::driver::DeviceRepr + Clone,
    {
        self.cuda_device.dtoh_sync_copy(device_data).map_err(|e| {
            crate::error::MinitensorError::memory_error(format!(
                "Device to host copy failed: {}",
                e
            ))
        })
    }

    /// Synchronize the device
    pub fn synchronize(&self) -> Result<()> {
        self.cuda_device.synchronize().map_err(|e| {
            crate::error::MinitensorError::backend_error(
                "CUDA",
                format!("CUDA synchronization failed: {}", e),
            )
        })
    }
}

impl Backend for CudaBackend {
    fn device(&self) -> Device {
        self.device
    }

    fn is_available() -> bool {
        CudaDevice::new(0).is_ok()
    }

    fn initialize() -> Result<Self> {
        let device_id = 0; // Default to device 0, could be configurable
        let cuda_device = CudaDevice::new(device_id).map_err(|e| {
            crate::error::MinitensorError::backend_error(
                "CUDA",
                format!("Failed to initialize CUDA device: {}", e),
            )
        })?;

        let cuda_device = Arc::new(cuda_device);
        let execution_context = Arc::new(Mutex::new(CudaExecutionContext::new(
            cuda_device.clone(),
            8,
        )));

        Ok(Self {
            device: Device::cuda(Some(device_id)),
            cuda_device,
            execution_context,
            kernels: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    fn allocate(&self, size_bytes: usize) -> Result<*mut u8> {
        if size_bytes == 0 {
            return Ok(std::ptr::null_mut());
        }

        let ptr = self.cuda_device.alloc::<u8>(size_bytes).map_err(|e| {
            crate::error::MinitensorError::memory_error(format!("CUDA allocation failed: {}", e))
        })?;

        Ok(ptr.device_ptr() as *mut u8)
    }

    fn deallocate(&self, ptr: *mut u8, _size_bytes: usize) -> Result<()> {
        if ptr.is_null() {
            return Ok(());
        }

        // Note: cudarc handles deallocation automatically when CudaSlice is dropped
        // This is a simplified implementation - in practice, we'd need to track allocations
        Ok(())
    }

    fn copy_from_host(&self, dst: *mut u8, src: &[u8]) -> Result<()> {
        if dst.is_null() {
            return Err(crate::error::MinitensorError::memory_error(
                "Null destination pointer",
            ));
        }

        // Create a temporary CudaSlice from the raw pointer
        // This is unsafe and simplified - in practice, we'd need better memory management
        unsafe {
            let device_ptr = DevicePtr::from_raw(dst as *mut cudarc::driver::sys::CUdeviceptr);
            self.cuda_device
                .htod_copy_into(src, &device_ptr)
                .map_err(|e| {
                    crate::error::MinitensorError::memory_error(format!(
                        "Host to device copy failed: {}",
                        e
                    ))
                })?;
        }

        Ok(())
    }

    fn copy_to_host(&self, dst: &mut [u8], src: *const u8) -> Result<()> {
        if src.is_null() {
            return Err(crate::error::MinitensorError::memory_error(
                "Null source pointer",
            ));
        }

        // Create a temporary CudaSlice from the raw pointer
        // This is unsafe and simplified - in practice, we'd need better memory management
        unsafe {
            let device_ptr = DevicePtr::from_raw(src as *mut cudarc::driver::sys::CUdeviceptr);
            self.cuda_device
                .dtoh_copy_into(&device_ptr, dst)
                .map_err(|e| {
                    crate::error::MinitensorError::memory_error(format!(
                        "Device to host copy failed: {}",
                        e
                    ))
                })?;
        }

        Ok(())
    }
}

/// CUDA kernel implementations for basic tensor operations
pub mod kernels {
    /// Element-wise addition kernel
    pub const ADD_KERNEL: &str = r#"
extern "C" __global__ void add_kernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
"#;

    /// Element-wise multiplication kernel
    pub const MUL_KERNEL: &str = r#"
extern "C" __global__ void mul_kernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}
"#;

    /// Matrix multiplication kernel (simplified)
    pub const MATMUL_KERNEL: &str = r#"
extern "C" __global__ void matmul_kernel(float* a, float* b, float* c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}
"#;

    /// ReLU activation kernel
    pub const RELU_KERNEL: &str = r#"
extern "C" __global__ void relu_kernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}
"#;

    /// Sigmoid activation kernel
    pub const SIGMOID_KERNEL: &str = r#"
extern "C" __global__ void sigmoid_kernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}
"#;
}

/// CUDA operations for tensor computations
pub struct CudaOps {
    backend: Arc<CudaBackend>,
}

impl CudaOps {
    /// Create new CUDA operations instance
    pub fn new(backend: Arc<CudaBackend>) -> Result<Self> {
        let ops = Self { backend };

        // Load basic kernels
        ops.backend.load_kernel("add_kernel", kernels::ADD_KERNEL)?;
        ops.backend.load_kernel("mul_kernel", kernels::MUL_KERNEL)?;
        ops.backend
            .load_kernel("matmul_kernel", kernels::MATMUL_KERNEL)?;
        ops.backend
            .load_kernel("relu_kernel", kernels::RELU_KERNEL)?;
        ops.backend
            .load_kernel("sigmoid_kernel", kernels::SIGMOID_KERNEL)?;

        Ok(ops)
    }

    /// Element-wise addition on GPU
    pub fn add(
        &self,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let n = a.len();
        if b.len() != n || c.len() != n {
            return Err(crate::error::MinitensorError::shape_error(
                "Tensor dimensions must match for addition",
            ));
        }

        let kernel = self.backend.get_kernel("add_kernel").ok_or_else(|| {
            crate::error::MinitensorError::backend_error("CUDA", "Add kernel not found")
        })?;

        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;

        unsafe {
            kernel
                .launch(
                    (grid_size as u32, 1, 1),
                    (block_size as u32, 1, 1),
                    0,
                    &[
                        a.device_ptr() as *const std::ffi::c_void,
                        b.device_ptr() as *const std::ffi::c_void,
                        c.device_ptr() as *const std::ffi::c_void,
                        &(n as i32) as *const i32 as *const std::ffi::c_void,
                    ],
                )
                .map_err(|e| {
                    crate::error::MinitensorError::backend_error(
                        "CUDA",
                        format!("Kernel launch failed: {}", e),
                    )
                })?;
        }

        Ok(())
    }

    /// Element-wise multiplication on GPU
    pub fn mul(
        &self,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let n = a.len();
        if b.len() != n || c.len() != n {
            return Err(crate::error::MinitensorError::shape_error(
                "Tensor dimensions must match for multiplication",
            ));
        }

        let kernel = self.backend.get_kernel("mul_kernel").ok_or_else(|| {
            crate::error::MinitensorError::backend_error("CUDA", "Mul kernel not found")
        })?;

        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;

        unsafe {
            kernel
                .launch(
                    (grid_size as u32, 1, 1),
                    (block_size as u32, 1, 1),
                    0,
                    &[
                        a.device_ptr() as *const std::ffi::c_void,
                        b.device_ptr() as *const std::ffi::c_void,
                        c.device_ptr() as *const std::ffi::c_void,
                        &(n as i32) as *const i32 as *const std::ffi::c_void,
                    ],
                )
                .map_err(|e| {
                    crate::error::MinitensorError::backend_error(
                        "CUDA",
                        format!("Kernel launch failed: {}", e),
                    )
                })?;
        }

        Ok(())
    }

    /// Matrix multiplication on GPU
    pub fn matmul(
        &self,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        if a.len() != m * k || b.len() != k * n || c.len() != m * n {
            return Err(crate::error::MinitensorError::shape_error(
                "Invalid matrix dimensions for multiplication",
            ));
        }

        let kernel = self.backend.get_kernel("matmul_kernel").ok_or_else(|| {
            crate::error::MinitensorError::backend_error("CUDA", "Matmul kernel not found")
        })?;

        let block_size_x = 16;
        let block_size_y = 16;
        let grid_size_x = (n + block_size_x - 1) / block_size_x;
        let grid_size_y = (m + block_size_y - 1) / block_size_y;

        unsafe {
            kernel
                .launch(
                    (grid_size_x as u32, grid_size_y as u32, 1),
                    (block_size_x as u32, block_size_y as u32, 1),
                    0,
                    &[
                        a.device_ptr() as *const std::ffi::c_void,
                        b.device_ptr() as *const std::ffi::c_void,
                        c.device_ptr() as *const std::ffi::c_void,
                        &(m as i32) as *const i32 as *const std::ffi::c_void,
                        &(n as i32) as *const i32 as *const std::ffi::c_void,
                        &(k as i32) as *const i32 as *const std::ffi::c_void,
                    ],
                )
                .map_err(|e| {
                    crate::error::MinitensorError::backend_error(
                        "CUDA",
                        format!("Kernel launch failed: {}", e),
                    )
                })?;
        }

        Ok(())
    }

    /// ReLU activation on GPU
    pub fn relu(&self, input: &CudaSlice<f32>, output: &mut CudaSlice<f32>) -> Result<()> {
        let n = input.len();
        if output.len() != n {
            return Err(crate::error::MinitensorError::shape_error(
                "Input and output tensors must have the same size",
            ));
        }

        let kernel = self.backend.get_kernel("relu_kernel").ok_or_else(|| {
            crate::error::MinitensorError::backend_error("CUDA", "ReLU kernel not found")
        })?;

        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;

        unsafe {
            kernel
                .launch(
                    (grid_size as u32, 1, 1),
                    (block_size as u32, 1, 1),
                    0,
                    &[
                        input.device_ptr() as *const std::ffi::c_void,
                        output.device_ptr() as *const std::ffi::c_void,
                        &(n as i32) as *const i32 as *const std::ffi::c_void,
                    ],
                )
                .map_err(|e| {
                    crate::error::MinitensorError::backend_error(
                        "CUDA",
                        format!("Kernel launch failed: {}", e),
                    )
                })?;
        }

        Ok(())
    }

    /// Sigmoid activation on GPU
    pub fn sigmoid(&self, input: &CudaSlice<f32>, output: &mut CudaSlice<f32>) -> Result<()> {
        let n = input.len();
        if output.len() != n {
            return Err(crate::error::MinitensorError::shape_error(
                "Input and output tensors must have the same size",
            ));
        }

        let kernel = self.backend.get_kernel("sigmoid_kernel").ok_or_else(|| {
            crate::error::MinitensorError::backend_error("CUDA", "Sigmoid kernel not found")
        })?;

        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;

        unsafe {
            kernel
                .launch(
                    (grid_size as u32, 1, 1),
                    (block_size as u32, 1, 1),
                    0,
                    &[
                        input.device_ptr() as *const std::ffi::c_void,
                        output.device_ptr() as *const std::ffi::c_void,
                        &(n as i32) as *const i32 as *const std::ffi::c_void,
                    ],
                )
                .map_err(|e| {
                    crate::error::MinitensorError::backend_error(
                        "CUDA",
                        format!("Kernel launch failed: {}", e),
                    )
                })?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_availability() {
        // This test will only pass if CUDA is available
        if CudaBackend::is_available() {
            let backend = CudaBackend::initialize().unwrap();
            assert!(backend.device().is_gpu());
        }
    }

    #[test]
    fn test_cuda_memory_operations() {
        if !CudaBackend::is_available() {
            return; // Skip test if CUDA not available
        }

        let backend = CudaBackend::initialize().unwrap();

        // Test allocation
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let device_data = backend.copy_to_device(&data).unwrap();

        // Test copy back
        let result = backend.copy_from_device(&device_data).unwrap();
        assert_eq!(data, result);
    }

    #[test]
    fn test_cuda_operations() {
        if !CudaBackend::is_available() {
            return; // Skip test if CUDA not available
        }

        let backend = Arc::new(CudaBackend::initialize().unwrap());
        let ops = CudaOps::new(backend.clone()).unwrap();

        // Test addition
        let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let b_data = vec![5.0f32, 6.0, 7.0, 8.0];

        let a = backend.copy_to_device(&a_data).unwrap();
        let b = backend.copy_to_device(&b_data).unwrap();
        let mut c = backend.allocate_slice::<f32>(4).unwrap();

        ops.add(&a, &b, &mut c).unwrap();
        backend.synchronize().unwrap();

        let result = backend.copy_from_device(&c).unwrap();
        let expected = vec![6.0f32, 8.0, 10.0, 12.0];

        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-6);
        }
    }
}

// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::Backend;
use crate::{device::Device, error::Result};
use metal::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Metal buffer wrapper for memory management
pub struct MetalBufferWrapper {
    buffer: metal::Buffer,
    size_bytes: usize,
}

unsafe impl Send for MetalBufferWrapper {}
unsafe impl Sync for MetalBufferWrapper {}

/// Metal backend for GPU tensor operations on Apple devices
pub struct MetalBackend {
    device: Device,
    metal_device: metal::Device,
    command_queue: metal::CommandQueue,
    library: Arc<Mutex<Option<metal::Library>>>,
    compute_pipelines: Arc<Mutex<HashMap<String, metal::ComputePipelineState>>>,
    buffers: Arc<Mutex<HashMap<usize, MetalBufferWrapper>>>,
    next_buffer_id: Arc<Mutex<usize>>,
}

impl MetalBackend {
    /// Get the Metal device
    pub fn metal_device(&self) -> &metal::Device {
        &self.metal_device
    }

    /// Get the command queue
    pub fn command_queue(&self) -> &metal::CommandQueue {
        &self.command_queue
    }

    /// Create a Metal buffer
    pub fn create_buffer(
        &self,
        length: u64,
        options: metal::MTLResourceOptions,
    ) -> Result<metal::Buffer> {
        let buffer = self.metal_device.new_buffer(length, options);
        Ok(buffer)
    }

    /// Create a Metal buffer with data
    pub fn create_buffer_with_data<T>(
        &self,
        data: &[T],
        options: metal::MTLResourceOptions,
    ) -> Result<metal::Buffer>
    where
        T: Copy,
    {
        let byte_length = (data.len() * std::mem::size_of::<T>()) as u64;
        let buffer = self.metal_device.new_buffer_with_data(
            data.as_ptr() as *const std::ffi::c_void,
            byte_length,
            options,
        );

        Ok(buffer)
    }

    /// Load Metal shaders from source
    pub fn load_library_from_source(&self, source: &str) -> Result<()> {
        let compile_options = metal::CompileOptions::new();

        let library = self
            .metal_device
            .new_library_with_source(source, &compile_options)
            .map_err(|e| {
                crate::error::MinitensorError::backend_error(format!(
                    "Failed to compile Metal library: {}",
                    e
                ))
            })?;

        let mut lib = self.library.lock().unwrap();
        *lib = Some(library);

        Ok(())
    }

    /// Create a compute pipeline state
    pub fn create_compute_pipeline(
        &self,
        function_name: &str,
    ) -> Result<metal::ComputePipelineState> {
        let library = self.library.lock().unwrap();
        let library = library.as_ref().ok_or_else(|| {
            crate::error::MinitensorError::backend_error("No Metal library loaded")
        })?;

        let function = library.get_function(function_name, None).ok_or_else(|| {
            crate::error::MinitensorError::backend_error(format!(
                "Function '{}' not found in Metal library",
                function_name
            ))
        })?;

        let pipeline_state = self
            .metal_device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| {
                crate::error::MinitensorError::backend_error(format!(
                    "Failed to create compute pipeline: {}",
                    e
                ))
            })?;

        // Cache the pipeline state
        let mut pipelines = self.compute_pipelines.lock().unwrap();
        pipelines.insert(function_name.to_string(), pipeline_state.clone());

        Ok(pipeline_state)
    }

    /// Get a cached compute pipeline state
    pub fn get_compute_pipeline(&self, function_name: &str) -> Option<metal::ComputePipelineState> {
        let pipelines = self.compute_pipelines.lock().unwrap();
        pipelines.get(function_name).cloned()
    }

    /// Execute a compute command
    pub fn execute_compute_command<F>(&self, setup: F) -> Result<()>
    where
        F: FnOnce(&metal::ComputeCommandEncoderRef) -> Result<()>,
    {
        let command_buffer = self.command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();

        setup(&compute_encoder)?;

        compute_encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    /// Copy data from buffer to host
    pub fn copy_buffer_to_host<T>(&self, buffer: &metal::Buffer, data: &mut [T]) -> Result<()>
    where
        T: Copy,
    {
        let byte_length = data.len() * std::mem::size_of::<T>();
        if buffer.length() as usize != byte_length {
            return Err(crate::error::MinitensorError::memory_error(
                "Buffer size mismatch",
            ));
        }

        unsafe {
            let contents = buffer.contents() as *const T;
            std::ptr::copy_nonoverlapping(contents, data.as_mut_ptr(), data.len());
        }

        Ok(())
    }

    /// Copy data from host to buffer
    pub fn copy_host_to_buffer<T>(&self, data: &[T], buffer: &metal::Buffer) -> Result<()>
    where
        T: Copy,
    {
        let byte_length = data.len() * std::mem::size_of::<T>();
        if buffer.length() as usize != byte_length {
            return Err(crate::error::MinitensorError::memory_error(
                "Buffer size mismatch",
            ));
        }

        unsafe {
            let contents = buffer.contents() as *mut T;
            std::ptr::copy_nonoverlapping(data.as_ptr(), contents, data.len());
        }

        Ok(())
    }

    /// Get buffer information for debugging
    pub fn get_buffer_info(&self, ptr: *const u8) -> Option<(usize, usize)> {
        let buffer_id = ptr as usize;
        let buffers = self.buffers.lock().unwrap();
        buffers
            .get(&buffer_id)
            .map(|buf| (buffer_id, buf.size_bytes))
    }

    /// Get total number of tracked buffers
    pub fn buffer_count(&self) -> usize {
        self.buffers.lock().unwrap().len()
    }

    /// Execute operation on buffers by pointer
    pub fn execute_buffer_operation<F, R>(&self, ptr: *const u8, operation: F) -> Result<R>
    where
        F: FnOnce(&metal::Buffer) -> Result<R>,
    {
        let buffer_id = ptr as usize;
        let buffers = self.buffers.lock().unwrap();
        if let Some(metal_buffer) = buffers.get(&buffer_id) {
            operation(&metal_buffer.buffer)
        } else {
            Err(crate::error::MinitensorError::memory_error(
                "Metal buffer not found for pointer",
            ))
        }
    }

    /// Get optimal thread group size for the device
    pub fn optimal_thread_group_size(
        &self,
        pipeline: &metal::ComputePipelineState,
    ) -> metal::MTLSize {
        let max_threads = pipeline.max_total_threads_per_threadgroup();
        let thread_execution_width = pipeline.thread_execution_width();

        // Use thread execution width as a good default for 1D operations
        let threads_per_group = std::cmp::min(max_threads, thread_execution_width);
        metal::MTLSize::new(threads_per_group, 1, 1)
    }

    /// Get optimal thread group size for 2D operations
    pub fn optimal_thread_group_size_2d(
        &self,
        pipeline: &metal::ComputePipelineState,
    ) -> metal::MTLSize {
        let max_threads = pipeline.max_total_threads_per_threadgroup();
        let thread_execution_width = pipeline.thread_execution_width();

        // For 2D operations, use square thread groups
        let side = (max_threads as f64).sqrt() as u64;
        let side = std::cmp::min(side, 16); // Cap at 16x16 for good occupancy
        metal::MTLSize::new(side, side, 1)
    }
}

impl Backend for MetalBackend {
    fn device(&self) -> Device {
        self.device
    }

    fn is_available() -> bool {
        // Metal is only available on Apple platforms
        #[cfg(target_os = "macos")]
        {
            metal::Device::system_default().is_some()
        }
        #[cfg(not(target_os = "macos"))]
        {
            false
        }
    }

    fn initialize() -> Result<Self> {
        #[cfg(target_os = "macos")]
        {
            let metal_device = metal::Device::system_default().ok_or_else(|| {
                crate::error::MinitensorError::backend_error("No Metal device available")
            })?;

            let command_queue = metal_device.new_command_queue();

            Ok(Self {
                device: Device::metal(Some(0)),
                metal_device,
                command_queue,
                library: Arc::new(Mutex::new(None)),
                compute_pipelines: Arc::new(Mutex::new(HashMap::new())),
                buffers: Arc::new(Mutex::new(HashMap::new())),
                next_buffer_id: Arc::new(Mutex::new(1)),
            })
        }
        #[cfg(not(target_os = "macos"))]
        {
            Err(crate::error::MinitensorError::backend_error(
                "Metal backend is only available on macOS",
            ))
        }
    }

    fn allocate(&self, size_bytes: usize) -> Result<*mut u8> {
        if size_bytes == 0 {
            return Ok(std::ptr::null_mut());
        }

        let buffer = self.create_buffer(
            size_bytes as u64,
            metal::MTLResourceOptions::StorageModeShared,
        )?;

        // Create a unique ID to track this buffer
        let buffer_id = {
            let mut next_id = self.next_buffer_id.lock().unwrap();
            let id = *next_id;
            *next_id += 1;
            id
        };

        let metal_buffer = MetalBufferWrapper { buffer, size_bytes };

        // Store the buffer for tracking
        let mut buffers = self.buffers.lock().unwrap();
        buffers.insert(buffer_id, metal_buffer);

        // Return the buffer ID as a pointer
        Ok(buffer_id as *mut u8)
    }

    fn deallocate(&self, ptr: *mut u8, _size_bytes: usize) -> Result<()> {
        if ptr.is_null() {
            return Ok(());
        }

        // Remove the buffer from tracking
        let buffer_id = ptr as usize;
        let mut buffers = self.buffers.lock().unwrap();
        buffers.remove(&buffer_id);

        // Metal buffer will be automatically dropped and freed by ARC
        Ok(())
    }

    fn copy_from_host(&self, dst: *mut u8, src: &[u8]) -> Result<()> {
        if dst.is_null() {
            return Err(crate::error::MinitensorError::memory_error(
                "Null destination pointer",
            ));
        }

        // Find the Metal buffer corresponding to this pointer
        let buffer_id = dst as usize;
        let buffers = self.buffers.lock().unwrap();
        if let Some(metal_buffer) = buffers.get(&buffer_id) {
            unsafe {
                let contents = metal_buffer.buffer.contents() as *mut u8;
                std::ptr::copy_nonoverlapping(src.as_ptr(), contents, src.len());
            }
        } else {
            return Err(crate::error::MinitensorError::memory_error(
                "Metal buffer not found for pointer",
            ));
        }

        Ok(())
    }

    fn copy_to_host(&self, dst: &mut [u8], src: *const u8) -> Result<()> {
        if src.is_null() {
            return Err(crate::error::MinitensorError::memory_error(
                "Null source pointer",
            ));
        }

        // Find the Metal buffer corresponding to this pointer
        let buffer_id = src as usize;
        let buffers = self.buffers.lock().unwrap();
        if let Some(metal_buffer) = buffers.get(&buffer_id) {
            unsafe {
                let contents = metal_buffer.buffer.contents() as *const u8;
                std::ptr::copy_nonoverlapping(contents, dst.as_mut_ptr(), dst.len());
            }
        } else {
            return Err(crate::error::MinitensorError::memory_error(
                "Metal buffer not found for pointer",
            ));
        }

        Ok(())
    }
}

/// Metal compute shaders for basic tensor operations
pub mod shaders {
    /// Element-wise addition shader
    pub const ADD_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void add_kernel(device const float* a [[buffer(0)]],
                      device const float* b [[buffer(1)]],
                      device float* c [[buffer(2)]],
                      constant uint& n [[buffer(3)]],
                      uint index [[thread_position_in_grid]]) {
    if (index >= n) return;
    c[index] = a[index] + b[index];
}
"#;

    /// Element-wise multiplication shader
    pub const MUL_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void mul_kernel(device const float* a [[buffer(0)]],
                      device const float* b [[buffer(1)]],
                      device float* c [[buffer(2)]],
                      constant uint& n [[buffer(3)]],
                      uint index [[thread_position_in_grid]]) {
    if (index >= n) return;
    c[index] = a[index] * b[index];
}
"#;

    /// Matrix multiplication shader
    pub const MATMUL_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void matmul_kernel(device const float* a [[buffer(0)]],
                         device const float* b [[buffer(1)]],
                         device float* c [[buffer(2)]],
                         constant uint& m [[buffer(3)]],
                         constant uint& n [[buffer(4)]],
                         constant uint& k [[buffer(5)]],
                         uint2 index [[thread_position_in_grid]]) {
    uint row = index.y;
    uint col = index.x;
    
    if (row >= m || col >= n) return;
    
    float sum = 0.0;
    for (uint i = 0; i < k; i++) {
        sum += a[row * k + i] * b[i * n + col];
    }
    c[row * n + col] = sum;
}
"#;

    /// ReLU activation shader
    pub const RELU_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void relu_kernel(device const float* input [[buffer(0)]],
                       device float* output [[buffer(1)]],
                       constant uint& n [[buffer(2)]],
                       uint index [[thread_position_in_grid]]) {
    if (index >= n) return;
    output[index] = max(0.0f, input[index]);
}
"#;

    /// Sigmoid activation shader
    pub const SIGMOID_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void sigmoid_kernel(device const float* input [[buffer(0)]],
                          device float* output [[buffer(1)]],
                          constant uint& n [[buffer(2)]],
                          uint index [[thread_position_in_grid]]) {
    if (index >= n) return;
    output[index] = 1.0f / (1.0f + exp(-input[index]));
}
"#;

    /// Combined shader source
    pub const ALL_SHADERS: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void add_kernel(device const float* a [[buffer(0)]],
                      device const float* b [[buffer(1)]],
                      device float* c [[buffer(2)]],
                      constant uint& n [[buffer(3)]],
                      uint index [[thread_position_in_grid]]) {
    if (index >= n) return;
    c[index] = a[index] + b[index];
}

kernel void mul_kernel(device const float* a [[buffer(0)]],
                      device const float* b [[buffer(1)]],
                      device float* c [[buffer(2)]],
                      constant uint& n [[buffer(3)]],
                      uint index [[thread_position_in_grid]]) {
    if (index >= n) return;
    c[index] = a[index] * b[index];
}

kernel void matmul_kernel(device const float* a [[buffer(0)]],
                         device const float* b [[buffer(1)]],
                         device float* c [[buffer(2)]],
                         constant uint& m [[buffer(3)]],
                         constant uint& n [[buffer(4)]],
                         constant uint& k [[buffer(5)]],
                         uint2 index [[thread_position_in_grid]]) {
    uint row = index.y;
    uint col = index.x;
    
    if (row >= m || col >= n) return;
    
    float sum = 0.0;
    for (uint i = 0; i < k; i++) {
        sum += a[row * k + i] * b[i * n + col];
    }
    c[row * n + col] = sum;
}

kernel void relu_kernel(device const float* input [[buffer(0)]],
                       device float* output [[buffer(1)]],
                       constant uint& n [[buffer(2)]],
                       uint index [[thread_position_in_grid]]) {
    if (index >= n) return;
    output[index] = max(0.0f, input[index]);
}

kernel void sigmoid_kernel(device const float* input [[buffer(0)]],
                          device float* output [[buffer(1)]],
                          constant uint& n [[buffer(2)]],
                          uint index [[thread_position_in_grid]]) {
    if (index >= n) return;
    output[index] = 1.0f / (1.0f + exp(-input[index]));
}
"#;
}

/// Metal operations for tensor computations
pub struct MetalOps {
    backend: Arc<MetalBackend>,
}

impl MetalOps {
    /// Create new Metal operations instance
    pub fn new(backend: Arc<MetalBackend>) -> Result<Self> {
        // Load all shaders
        backend.load_library_from_source(shaders::ALL_SHADERS)?;

        // Create compute pipelines
        backend.create_compute_pipeline("add_kernel")?;
        backend.create_compute_pipeline("mul_kernel")?;
        backend.create_compute_pipeline("matmul_kernel")?;
        backend.create_compute_pipeline("relu_kernel")?;
        backend.create_compute_pipeline("sigmoid_kernel")?;

        Ok(Self { backend })
    }

    /// Element-wise addition on GPU
    pub fn add(
        &self,
        a: &metal::Buffer,
        b: &metal::Buffer,
        c: &metal::Buffer,
        n: u32,
    ) -> Result<()> {
        let pipeline = self
            .backend
            .get_compute_pipeline("add_kernel")
            .ok_or_else(|| {
                crate::error::MinitensorError::backend_error("Add pipeline not found")
            })?;

        self.backend.execute_compute_command(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(a), 0);
            encoder.set_buffer(1, Some(b), 0);
            encoder.set_buffer(2, Some(c), 0);
            encoder.set_bytes(
                3,
                std::mem::size_of::<u32>() as u64,
                &n as *const u32 as *const std::ffi::c_void,
            );

            let thread_group_size = self.backend.optimal_thread_group_size(&pipeline);
            let threads_per_group = thread_group_size.width;
            let thread_group_count =
                metal::MTLSize::new((n as u64 + threads_per_group - 1) / threads_per_group, 1, 1);

            encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
            Ok(())
        })
    }

    /// Element-wise multiplication on GPU
    pub fn mul(
        &self,
        a: &metal::Buffer,
        b: &metal::Buffer,
        c: &metal::Buffer,
        n: u32,
    ) -> Result<()> {
        let pipeline = self
            .backend
            .get_compute_pipeline("mul_kernel")
            .ok_or_else(|| {
                crate::error::MinitensorError::backend_error("Mul pipeline not found")
            })?;

        self.backend.execute_compute_command(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(a), 0);
            encoder.set_buffer(1, Some(b), 0);
            encoder.set_buffer(2, Some(c), 0);
            encoder.set_bytes(
                3,
                std::mem::size_of::<u32>() as u64,
                &n as *const u32 as *const std::ffi::c_void,
            );

            let thread_group_count = metal::MTLSize::new((n as u64 + 255) / 256, 1, 1);
            let thread_group_size = metal::MTLSize::new(256, 1, 1);

            encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
            Ok(())
        })
    }

    /// Matrix multiplication on GPU
    pub fn matmul(
        &self,
        a: &metal::Buffer,
        b: &metal::Buffer,
        c: &metal::Buffer,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<()> {
        let pipeline = self
            .backend
            .get_compute_pipeline("matmul_kernel")
            .ok_or_else(|| {
                crate::error::MinitensorError::backend_error("Matmul pipeline not found")
            })?;

        self.backend.execute_compute_command(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(a), 0);
            encoder.set_buffer(1, Some(b), 0);
            encoder.set_buffer(2, Some(c), 0);
            encoder.set_bytes(
                3,
                std::mem::size_of::<u32>() as u64,
                &m as *const u32 as *const std::ffi::c_void,
            );
            encoder.set_bytes(
                4,
                std::mem::size_of::<u32>() as u64,
                &n as *const u32 as *const std::ffi::c_void,
            );
            encoder.set_bytes(
                5,
                std::mem::size_of::<u32>() as u64,
                &k as *const u32 as *const std::ffi::c_void,
            );

            let thread_group_count =
                metal::MTLSize::new((n as u64 + 15) / 16, (m as u64 + 15) / 16, 1);
            let thread_group_size = metal::MTLSize::new(16, 16, 1);

            encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
            Ok(())
        })
    }

    /// ReLU activation on GPU
    pub fn relu(&self, input: &metal::Buffer, output: &metal::Buffer, n: u32) -> Result<()> {
        let pipeline = self
            .backend
            .get_compute_pipeline("relu_kernel")
            .ok_or_else(|| {
                crate::error::MinitensorError::backend_error("ReLU pipeline not found")
            })?;

        self.backend.execute_compute_command(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(input), 0);
            encoder.set_buffer(1, Some(output), 0);
            encoder.set_bytes(
                2,
                std::mem::size_of::<u32>() as u64,
                &n as *const u32 as *const std::ffi::c_void,
            );

            let thread_group_count = metal::MTLSize::new((n as u64 + 255) / 256, 1, 1);
            let thread_group_size = metal::MTLSize::new(256, 1, 1);

            encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
            Ok(())
        })
    }

    /// Sigmoid activation on GPU
    pub fn sigmoid(&self, input: &metal::Buffer, output: &metal::Buffer, n: u32) -> Result<()> {
        let pipeline = self
            .backend
            .get_compute_pipeline("sigmoid_kernel")
            .ok_or_else(|| {
                crate::error::MinitensorError::backend_error("Sigmoid pipeline not found")
            })?;

        self.backend.execute_compute_command(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(input), 0);
            encoder.set_buffer(1, Some(output), 0);
            encoder.set_bytes(
                2,
                std::mem::size_of::<u32>() as u64,
                &n as *const u32 as *const std::ffi::c_void,
            );

            let thread_group_count = metal::MTLSize::new((n as u64 + 255) / 256, 1, 1);
            let thread_group_size = metal::MTLSize::new(256, 1, 1);

            encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
            Ok(())
        })
    }

    /// Execute element-wise addition using pointers
    pub fn add_ptr(
        &self,
        a_ptr: *const u8,
        b_ptr: *const u8,
        c_ptr: *mut u8,
        n: u32,
    ) -> Result<()> {
        let a_buffer_id = a_ptr as usize;
        let b_buffer_id = b_ptr as usize;
        let c_buffer_id = c_ptr as usize;

        let buffers = self.backend.buffers.lock().unwrap();

        if let (Some(a_buf), Some(b_buf), Some(c_buf)) = (
            buffers.get(&a_buffer_id),
            buffers.get(&b_buffer_id),
            buffers.get(&c_buffer_id),
        ) {
            self.add(&a_buf.buffer, &b_buf.buffer, &c_buf.buffer, n)
        } else {
            Err(crate::error::MinitensorError::memory_error(
                "Metal buffer not found for operation",
            ))
        }
    }

    /// Execute element-wise multiplication using pointers
    pub fn mul_ptr(
        &self,
        a_ptr: *const u8,
        b_ptr: *const u8,
        c_ptr: *mut u8,
        n: u32,
    ) -> Result<()> {
        let a_buffer_id = a_ptr as usize;
        let b_buffer_id = b_ptr as usize;
        let c_buffer_id = c_ptr as usize;

        let buffers = self.backend.buffers.lock().unwrap();

        if let (Some(a_buf), Some(b_buf), Some(c_buf)) = (
            buffers.get(&a_buffer_id),
            buffers.get(&b_buffer_id),
            buffers.get(&c_buffer_id),
        ) {
            self.mul(&a_buf.buffer, &b_buf.buffer, &c_buf.buffer, n)
        } else {
            Err(crate::error::MinitensorError::memory_error(
                "Metal buffer not found for operation",
            ))
        }
    }

    /// Execute matrix multiplication using pointers
    pub fn matmul_ptr(
        &self,
        a_ptr: *const u8,
        b_ptr: *const u8,
        c_ptr: *mut u8,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<()> {
        let a_buffer_id = a_ptr as usize;
        let b_buffer_id = b_ptr as usize;
        let c_buffer_id = c_ptr as usize;

        let buffers = self.backend.buffers.lock().unwrap();

        if let (Some(a_buf), Some(b_buf), Some(c_buf)) = (
            buffers.get(&a_buffer_id),
            buffers.get(&b_buffer_id),
            buffers.get(&c_buffer_id),
        ) {
            self.matmul(&a_buf.buffer, &b_buf.buffer, &c_buf.buffer, m, n, k)
        } else {
            Err(crate::error::MinitensorError::memory_error(
                "Metal buffer not found for operation",
            ))
        }
    }

    /// Execute ReLU activation using pointers
    pub fn relu_ptr(&self, input_ptr: *const u8, output_ptr: *mut u8, n: u32) -> Result<()> {
        let input_buffer_id = input_ptr as usize;
        let output_buffer_id = output_ptr as usize;

        let buffers = self.backend.buffers.lock().unwrap();

        if let (Some(input_buf), Some(output_buf)) = (
            buffers.get(&input_buffer_id),
            buffers.get(&output_buffer_id),
        ) {
            self.relu(&input_buf.buffer, &output_buf.buffer, n)
        } else {
            Err(crate::error::MinitensorError::memory_error(
                "Metal buffer not found for operation",
            ))
        }
    }

    /// Execute Sigmoid activation using pointers
    pub fn sigmoid_ptr(&self, input_ptr: *const u8, output_ptr: *mut u8, n: u32) -> Result<()> {
        let input_buffer_id = input_ptr as usize;
        let output_buffer_id = output_ptr as usize;

        let buffers = self.backend.buffers.lock().unwrap();

        if let (Some(input_buf), Some(output_buf)) = (
            buffers.get(&input_buffer_id),
            buffers.get(&output_buffer_id),
        ) {
            self.sigmoid(&input_buf.buffer, &output_buf.buffer, n)
        } else {
            Err(crate::error::MinitensorError::memory_error(
                "Metal buffer not found for operation",
            ))
        }
    }
}

#[cfg(test)]
mod integration_test;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_availability() {
        // This test will only pass on macOS with Metal support
        #[cfg(target_os = "macos")]
        {
            if MetalBackend::is_available() {
                let backend = MetalBackend::initialize().unwrap();
                assert!(backend.device().is_gpu());
            }
        }
    }

    #[test]
    fn test_metal_buffer_operations() {
        #[cfg(target_os = "macos")]
        {
            if !MetalBackend::is_available() {
                return; // Skip test if Metal not available
            }

            let backend = MetalBackend::initialize().unwrap();

            // Test buffer creation and data transfer
            let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
            let buffer = backend
                .create_buffer_with_data(&data, metal::MTLResourceOptions::StorageModeShared)
                .unwrap();

            let mut result = vec![0.0f32; 5];
            backend.copy_buffer_to_host(&buffer, &mut result).unwrap();

            assert_eq!(data, result);
        }
    }

    #[test]
    fn test_metal_operations() {
        #[cfg(target_os = "macos")]
        {
            if !MetalBackend::is_available() {
                return; // Skip test if Metal not available
            }

            let backend = Arc::new(MetalBackend::initialize().unwrap());
            let ops = MetalOps::new(backend.clone()).unwrap();

            // Test addition
            let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
            let b_data = vec![5.0f32, 6.0, 7.0, 8.0];

            let a_buffer = backend
                .create_buffer_with_data(&a_data, metal::MTLResourceOptions::StorageModeShared)
                .unwrap();
            let b_buffer = backend
                .create_buffer_with_data(&b_data, metal::MTLResourceOptions::StorageModeShared)
                .unwrap();
            let c_buffer = backend
                .create_buffer(16, metal::MTLResourceOptions::StorageModeShared)
                .unwrap();

            ops.add(&a_buffer, &b_buffer, &c_buffer, 4).unwrap();

            let mut result = vec![0.0f32; 4];
            backend.copy_buffer_to_host(&c_buffer, &mut result).unwrap();

            let expected = vec![6.0f32, 8.0, 10.0, 12.0];
            for (r, e) in result.iter().zip(expected.iter()) {
                assert!((r - e).abs() < 1e-6);
            }
        }
    }
}

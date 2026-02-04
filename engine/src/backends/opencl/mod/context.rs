// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::Backend;
use crate::{device::Device, error::Result};
use opencl3::command_queue::{CL_QUEUE_PROFILING_ENABLE, CommandQueue};
use opencl3::context::Context;
use opencl3::device::{CL_DEVICE_TYPE_GPU, Device as OpenCLDevice};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_WRITE};
use opencl3::platform::get_platforms;
use opencl3::program::Program;
use opencl3::types::{CL_BLOCKING, cl_float};
use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// OpenCL buffer wrapper for memory management
pub struct OpenCLBuffer {
    buffer: Buffer<cl_float>,
    size_bytes: usize,
}

unsafe impl Send for OpenCLBuffer {}
unsafe impl Sync for OpenCLBuffer {}

/// OpenCL backend for cross-platform GPU tensor operations
pub struct OpenCLBackend {
    device: Device,
    opencl_device: OpenCLDevice,
    context: Context,
    command_queue: CommandQueue,
    programs: Arc<RwLock<FxHashMap<String, Program>>>,
    kernels: Arc<RwLock<FxHashMap<String, Kernel>>>,
    buffers: Arc<RwLock<FxHashMap<usize, OpenCLBuffer>>>,
    buffer_pool: Arc<RwLock<FxHashMap<usize, Vec<Buffer<cl_float>>>>>,
    next_buffer_id: AtomicUsize,
}

unsafe impl Send for OpenCLBackend {}
unsafe impl Sync for OpenCLBackend {}

impl OpenCLBackend {
    /// Get the OpenCL device
    #[inline(always)]
    pub fn opencl_device(&self) -> &OpenCLDevice {
        &self.opencl_device
    }

    /// Get the OpenCL context
    #[inline(always)]
    pub fn context(&self) -> &Context {
        &self.context
    }

    /// Get the command queue
    #[inline(always)]
    pub fn command_queue(&self) -> &CommandQueue {
        &self.command_queue
    }

    /// Create an OpenCL buffer
    #[inline(always)]
    pub fn create_buffer(&self, size: usize, flags: u64) -> Result<Buffer<cl_float>> {
        let buffer =
            unsafe { Buffer::<cl_float>::create(&self.context, flags, size, ptr::null_mut()) }
                .map_err(|e| {
                    crate::error::MinitensorError::memory_error(format!(
                        "Failed to create OpenCL buffer: {}",
                        e
                    ))
                })?;

        Ok(buffer)
    }

    /// Create an OpenCL buffer with data
    #[inline(always)]
    pub fn create_buffer_with_data(&self, data: &[f32], flags: u64) -> Result<Buffer<cl_float>> {
        let mut buffer = unsafe {
            Buffer::<cl_float>::create(&self.context, flags, data.len(), ptr::null_mut())
        }
        .map_err(|e| {
            crate::error::MinitensorError::memory_error(format!(
                "Failed to create OpenCL buffer: {}",
                e
            ))
        })?;

        // Write data to buffer
        unsafe {
            self.command_queue
                .enqueue_write_buffer(&mut buffer, CL_BLOCKING, 0, data, &[])
                .map_err(|e| {
                    crate::error::MinitensorError::memory_error(format!(
                        "Failed to write to OpenCL buffer: {}",
                        e
                    ))
                })?;
        }

        Ok(buffer)
    }

    /// Build an OpenCL program
    #[inline(always)]
    pub fn build_program(&self, name: &str, source: &str) -> Result<()> {
        let program =
            Program::create_and_build_from_source(&self.context, source, "").map_err(|e| {
                crate::error::MinitensorError::backend_error(
                    "OpenCL",
                    format!("Failed to build OpenCL program: {}", e),
                )
            })?;

        let mut programs = self.programs.write();
        programs.insert(name.to_string(), program);

        Ok(())
    }

    /// Create a kernel from a program
    #[inline(always)]
    pub fn create_kernel(&self, program_name: &str, kernel_name: &str) -> Result<()> {
        let programs = self.programs.read();
        let program = programs.get(program_name).ok_or_else(|| {
            crate::error::MinitensorError::backend_error(
                "OpenCL",
                format!("Program '{}' not found", program_name),
            )
        })?;

        let kernel = Kernel::create(program, kernel_name).map_err(|e| {
            crate::error::MinitensorError::backend_error(
                "OpenCL",
                format!("Failed to create kernel '{}': {}", kernel_name, e),
            )
        })?;

        let mut kernels = self.kernels.write();
        kernels.insert(kernel_name.to_string(), kernel);

        Ok(())
    }

    /// Get a kernel (creates a new kernel instance to avoid borrowing issues)
    #[inline(always)]
    pub fn get_kernel(&self, kernel_name: &str) -> Option<Kernel> {
        let programs = self.programs.read();
        if let Some(program) = programs.get("tensor_ops") {
            Kernel::create(program, kernel_name).ok()
        } else {
            None
        }
    }

    /// Execute a kernel
    #[inline(always)]
    pub fn execute_kernel(
        &self,
        kernel_name: &str,
        global_work_size: &[usize],
        local_work_size: Option<&[usize]>,
    ) -> Result<()> {
        let kernel = self.get_kernel(kernel_name).ok_or_else(|| {
            crate::error::MinitensorError::backend_error(
                "OpenCL",
                format!("Kernel '{}' not found", kernel_name),
            )
        })?;

        let kernel_event = unsafe {
            ExecuteKernel::new(&kernel)
                .set_global_work_sizes(global_work_size)
                .set_local_work_sizes(local_work_size.unwrap_or(&[]))
                .enqueue_nd_range(&self.command_queue)
        }
        .map_err(|e| {
            crate::error::MinitensorError::backend_error(
                "OpenCL",
                format!("Failed to execute kernel: {}", e),
            )
        })?;

        kernel_event.wait().map_err(|e| {
            crate::error::MinitensorError::backend_error(
                "OpenCL",
                format!("Failed to wait for kernel completion: {}", e),
            )
        })?;

        Ok(())
    }

    /// Read data from buffer
    #[inline(always)]
    pub fn read_buffer(&self, buffer: &Buffer<cl_float>, data: &mut [f32]) -> Result<()> {
        unsafe {
            self.command_queue
                .enqueue_read_buffer(buffer, CL_BLOCKING, 0, data, &[])
                .map_err(|e| {
                    crate::error::MinitensorError::memory_error(format!(
                        "Failed to read from OpenCL buffer: {}",
                        e
                    ))
                })?;
        }

        Ok(())
    }

    /// Write data to buffer
    #[inline(always)]
    pub fn write_buffer(&self, buffer: &mut Buffer<cl_float>, data: &[f32]) -> Result<()> {
        unsafe {
            self.command_queue
                .enqueue_write_buffer(buffer, CL_BLOCKING, 0, data, &[])
                .map_err(|e| {
                    crate::error::MinitensorError::memory_error(format!(
                        "Failed to write to OpenCL buffer: {}",
                        e
                    ))
                })?;
        }

        Ok(())
    }

    /// Execute operation on buffers by pointer
    #[inline(always)]
    pub fn execute_buffer_operation<F, R>(&self, ptr: *const u8, operation: F) -> Result<R>
    where
        F: FnOnce(&Buffer<cl_float>) -> Result<R>,
    {
        let buffer_id = ptr as usize;
        let buffers = self.buffers.read();
        if let Some(opencl_buffer) = buffers.get(&buffer_id) {
            operation(&opencl_buffer.buffer)
        } else {
            Err(crate::error::MinitensorError::memory_error(
                "OpenCL buffer not found for pointer",
            ))
        }
    }

    /// Get buffer information for debugging
    #[inline(always)]
    pub fn get_buffer_info(&self, ptr: *const u8) -> Option<(usize, usize)> {
        let buffer_id = ptr as usize;
        let buffers = self.buffers.read();
        buffers
            .get(&buffer_id)
            .map(|buf| (buffer_id, buf.size_bytes))
    }

    /// Get total number of tracked buffers
    #[inline(always)]
    pub fn buffer_count(&self) -> usize {
        self.buffers.read().len()
    }

    /// Finish all operations in the command queue
    #[inline(always)]
    pub fn finish(&self) -> Result<()> {
        self.command_queue.finish().map_err(|e| {
            crate::error::MinitensorError::backend_error(
                "OpenCL",
                format!("Failed to finish OpenCL operations: {}", e),
            )
        })
    }
}

impl Backend for OpenCLBackend {
    #[inline(always)]
    fn device(&self) -> Device {
        self.device
    }

    #[inline(always)]
    fn is_available() -> bool {
        // Check if OpenCL platforms and GPU devices are available
        if let Ok(platforms) = get_platforms() {
            for platform in platforms {
                if let Ok(devices) =
                    opencl3::device::get_device_ids(platform.id(), CL_DEVICE_TYPE_GPU)
                {
                    if !devices.is_empty() {
                        return true;
                    }
                }
            }
        }
        false
    }

    #[inline(always)]
    fn initialize() -> Result<Self> {
        // Get the first available GPU device
        let platforms = get_platforms().map_err(|e| {
            crate::error::MinitensorError::backend_error(
                "OpenCL",
                format!("Failed to get OpenCL platforms: {}", e),
            )
        })?;

        let mut all_devices = Vec::new();
        for platform in platforms {
            if let Ok(platform_devices) =
                opencl3::device::get_device_ids(platform.id(), CL_DEVICE_TYPE_GPU)
            {
                all_devices.extend(platform_devices);
            }
        }
        let devices = all_devices;

        if devices.is_empty() {
            return Err(crate::error::MinitensorError::backend_error(
                "OpenCL",
                "No OpenCL GPU device found",
            ));
        }

        let opencl_device_id = devices[0];
        let opencl_device = opencl3::device::Device::new(opencl_device_id);

        // Create context and command queue
        let context = Context::from_device(&opencl_device).map_err(|e| {
            crate::error::MinitensorError::backend_error(
                "OpenCL",
                format!("Failed to create OpenCL context: {}", e),
            )
        })?;

        #[allow(deprecated)]
        let command_queue = CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE)
            .map_err(|e| {
                crate::error::MinitensorError::backend_error(
                    "OpenCL",
                    format!("Failed to create OpenCL command queue: {}", e),
                )
            })?;

        Ok(Self {
            device: Device::opencl(Some(0)),
            opencl_device,
            context,
            command_queue,
            programs: Arc::new(RwLock::new(FxHashMap::default())),
            kernels: Arc::new(RwLock::new(FxHashMap::default())),
            buffers: Arc::new(RwLock::new(FxHashMap::default())),
            buffer_pool: Arc::new(RwLock::new(FxHashMap::default())),
            next_buffer_id: AtomicUsize::new(1),
        })
    }

    #[inline(always)]
    fn allocate(&self, size_bytes: usize) -> Result<*mut u8> {
        if size_bytes == 0 {
            return Ok(std::ptr::null_mut());
        }

        let buffer = {
            let mut pool = self.buffer_pool.write();
            if let Some(buf) = pool.get_mut(&size_bytes).and_then(|v| v.pop()) {
                buf
            } else {
                let size_floats =
                    (size_bytes + std::mem::size_of::<f32>() - 1) / std::mem::size_of::<f32>();
                drop(pool);
                self.create_buffer(size_floats, CL_MEM_READ_WRITE)?
            }
        };

        // Create a unique ID to track this buffer
        let buffer_id = self.next_buffer_id.fetch_add(1, Ordering::Relaxed);

        let opencl_buffer = OpenCLBuffer { buffer, size_bytes };

        // Store the buffer for tracking
        let mut buffers = self.buffers.write();
        buffers.insert(buffer_id, opencl_buffer);

        // Return the buffer ID as a pointer
        Ok(buffer_id as *mut u8)
    }

    #[inline(always)]
    fn deallocate(&self, ptr: *mut u8, _size_bytes: usize) -> Result<()> {
        if ptr.is_null() {
            return Ok(());
        }

        // Remove the buffer from tracking and return to pool
        let buffer_id = ptr as usize;
        let mut buffers = self.buffers.write();
        if let Some(opencl_buffer) = buffers.remove(&buffer_id) {
            let mut pool = self.buffer_pool.write();
            pool.entry(opencl_buffer.size_bytes)
                .or_default()
                .push(opencl_buffer.buffer);
        }

        Ok(())
    }

    #[inline(always)]
    fn copy_from_host(&self, dst: *mut u8, src: &[u8]) -> Result<()> {
        if src.is_empty() {
            return Ok(());
        }
        if dst.is_null() {
            return Err(crate::error::MinitensorError::memory_error(
                "Null destination pointer",
            ));
        }

        // Find the OpenCL buffer corresponding to this pointer
        let buffer_id = dst as usize;
        let mut buffers = self.buffers.write();
        if let Some(opencl_buffer) = buffers.get_mut(&buffer_id) {
            // Convert bytes to f32 for OpenCL buffer
            let src_floats = unsafe {
                std::slice::from_raw_parts(
                    src.as_ptr() as *const f32,
                    src.len() / std::mem::size_of::<f32>(),
                )
            };

            unsafe {
                self.command_queue.enqueue_write_buffer(
                    &mut opencl_buffer.buffer,
                    CL_BLOCKING,
                    0,
                    src_floats,
                    &[],
                )
            }
            .map_err(|e| {
                crate::error::MinitensorError::memory_error(format!(
                    "Failed to copy data to OpenCL buffer: {}",
                    e
                ))
            })?;
        } else {
            return Err(crate::error::MinitensorError::memory_error(
                "OpenCL buffer not found for pointer",
            ));
        }

        Ok(())
    }

    #[inline(always)]
    fn copy_to_host(&self, dst: &mut [u8], src: *const u8) -> Result<()> {
        if dst.is_empty() {
            return Ok(());
        }
        if src.is_null() {
            return Err(crate::error::MinitensorError::memory_error(
                "Null source pointer",
            ));
        }

        // Find the OpenCL buffer corresponding to this pointer
        let buffer_id = src as usize;
        let buffers = self.buffers.read();
        if let Some(opencl_buffer) = buffers.get(&buffer_id) {
            // Convert bytes to f32 for OpenCL buffer
            let dst_floats = unsafe {
                std::slice::from_raw_parts_mut(
                    dst.as_mut_ptr() as *mut f32,
                    dst.len() / std::mem::size_of::<f32>(),
                )
            };

            unsafe {
                self.command_queue.enqueue_read_buffer(
                    &opencl_buffer.buffer,
                    CL_BLOCKING,
                    0,
                    dst_floats,
                    &[],
                )
            }
            .map_err(|e| {
                crate::error::MinitensorError::memory_error(format!(
                    "Failed to copy data from OpenCL buffer: {}",
                    e
                ))
            })?;
        } else {
            return Err(crate::error::MinitensorError::memory_error(
                "OpenCL buffer not found for pointer",
            ));
        }

        Ok(())
    }
}

impl Drop for OpenCLBackend {
    fn drop(&mut self) {
        {
            let mut buffers = self.buffers.write();
            for (_, buf) in buffers.drain() {
                drop(buf);
            }
        }
        let mut pool = self.buffer_pool.write();
        for (_, mut vec) in pool.drain() {
            for buf in vec.drain(..) {
                drop(buf);
            }
        }
    }
}

/// OpenCL kernel source code for basic tensor operations
pub mod kernels {
    /// Element-wise addition kernel
    pub const ADD_KERNEL: &str = r#"
__kernel void add_kernel(__global const float* a,
                        __global const float* b,
                        __global float* c,
                        const unsigned int n) {
    int gid = get_global_id(0);
    if (gid < n) {
        c[gid] = a[gid] + b[gid];
    }
}
"#;

    /// Element-wise multiplication kernel
    pub const MUL_KERNEL: &str = r#"
__kernel void mul_kernel(__global const float* a,
                        __global const float* b,
                        __global float* c,
                        const unsigned int n) {
    int gid = get_global_id(0);
    if (gid < n) {
        c[gid] = a[gid] * b[gid];
    }
}
"#;

    /// Matrix multiplication kernel
    pub const MATMUL_KERNEL: &str = r#"
__kernel void matmul_kernel(__global const float* a,
                           __global const float* b,
                           __global float* c,
                           const unsigned int m,
                           const unsigned int n,
                           const unsigned int k) {
    int row = get_global_id(1);
    int col = get_global_id(0);

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
__kernel void relu_kernel(__global const float* input,
                         __global float* output,
                         const unsigned int n) {
    int gid = get_global_id(0);
    if (gid < n) {
        output[gid] = fmax(0.0f, input[gid]);
    }
}
"#;

    /// Sigmoid activation kernel
    pub const SIGMOID_KERNEL: &str = r#"
__kernel void sigmoid_kernel(__global const float* input,
                            __global float* output,
                            const unsigned int n) {
    int gid = get_global_id(0);
    if (gid < n) {
        output[gid] = 1.0f / (1.0f + exp(-input[gid]));
    }
}
"#;

    /// Combined kernel source
    pub const ALL_KERNELS: &str = r#"
__kernel void add_kernel(__global const float* a,
                        __global const float* b,
                        __global float* c,
                        const unsigned int n) {
    int gid = get_global_id(0);
    if (gid < n) {
        c[gid] = a[gid] + b[gid];
    }
}

__kernel void mul_kernel(__global const float* a,
                        __global const float* b,
                        __global float* c,
                        const unsigned int n) {
    int gid = get_global_id(0);
    if (gid < n) {
        c[gid] = a[gid] * b[gid];
    }
}

__kernel void matmul_kernel(__global const float* a,
                           __global const float* b,
                           __global float* c,
                           const unsigned int m,
                           const unsigned int n,
                           const unsigned int k) {
    int row = get_global_id(1);
    int col = get_global_id(0);

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

__kernel void relu_kernel(__global const float* input,
                         __global float* output,
                         const unsigned int n) {
    int gid = get_global_id(0);
    if (gid < n) {
        output[gid] = fmax(0.0f, input[gid]);
    }
}

__kernel void sigmoid_kernel(__global const float* input,
                            __global float* output,
                            const unsigned int n) {
    int gid = get_global_id(0);
    if (gid < n) {
        output[gid] = 1.0f / (1.0f + exp(-input[gid]));
    }
}
"#;
}

/// OpenCL operations for tensor computations
pub struct OpenCLOps {
    backend: Arc<OpenCLBackend>,
}

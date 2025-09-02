// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

use super::Backend;
use crate::{device::Device, error::Result};
use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
use opencl3::context::Context;
use opencl3::device::{Device as OpenCLDevice, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_WRITE};
use opencl3::platform::get_platforms;
use opencl3::program::Program;
use opencl3::types::{cl_float, CL_BLOCKING};
use std::collections::HashMap;
use std::ptr;
use std::sync::{Arc, Mutex};

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
    programs: Arc<Mutex<HashMap<String, Program>>>,
    kernels: Arc<Mutex<HashMap<String, Kernel>>>,
    buffers: Arc<Mutex<HashMap<usize, OpenCLBuffer>>>,
    next_buffer_id: Arc<Mutex<usize>>,
}

impl OpenCLBackend {
    /// Get the OpenCL device
    pub fn opencl_device(&self) -> &OpenCLDevice {
        &self.opencl_device
    }

    /// Get the OpenCL context
    pub fn context(&self) -> &Context {
        &self.context
    }

    /// Get the command queue
    pub fn command_queue(&self) -> &CommandQueue {
        &self.command_queue
    }

    /// Create an OpenCL buffer
    pub fn create_buffer(&self, size: usize, flags: u64) -> Result<Buffer<cl_float>> {
        let buffer = unsafe {
            Buffer::<cl_float>::create(&self.context, flags, size, ptr::null_mut())
        }.map_err(|e| crate::error::MinitensorError::memory_error(format!("Failed to create OpenCL buffer: {}", e)))?;

        Ok(buffer)
    }

    /// Create an OpenCL buffer with data
    pub fn create_buffer_with_data(&self, data: &[f32], flags: u64) -> Result<Buffer<cl_float>> {
        let mut buffer = unsafe {
            Buffer::<cl_float>::create(&self.context, flags, data.len(), ptr::null_mut())
        }.map_err(|e| crate::error::MinitensorError::memory_error(format!("Failed to create OpenCL buffer: {}", e)))?;

        // Write data to buffer
        unsafe {
            self.command_queue
                .enqueue_write_buffer(&mut buffer, CL_BLOCKING, 0, data, &[])
                .map_err(|e| crate::error::MinitensorError::memory_error(format!("Failed to write to OpenCL buffer: {}", e)))?;
        }

        Ok(buffer)
    }

    /// Build an OpenCL program from source
    pub fn build_program(&self, name: &str, source: &str) -> Result<()> {
        let program = Program::create_and_build_from_source(&self.context, source, "")
            .map_err(|e| crate::error::MinitensorError::backend_error(format!("Failed to build OpenCL program: {}", e)))?;

        let mut programs = self.programs.lock().unwrap();
        programs.insert(name.to_string(), program);

        Ok(())
    }

    /// Create a kernel from a program
    pub fn create_kernel(&self, program_name: &str, kernel_name: &str) -> Result<()> {
        let programs = self.programs.lock().unwrap();
        let program = programs.get(program_name)
            .ok_or_else(|| crate::error::MinitensorError::backend_error(format!("Program '{}' not found", program_name)))?;

        let kernel = Kernel::create(program, kernel_name)
            .map_err(|e| crate::error::MinitensorError::backend_error(format!("Failed to create kernel '{}': {}", kernel_name, e)))?;

        let mut kernels = self.kernels.lock().unwrap();
        kernels.insert(kernel_name.to_string(), kernel);

        Ok(())
    }

    /// Get a kernel (creates a new kernel instance to avoid borrowing issues)
    pub fn get_kernel(&self, kernel_name: &str) -> Option<Kernel> {
        let programs = self.programs.lock().unwrap();
        if let Some(program) = programs.get("tensor_ops") {
            Kernel::create(program, kernel_name).ok()
        } else {
            None
        }
    }

    /// Execute a kernel
    pub fn execute_kernel(&self, kernel_name: &str, global_work_size: &[usize], local_work_size: Option<&[usize]>) -> Result<()> {
        let kernel = self.get_kernel(kernel_name)
            .ok_or_else(|| crate::error::MinitensorError::backend_error(format!("Kernel '{}' not found", kernel_name)))?;

        let kernel_event = unsafe {
            ExecuteKernel::new(&kernel)
                .set_global_work_sizes(global_work_size)
                .set_local_work_sizes(local_work_size.unwrap_or(&[]))
                .enqueue_nd_range(&self.command_queue)
        }.map_err(|e| crate::error::MinitensorError::backend_error(format!("Failed to execute kernel: {}", e)))?;

        kernel_event.wait()
            .map_err(|e| crate::error::MinitensorError::backend_error(format!("Failed to wait for kernel completion: {}", e)))?;

        Ok(())
    }

    /// Read data from buffer
    pub fn read_buffer(&self, buffer: &Buffer<cl_float>, data: &mut [f32]) -> Result<()> {
        unsafe {
            self.command_queue
                .enqueue_read_buffer(buffer, CL_BLOCKING, 0, data, &[])
                .map_err(|e| crate::error::MinitensorError::memory_error(format!("Failed to read from OpenCL buffer: {}", e)))?;
        }

        Ok(())
    }

    /// Write data to buffer
    pub fn write_buffer(&self, buffer: &mut Buffer<cl_float>, data: &[f32]) -> Result<()> {
        unsafe {
            self.command_queue
                .enqueue_write_buffer(buffer, CL_BLOCKING, 0, data, &[])
                .map_err(|e| crate::error::MinitensorError::memory_error(format!("Failed to write to OpenCL buffer: {}", e)))?;
        }

        Ok(())
    }

    /// Execute operation on buffers by pointer
    pub fn execute_buffer_operation<F, R>(&self, ptr: *const u8, operation: F) -> Result<R>
    where
        F: FnOnce(&Buffer<cl_float>) -> Result<R>,
    {
        let buffer_id = ptr as usize;
        let buffers = self.buffers.lock().unwrap();
        if let Some(opencl_buffer) = buffers.get(&buffer_id) {
            operation(&opencl_buffer.buffer)
        } else {
            Err(crate::error::MinitensorError::memory_error(
                "OpenCL buffer not found for pointer"
            ))
        }
    }

    /// Get buffer information for debugging
    pub fn get_buffer_info(&self, ptr: *const u8) -> Option<(usize, usize)> {
        let buffer_id = ptr as usize;
        let buffers = self.buffers.lock().unwrap();
        buffers.get(&buffer_id).map(|buf| (buffer_id, buf.size_bytes))
    }

    /// Get total number of tracked buffers
    pub fn buffer_count(&self) -> usize {
        self.buffers.lock().unwrap().len()
    }

    /// Finish all operations in the command queue
    pub fn finish(&self) -> Result<()> {
        self.command_queue.finish()
            .map_err(|e| crate::error::MinitensorError::backend_error(format!("Failed to finish OpenCL operations: {}", e)))
    }
}

impl Backend for OpenCLBackend {
    fn device(&self) -> Device {
        self.device
    }

    fn is_available() -> bool {
        // Check if OpenCL platforms and GPU devices are available
        if let Ok(platforms) = get_platforms() {
            for platform in platforms {
                if let Ok(devices) = opencl3::device::get_device_ids(platform.id(), CL_DEVICE_TYPE_GPU) {
                    if !devices.is_empty() {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn initialize() -> Result<Self> {
        // Get the first available GPU device
        let platforms = get_platforms()
            .map_err(|e| crate::error::MinitensorError::backend_error(format!("Failed to get OpenCL platforms: {}", e)))?;

        let mut all_devices = Vec::new();
        for platform in platforms {
            if let Ok(platform_devices) = opencl3::device::get_device_ids(platform.id(), CL_DEVICE_TYPE_GPU) {
                all_devices.extend(platform_devices);
            }
        }
        let devices = all_devices;

        if devices.is_empty() {
            return Err(crate::error::MinitensorError::backend_error("No OpenCL GPU device found"));
        }

        let opencl_device_id = devices[0];
        let opencl_device = opencl3::device::Device::new(opencl_device_id);

        // Create context and command queue
        let context = Context::from_device(&opencl_device)
            .map_err(|e| crate::error::MinitensorError::backend_error(format!("Failed to create OpenCL context: {}", e)))?;

        let command_queue = CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE)
            .map_err(|e| crate::error::MinitensorError::backend_error(format!("Failed to create OpenCL command queue: {}", e)))?;

        Ok(Self {
            device: Device::opencl(Some(0)),
            opencl_device,
            context,
            command_queue,
            programs: Arc::new(Mutex::new(HashMap::new())),
            kernels: Arc::new(Mutex::new(HashMap::new())),
            buffers: Arc::new(Mutex::new(HashMap::new())),
            next_buffer_id: Arc::new(Mutex::new(1)),
        })
    }

    fn allocate(&self, size_bytes: usize) -> Result<*mut u8> {
        if size_bytes == 0 {
            return Ok(std::ptr::null_mut());
        }

        let size_floats = (size_bytes + std::mem::size_of::<f32>() - 1) / std::mem::size_of::<f32>();
        let buffer = self.create_buffer(size_floats, CL_MEM_READ_WRITE)?;

        // Create a unique ID to track this buffer
        let buffer_id = {
            let mut next_id = self.next_buffer_id.lock().unwrap();
            let id = *next_id;
            *next_id += 1;
            id
        };
        
        let opencl_buffer = OpenCLBuffer {
            buffer,
            size_bytes,
        };

        // Store the buffer for tracking
        let mut buffers = self.buffers.lock().unwrap();
        buffers.insert(buffer_id, opencl_buffer);

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

        // OpenCL buffer will be automatically dropped and freed
        Ok(())
    }

    fn copy_from_host(&self, dst: *mut u8, src: &[u8]) -> Result<()> {
        if dst.is_null() {
            return Err(crate::error::MinitensorError::memory_error(
                "Null destination pointer"
            ));
        }

        // Find the OpenCL buffer corresponding to this pointer
        let buffer_id = dst as usize;
        let mut buffers = self.buffers.lock().unwrap();
        if let Some(opencl_buffer) = buffers.get_mut(&buffer_id) {
            // Convert bytes to f32 for OpenCL buffer
            let src_floats = unsafe {
                std::slice::from_raw_parts(
                    src.as_ptr() as *const f32,
                    src.len() / std::mem::size_of::<f32>()
                )
            };
            
            unsafe {
                self.command_queue
                    .enqueue_write_buffer(&mut opencl_buffer.buffer, CL_BLOCKING, 0, src_floats, &[])
            }
                .map_err(|e| crate::error::MinitensorError::memory_error(
                    format!("Failed to copy data to OpenCL buffer: {}", e)
                ))?;
        } else {
            return Err(crate::error::MinitensorError::memory_error(
                "OpenCL buffer not found for pointer"
            ));
        }

        Ok(())
    }

    fn copy_to_host(&self, dst: &mut [u8], src: *const u8) -> Result<()> {
        if src.is_null() {
            return Err(crate::error::MinitensorError::memory_error(
                "Null source pointer"
            ));
        }

        // Find the OpenCL buffer corresponding to this pointer
        let buffer_id = src as usize;
        let buffers = self.buffers.lock().unwrap();
        if let Some(opencl_buffer) = buffers.get(&buffer_id) {
            // Convert bytes to f32 for OpenCL buffer
            let dst_floats = unsafe {
                std::slice::from_raw_parts_mut(
                    dst.as_mut_ptr() as *mut f32,
                    dst.len() / std::mem::size_of::<f32>()
                )
            };
            
            unsafe {
                self.command_queue
                    .enqueue_read_buffer(&opencl_buffer.buffer, CL_BLOCKING, 0, dst_floats, &[])
            }
                .map_err(|e| crate::error::MinitensorError::memory_error(
                    format!("Failed to copy data from OpenCL buffer: {}", e)
                ))?;
        } else {
            return Err(crate::error::MinitensorError::memory_error(
                "OpenCL buffer not found for pointer"
            ));
        }

        Ok(())
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

impl OpenCLOps {
    /// Create new OpenCL operations instance
    pub fn new(backend: Arc<OpenCLBackend>) -> Result<Self> {
        // Build the kernel program
        backend.build_program("tensor_ops", kernels::ALL_KERNELS)?;

        // Create all kernels
        backend.create_kernel("tensor_ops", "add_kernel")?;
        backend.create_kernel("tensor_ops", "mul_kernel")?;
        backend.create_kernel("tensor_ops", "matmul_kernel")?;
        backend.create_kernel("tensor_ops", "relu_kernel")?;
        backend.create_kernel("tensor_ops", "sigmoid_kernel")?;

        Ok(Self { backend })
    }

    /// Element-wise addition on GPU
    pub fn add(&self, a: &Buffer<cl_float>, b: &Buffer<cl_float>, c: &Buffer<cl_float>, n: u32) -> Result<()> {
        let kernel = self.backend.get_kernel("add_kernel")
            .ok_or_else(|| crate::error::MinitensorError::backend_error("Add kernel not found"))?;

        unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(a)
                .set_arg(b)
                .set_arg(c)
                .set_arg(&n)
                .set_global_work_size(n as usize)
                .enqueue_nd_range(&self.backend.command_queue)
        }.map_err(|e| crate::error::MinitensorError::backend_error(format!("Failed to execute add kernel: {}", e)))?
        .wait()
        .map_err(|e| crate::error::MinitensorError::backend_error(format!("Failed to wait for add kernel: {}", e)))?;

        Ok(())
    }

    /// Element-wise multiplication on GPU
    pub fn mul(&self, a: &Buffer<cl_float>, b: &Buffer<cl_float>, c: &Buffer<cl_float>, n: u32) -> Result<()> {
        let kernel = self.backend.get_kernel("mul_kernel")
            .ok_or_else(|| crate::error::MinitensorError::backend_error("Mul kernel not found"))?;

        unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(a)
                .set_arg(b)
                .set_arg(c)
                .set_arg(&n)
                .set_global_work_size(n as usize)
                .enqueue_nd_range(&self.backend.command_queue)
        }.map_err(|e| crate::error::MinitensorError::backend_error(format!("Failed to execute mul kernel: {}", e)))?
        .wait()
        .map_err(|e| crate::error::MinitensorError::backend_error(format!("Failed to wait for mul kernel: {}", e)))?;

        Ok(())
    }

    /// Matrix multiplication on GPU
    pub fn matmul(&self, a: &Buffer<cl_float>, b: &Buffer<cl_float>, c: &Buffer<cl_float>, m: u32, n: u32, k: u32) -> Result<()> {
        let kernel = self.backend.get_kernel("matmul_kernel")
            .ok_or_else(|| crate::error::MinitensorError::backend_error("Matmul kernel not found"))?;

        unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(a)
                .set_arg(b)
                .set_arg(c)
                .set_arg(&m)
                .set_arg(&n)
                .set_arg(&k)
                .set_global_work_sizes(&[n as usize, m as usize])
                .enqueue_nd_range(&self.backend.command_queue)
        }.map_err(|e| crate::error::MinitensorError::backend_error(format!("Failed to execute matmul kernel: {}", e)))?
        .wait()
        .map_err(|e| crate::error::MinitensorError::backend_error(format!("Failed to wait for matmul kernel: {}", e)))?;

        Ok(())
    }

    /// ReLU activation on GPU
    pub fn relu(&self, input: &Buffer<cl_float>, output: &Buffer<cl_float>, n: u32) -> Result<()> {
        let kernel = self.backend.get_kernel("relu_kernel")
            .ok_or_else(|| crate::error::MinitensorError::backend_error("ReLU kernel not found"))?;

        unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(input)
                .set_arg(output)
                .set_arg(&n)
                .set_global_work_size(n as usize)
                .enqueue_nd_range(&self.backend.command_queue)
        }.map_err(|e| crate::error::MinitensorError::backend_error(format!("Failed to execute relu kernel: {}", e)))?
        .wait()
        .map_err(|e| crate::error::MinitensorError::backend_error(format!("Failed to wait for relu kernel: {}", e)))?;

        Ok(())
    }

    /// Sigmoid activation on GPU
    pub fn sigmoid(&self, input: &Buffer<cl_float>, output: &Buffer<cl_float>, n: u32) -> Result<()> {
        let kernel = self.backend.get_kernel("sigmoid_kernel")
            .ok_or_else(|| crate::error::MinitensorError::backend_error("Sigmoid kernel not found"))?;

        unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(input)
                .set_arg(output)
                .set_arg(&n)
                .set_global_work_size(n as usize)
                .enqueue_nd_range(&self.backend.command_queue)
        }.map_err(|e| crate::error::MinitensorError::backend_error(format!("Failed to execute sigmoid kernel: {}", e)))?
        .wait()
        .map_err(|e| crate::error::MinitensorError::backend_error(format!("Failed to wait for sigmoid kernel: {}", e)))?;

        Ok(())
    }

    /// Execute element-wise addition using pointers
    pub fn add_ptr(&self, a_ptr: *const u8, b_ptr: *const u8, c_ptr: *mut u8, n: u32) -> Result<()> {
        // Get buffers from the backend's buffer tracking system
        let a_buffer_id = a_ptr as usize;
        let b_buffer_id = b_ptr as usize;
        let c_buffer_id = c_ptr as usize;
        
        let buffers = self.backend.buffers.lock().unwrap();
        
        if let (Some(a_buf), Some(b_buf), Some(c_buf)) = (
            buffers.get(&a_buffer_id),
            buffers.get(&b_buffer_id),
            buffers.get(&c_buffer_id)
        ) {
            self.add(&a_buf.buffer, &b_buf.buffer, &c_buf.buffer, n)
        } else {
            Err(crate::error::MinitensorError::memory_error("OpenCL buffer not found for operation"))
        }
    }

    /// Execute element-wise multiplication using pointers
    pub fn mul_ptr(&self, a_ptr: *const u8, b_ptr: *const u8, c_ptr: *mut u8, n: u32) -> Result<()> {
        let a_buffer_id = a_ptr as usize;
        let b_buffer_id = b_ptr as usize;
        let c_buffer_id = c_ptr as usize;
        
        let buffers = self.backend.buffers.lock().unwrap();
        
        if let (Some(a_buf), Some(b_buf), Some(c_buf)) = (
            buffers.get(&a_buffer_id),
            buffers.get(&b_buffer_id),
            buffers.get(&c_buffer_id)
        ) {
            self.mul(&a_buf.buffer, &b_buf.buffer, &c_buf.buffer, n)
        } else {
            Err(crate::error::MinitensorError::memory_error("OpenCL buffer not found for operation"))
        }
    }

    /// Execute matrix multiplication using pointers
    pub fn matmul_ptr(&self, a_ptr: *const u8, b_ptr: *const u8, c_ptr: *mut u8, m: u32, n: u32, k: u32) -> Result<()> {
        let a_buffer_id = a_ptr as usize;
        let b_buffer_id = b_ptr as usize;
        let c_buffer_id = c_ptr as usize;
        
        let buffers = self.backend.buffers.lock().unwrap();
        
        if let (Some(a_buf), Some(b_buf), Some(c_buf)) = (
            buffers.get(&a_buffer_id),
            buffers.get(&b_buffer_id),
            buffers.get(&c_buffer_id)
        ) {
            self.matmul(&a_buf.buffer, &b_buf.buffer, &c_buf.buffer, m, n, k)
        } else {
            Err(crate::error::MinitensorError::memory_error("OpenCL buffer not found for operation"))
        }
    }

    /// Execute ReLU activation using pointers
    pub fn relu_ptr(&self, input_ptr: *const u8, output_ptr: *mut u8, n: u32) -> Result<()> {
        let input_buffer_id = input_ptr as usize;
        let output_buffer_id = output_ptr as usize;
        
        let buffers = self.backend.buffers.lock().unwrap();
        
        if let (Some(input_buf), Some(output_buf)) = (
            buffers.get(&input_buffer_id),
            buffers.get(&output_buffer_id)
        ) {
            self.relu(&input_buf.buffer, &output_buf.buffer, n)
        } else {
            Err(crate::error::MinitensorError::memory_error("OpenCL buffer not found for operation"))
        }
    }

    /// Execute Sigmoid activation using pointers
    pub fn sigmoid_ptr(&self, input_ptr: *const u8, output_ptr: *mut u8, n: u32) -> Result<()> {
        let input_buffer_id = input_ptr as usize;
        let output_buffer_id = output_ptr as usize;
        
        let buffers = self.backend.buffers.lock().unwrap();
        
        if let (Some(input_buf), Some(output_buf)) = (
            buffers.get(&input_buffer_id),
            buffers.get(&output_buffer_id)
        ) {
            self.sigmoid(&input_buf.buffer, &output_buf.buffer, n)
        } else {
            Err(crate::error::MinitensorError::memory_error("OpenCL buffer not found for operation"))
        }
    }
}

#[cfg(test)]
mod integration_test;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opencl_availability() {
        // This test will only pass if OpenCL is available
        if OpenCLBackend::is_available() {
            let backend = OpenCLBackend::initialize().unwrap();
            assert!(backend.device().is_gpu());
        }
    }

    #[test]
    fn test_opencl_buffer_operations() {
        if !OpenCLBackend::is_available() {
            return; // Skip test if OpenCL not available
        }

        let backend = OpenCLBackend::initialize().unwrap();
        
        // Test buffer creation and data transfer
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let buffer = backend.create_buffer_with_data(&data, CL_MEM_READ_WRITE).unwrap();
        
        let mut result = vec![0.0f32; 5];
        backend.read_buffer(&buffer, &mut result).unwrap();
        
        assert_eq!(data, result);
    }

    #[test]
    fn test_opencl_operations() {
        if !OpenCLBackend::is_available() {
            return; // Skip test if OpenCL not available
        }

        let backend = Arc::new(OpenCLBackend::initialize().unwrap());
        let ops = OpenCLOps::new(backend.clone()).unwrap();
        
        // Test addition
        let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let b_data = vec![5.0f32, 6.0, 7.0, 8.0];
        
        let a_buffer = backend.create_buffer_with_data(&a_data, CL_MEM_READ_ONLY).unwrap();
        let b_buffer = backend.create_buffer_with_data(&b_data, CL_MEM_READ_ONLY).unwrap();
        let c_buffer = backend.create_buffer(4, CL_MEM_WRITE_ONLY).unwrap();
        
        ops.add(&a_buffer, &b_buffer, &c_buffer, 4).unwrap();
        
        let mut result = vec![0.0f32; 4];
        backend.read_buffer(&c_buffer, &mut result).unwrap();
        
        let expected = vec![6.0f32, 8.0, 10.0, 12.0];
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-6);
        }
    }
}
// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

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
    pub fn add(
        &self,
        a: &Buffer<cl_float>,
        b: &Buffer<cl_float>,
        c: &Buffer<cl_float>,
        n: u32,
    ) -> Result<()> {
        let kernel = self.backend.get_kernel("add_kernel").ok_or_else(|| {
            crate::error::MinitensorError::backend_error("OpenCL", "Add kernel not found")
        })?;

        unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(a)
                .set_arg(b)
                .set_arg(c)
                .set_arg(&n)
                .set_global_work_size(n as usize)
                .enqueue_nd_range(&self.backend.command_queue)
        }
        .map_err(|e| {
            crate::error::MinitensorError::backend_error(
                "OpenCL",
                format!("Failed to execute add kernel: {}", e),
            )
        })?
        .wait()
        .map_err(|e| {
            crate::error::MinitensorError::backend_error(
                "OpenCL",
                format!("Failed to wait for add kernel: {}", e),
            )
        })?;

        Ok(())
    }

    /// Element-wise multiplication on GPU
    pub fn mul(
        &self,
        a: &Buffer<cl_float>,
        b: &Buffer<cl_float>,
        c: &Buffer<cl_float>,
        n: u32,
    ) -> Result<()> {
        let kernel = self.backend.get_kernel("mul_kernel").ok_or_else(|| {
            crate::error::MinitensorError::backend_error("OpenCL", "Mul kernel not found")
        })?;

        unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(a)
                .set_arg(b)
                .set_arg(c)
                .set_arg(&n)
                .set_global_work_size(n as usize)
                .enqueue_nd_range(&self.backend.command_queue)
        }
        .map_err(|e| {
            crate::error::MinitensorError::backend_error(
                "OpenCL",
                format!("Failed to execute mul kernel: {}", e),
            )
        })?
        .wait()
        .map_err(|e| {
            crate::error::MinitensorError::backend_error(
                "OpenCL",
                format!("Failed to wait for mul kernel: {}", e),
            )
        })?;

        Ok(())
    }

    /// Matrix multiplication on GPU
    pub fn matmul(
        &self,
        a: &Buffer<cl_float>,
        b: &Buffer<cl_float>,
        c: &Buffer<cl_float>,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<()> {
        let kernel = self.backend.get_kernel("matmul_kernel").ok_or_else(|| {
            crate::error::MinitensorError::backend_error("OpenCL", "Matmul kernel not found")
        })?;

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
        }
        .map_err(|e| {
            crate::error::MinitensorError::backend_error(
                "OpenCL",
                format!("Failed to execute matmul kernel: {}", e),
            )
        })?
        .wait()
        .map_err(|e| {
            crate::error::MinitensorError::backend_error(
                "OpenCL",
                format!("Failed to wait for matmul kernel: {}", e),
            )
        })?;

        Ok(())
    }

    /// ReLU activation on GPU
    pub fn relu(&self, input: &Buffer<cl_float>, output: &Buffer<cl_float>, n: u32) -> Result<()> {
        let kernel = self.backend.get_kernel("relu_kernel").ok_or_else(|| {
            crate::error::MinitensorError::backend_error("OpenCL", "ReLU kernel not found")
        })?;

        unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(input)
                .set_arg(output)
                .set_arg(&n)
                .set_global_work_size(n as usize)
                .enqueue_nd_range(&self.backend.command_queue)
        }
        .map_err(|e| {
            crate::error::MinitensorError::backend_error(
                "OpenCL",
                format!("Failed to execute relu kernel: {}", e),
            )
        })?
        .wait()
        .map_err(|e| {
            crate::error::MinitensorError::backend_error(
                "OpenCL",
                format!("Failed to wait for relu kernel: {}", e),
            )
        })?;

        Ok(())
    }

    /// Sigmoid activation on GPU
    pub fn sigmoid(
        &self,
        input: &Buffer<cl_float>,
        output: &Buffer<cl_float>,
        n: u32,
    ) -> Result<()> {
        let kernel = self.backend.get_kernel("sigmoid_kernel").ok_or_else(|| {
            crate::error::MinitensorError::backend_error("OpenCL", "Sigmoid kernel not found")
        })?;

        unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(input)
                .set_arg(output)
                .set_arg(&n)
                .set_global_work_size(n as usize)
                .enqueue_nd_range(&self.backend.command_queue)
        }
        .map_err(|e| {
            crate::error::MinitensorError::backend_error(
                "OpenCL",
                format!("Failed to execute sigmoid kernel: {}", e),
            )
        })?
        .wait()
        .map_err(|e| {
            crate::error::MinitensorError::backend_error(
                "OpenCL",
                format!("Failed to wait for sigmoid kernel: {}", e),
            )
        })?;

        Ok(())
    }

    /// Execute element-wise addition using pointers
    pub fn add_ptr(
        &self,
        a_ptr: *const u8,
        b_ptr: *const u8,
        c_ptr: *mut u8,
        n: u32,
    ) -> Result<()> {
        // Get buffers from the backend's buffer tracking system
        let a_buffer_id = a_ptr as usize;
        let b_buffer_id = b_ptr as usize;
        let c_buffer_id = c_ptr as usize;

        let buffers = self.backend.buffers.read();

        if let (Some(a_buf), Some(b_buf), Some(c_buf)) = (
            buffers.get(&a_buffer_id),
            buffers.get(&b_buffer_id),
            buffers.get(&c_buffer_id),
        ) {
            self.add(&a_buf.buffer, &b_buf.buffer, &c_buf.buffer, n)
        } else {
            Err(crate::error::MinitensorError::memory_error(
                "OpenCL buffer not found for operation",
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

        let buffers = self.backend.buffers.read();

        if let (Some(a_buf), Some(b_buf), Some(c_buf)) = (
            buffers.get(&a_buffer_id),
            buffers.get(&b_buffer_id),
            buffers.get(&c_buffer_id),
        ) {
            self.mul(&a_buf.buffer, &b_buf.buffer, &c_buf.buffer, n)
        } else {
            Err(crate::error::MinitensorError::memory_error(
                "OpenCL buffer not found for operation",
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

        let buffers = self.backend.buffers.read();

        if let (Some(a_buf), Some(b_buf), Some(c_buf)) = (
            buffers.get(&a_buffer_id),
            buffers.get(&b_buffer_id),
            buffers.get(&c_buffer_id),
        ) {
            self.matmul(&a_buf.buffer, &b_buf.buffer, &c_buf.buffer, m, n, k)
        } else {
            Err(crate::error::MinitensorError::memory_error(
                "OpenCL buffer not found for operation",
            ))
        }
    }

    /// Execute ReLU activation using pointers
    pub fn relu_ptr(&self, input_ptr: *const u8, output_ptr: *mut u8, n: u32) -> Result<()> {
        let input_buffer_id = input_ptr as usize;
        let output_buffer_id = output_ptr as usize;

        let buffers = self.backend.buffers.read();

        if let (Some(input_buf), Some(output_buf)) = (
            buffers.get(&input_buffer_id),
            buffers.get(&output_buffer_id),
        ) {
            self.relu(&input_buf.buffer, &output_buf.buffer, n)
        } else {
            Err(crate::error::MinitensorError::memory_error(
                "OpenCL buffer not found for operation",
            ))
        }
    }

    /// Execute Sigmoid activation using pointers
    pub fn sigmoid_ptr(&self, input_ptr: *const u8, output_ptr: *mut u8, n: u32) -> Result<()> {
        let input_buffer_id = input_ptr as usize;
        let output_buffer_id = output_ptr as usize;

        let buffers = self.backend.buffers.read();

        if let (Some(input_buf), Some(output_buf)) = (
            buffers.get(&input_buffer_id),
            buffers.get(&output_buffer_id),
        ) {
            self.sigmoid(&input_buf.buffer, &output_buf.buffer, n)
        } else {
            Err(crate::error::MinitensorError::memory_error(
                "OpenCL buffer not found for operation",
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
        let buffer = backend
            .create_buffer_with_data(&data, CL_MEM_READ_WRITE)
            .unwrap();

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

        let a_buffer = backend
            .create_buffer_with_data(&a_data, CL_MEM_READ_ONLY)
            .unwrap();
        let b_buffer = backend
            .create_buffer_with_data(&b_data, CL_MEM_READ_ONLY)
            .unwrap();
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

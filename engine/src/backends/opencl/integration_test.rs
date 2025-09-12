// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::backends::Backend;
    use std::sync::Arc;

    #[test]
    fn test_opencl_backend_initialization() {
        if !OpenCLBackend::is_available() {
            println!("OpenCL not available, skipping test");
            return;
        }

        let backend = OpenCLBackend::initialize();
        assert!(backend.is_ok(), "Failed to initialize OpenCL backend");

        let backend = backend.unwrap();
        assert!(backend.device().is_gpu());
        assert_eq!(
            backend.device().device_type,
            crate::device::DeviceType::OpenCL
        );
    }

    #[test]
    fn test_opencl_memory_operations() {
        if !OpenCLBackend::is_available() {
            println!("OpenCL not available, skipping test");
            return;
        }

        let backend = OpenCLBackend::initialize().unwrap();

        // Test allocation
        let size = 1024;
        let ptr = backend.allocate(size).unwrap();
        assert!(!ptr.is_null());

        // Test host to device copy
        let data = vec![1u8; size];
        backend.copy_from_host(ptr, &data).unwrap();

        // Test device to host copy
        let mut result = vec![0u8; size];
        backend.copy_to_host(&mut result, ptr).unwrap();

        // Verify data integrity
        assert_eq!(data, result);

        // Test deallocation
        backend.deallocate(ptr, size).unwrap();
    }

    #[test]
    fn test_opencl_kernel_operations() {
        if !OpenCLBackend::is_available() {
            println!("OpenCL not available, skipping test");
            return;
        }

        let backend = Arc::new(OpenCLBackend::initialize().unwrap());
        let ops = OpenCLOps::new(backend.clone());

        if ops.is_err() {
            println!("Failed to create OpenCL operations, skipping test");
            return;
        }

        let ops = ops.unwrap();

        // Test basic kernel execution
        let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let b_data = vec![5.0f32, 6.0, 7.0, 8.0];

        let a_buffer = backend
            .create_buffer_with_data(&a_data, opencl3::memory::CL_MEM_READ_ONLY)
            .unwrap();
        let b_buffer = backend
            .create_buffer_with_data(&b_data, opencl3::memory::CL_MEM_READ_ONLY)
            .unwrap();
        let c_buffer = backend
            .create_buffer(4, opencl3::memory::CL_MEM_WRITE_ONLY)
            .unwrap();

        // Test addition
        let result = ops.add(&a_buffer, &b_buffer, &c_buffer, 4);
        assert!(result.is_ok(), "Addition operation failed");

        // Read back result
        let mut result_data = vec![0.0f32; 4];
        backend.read_buffer(&c_buffer, &mut result_data).unwrap();

        let expected = vec![6.0f32, 8.0, 10.0, 12.0];
        for (r, e) in result_data.iter().zip(expected.iter()) {
            assert!(
                (r - e).abs() < 1e-6,
                "Addition result mismatch: {} != {}",
                r,
                e
            );
        }
    }

    #[test]
    fn test_opencl_matrix_multiplication() {
        if !OpenCLBackend::is_available() {
            println!("OpenCL not available, skipping test");
            return;
        }

        let backend = Arc::new(OpenCLBackend::initialize().unwrap());
        let ops = OpenCLOps::new(backend.clone());

        if ops.is_err() {
            println!("Failed to create OpenCL operations, skipping test");
            return;
        }

        let ops = ops.unwrap();

        // Test 2x2 matrix multiplication
        let a_data = vec![1.0f32, 2.0, 3.0, 4.0]; // 2x2 matrix
        let b_data = vec![5.0f32, 6.0, 7.0, 8.0]; // 2x2 matrix

        let a_buffer = backend
            .create_buffer_with_data(&a_data, opencl3::memory::CL_MEM_READ_ONLY)
            .unwrap();
        let b_buffer = backend
            .create_buffer_with_data(&b_data, opencl3::memory::CL_MEM_READ_ONLY)
            .unwrap();
        let c_buffer = backend
            .create_buffer(4, opencl3::memory::CL_MEM_WRITE_ONLY)
            .unwrap();

        // Test matrix multiplication (2x2 * 2x2 = 2x2)
        let result = ops.matmul(&a_buffer, &b_buffer, &c_buffer, 2, 2, 2);
        assert!(result.is_ok(), "Matrix multiplication failed");

        // Read back result
        let mut result_data = vec![0.0f32; 4];
        backend.read_buffer(&c_buffer, &mut result_data).unwrap();

        // Expected result: [1*5+2*7, 1*6+2*8, 3*5+4*7, 3*6+4*8] = [19, 22, 43, 50]
        let expected = vec![19.0f32, 22.0, 43.0, 50.0];
        for (r, e) in result_data.iter().zip(expected.iter()) {
            assert!(
                (r - e).abs() < 1e-6,
                "Matrix multiplication result mismatch: {} != {}",
                r,
                e
            );
        }
    }

    #[test]
    fn test_opencl_activation_functions() {
        if !OpenCLBackend::is_available() {
            println!("OpenCL not available, skipping test");
            return;
        }

        let backend = Arc::new(OpenCLBackend::initialize().unwrap());
        let ops = OpenCLOps::new(backend.clone());

        if ops.is_err() {
            println!("Failed to create OpenCL operations, skipping test");
            return;
        }

        let ops = ops.unwrap();

        // Test ReLU activation
        let input_data = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
        let input_buffer = backend
            .create_buffer_with_data(&input_data, opencl3::memory::CL_MEM_READ_ONLY)
            .unwrap();
        let output_buffer = backend
            .create_buffer(5, opencl3::memory::CL_MEM_WRITE_ONLY)
            .unwrap();

        let result = ops.relu(&input_buffer, &output_buffer, 5);
        assert!(result.is_ok(), "ReLU operation failed");

        let mut result_data = vec![0.0f32; 5];
        backend
            .read_buffer(&output_buffer, &mut result_data)
            .unwrap();

        let expected = vec![0.0f32, 0.0, 0.0, 1.0, 2.0];
        for (r, e) in result_data.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-6, "ReLU result mismatch: {} != {}", r, e);
        }

        // Test Sigmoid activation
        let sigmoid_input = vec![0.0f32, 1.0, -1.0];
        let sigmoid_input_buffer = backend
            .create_buffer_with_data(&sigmoid_input, opencl3::memory::CL_MEM_READ_ONLY)
            .unwrap();
        let sigmoid_output_buffer = backend
            .create_buffer(3, opencl3::memory::CL_MEM_WRITE_ONLY)
            .unwrap();

        let result = ops.sigmoid(&sigmoid_input_buffer, &sigmoid_output_buffer, 3);
        assert!(result.is_ok(), "Sigmoid operation failed");

        let mut sigmoid_result = vec![0.0f32; 3];
        backend
            .read_buffer(&sigmoid_output_buffer, &mut sigmoid_result)
            .unwrap();

        // Expected: sigmoid(0) = 0.5, sigmoid(1) ≈ 0.731, sigmoid(-1) ≈ 0.269
        assert!((sigmoid_result[0] - 0.5).abs() < 1e-6);
        assert!((sigmoid_result[1] - 0.7310586).abs() < 1e-6);
        assert!((sigmoid_result[2] - 0.26894143).abs() < 1e-6);
    }

    #[test]
    fn test_opencl_zero_length_operations() {
        if !OpenCLBackend::is_available() {
            println!("OpenCL not available, skipping test");
            return;
        }

        let backend = OpenCLBackend::initialize().unwrap();

        let ptr = backend.allocate(0).unwrap();
        assert!(ptr.is_null());

        backend.copy_from_host(ptr, &[]).unwrap();
        backend.copy_to_host(&mut [], ptr).unwrap();

        backend.deallocate(ptr, 0).unwrap();
    }

    #[test]
    fn test_opencl_zero_length_copy_to_valid_pointer() {
        if !OpenCLBackend::is_available() {
            println!("OpenCL not available, skipping test");
            return;
        }

        let backend = OpenCLBackend::initialize().unwrap();
        let ptr = backend.allocate(8).unwrap();
        backend.copy_from_host(ptr, &[]).unwrap();
        let mut buf = vec![0u8; 0];
        backend.copy_to_host(&mut buf, ptr).unwrap();
        backend.deallocate(ptr, 8).unwrap();
    }

    #[test]
    fn test_opencl_null_pointer_errors() {
        if !OpenCLBackend::is_available() {
            println!("OpenCL not available, skipping test");
            return;
        }

        let backend = OpenCLBackend::initialize().unwrap();
        assert!(
            backend
                .copy_from_host(std::ptr::null_mut(), &[1u8])
                .is_err()
        );
        let mut buf = [0u8; 1];
        assert!(backend.copy_to_host(&mut buf, std::ptr::null()).is_err());
    }

    #[test]
    fn test_opencl_multiple_allocations_and_copies() {
        if !OpenCLBackend::is_available() {
            println!("OpenCL not available, skipping test");
            return;
        }

        let backend = OpenCLBackend::initialize().unwrap();
        let ptr1 = backend.allocate(4).unwrap();
        let ptr2 = backend.allocate(4).unwrap();

        let data1 = [1u8, 2, 3, 4];
        let data2 = [5u8, 6, 7, 8];

        backend.copy_from_host(ptr1, &data1).unwrap();
        backend.copy_from_host(ptr2, &data2).unwrap();

        let mut out1 = [0u8; 4];
        let mut out2 = [0u8; 4];
        backend.copy_to_host(&mut out1, ptr1).unwrap();
        backend.copy_to_host(&mut out2, ptr2).unwrap();

        assert_eq!(data1, out1);
        assert_eq!(data2, out2);

        backend.deallocate(ptr1, 4).unwrap();
        backend.deallocate(ptr2, 4).unwrap();
    }

    #[test]
    fn test_opencl_deallocate_null_pointer() {
        if !OpenCLBackend::is_available() {
            println!("OpenCL not available, skipping test");
            return;
        }

        let backend = OpenCLBackend::initialize().unwrap();
        backend.deallocate(std::ptr::null_mut(), 128).unwrap();
    }

    #[test]
    fn test_opencl_memory_pool_reuse() {
        if !OpenCLBackend::is_available() {
            println!("OpenCL not available, skipping test");
            return;
        }

        let backend = OpenCLBackend::initialize().unwrap();

        let ptr1 = backend.allocate(256).unwrap();
        backend.deallocate(ptr1, 256).unwrap();
        let ptr2 = backend.allocate(256).unwrap();

        assert_eq!(ptr1, ptr2);

        backend.deallocate(ptr2, 256).unwrap();
    }
}

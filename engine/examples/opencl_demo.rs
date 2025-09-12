// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! OpenCL backend demonstration
//!
//! This example shows how to use the OpenCL backend for tensor operations.
//! Run with: cargo run --example opencl_demo --features opencl

#[cfg(feature = "opencl")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use engine::backends::Backend;
    use engine::backends::opencl::{OpenCLBackend, OpenCLOps};
    use opencl3::memory::{CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY};
    use std::sync::Arc;

    println!("OpenCL Backend Demo");
    println!("==================");

    // Check if OpenCL is available
    if !OpenCLBackend::is_available() {
        println!("OpenCL is not available on this system.");
        println!("Please ensure you have OpenCL drivers installed.");
        return Ok(());
    }

    println!("OpenCL is available");

    // Initialize the OpenCL backend
    let backend = Arc::new(OpenCLBackend::initialize()?);
    println!("OpenCL backend initialized");
    println!("  Device: {}", backend.device());

    // Create OpenCL operations
    let ops = OpenCLOps::new(backend.clone())?;
    println!("OpenCL operations created");

    // Demonstrate element-wise addition
    println!("\n--- Element-wise Addition ---");
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let b_data = vec![10.0f32, 20.0, 30.0, 40.0, 50.0];

    println!("Input A: {:?}", a_data);
    println!("Input B: {:?}", b_data);

    let a_buffer = backend.create_buffer_with_data(&a_data, CL_MEM_READ_ONLY)?;
    let b_buffer = backend.create_buffer_with_data(&b_data, CL_MEM_READ_ONLY)?;
    let c_buffer = backend.create_buffer(5, CL_MEM_WRITE_ONLY)?;

    ops.add(&a_buffer, &b_buffer, &c_buffer, 5)?;

    let mut result = vec![0.0f32; 5];
    backend.read_buffer(&c_buffer, &mut result)?;
    println!("Result:  {:?}", result);

    // Demonstrate matrix multiplication
    println!("\n--- Matrix Multiplication ---");
    let matrix_a = vec![1.0f32, 2.0, 3.0, 4.0]; // 2x2 matrix
    let matrix_b = vec![5.0f32, 6.0, 7.0, 8.0]; // 2x2 matrix

    println!(
        "Matrix A (2x2): [{}]",
        matrix_a[0..2]
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "                [{}]",
        matrix_a[2..4]
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "Matrix B (2x2): [{}]",
        matrix_b[0..2]
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "                [{}]",
        matrix_b[2..4]
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );

    let a_mat_buffer = backend.create_buffer_with_data(&matrix_a, CL_MEM_READ_ONLY)?;
    let b_mat_buffer = backend.create_buffer_with_data(&matrix_b, CL_MEM_READ_ONLY)?;
    let c_mat_buffer = backend.create_buffer(4, CL_MEM_WRITE_ONLY)?;

    ops.matmul(&a_mat_buffer, &b_mat_buffer, &c_mat_buffer, 2, 2, 2)?;

    let mut mat_result = vec![0.0f32; 4];
    backend.read_buffer(&c_mat_buffer, &mut mat_result)?;
    println!(
        "Result (2x2):   [{}]",
        mat_result[0..2]
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "                [{}]",
        mat_result[2..4]
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );

    // Demonstrate activation functions
    println!("\n--- Activation Functions ---");
    let activation_input = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    println!("Input: {:?}", activation_input);

    // ReLU
    let relu_input_buffer = backend.create_buffer_with_data(&activation_input, CL_MEM_READ_ONLY)?;
    let relu_output_buffer = backend.create_buffer(5, CL_MEM_WRITE_ONLY)?;

    ops.relu(&relu_input_buffer, &relu_output_buffer, 5)?;

    let mut relu_result = vec![0.0f32; 5];
    backend.read_buffer(&relu_output_buffer, &mut relu_result)?;
    println!("ReLU:  {:?}", relu_result);

    // Sigmoid
    let sigmoid_input_buffer =
        backend.create_buffer_with_data(&activation_input, CL_MEM_READ_ONLY)?;
    let sigmoid_output_buffer = backend.create_buffer(5, CL_MEM_WRITE_ONLY)?;

    ops.sigmoid(&sigmoid_input_buffer, &sigmoid_output_buffer, 5)?;

    let mut sigmoid_result = vec![0.0f32; 5];
    backend.read_buffer(&sigmoid_output_buffer, &mut sigmoid_result)?;
    println!(
        "Sigmoid: {:?}",
        sigmoid_result
            .iter()
            .map(|x| format!("{:.4}", x))
            .collect::<Vec<_>>()
    );

    // Memory management demonstration
    println!("\n--- Memory Management ---");
    println!("Tracked buffers: {}", backend.buffer_count());

    // Test backend memory allocation interface
    let ptr = backend.allocate(1024)?;
    println!("Allocated 1024 bytes at pointer: {:p}", ptr);

    let test_data = vec![42u8; 1024];
    backend.copy_from_host(ptr, &test_data)?;
    println!("Copied data to device");

    let mut read_back = vec![0u8; 1024];
    backend.copy_to_host(&mut read_back, ptr)?;
    println!("Copied data from device");

    assert_eq!(test_data, read_back);
    println!("Data integrity verified");

    backend.deallocate(ptr, 1024)?;
    println!("Memory deallocated");

    backend.finish()?;
    println!("\nAll operations completed successfully!");

    Ok(())
}

#[cfg(not(feature = "opencl"))]
fn main() {
    println!("This example requires the 'opencl' feature to be enabled.");
    println!("Run with: cargo run --example opencl_demo --features opencl");
}

// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Metal backend demonstration
//!
//! This example shows how to use the Metal backend for tensor operations on Apple devices.
//! Run with: cargo run --example metal_demo --features metal

#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use engine::backends::Backend;
    use engine::backends::metal::{MetalBackend, MetalOps};

    use std::sync::Arc;

    println!("Metal Backend Demo");
    println!("==================");

    // Check if Metal is available
    if !MetalBackend::is_available() {
        println!("Metal is not available on this system.");
        println!("Please ensure you're running on macOS with Metal support.");
        return Ok(());
    }

    println!("Metal is available");

    // Initialize the Metal backend
    let backend = Arc::new(MetalBackend::initialize()?);
    println!("Metal backend initialized");
    println!("  Device: {}", backend.device());

    // Get device information
    let metal_device = backend.metal_device();
    println!("  Metal Device Name: {}", metal_device.name());
    println!("  Registry ID: {}", metal_device.registry_id());
    println!(
        "  Max Threads Per Threadgroup: {}",
        metal_device.max_threads_per_threadgroup().width
    );

    // Create Metal operations
    let ops = MetalOps::new(backend.clone())?;
    println!("Metal operations created");

    // Demonstrate element-wise addition
    println!("\n--- Element-wise Addition ---");
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let b_data = vec![10.0f32, 20.0, 30.0, 40.0, 50.0];

    println!("Input A: {:?}", a_data);
    println!("Input B: {:?}", b_data);

    let a_buffer =
        backend.create_buffer_with_data(&a_data, metal::MTLResourceOptions::StorageModeShared)?;
    let b_buffer =
        backend.create_buffer_with_data(&b_data, metal::MTLResourceOptions::StorageModeShared)?;
    let c_buffer = backend.create_buffer(20, metal::MTLResourceOptions::StorageModeShared)?;

    ops.add(&a_buffer, &b_buffer, &c_buffer, 5)?;

    let mut result = vec![0.0f32; 5];
    backend.copy_buffer_to_host(&c_buffer, &mut result)?;
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

    let a_mat_buffer =
        backend.create_buffer_with_data(&matrix_a, metal::MTLResourceOptions::StorageModeShared)?;
    let b_mat_buffer =
        backend.create_buffer_with_data(&matrix_b, metal::MTLResourceOptions::StorageModeShared)?;
    let c_mat_buffer = backend.create_buffer(16, metal::MTLResourceOptions::StorageModeShared)?;

    ops.matmul(&a_mat_buffer, &b_mat_buffer, &c_mat_buffer, 2, 2, 2)?;

    let mut mat_result = vec![0.0f32; 4];
    backend.copy_buffer_to_host(&c_mat_buffer, &mut mat_result)?;
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
    let relu_input_buffer = backend.create_buffer_with_data(
        &activation_input,
        metal::MTLResourceOptions::StorageModeShared,
    )?;
    let relu_output_buffer =
        backend.create_buffer(20, metal::MTLResourceOptions::StorageModeShared)?;

    ops.relu(&relu_input_buffer, &relu_output_buffer, 5)?;

    let mut relu_result = vec![0.0f32; 5];
    backend.copy_buffer_to_host(&relu_output_buffer, &mut relu_result)?;
    println!("ReLU:  {:?}", relu_result);

    // Sigmoid
    let sigmoid_input_buffer = backend.create_buffer_with_data(
        &activation_input,
        metal::MTLResourceOptions::StorageModeShared,
    )?;
    let sigmoid_output_buffer =
        backend.create_buffer(20, metal::MTLResourceOptions::StorageModeShared)?;

    ops.sigmoid(&sigmoid_input_buffer, &sigmoid_output_buffer, 5)?;

    let mut sigmoid_result = vec![0.0f32; 5];
    backend.copy_buffer_to_host(&sigmoid_output_buffer, &mut sigmoid_result)?;
    println!(
        "Sigmoid: {:?}",
        sigmoid_result
            .iter()
            .map(|x| format!("{:.4}", x))
            .collect::<Vec<_>>()
    );

    // Performance characteristics demonstration
    println!("\n--- Performance Characteristics ---");
    if let Some(pipeline) = backend.get_compute_pipeline("add_kernel") {
        let thread_group_size = backend.optimal_thread_group_size(&pipeline);
        let thread_group_size_2d = backend.optimal_thread_group_size_2d(&pipeline);

        println!(
            "Optimal 1D thread group size: {}x{}x{}",
            thread_group_size.width, thread_group_size.height, thread_group_size.depth
        );
        println!(
            "Optimal 2D thread group size: {}x{}x{}",
            thread_group_size_2d.width, thread_group_size_2d.height, thread_group_size_2d.depth
        );
        println!(
            "Max threads per threadgroup: {}",
            pipeline.max_total_threads_per_threadgroup()
        );
        println!(
            "Thread execution width: {}",
            pipeline.thread_execution_width()
        );
    }

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

    // Apple Silicon specific features
    println!("\n--- Apple Silicon Features ---");
    println!("Unified Memory Architecture: Available");
    println!("Shared Memory Mode: Enabled");
    println!("Zero-copy operations: Supported");

    println!("\nAll operations completed successfully!");
    println!("Metal backend provides excellent performance on Apple Silicon!");

    Ok(())
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    println!("This example requires the 'metal' feature and macOS to run.");
    println!("Run with: cargo run --example metal_demo --features metal");
    println!("Note: Metal is only available on Apple devices (macOS, iOS, etc.)");
}

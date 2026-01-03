// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use engine::{
    operations::arithmetic,
    tensor::{DataType, Shape, Tensor, TensorData},
};
use ndarray::Array2;
use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

fn create_tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
    let shape_obj = Shape::new(shape);
    let mut tensor_data = TensorData::zeros(shape_obj.numel(), DataType::Float32);
    if let Some(slice) = tensor_data.as_f32_slice_mut() {
        slice.copy_from_slice(&data);
    }
    Tensor::new(
        Arc::new(tensor_data),
        shape_obj,
        DataType::Float32,
        engine::device::Device::cpu(),
        false,
    )
}

#[test]
fn test_add_performance_vs_ndarray() {
    let size = 128;
    let data1 = vec![1.0f32; size * size];
    let data2 = vec![2.0f32; size * size];
    let a = create_tensor(data1.clone(), vec![size, size]);
    let b = create_tensor(data2.clone(), vec![size, size]);
    let iterations = 200;

    for _ in 0..10 {
        black_box(arithmetic::add(&a, &b).unwrap());
    }

    let start = Instant::now();
    for _ in 0..iterations {
        black_box(arithmetic::add(&a, &b).unwrap());
    }
    let engine_time = start.elapsed();

    let arr1 = Array2::from_shape_vec((size, size), data1).unwrap();
    let arr2 = Array2::from_shape_vec((size, size), data2).unwrap();
    for _ in 0..10 {
        black_box(&arr1 + &arr2);
    }
    let start = Instant::now();
    for _ in 0..iterations {
        black_box(&arr1 + &arr2);
    }
    let ndarray_time = start.elapsed();

    assert!(engine_time.as_secs_f64() <= ndarray_time.as_secs_f64() * 10.0);
}

#[test]
fn test_tensor_memory_is_released() {
    let data = Arc::new(TensorData::zeros(10, DataType::Float32));
    let count_before = Arc::strong_count(&data);
    {
        let tensor = Tensor::new(
            data.clone(),
            Shape::new(vec![10]),
            DataType::Float32,
            engine::device::Device::cpu(),
            false,
        );
        assert_eq!(Arc::strong_count(&data), count_before + 1);
        drop(tensor);
    }
    assert_eq!(Arc::strong_count(&data), count_before);
}

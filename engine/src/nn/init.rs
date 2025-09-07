// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    device::Device,
    error::Result,
    tensor::{DataType, Shape, Tensor, TensorData},
};
use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;
use rand_distr::Normal;
use std::sync::Arc;
// use std::f64::consts::PI; // Not needed yet

/// Parameter initialization methods
#[derive(Debug, Clone, Copy)]
pub enum InitMethod {
    /// Initialize with zeros
    Zeros,
    /// Initialize with ones
    Ones,
    /// Initialize with constant value
    Constant(f64),
    /// Initialize with uniform distribution in range [a, b]
    Uniform { a: f64, b: f64 },
    /// Initialize with normal distribution (mean, std)
    Normal { mean: f64, std: f64 },
    /// Xavier/Glorot uniform initialization
    XavierUniform,
    /// Xavier/Glorot normal initialization
    XavierNormal,
    /// He uniform initialization (for ReLU networks)
    HeUniform,
    /// He normal initialization (for ReLU networks)
    HeNormal,
    /// LeCun uniform initialization
    LeCunUniform,
    /// LeCun normal initialization
    LeCunNormal,
}

impl InitMethod {
    /// Initialize a tensor with the specified method
    pub fn init_tensor(
        &self,
        shape: Shape,
        dtype: DataType,
        device: Device,
        requires_grad: bool,
    ) -> Result<Tensor> {
        match self {
            InitMethod::Zeros => Ok(Tensor::zeros(shape, dtype, device, requires_grad)),
            InitMethod::Ones => Ok(Tensor::ones(shape, dtype, device, requires_grad)),
            InitMethod::Constant(value) => {
                init_constant(shape, *value, dtype, device, requires_grad)
            }
            InitMethod::Uniform { a, b } => {
                init_uniform(shape, *a, *b, dtype, device, requires_grad)
            }
            InitMethod::Normal { mean, std } => {
                init_normal(shape, *mean, *std, dtype, device, requires_grad)
            }
            InitMethod::XavierUniform => xavier_uniform_init(shape, dtype, device, requires_grad),
            InitMethod::XavierNormal => xavier_normal_init(shape, dtype, device, requires_grad),
            InitMethod::HeUniform => he_uniform_init(shape, dtype, device, requires_grad),
            InitMethod::HeNormal => he_normal_init(shape, dtype, device, requires_grad),
            InitMethod::LeCunUniform => lecun_uniform_init(shape, dtype, device, requires_grad),
            InitMethod::LeCunNormal => lecun_normal_init(shape, dtype, device, requires_grad),
        }
    }
}

/// Initialize tensor with constant value
pub fn init_constant(
    shape: Shape,
    value: f64,
    dtype: DataType,
    device: Device,
    requires_grad: bool,
) -> Result<Tensor> {
    let numel = shape.numel();
    let data = match dtype {
        DataType::Float32 => {
            let vec = vec![value as f32; numel];
            TensorData::from_vec_f32(vec, device)
        }
        DataType::Float64 => {
            let vec = vec![value as f64; numel];
            TensorData::from_vec_f64(vec, device)
        }
        DataType::Int32 => {
            let vec = vec![value as i32; numel];
            TensorData::from_vec_i32(vec, device)
        }
        DataType::Int64 => {
            let vec = vec![value as i64; numel];
            TensorData::from_vec_i64(vec, device)
        }
        DataType::Bool => {
            let vec = vec![value != 0.0; numel];
            TensorData::from_vec_bool(vec, device)
        }
    };
    Ok(Tensor::new(
        Arc::new(data),
        shape,
        dtype,
        device,
        requires_grad,
    ))
}

/// Initialize tensor with uniform distribution
pub fn init_uniform(
    shape: Shape,
    a: f64,
    b: f64,
    dtype: DataType,
    device: Device,
    requires_grad: bool,
) -> Result<Tensor> {
    let numel = shape.numel();
    let mut rng = thread_rng();
    let data = match dtype {
        DataType::Float32 => {
            let dist = Uniform::new(a as f32, b as f32);
            let mut vec = vec![0.0f32; numel];
            for v in &mut vec {
                *v = dist.sample(&mut rng);
            }
            TensorData::from_vec_f32(vec, device)
        }
        DataType::Float64 => {
            let dist = Uniform::new(a, b);
            let mut vec = vec![0.0f64; numel];
            for v in &mut vec {
                *v = dist.sample(&mut rng);
            }
            TensorData::from_vec_f64(vec, device)
        }
        DataType::Int32 => {
            let dist = Uniform::new(a as i32, b as i32);
            let mut vec = vec![0i32; numel];
            for v in &mut vec {
                *v = dist.sample(&mut rng);
            }
            TensorData::from_vec_i32(vec, device)
        }
        DataType::Int64 => {
            let dist = Uniform::new(a as i64, b as i64);
            let mut vec = vec![0i64; numel];
            for v in &mut vec {
                *v = dist.sample(&mut rng);rng));
            }
            TensorData::from_vec_i64(vec, device)
        }
        DataType::Bool => {
            let dist = Uniform::new(0.0, 1.0);
            let mut vec = vec![false; numel];
            for v in &mut vec {
                *v = dist.sample(&mut rng) > 0.5
            }
            TensorData::from_vec_bool(vec, device)
        }
    };
    Ok(Tensor::new(
        Arc::new(data),
        shape,
        dtype,
        device,
        requires_grad,
    ))
}

/// Initialize tensor with normal distribution
pub fn init_normal(
    shape: Shape,
    mean: f64,
    std: f64,
    dtype: DataType,
    device: Device,
    requires_grad: bool,
) -> Result<Tensor> {
    let numel = shape.numel();
    let mut rng = thread_rng();
    let data = match dtype {
        DataType::Float32 => {
            let dist = Normal::new(mean as f32, std as f32).unwrap();
            let mut vec = vec![0.0f32; numel];
            for v in &mut vec {
                *v = dist.sample(&mut rng);
            }
            TensorData::from_vec_f32(vec, device)
        }
        DataType::Float64 => {
            let dist = Normal::new(mean, std).unwrap();
            let mut vec = vec![0.0f64; numel];
            for v in &mut vec {
                *v = dist.sample(&mut rng);
            }
            TensorData::from_vec_f64(vec, device)
        }
        DataType::Int32 => {
            let dist = Normal::new(mean, std).unwrap();
            let mut vec = vec![0i32; numel];
            for v in &mut vec {
                *v = dist.sample(&mut rng).round() as i32;
            }
            TensorData::from_vec_i32(vec, device)
        }
        DataType::Int64 => {
            let dist = Normal::new(mean, std).unwrap();
            let mut vec = vec![0i64; numel];
            for v in &mut vec {
                *v = dist.sample(&mut rng).round() as i64;
            }
            TensorData::from_vec_i64(vec, device)
        }
        DataType::Bool => {
            let dist = Normal::new(mean, std).unwrap();
            let mut vec = vec![false; numel];
            for v in &mut vec {
                *v = dist.sample(&mut rng) > 0.0;
            }
            TensorData::from_vec_bool(vec, device)
        }
    };
    Ok(Tensor::new(
        Arc::new(data),
        shape,
        dtype,
        device,
        requires_grad,
    ))
}

/// Xavier/Glorot uniform initialization
/// Uniform distribution with bounds: sqrt(6 / (fan_in + fan_out))
pub fn xavier_uniform_init(
    shape: Shape,
    dtype: DataType,
    device: Device,
    requires_grad: bool,
) -> Result<Tensor> {
    let (fan_in, fan_out) = calculate_fan_in_fan_out(&shape)?;
    let bound = (6.0 / (fan_in + fan_out) as f64).sqrt();
    init_uniform(shape, -bound, bound, dtype, device, requires_grad)
}

/// Xavier/Glorot normal initialization
/// Normal distribution with std: sqrt(2 / (fan_in + fan_out))
pub fn xavier_normal_init(
    shape: Shape,
    dtype: DataType,
    device: Device,
    requires_grad: bool,
) -> Result<Tensor> {
    let (fan_in, fan_out) = calculate_fan_in_fan_out(&shape)?;
    let std = (2.0 / (fan_in + fan_out) as f64).sqrt();
    init_normal(shape, 0.0, std, dtype, device, requires_grad)
}

/// He uniform initialization (for ReLU networks)
/// Uniform distribution with bounds: sqrt(6 / fan_in)
pub fn he_uniform_init(
    shape: Shape,
    dtype: DataType,
    device: Device,
    requires_grad: bool,
) -> Result<Tensor> {
    let (fan_in, _) = calculate_fan_in_fan_out(&shape)?;
    let bound = (6.0 / fan_in as f64).sqrt();
    init_uniform(shape, -bound, bound, dtype, device, requires_grad)
}

/// He normal initialization (for ReLU networks)
/// Normal distribution with std: sqrt(2 / fan_in)
pub fn he_normal_init(
    shape: Shape,
    dtype: DataType,
    device: Device,
    requires_grad: bool,
) -> Result<Tensor> {
    let (fan_in, _) = calculate_fan_in_fan_out(&shape)?;
    let std = (2.0 / fan_in as f64).sqrt();
    init_normal(shape, 0.0, std, dtype, device, requires_grad)
}

/// LeCun uniform initialization
/// Uniform distribution with bounds: sqrt(3 / fan_in)
pub fn lecun_uniform_init(
    shape: Shape,
    dtype: DataType,
    device: Device,
    requires_grad: bool,
) -> Result<Tensor> {
    let (fan_in, _) = calculate_fan_in_fan_out(&shape)?;
    let bound = (3.0 / fan_in as f64).sqrt();
    init_uniform(shape, -bound, bound, dtype, device, requires_grad)
}

/// LeCun normal initialization
/// Normal distribution with std: sqrt(1 / fan_in)
pub fn lecun_normal_init(
    shape: Shape,
    dtype: DataType,
    device: Device,
    requires_grad: bool,
) -> Result<Tensor> {
    let (fan_in, _) = calculate_fan_in_fan_out(&shape)?;
    let std = (1.0 / fan_in as f64).sqrt();
    init_normal(shape, 0.0, std, dtype, device, requires_grad)
}

/// Calculate fan_in and fan_out for a tensor shape
fn calculate_fan_in_fan_out(shape: &Shape) -> Result<(usize, usize)> {
    let dims = shape.dims();

    match dims.len() {
        0 => Ok((1, 1)),             // Scalar
        1 => Ok((dims[0], dims[0])), // 1D tensor
        2 => Ok((dims[1], dims[0])), // 2D tensor (weight matrix)
        _ => {
            // For higher dimensional tensors (e.g., conv weights)
            let num_input_fmaps = dims[1];
            let num_output_fmaps = dims[0];
            let receptive_field_size: usize = dims[2..].iter().product();

            let fan_in = num_input_fmaps * receptive_field_size;
            let fan_out = num_output_fmaps * receptive_field_size;

            Ok((fan_in, fan_out))
        }
    }
}

/// Utility function to initialize a parameter tensor with a given method
pub fn init_parameter(
    shape: Shape,
    init_method: InitMethod,
    dtype: DataType,
    device: Device,
) -> Result<Tensor> {
    init_method.init_tensor(shape, dtype, device, true) // Parameters require gradients
}

/// Utility function to initialize a bias tensor (typically zeros)
pub fn init_bias(shape: Shape, dtype: DataType, device: Device) -> Result<Tensor> {
    InitMethod::Zeros.init_tensor(shape, dtype, device, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Shape;

    #[test]
    fn test_init_methods() {
        let shape = Shape::new(vec![3, 4]);
        let dtype = DataType::Float32;
        let device = Device::cpu();

        // Test zeros initialization
        let zeros = InitMethod::Zeros
            .init_tensor(shape.clone(), dtype, device, true)
            .unwrap();
        assert_eq!(zeros.shape(), &shape);
        assert!(zeros.requires_grad());

        // Test ones initialization
        let ones = InitMethod::Ones
            .init_tensor(shape.clone(), dtype, device, true)
            .unwrap();
        assert_eq!(ones.shape(), &shape);
        assert!(ones.requires_grad());
    }

    #[test]
    fn test_fan_in_fan_out_calculation() {
        // Test 2D tensor (dense layer weight)
        let shape_2d = Shape::new(vec![10, 5]); // output_size x input_size
        let (fan_in, fan_out) = calculate_fan_in_fan_out(&shape_2d).unwrap();
        assert_eq!(fan_in, 5);
        assert_eq!(fan_out, 10);

        // Test 4D tensor (conv layer weight)
        let shape_4d = Shape::new(vec![32, 16, 3, 3]); // out_channels x in_channels x kernel_h x kernel_w
        let (fan_in, fan_out) = calculate_fan_in_fan_out(&shape_4d).unwrap();
        assert_eq!(fan_in, 16 * 3 * 3); // in_channels * kernel_size
        assert_eq!(fan_out, 32 * 3 * 3); // out_channels * kernel_size
    }

    #[test]
    fn test_parameter_initialization() {
        let shape = Shape::new(vec![4, 3]);
        let dtype = DataType::Float32;
        let device = Device::cpu();

        // Test parameter initialization
        let param =
            init_parameter(shape.clone(), InitMethod::XavierUniform, dtype, device).unwrap();
        assert_eq!(param.shape(), &shape);
        assert!(param.requires_grad());

        // Test bias initialization
        let bias_shape = Shape::new(vec![4]);
        let bias = init_bias(bias_shape.clone(), dtype, device).unwrap();
        assert_eq!(bias.shape(), &bias_shape);
        assert!(bias.requires_grad());
    }

    #[test]
    fn test_uniform_range() {
        let shape = Shape::new(vec![100]);
        let tensor = init_uniform(
            shape.clone(),
            -0.5,
            0.5,
            DataType::Float32,
            Device::cpu(),
            false,
        )
        .unwrap();
        let slice = tensor.data().as_f32_slice().unwrap();
        for &v in slice {
            assert!(v >= -0.5 && v <= 0.5);
        }
    }

    #[test]
    fn test_normal_distribution_statistics() {
        let shape = Shape::new(vec![10_000]);
        let tensor = init_normal(shape, 0.0, 1.0, DataType::Float32, Device::cpu(), false).unwrap();
        let slice = tensor.data().as_f32_slice().unwrap();
        let mean: f32 = slice.iter().sum::<f32>() / slice.len() as f32;
        assert!(mean.abs() < 0.1);
    }
}

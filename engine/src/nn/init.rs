// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    device::Device,
    error::{MinitensorError, Result},
    random,
    tensor::{DataType, Shape, Tensor, TensorData},
};
use rand_distr::{Distribution, Normal, Uniform};
use statrs::distribution::{ContinuousCDF, Normal as StatrsNormal};
use std::sync::Arc;

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
            let vec = vec![value; numel];
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
    let data = match dtype {
        DataType::Float32 => {
            let dist = Uniform::new(a as f32, b as f32).unwrap();
            let mut vec = Vec::with_capacity(numel);
            random::with_rng(|rng| {
                vec.extend((0..numel).map(|_| dist.sample(rng)));
            });
            TensorData::from_vec_f32(vec, device)
        }
        DataType::Float64 => {
            let dist = Uniform::new(a, b).unwrap();
            let mut vec = Vec::with_capacity(numel);
            random::with_rng(|rng| {
                vec.extend((0..numel).map(|_| dist.sample(rng)));
            });
            TensorData::from_vec_f64(vec, device)
        }
        DataType::Int32 => {
            let dist = Uniform::new(a as i32, b as i32).unwrap();
            let mut vec = Vec::with_capacity(numel);
            random::with_rng(|rng| {
                vec.extend((0..numel).map(|_| dist.sample(rng)));
            });
            TensorData::from_vec_i32(vec, device)
        }
        DataType::Int64 => {
            let dist = Uniform::new(a as i64, b as i64).unwrap();
            let mut vec = Vec::with_capacity(numel);
            random::with_rng(|rng| {
                vec.extend((0..numel).map(|_| dist.sample(rng)));
            });
            TensorData::from_vec_i64(vec, device)
        }
        DataType::Bool => {
            let dist = Uniform::new(0.0, 1.0).unwrap();
            let mut vec = Vec::with_capacity(numel);
            random::with_rng(|rng| {
                vec.extend((0..numel).map(|_| dist.sample(rng) > 0.5));
            });
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
    let data = match dtype {
        DataType::Float32 => {
            let dist = Normal::new(mean as f32, std as f32).unwrap();
            let mut vec = Vec::with_capacity(numel);
            random::with_rng(|rng| {
                vec.extend((0..numel).map(|_| dist.sample(rng)));
            });
            TensorData::from_vec_f32(vec, device)
        }
        DataType::Float64 => {
            let dist = Normal::new(mean, std).unwrap();
            let mut vec = Vec::with_capacity(numel);
            random::with_rng(|rng| {
                vec.extend((0..numel).map(|_| dist.sample(rng)));
            });
            TensorData::from_vec_f64(vec, device)
        }
        DataType::Int32 => {
            let dist = Normal::new(mean, std).unwrap();
            let mut vec = Vec::with_capacity(numel);
            random::with_rng(|rng| {
                vec.extend((0..numel).map(|_| dist.sample(rng).round() as i32));
            });
            TensorData::from_vec_i32(vec, device)
        }
        DataType::Int64 => {
            let dist = Normal::new(mean, std).unwrap();
            let mut vec = Vec::with_capacity(numel);
            random::with_rng(|rng| {
                vec.extend((0..numel).map(|_| dist.sample(rng).round() as i64));
            });
            TensorData::from_vec_i64(vec, device)
        }
        DataType::Bool => {
            let dist = Normal::new(mean, std).unwrap();
            let mut vec = Vec::with_capacity(numel);
            random::with_rng(|rng| {
                vec.extend((0..numel).map(|_| dist.sample(rng) > 0.0));
            });
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

/// Initialize tensor with a normal distribution truncated to ``[lower, upper]``.
pub fn truncated_normal_init(
    shape: Shape,
    mean: f64,
    std: f64,
    lower: f64,
    upper: f64,
    dtype: DataType,
    device: Device,
    requires_grad: bool,
) -> Result<Tensor> {
    if !mean.is_finite() {
        return Err(MinitensorError::invalid_argument(
            "truncated_normal requires a finite mean",
        ));
    }

    if !std.is_finite() || std <= 0.0 {
        return Err(MinitensorError::invalid_argument(
            "truncated_normal requires a positive, finite std deviation",
        ));
    }

    if lower.is_nan() || upper.is_nan() {
        return Err(MinitensorError::invalid_argument(
            "truncated_normal requires non-NaN bounds",
        ));
    }

    if upper <= lower {
        return Err(MinitensorError::invalid_argument(
            "truncated_normal requires upper bound to be greater than lower bound",
        ));
    }

    let normal = StatrsNormal::new(mean, std).map_err(|err| {
        MinitensorError::invalid_argument(format!(
            "truncated_normal could not construct distribution: {err}",
        ))
    })?;

    // Inverse-CDF sampling needs `Phi(lower)` and `Phi(upper)` to be far enough
    // apart to tell sample points inside the interval apart. Above the mean the
    // CDF saturates at 1: for [8, 9] on a standard normal both endpoints round
    // to within a few ULPs of 1.0, so `inverse_cdf` was fed noise and returned
    // values below the requested lower bound (~8% of draws), and [10, 12] was
    // rejected outright as "zero probability mass".
    //
    // The normal is symmetric about its mean, so an interval lying above the
    // mean is sampled from its mirror image below the mean -- where the CDF has
    // full relative precision -- and each draw is reflected back. Same
    // distribution, computed where the arithmetic works.
    let reflected = lower + upper > 2.0 * mean;
    let (sample_lower, sample_upper) = if reflected {
        (2.0 * mean - upper, 2.0 * mean - lower)
    } else {
        (lower, upper)
    };

    let lower_cdf = normal.cdf(sample_lower);
    let upper_cdf = normal.cdf(sample_upper);

    if upper_cdf <= lower_cdf {
        return Err(MinitensorError::invalid_argument(
            "truncated_normal bounds must span non-zero probability mass",
        ));
    }

    let uniform = Uniform::new(lower_cdf, upper_cdf).map_err(|err| {
        MinitensorError::invalid_argument(format!(
            "truncated_normal could not construct sampler: {err}",
        ))
    })?;

    // One draw, in f64 throughout. The final clamp makes the requested bounds
    // exact: reflection and `inverse_cdf` are both accurate to a few ULPs, but
    // "a few ULPs outside the interval the caller asked for" is still outside.
    let draw = |rng: &mut _| {
        // Keep the draw strictly inside (0, 1); `inverse_cdf` is infinite at
        // the endpoints. The floor has to be the smallest positive double, not
        // machine epsilon: in the far tail (`[-40, -38]`) every legitimate
        // draw is orders of magnitude below `f64::EPSILON`, and flooring there
        // would collapse the whole tensor onto one bound.
        let sample_cdf = uniform.sample(rng);
        let sample_cdf = if sample_cdf <= 0.0 {
            f64::MIN_POSITIVE
        } else if sample_cdf >= 1.0 {
            1.0 - f64::EPSILON
        } else {
            sample_cdf
        };
        let value = normal.inverse_cdf(sample_cdf);
        let value = if reflected { 2.0 * mean - value } else { value };
        value.clamp(lower, upper)
    };

    let numel = shape.numel();
    let data = match dtype {
        DataType::Float32 => {
            let mut vec = Vec::with_capacity(numel);
            random::with_rng(|rng| vec.extend((0..numel).map(|_| draw(rng) as f32)));
            TensorData::from_vec_f32(vec, device)
        }
        DataType::Float64 => {
            let mut vec = Vec::with_capacity(numel);
            random::with_rng(|rng| vec.extend((0..numel).map(|_| draw(rng))));
            TensorData::from_vec_f64(vec, device)
        }
        _ => {
            return Err(MinitensorError::invalid_argument(
                "truncated_normal only supports float32 or float64 dtypes",
            ));
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

/// Calculate fan_in and fan_out for a tensor shape.
///
/// Follows the same convention as the layers here and as PyTorch: a weight is
/// stored `[out_features, in_features]`, so `fan_in` is the *trailing*
/// dimension, and a convolution weight's fans are scaled by its receptive
/// field. Public because a caller writing their own scheme needs the same
/// numbers the built-in initializers use -- deriving them independently is how
/// a hand-rolled initializer ends up transposed.
pub fn calculate_fan_in_fan_out(shape: &Shape) -> Result<(usize, usize)> {
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
    fn truncated_normal_keeps_its_bounds_in_both_tails() {
        // Sampling by inverting the CDF breaks above the mean, where `Phi`
        // saturates at 1: [8, 9] used to put ~8% of its draws below 8 and
        // never reach 9, and [10, 12] was rejected as spanning zero
        // probability mass. The mirrored intervals always worked, which is
        // what identifies the cause.
        for (lower, upper) in [
            (-1.0, 1.0),
            (8.0, 9.0),
            (-9.0, -8.0),
            (10.0, 12.0),
            (-12.0, -10.0),
            (20.0, 22.0),
            (-22.0, -20.0),
        ] {
            let tensor = truncated_normal_init(
                Shape::new(vec![4096]),
                0.0,
                1.0,
                lower,
                upper,
                DataType::Float64,
                Device::cpu(),
                false,
            )
            .unwrap_or_else(|err| panic!("[{lower}, {upper}] failed: {err}"));

            let values = tensor.data().as_f64_slice().unwrap();
            let min = values.iter().copied().fold(f64::INFINITY, f64::min);
            let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            assert!(
                min >= lower && max <= upper,
                "[{lower}, {upper}] produced [{min}, {max}]"
            );
            // Guarding the inverse CDF too aggressively keeps every value in
            // range while collapsing the tensor onto one bound, so check the
            // interval is actually covered rather than merely respected.
            assert!(
                max - min > (upper - lower) * 0.1,
                "[{lower}, {upper}] collapsed to [{min}, {max}]"
            );
        }
    }

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
            assert!((-0.5..=0.5).contains(&v));
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

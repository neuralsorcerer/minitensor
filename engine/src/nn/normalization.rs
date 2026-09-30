// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::{
    Layer,
    init::{InitMethod, init_parameter},
};
use crate::nn::layer::{FeatureAxis, check_feature_dim};
use crate::{
    device::Device,
    error::{MinitensorError, Result},
    tensor::{DataType, Shape, Tensor},
};
use std::collections::HashMap;

/// State and arithmetic shared by [`BatchNorm1d`] and [`BatchNorm2d`], which
/// differ only in the input rank they accept.
///
/// The scale and shift are optional: without them (`affine = false`) the layer
/// only normalizes, and has no parameters for an optimizer to train.
#[derive(Clone)]
struct BatchNormCore {
    weight: Option<Tensor>, // γ (gamma) - learnable scale
    bias: Option<Tensor>,   // β (beta) - learnable shift
    running_mean: Tensor,
    running_var: Tensor,
    num_features: usize,
    eps: f64,
    momentum: f64,
    training: bool,
}

impl BatchNormCore {
    fn new(
        num_features: usize,
        eps: Option<f64>,
        momentum: Option<f64>,
        affine: bool,
        device: Device,
        dtype: DataType,
    ) -> Result<Self> {
        let eps = eps.unwrap_or(1e-5);
        let momentum = momentum.unwrap_or(0.1);
        crate::ops::normalization::check_eps("BatchNorm", eps)?;
        crate::ops::normalization::check_momentum("BatchNorm", momentum)?;

        let param_shape = Shape::new(vec![num_features]);
        let (weight, bias) = if affine {
            (
                Some(init_parameter(
                    param_shape.clone(),
                    InitMethod::Ones,
                    dtype,
                    device,
                )?),
                Some(init_parameter(
                    param_shape.clone(),
                    InitMethod::Zeros,
                    dtype,
                    device,
                )?),
            )
        } else {
            (None, None)
        };
        // Running statistics are state, not parameters: they take no gradient.
        let running_mean = Tensor::zeros(param_shape.clone(), dtype, device, false);
        let running_var = Tensor::ones(param_shape, dtype, device, false);

        Ok(Self {
            weight,
            bias,
            running_mean,
            running_var,
            num_features,
            eps,
            momentum,
            training: true,
        })
    }

    /// Normalize through the functional kernel, which stores the *unbiased*
    /// batch variance in `running_var`, so the layer and `batch_norm` agree
    /// about eval-time normalization for the same input.
    fn forward(&mut self, input: &Tensor, layer: &str) -> Result<Tensor> {
        check_feature_dim(
            layer,
            "num_features",
            self.num_features,
            input,
            FeatureAxis::At(1),
        )?;
        crate::ops::normalization::batch_norm(
            input,
            Some(&mut self.running_mean),
            Some(&mut self.running_var),
            self.weight.as_ref(),
            self.bias.as_ref(),
            self.training,
            self.momentum,
            self.eps,
        )
    }

    fn named_parameters(&self) -> HashMap<String, &Tensor> {
        let mut params = HashMap::with_capacity(2);
        if let Some(weight) = &self.weight {
            params.insert("weight".to_string(), weight);
        }
        if let Some(bias) = &self.bias {
            params.insert("bias".to_string(), bias);
        }
        params
    }

    fn named_parameters_mut(&mut self) -> HashMap<String, &mut Tensor> {
        let mut params = HashMap::with_capacity(2);
        if let Some(weight) = &mut self.weight {
            params.insert("weight".to_string(), weight);
        }
        if let Some(bias) = &mut self.bias {
            params.insert("bias".to_string(), bias);
        }
        params
    }

    /// Must mirror [`Self::named_buffers_mut`]: saving under names and loading
    /// by position would put `running_var` wherever the order placed it.
    fn named_buffers(&self) -> HashMap<String, &Tensor> {
        let mut buffers = HashMap::with_capacity(2);
        buffers.insert("running_mean".to_string(), &self.running_mean);
        buffers.insert("running_var".to_string(), &self.running_var);
        buffers
    }

    fn named_buffers_mut(&mut self) -> HashMap<String, &mut Tensor> {
        let mut buffers = HashMap::with_capacity(2);
        buffers.insert("running_mean".to_string(), &mut self.running_mean);
        buffers.insert("running_var".to_string(), &mut self.running_var);
        buffers
    }

    fn parameters(&self) -> Vec<&Tensor> {
        self.weight.iter().chain(self.bias.iter()).collect()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        self.weight.iter_mut().chain(self.bias.iter_mut()).collect()
    }

    /// Order must stay stable: it defines the indexed buffer names used for
    /// serialization (buffer_0 = running_mean, buffer_1 = running_var).
    fn buffers(&self) -> Vec<&Tensor> {
        vec![&self.running_mean, &self.running_var]
    }

    fn buffers_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.running_mean, &mut self.running_var]
    }
}

/// The methods `BatchNorm1d` and `BatchNorm2d` share, forwarded to their core.
macro_rules! batch_norm_layer {
    ($name:ident, $doc_input:literal, $check_rank:expr, $rank_message:literal) => {
        impl $name {
            #[doc = concat!("Create a batch normalization layer over ", $doc_input, ", with a")]
            /// learnable scale and shift.
            ///
            /// # Arguments
            /// * `num_features` - The channel count C, the size of axis 1
            /// * `eps` - Added to the variance for numerical stability. Default: 1e-5
            /// * `momentum` - Weight of each batch in the running statistics. Default: 0.1
            /// * `device` - Device to place the layer parameters on
            /// * `dtype` - Data type for the layer parameters
            pub fn new(
                num_features: usize,
                eps: Option<f64>,
                momentum: Option<f64>,
                device: Device,
                dtype: DataType,
            ) -> Result<Self> {
                Self::with_affine(num_features, eps, momentum, true, device, dtype)
            }

            /// As [`Self::new`], with `affine` saying whether the layer has a
            /// learnable scale and shift. Without them it only normalizes.
            pub fn with_affine(
                num_features: usize,
                eps: Option<f64>,
                momentum: Option<f64>,
                affine: bool,
                device: Device,
                dtype: DataType,
            ) -> Result<Self> {
                Ok(Self {
                    core: BatchNormCore::new(num_features, eps, momentum, affine, device, dtype)?,
                })
            }

            /// Get number of features
            pub fn num_features(&self) -> usize {
                self.core.num_features
            }

            /// The value added to the variance for numerical stability.
            pub fn eps(&self) -> f64 {
                self.core.eps
            }

            /// The weight each batch gets in the running statistics.
            pub fn momentum(&self) -> f64 {
                self.core.momentum
            }

            /// Whether the layer has a learnable scale and shift.
            pub fn affine(&self) -> bool {
                self.core.weight.is_some()
            }

            /// Whether the layer normalizes with batch statistics.
            pub fn is_training(&self) -> bool {
                self.core.training
            }

            /// The scale (gamma), when the layer is affine.
            pub fn weight(&self) -> Option<&Tensor> {
                self.core.weight.as_ref()
            }

            /// The shift (beta), when the layer is affine.
            pub fn bias(&self) -> Option<&Tensor> {
                self.core.bias.as_ref()
            }

            /// Get the running mean tensor
            pub fn running_mean(&self) -> &Tensor {
                &self.core.running_mean
            }

            /// Get the running variance tensor
            pub fn running_var(&self) -> &Tensor {
                &self.core.running_var
            }

            /// Set training mode
            pub fn train(&mut self) {
                self.core.training = true;
            }

            /// Set evaluation mode
            pub fn eval(&mut self) {
                self.core.training = false;
            }
        }

        impl Layer for $name {
            crate::nn::layer::cloneable_layer!();

            fn named_parameters(&self) -> HashMap<String, &Tensor> {
                self.core.named_parameters()
            }

            fn named_parameters_mut(&mut self) -> HashMap<String, &mut Tensor> {
                self.core.named_parameters_mut()
            }

            fn named_buffers(&self) -> HashMap<String, &Tensor> {
                self.core.named_buffers()
            }

            fn named_buffers_mut(&mut self) -> HashMap<String, &mut Tensor> {
                self.core.named_buffers_mut()
            }

            fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
                let check_rank: fn(usize) -> bool = $check_rank;
                if !check_rank(input.ndim()) {
                    return Err(MinitensorError::invalid_operation($rank_message));
                }
                self.core.forward(input, stringify!($name))
            }

            fn parameters(&self) -> Vec<&Tensor> {
                self.core.parameters()
            }

            fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
                self.core.parameters_mut()
            }

            fn buffers(&self) -> Vec<&Tensor> {
                self.core.buffers()
            }

            fn buffers_mut(&mut self) -> Vec<&mut Tensor> {
                self.core.buffers_mut()
            }

            fn train(&mut self) {
                self.core.training = true;
            }

            fn eval(&mut self) {
                self.core.training = false;
            }
        }
    };
}

/// 1D Batch normalization layer
///
/// Applies Batch Normalization over a 2D or 3D input (a mini-batch of 1D inputs
/// with optional additional channel dimension).
///
/// The mean and standard-deviation are calculated per-dimension over the mini-batches
/// and γ and β are learnable parameter vectors of size C (where C is the input size).
#[derive(Clone)]
pub struct BatchNorm1d {
    core: BatchNormCore,
}

batch_norm_layer!(
    BatchNorm1d,
    "a 2D `[N, C]` or 3D `[N, C, L]` input",
    |ndim| (2..=3).contains(&ndim),
    "BatchNorm1d expects 2D input [batch_size, features] or 3D input [batch_size, features, length]"
);

/// 2D Batch normalization layer for convolutional layers
///
/// Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs
/// with additional channel dimension).
#[derive(Clone)]
pub struct BatchNorm2d {
    core: BatchNormCore,
}

batch_norm_layer!(
    BatchNorm2d,
    "a 4D `[N, C, H, W]` input",
    |ndim| ndim == 4,
    "BatchNorm2d expects 4D input [batch_size, channels, height, width]"
);

/// Shared validation for the shape-normalizing layers: `normalized_shape` must
/// be non-empty and match the trailing dimensions of the input.
fn check_normalized_suffix(input: &Tensor, normalized_shape: &[usize], layer: &str) -> Result<()> {
    let dims = input.shape().dims();
    if dims.len() < normalized_shape.len() {
        return Err(MinitensorError::invalid_operation(format!(
            "{} expects an input with at least {} dimensions, got {}",
            layer,
            normalized_shape.len(),
            dims.len()
        )));
    }
    let suffix = &dims[dims.len() - normalized_shape.len()..];
    if suffix != normalized_shape {
        // `shape_mismatch(normalized_shape, dims)` compared a suffix against a
        // whole shape, so `LayerNorm([512])` on a `[4, 256]` input read
        // "expected [512], got [4, 256]" -- as if the input were meant to be
        // one-dimensional. Only the trailing dimensions are in question.
        let n = normalized_shape.len();
        return Err(MinitensorError::invalid_argument_with_suggestion(
            format!(
                "{layer} normalizes over the last {n} dimension{} {normalized_shape:?}, \
                 but the input's last {n} dimension{} {suffix:?} (input shape {dims:?})",
                if n == 1 { "" } else { "s" },
                if n == 1 { " is" } else { "s are" },
            ),
            format!(
                "Either construct the layer with normalized_shape={suffix:?}, or give it \
                 an input ending in {normalized_shape:?}"
            ),
        ));
    }
    Ok(())
}

/// Layer normalization (Ba et al., 2016) as a stateful layer.
///
/// Normalizes over the trailing `normalized_shape` dimensions using that
/// slice's own mean and variance, then applies a learned elementwise scale and
/// shift. Unlike BatchNorm it carries no running statistics and behaves
/// identically in training and evaluation.
#[derive(Clone)]
pub struct LayerNorm {
    weight: Option<Tensor>,
    bias: Option<Tensor>,
    normalized_shape: Vec<usize>,
    eps: f64,
}

impl LayerNorm {
    /// Create a new LayerNorm over the trailing `normalized_shape` dimensions.
    pub fn new(
        normalized_shape: Vec<usize>,
        eps: Option<f64>,
        elementwise_affine: bool,
        device: Device,
        dtype: DataType,
    ) -> Result<Self> {
        if normalized_shape.is_empty() {
            return Err(MinitensorError::invalid_argument(
                "LayerNorm requires normalized_shape to contain at least one dimension",
            ));
        }
        if let Some(eps) = eps {
            crate::ops::normalization::check_eps("LayerNorm", eps)?;
        }
        if !dtype.is_float() {
            return Err(MinitensorError::invalid_argument(
                "LayerNorm parameters must have a floating point dtype",
            ));
        }

        let (weight, bias) = if elementwise_affine {
            let shape = Shape::new(normalized_shape.clone());
            (
                Some(init_parameter(
                    shape.clone(),
                    InitMethod::Ones,
                    dtype,
                    device,
                )?),
                Some(init_parameter(shape, InitMethod::Zeros, dtype, device)?),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            weight,
            bias,
            normalized_shape,
            eps: eps.unwrap_or(1e-5),
        })
    }

    /// Dimensions this layer normalizes over.
    pub fn normalized_shape(&self) -> &[usize] {
        &self.normalized_shape
    }

    /// Numerical stability epsilon.
    pub fn eps(&self) -> f64 {
        self.eps
    }

    /// Whether a learned scale and shift are applied.
    pub fn elementwise_affine(&self) -> bool {
        self.weight.is_some()
    }
}

impl Layer for LayerNorm {
    crate::nn::layer::cloneable_layer!();

    /// Named parameters for serialization.
    fn named_parameters(&self) -> HashMap<String, &Tensor> {
        let mut params = HashMap::new();
        if let Some(ref w) = self.weight {
            params.insert("weight".to_string(), w);
        }
        if let Some(ref b) = self.bias {
            params.insert("bias".to_string(), b);
        }
        params
    }
    /// Named mutable parameters for state-dict loading.
    fn named_parameters_mut(&mut self) -> HashMap<String, &mut Tensor> {
        let mut params = HashMap::new();
        if let Some(ref mut w) = self.weight {
            params.insert("weight".to_string(), w);
        }
        if let Some(ref mut b) = self.bias {
            params.insert("bias".to_string(), b);
        }
        params
    }

    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        check_normalized_suffix(input, &self.normalized_shape, "LayerNorm")?;
        crate::ops::normalization::layer_norm(
            input,
            &self.normalized_shape,
            self.weight.as_ref(),
            self.bias.as_ref(),
            self.eps,
        )
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        if let Some(ref w) = self.weight {
            params.push(w);
        }
        if let Some(ref b) = self.bias {
            params.push(b);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        if let Some(ref mut w) = self.weight {
            params.push(w);
        }
        if let Some(ref mut b) = self.bias {
            params.push(b);
        }
        params
    }
}

/// Root-mean-square layer normalization (Zhang & Sennrich, 2019) as a stateful
/// layer — the normalization used by LLaMA, Mistral, Gemma and Qwen.
///
/// Rescales by the root mean square over the trailing `normalized_shape`
/// dimensions and applies a learned gain. There is no mean subtraction and no
/// bias, which makes it cheaper than LayerNorm while matching its quality on
/// large language models.
#[derive(Clone)]
pub struct RMSNorm {
    weight: Option<Tensor>,
    normalized_shape: Vec<usize>,
    eps: f64,
}

impl RMSNorm {
    /// Create a new RMSNorm over the trailing `normalized_shape` dimensions.
    pub fn new(
        normalized_shape: Vec<usize>,
        eps: Option<f64>,
        elementwise_affine: bool,
        device: Device,
        dtype: DataType,
    ) -> Result<Self> {
        if normalized_shape.is_empty() {
            return Err(MinitensorError::invalid_argument(
                "RMSNorm requires normalized_shape to contain at least one dimension",
            ));
        }
        if let Some(eps) = eps {
            crate::ops::normalization::check_eps("RMSNorm", eps)?;
        }
        if !dtype.is_float() {
            return Err(MinitensorError::invalid_argument(
                "RMSNorm parameters must have a floating point dtype",
            ));
        }

        let weight = if elementwise_affine {
            Some(init_parameter(
                Shape::new(normalized_shape.clone()),
                InitMethod::Ones,
                dtype,
                device,
            )?)
        } else {
            None
        };

        Ok(Self {
            weight,
            normalized_shape,
            eps: eps.unwrap_or(1e-6),
        })
    }

    /// Dimensions this layer normalizes over.
    pub fn normalized_shape(&self) -> &[usize] {
        &self.normalized_shape
    }

    /// Numerical stability epsilon.
    pub fn eps(&self) -> f64 {
        self.eps
    }

    /// Whether a learned gain is applied.
    pub fn elementwise_affine(&self) -> bool {
        self.weight.is_some()
    }
}

impl Layer for RMSNorm {
    crate::nn::layer::cloneable_layer!();

    /// Named parameters for serialization.
    fn named_parameters(&self) -> HashMap<String, &Tensor> {
        let mut params = HashMap::new();
        if let Some(ref w) = self.weight {
            params.insert("weight".to_string(), w);
        }
        params
    }
    /// Named mutable parameters for state-dict loading.
    fn named_parameters_mut(&mut self) -> HashMap<String, &mut Tensor> {
        let mut params = HashMap::new();
        if let Some(ref mut w) = self.weight {
            params.insert("weight".to_string(), w);
        }
        params
    }

    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        check_normalized_suffix(input, &self.normalized_shape, "RMSNorm")?;
        crate::ops::normalization::rms_norm(
            input,
            &self.normalized_shape,
            self.weight.as_ref(),
            self.eps,
        )
    }

    fn parameters(&self) -> Vec<&Tensor> {
        self.weight.as_ref().into_iter().collect()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        self.weight.as_mut().into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;
    use crate::nn::Module;
    use crate::tensor::{DataType, Shape, TensorData};
    use std::sync::Arc;

    #[test]
    fn test_batchnorm_state_dict_includes_running_stats() {
        // Running stats are buffers, so the state dict must carry them (2
        // parameters + 2 buffers), otherwise a reloaded model loses its
        // inference statistics.
        let layer =
            BatchNorm1d::new(8, Some(1e-5), Some(0.1), Device::cpu(), DataType::Float32).unwrap();
        assert_eq!(layer.buffers().len(), 2);
        let sd = layer.state_dict();
        assert_eq!(sd.parameters.len(), 2);
        assert_eq!(
            sd.buffers.len(),
            2,
            "running_mean and running_var must serialize"
        );
        // Named, not positional: a checkpoint that says `buffer_0` cannot be
        // checked by eye, and cannot survive a change in buffer order.
        assert!(sd.buffers.contains_key("running_mean"));
        assert!(sd.buffers.contains_key("running_var"));
        assert!(sd.parameters.contains_key("weight"));
        assert!(sd.parameters.contains_key("bias"));
    }

    #[test]
    fn test_batchnorm1d_creation() {
        let layer =
            BatchNorm1d::new(128, Some(1e-5), Some(0.1), Device::cpu(), DataType::Float32).unwrap();

        assert_eq!(layer.num_features(), 128);
        assert_eq!(layer.eps(), 1e-5);
        assert_eq!(layer.momentum(), 0.1);
        assert!(layer.is_training());
        assert_eq!(layer.weight().unwrap().shape(), &Shape::new(vec![128]));
        assert_eq!(layer.bias().unwrap().shape(), &Shape::new(vec![128]));
        assert_eq!(layer.running_mean().shape(), &Shape::new(vec![128]));
        assert_eq!(layer.running_var().shape(), &Shape::new(vec![128]));
    }

    #[test]
    fn test_batchnorm1d_training_mode() {
        let mut layer =
            BatchNorm1d::new(128, None, None, Device::cpu(), DataType::Float32).unwrap();

        assert!(layer.is_training());

        layer.eval();
        assert!(!layer.is_training());

        layer.train();
        assert!(layer.is_training());
    }

    #[test]
    fn test_batchnorm1d_parameters() {
        let mut layer =
            BatchNorm1d::new(128, None, None, Device::cpu(), DataType::Float32).unwrap();

        let params = layer.parameters();
        assert_eq!(params.len(), 2); // weight + bias

        let mut_params = layer.parameters_mut();
        assert_eq!(mut_params.len(), 2);

        let named_params = layer.named_parameters();
        assert_eq!(named_params.len(), 2);
        assert!(named_params.contains_key("weight"));
        assert!(named_params.contains_key("bias"));

        let buffers = layer.named_buffers();
        assert_eq!(buffers.len(), 2);
        assert!(buffers.contains_key("running_mean"));
        assert!(buffers.contains_key("running_var"));
    }

    #[test]
    fn test_batchnorm1d_forward_shape_validation() {
        let mut layer =
            BatchNorm1d::new(128, None, None, Device::cpu(), DataType::Float32).unwrap();

        // Test with correct 2D input [batch=32, features=128]
        let input_2d = Tensor::zeros(
            Shape::new(vec![32, 128]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let output = layer.forward(&input_2d).unwrap();
        assert_eq!(output.shape(), input_2d.shape());

        // Test with correct 3D input [batch=32, features=128, length=10]
        let input_3d = Tensor::zeros(
            Shape::new(vec![32, 128, 10]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let output = layer.forward(&input_3d).unwrap();
        assert_eq!(output.shape(), input_3d.shape());

        // Test with incorrect number of features
        let wrong_input = Tensor::zeros(
            Shape::new(vec![32, 64]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let result = layer.forward(&wrong_input);
        assert!(result.is_err());

        // Test with wrong number of dimensions
        let wrong_dim_input = Tensor::zeros(
            Shape::new(vec![128]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let result = layer.forward(&wrong_dim_input);
        assert!(result.is_err());
    }

    #[test]
    fn test_batchnorm1d_running_stats() {
        let mut layer = BatchNorm1d::new(2, None, None, Device::cpu(), DataType::Float32).unwrap();

        let data = TensorData::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], Device::cpu());
        let input = Tensor::new(
            Arc::new(data),
            Shape::new(vec![2, 2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );

        layer.forward(&input).unwrap();
        assert!(layer.running_mean().data().as_f32_slice().unwrap()[0] != 0.0);

        let mean_before = layer.running_mean().clone();
        layer.eval();
        layer.forward(&input).unwrap();
        let before = mean_before.data().as_f32_slice().unwrap()[0];
        let after = layer.running_mean().data().as_f32_slice().unwrap()[0];
        assert!((after - before).abs() < 1e-6);
    }

    #[test]
    fn test_batchnorm1d_eval_uses_running_stats() {
        // With momentum = 1.0 a single training forward sets the running stats
        // to exactly that batch's statistics (mean, and the UNBIASED variance
        // `running_var` stores). Eval must then normalize a *different* batch with
        // those running stats, independent of the eval batch — not recompute
        // batch statistics.
        let mut layer =
            BatchNorm1d::new(2, Some(0.0), Some(1.0), Device::cpu(), DataType::Float32).unwrap();

        // feature 0 = [1, 3] -> mean 2, var 1; feature 1 = [2, 4] -> mean 3, var 1.
        let train = Tensor::new(
            Arc::new(TensorData::from_vec_f32(
                vec![1.0, 2.0, 3.0, 4.0],
                Device::cpu(),
            )),
            Shape::new(vec![2, 2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        layer.forward(&train).unwrap();

        layer.eval();
        // A different batch with different statistics.
        let test = Tensor::new(
            Arc::new(TensorData::from_vec_f32(
                vec![5.0, 6.0, 7.0, 8.0],
                Device::cpu(),
            )),
            Shape::new(vec![2, 2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let out = layer.forward(&test).unwrap();
        let o = out.data().as_f32_slice().unwrap();
        // (x - running_mean) / sqrt(running_var + eps=0), running_mean=[2,3].
        // n = 2, so the stored variance is the unbiased 1 * 2/1 = 2.
        // row0: (5-2, 6-3) = (3, 3); row1: (7-2, 8-3) = (5, 5), each / sqrt(2).
        let scale = 2.0f32.sqrt();
        assert!((o[0] - 3.0 / scale).abs() < 1e-4, "{}", o[0]);
        assert!((o[1] - 3.0 / scale).abs() < 1e-4, "{}", o[1]);
        assert!((o[2] - 5.0 / scale).abs() < 1e-4, "{}", o[2]);
        assert!((o[3] - 5.0 / scale).abs() < 1e-4, "{}", o[3]);
    }

    #[test]
    fn test_batchnorm1d_3d_per_channel_statistics() {
        // momentum = 1.0 makes running stats equal the batch statistics, so we
        // can read the per-channel mean directly. Channel 0 = [1, 2, 3, 4]
        // (mean 2.5), channel 1 = [10, 20, 30, 40] (mean 25). A flat reshape
        // of [N, C, L] to [N*L, C] would interleave the channels and produce
        // wrong statistics here.
        let mut layer =
            BatchNorm1d::new(2, Some(0.0), Some(1.0), Device::cpu(), DataType::Float32).unwrap();
        let input = Tensor::new(
            Arc::new(TensorData::from_vec_f32(
                vec![1.0, 2.0, 10.0, 20.0, 3.0, 4.0, 30.0, 40.0],
                Device::cpu(),
            )),
            Shape::new(vec![2, 2, 2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        layer.forward(&input).unwrap();

        let mean = layer.running_mean().data().as_f32_slice().unwrap();
        let var = layer.running_var().data().as_f32_slice().unwrap();
        assert!((mean[0] - 2.5).abs() < 1e-5, "channel 0 mean: {}", mean[0]);
        assert!((mean[1] - 25.0).abs() < 1e-4, "channel 1 mean: {}", mean[1]);
        // `running_var` stores the UNBIASED batch variance (Bessel's
        // correction), while the normalization itself uses the biased one.
        // n = N * L = 4, so biased 1.25 -> 1.25 * 4/3, and 125 -> 125 * 4/3.
        assert!(
            (var[0] - 1.25 * 4.0 / 3.0).abs() < 1e-5,
            "channel 0 var: {}",
            var[0]
        );
        assert!(
            (var[1] - 125.0 * 4.0 / 3.0).abs() < 1e-3,
            "channel 1 var: {}",
            var[1]
        );
    }

    #[test]
    fn test_batchnorm2d_per_channel_statistics() {
        // Channel 0 = [1..4] (mean 2.5, var 1.25); channel 1 = [5..8]
        // (mean 6.5, var 1.25). With momentum = 1.0 the running stats equal
        // the batch statistics after one forward pass.
        let mut layer =
            BatchNorm2d::new(2, Some(0.0), Some(1.0), Device::cpu(), DataType::Float32).unwrap();
        let input = Tensor::new(
            Arc::new(TensorData::from_vec_f32(
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                Device::cpu(),
            )),
            Shape::new(vec![1, 2, 2, 2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let out = layer.forward(&input).unwrap();

        let mean = layer.named_buffers()["running_mean"]
            .data()
            .as_f32_slice()
            .unwrap();
        let var = layer.named_buffers()["running_var"]
            .data()
            .as_f32_slice()
            .unwrap();
        assert!((mean[0] - 2.5).abs() < 1e-5, "channel 0 mean: {}", mean[0]);
        assert!((mean[1] - 6.5).abs() < 1e-5, "channel 1 mean: {}", mean[1]);
        // Unbiased (Bessel-corrected) variance in the running buffer:
        // n = N * H * W = 4, so biased 1.25 -> 1.25 * 4/3.
        assert!(
            (var[0] - 1.25 * 4.0 / 3.0).abs() < 1e-5,
            "channel 0 var: {}",
            var[0]
        );
        assert!(
            (var[1] - 1.25 * 4.0 / 3.0).abs() < 1e-5,
            "channel 1 var: {}",
            var[1]
        );

        // Each channel is normalized with its own statistics.
        let o = out.data().as_f32_slice().unwrap();
        let expected = [-1.3416407f32, -0.4472136, 0.4472136, 1.3416407];
        for c in 0..2 {
            for i in 0..4 {
                assert!(
                    (o[c * 4 + i] - expected[i]).abs() < 1e-4,
                    "channel {c} element {i}: {}",
                    o[c * 4 + i]
                );
            }
        }
    }

    #[test]
    fn test_batchnorm_layers_agree_with_functional_batch_norm() {
        // The layers delegate to `ops::normalization::batch_norm`; this pins
        // that they cannot drift apart again. They previously stored the
        // *biased* batch variance in `running_var` while the functional kernel
        // stored the unbiased one, so the same input normalized differently at
        // eval time depending on which entry point the caller used.
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let make = |dims: Vec<usize>| {
            Tensor::new(
                Arc::new(TensorData::from_vec_f32(values.clone(), Device::cpu())),
                Shape::new(dims),
                DataType::Float32,
                Device::cpu(),
                false,
            )
        };

        for (dims, is_2d) in [
            (vec![4usize, 2usize], false),
            (vec![2, 2, 2], false),
            (vec![1, 2, 2, 2], true),
        ] {
            let input = make(dims.clone());

            let mut rm =
                Tensor::zeros(Shape::new(vec![2]), DataType::Float32, Device::cpu(), false);
            let mut rv = Tensor::ones(Shape::new(vec![2]), DataType::Float32, Device::cpu(), false);
            let weight = Tensor::ones(Shape::new(vec![2]), DataType::Float32, Device::cpu(), false);
            let bias = Tensor::zeros(Shape::new(vec![2]), DataType::Float32, Device::cpu(), false);
            let functional_out = crate::ops::normalization::batch_norm(
                &input,
                Some(&mut rm),
                Some(&mut rv),
                Some(&weight),
                Some(&bias),
                true,
                0.1,
                1e-5,
            )
            .unwrap();

            let layer_out = if is_2d {
                let mut layer =
                    BatchNorm2d::new(2, Some(1e-5), Some(0.1), Device::cpu(), DataType::Float32)
                        .unwrap();
                let out = layer.forward(&input).unwrap();
                let buffers = layer.named_buffers();
                assert_eq!(
                    buffers["running_var"].data().as_f32_slice().unwrap(),
                    rv.data().as_f32_slice().unwrap(),
                    "running_var mismatch for {dims:?}"
                );
                assert_eq!(
                    buffers["running_mean"].data().as_f32_slice().unwrap(),
                    rm.data().as_f32_slice().unwrap(),
                    "running_mean mismatch for {dims:?}"
                );
                out
            } else {
                let mut layer =
                    BatchNorm1d::new(2, Some(1e-5), Some(0.1), Device::cpu(), DataType::Float32)
                        .unwrap();
                let out = layer.forward(&input).unwrap();
                assert_eq!(
                    layer.running_var().data().as_f32_slice().unwrap(),
                    rv.data().as_f32_slice().unwrap(),
                    "running_var mismatch for {dims:?}"
                );
                assert_eq!(
                    layer.running_mean().data().as_f32_slice().unwrap(),
                    rm.data().as_f32_slice().unwrap(),
                    "running_mean mismatch for {dims:?}"
                );
                out
            };

            assert_eq!(
                layer_out.data().as_f32_slice().unwrap(),
                functional_out.data().as_f32_slice().unwrap(),
                "output mismatch for {dims:?}"
            );
        }
    }

    #[test]
    fn test_batchnorm2d_creation() {
        let layer =
            BatchNorm2d::new(64, Some(1e-5), Some(0.1), Device::cpu(), DataType::Float32).unwrap();

        assert_eq!(layer.num_features(), 64);
    }

    #[test]
    fn test_batchnorm2d_forward_shape_validation() {
        let mut layer = BatchNorm2d::new(64, None, None, Device::cpu(), DataType::Float32).unwrap();

        // Test with correct 4D input [batch=16, channels=64, height=32, width=32]
        let input = Tensor::zeros(
            Shape::new(vec![16, 64, 32, 32]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), input.shape());

        // Test with incorrect number of channels
        let wrong_input = Tensor::zeros(
            Shape::new(vec![16, 32, 32, 32]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let result = layer.forward(&wrong_input);
        assert!(result.is_err());

        // Test with wrong number of dimensions
        let wrong_dim_input = Tensor::zeros(
            Shape::new(vec![16, 64, 32]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let result = layer.forward(&wrong_dim_input);
        assert!(result.is_err());
    }

    #[test]
    fn test_batchnorm2d_running_stats() {
        let mut layer = BatchNorm2d::new(2, None, None, Device::cpu(), DataType::Float32).unwrap();

        let data =
            TensorData::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], Device::cpu());
        let input = Tensor::new(
            Arc::new(data),
            Shape::new(vec![1, 2, 2, 2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );

        layer.forward(&input).unwrap();
        assert!(
            layer.named_buffers()["running_mean"]
                .data()
                .as_f32_slice()
                .unwrap()[0]
                != 0.0
        );

        let mean_before = layer.named_buffers()["running_mean"].clone();
        layer.eval();
        let data2 = TensorData::from_vec_f32(
            vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            Device::cpu(),
        );
        let input2 = Tensor::new(
            Arc::new(data2),
            Shape::new(vec![1, 2, 2, 2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        layer.forward(&input2).unwrap();
        let before = mean_before.data().as_f32_slice().unwrap()[0];
        let after = layer.named_buffers()["running_mean"]
            .data()
            .as_f32_slice()
            .unwrap()[0];
        assert!((after - before).abs() < 1e-6);
    }

    #[test]
    fn test_batchnorm2d_inference_output() {
        let mut layer =
            BatchNorm2d::new(1, Some(1e-5), Some(1.0), Device::cpu(), DataType::Float32).unwrap();

        // First batch to set running statistics
        let data1 = TensorData::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], Device::cpu());
        let input1 = Tensor::new(
            Arc::new(data1),
            Shape::new(vec![1, 1, 2, 2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        layer.forward(&input1).unwrap();

        // Inference with different input should use stored running stats
        layer.eval();
        let data2 = TensorData::from_vec_f32(vec![5.0, 5.0, 5.0, 5.0], Device::cpu());
        let input2 = Tensor::new(
            Arc::new(data2),
            Shape::new(vec![1, 1, 2, 2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let output = layer.forward(&input2).unwrap();

        // n = H * W = 4, so the stored (unbiased) variance is 1.25 * 4/3.
        let running_var = 1.25f32 * 4.0 / 3.0;
        let expected = (5.0 - 2.5) / (running_var + layer.eps() as f32).sqrt();
        let out_slice = output.data().as_f32_slice().unwrap();
        assert!(out_slice.iter().all(|&v| (v - expected).abs() < 1e-4));
    }
}

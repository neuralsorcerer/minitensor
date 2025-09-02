// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    tensor::{Tensor, Shape, TensorData},
    autograd::TensorId,
    error::{Result, MinitensorError},
    operations::{arithmetic, reduction, linalg, activation},
    tensor::DataType,
    device::Device,
};
use std::sync::Arc;

/// Handle gradient broadcasting by summing over broadcasted dimensions
fn handle_broadcast_gradient(grad_output: &Tensor, target_shape: &[usize]) -> Result<Tensor> {
    let target_shape_obj = Shape::new(target_shape.to_vec());

    // If shapes are the same, no broadcasting occurred
    if grad_output.shape().dims() == target_shape {
        return Ok(grad_output.clone());
    }

    let mut grad = grad_output.clone();
    let grad_dims = grad_output.shape().dims();
    let mut axes_to_sum = Vec::new();

    // Sum over extra leading dimensions introduced by broadcasting
    if grad_dims.len() > target_shape.len() {
        axes_to_sum.extend(0..grad_dims.len() - target_shape.len());
    }

    // Sum over dimensions where target shape had size 1
    for (i, (&gdim, &tdim)) in grad_dims.iter().rev().zip(target_shape.iter().rev()).enumerate() {
        let axis = grad_dims.len() - 1 - i;
        if tdim == 1 && gdim > 1 {
            axes_to_sum.push(axis);
        }
    }

    if !axes_to_sum.is_empty() {
        axes_to_sum.sort_unstable();
        axes_to_sum.dedup();
        grad = grad.sum(Some(axes_to_sum), true)?;
    }

    if grad.shape().dims() != target_shape {
        grad = grad.view(target_shape_obj)?;
    }

    Ok(grad)
}

/// Trait for gradient functions in automatic differentiation
pub trait GradientFunction: Send + Sync + std::fmt::Debug {
    /// Compute gradients for inputs given output gradient
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>>;
    
    /// Get the input tensor IDs this function depends on
    fn inputs(&self) -> &[TensorId];
    
    /// Get a human-readable name for this function
    fn name(&self) -> &'static str {
        "GradientFunction"
    }
    
    /// Get the number of inputs this function expects
    fn num_inputs(&self) -> usize {
        self.inputs().len()
    }
    
    /// Check if this function can handle the given input shapes
    fn validate_inputs(&self, _input_shapes: &[Vec<usize>]) -> Result<()> {
        // Default implementation does no validation
        Ok(())
    }
    
    /// Get a description of what this function does (for debugging)
    fn description(&self) -> String {
        format!("{} with {} inputs", self.name(), self.num_inputs())
    }
}

/// Placeholder gradient function for addition
#[derive(Debug)]
pub struct AddBackward {
    pub input_shapes: [Vec<usize>; 2],
    pub input_ids: [TensorId; 2],
}

impl GradientFunction for AddBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        // For addition, the gradient flows through unchanged to both inputs
        // grad_input1 = grad_output
        // grad_input2 = grad_output
        
        // Handle broadcasting by summing over broadcasted dimensions and reshaping
        let grad1 = handle_broadcast_gradient(grad_output, &self.input_shapes[0])?;
        let grad2 = handle_broadcast_gradient(grad_output, &self.input_shapes[1])?;
        
        Ok(vec![Some(grad1), Some(grad2)])
    }

    fn inputs(&self) -> &[TensorId] {
        &self.input_ids
    }

    fn name(&self) -> &'static str {
        "AddBackward"
    }
    
    fn validate_inputs(&self, input_shapes: &[Vec<usize>]) -> Result<()> {
        if input_shapes.len() != 2 {
            return Err(crate::error::MinitensorError::gradient_error(
                format!("AddBackward expects 2 inputs, got {}", input_shapes.len())
            ));
        }
        // Additional shape validation can be added here
        Ok(())
    }
}

/// Gradient function for element-wise multiplication
#[derive(Debug)]
pub struct MulBackward {
    /// Left-hand side input tensor saved from forward pass
    pub lhs: Tensor,
    /// Right-hand side input tensor saved from forward pass
    pub rhs: Tensor,
    /// IDs of the original input tensors
    pub input_ids: [TensorId; 2],
}

/// Gradient function for matrix multiplication
#[derive(Debug)]
pub struct MatMulBackward {
    /// Left-hand side input tensor saved from forward pass
    pub lhs: Tensor,
    /// Right-hand side input tensor saved from forward pass
    pub rhs: Tensor,
    /// IDs of the original input tensors
    pub input_ids: [TensorId; 2],
}

/// Gradient function for reshape operations
#[derive(Debug)]
pub struct ReshapeBackward {
    pub original_shape: Vec<usize>,
    pub input_id: TensorId,
}

/// Gradient function for transpose operations
#[derive(Debug)]
pub struct TransposeBackward {
    pub dims: Vec<usize>,
    pub input_id: TensorId,
}

/// Gradient function for sum reduction
#[derive(Debug)]
pub struct SumBackward {
    pub input_shape: Vec<usize>,
    pub dims: Option<Vec<usize>>,
    pub keepdim: bool,
    pub input_id: TensorId,
}

/// Gradient function for exponential
#[derive(Debug)]
pub struct ExpBackward {
    pub input_id: TensorId,
    pub output: Tensor,
}

/// Gradient function for logarithm
#[derive(Debug)]
pub struct LogBackward {
    pub input_id: TensorId,
    pub input: Tensor,
}

/// Gradient function for sine
#[derive(Debug)]
pub struct SinBackward {
    pub input_id: TensorId,
    pub input: Tensor,
}

/// Gradient function for cosine
#[derive(Debug)]
pub struct CosBackward {
    pub input_id: TensorId,
    pub input: Tensor,
}

/// Gradient function for tangent
#[derive(Debug)]
pub struct TanBackward {
    pub input_id: TensorId,
    pub output: Tensor,
}

/// Gradient function for hyperbolic tangent
#[derive(Debug)]
pub struct TanhBackward {
    pub input_id: TensorId,
    pub output: Tensor,
}

/// Gradient function for sigmoid
#[derive(Debug)]
pub struct SigmoidBackward {
    pub input_id: TensorId,
    pub output: Tensor,
}

/// Gradient function for ReLU
#[derive(Debug)]
pub struct ReluBackward {
    pub input_id: TensorId,
    pub mask: Vec<bool>,
}

/// Gradient function for softmax
#[derive(Debug)]
pub struct SoftmaxBackward {
    pub input_id: TensorId,
    pub output: Tensor,
    pub dim: usize,
}

/// Gradient function for MSE loss
#[derive(Debug)]
pub struct MSELossBackward {
    pub predictions_shape: Vec<usize>,
    pub targets_shape: Vec<usize>,
    pub input_ids: [TensorId; 2],
    pub reduction: String,
    pub diff: Tensor,
}

/// Gradient function for MAE loss
#[derive(Debug)]
pub struct MAELossBackward {
    pub predictions_shape: Vec<usize>,
    pub targets_shape: Vec<usize>,
    pub input_ids: [TensorId; 2],
    pub reduction: String,
    pub sign: Tensor,
}

/// Gradient function for Huber loss
#[derive(Debug)]
pub struct HuberLossBackward {
    pub predictions_shape: Vec<usize>,
    pub targets_shape: Vec<usize>,
    pub input_ids: [TensorId; 2],
    pub delta: f64,
    pub reduction: String,
    pub diff: Tensor,
}

/// Gradient function for Cross Entropy loss
#[derive(Debug)]
pub struct CrossEntropyLossBackward {
    pub predictions_shape: Vec<usize>,
    pub targets_shape: Vec<usize>,
    pub input_ids: [TensorId; 2],
    pub reduction: String,
    pub softmax_predictions: Tensor,
    pub targets: Tensor,
}

/// Gradient function for Binary Cross Entropy loss
#[derive(Debug)]
pub struct BCELossBackward {
    pub predictions_shape: Vec<usize>,
    pub targets_shape: Vec<usize>,
    pub input_ids: [TensorId; 2],
    pub reduction: String,
    pub predictions: Tensor,
    pub targets: Tensor,
}

/// Gradient function for Focal loss
#[derive(Debug)]
pub struct FocalLossBackward {
    pub predictions_shape: Vec<usize>,
    pub targets_shape: Vec<usize>,
    pub input_ids: [TensorId; 2],
    pub alpha: f64,
    pub gamma: f64,
    pub reduction: String,
    pub softmax_predictions: Tensor,
    pub targets: Tensor,
}

impl GradientFunction for MulBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        // For multiplication: z = x * y
        // grad_x = grad_output * y
        // grad_y = grad_output * x
        let grad_lhs = arithmetic::mul(grad_output, &self.rhs)?;
        let grad_rhs = arithmetic::mul(grad_output, &self.lhs)?;

        let grad1 = handle_broadcast_gradient(&grad_lhs, self.lhs.shape().dims())?;
        let grad2 = handle_broadcast_gradient(&grad_rhs, self.rhs.shape().dims())?;

        Ok(vec![Some(grad1), Some(grad2)])
    }

    fn inputs(&self) -> &[TensorId] {
        &self.input_ids
    }

    fn name(&self) -> &'static str {
        "MulBackward"
    }

    fn validate_inputs(&self, input_shapes: &[Vec<usize>]) -> Result<()> {
        if input_shapes.len() != 2 {
            return Err(crate::error::MinitensorError::gradient_error(
                format!("MulBackward expects 2 inputs, got {}", input_shapes.len())
            ));
        }
        Ok(())
    }
}

impl GradientFunction for MatMulBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        // For matrix multiplication Z = X @ Y, the gradients are:
        // dL/dX = dL/dZ @ Y^T and dL/dY = X^T @ dL/dZ
        let rhs_t = linalg::transpose(&self.rhs, 0, 1)?;
        let lhs_grad = linalg::matmul(grad_output, &rhs_t)?;

        let lhs_t = linalg::transpose(&self.lhs, 0, 1)?;
        let rhs_grad = linalg::matmul(&lhs_t, grad_output)?;

        Ok(vec![Some(lhs_grad), Some(rhs_grad)])
    }

    fn inputs(&self) -> &[TensorId] {
        &self.input_ids
    }

    fn name(&self) -> &'static str {
        "MatMulBackward"
    }
    
    fn validate_inputs(&self, input_shapes: &[Vec<usize>]) -> Result<()> {
        if input_shapes.len() != 2 {
            return Err(crate::error::MinitensorError::gradient_error(
                format!("MatMulBackward expects 2 inputs, got {}", input_shapes.len())
            ));
        }
        
        let shape1 = &input_shapes[0];
        let shape2 = &input_shapes[1];
        
        if shape1.len() < 2 || shape2.len() < 2 {
            return Err(crate::error::MinitensorError::gradient_error(
                "MatMul requires tensors with at least 2 dimensions".to_string()
            ));
        }
        
        Ok(())
    }
}

impl GradientFunction for ReshapeBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        // For reshape, we need to reshape the gradient back to the original shape
        let original_shape = Shape::new(self.original_shape.clone());
        match grad_output.view(original_shape) {
            Ok(reshaped_grad) => Ok(vec![Some(reshaped_grad)]),
            Err(e) => Err(e),
        }
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn name(&self) -> &'static str {
        "ReshapeBackward"
    }
}

impl GradientFunction for TransposeBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let transposed_grad = linalg::transpose(grad_output, self.dims[0], self.dims[1])?;
        Ok(vec![Some(transposed_grad)])
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn name(&self) -> &'static str {
        "TransposeBackward"
    }
}

impl GradientFunction for SumBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let mut grad = grad_output.clone();
        if let Some(dims) = &self.dims {
            if !self.keepdim {
                let mut dims_sorted = dims.clone();
                dims_sorted.sort_unstable();
                for (i, &d) in dims_sorted.iter().enumerate() {
                    grad = grad.unsqueeze(d + i)?;
                }
            }
            let ones = Tensor::ones(
                Shape::new(self.input_shape.clone()),
                grad.dtype(),
                grad.device(),
                false,
            );
            grad = arithmetic::mul(&grad, &ones)?;
        } else {
            let ones = Tensor::ones(
                Shape::new(self.input_shape.clone()),
                grad.dtype(),
                grad.device(),
                false,
            );
            grad = arithmetic::mul(&grad, &ones)?;
        }
        Ok(vec![Some(grad)])
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn name(&self) -> &'static str {
        "SumBackward"
    }
}

impl GradientFunction for ExpBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let grad = arithmetic::mul(grad_output, &self.output)?;
        Ok(vec![Some(grad)])
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn name(&self) -> &'static str {
        "ExpBackward"
    }
}

impl GradientFunction for LogBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let inv = arithmetic::div(
            grad_output,
            &self.input,
        )?;
        Ok(vec![Some(inv)])
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn name(&self) -> &'static str {
        "LogBackward"
    }
}

impl GradientFunction for SinBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let cosx = activation::cos(&self.input)?;
        let grad = arithmetic::mul(grad_output, &cosx)?;
        Ok(vec![Some(grad)])
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn name(&self) -> &'static str {
        "SinBackward"
    }
}

impl GradientFunction for CosBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let sinx = activation::sin(&self.input)?;
        let neg_one = create_scalar_tensor(-1.0, grad_output.dtype(), grad_output.device())?;
        let sinx = arithmetic::mul(&sinx, &neg_one)?;
        let grad = arithmetic::mul(grad_output, &sinx)?;
        Ok(vec![Some(grad)])
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn name(&self) -> &'static str {
        "CosBackward"
    }
}

impl GradientFunction for TanBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let tan_sq = arithmetic::mul(&self.output, &self.output)?;
        let one = Tensor::ones(self.output.shape().clone(), self.output.dtype(), self.output.device(), false);
        let term = arithmetic::add(&one, &tan_sq)?;
        let grad = arithmetic::mul(grad_output, &term)?;
        Ok(vec![Some(grad)])
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn name(&self) -> &'static str {
        "TanBackward"
    }
}

impl GradientFunction for TanhBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let one = Tensor::ones(self.output.shape().clone(), self.output.dtype(), self.output.device(), false);
        let tanh_sq = arithmetic::mul(&self.output, &self.output)?;
        let diff = arithmetic::sub(&one, &tanh_sq)?;
        let grad = arithmetic::mul(grad_output, &diff)?;
        Ok(vec![Some(grad)])
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn name(&self) -> &'static str {
        "TanhBackward"
    }
}

impl GradientFunction for SigmoidBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let one = Tensor::ones(self.output.shape().clone(), self.output.dtype(), self.output.device(), false);
        let one_minus = arithmetic::sub(&one, &self.output)?;
        let base = arithmetic::mul(&self.output, &one_minus)?;
        let grad = arithmetic::mul(grad_output, &base)?;
        Ok(vec![Some(grad)])
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn name(&self) -> &'static str {
        "SigmoidBackward"
    }
}

impl GradientFunction for ReluBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let mut grad_data = TensorData::zeros_on_device(self.mask.len(), grad_output.dtype(), grad_output.device());
        match grad_output.dtype() {
            DataType::Float32 => {
                let go = grad_output.data().as_f32_slice().unwrap();
                let out = grad_data.as_f32_slice_mut().unwrap();
                for (i, &m) in self.mask.iter().enumerate() {
                    out[i] = if m { go[i] } else { 0.0 };
                }
            }
            DataType::Float64 => {
                let go = grad_output.data().as_f64_slice().unwrap();
                let out = grad_data.as_f64_slice_mut().unwrap();
                for (i, &m) in self.mask.iter().enumerate() {
                    out[i] = if m { go[i] } else { 0.0 };
                }
            }
            _ => unreachable!("ReLU only supports float"),
        }
        let grad = Tensor::new(
            Arc::new(grad_data),
            grad_output.shape().clone(),
            grad_output.dtype(),
            grad_output.device(),
            false,
        );
        Ok(vec![Some(grad)])
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn name(&self) -> &'static str {
        "ReluBackward"
    }
}

impl GradientFunction for SoftmaxBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let y = self.output.detach();
        let grad_y = arithmetic::mul(grad_output, &y)?;
        let sum = reduction::sum(&grad_y, Some(vec![self.dim]), true)?;
        let sub = arithmetic::sub(grad_output, &sum)?;
        let grad_input = arithmetic::mul(&y, &sub)?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn name(&self) -> &'static str {
        "SoftmaxBackward"
    }
}

impl GradientFunction for MSELossBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let two = create_scalar_tensor(2.0, self.diff.dtype(), self.diff.device())?;
        let mut base = arithmetic::mul(&self.diff, &two)?;
        if self.reduction == "mean" {
            let n = self.diff.numel() as f64;
            let scale = create_scalar_tensor(1.0 / n, base.dtype(), base.device())?;
            base = arithmetic::mul(&base, &scale)?;
        }
        let pred_grad = arithmetic::mul(&base, grad_output)?;
        let neg_one = create_scalar_tensor(-1.0, pred_grad.dtype(), pred_grad.device())?;
        let targ_grad = arithmetic::mul(&pred_grad, &neg_one)?;
        Ok(vec![Some(pred_grad), Some(targ_grad)])
    }

    fn inputs(&self) -> &[TensorId] {
        &self.input_ids
    }

    fn name(&self) -> &'static str {
        "MSELossBackward"
    }
    
    fn validate_inputs(&self, input_shapes: &[Vec<usize>]) -> Result<()> {
        if input_shapes.len() != 2 {
            return Err(crate::error::MinitensorError::gradient_error(
                format!("MSELossBackward expects 2 inputs, got {}", input_shapes.len())
            ));
        }
        
        if input_shapes[0] != input_shapes[1] {
            return Err(crate::error::MinitensorError::gradient_error(
                "MSE loss requires predictions and targets to have the same shape".to_string()
            ));
        }
        
        Ok(())
    }
}

impl GradientFunction for MAELossBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let mut base = self.sign.clone();
        if self.reduction == "mean" {
            let n = self.sign.numel() as f64;
            let scale = create_scalar_tensor(1.0 / n, base.dtype(), base.device())?;
            base = arithmetic::mul(&base, &scale)?;
        }
        let pred_grad = arithmetic::mul(&base, grad_output)?;
        let neg_one = create_scalar_tensor(-1.0, pred_grad.dtype(), pred_grad.device())?;
        let targ_grad = arithmetic::mul(&pred_grad, &neg_one)?;
        Ok(vec![Some(pred_grad), Some(targ_grad)])
    }

    fn inputs(&self) -> &[TensorId] {
        &self.input_ids
    }

    fn name(&self) -> &'static str {
        "MAELossBackward"
    }
    
    fn validate_inputs(&self, input_shapes: &[Vec<usize>]) -> Result<()> {
        if input_shapes.len() != 2 {
            return Err(crate::error::MinitensorError::gradient_error(
                format!("MAELossBackward expects 2 inputs, got {}", input_shapes.len())
            ));
        }
        
        if input_shapes[0] != input_shapes[1] {
            return Err(crate::error::MinitensorError::gradient_error(
                "MAE loss requires predictions and targets to have the same shape".to_string()
            ));
        }
        
        Ok(())
    }
}

impl GradientFunction for HuberLossBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let numel = self.diff.numel();
        let dtype = self.diff.dtype();
        let device = self.diff.device();
        let mut grad_data = TensorData::zeros_on_device(numel, dtype, device);
        match dtype {
            DataType::Float32 => {
                let diff_slice = self.diff.data().as_f32_slice().unwrap();
                let grad_slice = grad_data.as_f32_slice_mut().unwrap();
                let delta = self.delta as f32;
                for (g, &d) in grad_slice.iter_mut().zip(diff_slice.iter()) {
                    *g = if d.abs() <= delta { d } else { delta * d.signum() };
                }
            }
            DataType::Float64 => {
                let diff_slice = self.diff.data().as_f64_slice().unwrap();
                let grad_slice = grad_data.as_f64_slice_mut().unwrap();
                for (g, &d) in grad_slice.iter_mut().zip(diff_slice.iter()) {
                    *g = if d.abs() <= self.delta { d } else { self.delta * d.signum() };
                }
            }
            _ => return Err(MinitensorError::invalid_operation("Huber only supports float")),
        }
        let mut base = Tensor::new(Arc::new(grad_data), self.diff.shape().clone(), dtype, device, false);
        if self.reduction == "mean" {
            let n = self.diff.numel() as f64;
            let scale = create_scalar_tensor(1.0 / n, dtype, device)?;
            base = arithmetic::mul(&base, &scale)?;
        }
        let pred_grad = arithmetic::mul(&base, grad_output)?;
        let neg_one = create_scalar_tensor(-1.0, dtype, device)?;
        let targ_grad = arithmetic::mul(&pred_grad, &neg_one)?;
        Ok(vec![Some(pred_grad), Some(targ_grad)])
    }

    fn inputs(&self) -> &[TensorId] {
        &self.input_ids
    }

    fn name(&self) -> &'static str {
        "HuberLossBackward"
    }
    
    fn validate_inputs(&self, input_shapes: &[Vec<usize>]) -> Result<()> {
        if input_shapes.len() != 2 {
            return Err(crate::error::MinitensorError::gradient_error(
                format!("HuberLossBackward expects 2 inputs, got {}", input_shapes.len())
            ));
        }
        
        if input_shapes[0] != input_shapes[1] {
            return Err(crate::error::MinitensorError::gradient_error(
                "Huber loss requires predictions and targets to have the same shape".to_string()
            ));
        }
        
        if self.delta <= 0.0 {
            return Err(crate::error::MinitensorError::gradient_error(
                "Huber loss delta must be positive".to_string()
            ));
        }
        
        Ok(())
    }
}

impl GradientFunction for CrossEntropyLossBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let mut base_grad = arithmetic::sub(&self.softmax_predictions, &self.targets)?;
        if self.reduction == "mean" {
            let batch = self.targets_shape[0] as f64;
            let scale = create_scalar_tensor(1.0 / batch, base_grad.dtype(), base_grad.device())?;
            base_grad = arithmetic::mul(&base_grad, &scale)?;
        }
        let pred_grad = arithmetic::mul(&base_grad, grad_output)?;
        Ok(vec![Some(pred_grad), None])
    }

    fn inputs(&self) -> &[TensorId] {
        &self.input_ids
    }

    fn name(&self) -> &'static str {
        "CrossEntropyLossBackward"
    }
    
    fn validate_inputs(&self, input_shapes: &[Vec<usize>]) -> Result<()> {
        if input_shapes.len() != 2 {
            return Err(crate::error::MinitensorError::gradient_error(
                format!("CrossEntropyLossBackward expects 2 inputs, got {}", input_shapes.len())
            ));
        }
        
        let predictions_shape = &input_shapes[0];
        let targets_shape = &input_shapes[1];
        
        // Predictions should be at least 2D (batch_size, num_classes)
        if predictions_shape.len() < 2 {
            return Err(crate::error::MinitensorError::gradient_error(
                "Cross entropy predictions must be at least 2D".to_string()
            ));
        }
        
        // Targets can be 1D (class indices) or same shape as predictions (one-hot)
        if targets_shape != predictions_shape && targets_shape.len() != predictions_shape.len() - 1 {
            return Err(crate::error::MinitensorError::gradient_error(
                "Cross entropy targets must be class indices or one-hot encoded".to_string()
            ));
        }
        
        Ok(())
    }
}

impl GradientFunction for BCELossBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let one = Tensor::ones(
            Shape::new(self.predictions_shape.clone()),
            self.predictions.dtype(),
            self.predictions.device(),
            false,
        );
        let one_minus_pred = arithmetic::sub(&one, &self.predictions)?;
        let numerator = arithmetic::sub(&self.predictions, &self.targets)?;
        let denom = arithmetic::mul(&self.predictions, &one_minus_pred)?;
        let mut base = arithmetic::div(&numerator, &denom)?;
        if self.reduction == "mean" {
            let n = self.predictions.numel() as f64;
            let scale = create_scalar_tensor(1.0 / n, base.dtype(), base.device())?;
            base = arithmetic::mul(&base, &scale)?;
        }
        let pred_grad = arithmetic::mul(&base, grad_output)?;
        Ok(vec![Some(pred_grad), None])
    }

    fn inputs(&self) -> &[TensorId] {
        &self.input_ids
    }

    fn name(&self) -> &'static str {
        "BCELossBackward"
    }
    
    fn validate_inputs(&self, input_shapes: &[Vec<usize>]) -> Result<()> {
        if input_shapes.len() != 2 {
            return Err(crate::error::MinitensorError::gradient_error(
                format!("BCELossBackward expects 2 inputs, got {}", input_shapes.len())
            ));
        }
        
        if input_shapes[0] != input_shapes[1] {
            return Err(crate::error::MinitensorError::gradient_error(
                "BCE loss requires predictions and targets to have the same shape".to_string()
            ));
        }
        
        Ok(())
    }
}

impl GradientFunction for FocalLossBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let p = self.softmax_predictions.detach();
        let t = self.targets.detach();
        let mut base = arithmetic::sub(&p, &t)?;
        let one = Tensor::ones(p.shape().clone(), p.dtype(), p.device(), false);
        let one_minus_p = arithmetic::sub(&one, &p)?;
        let mut weight = tensor_power(&one_minus_p, self.gamma)?;
        let alpha_tensor = create_scalar_tensor(self.alpha, p.dtype(), p.device())?;
        weight = arithmetic::mul(&weight, &alpha_tensor)?;
        base = arithmetic::mul(&base, &weight)?;
        if self.reduction == "mean" {
            let batch = self.targets_shape[0] as f64;
            let scale = create_scalar_tensor(1.0 / batch, base.dtype(), base.device())?;
            base = arithmetic::mul(&base, &scale)?;
        }
        let pred_grad = arithmetic::mul(&base, grad_output)?;
        Ok(vec![Some(pred_grad), None])
    }

    fn inputs(&self) -> &[TensorId] {
        &self.input_ids
    }

    fn name(&self) -> &'static str {
        "FocalLossBackward"
    }
    
    fn validate_inputs(&self, input_shapes: &[Vec<usize>]) -> Result<()> {
        if input_shapes.len() != 2 {
            return Err(crate::error::MinitensorError::gradient_error(
                format!("FocalLossBackward expects 2 inputs, got {}", input_shapes.len())
            ));
        }
        
        let predictions_shape = &input_shapes[0];
        let targets_shape = &input_shapes[1];
        
        // Predictions should be at least 2D (batch_size, num_classes)
        if predictions_shape.len() < 2 {
            return Err(crate::error::MinitensorError::gradient_error(
                "Focal loss predictions must be at least 2D".to_string()
            ));
        }
        
        // Targets can be 1D (class indices) or same shape as predictions (one-hot)
        if targets_shape != predictions_shape && targets_shape.len() != predictions_shape.len() - 1 {
            return Err(crate::error::MinitensorError::gradient_error(
                "Focal loss targets must be class indices or one-hot encoded".to_string()
            ));
        }
        
        if self.alpha <= 0.0 || self.alpha >= 1.0 {
            return Err(crate::error::MinitensorError::gradient_error(
                "Focal loss alpha must be between 0 and 1".to_string()
            ));
        }
        
        if self.gamma < 0.0 {
            return Err(crate::error::MinitensorError::gradient_error(
                "Focal loss gamma must be non-negative".to_string()
            ));
        }
        
        Ok(())
    }
}

fn create_scalar_tensor(value: f64, dtype: DataType, device: Device) -> Result<Tensor> {
    let mut data = TensorData::zeros_on_device(1, dtype, device);
    match dtype {
        DataType::Float32 => {
            let slice = data.as_f32_slice_mut().ok_or_else(|| MinitensorError::internal_error("f32 slice"))?;
            slice[0] = value as f32;
        }
        DataType::Float64 => {
            let slice = data.as_f64_slice_mut().ok_or_else(|| MinitensorError::internal_error("f64 slice"))?;
            slice[0] = value;
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Scalar tensors only supported for floating point types",
            ))
        }
    }
    Ok(Tensor::new(
        Arc::new(data),
        Shape::new(vec![1]),
        dtype,
        device,
        false,
    ))
}

fn tensor_power(tensor: &Tensor, exponent: f64) -> Result<Tensor> {
    let mut output = TensorData::zeros_on_device(tensor.numel(), tensor.dtype(), tensor.device());
    match tensor.dtype() {
        DataType::Float32 => {
            let inp = tensor.data().as_f32_slice().ok_or_else(|| MinitensorError::internal_error("f32 slice"))?;
            let out = output.as_f32_slice_mut().ok_or_else(|| MinitensorError::internal_error("f32 slice mut"))?;
            let exp = exponent as f32;
            for (o, &i) in out.iter_mut().zip(inp.iter()) { *o = i.powf(exp); }
        }
        DataType::Float64 => {
            let inp = tensor.data().as_f64_slice().ok_or_else(|| MinitensorError::internal_error("f64 slice"))?;
            let out = output.as_f64_slice_mut().ok_or_else(|| MinitensorError::internal_error("f64 slice mut"))?;
            for (o, &i) in out.iter_mut().zip(inp.iter()) { *o = i.powf(exponent); }
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Power operation only supported for floating point tensors",
            ))
        }
    }
    Ok(Tensor::new(
        Arc::new(output),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        false,
    ))
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;
    use crate::operations::linalg;
    use crate::tensor::{DataType, Shape, Tensor, TensorData};
    use std::sync::Arc;

    fn create_test_tensor(shape: Vec<usize>) -> Tensor {
        let shape_obj = Shape::new(shape.clone());
        let data = Arc::new(TensorData::zeros(shape_obj.numel(), DataType::Float32));
        Tensor::new(data, shape_obj, DataType::Float32, Device::cpu(), true)
    }

    #[test]
    fn test_add_backward_trait() {
        let id1 = TensorId::new();
        let id2 = TensorId::new();
        
        let add_fn = AddBackward {
            input_shapes: [vec![2, 3], vec![2, 3]],
            input_ids: [id1, id2],
        };
        
        assert_eq!(add_fn.name(), "AddBackward");
        assert_eq!(add_fn.num_inputs(), 2);
        assert_eq!(add_fn.inputs(), &[id1, id2]);
        
        let tensor = create_test_tensor(vec![2, 3]);
        let result = add_fn.backward(&tensor);
        assert!(result.is_ok());
        
        let gradients = result.unwrap();
        assert_eq!(gradients.len(), 2);
    }

    #[test]
    fn test_mul_backward_trait() {
        let lhs = create_test_tensor(vec![3, 4]);
        let rhs = create_test_tensor(vec![3, 4]);

        let mul_fn = MulBackward {
            lhs: lhs.clone(),
            rhs: rhs.clone(),
            input_ids: [lhs.id(), rhs.id()],
        };

        assert_eq!(mul_fn.name(), "MulBackward");
        assert_eq!(mul_fn.num_inputs(), 2);

        let tensor = create_test_tensor(vec![3, 4]);
        let result = mul_fn.backward(&tensor);
        assert!(result.is_ok());

        let gradients = result.unwrap();
        assert_eq!(gradients.len(), 2);
    }

    #[test]
    fn test_matmul_backward_validation() {
        let lhs = create_test_tensor(vec![3, 4]);
        let rhs = create_test_tensor(vec![4, 5]);

        let matmul_fn = MatMulBackward {
            lhs: lhs.clone(),
            rhs: rhs.clone(),
            input_ids: [lhs.id(), rhs.id()],
        };

        // Valid shapes
        let valid_shapes = vec![vec![3, 4], vec![4, 5]];
        assert!(matmul_fn.validate_inputs(&valid_shapes).is_ok());

        // Invalid shapes (1D tensors)
        let invalid_shapes = vec![vec![3], vec![4]];
        assert!(matmul_fn.validate_inputs(&invalid_shapes).is_err());

        // Wrong number of inputs
        let wrong_count = vec![vec![3, 4]];
        assert!(matmul_fn.validate_inputs(&wrong_count).is_err());
    }

    #[test]
    fn test_matmul_backward_gradients() {
        let lhs = Tensor::new(
            Arc::new(TensorData::from_vec_f32(
                vec![1.0, 2.0, 3.0, 4.0],
                Device::cpu(),
            )),
            Shape::new(vec![2, 2]),
            DataType::Float32,
            Device::cpu(),
            true,
        );
        let rhs = Tensor::new(
            Arc::new(TensorData::from_vec_f32(
                vec![5.0, 6.0, 7.0, 8.0],
                Device::cpu(),
            )),
            Shape::new(vec![2, 2]),
            DataType::Float32,
            Device::cpu(),
            true,
        );

        let grad_fn = MatMulBackward {
            lhs: lhs.clone(),
            rhs: rhs.clone(),
            input_ids: [lhs.id(), rhs.id()],
        };
        let grad_output = Tensor::ones(
            Shape::new(vec![2, 2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let grads = grad_fn.backward(&grad_output).unwrap();
        let rhs_t = linalg::transpose(&rhs, 0, 1).unwrap();
        let expected_lhs = linalg::matmul(&grad_output, &rhs_t).unwrap();
        assert!(grads[0].as_ref().unwrap().allclose(&expected_lhs, 1e-6, 1e-6));
    }

    #[test]
    fn test_reshape_backward() {
        let id = TensorId::new();
        
        let reshape_fn = ReshapeBackward {
            original_shape: vec![2, 3, 4],
            input_id: id,
        };
        
        assert_eq!(reshape_fn.name(), "ReshapeBackward");
        assert_eq!(reshape_fn.num_inputs(), 1);
        assert_eq!(reshape_fn.inputs(), &[id]);
        
        let tensor = create_test_tensor(vec![6, 4]);
        let result = reshape_fn.backward(&tensor);
        assert!(result.is_ok());
    }

    #[test]
    fn test_transpose_backward() {
        let id = TensorId::new();
        
        let transpose_fn = TransposeBackward {
            dims: vec![1, 0],
            input_id: id,
        };
        
        assert_eq!(transpose_fn.name(), "TransposeBackward");
        assert_eq!(transpose_fn.num_inputs(), 1);
        
        let tensor = create_test_tensor(vec![3, 4]);
        let result = transpose_fn.backward(&tensor);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sum_backward() {
        let id = TensorId::new();
        
        let sum_fn = SumBackward {
            input_shape: vec![2, 3, 4],
            dims: Some(vec![1]),
            keepdim: false,
            input_id: id,
        };
        
        assert_eq!(sum_fn.name(), "SumBackward");
        assert_eq!(sum_fn.num_inputs(), 1);
        
        let tensor = create_test_tensor(vec![2, 4]);
        let result = sum_fn.backward(&tensor);
        assert!(result.is_ok());
    }

    #[test]
    fn test_gradient_function_description() {
        let id1 = TensorId::new();
        let id2 = TensorId::new();
        
        let add_fn = AddBackward {
            input_shapes: [vec![2], vec![2]],
            input_ids: [id1, id2],
        };
        
        let description = add_fn.description();
        assert!(description.contains("AddBackward"));
        assert!(description.contains("2 inputs"));
    }

    #[test]
    fn test_add_backward_validation() {
        let id1 = TensorId::new();
        let id2 = TensorId::new();
        
        let add_fn = AddBackward {
            input_shapes: [vec![2], vec![2]],
            input_ids: [id1, id2],
        };
        
        // Valid input count
        let valid_shapes = vec![vec![2], vec![2]];
        assert!(add_fn.validate_inputs(&valid_shapes).is_ok());
        
        // Invalid input count
        let invalid_shapes = vec![vec![2]];
        assert!(add_fn.validate_inputs(&invalid_shapes).is_err());
    }

    #[test]
    fn test_cross_entropy_loss_backward() {
        let id1 = TensorId::new();
        let id2 = TensorId::new();
        
        let ce_fn = CrossEntropyLossBackward {
            predictions_shape: vec![2, 3],
            targets_shape: vec![2, 3],
            input_ids: [id1, id2],
            reduction: "mean".to_string(),
        };
        
        assert_eq!(ce_fn.name(), "CrossEntropyLossBackward");
        assert_eq!(ce_fn.num_inputs(), 2);
        
        let tensor = create_test_tensor(vec![1]);
        let result = ce_fn.backward(&tensor);
        assert!(result.is_ok());
        
        let gradients = result.unwrap();
        assert_eq!(gradients.len(), 2);
        assert!(gradients[0].is_some()); // predictions gradient
        assert!(gradients[1].is_none());  // targets gradient (not needed)
    }

    #[test]
    fn test_bce_loss_backward() {
        let id1 = TensorId::new();
        let id2 = TensorId::new();
        
        let bce_fn = BCELossBackward {
            predictions_shape: vec![4],
            targets_shape: vec![4],
            input_ids: [id1, id2],
            reduction: "mean".to_string(),
        };
        
        assert_eq!(bce_fn.name(), "BCELossBackward");
        assert_eq!(bce_fn.num_inputs(), 2);
        
        let tensor = create_test_tensor(vec![1]);
        let result = bce_fn.backward(&tensor);
        assert!(result.is_ok());
        
        let gradients = result.unwrap();
        assert_eq!(gradients.len(), 2);
        assert!(gradients[0].is_some()); // predictions gradient
        assert!(gradients[1].is_none());  // targets gradient (not needed)
    }

    #[test]
    fn test_focal_loss_backward() {
        let id1 = TensorId::new();
        let id2 = TensorId::new();
        
        let focal_fn = FocalLossBackward {
            predictions_shape: vec![2, 3],
            targets_shape: vec![2, 3],
            input_ids: [id1, id2],
            alpha: 0.25,
            gamma: 2.0,
            reduction: "mean".to_string(),
        };
        
        assert_eq!(focal_fn.name(), "FocalLossBackward");
        assert_eq!(focal_fn.num_inputs(), 2);
        
        let tensor = create_test_tensor(vec![1]);
        let result = focal_fn.backward(&tensor);
        assert!(result.is_ok());
        
        let gradients = result.unwrap();
        assert_eq!(gradients.len(), 2);
        assert!(gradients[0].is_some()); // predictions gradient
        assert!(gradients[1].is_none());  // targets gradient (not needed)
    }

    #[test]
    fn test_cross_entropy_loss_validation() {
        let id1 = TensorId::new();
        let id2 = TensorId::new();
        
        let ce_fn = CrossEntropyLossBackward {
            predictions_shape: vec![2, 3],
            targets_shape: vec![2, 3],
            input_ids: [id1, id2],
            reduction: "mean".to_string(),
        };
        
        // Valid shapes (one-hot targets)
        let valid_shapes = vec![vec![2, 3], vec![2, 3]];
        assert!(ce_fn.validate_inputs(&valid_shapes).is_ok());
        
        // Valid shapes (class indices)
        let valid_indices = vec![vec![2, 3], vec![2]];
        assert!(ce_fn.validate_inputs(&valid_indices).is_ok());
        
        // Invalid shapes (1D predictions)
        let invalid_shapes = vec![vec![3], vec![3]];
        assert!(ce_fn.validate_inputs(&invalid_shapes).is_err());
        
        // Wrong number of inputs
        let wrong_count = vec![vec![2, 3]];
        assert!(ce_fn.validate_inputs(&wrong_count).is_err());
    }

    #[test]
    fn test_bce_loss_validation() {
        let id1 = TensorId::new();
        let id2 = TensorId::new();
        
        let bce_fn = BCELossBackward {
            predictions_shape: vec![4],
            targets_shape: vec![4],
            input_ids: [id1, id2],
            reduction: "mean".to_string(),
        };
        
        // Valid shapes
        let valid_shapes = vec![vec![4], vec![4]];
        assert!(bce_fn.validate_inputs(&valid_shapes).is_ok());
        
        // Invalid shapes (mismatch)
        let invalid_shapes = vec![vec![4], vec![3]];
        assert!(bce_fn.validate_inputs(&invalid_shapes).is_err());
        
        // Wrong number of inputs
        let wrong_count = vec![vec![4]];
        assert!(bce_fn.validate_inputs(&wrong_count).is_err());
    }

    #[test]
    fn test_focal_loss_validation() {
        let id1 = TensorId::new();
        let id2 = TensorId::new();
        
        let focal_fn = FocalLossBackward {
            predictions_shape: vec![2, 3],
            targets_shape: vec![2, 3],
            input_ids: [id1, id2],
            alpha: 0.25,
            gamma: 2.0,
            reduction: "mean".to_string(),
        };
        
        // Valid shapes
        let valid_shapes = vec![vec![2, 3], vec![2, 3]];
        assert!(focal_fn.validate_inputs(&valid_shapes).is_ok());
        
        // Invalid alpha (out of range)
        let invalid_alpha_fn = FocalLossBackward {
            predictions_shape: vec![2, 3],
            targets_shape: vec![2, 3],
            input_ids: [id1, id2],
            alpha: 1.5,
            gamma: 2.0,
            reduction: "mean".to_string(),
        };
        assert!(invalid_alpha_fn.validate_inputs(&valid_shapes).is_err());
        
        // Invalid gamma (negative)
        let invalid_gamma_fn = FocalLossBackward {
            predictions_shape: vec![2, 3],
            targets_shape: vec![2, 3],
            input_ids: [id1, id2],
            alpha: 0.25,
            gamma: -1.0,
            reduction: "mean".to_string(),
        };
        assert!(invalid_gamma_fn.validate_inputs(&valid_shapes).is_err());
    }
}
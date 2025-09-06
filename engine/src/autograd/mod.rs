// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    device::Device,
    error::{MinitensorError, Result},
    operations::{activation, arithmetic, reduction, shape_ops},
    tensor::{DataType, Shape, Tensor, TensorData},
};
use rayon::prelude::*;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

const PAR_THRESHOLD: usize = 1 << 12; // 4096 elements

/// Unique identifier for tensors in the computation graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId(usize);

impl TensorId {
    /// Create a new unique tensor ID
    pub fn new() -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

impl Default for TensorId {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for gradient functions in the computation graph
pub trait GradientFunction: Send + Sync {
    /// Compute gradients for inputs given the output gradient
    fn backward(&self, grad_output: &Tensor) -> Result<HashMap<TensorId, Tensor>>;

    /// Get the input tensor IDs that this function depends on
    fn input_ids(&self) -> Vec<TensorId>;
}

/// Computation graph for automatic differentiation
pub struct ComputationGraph {
    nodes: HashMap<TensorId, Arc<dyn GradientFunction>>,
}

impl ComputationGraph {
    /// Create a new computation graph
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
        }
    }

    /// Add a node to the computation graph
    pub fn add_node(&mut self, tensor_id: TensorId, grad_fn: Arc<dyn GradientFunction>) {
        self.nodes.insert(tensor_id, grad_fn);
    }

    /// Get a reference to the nodes in the computation graph
    pub fn nodes(&self) -> &HashMap<TensorId, Arc<dyn GradientFunction>> {
        &self.nodes
    }

    /// Perform backward pass from the given tensor
    pub fn backward(
        &self,
        output_tensor: &Tensor,
        grad_output: Option<Tensor>,
    ) -> Result<HashMap<TensorId, Tensor>> {
        let mut gradients = HashMap::with_capacity(self.nodes.len());

        // Initialize with output gradient
        let initial_grad = match grad_output {
            Some(grad) => grad,
            None => {
                if output_tensor.numel() != 1 {
                    return Err(MinitensorError::gradient_error(
                        "Gradient can only be implicitly created for scalar tensors",
                    ));
                }
                Tensor::ones(
                    output_tensor.shape().clone(),
                    output_tensor.dtype(),
                    output_tensor.device(),
                    false,
                )
            }
        };

        // Iterative depth-first traversal using an explicit stack avoids the
        // overhead of recursive function calls and additional visited sets.
        let mut stack = Vec::with_capacity(self.nodes.len().max(1));
        stack.push((output_tensor.id(), initial_grad));

        while let Some((tensor_id, grad_output)) = stack.pop() {
            match gradients.entry(tensor_id) {
                Entry::Occupied(mut e) => {
                    let new_grad = crate::operations::arithmetic::add(e.get(), &grad_output)?;
                    *e.get_mut() = new_grad;
                }
                Entry::Vacant(e) => {
                    e.insert(grad_output.clone());
                }
            }

            if let Some(grad_fn) = self.nodes.get(&tensor_id) {
                let input_grads = grad_fn.backward(&grad_output)?;
                stack.extend(input_grads.into_iter());
            }
        }

        Ok(gradients)
    }
}

impl Default for ComputationGraph {
    fn default() -> Self {
        Self::new()
    }
}

// Thread-local computation graph to avoid cross-test interference
thread_local! {
    static GLOBAL_GRAPH: std::cell::RefCell<ComputationGraph> =
        std::cell::RefCell::new(ComputationGraph::new());
}

/// Add a tensor and its gradient function to the global computation graph
pub fn add_to_graph(tensor: &Tensor, grad_fn: Option<Arc<dyn GradientFunction>>) -> Result<()> {
    if let Some(grad_fn) = grad_fn {
        GLOBAL_GRAPH.with(|graph| {
            if let Ok(mut g) = graph.try_borrow_mut() {
                g.add_node(tensor.id(), grad_fn);
            }
        });
    }
    Ok(())
}

/// Perform backward pass from the given tensor using the global computation graph
pub fn backward(tensor: &Tensor, grad_output: Option<Tensor>) -> Result<HashMap<TensorId, Tensor>> {
    GLOBAL_GRAPH.with(|graph| graph.borrow().backward(tensor, grad_output))
}

/// Clear the global computation graph
pub fn clear_graph() -> Result<()> {
    GLOBAL_GRAPH.with(|graph| {
        graph.borrow_mut().nodes.clear();
    });
    Ok(())
}

// Gradient function implementations for common operations

/// Gradient function for addition operation
pub struct AddBackward {
    pub input_shapes: [Vec<usize>; 2],
    pub input_ids: [TensorId; 2],
}

impl GradientFunction for AddBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<HashMap<TensorId, Tensor>> {
        let mut gradients = HashMap::new();

        // For addition, gradients flow through unchanged, but we need to handle broadcasting
        let lhs_shape = Shape::new(self.input_shapes[0].clone());
        let rhs_shape = Shape::new(self.input_shapes[1].clone());

        // Reduce gradients to match input shapes if broadcasting occurred
        let lhs_grad = reduce_gradient_for_broadcasting(grad_output, &lhs_shape)?;
        let rhs_grad = reduce_gradient_for_broadcasting(grad_output, &rhs_shape)?;

        gradients.insert(self.input_ids[0], lhs_grad);
        gradients.insert(self.input_ids[1], rhs_grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> Vec<TensorId> {
        vec![self.input_ids[0], self.input_ids[1]]
    }
}

/// Gradient function for multiplication operation
pub struct MulBackward {
    pub lhs: Tensor,
    pub rhs: Tensor,
    pub input_ids: [TensorId; 2],
}

impl GradientFunction for MulBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<HashMap<TensorId, Tensor>> {
        let mut gradients = HashMap::new();

        // d/dx(x*y) = y and d/dy(x*y) = x
        let lhs_term = arithmetic::mul(grad_output, &self.rhs.detach())?;
        let rhs_term = arithmetic::mul(grad_output, &self.lhs.detach())?;

        let lhs_grad = reduce_gradient_for_broadcasting(&lhs_term, self.lhs.shape())?;
        let rhs_grad = reduce_gradient_for_broadcasting(&rhs_term, self.rhs.shape())?;

        gradients.insert(self.input_ids[0], lhs_grad);
        gradients.insert(self.input_ids[1], rhs_grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> Vec<TensorId> {
        vec![self.input_ids[0], self.input_ids[1]]
    }
}

/// Gradient function for division operation
pub struct DivBackward {
    pub lhs: Tensor,
    pub rhs: Tensor,
    pub input_ids: [TensorId; 2],
}

impl GradientFunction for DivBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<HashMap<TensorId, Tensor>> {
        let mut gradients = HashMap::new();

        // d/dx(x/y) = 1 / y
        let rhs_inv = arithmetic::div(
            &Tensor::ones(
                self.rhs.shape().clone(),
                self.rhs.dtype(),
                self.rhs.device(),
                false,
            ),
            &self.rhs.detach(),
        )?;
        let lhs_term = arithmetic::mul(grad_output, &rhs_inv)?;
        let lhs_grad = reduce_gradient_for_broadcasting(&lhs_term, self.lhs.shape())?;

        // d/dy(x/y) = -x / y^2
        let num = arithmetic::mul(grad_output, &self.lhs.detach())?;
        let rhs_sq = arithmetic::mul(&self.rhs.detach(), &self.rhs.detach())?;
        let rhs_term = arithmetic::div(&num, &rhs_sq)?;
        let zero = Tensor::zeros(
            rhs_term.shape().clone(),
            rhs_term.dtype(),
            rhs_term.device(),
            false,
        );
        let rhs_term = arithmetic::sub(&zero, &rhs_term)?; // negate
        let rhs_grad = reduce_gradient_for_broadcasting(&rhs_term, self.rhs.shape())?;

        gradients.insert(self.input_ids[0], lhs_grad);
        gradients.insert(self.input_ids[1], rhs_grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> Vec<TensorId> {
        vec![self.input_ids[0], self.input_ids[1]]
    }
}

/// Gradient function for matrix multiplication
pub struct MatMulBackward {
    pub lhs: Tensor,
    pub rhs: Tensor,
    pub input_ids: [TensorId; 2],
}

impl GradientFunction for MatMulBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<HashMap<TensorId, Tensor>> {
        let mut gradients = HashMap::new();

        if self.lhs.ndim() != 2 || self.rhs.ndim() != 2 {
            return Err(MinitensorError::not_implemented(
                "MatMulBackward currently supports only 2D tensors".to_string(),
            ));
        }

        let rhs_t = crate::operations::linalg::transpose(&self.rhs.detach(), 0, 1)?;
        let lhs_grad = crate::operations::linalg::matmul(grad_output, &rhs_t)?;

        let lhs_t = crate::operations::linalg::transpose(&self.lhs.detach(), 0, 1)?;
        let rhs_grad = crate::operations::linalg::matmul(&lhs_t, grad_output)?;

        gradients.insert(self.input_ids[0], lhs_grad);
        gradients.insert(self.input_ids[1], rhs_grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> Vec<TensorId> {
        vec![self.input_ids[0], self.input_ids[1]]
    }
}

/// Gradient function for transpose operation
pub struct TransposeBackward {
    pub dims: Vec<usize>,
    pub input_id: TensorId,
}

impl GradientFunction for TransposeBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<HashMap<TensorId, Tensor>> {
        let mut gradients = HashMap::new();

        // Transpose gradient: transpose back
        if self.dims.len() == 2 {
            let grad_input =
                crate::operations::linalg::transpose(grad_output, self.dims[0], self.dims[1])?;
            gradients.insert(self.input_id, grad_input);
        } else {
            // For now, just pass through the gradient
            gradients.insert(self.input_id, grad_output.clone());
        }

        Ok(gradients)
    }

    fn input_ids(&self) -> Vec<TensorId> {
        vec![self.input_id]
    }
}

/// Gradient function for sum reduction
pub struct SumBackward {
    pub input_id: TensorId,
    pub input_shape: Vec<usize>,
    pub dims: Option<Vec<usize>>,
    pub keepdim: bool,
}

impl GradientFunction for SumBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<HashMap<TensorId, Tensor>> {
        let mut gradients = HashMap::new();

        let mut grad = grad_output.clone();
        if !self.keepdim {
            if let Some(dims) = &self.dims {
                let mut shape = grad.shape().dims().to_vec();
                let mut sorted = dims.clone();
                sorted.sort_unstable();
                for &d in &sorted {
                    shape.insert(d, 1);
                }
                grad = shape_ops::reshape(&grad, Shape::new(shape))?;
            } else {
                grad = shape_ops::reshape(&grad, Shape::new(vec![1; self.input_shape.len()]))?;
            }
        }

        let ones = Tensor::ones(
            Shape::new(self.input_shape.clone()),
            grad_output.dtype(),
            grad_output.device(),
            false,
        );
        let grad_input = arithmetic::mul(&ones, &grad)?;
        gradients.insert(self.input_id, grad_input);

        Ok(gradients)
    }

    fn input_ids(&self) -> Vec<TensorId> {
        vec![self.input_id]
    }
}

// Gradient functions for activation functions

/// Gradient function for exponential
pub struct ExpBackward {
    pub input_id: TensorId,
    pub output: Tensor,
}

impl GradientFunction for ExpBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<HashMap<TensorId, Tensor>> {
        let mut gradients = HashMap::new();

        // d/dx(exp(x)) = exp(x) * grad_output
        let grad = arithmetic::mul(&self.output, grad_output)?;
        gradients.insert(self.input_id, grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> Vec<TensorId> {
        vec![self.input_id]
    }
}

/// Gradient function for logarithm
pub struct LogBackward {
    pub input_id: TensorId,
    pub input: Tensor,
}

impl GradientFunction for LogBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<HashMap<TensorId, Tensor>> {
        let mut gradients = HashMap::new();

        // d/dx(log(x)) = 1/x * grad_output
        let ones = Tensor::ones(
            self.input.shape().clone(),
            self.input.dtype(),
            self.input.device(),
            false,
        );
        let inv = arithmetic::div(&ones, &self.input.detach())?;
        let grad = arithmetic::mul(&inv, grad_output)?;
        gradients.insert(self.input_id, grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> Vec<TensorId> {
        vec![self.input_id]
    }
}

/// Gradient function for sine
pub struct SinBackward {
    pub input_id: TensorId,
    pub input: Tensor,
}

impl GradientFunction for SinBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<HashMap<TensorId, Tensor>> {
        let mut gradients = HashMap::new();

        // d/dx(sin(x)) = cos(x) * grad_output
        let cos_x = self.input.cos()?;
        let grad = arithmetic::mul(&cos_x, grad_output)?;
        gradients.insert(self.input_id, grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> Vec<TensorId> {
        vec![self.input_id]
    }
}

/// Gradient function for cosine
pub struct CosBackward {
    pub input_id: TensorId,
    pub input: Tensor,
}

impl GradientFunction for CosBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<HashMap<TensorId, Tensor>> {
        let mut gradients = HashMap::new();

        // d/dx(cos(x)) = -sin(x) * grad_output
        let sin_x = self.input.sin()?;
        let mul = arithmetic::mul(&sin_x, grad_output)?;
        let zeros = Tensor::zeros(mul.shape().clone(), mul.dtype(), mul.device(), false);
        let grad = arithmetic::sub(&zeros, &mul)?;
        gradients.insert(self.input_id, grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> Vec<TensorId> {
        vec![self.input_id]
    }
}

/// Gradient function for tangent
pub struct TanBackward {
    pub input_id: TensorId,
    pub output: Tensor,
}

impl GradientFunction for TanBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<HashMap<TensorId, Tensor>> {
        let mut gradients = HashMap::new();

        // d/dx(tan(x)) = (1 + tan²(x)) * grad_output
        let tan_sq = arithmetic::mul(&self.output, &self.output)?;
        let ones = Tensor::ones(
            self.output.shape().clone(),
            self.output.dtype(),
            self.output.device(),
            false,
        );
        let term = arithmetic::add(&ones, &tan_sq)?;
        let grad = arithmetic::mul(&term, grad_output)?;
        gradients.insert(self.input_id, grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> Vec<TensorId> {
        vec![self.input_id]
    }
}

/// Gradient function for tanh
pub struct TanhBackward {
    pub input_id: TensorId,
    pub output: Tensor,
}

impl GradientFunction for TanhBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<HashMap<TensorId, Tensor>> {
        let mut gradients = HashMap::new();

        // d/dx(tanh(x)) = (1 - tanh²(x)) * grad_output
        let y2 = arithmetic::mul(&self.output, &self.output)?;
        let ones = Tensor::ones(
            self.output.shape().clone(),
            self.output.dtype(),
            self.output.device(),
            false,
        );
        let term = arithmetic::sub(&ones, &y2)?;
        let grad = arithmetic::mul(&term, grad_output)?;
        gradients.insert(self.input_id, grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> Vec<TensorId> {
        vec![self.input_id]
    }
}

/// Gradient function for sigmoid
pub struct SigmoidBackward {
    pub input_id: TensorId,
    pub output: Tensor,
}

impl GradientFunction for SigmoidBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<HashMap<TensorId, Tensor>> {
        let mut gradients = HashMap::new();

        // d/dx(sigmoid(x)) = sigmoid(x) * (1 - sigmoid(x)) * grad_output
        let ones = Tensor::ones(
            self.output.shape().clone(),
            self.output.dtype(),
            self.output.device(),
            false,
        );
        let one_minus = arithmetic::sub(&ones, &self.output)?;
        let term = arithmetic::mul(&self.output, &one_minus)?;
        let grad = arithmetic::mul(&term, grad_output)?;
        gradients.insert(self.input_id, grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> Vec<TensorId> {
        vec![self.input_id]
    }
}

/// Gradient function for power operation
pub struct PowBackward {
    pub base: Tensor,
    pub exponent: Tensor,
    pub output: Tensor,
    pub input_ids: [TensorId; 2],
    pub base_requires_grad: bool,
    pub exp_requires_grad: bool,
}

impl GradientFunction for PowBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<HashMap<TensorId, Tensor>> {
        let mut gradients = HashMap::new();

        match self.output.dtype() {
            DataType::Float32 => {
                let base_slice = self.base.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from base tensor")
                })?;
                let exp_slice = self.exponent.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from exponent tensor")
                })?;
                let out_slice = self.output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from output tensor")
                })?;
                let grad_out = grad_output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from grad_output")
                })?;

                if self.base_requires_grad {
                    let mut grad_data = TensorData::zeros_on_device(
                        self.base.numel(),
                        self.base.dtype(),
                        self.base.device(),
                    );
                    let grad_slice = grad_data.as_f32_slice_mut().ok_or_else(|| {
                        MinitensorError::internal_error(
                            "Failed to get mutable f32 slice from grad_data",
                        )
                    })?;
                    let len = base_slice.len();
                    if len < PAR_THRESHOLD {
                        for i in 0..len {
                            grad_slice[i] =
                                exp_slice[i] * base_slice[i].powf(exp_slice[i] - 1.0) * grad_out[i];
                        }
                    } else {
                        let base_ptr = base_slice.as_ptr() as usize;
                        let exp_ptr = exp_slice.as_ptr() as usize;
                        let go_ptr = grad_out.as_ptr() as usize;
                        let grad_ptr = grad_slice.as_mut_ptr() as usize;
                        (0..len).into_par_iter().for_each(|i| unsafe {
                            let base_ptr = base_ptr as *const f32;
                            let exp_ptr = exp_ptr as *const f32;
                            let go_ptr = go_ptr as *const f32;
                            let grad_ptr = grad_ptr as *mut f32;
                            *grad_ptr.add(i) = *exp_ptr.add(i)
                                * (*base_ptr.add(i)).powf(*exp_ptr.add(i) - 1.0)
                                * *go_ptr.add(i);
                        });
                    }
                    let grad_tensor = Tensor::new(
                        Arc::new(grad_data),
                        self.base.shape().clone(),
                        self.base.dtype(),
                        self.base.device(),
                        false,
                    );
                    gradients.insert(self.input_ids[0], grad_tensor);
                }

                if self.exp_requires_grad {
                    let mut grad_data = TensorData::zeros_on_device(
                        self.exponent.numel(),
                        self.exponent.dtype(),
                        self.exponent.device(),
                    );
                    let grad_slice = grad_data.as_f32_slice_mut().ok_or_else(|| {
                        MinitensorError::internal_error(
                            "Failed to get mutable f32 slice from grad_data",
                        )
                    })?;
                    let len = exp_slice.len();
                    if len < PAR_THRESHOLD {
                        for i in 0..len {
                            grad_slice[i] = out_slice[i] * base_slice[i].ln() * grad_out[i];
                        }
                    } else {
                        let out_ptr = out_slice.as_ptr() as usize;
                        let base_ptr = base_slice.as_ptr() as usize;
                        let go_ptr = grad_out.as_ptr() as usize;
                        let grad_ptr = grad_slice.as_mut_ptr() as usize;
                        (0..len).into_par_iter().for_each(|i| unsafe {
                            let out_ptr = out_ptr as *const f32;
                            let base_ptr = base_ptr as *const f32;
                            let go_ptr = go_ptr as *const f32;
                            let grad_ptr = grad_ptr as *mut f32;
                            *grad_ptr.add(i) =
                                *out_ptr.add(i) * (*base_ptr.add(i)).ln() * *go_ptr.add(i);
                        });
                    }
                    let grad_tensor = Tensor::new(
                        Arc::new(grad_data),
                        self.exponent.shape().clone(),
                        self.exponent.dtype(),
                        self.exponent.device(),
                        false,
                    );
                    gradients.insert(self.input_ids[1], grad_tensor);
                }
            }
            DataType::Float64 => {
                let base_slice = self.base.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from base tensor")
                })?;
                let exp_slice = self.exponent.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from exponent tensor")
                })?;
                let out_slice = self.output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from output tensor")
                })?;
                let grad_out = grad_output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from grad_output")
                })?;

                if self.base_requires_grad {
                    let mut grad_data = TensorData::zeros_on_device(
                        self.base.numel(),
                        self.base.dtype(),
                        self.base.device(),
                    );
                    let grad_slice = grad_data.as_f64_slice_mut().ok_or_else(|| {
                        MinitensorError::internal_error(
                            "Failed to get mutable f64 slice from grad_data",
                        )
                    })?;
                    let len = base_slice.len();
                    if len < PAR_THRESHOLD {
                        for i in 0..len {
                            grad_slice[i] =
                                exp_slice[i] * base_slice[i].powf(exp_slice[i] - 1.0) * grad_out[i];
                        }
                    } else {
                        let base_ptr = base_slice.as_ptr() as usize;
                        let exp_ptr = exp_slice.as_ptr() as usize;
                        let go_ptr = grad_out.as_ptr() as usize;
                        let grad_ptr = grad_slice.as_mut_ptr() as usize;
                        (0..len).into_par_iter().for_each(|i| unsafe {
                            let base_ptr = base_ptr as *const f64;
                            let exp_ptr = exp_ptr as *const f64;
                            let go_ptr = go_ptr as *const f64;
                            let grad_ptr = grad_ptr as *mut f64;
                            *grad_ptr.add(i) = *exp_ptr.add(i)
                                * (*base_ptr.add(i)).powf(*exp_ptr.add(i) - 1.0)
                                * *go_ptr.add(i);
                        });
                    }
                    let grad_tensor = Tensor::new(
                        Arc::new(grad_data),
                        self.base.shape().clone(),
                        self.base.dtype(),
                        self.base.device(),
                        false,
                    );
                    gradients.insert(self.input_ids[0], grad_tensor);
                }

                if self.exp_requires_grad {
                    let mut grad_data = TensorData::zeros_on_device(
                        self.exponent.numel(),
                        self.exponent.dtype(),
                        self.exponent.device(),
                    );
                    let grad_slice = grad_data.as_f64_slice_mut().ok_or_else(|| {
                        MinitensorError::internal_error(
                            "Failed to get mutable f64 slice from grad_data",
                        )
                    })?;
                    let len = exp_slice.len();
                    if len < PAR_THRESHOLD {
                        for i in 0..len {
                            grad_slice[i] = out_slice[i] * base_slice[i].ln() * grad_out[i];
                        }
                    } else {
                        let out_ptr = out_slice.as_ptr() as usize;
                        let base_ptr = base_slice.as_ptr() as usize;
                        let go_ptr = grad_out.as_ptr() as usize;
                        let grad_ptr = grad_slice.as_mut_ptr() as usize;
                        (0..len).into_par_iter().for_each(|i| unsafe {
                            let out_ptr = out_ptr as *const f64;
                            let base_ptr = base_ptr as *const f64;
                            let go_ptr = go_ptr as *const f64;
                            let grad_ptr = grad_ptr as *mut f64;
                            *grad_ptr.add(i) =
                                *out_ptr.add(i) * (*base_ptr.add(i)).ln() * *go_ptr.add(i);
                        });
                    }
                    let grad_tensor = Tensor::new(
                        Arc::new(grad_data),
                        self.exponent.shape().clone(),
                        self.exponent.dtype(),
                        self.exponent.device(),
                        false,
                    );
                    gradients.insert(self.input_ids[1], grad_tensor);
                }
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "Power backward only supported for floating point tensors",
                ))
            }
        }

        Ok(gradients)
    }

    fn input_ids(&self) -> Vec<TensorId> {
        vec![self.input_ids[0], self.input_ids[1]]
    }
}

/// Gradient function for ReLU
pub struct ReluBackward {
    pub input_id: TensorId,
    pub mask: Vec<bool>,
}

impl GradientFunction for ReluBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<HashMap<TensorId, Tensor>> {
        let mut gradients = HashMap::new();

        let mut grad_data = TensorData::zeros_on_device(
            grad_output.numel(),
            grad_output.dtype(),
            grad_output.device(),
        );

        match grad_output.dtype() {
            DataType::Float32 => {
                let go = grad_output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from grad_output")
                })?;
                let grad_slice = grad_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice from grad_data",
                    )
                })?;
                let len = go.len();
                if len < PAR_THRESHOLD {
                    for i in 0..len {
                        grad_slice[i] = go[i] * if self.mask[i] { 1.0 } else { 0.0 };
                    }
                } else {
                    let mask = &self.mask;
                    let go_ptr = go.as_ptr() as usize;
                    let grad_ptr = grad_slice.as_mut_ptr() as usize;
                    (0..len).into_par_iter().for_each(|i| unsafe {
                        let go_ptr = go_ptr as *const f32;
                        let grad_ptr = grad_ptr as *mut f32;
                        let m = if *mask.get_unchecked(i) { 1.0 } else { 0.0 };
                        *grad_ptr.add(i) = *go_ptr.add(i) * m;
                    });
                }
            }
            DataType::Float64 => {
                let go = grad_output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from grad_output")
                })?;
                let grad_slice = grad_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice from grad_data",
                    )
                })?;
                let len = go.len();
                if len < PAR_THRESHOLD {
                    for i in 0..len {
                        grad_slice[i] = go[i] * if self.mask[i] { 1.0 } else { 0.0 };
                    }
                } else {
                    let mask = &self.mask;
                    let go_ptr = go.as_ptr() as usize;
                    let grad_ptr = grad_slice.as_mut_ptr() as usize;
                    (0..len).into_par_iter().for_each(|i| unsafe {
                        let go_ptr = go_ptr as *const f64;
                        let grad_ptr = grad_ptr as *mut f64;
                        let m = if *mask.get_unchecked(i) { 1.0 } else { 0.0 };
                        *grad_ptr.add(i) = *go_ptr.add(i) * m;
                    });
                }
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "ReLU backward only supported for floating point tensors",
                ))
            }
        }

        let grad_input = Tensor::new(
            Arc::new(grad_data),
            grad_output.shape().clone(),
            grad_output.dtype(),
            grad_output.device(),
            grad_output.requires_grad(),
        );
        gradients.insert(self.input_id, grad_input);

        Ok(gradients)
    }

    fn input_ids(&self) -> Vec<TensorId> {
        vec![self.input_id]
    }
}

/// Gradient function for LeakyReLU
pub struct LeakyReluBackward {
    pub input_id: TensorId,
    pub negative_slope: f64,
    pub mask: Vec<bool>,
}

impl GradientFunction for LeakyReluBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<HashMap<TensorId, Tensor>> {
        let mut gradients = HashMap::new();

        let mut grad_data = TensorData::zeros_on_device(
            grad_output.numel(),
            grad_output.dtype(),
            grad_output.device(),
        );

        match grad_output.dtype() {
            DataType::Float32 => {
                let go = grad_output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from grad_output")
                })?;
                let grad_slice = grad_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice from grad_data",
                    )
                })?;
                let len = go.len();
                let slope = self.negative_slope as f32;
                if len < PAR_THRESHOLD {
                    for i in 0..len {
                        grad_slice[i] = if self.mask[i] { go[i] } else { go[i] * slope };
                    }
                } else {
                    let mask = &self.mask;
                    let go_ptr = go.as_ptr() as usize;
                    let grad_ptr = grad_slice.as_mut_ptr() as usize;
                    (0..len).into_par_iter().for_each(|i| unsafe {
                        let go_ptr = go_ptr as *const f32;
                        let grad_ptr = grad_ptr as *mut f32;
                        let val = if *mask.get_unchecked(i) {
                            *go_ptr.add(i)
                        } else {
                            *go_ptr.add(i) * slope
                        };
                        *grad_ptr.add(i) = val;
                    });
                }
            }
            DataType::Float64 => {
                let go = grad_output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from grad_output")
                })?;
                let grad_slice = grad_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice from grad_data",
                    )
                })?;
                let len = go.len();
                let slope = self.negative_slope;
                if len < PAR_THRESHOLD {
                    for i in 0..len {
                        grad_slice[i] = if self.mask[i] { go[i] } else { go[i] * slope };
                    }
                } else {
                    let mask = &self.mask;
                    let go_ptr = go.as_ptr() as usize;
                    let grad_ptr = grad_slice.as_mut_ptr() as usize;
                    (0..len).into_par_iter().for_each(|i| unsafe {
                        let go_ptr = go_ptr as *const f64;
                        let grad_ptr = grad_ptr as *mut f64;
                        let val = if *mask.get_unchecked(i) {
                            *go_ptr.add(i)
                        } else {
                            *go_ptr.add(i) * slope
                        };
                        *grad_ptr.add(i) = val;
                    });
                }
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "LeakyReLU backward only supported for floating point tensors",
                ))
            }
        }

        let grad_input = Tensor::new(
            Arc::new(grad_data),
            grad_output.shape().clone(),
            grad_output.dtype(),
            grad_output.device(),
            grad_output.requires_grad(),
        );
        gradients.insert(self.input_id, grad_input);

        Ok(gradients)
    }

    fn input_ids(&self) -> Vec<TensorId> {
        vec![self.input_id]
    }
}

/// Gradient function for softmax
pub struct SoftmaxBackward {
    pub input_id: TensorId,
    pub output: Tensor,
    pub dim: usize,
}

impl GradientFunction for SoftmaxBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<HashMap<TensorId, Tensor>> {
        let mut gradients = HashMap::new();

        // Softmax gradient: y * (grad_output - sum(grad_output * y, dim))
        let y = self.output.detach();
        let grad_y = arithmetic::mul(grad_output, &y)?;
        let sum = reduction::sum(&grad_y, Some(vec![self.dim]), true)?;
        let sub = arithmetic::sub(grad_output, &sum)?;
        let grad_input = arithmetic::mul(&y, &sub)?;

        gradients.insert(self.input_id, grad_input);

        Ok(gradients)
    }

    fn input_ids(&self) -> Vec<TensorId> {
        vec![self.input_id]
    }
}

/// Gradient function for reshape operation
pub struct ReshapeBackward {
    pub input_shape: Vec<usize>,
    pub input_id: TensorId,
}

impl GradientFunction for ReshapeBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<HashMap<TensorId, Tensor>> {
        let mut gradients = HashMap::new();

        // Reshape gradient: reshape back to original shape
        let original_shape = Shape::new(self.input_shape.clone());
        let grad_input = crate::operations::shape_ops::reshape(grad_output, original_shape)?;
        gradients.insert(self.input_id, grad_input);

        Ok(gradients)
    }

    fn input_ids(&self) -> Vec<TensorId> {
        vec![self.input_id]
    }
}

// Loss function gradient implementations

/// Gradient function for MSE loss
pub struct MSELossBackward {
    pub predictions_shape: Vec<usize>,
    pub targets_shape: Vec<usize>,
    pub input_ids: [TensorId; 2],
    pub reduction: String,
    pub diff: Tensor,
}

impl GradientFunction for MSELossBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<HashMap<TensorId, Tensor>> {
        let mut gradients = HashMap::new();

        // Base gradient: 2 * (predictions - targets)
        let two = create_scalar_tensor(2.0, self.diff.dtype(), self.diff.device())?;
        let mut base_grad = arithmetic::mul(&self.diff, &two)?;

        // Apply reduction scaling
        match self.reduction.as_str() {
            "mean" => {
                let n = self.diff.numel() as f64;
                let scale = create_scalar_tensor(1.0 / n, base_grad.dtype(), base_grad.device())?;
                base_grad = arithmetic::mul(&base_grad, &scale)?;
            }
            "sum" | "none" => {}
            _ => {
                return Err(MinitensorError::gradient_error(format!(
                    "Unknown reduction mode: {}",
                    self.reduction
                )))
            }
        }

        // Multiply by upstream gradient
        let pred_grad = arithmetic::mul(&base_grad, grad_output)?;
        let neg_one = create_scalar_tensor(-1.0, pred_grad.dtype(), pred_grad.device())?;
        let target_grad = arithmetic::mul(&pred_grad, &neg_one)?;

        gradients.insert(self.input_ids[0], pred_grad);
        gradients.insert(self.input_ids[1], target_grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> Vec<TensorId> {
        vec![self.input_ids[0], self.input_ids[1]]
    }
}

/// Gradient function for MAE loss
pub struct MAELossBackward {
    pub predictions_shape: Vec<usize>,
    pub targets_shape: Vec<usize>,
    pub input_ids: [TensorId; 2],
    pub reduction: String,
    pub sign: Tensor,
}

impl GradientFunction for MAELossBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<HashMap<TensorId, Tensor>> {
        let mut gradients = HashMap::new();

        let mut base_grad = self.sign.clone();
        match self.reduction.as_str() {
            "mean" => {
                let n = self.sign.numel() as f64;
                let scale = create_scalar_tensor(1.0 / n, base_grad.dtype(), base_grad.device())?;
                base_grad = arithmetic::mul(&base_grad, &scale)?;
            }
            "sum" | "none" => {}
            _ => {
                return Err(MinitensorError::gradient_error(format!(
                    "Unknown reduction mode: {}",
                    self.reduction
                )))
            }
        }

        let pred_grad = arithmetic::mul(&base_grad, grad_output)?;
        let neg_one = create_scalar_tensor(-1.0, pred_grad.dtype(), pred_grad.device())?;
        let target_grad = arithmetic::mul(&pred_grad, &neg_one)?;

        gradients.insert(self.input_ids[0], pred_grad);
        gradients.insert(self.input_ids[1], target_grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> Vec<TensorId> {
        vec![self.input_ids[0], self.input_ids[1]]
    }
}

/// Gradient function for Huber loss
pub struct HuberLossBackward {
    pub predictions_shape: Vec<usize>,
    pub targets_shape: Vec<usize>,
    pub input_ids: [TensorId; 2],
    pub delta: f64,
    pub reduction: String,
    pub diff: Tensor,
}

impl GradientFunction for HuberLossBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<HashMap<TensorId, Tensor>> {
        let mut gradients = HashMap::new();

        let numel = self.diff.numel();
        let dtype = self.diff.dtype();
        let device = self.diff.device();
        let mut grad_data = TensorData::zeros_on_device(numel, dtype, device);

        match dtype {
            DataType::Float32 => {
                let diff_slice = self.diff.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from diff")
                })?;
                let grad_slice = grad_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get mutable f32 slice from grad")
                })?;
                let delta = self.delta as f32;
                if numel < PAR_THRESHOLD {
                    for i in 0..numel {
                        let d = diff_slice[i];
                        grad_slice[i] = if d.abs() <= delta {
                            d
                        } else {
                            delta * d.signum()
                        };
                    }
                } else {
                    let diff_ptr = diff_slice.as_ptr() as usize;
                    let grad_ptr = grad_slice.as_mut_ptr() as usize;
                    (0..numel).into_par_iter().for_each(|i| unsafe {
                        let diff_ptr = diff_ptr as *const f32;
                        let grad_ptr = grad_ptr as *mut f32;
                        let d = *diff_ptr.add(i);
                        *grad_ptr.add(i) = if d.abs() <= delta {
                            d
                        } else {
                            delta * d.signum()
                        };
                    });
                }
            }
            DataType::Float64 => {
                let diff_slice = self.diff.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from diff")
                })?;
                let grad_slice = grad_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get mutable f64 slice from grad")
                })?;
                if numel < PAR_THRESHOLD {
                    for i in 0..numel {
                        let d = diff_slice[i];
                        grad_slice[i] = if d.abs() <= self.delta {
                            d
                        } else {
                            self.delta * d.signum()
                        };
                    }
                } else {
                    let diff_ptr = diff_slice.as_ptr() as usize;
                    let grad_ptr = grad_slice.as_mut_ptr() as usize;
                    let delta = self.delta;
                    (0..numel).into_par_iter().for_each(|i| unsafe {
                        let diff_ptr = diff_ptr as *const f64;
                        let grad_ptr = grad_ptr as *mut f64;
                        let d = *diff_ptr.add(i);
                        *grad_ptr.add(i) = if d.abs() <= delta {
                            d
                        } else {
                            delta * d.signum()
                        };
                    });
                }
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "Huber loss only supports floating point tensors",
                ))
            }
        }

        let mut base_grad = Tensor::new(
            Arc::new(grad_data),
            Shape::new(self.predictions_shape.clone()),
            dtype,
            device,
            false,
        );

        if self.reduction == "mean" {
            let scale = create_scalar_tensor(1.0 / numel as f64, dtype, device)?;
            base_grad = arithmetic::mul(&base_grad, &scale)?;
        }

        let pred_grad = arithmetic::mul(&base_grad, grad_output)?;
        let neg_one = create_scalar_tensor(-1.0, dtype, device)?;
        let target_grad = arithmetic::mul(&pred_grad, &neg_one)?;

        gradients.insert(self.input_ids[0], pred_grad);
        gradients.insert(self.input_ids[1], target_grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> Vec<TensorId> {
        vec![self.input_ids[0], self.input_ids[1]]
    }
}

/// Gradient function for Cross Entropy loss
pub struct CrossEntropyLossBackward {
    pub predictions_shape: Vec<usize>,
    pub targets_shape: Vec<usize>,
    pub input_ids: [TensorId; 2],
    pub reduction: String,
    pub softmax_predictions: Tensor,
    pub targets: Tensor,
}

impl GradientFunction for CrossEntropyLossBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<HashMap<TensorId, Tensor>> {
        let mut gradients = HashMap::new();

        // Compute base gradient: softmax(predictions) - targets
        let mut base_grad =
            arithmetic::sub(&self.softmax_predictions.detach(), &self.targets.detach())?;

        // Apply reduction scaling
        match self.reduction.as_str() {
            "mean" => {
                let batch = self.targets_shape[0] as f64;
                let mut scalar_data =
                    TensorData::zeros_on_device(1, base_grad.dtype(), base_grad.device());
                match base_grad.dtype() {
                    DataType::Float32 => {
                        let slice = scalar_data.as_f32_slice_mut().ok_or_else(|| {
                            MinitensorError::internal_error(
                                "Failed to get mutable f32 slice from scalar",
                            )
                        })?;
                        slice[0] = (1.0 / batch) as f32;
                    }
                    DataType::Float64 => {
                        let slice = scalar_data.as_f64_slice_mut().ok_or_else(|| {
                            MinitensorError::internal_error(
                                "Failed to get mutable f64 slice from scalar",
                            )
                        })?;
                        slice[0] = 1.0 / batch;
                    }
                    _ => {
                        return Err(MinitensorError::invalid_operation(
                            "CrossEntropy backward only supports floating point tensors",
                        ))
                    }
                }
                let scalar_tensor = Tensor::new(
                    Arc::new(scalar_data),
                    Shape::new(vec![1]),
                    base_grad.dtype(),
                    base_grad.device(),
                    false,
                );
                base_grad = arithmetic::mul(&base_grad, &scalar_tensor)?;
            }
            "sum" | "none" => {}
            _ => {
                return Err(MinitensorError::gradient_error(format!(
                    "Unknown reduction mode: {}",
                    self.reduction
                )))
            }
        }

        // Multiply by upstream gradient (handles broadcasting)
        let pred_grad = arithmetic::mul(&base_grad, grad_output)?;

        // Targets typically have no gradient
        gradients.insert(self.input_ids[0], pred_grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> Vec<TensorId> {
        vec![self.input_ids[0], self.input_ids[1]]
    }
}

/// Gradient function for Binary Cross Entropy loss
pub struct BCELossBackward {
    pub predictions_shape: Vec<usize>,
    pub targets_shape: Vec<usize>,
    pub input_ids: [TensorId; 2],
    pub reduction: String,
    pub predictions: Tensor,
    pub targets: Tensor,
}

impl GradientFunction for BCELossBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<HashMap<TensorId, Tensor>> {
        let mut gradients = HashMap::new();

        // BCE gradient: (predictions - targets) / (predictions * (1 - predictions))
        let one = Tensor::ones(
            Shape::new(self.predictions_shape.clone()),
            self.predictions.dtype(),
            self.predictions.device(),
            false,
        );
        let one_minus_pred = arithmetic::sub(&one, &self.predictions)?;
        let numerator = arithmetic::sub(&self.predictions, &self.targets)?;
        let denom = arithmetic::mul(&self.predictions, &one_minus_pred)?;
        let mut base_grad = arithmetic::div(&numerator, &denom)?;

        if self.reduction == "mean" {
            let n = self.predictions.numel() as f64;
            let scale = create_scalar_tensor(1.0 / n, base_grad.dtype(), base_grad.device())?;
            base_grad = arithmetic::mul(&base_grad, &scale)?;
        }

        let pred_grad = arithmetic::mul(&base_grad, grad_output)?;
        gradients.insert(self.input_ids[0], pred_grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> Vec<TensorId> {
        vec![self.input_ids[0], self.input_ids[1]]
    }
}

/// Gradient function for KL Divergence loss
pub struct KLDivLossBackward {
    pub predictions_shape: Vec<usize>,
    pub targets_shape: Vec<usize>,
    pub input_ids: [TensorId; 2],
    pub reduction: String,
    pub predictions: Tensor,
    pub targets: Tensor,
}

impl GradientFunction for KLDivLossBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<HashMap<TensorId, Tensor>> {
        let mut gradients = HashMap::new();

        // Gradient w.r.t predictions: -(targets / predictions)
        let mut pred_grad = arithmetic::div(&self.targets, &self.predictions)?;
        let neg_one = create_scalar_tensor(-1.0, pred_grad.dtype(), pred_grad.device())?;
        pred_grad = arithmetic::mul(&pred_grad, &neg_one)?;
        if self.reduction == "mean" {
            let n = self.predictions.numel() as f64;
            let scale = create_scalar_tensor(1.0 / n, pred_grad.dtype(), pred_grad.device())?;
            pred_grad = arithmetic::mul(&pred_grad, &scale)?;
        }
        let pred_grad = arithmetic::mul(&pred_grad, grad_output)?;
        gradients.insert(self.input_ids[0], pred_grad);

        // Gradient w.r.t targets: log(targets) - log(predictions) + 1
        let log_targets = activation::log(&self.targets)?;
        let log_preds = activation::log(&self.predictions)?;
        let diff = arithmetic::sub(&log_targets, &log_preds)?;
        let one = Tensor::ones(
            self.targets.shape().clone(),
            self.targets.dtype(),
            self.targets.device(),
            false,
        );
        let mut target_grad = arithmetic::add(&diff, &one)?;
        if self.reduction == "mean" {
            let n = self.predictions.numel() as f64;
            let scale = create_scalar_tensor(1.0 / n, target_grad.dtype(), target_grad.device())?;
            target_grad = arithmetic::mul(&target_grad, &scale)?;
        }
        let target_grad = arithmetic::mul(&target_grad, grad_output)?;
        gradients.insert(self.input_ids[1], target_grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> Vec<TensorId> {
        vec![self.input_ids[0], self.input_ids[1]]
    }
}

/// Gradient function for Focal loss
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

impl GradientFunction for FocalLossBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<HashMap<TensorId, Tensor>> {
        let mut gradients = HashMap::new();

        // Compute base gradient similar to cross entropy
        let p = self.softmax_predictions.detach();
        let t = self.targets.detach();
        let mut base_grad = arithmetic::sub(&p, &t)?;

        // Compute focal weight: alpha * (1 - p)^gamma
        let one = Tensor::ones(p.shape().clone(), p.dtype(), p.device(), false);
        let one_minus_p = arithmetic::sub(&one, &p)?;
        let mut weight = tensor_power(&one_minus_p, self.gamma)?;
        let alpha_tensor = create_scalar_tensor(self.alpha, p.dtype(), p.device())?;
        weight = arithmetic::mul(&weight, &alpha_tensor)?;

        base_grad = arithmetic::mul(&base_grad, &weight)?;

        if self.reduction == "mean" {
            let batch = self.targets_shape[0] as f64;
            let scale = create_scalar_tensor(1.0 / batch, base_grad.dtype(), base_grad.device())?;
            base_grad = arithmetic::mul(&base_grad, &scale)?;
        }

        let pred_grad = arithmetic::mul(&base_grad, grad_output)?;
        gradients.insert(self.input_ids[0], pred_grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> Vec<TensorId> {
        vec![self.input_ids[0], self.input_ids[1]]
    }
}

/// Create a scalar tensor with the given value
fn create_scalar_tensor(value: f64, dtype: DataType, device: Device) -> Result<Tensor> {
    let mut data = TensorData::zeros_on_device(1, dtype, device);
    match dtype {
        DataType::Float32 => {
            let slice = data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice from scalar")
            })?;
            slice[0] = value as f32;
        }
        DataType::Float64 => {
            let slice = data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice from scalar")
            })?;
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

/// Raise each tensor element to the given power
fn tensor_power(tensor: &Tensor, exponent: f64) -> Result<Tensor> {
    let mut output_data =
        TensorData::zeros_on_device(tensor.numel(), tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => {
            let input = tensor.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 slice from tensor")
            })?;
            let output = output_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice from output")
            })?;
            let exp = exponent as f32;
            let len = input.len();
            debug_assert_eq!(len, output.len());
            if len < PAR_THRESHOLD {
                for i in 0..len {
                    output[i] = input[i].powf(exp);
                }
            } else {
                let in_ptr = input.as_ptr() as usize;
                let out_ptr = output.as_mut_ptr() as usize;
                (0..len).into_par_iter().for_each(|i| unsafe {
                    let in_ptr = in_ptr as *const f32;
                    let out_ptr = out_ptr as *mut f32;
                    *out_ptr.add(i) = (*in_ptr.add(i)).powf(exp);
                });
            }
        }
        DataType::Float64 => {
            let input = tensor.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 slice from tensor")
            })?;
            let output = output_data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice from output")
            })?;
            let len = input.len();
            debug_assert_eq!(len, output.len());
            if len < PAR_THRESHOLD {
                for i in 0..len {
                    output[i] = input[i].powf(exponent);
                }
            } else {
                let in_ptr = input.as_ptr() as usize;
                let out_ptr = output.as_mut_ptr() as usize;
                (0..len).into_par_iter().for_each(|i| unsafe {
                    let in_ptr = in_ptr as *const f64;
                    let out_ptr = out_ptr as *mut f64;
                    *out_ptr.add(i) = (*in_ptr.add(i)).powf(exponent);
                });
            }
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "Power operation only supported for floating point tensors",
            ))
        }
    }

    Ok(Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        false,
    ))
}

/// Helper function to reduce gradients for broadcasting
fn reduce_gradient_for_broadcasting(grad_output: &Tensor, target_shape: &Shape) -> Result<Tensor> {
    if grad_output.shape() == target_shape {
        return Ok(grad_output.clone());
    }

    let mut grad = grad_output.clone();
    let grad_dims = grad.shape().dims();
    let target_dims = target_shape.dims();
    let mut axes_to_sum = Vec::new();

    if grad_dims.len() > target_dims.len() {
        axes_to_sum.extend(0..grad_dims.len() - target_dims.len());
    }

    for (i, (&gdim, &tdim)) in grad_dims
        .iter()
        .rev()
        .zip(target_dims.iter().rev())
        .enumerate()
    {
        let axis = grad_dims.len() - 1 - i;
        if tdim == 1 && gdim > 1 {
            axes_to_sum.push(axis);
        }
    }

    if !axes_to_sum.is_empty() {
        axes_to_sum.sort_unstable();
        axes_to_sum.dedup();
        grad = reduction::sum(&grad, Some(axes_to_sum), true)?;
    }

    if grad.shape() != target_shape {
        grad = shape_ops::reshape(&grad, target_shape.clone())?;
    }

    Ok(grad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;

    #[test]
    fn test_tensor_id_generation() {
        let id1 = TensorId::new();
        let id2 = TensorId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_computation_graph() {
        let mut graph = ComputationGraph::new();
        let tensor_id = TensorId::new();

        let grad_fn = Arc::new(AddBackward {
            input_shapes: [vec![2, 2], vec![2, 2]],
            input_ids: [TensorId::new(), TensorId::new()],
        });

        graph.add_node(tensor_id, grad_fn);
        assert!(graph.nodes.contains_key(&tensor_id));
    }

    #[test]
    fn test_add_backward() {
        let grad_fn = AddBackward {
            input_shapes: [vec![2, 2], vec![2, 2]],
            input_ids: [TensorId::new(), TensorId::new()],
        };

        let grad_output = Tensor::ones(
            Shape::new(vec![2, 2]),
            crate::tensor::DataType::Float32,
            Device::cpu(),
            false,
        );
        let gradients = grad_fn.backward(&grad_output).unwrap();

        assert_eq!(gradients.len(), 2);
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
            false,
        );
        let rhs = Tensor::new(
            Arc::new(TensorData::from_vec_f32(
                vec![5.0, 6.0, 7.0, 8.0],
                Device::cpu(),
            )),
            Shape::new(vec![2, 2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let input_ids = [TensorId::new(), TensorId::new()];
        let grad_fn = MatMulBackward {
            lhs: lhs.clone(),
            rhs: rhs.clone(),
            input_ids,
        };
        let grad_output = Tensor::ones(
            Shape::new(vec![2, 2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let grads = grad_fn.backward(&grad_output).unwrap();
        let rhs_t = crate::operations::linalg::transpose(&rhs, 0, 1).unwrap();
        let expected_lhs = crate::operations::linalg::matmul(&grad_output, &rhs_t).unwrap();
        let lhs_grad = grads.get(&input_ids[0]).unwrap();
        assert!(lhs_grad.allclose(&expected_lhs, 1e-6, 1e-6));
    }
}

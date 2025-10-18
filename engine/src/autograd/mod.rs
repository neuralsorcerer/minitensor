// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    device::Device,
    error::{MinitensorError, Result},
    operations::{activation, arithmetic, linalg, minmax, reduction, selection, shape_ops},
    tensor::{DataType, Shape, Tensor, TensorData},
};
pub mod graph;
use libm::{erf, erff};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::cell::Cell;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

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

impl std::fmt::Display for TensorId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TensorId({})", self.0)
    }
}

/// Trait for gradient functions in the computation graph
pub trait GradientFunction: Send + Sync {
    /// Compute gradients for inputs given the output gradient
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>>;

    /// Get the input tensor IDs that this function depends on
    fn input_ids(&self) -> &[TensorId];

    /// Name of the gradient function used for debugging and introspection
    fn name(&self) -> &'static str {
        let full = std::any::type_name::<Self>();
        match full.rsplit("::").next() {
            Some(name) => name,
            None => full,
        }
    }
}

pub use graph::ComputationGraph;

// Thread-local computation graph to avoid cross-test interference
thread_local! {
    static GLOBAL_GRAPH: std::cell::RefCell<ComputationGraph> =
        std::cell::RefCell::new(ComputationGraph::new());
}

thread_local! {
    static GRAPH_CONSUMED: Cell<bool> = Cell::new(false);
}

/// Add a tensor and its gradient function to the global computation graph
pub fn add_to_graph(tensor: &Tensor, grad_fn: Option<Arc<dyn GradientFunction>>) -> Result<()> {
    GLOBAL_GRAPH.with(|graph| {
        if let Ok(mut g) = graph.try_borrow_mut() {
            g.add_tensor_with_grad_req(tensor.id(), grad_fn, tensor.requires_grad());
        }
    });
    reset_graph_consumed();
    Ok(())
}

/// Perform backward pass from the given tensor using the global computation graph
pub fn backward(
    tensor: &Tensor,
    grad_output: Option<Tensor>,
) -> Result<FxHashMap<TensorId, Tensor>> {
    GLOBAL_GRAPH.with(|graph| {
        let grad = match grad_output {
            Some(g) => g,
            None => {
                if tensor.numel() != 1 {
                    return Err(MinitensorError::gradient_error(
                        "Gradient can only be implicitly created for scalar tensors",
                    ));
                }
                Tensor::ones(
                    tensor.shape().clone(),
                    tensor.dtype(),
                    tensor.device(),
                    false,
                )
            }
        };
        graph.borrow_mut().backward(tensor.id(), Some(grad))
    })
}

/// Get the gradient for a tensor from the last backward pass
pub fn get_gradient(tensor: &Tensor) -> Option<Tensor> {
    GLOBAL_GRAPH.with(|graph| graph.borrow().get_gradient(tensor.id()).cloned())
}

/// Clear all stored gradients in the global computation graph
pub fn zero_gradients() {
    GLOBAL_GRAPH.with(|graph| graph.borrow_mut().zero_grad());
}

/// Clear the global computation graph
pub fn clear_graph() -> Result<()> {
    GLOBAL_GRAPH.with(|graph| {
        *graph.borrow_mut() = ComputationGraph::new();
    });
    reset_graph_consumed();
    Ok(())
}

/// Mark the computation graph as consumed after a backward pass completes.
pub fn mark_graph_consumed() {
    GRAPH_CONSUMED.with(|flag| flag.set(true));
}

/// Reset the consumed flag so that future backward passes are permitted.
pub fn reset_graph_consumed() {
    GRAPH_CONSUMED.with(|flag| flag.set(false));
}

/// Query whether the active computation graph has already been consumed.
pub fn is_graph_consumed() -> bool {
    GRAPH_CONSUMED.with(|flag| flag.get())
}

// Gradient function implementations for common operations

/// Gradient function for tensor cloning operation
pub struct CloneBackward {
    pub input_id: TensorId,
}

impl GradientFunction for CloneBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);
        gradients.insert(self.input_id, grad_output.deep_clone()?);
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for addition operation
pub struct AddBackward {
    pub input_shapes: [Vec<usize>; 2],
    pub input_ids: [TensorId; 2],
}

impl GradientFunction for AddBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(2);

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

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}

/// Gradient function for subtraction operation
pub struct SubBackward {
    pub input_shapes: [Vec<usize>; 2],
    pub input_ids: [TensorId; 2],
}

impl GradientFunction for SubBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(2);

        let lhs_shape = Shape::new(self.input_shapes[0].clone());
        let rhs_shape = Shape::new(self.input_shapes[1].clone());

        let lhs_grad = reduce_gradient_for_broadcasting(grad_output, &lhs_shape)?;
        let rhs_base = reduce_gradient_for_broadcasting(grad_output, &rhs_shape)?;
        let rhs_grad = arithmetic::neg(&rhs_base)?;

        gradients.insert(self.input_ids[0], lhs_grad);
        gradients.insert(self.input_ids[1], rhs_grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}

/// Gradient function for multiplication operation
pub struct MulBackward {
    pub lhs: Tensor,
    pub rhs: Tensor,
    pub input_ids: [TensorId; 2],
}

impl GradientFunction for MulBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(2);

        // d/dx(x*y) = y and d/dy(x*y) = x
        let lhs_term = arithmetic::mul(grad_output, &self.rhs.detach())?;
        let rhs_term = arithmetic::mul(grad_output, &self.lhs.detach())?;

        let lhs_grad = reduce_gradient_for_broadcasting(&lhs_term, self.lhs.shape())?;
        let rhs_grad = reduce_gradient_for_broadcasting(&rhs_term, self.rhs.shape())?;

        gradients.insert(self.input_ids[0], lhs_grad);
        gradients.insert(self.input_ids[1], rhs_grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}

/// Gradient function for division operation
pub struct DivBackward {
    pub lhs: Tensor,
    pub rhs: Tensor,
    pub input_ids: [TensorId; 2],
}

impl GradientFunction for DivBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(2);

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
        let rhs_term = arithmetic::neg(&rhs_term)?;
        let rhs_grad = reduce_gradient_for_broadcasting(&rhs_term, self.rhs.shape())?;

        gradients.insert(self.input_ids[0], lhs_grad);
        gradients.insert(self.input_ids[1], rhs_grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}

/// Gradient function for where/select operation
pub struct WhereBackward {
    pub condition: Tensor,
    pub input_shape: Vec<usize>,
    pub other_shape: Vec<usize>,
    pub input_requires_grad: bool,
    pub other_requires_grad: bool,
    pub input_ids: [TensorId; 2],
}

impl GradientFunction for WhereBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(self.input_requires_grad as usize + self.other_requires_grad as usize);

        let mut zero_tensor: Option<Tensor> = None;

        if self.input_requires_grad {
            let zeros = zero_tensor.get_or_insert_with(|| {
                Tensor::zeros(
                    grad_output.shape().clone(),
                    grad_output.dtype(),
                    grad_output.device(),
                    false,
                )
            });
            let selected = selection::where_op(&self.condition, grad_output, zeros)?;
            let reduced =
                reduce_gradient_for_broadcasting(&selected, &Shape::new(self.input_shape.clone()))?;
            gradients.insert(self.input_ids[0], reduced);
        }

        if self.other_requires_grad {
            let zeros = zero_tensor.get_or_insert_with(|| {
                Tensor::zeros(
                    grad_output.shape().clone(),
                    grad_output.dtype(),
                    grad_output.device(),
                    false,
                )
            });
            let selected = selection::where_op(&self.condition, zeros, grad_output)?;
            let reduced =
                reduce_gradient_for_broadcasting(&selected, &Shape::new(self.other_shape.clone()))?;
            gradients.insert(self.input_ids[1], reduced);
        }

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}

/// Gradient function for diagonal extraction.
pub struct DiagonalBackward {
    pub input_shape: Vec<usize>,
    pub input_strides: Vec<usize>,
    pub input_dtype: DataType,
    pub dim1: usize,
    pub dim2: usize,
    pub offset: isize,
    pub input_requires_grad: bool,
    pub input_id: TensorId,
}

impl GradientFunction for DiagonalBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();

        if !self.input_requires_grad {
            return Ok(gradients);
        }

        if grad_output.dtype() != self.input_dtype {
            return Err(MinitensorError::type_mismatch(
                format!("{:?}", grad_output.dtype()),
                format!("{:?}", self.input_dtype),
            ));
        }

        let spec = linalg::compute_diagonal_spec(
            &self.input_shape,
            &self.input_strides,
            self.dim1,
            self.dim2,
            self.offset,
        )?;

        if grad_output.shape().dims() != spec.output_dims {
            return Err(MinitensorError::shape_mismatch(
                grad_output.shape().dims().to_vec(),
                spec.output_dims.clone(),
            ));
        }

        let numel = self.input_shape.iter().product();
        let mut grad_data =
            TensorData::zeros_on_device(numel, self.input_dtype, grad_output.device());

        match self.input_dtype {
            DataType::Float32 => {
                let grad_out = grad_output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice for diagonal backward")
                })?;
                let grad_in = grad_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice for diagonal backward",
                    )
                })?;
                linalg::diagonal_scatter(
                    grad_out,
                    grad_in,
                    &self.input_shape,
                    &self.input_strides,
                    &spec,
                );
            }
            DataType::Float64 => {
                let grad_out = grad_output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice for diagonal backward")
                })?;
                let grad_in = grad_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice for diagonal backward",
                    )
                })?;
                linalg::diagonal_scatter(
                    grad_out,
                    grad_in,
                    &self.input_shape,
                    &self.input_strides,
                    &spec,
                );
            }
            DataType::Int32 => {
                let grad_out = grad_output.data().as_i32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get i32 slice for diagonal backward")
                })?;
                let grad_in = grad_data.as_i32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable i32 slice for diagonal backward",
                    )
                })?;
                linalg::diagonal_scatter(
                    grad_out,
                    grad_in,
                    &self.input_shape,
                    &self.input_strides,
                    &spec,
                );
            }
            DataType::Int64 => {
                let grad_out = grad_output.data().as_i64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get i64 slice for diagonal backward")
                })?;
                let grad_in = grad_data.as_i64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable i64 slice for diagonal backward",
                    )
                })?;
                linalg::diagonal_scatter(
                    grad_out,
                    grad_in,
                    &self.input_shape,
                    &self.input_strides,
                    &spec,
                );
            }
            DataType::Bool => {
                return Err(MinitensorError::invalid_operation(
                    "diagonal backward is not defined for bool tensors",
                ));
            }
        }

        let grad_tensor = Tensor::new(
            Arc::new(grad_data),
            Shape::new(self.input_shape.clone()),
            self.input_dtype,
            grad_output.device(),
            false,
        );
        gradients.insert(self.input_id, grad_tensor);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for triangular masking operations (triu/tril)
pub struct TriangularBackward {
    pub input_shape: Vec<usize>,
    pub diagonal: isize,
    pub upper: bool,
    pub input_requires_grad: bool,
    pub input_id: TensorId,
}

impl GradientFunction for TriangularBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();

        if self.input_requires_grad {
            if grad_output.shape().dims() != self.input_shape {
                return Err(MinitensorError::shape_mismatch(
                    grad_output.shape().dims().to_vec(),
                    self.input_shape.clone(),
                ));
            }

            let mut grad_data = TensorData::uninitialized_on_device(
                grad_output.numel(),
                grad_output.dtype(),
                grad_output.device(),
            );
            linalg::apply_triangular_mask(grad_output, &mut grad_data, self.diagonal, self.upper)?;
            let grad = Tensor::new(
                Arc::new(grad_data),
                grad_output.shape().clone(),
                grad_output.dtype(),
                grad_output.device(),
                false,
            );
            gradients.insert(self.input_id, grad);
        }

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for element-wise maximum operation
pub struct MaximumBackward {
    pub lhs: Tensor,
    pub rhs: Tensor,
    pub input_shapes: [Vec<usize>; 2],
    pub input_requires_grad: [bool; 2],
    pub input_ids: [TensorId; 2],
}

impl GradientFunction for MaximumBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(self.input_requires_grad.iter().filter(|&&b| b).count());

        if !self.input_requires_grad[0] && !self.input_requires_grad[1] {
            return Ok(gradients);
        }

        let mask = minmax::maximum_backward_mask(&self.lhs, &self.rhs)?;
        let mut zeros: Option<Tensor> = None;

        if self.input_requires_grad[0] {
            let zero = zeros.get_or_insert_with(|| {
                Tensor::zeros(
                    grad_output.shape().clone(),
                    grad_output.dtype(),
                    grad_output.device(),
                    false,
                )
            });
            let selected = minmax::select_with_mask(&mask, grad_output, zero)?;
            let reduced = reduce_gradient_for_broadcasting(
                &selected,
                &Shape::new(self.input_shapes[0].clone()),
            )?;
            gradients.insert(self.input_ids[0], reduced);
        }

        if self.input_requires_grad[1] {
            let zero = zeros.get_or_insert_with(|| {
                Tensor::zeros(
                    grad_output.shape().clone(),
                    grad_output.dtype(),
                    grad_output.device(),
                    false,
                )
            });
            let selected = minmax::select_with_mask(&mask, zero, grad_output)?;
            let reduced = reduce_gradient_for_broadcasting(
                &selected,
                &Shape::new(self.input_shapes[1].clone()),
            )?;
            gradients.insert(self.input_ids[1], reduced);
        }

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}

/// Gradient function for element-wise minimum operation
pub struct MinimumBackward {
    pub lhs: Tensor,
    pub rhs: Tensor,
    pub input_shapes: [Vec<usize>; 2],
    pub input_requires_grad: [bool; 2],
    pub input_ids: [TensorId; 2],
}

impl GradientFunction for MinimumBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(self.input_requires_grad.iter().filter(|&&b| b).count());

        if !self.input_requires_grad[0] && !self.input_requires_grad[1] {
            return Ok(gradients);
        }

        let mask = minmax::minimum_backward_mask(&self.lhs, &self.rhs)?;
        let mut zeros: Option<Tensor> = None;

        if self.input_requires_grad[0] {
            let zero = zeros.get_or_insert_with(|| {
                Tensor::zeros(
                    grad_output.shape().clone(),
                    grad_output.dtype(),
                    grad_output.device(),
                    false,
                )
            });
            let selected = minmax::select_with_mask(&mask, grad_output, zero)?;
            let reduced = reduce_gradient_for_broadcasting(
                &selected,
                &Shape::new(self.input_shapes[0].clone()),
            )?;
            gradients.insert(self.input_ids[0], reduced);
        }

        if self.input_requires_grad[1] {
            let zero = zeros.get_or_insert_with(|| {
                Tensor::zeros(
                    grad_output.shape().clone(),
                    grad_output.dtype(),
                    grad_output.device(),
                    false,
                )
            });
            let selected = minmax::select_with_mask(&mask, zero, grad_output)?;
            let reduced = reduce_gradient_for_broadcasting(
                &selected,
                &Shape::new(self.input_shapes[1].clone()),
            )?;
            gradients.insert(self.input_ids[1], reduced);
        }

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}

/// Gradient function for dot product
pub struct DotBackward {
    pub lhs: Tensor,
    pub rhs: Tensor,
    pub input_ids: [TensorId; 2],
    pub lhs_requires_grad: bool,
    pub rhs_requires_grad: bool,
}

impl GradientFunction for DotBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve((self.lhs_requires_grad as usize) + (self.rhs_requires_grad as usize));

        if self.lhs_requires_grad {
            let grad = crate::operations::arithmetic::mul(&self.rhs, grad_output)?;
            gradients.insert(self.input_ids[0], grad);
        }

        if self.rhs_requires_grad {
            let grad = crate::operations::arithmetic::mul(&self.lhs, grad_output)?;
            gradients.insert(self.input_ids[1], grad);
        }

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}

/// Gradient function for negation
pub struct NegBackward {
    pub input_id: TensorId,
}

impl GradientFunction for NegBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);
        let grad = arithmetic::neg(grad_output)?;
        gradients.insert(self.input_id, grad);
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for matrix multiplication
pub struct MatMulBackward {
    pub lhs: Tensor,
    pub rhs: Tensor,
    pub input_ids: [TensorId; 2],
    pub lhs_requires_grad: bool,
    pub rhs_requires_grad: bool,
}

impl GradientFunction for MatMulBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve((self.lhs_requires_grad as usize) + (self.rhs_requires_grad as usize));

        if self.lhs.ndim() < 2 || self.rhs.ndim() < 2 {
            return Err(MinitensorError::invalid_operation(
                "MatMulBackward requires tensors with at least 2 dimensions",
            ));
        }

        if self.lhs_requires_grad {
            let rhs_t = crate::operations::linalg::transpose(
                &self.rhs,
                (self.rhs.ndim() - 2) as isize,
                (self.rhs.ndim() - 1) as isize,
            )?;
            let lhs_grad = crate::operations::linalg::matmul(grad_output, &rhs_t)?;
            gradients.insert(self.input_ids[0], lhs_grad);
        }

        if self.rhs_requires_grad {
            let lhs_t = crate::operations::linalg::transpose(
                &self.lhs,
                (self.lhs.ndim() - 2) as isize,
                (self.lhs.ndim() - 1) as isize,
            )?;
            let rhs_grad = crate::operations::linalg::matmul(&lhs_t, grad_output)?;
            gradients.insert(self.input_ids[1], rhs_grad);
        }

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}

/// Gradient function for transpose operation
pub struct TransposeBackward {
    pub dims: Vec<usize>,
    pub input_id: TensorId,
}

impl GradientFunction for TransposeBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        // Transpose gradient: transpose back. Support both simple swaps and
        // arbitrary dimension permutations by applying the inverse permutation.
        let grad_input = if self.dims.len() == 2 {
            crate::operations::linalg::transpose(
                grad_output,
                self.dims[0] as isize,
                self.dims[1] as isize,
            )?
        } else {
            let mut inverse = vec![0; self.dims.len()];
            for (i, &d) in self.dims.iter().enumerate() {
                inverse[d] = i;
            }
            let mut grad = grad_output.clone();
            let mut current: Vec<usize> = (0..inverse.len()).collect();
            for i in 0..inverse.len() {
                let j = current
                    .iter()
                    .position(|&x| x == inverse[i])
                    .expect("invalid permutation");
                if i != j {
                    grad = crate::operations::linalg::transpose(&grad, i as isize, j as isize)?;
                    current.swap(i, j);
                }
            }
            grad
        };

        gradients.insert(self.input_id, grad_input);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
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
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

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

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for product reduction
pub struct ProdBackward {
    pub input: Tensor,
    pub result: Tensor,
    pub input_id: TensorId,
    pub dims: Option<Vec<usize>>,
    pub keepdim: bool,
}

impl GradientFunction for ProdBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

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
                grad = shape_ops::reshape(&grad, Shape::new(vec![1; self.input.ndim()]))?;
            }
        }

        let mut prod = self.result.clone();
        if !self.keepdim {
            if let Some(dims) = &self.dims {
                let mut shape = prod.shape().dims().to_vec();
                let mut sorted = dims.clone();
                sorted.sort_unstable();
                for &d in &sorted {
                    shape.insert(d, 1);
                }
                prod = shape_ops::reshape(&prod, Shape::new(shape))?;
            } else {
                prod = shape_ops::reshape(&prod, Shape::new(vec![1; self.input.ndim()]))?;
            }
        }

        let div = arithmetic::div(&prod, &self.input)?;
        let grad_input = arithmetic::mul(&grad, &div)?;
        gradients.insert(self.input_id, grad_input);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for cumulative sum operation
pub struct CumsumBackward {
    pub input_id: TensorId,
    pub dim: usize,
}

/// Gradient function for cumulative product operation
pub struct CumprodBackward {
    pub input_id: TensorId,
    pub input: Tensor,
    pub output: Tensor,
    pub dim: usize,
}

impl GradientFunction for CumprodBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        let grad_input =
            reduction::cumprod_backward(&self.input, &self.output, grad_output, self.dim)?;
        gradients.insert(self.input_id, grad_input);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

impl GradientFunction for CumsumBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        let grad_input = reduction::cumsum_backward(grad_output, self.dim)?;
        gradients.insert(self.input_id, grad_input);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

// Gradient functions for activation functions

/// Gradient function for exponential
pub struct ExpBackward {
    pub input_id: TensorId,
    pub output: Tensor,
}

impl GradientFunction for ExpBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        // d/dx(exp(x)) = exp(x) * grad_output
        let grad = arithmetic::mul(&self.output, grad_output)?;
        gradients.insert(self.input_id, grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for logarithm
pub struct LogBackward {
    pub input_id: TensorId,
    pub input: Tensor,
}

impl GradientFunction for LogBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

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

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for log1p
pub struct Log1pBackward {
    pub input_id: TensorId,
    pub input: Tensor,
}

impl GradientFunction for Log1pBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        let ones = Tensor::ones(
            self.input.shape().clone(),
            self.input.dtype(),
            self.input.device(),
            false,
        );
        let denom = arithmetic::add(&ones, &self.input.detach())?;
        let grad = arithmetic::div(grad_output, &denom)?;
        gradients.insert(self.input_id, grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for expm1
pub struct Expm1Backward {
    pub input_id: TensorId,
    pub output: Tensor,
}

impl GradientFunction for Expm1Backward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        let ones = Tensor::ones(
            self.output.shape().clone(),
            self.output.dtype(),
            self.output.device(),
            false,
        );
        let term = arithmetic::add(&self.output.detach(), &ones)?;
        let grad = arithmetic::mul(&term, grad_output)?;
        gradients.insert(self.input_id, grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for sine
pub struct SinBackward {
    pub input_id: TensorId,
    pub input: Tensor,
}

impl GradientFunction for SinBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        // d/dx(sin(x)) = cos(x) * grad_output
        let cos_x = self.input.cos()?;
        let grad = arithmetic::mul(&cos_x, grad_output)?;
        gradients.insert(self.input_id, grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for cosine
pub struct CosBackward {
    pub input_id: TensorId,
    pub input: Tensor,
}

impl GradientFunction for CosBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        // d/dx(cos(x)) = -sin(x) * grad_output
        let sin_x = self.input.sin()?;
        let mul = arithmetic::mul(&sin_x, grad_output)?;
        let grad = arithmetic::neg(&mul)?;
        gradients.insert(self.input_id, grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for tangent
pub struct TanBackward {
    pub input_id: TensorId,
    pub output: Tensor,
}

impl GradientFunction for TanBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

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

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for tanh
pub struct TanhBackward {
    pub input_id: TensorId,
    pub output: Tensor,
}

impl GradientFunction for TanhBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

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

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for sigmoid
pub struct SigmoidBackward {
    pub input_id: TensorId,
    pub output: Tensor,
}

impl GradientFunction for SigmoidBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

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

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for Softplus
pub struct SoftplusBackward {
    pub input_id: TensorId,
    pub input: Tensor,
    pub beta: f64,
    pub threshold: f64,
}

impl GradientFunction for SoftplusBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        match self.input.dtype() {
            DataType::Float32 => {
                let input_slice = self.input.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from input tensor")
                })?;
                let grad_out_slice = grad_output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get f32 slice from grad_output tensor",
                    )
                })?;

                let mut grad_data = TensorData::uninitialized_on_device(
                    input_slice.len(),
                    DataType::Float32,
                    self.input.device(),
                );
                let grad_slice = grad_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice from gradient tensor",
                    )
                })?;

                let beta = self.beta as f32;
                let threshold = self.threshold as f32;
                for ((grad_slot, &x), &gout) in grad_slice
                    .iter_mut()
                    .zip(input_slice.iter())
                    .zip(grad_out_slice.iter())
                {
                    let scaled = beta * x;
                    *grad_slot = if scaled > threshold {
                        gout
                    } else {
                        gout / (1.0 + (-scaled).exp())
                    };
                }

                let grad_tensor = Tensor::new(
                    Arc::new(grad_data),
                    self.input.shape().clone(),
                    DataType::Float32,
                    self.input.device(),
                    false,
                );
                gradients.insert(self.input_id, grad_tensor);
            }
            DataType::Float64 => {
                let input_slice = self.input.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from input tensor")
                })?;
                let grad_out_slice = grad_output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get f64 slice from grad_output tensor",
                    )
                })?;

                let mut grad_data = TensorData::uninitialized_on_device(
                    input_slice.len(),
                    DataType::Float64,
                    self.input.device(),
                );
                let grad_slice = grad_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice from gradient tensor",
                    )
                })?;

                let beta = self.beta;
                let threshold = self.threshold;
                for ((grad_slot, &x), &gout) in grad_slice
                    .iter_mut()
                    .zip(input_slice.iter())
                    .zip(grad_out_slice.iter())
                {
                    let scaled = beta * x;
                    *grad_slot = if scaled > threshold {
                        gout
                    } else {
                        gout / (1.0 + (-scaled).exp())
                    };
                }

                let grad_tensor = Tensor::new(
                    Arc::new(grad_data),
                    self.input.shape().clone(),
                    DataType::Float64,
                    self.input.device(),
                    false,
                );
                gradients.insert(self.input_id, grad_tensor);
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "Softplus gradient only defined for floating point tensors",
                ));
            }
        }

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for GELU activation
pub struct GeluBackward {
    pub input_id: TensorId,
    pub input: Tensor,
    pub approximate: bool,
}

impl GradientFunction for GeluBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        match self.input.dtype() {
            DataType::Float32 => {
                let input_slice = self.input.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from input tensor")
                })?;
                let grad_out_slice = grad_output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get f32 slice from grad_output tensor",
                    )
                })?;

                let mut grad_data = TensorData::uninitialized_on_device(
                    input_slice.len(),
                    DataType::Float32,
                    self.input.device(),
                );
                let grad_slice = grad_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice from gradient tensor",
                    )
                })?;

                if self.approximate {
                    let coeff = (2.0f32 / std::f32::consts::PI).sqrt();
                    for ((grad_slot, &x), &gout) in grad_slice
                        .iter_mut()
                        .zip(input_slice.iter())
                        .zip(grad_out_slice.iter())
                    {
                        let x2 = x * x;
                        let inner = coeff * (x + 0.044715f32 * x * x2);
                        let tanh_inner = inner.tanh();
                        let sech2 = 1.0f32 - tanh_inner * tanh_inner;
                        let grad_val = 0.5f32 * (1.0f32 + tanh_inner)
                            + 0.5f32 * x * sech2 * coeff * (1.0f32 + 3.0f32 * 0.044715f32 * x2);
                        *grad_slot = gout * grad_val;
                    }
                } else {
                    let inv_sqrt_2 = std::f32::consts::FRAC_1_SQRT_2;
                    let inv_sqrt_2pi = 1.0f32 / ((2.0f32 * std::f32::consts::PI).sqrt());
                    for ((grad_slot, &x), &gout) in grad_slice
                        .iter_mut()
                        .zip(input_slice.iter())
                        .zip(grad_out_slice.iter())
                    {
                        let cdf = 0.5f32 * (1.0f32 + erff(x * inv_sqrt_2));
                        let pdf = (-0.5f32 * x * x).exp() * inv_sqrt_2pi;
                        let grad_val = cdf + x * pdf;
                        *grad_slot = gout * grad_val;
                    }
                }

                let grad_tensor = Tensor::new(
                    Arc::new(grad_data),
                    self.input.shape().clone(),
                    DataType::Float32,
                    self.input.device(),
                    false,
                );
                gradients.insert(self.input_id, grad_tensor);
            }
            DataType::Float64 => {
                let input_slice = self.input.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from input tensor")
                })?;
                let grad_out_slice = grad_output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get f64 slice from grad_output tensor",
                    )
                })?;

                let mut grad_data = TensorData::uninitialized_on_device(
                    input_slice.len(),
                    DataType::Float64,
                    self.input.device(),
                );
                let grad_slice = grad_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice from gradient tensor",
                    )
                })?;

                if self.approximate {
                    let coeff = (2.0f64 / std::f64::consts::PI).sqrt();
                    for ((grad_slot, &x), &gout) in grad_slice
                        .iter_mut()
                        .zip(input_slice.iter())
                        .zip(grad_out_slice.iter())
                    {
                        let x2 = x * x;
                        let inner = coeff * (x + 0.044715f64 * x * x2);
                        let tanh_inner = inner.tanh();
                        let sech2 = 1.0f64 - tanh_inner * tanh_inner;
                        let grad_val = 0.5f64 * (1.0f64 + tanh_inner)
                            + 0.5f64 * x * sech2 * coeff * (1.0f64 + 3.0f64 * 0.044715f64 * x2);
                        *grad_slot = gout * grad_val;
                    }
                } else {
                    let inv_sqrt_2 = std::f64::consts::FRAC_1_SQRT_2;
                    let inv_sqrt_2pi = 1.0f64 / ((2.0f64 * std::f64::consts::PI).sqrt());
                    for ((grad_slot, &x), &gout) in grad_slice
                        .iter_mut()
                        .zip(input_slice.iter())
                        .zip(grad_out_slice.iter())
                    {
                        let cdf = 0.5f64 * (1.0f64 + erf(x * inv_sqrt_2));
                        let pdf = (-0.5f64 * x * x).exp() * inv_sqrt_2pi;
                        let grad_val = cdf + x * pdf;
                        *grad_slot = gout * grad_val;
                    }
                }

                let grad_tensor = Tensor::new(
                    Arc::new(grad_data),
                    self.input.shape().clone(),
                    DataType::Float64,
                    self.input.device(),
                    false,
                );
                gradients.insert(self.input_id, grad_tensor);
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "GELU backward only supports floating point tensors",
                ));
            }
        }

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for ELU activation
pub struct EluBackward {
    pub input_id: TensorId,
    pub output: Tensor,
    pub alpha: f64,
}

impl GradientFunction for EluBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        match self.output.dtype() {
            DataType::Float32 => {
                let output_slice = self.output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from output tensor")
                })?;
                let grad_out_slice = grad_output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get f32 slice from grad_output tensor",
                    )
                })?;

                let mut grad_data = TensorData::uninitialized_on_device(
                    output_slice.len(),
                    DataType::Float32,
                    self.output.device(),
                );
                let grad_slice = grad_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice from gradient tensor",
                    )
                })?;

                let alpha = self.alpha as f32;
                for ((grad_slot, &out), &gout) in grad_slice
                    .iter_mut()
                    .zip(output_slice.iter())
                    .zip(grad_out_slice.iter())
                {
                    let local_grad = if out > 0.0f32 { 1.0f32 } else { out + alpha };
                    *grad_slot = gout * local_grad;
                }

                let grad_tensor = Tensor::new(
                    Arc::new(grad_data),
                    self.output.shape().clone(),
                    DataType::Float32,
                    self.output.device(),
                    false,
                );
                gradients.insert(self.input_id, grad_tensor);
            }
            DataType::Float64 => {
                let output_slice = self.output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from output tensor")
                })?;
                let grad_out_slice = grad_output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get f64 slice from grad_output tensor",
                    )
                })?;

                let mut grad_data = TensorData::uninitialized_on_device(
                    output_slice.len(),
                    DataType::Float64,
                    self.output.device(),
                );
                let grad_slice = grad_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice from gradient tensor",
                    )
                })?;

                for ((grad_slot, &out), &gout) in grad_slice
                    .iter_mut()
                    .zip(output_slice.iter())
                    .zip(grad_out_slice.iter())
                {
                    let local_grad = if out > 0.0f64 {
                        1.0f64
                    } else {
                        out + self.alpha
                    };
                    *grad_slot = gout * local_grad;
                }

                let grad_tensor = Tensor::new(
                    Arc::new(grad_data),
                    self.output.shape().clone(),
                    DataType::Float64,
                    self.output.device(),
                    false,
                );
                gradients.insert(self.input_id, grad_tensor);
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "ELU backward only supports floating point tensors",
                ));
            }
        }

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for SELU activation
pub struct SeluBackward {
    pub input_id: TensorId,
    pub output: Tensor,
}

impl GradientFunction for SeluBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        match self.output.dtype() {
            DataType::Float32 => {
                let output_slice = self.output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from output tensor")
                })?;
                let grad_out_slice = grad_output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get f32 slice from grad_output tensor",
                    )
                })?;

                let mut grad_data = TensorData::uninitialized_on_device(
                    output_slice.len(),
                    DataType::Float32,
                    self.output.device(),
                );
                let grad_slice = grad_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice from gradient tensor",
                    )
                })?;

                const SCALE: f32 = 1.050701;
                const ALPHA: f32 = 1.6732632;
                for ((grad_slot, &out), &gout) in grad_slice
                    .iter_mut()
                    .zip(output_slice.iter())
                    .zip(grad_out_slice.iter())
                {
                    let local_grad = if out > 0.0f32 {
                        SCALE
                    } else {
                        out + SCALE * ALPHA
                    };
                    *grad_slot = gout * local_grad;
                }

                let grad_tensor = Tensor::new(
                    Arc::new(grad_data),
                    self.output.shape().clone(),
                    DataType::Float32,
                    self.output.device(),
                    false,
                );
                gradients.insert(self.input_id, grad_tensor);
            }
            DataType::Float64 => {
                let output_slice = self.output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from output tensor")
                })?;
                let grad_out_slice = grad_output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get f64 slice from grad_output tensor",
                    )
                })?;

                let mut grad_data = TensorData::uninitialized_on_device(
                    output_slice.len(),
                    DataType::Float64,
                    self.output.device(),
                );
                let grad_slice = grad_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice from gradient tensor",
                    )
                })?;

                const SCALE: f64 = 1.0507009873554804934193349852946;
                const ALPHA: f64 = 1.6732632423543772848170429916717;
                for ((grad_slot, &out), &gout) in grad_slice
                    .iter_mut()
                    .zip(output_slice.iter())
                    .zip(grad_out_slice.iter())
                {
                    let local_grad = if out > 0.0f64 {
                        SCALE
                    } else {
                        out + SCALE * ALPHA
                    };
                    *grad_slot = gout * local_grad;
                }

                let grad_tensor = Tensor::new(
                    Arc::new(grad_data),
                    self.output.shape().clone(),
                    DataType::Float64,
                    self.output.device(),
                    false,
                );
                gradients.insert(self.input_id, grad_tensor);
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "SELU backward only supports floating point tensors",
                ));
            }
        }

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for SiLU activation
pub struct SiluBackward {
    pub input_id: TensorId,
    pub input: Tensor,
}

impl GradientFunction for SiluBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        match self.input.dtype() {
            DataType::Float32 => {
                let input_slice = self.input.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from input tensor")
                })?;
                let grad_out_slice = grad_output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get f32 slice from grad_output tensor",
                    )
                })?;

                let mut grad_data = TensorData::uninitialized_on_device(
                    input_slice.len(),
                    DataType::Float32,
                    self.input.device(),
                );
                let grad_slice = grad_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice from gradient tensor",
                    )
                })?;

                for ((grad_slot, &x), &gout) in grad_slice
                    .iter_mut()
                    .zip(input_slice.iter())
                    .zip(grad_out_slice.iter())
                {
                    let sigmoid = 1.0f32 / (1.0f32 + (-x).exp());
                    let grad_val = sigmoid * (1.0f32 + x * (1.0f32 - sigmoid));
                    *grad_slot = gout * grad_val;
                }

                let grad_tensor = Tensor::new(
                    Arc::new(grad_data),
                    self.input.shape().clone(),
                    DataType::Float32,
                    self.input.device(),
                    false,
                );
                gradients.insert(self.input_id, grad_tensor);
            }
            DataType::Float64 => {
                let input_slice = self.input.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from input tensor")
                })?;
                let grad_out_slice = grad_output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get f64 slice from grad_output tensor",
                    )
                })?;

                let mut grad_data = TensorData::uninitialized_on_device(
                    input_slice.len(),
                    DataType::Float64,
                    self.input.device(),
                );
                let grad_slice = grad_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice from gradient tensor",
                    )
                })?;

                for ((grad_slot, &x), &gout) in grad_slice
                    .iter_mut()
                    .zip(input_slice.iter())
                    .zip(grad_out_slice.iter())
                {
                    let sigmoid = 1.0f64 / (1.0f64 + (-x).exp());
                    let grad_val = sigmoid * (1.0f64 + x * (1.0f64 - sigmoid));
                    *grad_slot = gout * grad_val;
                }

                let grad_tensor = Tensor::new(
                    Arc::new(grad_data),
                    self.input.shape().clone(),
                    DataType::Float64,
                    self.input.device(),
                    false,
                );
                gradients.insert(self.input_id, grad_tensor);
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "SiLU backward only supports floating point tensors",
                ));
            }
        }

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for Softsign activation
pub struct SoftsignBackward {
    pub input_id: TensorId,
    pub input: Tensor,
}

impl GradientFunction for SoftsignBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        match self.input.dtype() {
            DataType::Float32 => {
                let input_slice = self.input.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from input tensor")
                })?;
                let grad_out_slice = grad_output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get f32 slice from grad_output tensor",
                    )
                })?;

                let mut grad_data = TensorData::uninitialized_on_device(
                    input_slice.len(),
                    DataType::Float32,
                    self.input.device(),
                );
                let grad_slice = grad_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice from gradient tensor",
                    )
                })?;

                for ((grad_slot, &x), &gout) in grad_slice
                    .iter_mut()
                    .zip(input_slice.iter())
                    .zip(grad_out_slice.iter())
                {
                    let denom = 1.0f32 + x.abs();
                    let local_grad = 1.0f32 / (denom * denom);
                    *grad_slot = gout * local_grad;
                }

                let grad_tensor = Tensor::new(
                    Arc::new(grad_data),
                    self.input.shape().clone(),
                    DataType::Float32,
                    self.input.device(),
                    false,
                );
                gradients.insert(self.input_id, grad_tensor);
            }
            DataType::Float64 => {
                let input_slice = self.input.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from input tensor")
                })?;
                let grad_out_slice = grad_output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get f64 slice from grad_output tensor",
                    )
                })?;

                let mut grad_data = TensorData::uninitialized_on_device(
                    input_slice.len(),
                    DataType::Float64,
                    self.input.device(),
                );
                let grad_slice = grad_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice from gradient tensor",
                    )
                })?;

                for ((grad_slot, &x), &gout) in grad_slice
                    .iter_mut()
                    .zip(input_slice.iter())
                    .zip(grad_out_slice.iter())
                {
                    let denom = 1.0f64 + x.abs();
                    let local_grad = 1.0f64 / (denom * denom);
                    *grad_slot = gout * local_grad;
                }

                let grad_tensor = Tensor::new(
                    Arc::new(grad_data),
                    self.input.shape().clone(),
                    DataType::Float64,
                    self.input.device(),
                    false,
                );
                gradients.insert(self.input_id, grad_tensor);
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "Softsign backward only supports floating point tensors",
                ));
            }
        }

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for power operation
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PowBroadcast {
    None,
    BaseScalar,
    ExponentScalar,
}
pub struct PowBackward {
    pub base: Tensor,
    pub exponent: Tensor,
    pub output: Tensor,
    pub input_ids: [TensorId; 2],
    pub base_requires_grad: bool,
    pub exp_requires_grad: bool,
    pub broadcast: PowBroadcast,
}

/// Gradient function for logaddexp
pub struct LogAddExpBackward {
    pub lhs: Tensor,
    pub rhs: Tensor,
    pub output: Tensor,
    pub input_ids: [TensorId; 2],
    pub input_shapes: [Vec<usize>; 2],
}

impl GradientFunction for LogAddExpBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(2);

        let lhs_diff = arithmetic::sub(&self.lhs.detach(), &self.output.detach())?;
        let lhs_term = lhs_diff.exp()?;
        let lhs_mul = arithmetic::mul(&lhs_term, grad_output)?;
        let lhs_grad =
            reduce_gradient_for_broadcasting(&lhs_mul, &Shape::new(self.input_shapes[0].clone()))?;
        gradients.insert(self.input_ids[0], lhs_grad);

        let rhs_diff = arithmetic::sub(&self.rhs.detach(), &self.output.detach())?;
        let rhs_term = rhs_diff.exp()?;
        let rhs_mul = arithmetic::mul(&rhs_term, grad_output)?;
        let rhs_grad =
            reduce_gradient_for_broadcasting(&rhs_mul, &Shape::new(self.input_shapes[1].clone()))?;
        gradients.insert(self.input_ids[1], rhs_grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}

impl GradientFunction for PowBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(2);

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

                    match self.broadcast {
                        PowBroadcast::None => {
                            let len = base_slice.len();
                            if len < PAR_THRESHOLD {
                                for i in 0..len {
                                    grad_slice[i] = exp_slice[i]
                                        * base_slice[i].powf(exp_slice[i] - 1.0)
                                        * grad_out[i];
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
                        }
                        PowBroadcast::BaseScalar => {
                            let base_val = base_slice[0];
                            let mut accum = 0.0_f32;
                            for i in 0..grad_out.len() {
                                accum +=
                                    exp_slice[i] * base_val.powf(exp_slice[i] - 1.0) * grad_out[i];
                            }
                            grad_slice[0] = accum;
                        }
                        PowBroadcast::ExponentScalar => {
                            let exp_val = exp_slice[0];
                            let len = base_slice.len();
                            if len < PAR_THRESHOLD {
                                for i in 0..len {
                                    grad_slice[i] =
                                        exp_val * base_slice[i].powf(exp_val - 1.0) * grad_out[i];
                                }
                            } else {
                                let base_ptr = base_slice.as_ptr() as usize;
                                let go_ptr = grad_out.as_ptr() as usize;
                                let grad_ptr = grad_slice.as_mut_ptr() as usize;
                                (0..len).into_par_iter().for_each(|i| unsafe {
                                    let base_ptr = base_ptr as *const f32;
                                    let go_ptr = go_ptr as *const f32;
                                    let grad_ptr = grad_ptr as *mut f32;
                                    *grad_ptr.add(i) = exp_val
                                        * (*base_ptr.add(i)).powf(exp_val - 1.0)
                                        * *go_ptr.add(i);
                                });
                            }
                        }
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

                    match self.broadcast {
                        PowBroadcast::None => {
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
                        }
                        PowBroadcast::BaseScalar => {
                            let base_val = base_slice[0];
                            for i in 0..grad_out.len() {
                                grad_slice[i] = out_slice[i] * base_val.ln() * grad_out[i];
                            }
                        }
                        PowBroadcast::ExponentScalar => {
                            let mut accum = 0.0_f32;
                            for i in 0..grad_out.len() {
                                accum += out_slice[i] * base_slice[i].ln() * grad_out[i];
                            }
                            grad_slice[0] = accum;
                        }
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

                    match self.broadcast {
                        PowBroadcast::None => {
                            let len = base_slice.len();
                            if len < PAR_THRESHOLD {
                                for i in 0..len {
                                    grad_slice[i] = exp_slice[i]
                                        * base_slice[i].powf(exp_slice[i] - 1.0)
                                        * grad_out[i];
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
                        }
                        PowBroadcast::BaseScalar => {
                            let base_val = base_slice[0];
                            let mut accum = 0.0_f64;
                            for i in 0..grad_out.len() {
                                accum +=
                                    exp_slice[i] * base_val.powf(exp_slice[i] - 1.0) * grad_out[i];
                            }
                            grad_slice[0] = accum;
                        }
                        PowBroadcast::ExponentScalar => {
                            let exp_val = exp_slice[0];
                            let len = base_slice.len();
                            if len < PAR_THRESHOLD {
                                for i in 0..len {
                                    grad_slice[i] =
                                        exp_val * base_slice[i].powf(exp_val - 1.0) * grad_out[i];
                                }
                            } else {
                                let base_ptr = base_slice.as_ptr() as usize;
                                let go_ptr = grad_out.as_ptr() as usize;
                                let grad_ptr = grad_slice.as_mut_ptr() as usize;
                                (0..len).into_par_iter().for_each(|i| unsafe {
                                    let base_ptr = base_ptr as *const f64;
                                    let go_ptr = go_ptr as *const f64;
                                    let grad_ptr = grad_ptr as *mut f64;
                                    *grad_ptr.add(i) = exp_val
                                        * (*base_ptr.add(i)).powf(exp_val - 1.0)
                                        * *go_ptr.add(i);
                                });
                            }
                        }
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

                    match self.broadcast {
                        PowBroadcast::None => {
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
                        }
                        PowBroadcast::BaseScalar => {
                            let base_val = base_slice[0];
                            for i in 0..grad_out.len() {
                                grad_slice[i] = out_slice[i] * base_val.ln() * grad_out[i];
                            }
                        }
                        PowBroadcast::ExponentScalar => {
                            let mut accum = 0.0_f64;
                            for i in 0..grad_out.len() {
                                accum += out_slice[i] * base_slice[i].ln() * grad_out[i];
                            }
                            grad_slice[0] = accum;
                        }
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
                ));
            }
        }

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}

/// Gradient function for Hardshrink
pub struct HardshrinkBackward {
    pub input_id: TensorId,
    pub mask: Vec<bool>,
}

impl GradientFunction for HardshrinkBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

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
                        grad_slice[i] = if self.mask[i] { go[i] } else { 0.0 };
                    }
                } else {
                    let mask = &self.mask;
                    let go_ptr = go.as_ptr() as usize;
                    let grad_ptr = grad_slice.as_mut_ptr() as usize;
                    (0..len).into_par_iter().for_each(|i| unsafe {
                        let go_ptr = go_ptr as *const f32;
                        let grad_ptr = grad_ptr as *mut f32;
                        if *mask.get_unchecked(i) {
                            *grad_ptr.add(i) = *go_ptr.add(i);
                        } else {
                            *grad_ptr.add(i) = 0.0;
                        }
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
                        grad_slice[i] = if self.mask[i] { go[i] } else { 0.0 };
                    }
                } else {
                    let mask = &self.mask;
                    let go_ptr = go.as_ptr() as usize;
                    let grad_ptr = grad_slice.as_mut_ptr() as usize;
                    (0..len).into_par_iter().for_each(|i| unsafe {
                        let go_ptr = go_ptr as *const f64;
                        let grad_ptr = grad_ptr as *mut f64;
                        if *mask.get_unchecked(i) {
                            *grad_ptr.add(i) = *go_ptr.add(i);
                        } else {
                            *grad_ptr.add(i) = 0.0;
                        }
                    });
                }
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "hardshrink backward only supported for floating point tensors",
                ));
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

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for ReLU
pub struct ReluBackward {
    pub input_id: TensorId,
    pub mask: Vec<bool>,
}

impl GradientFunction for ReluBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

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
                ));
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

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for LeakyReLU
pub struct LeakyReluBackward {
    pub input_id: TensorId,
    pub negative_slope: f64,
    pub mask: Vec<bool>,
}

impl GradientFunction for LeakyReluBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

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
                ));
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

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for softmax
pub struct SoftmaxBackward {
    pub input_id: TensorId,
    pub output: Tensor,
    pub dim: usize,
}

impl GradientFunction for SoftmaxBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        // Allocate gradient buffer
        let mut grad_data = TensorData::zeros_on_device(
            self.output.numel(),
            self.output.dtype(),
            self.output.device(),
        );

        match grad_output.dtype() {
            DataType::Float32 => {
                let go = grad_output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from grad_output")
                })?;
                let y = self.output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from softmax output")
                })?;
                let grad_slice = grad_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice from grad_data",
                    )
                })?;
                softmax_backward_f32(go, y, grad_slice, self.output.shape().dims(), self.dim);
            }
            DataType::Float64 => {
                let go = grad_output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from grad_output")
                })?;
                let y = self.output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from softmax output")
                })?;
                let grad_slice = grad_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice from grad_data",
                    )
                })?;
                softmax_backward_f64(go, y, grad_slice, self.output.shape().dims(), self.dim);
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "Softmax backward only supported for floating point tensors",
                ));
            }
        }

        let grad_input = Tensor::new(
            Arc::new(grad_data),
            self.output.shape().clone(),
            self.output.dtype(),
            self.output.device(),
            grad_output.requires_grad(),
        );

        gradients.insert(self.input_id, grad_input);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for log-softmax
pub struct LogSoftmaxBackward {
    pub input_id: TensorId,
    pub output: Tensor,
    pub dim: usize,
}

impl GradientFunction for LogSoftmaxBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        let mut grad_data = TensorData::zeros_on_device(
            self.output.numel(),
            self.output.dtype(),
            self.output.device(),
        );

        match grad_output.dtype() {
            DataType::Float32 => {
                let go = grad_output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from grad_output")
                })?;
                let log_y = self.output.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get f32 slice from log_softmax output",
                    )
                })?;
                let grad_slice = grad_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice from grad_data",
                    )
                })?;
                log_softmax_backward_f32(
                    go,
                    log_y,
                    grad_slice,
                    self.output.shape().dims(),
                    self.dim,
                );
            }
            DataType::Float64 => {
                let go = grad_output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from grad_output")
                })?;
                let log_y = self.output.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get f64 slice from log_softmax output",
                    )
                })?;
                let grad_slice = grad_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice from grad_data",
                    )
                })?;
                log_softmax_backward_f64(
                    go,
                    log_y,
                    grad_slice,
                    self.output.shape().dims(),
                    self.dim,
                );
            }
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "LogSoftmax backward only supported for floating point tensors",
                ));
            }
        }

        let grad_input = Tensor::new(
            Arc::new(grad_data),
            self.output.shape().clone(),
            self.output.dtype(),
            self.output.device(),
            grad_output.requires_grad(),
        );

        gradients.insert(self.input_id, grad_input);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for layer normalization
pub struct LayerNormBackward {
    pub input_ids: SmallVec<[TensorId; 3]>,
    pub input_id: TensorId,
    pub weight_id: Option<TensorId>,
    pub bias_id: Option<TensorId>,
    pub normalized: Tensor,
    pub inv_std: Tensor,
    pub weight_broadcast: Option<Tensor>,
    pub normalized_shape: Vec<usize>,
    pub axis_start: usize,
    pub element_count: usize,
    pub input_requires_grad: bool,
    pub weight_requires_grad: bool,
    pub bias_requires_grad: bool,
}

impl GradientFunction for LayerNormBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();

        let grad_output_detached = grad_output.detach();
        let normalized = self.normalized.detach();

        if self.element_count == 0 {
            if self.input_requires_grad {
                let zero = Tensor::zeros(
                    grad_output.shape().clone(),
                    grad_output.dtype(),
                    grad_output.device(),
                    false,
                );
                gradients.insert(self.input_id, zero);
            }
            if self.weight_requires_grad {
                if let Some(weight_id) = self.weight_id {
                    let zero = Tensor::zeros(
                        Shape::new(self.normalized_shape.clone()),
                        grad_output.dtype(),
                        grad_output.device(),
                        false,
                    );
                    gradients.insert(weight_id, zero);
                }
            }
            if self.bias_requires_grad {
                if let Some(bias_id) = self.bias_id {
                    let zero = Tensor::zeros(
                        Shape::new(self.normalized_shape.clone()),
                        grad_output.dtype(),
                        grad_output.device(),
                        false,
                    );
                    gradients.insert(bias_id, zero);
                }
            }

            return Ok(gradients);
        }

        if self.input_requires_grad {
            let mut grad_output_hat = if let Some(weight) = &self.weight_broadcast {
                arithmetic::mul(&grad_output_detached, weight)?
            } else {
                grad_output_detached.clone()
            };

            let axes: Vec<isize> = (self.axis_start..grad_output_hat.ndim())
                .map(|d| d as isize)
                .collect();
            let sum_grad = reduction::sum(&grad_output_hat, Some(axes.clone()), true)?;
            let grad_norm_mul = arithmetic::mul(&grad_output_hat, &normalized)?;
            let sum_grad_norm = reduction::sum(&grad_norm_mul, Some(axes), true)?;

            let count = self.element_count as f64;
            let m_tensor = create_scalar_tensor(count, grad_output.dtype(), grad_output.device())?;
            let inv_m_tensor =
                create_scalar_tensor(1.0 / count, grad_output.dtype(), grad_output.device())?;
            grad_output_hat = arithmetic::mul(&grad_output_hat, &m_tensor)?;
            let tmp = arithmetic::sub(&grad_output_hat, &sum_grad)?;
            let norm_term = arithmetic::mul(&normalized, &sum_grad_norm)?;
            let numerator = arithmetic::sub(&tmp, &norm_term)?;
            let grad_input = arithmetic::mul(&numerator, &self.inv_std)?;
            let grad_input = arithmetic::mul(&grad_input, &inv_m_tensor)?;
            gradients.insert(self.input_id, grad_input);
        }

        if self.weight_requires_grad {
            if let Some(weight_id) = self.weight_id {
                let mut grad_weight = arithmetic::mul(&grad_output_detached, &normalized)?;
                if self.axis_start > 0 {
                    let axes: Vec<isize> = (0..self.axis_start).map(|d| d as isize).collect();
                    grad_weight = reduction::sum(&grad_weight, Some(axes), false)?;
                }
                if grad_weight.shape().dims() != self.normalized_shape.as_slice() {
                    grad_weight = grad_weight.view(Shape::new(self.normalized_shape.clone()))?;
                }
                gradients.insert(weight_id, grad_weight);
            }
        }

        if self.bias_requires_grad {
            if let Some(bias_id) = self.bias_id {
                let mut grad_bias = grad_output_detached.clone();
                if self.axis_start > 0 {
                    let axes: Vec<isize> = (0..self.axis_start).map(|d| d as isize).collect();
                    grad_bias = reduction::sum(&grad_bias, Some(axes), false)?;
                }
                if grad_bias.shape().dims() != self.normalized_shape.as_slice() {
                    grad_bias = grad_bias.view(Shape::new(self.normalized_shape.clone()))?;
                }
                gradients.insert(bias_id, grad_bias);
            }
        }

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}

fn softmax_backward_f32(
    grad_output: &[f32],
    y: &[f32],
    grad_input: &mut [f32],
    dims: &[usize],
    dim: usize,
) {
    let dim_size = dims[dim];
    let after: usize = if dim + 1 >= dims.len() {
        1
    } else {
        dims[dim + 1..].iter().product()
    };
    let group = dim_size * after;
    if grad_output.len() < PAR_THRESHOLD {
        for ((go_block, y_block), out_block) in grad_output
            .chunks(group)
            .zip(y.chunks(group))
            .zip(grad_input.chunks_mut(group))
        {
            for a in 0..after {
                let base = a;
                let mut dot = 0.0f32;
                for k in 0..dim_size {
                    let idx = base + k * after;
                    dot += go_block[idx] * y_block[idx];
                }
                for k in 0..dim_size {
                    let idx = base + k * after;
                    out_block[idx] = y_block[idx] * (go_block[idx] - dot);
                }
            }
        }
    } else {
        grad_output
            .par_chunks(group)
            .zip(y.par_chunks(group))
            .zip(grad_input.par_chunks_mut(group))
            .for_each(|((go_block, y_block), out_block)| {
                for a in 0..after {
                    let base = a;
                    let mut dot = 0.0f32;
                    for k in 0..dim_size {
                        let idx = base + k * after;
                        dot += go_block[idx] * y_block[idx];
                    }
                    for k in 0..dim_size {
                        let idx = base + k * after;
                        out_block[idx] = y_block[idx] * (go_block[idx] - dot);
                    }
                }
            });
    }
}

fn softmax_backward_f64(
    grad_output: &[f64],
    y: &[f64],
    grad_input: &mut [f64],
    dims: &[usize],
    dim: usize,
) {
    let dim_size = dims[dim];
    let after: usize = if dim + 1 >= dims.len() {
        1
    } else {
        dims[dim + 1..].iter().product()
    };
    let group = dim_size * after;
    if grad_output.len() < PAR_THRESHOLD {
        for ((go_block, y_block), out_block) in grad_output
            .chunks(group)
            .zip(y.chunks(group))
            .zip(grad_input.chunks_mut(group))
        {
            for a in 0..after {
                let base = a;
                let mut dot = 0.0f64;
                for k in 0..dim_size {
                    let idx = base + k * after;
                    dot += go_block[idx] * y_block[idx];
                }
                for k in 0..dim_size {
                    let idx = base + k * after;
                    out_block[idx] = y_block[idx] * (go_block[idx] - dot);
                }
            }
        }
    } else {
        grad_output
            .par_chunks(group)
            .zip(y.par_chunks(group))
            .zip(grad_input.par_chunks_mut(group))
            .for_each(|((go_block, y_block), out_block)| {
                for a in 0..after {
                    let base = a;
                    let mut dot = 0.0f64;
                    for k in 0..dim_size {
                        let idx = base + k * after;
                        dot += go_block[idx] * y_block[idx];
                    }
                    for k in 0..dim_size {
                        let idx = base + k * after;
                        out_block[idx] = y_block[idx] * (go_block[idx] - dot);
                    }
                }
            });
    }
}

fn log_softmax_backward_f32(
    grad_output: &[f32],
    log_y: &[f32],
    grad_input: &mut [f32],
    dims: &[usize],
    dim: usize,
) {
    let dim_size = dims[dim];
    let after: usize = if dim + 1 >= dims.len() {
        1
    } else {
        dims[dim + 1..].iter().product()
    };
    let group = dim_size * after;

    if grad_output.len() < PAR_THRESHOLD {
        for ((go_block, log_block), out_block) in grad_output
            .chunks(group)
            .zip(log_y.chunks(group))
            .zip(grad_input.chunks_mut(group))
        {
            for a in 0..after {
                let base = a;
                let mut sum = 0.0f32;
                for k in 0..dim_size {
                    let idx = base + k * after;
                    sum += go_block[idx];
                }
                for k in 0..dim_size {
                    let idx = base + k * after;
                    let prob = log_block[idx].exp();
                    out_block[idx] = go_block[idx] - prob * sum;
                }
            }
        }
    } else {
        grad_output
            .par_chunks(group)
            .zip(log_y.par_chunks(group))
            .zip(grad_input.par_chunks_mut(group))
            .for_each(|((go_block, log_block), out_block)| {
                for a in 0..after {
                    let base = a;
                    let mut sum = 0.0f32;
                    for k in 0..dim_size {
                        let idx = base + k * after;
                        sum += go_block[idx];
                    }
                    for k in 0..dim_size {
                        let idx = base + k * after;
                        let prob = log_block[idx].exp();
                        out_block[idx] = go_block[idx] - prob * sum;
                    }
                }
            });
    }
}

fn log_softmax_backward_f64(
    grad_output: &[f64],
    log_y: &[f64],
    grad_input: &mut [f64],
    dims: &[usize],
    dim: usize,
) {
    let dim_size = dims[dim];
    let after: usize = if dim + 1 >= dims.len() {
        1
    } else {
        dims[dim + 1..].iter().product()
    };
    let group = dim_size * after;

    if grad_output.len() < PAR_THRESHOLD {
        for ((go_block, log_block), out_block) in grad_output
            .chunks(group)
            .zip(log_y.chunks(group))
            .zip(grad_input.chunks_mut(group))
        {
            for a in 0..after {
                let base = a;
                let mut sum = 0.0f64;
                for k in 0..dim_size {
                    let idx = base + k * after;
                    sum += go_block[idx];
                }
                for k in 0..dim_size {
                    let idx = base + k * after;
                    let prob = log_block[idx].exp();
                    out_block[idx] = go_block[idx] - prob * sum;
                }
            }
        }
    } else {
        grad_output
            .par_chunks(group)
            .zip(log_y.par_chunks(group))
            .zip(grad_input.par_chunks_mut(group))
            .for_each(|((go_block, log_block), out_block)| {
                for a in 0..after {
                    let base = a;
                    let mut sum = 0.0f64;
                    for k in 0..dim_size {
                        let idx = base + k * after;
                        sum += go_block[idx];
                    }
                    for k in 0..dim_size {
                        let idx = base + k * after;
                        let prob = log_block[idx].exp();
                        out_block[idx] = go_block[idx] - prob * sum;
                    }
                }
            });
    }
}

/// Gradient function for reshape operation
pub struct ReshapeBackward {
    pub input_shape: Vec<usize>,
    pub input_id: TensorId,
}

impl GradientFunction for ReshapeBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        // Reshape gradient: reshape back to original shape
        let original_shape = Shape::new(self.input_shape.clone());
        let grad_input = crate::operations::shape_ops::reshape(grad_output, original_shape)?;
        gradients.insert(self.input_id, grad_input);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for repeat_interleave operation
pub struct RepeatInterleaveBackward {
    pub input_shape: Vec<usize>,
    pub repeats: Vec<usize>,
    pub input_id: TensorId,
    pub dim: usize,
}

impl GradientFunction for RepeatInterleaveBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let grad_input = repeat_interleave_backward_impl(
            grad_output,
            &self.input_shape,
            &self.repeats,
            self.dim,
        )?;

        let mut gradients = FxHashMap::default();
        gradients.insert(self.input_id, grad_input);
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

fn repeat_interleave_backward_impl(
    grad_output: &Tensor,
    input_shape: &[usize],
    repeats: &[usize],
    dim: usize,
) -> Result<Tensor> {
    if dim >= input_shape.len() {
        return Err(MinitensorError::index_error(
            dim as isize,
            0,
            input_shape.len(),
        ));
    }

    let dim_size = input_shape[dim];
    if repeats.len() != dim_size {
        return Err(MinitensorError::invalid_operation(
            "repeat_interleave backward: repeats must match input dimension size".to_string(),
        ));
    }

    let grad_shape_vec = input_shape.to_vec();
    let grad_shape = Shape::new(grad_shape_vec.clone());
    let numel = grad_shape.numel();
    let dtype = grad_output.dtype();
    let device = grad_output.device();
    let total_repeats: usize = repeats.iter().sum();

    let inner: usize = if dim + 1 >= input_shape.len() {
        1
    } else {
        input_shape[dim + 1..].iter().product()
    };
    let outer: usize = if dim == 0 {
        1
    } else {
        input_shape[..dim].iter().product()
    };

    if numel == 0 || total_repeats == 0 || inner == 0 || outer == 0 {
        return Ok(Tensor::zeros(
            Shape::new(grad_shape_vec),
            dtype,
            device,
            false,
        ));
    }

    let output_dims = grad_output.shape().dims();
    if output_dims.len() != input_shape.len() || output_dims[dim] != total_repeats {
        return Err(MinitensorError::shape_mismatch(
            input_shape.to_vec(),
            output_dims.to_vec(),
        ));
    }

    macro_rules! repeat_interleave_backward_impl_inner {
        ($ty:ty, $slice:ident, $from_vec:ident) => {{
            let src = grad_output.data().$slice().ok_or_else(|| {
                MinitensorError::invalid_operation(
                    "repeat_interleave backward: gradient tensor must be contiguous".to_string(),
                )
            })?;
            let mut dst = vec![<$ty>::default(); numel];
            let chunk = total_repeats * inner;
            dst.par_chunks_mut(dim_size * inner)
                .enumerate()
                .for_each(|(outer_idx, dst_chunk)| {
                    let mut src_offset = outer_idx * chunk;
                    for (i, &rep) in repeats.iter().enumerate() {
                        if rep == 0 {
                            continue;
                        }
                        let dst_start = i * inner;
                        let dst_slice = &mut dst_chunk[dst_start..dst_start + inner];
                        for _ in 0..rep {
                            let src_slice = &src[src_offset..src_offset + inner];
                            dst_slice.iter_mut().zip(src_slice.iter()).for_each(
                                |(dst_val, &src_val)| {
                                    *dst_val += src_val;
                                },
                            );
                            src_offset += inner;
                        }
                    }
                });
            TensorData::$from_vec(dst, device)
        }};
    }

    let data = match dtype {
        DataType::Float32 => {
            repeat_interleave_backward_impl_inner!(f32, as_f32_slice, from_vec_f32)
        }
        DataType::Float64 => {
            repeat_interleave_backward_impl_inner!(f64, as_f64_slice, from_vec_f64)
        }
        DataType::Int32 => repeat_interleave_backward_impl_inner!(i32, as_i32_slice, from_vec_i32),
        DataType::Int64 => repeat_interleave_backward_impl_inner!(i64, as_i64_slice, from_vec_i64),
        DataType::Bool => {
            return Ok(Tensor::zeros(grad_shape, dtype, device, false));
        }
    };

    Ok(Tensor::new(
        Arc::new(data),
        grad_shape,
        dtype,
        device,
        false,
    ))
}

/// Gradient function for expand operation which reduces broadcasted gradients
pub struct ExpandBackward {
    pub input_shape: Vec<usize>,
    pub input_id: TensorId,
}

impl GradientFunction for ExpandBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

        let shape = Shape::new(self.input_shape.clone());
        let grad_input = reduce_gradient_for_broadcasting(grad_output, &shape)?;
        gradients.insert(self.input_id, grad_input);
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }
}

/// Gradient function for MSE loss
pub struct MSELossBackward {
    pub predictions_shape: Vec<usize>,
    pub targets_shape: Vec<usize>,
    pub input_ids: [TensorId; 2],
    pub reduction: String,
    pub diff: Tensor,
}

impl GradientFunction for MSELossBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(2);

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
                )));
            }
        }

        // Multiply by upstream gradient
        let pred_grad = arithmetic::mul(&base_grad, grad_output)?;
        let target_grad = arithmetic::neg(&pred_grad)?;

        gradients.insert(self.input_ids[0], pred_grad);
        gradients.insert(self.input_ids[1], target_grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
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
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(2);

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
                )));
            }
        }

        let pred_grad = arithmetic::mul(&base_grad, grad_output)?;
        let target_grad = arithmetic::neg(&pred_grad)?;

        gradients.insert(self.input_ids[0], pred_grad);
        gradients.insert(self.input_ids[1], target_grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
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
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(2);

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
                ));
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
        let target_grad = arithmetic::neg(&pred_grad)?;

        gradients.insert(self.input_ids[0], pred_grad);
        gradients.insert(self.input_ids[1], target_grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
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
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

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
                        ));
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
                )));
            }
        }

        // Multiply by upstream gradient (handles broadcasting)
        let pred_grad = arithmetic::mul(&base_grad, grad_output)?;

        // Targets typically have no gradient
        gradients.insert(self.input_ids[0], pred_grad);

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
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
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

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

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
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
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(2);

        // Gradient w.r.t predictions: -(targets / predictions)
        let mut pred_grad = arithmetic::div(&self.targets, &self.predictions)?;
        pred_grad = arithmetic::neg(&pred_grad)?;
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

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
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
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(1);

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

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
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
            ));
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
            ));
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

    let grad_dims = grad_output.shape().dims();
    let target_dims = target_shape.dims();
    let extra = grad_dims.len() - target_dims.len();

    // Use a stack-allocated small vector and pre-allocate enough capacity to
    // hold all potential broadcast axes. This avoids repeated reallocations for
    // higher dimensional tensors.
    let mut axes_to_sum: SmallVec<[usize; 8]> = SmallVec::with_capacity(grad_dims.len());
    axes_to_sum.extend(0..extra);
    for i in 0..target_dims.len() {
        let gdim = grad_dims[extra + i];
        let tdim = target_dims[i];
        if tdim == 1 && gdim > 1 {
            axes_to_sum.push(extra + i);
        }
    }

    if axes_to_sum.is_empty() {
        return Ok(grad_output.clone());
    }

    let axes: Vec<isize> = axes_to_sum
        .into_vec()
        .into_iter()
        .map(|d| d as isize)
        .collect();
    let mut grad = reduction::sum(grad_output, Some(axes), true)?;

    if grad.shape() != target_shape {
        grad = grad.view(target_shape.clone())?;
    }

    Ok(grad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;
    use crate::tensor::DataType;

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

        graph.add_tensor_with_grad_req(tensor_id, Some(grad_fn), true);
        assert!(graph.nodes().contains_key(&tensor_id));
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
    fn test_reduce_gradient_for_broadcasting() {
        let grad_output = Tensor::ones(
            Shape::new(vec![2, 3]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let target_shape = Shape::new(vec![2, 1]);
        let reduced = reduce_gradient_for_broadcasting(&grad_output, &target_shape).unwrap();
        assert_eq!(reduced.shape().dims(), &[2, 1]);
        let slice = reduced.data().as_f32_slice().unwrap();
        assert!(slice.iter().all(|&x| (x - 3.0).abs() < 1e-6));
    }

    #[test]
    fn test_reduce_gradient_multiple_axes() {
        let grad_output = Tensor::ones(
            Shape::new(vec![2, 3]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let target_shape = Shape::new(vec![1, 1]);
        let reduced = reduce_gradient_for_broadcasting(&grad_output, &target_shape).unwrap();
        assert_eq!(reduced.shape().dims(), &[1, 1]);
        let slice = reduced.data().as_f32_slice().unwrap();
        assert!((slice[0] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduce_gradient_with_leading_and_inner_axes() {
        let grad_output = Tensor::ones(
            Shape::new(vec![4, 2, 3, 5]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let target_shape = Shape::new(vec![2, 1, 5]);
        let reduced = reduce_gradient_for_broadcasting(&grad_output, &target_shape).unwrap();
        assert_eq!(reduced.shape().dims(), &[2, 1, 5]);
        let slice = reduced.data().as_f32_slice().unwrap();
        assert!(slice.iter().all(|&x| (x - 12.0).abs() < 1e-6));
    }

    #[test]
    fn test_reduce_gradient_noop_for_same_shape() {
        let grad_output = Tensor::ones(
            Shape::new(vec![2, 3, 4]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let target_shape = grad_output.shape().clone();
        let reduced = reduce_gradient_for_broadcasting(&grad_output, &target_shape).unwrap();
        assert_eq!(reduced.shape().dims(), &[2, 3, 4]);
        assert!(reduced.allclose(&grad_output, 1e-6, 1e-6));
    }

    #[test]
    fn test_softmax_backward_dim1() {
        let input = Tensor::new(
            Arc::new(TensorData::from_vec_f32(
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                Device::cpu(),
            )),
            Shape::new(vec![2, 3]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let softmax_out = activation::softmax(&input, Some(1)).unwrap();
        let grad_output = Tensor::ones(
            Shape::new(vec![2, 3]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let grad_y = arithmetic::mul(&grad_output, &softmax_out).unwrap();
        let sum = reduction::sum(&grad_y, Some(vec![1]), true).unwrap();
        let sub = arithmetic::sub(&grad_output, &sum).unwrap();
        let expected = arithmetic::mul(&softmax_out, &sub).unwrap();

        let grad_fn = SoftmaxBackward {
            input_id: TensorId::new(),
            output: softmax_out.clone(),
            dim: 1,
        };
        let grads = grad_fn.backward(&grad_output).unwrap();
        let grad_input = grads.values().next().unwrap();
        assert!(grad_input.allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_softmax_backward_dim0_f64() {
        let data: Vec<f64> = (1..=6).map(|v| v as f64).collect();
        let input = Tensor::new(
            Arc::new(TensorData::from_vec_f64(data, Device::cpu())),
            Shape::new(vec![2, 3]),
            DataType::Float64,
            Device::cpu(),
            false,
        );
        let softmax_out = activation::softmax(&input, Some(0)).unwrap();
        let grad_output = Tensor::ones(
            Shape::new(vec![2, 3]),
            DataType::Float64,
            Device::cpu(),
            false,
        );
        let grad_y = arithmetic::mul(&grad_output, &softmax_out).unwrap();
        let sum = reduction::sum(&grad_y, Some(vec![0]), true).unwrap();
        let sub = arithmetic::sub(&grad_output, &sum).unwrap();
        let expected = arithmetic::mul(&softmax_out, &sub).unwrap();

        let grad_fn = SoftmaxBackward {
            input_id: TensorId::new(),
            output: softmax_out.clone(),
            dim: 0,
        };
        let grads = grad_fn.backward(&grad_output).unwrap();
        let grad_input = grads.values().next().unwrap();
        assert!(grad_input.allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_backward_broadcast_addition() {
        clear_graph().unwrap();

        let a = Tensor::ones(
            Shape::new(vec![2, 3]),
            DataType::Float32,
            Device::cpu(),
            true,
        );
        let b = Tensor::ones(Shape::new(vec![3]), DataType::Float32, Device::cpu(), true);
        let out = arithmetic::add(&a, &b).unwrap();

        let grad = Tensor::ones(out.shape().clone(), out.dtype(), out.device(), false);
        let grads = backward(&out, Some(grad)).unwrap();

        let grad_a = grads.get(&a.id()).unwrap();
        let grad_b = grads.get(&b.id()).unwrap();
        assert_eq!(grad_a.shape().dims(), &[2, 3]);
        assert_eq!(grad_b.shape().dims(), &[3]);
        let slice_b = grad_b.data().as_f32_slice().unwrap();
        assert!(slice_b.iter().all(|&x| (x - 2.0).abs() < 1e-6));
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
            lhs_requires_grad: true,
            rhs_requires_grad: true,
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

    #[test]
    fn test_matmul_backward_batched() {
        let lhs = Tensor::new(
            Arc::new(TensorData::from_vec_f32(
                (0..12).map(|x| x as f32).collect(),
                Device::cpu(),
            )),
            Shape::new(vec![2, 2, 3]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let rhs = Tensor::new(
            Arc::new(TensorData::from_vec_f32(
                (0..24).map(|x| x as f32).collect(),
                Device::cpu(),
            )),
            Shape::new(vec![2, 3, 4]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let input_ids = [TensorId::new(), TensorId::new()];
        let grad_fn = MatMulBackward {
            lhs: lhs.clone(),
            rhs: rhs.clone(),
            input_ids,
            lhs_requires_grad: true,
            rhs_requires_grad: true,
        };
        let grad_output = Tensor::ones(
            Shape::new(vec![2, 2, 4]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let grads = grad_fn.backward(&grad_output).unwrap();
        let rhs_t = crate::operations::linalg::transpose(
            &rhs,
            (rhs.ndim() - 2) as isize,
            (rhs.ndim() - 1) as isize,
        )
        .unwrap();
        let expected_lhs = crate::operations::linalg::matmul(&grad_output, &rhs_t).unwrap();
        assert!(
            grads
                .get(&input_ids[0])
                .unwrap()
                .allclose(&expected_lhs, 1e-6, 1e-6)
        );
        let lhs_t = crate::operations::linalg::transpose(
            &lhs,
            (lhs.ndim() - 2) as isize,
            (lhs.ndim() - 1) as isize,
        )
        .unwrap();
        let expected_rhs = crate::operations::linalg::matmul(&lhs_t, &grad_output).unwrap();
        assert!(
            grads
                .get(&input_ids[1])
                .unwrap()
                .allclose(&expected_rhs, 1e-6, 1e-6)
        );
    }

    #[test]
    fn test_matmul_backward_requires_grad_flags() {
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
        let ids = [TensorId::new(), TensorId::new()];
        let grad_fn = MatMulBackward {
            lhs: lhs.clone(),
            rhs: rhs.clone(),
            input_ids: ids,
            lhs_requires_grad: true,
            rhs_requires_grad: false,
        };
        let grad_output = Tensor::ones(
            Shape::new(vec![2, 2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let grads = grad_fn.backward(&grad_output).unwrap();
        assert!(grads.contains_key(&ids[0]));
        assert!(!grads.contains_key(&ids[1]));
    }

    #[test]
    fn test_transpose_backward_permutation() {
        let _input = Tensor::new(
            Arc::new(TensorData::from_vec_f32(
                (0..24).map(|x| x as f32).collect(),
                Device::cpu(),
            )),
            Shape::new(vec![2, 3, 4]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let dims = vec![1, 2, 0];
        let grad_fn = TransposeBackward {
            dims: dims.clone(),
            input_id: TensorId::new(),
        };
        let grad_output = Tensor::ones(
            Shape::new(vec![3, 4, 2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let grads = grad_fn.backward(&grad_output).unwrap();
        let grad_input = grads.values().next().unwrap();
        let mut inverse = vec![0; dims.len()];
        for (i, &d) in dims.iter().enumerate() {
            inverse[d] = i;
        }
        let mut expected = grad_output.clone();
        let mut current: Vec<usize> = (0..inverse.len()).collect();
        for i in 0..inverse.len() {
            let j = current.iter().position(|&x| x == inverse[i]).unwrap();
            if i != j {
                expected = crate::operations::linalg::transpose(&expected, i as isize, j as isize)
                    .unwrap();
                current.swap(i, j);
            }
        }
        assert!(grad_input.allclose(&expected, 1e-6, 1e-6));
    }
}

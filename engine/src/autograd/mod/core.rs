// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    device::Device,
    error::{MinitensorError, Result},
    operations::{activation, arithmetic, linalg, minmax, reduction, selection, shape_ops},
    tensor::{DataType, Shape, Strides, Tensor, TensorData},
};
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

/// Gradient function for solving linear systems.
pub struct SolveBackward {
    pub lhs: Tensor,
    pub solution: Tensor,
    pub input_ids: [TensorId; 2],
    pub lhs_requires_grad: bool,
    pub rhs_requires_grad: bool,
}

impl GradientFunction for SolveBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve((self.lhs_requires_grad as usize) + (self.rhs_requires_grad as usize));

        let lhs_t = crate::operations::linalg::transpose(
            &self.lhs,
            (self.lhs.ndim() - 2) as isize,
            (self.lhs.ndim() - 1) as isize,
        )?;

        if self.rhs_requires_grad {
            let grad_rhs = crate::operations::linalg::solve(&lhs_t, grad_output)?;
            gradients.insert(self.input_ids[1], grad_rhs);
        }

        if self.lhs_requires_grad {
            let solution_view = if self.solution.ndim() == self.lhs.ndim() - 1 {
                crate::operations::shape_ops::unsqueeze(
                    &self.solution,
                    self.solution.ndim() as isize,
                )?
            } else {
                self.solution.clone()
            };

            let grad_output_view = if grad_output.ndim() == self.lhs.ndim() - 1 {
                crate::operations::shape_ops::unsqueeze(grad_output, grad_output.ndim() as isize)?
            } else {
                grad_output.clone()
            };

            let solution_t = crate::operations::linalg::transpose(
                &solution_view,
                (solution_view.ndim() - 2) as isize,
                (solution_view.ndim() - 1) as isize,
            )?;
            let gram = crate::operations::linalg::matmul(&grad_output_view, &solution_t)?;
            let lhs_grad = crate::operations::linalg::solve(&lhs_t, &gram)?;
            let lhs_grad = crate::operations::arithmetic::neg(&lhs_grad)?;
            gradients.insert(self.input_ids[0], lhs_grad);
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

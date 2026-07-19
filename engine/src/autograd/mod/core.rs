// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    error::{MinitensorError, Result},
    operations::{arithmetic, linalg, minmax, reduction, selection},
    tensor::{DataType, Shape, Tensor, TensorData},
};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::cell::Cell;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

pub(crate) use crate::operations::map::PAR_THRESHOLD;

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

pub use super::graph::{BackwardStep, ComputationGraph, execute_backward_plan};

// Thread-local computation graph to avoid cross-test interference
thread_local! {
    static GLOBAL_GRAPH: std::cell::RefCell<ComputationGraph> =
        std::cell::RefCell::new(ComputationGraph::new());
}

thread_local! {
    static GRAPH_CONSUMED: Cell<bool> = const { Cell::new(false) };
}

// Thread-local gradient recording mode. While disabled, `add_to_graph` is a
// no-op, so tensor operations do not record autograd nodes. The backward pass
// disables recording while it executes gradient kernels; previously this was
// enforced only by the accident that the graph happened to be borrowed, which
// silently dropped registrations instead of making the policy explicit.
thread_local! {
    static GRAD_ENABLED: Cell<bool> = const { Cell::new(true) };
}

/// Query whether autograd recording is currently enabled on this thread.
pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED.with(|flag| flag.get())
}

/// Enable or disable autograd recording on this thread, returning the
/// previous state. Building block for user-facing `no_grad()` /
/// `enable_grad()` context managers.
pub fn set_grad_enabled(enabled: bool) -> bool {
    GRAD_ENABLED.with(|flag| flag.replace(enabled))
}

/// RAII guard that disables autograd recording for its lifetime.
///
/// Used internally by the backward pass; also usable as a building block for a
/// user-facing `no_grad` mode.
pub struct NoGradGuard {
    prev: bool,
}

impl NoGradGuard {
    pub fn new() -> Self {
        let prev = GRAD_ENABLED.with(|flag| flag.replace(false));
        Self { prev }
    }
}

impl Default for NoGradGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        let prev = self.prev;
        GRAD_ENABLED.with(|flag| flag.set(prev));
    }
}

/// Add a tensor and its gradient function to the global computation graph
pub fn add_to_graph(tensor: &Tensor, grad_fn: Option<Arc<dyn GradientFunction>>) -> Result<()> {
    if !is_grad_enabled() {
        return Ok(());
    }
    GLOBAL_GRAPH.with(|graph| {
        graph
            .borrow_mut()
            .add_tensor_with_grad_req(tensor.id(), grad_fn, tensor.requires_grad());
    });
    reset_graph_consumed();
    Ok(())
}

fn implicit_gradient(tensor: &Tensor, grad_output: Option<Tensor>) -> Result<Tensor> {
    match grad_output {
        Some(g) => Ok(g),
        None => {
            if tensor.numel() != 1 {
                return Err(MinitensorError::gradient_error(
                    "Gradient can only be implicitly created for scalar tensors",
                ));
            }
            Ok(Tensor::ones(
                tensor.shape().clone(),
                tensor.dtype(),
                tensor.device(),
                false,
            ))
        }
    }
}

/// Perform backward pass from the given tensor using the global computation
/// graph. Gradients are stored in the graph and can be read individually with
/// [`get_gradient`]; nothing is cloned on this path.
///
/// The graph is only borrowed to plan the pass and to store the results, so
/// gradient kernels run without holding the thread-local borrow.
pub fn backward(tensor: &Tensor, grad_output: Option<Tensor>) -> Result<()> {
    let grad = implicit_gradient(tensor, grad_output)?;

    let plan = GLOBAL_GRAPH.with(|graph| {
        let graph = graph.borrow();
        if !graph.contains_tensor(tensor.id()) && is_graph_consumed() {
            return Err(MinitensorError::gradient_error_with_suggestion(
                "Computation graph for this tensor has already been freed",
                "Re-run the forward pass or call backward(retain_graph=True)",
                None,
            ));
        }
        graph.plan_backward(tensor.id())
    })?;

    let gradients = {
        // Gradient kernels must not record new autograd nodes.
        let _guard = NoGradGuard::new();
        execute_backward_plan(&plan, tensor.id(), grad)?
    };

    GLOBAL_GRAPH.with(|graph| graph.borrow_mut().set_gradients(gradients));
    Ok(())
}

/// Perform a backward pass and return a snapshot of every gradient computed.
///
/// This clones the full gradient map and exists for tests and diagnostics;
/// production code should call [`backward`] and read the gradients it needs
/// via [`get_gradient`].
pub fn backward_collect(
    tensor: &Tensor,
    grad_output: Option<Tensor>,
) -> Result<FxHashMap<TensorId, Tensor>> {
    backward(tensor, grad_output)?;
    Ok(GLOBAL_GRAPH.with(|graph| graph.borrow().gradients_snapshot()))
}

/// Release the autograd nodes (and the tensors they saved for backward)
/// reachable from `tensor`. Stored gradients remain available. Called by the
/// bindings after a non-retaining backward pass so saved activations are freed
/// immediately rather than at the next optimizer step.
pub fn release_saved_subgraph(tensor: &Tensor) {
    GLOBAL_GRAPH.with(|graph| graph.borrow_mut().release_saved_subgraph(tensor.id()));
}

/// Get the gradient for a tensor from the last backward pass
pub fn get_gradient(tensor: &Tensor) -> Option<Tensor> {
    GLOBAL_GRAPH.with(|graph| graph.borrow().get_gradient(tensor.id()).cloned())
}

/// Clear all stored gradients in the global computation graph
pub fn zero_gradients() {
    GLOBAL_GRAPH.with(|graph| graph.borrow_mut().zero_grad());
}

/// Remove the stored gradient for a single tensor from the global graph.
pub fn clear_gradient(tensor: &Tensor) -> Option<Tensor> {
    GLOBAL_GRAPH.with(|graph| graph.borrow_mut().remove_gradient(tensor.id()))
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

/// Helper function to reduce gradients for broadcasting
pub(crate) fn reduce_gradient_for_broadcasting(
    grad_output: &Tensor,
    target_shape: &Shape,
) -> Result<Tensor> {
    if grad_output.shape() == target_shape {
        return Ok(grad_output.clone());
    }

    let grad_dims = grad_output.shape().dims();
    let target_dims = target_shape.dims();
    if target_dims.len() > grad_dims.len() {
        return Err(MinitensorError::BroadcastError {
            shape1: grad_dims.to_vec(),
            shape2: target_dims.to_vec(),
            suggestion: Some(
                "Ensure the target shape has no more dimensions than the gradient output."
                    .to_string(),
            ),
            context: Some("reduce_gradient_for_broadcasting".to_string()),
        });
    }
    let extra = grad_dims.len() - target_dims.len();

    // Use a stack-allocated small vector and pre-allocate enough capacity to
    // hold all potential broadcast axes. This avoids repeated reallocations for
    // higher dimensional tensors.
    let mut axes_to_sum: SmallVec<[usize; 8]> = SmallVec::with_capacity(grad_dims.len());
    axes_to_sum.extend(0..extra);
    for i in 0..target_dims.len() {
        let gdim = grad_dims[extra + i];
        let tdim = target_dims[i];
        if tdim == 1 {
            if gdim != 1 {
                axes_to_sum.push(extra + i);
            }
        } else if gdim != tdim {
            return Err(MinitensorError::BroadcastError {
                shape1: grad_dims.to_vec(),
                shape2: target_dims.to_vec(),
                suggestion: Some(
                    "Ensure each target dimension is 1 or matches the gradient dimension."
                        .to_string(),
                ),
                context: Some("reduce_gradient_for_broadcasting".to_string()),
            });
        }
    }

    if axes_to_sum.is_empty() {
        return Ok(grad_output.clone());
    }

    let mut axes = Vec::with_capacity(axes_to_sum.len());
    for axis in axes_to_sum {
        axes.push(axis as isize);
    }
    let mut grad = reduction::sum(grad_output, Some(axes), true)?;

    if grad.shape() != target_shape {
        grad = grad.view(target_shape.clone())?;
    }

    Ok(grad)
}

// Gradient function implementations for common operations

/// Accumulate a gradient contribution for `input_id` into `gradients`.
///
/// A single backward pass may produce more than one gradient for the same input
/// when a tensor is used as several operands of one operation (`x * x`, `x + x`,
/// `x.matmul(x)`, `pow(x, x)`, ...). The gradients returned by a
/// [`GradientFunction`] are keyed by [`TensorId`], so a plain `insert` would let
/// the later contribution silently overwrite the earlier one and halve (or worse)
/// the gradient. Summing on collision matches the mathematically correct result
/// and mirrors the cross-node accumulation performed by the graph itself.
#[inline]
pub(crate) fn accumulate_grad(
    gradients: &mut FxHashMap<TensorId, Tensor>,
    input_id: TensorId,
    grad: Tensor,
) -> Result<()> {
    use std::collections::hash_map::Entry;
    match gradients.entry(input_id) {
        Entry::Occupied(mut existing) => {
            arithmetic::add_inplace(existing.get_mut(), &grad)?;
        }
        Entry::Vacant(slot) => {
            slot.insert(grad);
        }
    }
    Ok(())
}

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
    /// Which inputs actually need a gradient; contributions for frozen
    /// inputs are skipped entirely (no broadcast reduction, no map entry).
    pub input_requires_grad: [bool; 2],
}

impl GradientFunction for AddBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(2);

        // For addition, gradients flow through unchanged, but broadcasting must
        // be undone. Inputs that do not require a gradient are skipped.
        if self.input_requires_grad[0] {
            let lhs_shape = Shape::new(self.input_shapes[0].clone());
            let lhs_grad = reduce_gradient_for_broadcasting(grad_output, &lhs_shape)?;
            accumulate_grad(&mut gradients, self.input_ids[0], lhs_grad)?;
        }
        if self.input_requires_grad[1] {
            let rhs_shape = Shape::new(self.input_shapes[1].clone());
            let rhs_grad = reduce_gradient_for_broadcasting(grad_output, &rhs_shape)?;
            accumulate_grad(&mut gradients, self.input_ids[1], rhs_grad)?;
        }

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
    /// Which inputs actually need a gradient (see [`AddBackward`]).
    pub input_requires_grad: [bool; 2],
}

impl GradientFunction for SubBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(2);

        if self.input_requires_grad[0] {
            let lhs_shape = Shape::new(self.input_shapes[0].clone());
            let lhs_grad = reduce_gradient_for_broadcasting(grad_output, &lhs_shape)?;
            accumulate_grad(&mut gradients, self.input_ids[0], lhs_grad)?;
        }
        if self.input_requires_grad[1] {
            let rhs_shape = Shape::new(self.input_shapes[1].clone());
            let rhs_base = reduce_gradient_for_broadcasting(grad_output, &rhs_shape)?;
            let rhs_grad = arithmetic::neg(&rhs_base)?;
            accumulate_grad(&mut gradients, self.input_ids[1], rhs_grad)?;
        }

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
    /// Which inputs actually need a gradient (see [`AddBackward`]).
    pub input_requires_grad: [bool; 2],
}

impl GradientFunction for MulBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(2);

        // d/dx(x*y) = y and d/dy(x*y) = x; skip frozen inputs entirely.
        if self.input_requires_grad[0] {
            let lhs_term = arithmetic::mul(grad_output, &self.rhs.detach())?;
            let lhs_grad = reduce_gradient_for_broadcasting(&lhs_term, self.lhs.shape())?;
            accumulate_grad(&mut gradients, self.input_ids[0], lhs_grad)?;
        }
        if self.input_requires_grad[1] {
            let rhs_term = arithmetic::mul(grad_output, &self.lhs.detach())?;
            let rhs_grad = reduce_gradient_for_broadcasting(&rhs_term, self.rhs.shape())?;
            accumulate_grad(&mut gradients, self.input_ids[1], rhs_grad)?;
        }

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
    /// Which inputs actually need a gradient (see [`AddBackward`]).
    pub input_requires_grad: [bool; 2],
}

impl GradientFunction for DivBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(2);

        let rhs = self.rhs.detach();

        // grad/y is needed by both branches; compute it once if either input
        // requires a gradient.
        let grad_over_rhs = arithmetic::div(grad_output, &rhs)?;

        // d/dx(x/y) = 1/y  =>  grad_x = grad / y
        if self.input_requires_grad[0] {
            let lhs_grad = reduce_gradient_for_broadcasting(&grad_over_rhs, self.lhs.shape())?;
            accumulate_grad(&mut gradients, self.input_ids[0], lhs_grad)?;
        }

        // d/dy(x/y) = -x/y^2  =>  grad_y = -(grad / y) * (x / y)
        if self.input_requires_grad[1] {
            let lhs_over_rhs = arithmetic::div(&self.lhs.detach(), &rhs)?;
            let rhs_term = arithmetic::mul(&grad_over_rhs, &lhs_over_rhs)?;
            let rhs_term = arithmetic::neg(&rhs_term)?;
            let rhs_grad = reduce_gradient_for_broadcasting(&rhs_term, self.rhs.shape())?;
            accumulate_grad(&mut gradients, self.input_ids[1], rhs_grad)?;
        }

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}

/// Gradient function for the Python-style remainder operation
pub struct RemainderBackward {
    pub lhs: Tensor,
    pub rhs: Tensor,
    pub input_ids: [TensorId; 2],
    /// Which inputs actually need a gradient (see [`AddBackward`]).
    pub input_requires_grad: [bool; 2],
}

impl GradientFunction for RemainderBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(2);

        // rem(x, y) = x - floor(x/y) * y with floor(x/y) locally constant, so
        // d/dx = 1 and d/dy = -floor(x/y).
        if self.input_requires_grad[0] {
            let lhs_grad = reduce_gradient_for_broadcasting(grad_output, self.lhs.shape())?;
            accumulate_grad(&mut gradients, self.input_ids[0], lhs_grad)?;
        }
        if self.input_requires_grad[1] {
            let quotient = arithmetic::floor_div(&self.lhs.detach(), &self.rhs.detach())?;
            let rhs_term = arithmetic::mul(grad_output, &quotient)?;
            let rhs_term = arithmetic::neg(&rhs_term)?;
            let rhs_grad = reduce_gradient_for_broadcasting(&rhs_term, self.rhs.shape())?;
            accumulate_grad(&mut gradients, self.input_ids[1], rhs_grad)?;
        }

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}

/// Gradient function for NumPy-style boolean indexing (`masked_index`)
pub struct MaskedIndexBackward {
    /// The (contiguous, detached) boolean mask that produced the selection.
    pub mask: Tensor,
    pub input_shape: crate::tensor::Shape,
    /// Elements per selected block (product of the trailing input dims).
    pub inner: usize,
    pub input_id: TensorId,
}

impl GradientFunction for MaskedIndexBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();

        // Scatter the selected blocks back to their positions; unselected
        // positions receive zero.
        let mut grad_input = Tensor::zeros(
            self.input_shape.clone(),
            grad_output.dtype(),
            grad_output.device(),
            false,
        );
        let mask_slice = self.mask.data().as_bool_slice().ok_or_else(|| {
            crate::error::MinitensorError::internal_error("Failed to get bool slice from mask")
        })?;
        let grad_c = grad_output.contiguous()?;

        macro_rules! scatter_arm {
            ($accessor:ident, $accessor_mut:ident, $tyname:literal) => {{
                let src = grad_c.data().$accessor().ok_or_else(|| {
                    crate::error::MinitensorError::internal_error(concat!(
                        "Failed to get ",
                        $tyname,
                        " slice from gradient"
                    ))
                })?;
                let dst = grad_input.data_mut().$accessor_mut().ok_or_else(|| {
                    crate::error::MinitensorError::internal_error(concat!(
                        "Failed to get mutable ",
                        $tyname,
                        " slice for input gradient"
                    ))
                })?;
                let mut k = 0usize;
                for (blk, &selected) in mask_slice.iter().enumerate() {
                    if selected {
                        dst[blk * self.inner..(blk + 1) * self.inner]
                            .copy_from_slice(&src[k * self.inner..(k + 1) * self.inner]);
                        k += 1;
                    }
                }
            }};
        }

        if self.inner > 0 {
            match grad_output.dtype() {
                crate::tensor::DataType::Float32 => {
                    scatter_arm!(as_f32_slice, as_f32_slice_mut, "f32")
                }
                crate::tensor::DataType::Float64 => {
                    scatter_arm!(as_f64_slice, as_f64_slice_mut, "f64")
                }
                dtype => {
                    return Err(crate::error::MinitensorError::invalid_operation(format!(
                        "masked_index backward not supported for {dtype:?} gradients"
                    )));
                }
            }
        }

        accumulate_grad(&mut gradients, self.input_id, grad_input)?;
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
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
            accumulate_grad(&mut gradients, self.input_ids[0], reduced)?;
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
            accumulate_grad(&mut gradients, self.input_ids[1], reduced)?;
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

            let grad_data = linalg::apply_triangular_mask(grad_output, self.diagonal, self.upper)?;
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
            accumulate_grad(&mut gradients, self.input_ids[0], reduced)?;
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
            accumulate_grad(&mut gradients, self.input_ids[1], reduced)?;
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
            accumulate_grad(&mut gradients, self.input_ids[0], reduced)?;
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
            accumulate_grad(&mut gradients, self.input_ids[1], reduced)?;
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
            accumulate_grad(&mut gradients, self.input_ids[0], grad)?;
        }

        if self.rhs_requires_grad {
            let grad = crate::operations::arithmetic::mul(&self.lhs, grad_output)?;
            accumulate_grad(&mut gradients, self.input_ids[1], grad)?;
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
            accumulate_grad(&mut gradients, self.input_ids[0], lhs_grad)?;
        }

        if self.rhs_requires_grad {
            let lhs_t = crate::operations::linalg::transpose(
                &self.lhs,
                (self.lhs.ndim() - 2) as isize,
                (self.lhs.ndim() - 1) as isize,
            )?;
            let rhs_grad = crate::operations::linalg::matmul(&lhs_t, grad_output)?;
            accumulate_grad(&mut gradients, self.input_ids[1], rhs_grad)?;
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
            accumulate_grad(&mut gradients, self.input_ids[1], grad_rhs)?;
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
            accumulate_grad(&mut gradients, self.input_ids[0], lhs_grad)?;
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

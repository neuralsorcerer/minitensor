// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! The core `Tensor` type: its fields, constructors, and the method impls
//! grouped by concern (core metadata and autograd storage, math ops, indexing,
//! dtype casts, and formatting). Kept in one module so every method retains
//! access to `Tensor`'s private fields; everything public is re-exported up to
//! `crate::tensor`.

pub use crate::tensor::dtype::DataType;
pub use crate::tensor::shape::{Shape, Strides};
pub use crate::tensor::storage::{DataMut, TensorData};

use crate::ops::map::{BINARY_PAR_THRESHOLD as CAST_PAR_THRESHOLD, unary_map_threshold};
use crate::{
    autograd::{self, CloneBackward, GradientFunction, TensorId},
    device::Device,
    error::{MinitensorError, Result},
    ops::{arithmetic::add, reduction::QuantileInterpolation},
};
use rayon::prelude::*;
use std::{borrow::Cow, sync::Arc};

// ===== core: struct definition, constructors, autograd storage =====
/// Core tensor structure for minitensor
#[derive(Clone)]
pub struct Tensor {
    /// Tensor data storage
    data: Arc<TensorData>,
    /// Tensor shape (dimensions)
    shape: Shape,
    /// Memory strides for each dimension
    strides: Strides,
    /// Data type of tensor elements
    dtype: DataType,
    /// Device where tensor is stored
    device: Device,
    /// Whether this tensor requires gradient computation
    requires_grad: bool,
    /// Gradient function for automatic differentiation
    grad_fn: Option<Arc<dyn GradientFunction>>,
    /// Stored gradient for this tensor
    grad: Option<Arc<Tensor>>,
    /// Unique identifier for this tensor
    tensor_id: TensorId,
}

/// Index specification for tensor slicing and indexing
#[derive(Clone, Copy, Debug)]
pub enum TensorIndex {
    /// Select a single index along the dimension
    Index(usize),
    /// Select a range with optional step (step defaults to 1)
    Slice {
        start: usize,
        end: usize,
        step: usize,
    },
}

impl Tensor {
    /// Create a new tensor with the given data, shape, and properties
    #[inline(always)]
    pub fn new(
        data: Arc<TensorData>,
        shape: Shape,
        dtype: DataType,
        device: Device,
        requires_grad: bool,
    ) -> Self {
        let strides = Strides::from_shape(&shape);
        Self {
            data,
            shape,
            strides,
            dtype,
            device,
            // While grad recording is disabled (`no_grad`), newly created
            // tensors never require gradients. Callers that genuinely need a
            // trainable leaf inside a no-grad scope can opt back in with
            // `requires_grad_(true)`, which expresses explicit intent and is
            // not gated.
            requires_grad: requires_grad && autograd::is_grad_enabled(),
            grad_fn: None,
            grad: None,
            tensor_id: TensorId::new(),
        }
    }

    /// Create a tensor with uninitialized data
    #[inline(always)]
    pub fn empty(shape: Shape, dtype: DataType, device: Device, requires_grad: bool) -> Self {
        let data = Arc::new(TensorData::uninitialized_on_device(
            shape.numel(),
            dtype,
            device,
        ));
        Self::new(data, shape, dtype, device, requires_grad)
    }

    /// Create a tensor filled with zeros
    #[inline(always)]
    pub fn zeros(shape: Shape, dtype: DataType, device: Device, requires_grad: bool) -> Self {
        let data = Arc::new(TensorData::zeros_on_device(shape.numel(), dtype, device));
        Self::new(data, shape, dtype, device, requires_grad)
    }

    /// Create a tensor filled with ones
    #[inline(always)]
    pub fn ones(shape: Shape, dtype: DataType, device: Device, requires_grad: bool) -> Self {
        let data = Arc::new(TensorData::ones_on_device(shape.numel(), dtype, device));
        Self::new(data, shape, dtype, device, requires_grad)
    }

    /// Get the tensor's shape
    #[inline(always)]
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Get the tensor's strides
    #[inline(always)]
    pub fn strides(&self) -> &Strides {
        &self.strides
    }

    /// Get the tensor's data type
    #[inline(always)]
    pub fn dtype(&self) -> DataType {
        self.dtype
    }

    /// Get the tensor's device
    #[inline(always)]
    pub fn device(&self) -> Device {
        self.device
    }

    /// Check if this tensor requires gradients
    #[inline(always)]
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Get the tensor's unique ID
    #[inline(always)]
    pub fn id(&self) -> TensorId {
        self.tensor_id
    }

    /// Get the number of dimensions
    #[inline(always)]
    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }

    /// Get the total number of elements
    #[inline(always)]
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// Get the size of a specific dimension
    #[inline(always)]
    pub fn size(&self, dim: usize) -> Result<usize> {
        self.shape.size(dim)
    }

    /// Check if the tensor is contiguous in memory
    #[inline(always)]
    pub fn is_contiguous(&self) -> bool {
        self.strides.is_contiguous(&self.shape)
    }

    /// Get a reference to the tensor data
    #[inline(always)]
    pub fn data(&self) -> &Arc<TensorData> {
        &self.data
    }

    /// Get mutable access to the tensor data.
    ///
    /// Non-leaf tensors (and tensors that do not require gradients) get
    /// copy-on-write semantics: if the storage is shared, it is cloned first
    /// so in-place mutation cannot corrupt other tensors or saved autograd
    /// state.
    ///
    /// Leaf tensors that require gradients (i.e. parameters) are mutated in
    /// place even when the storage is shared, so that optimizer updates stay
    /// visible through every handle to the parameter — mirroring PyTorch's
    /// in-place parameter update semantics. That path goes through the
    /// storage layer's interior mutability (see [`DataMut`]) instead of
    /// fabricating a `&mut TensorData` from the shared `Arc`.
    #[inline(always)]
    pub(crate) fn data_mut(&mut self) -> DataMut<'_> {
        let needs_detach = self.grad_fn.is_some() || !self.requires_grad;
        if needs_detach {
            if Arc::get_mut(&mut self.data).is_none() {
                let cloned = self.data.as_ref().clone_data();
                self.data = Arc::new(cloned);
            }
            return DataMut::Unique(
                Arc::get_mut(&mut self.data).expect("Tensor data should be uniquely owned"),
            );
        }
        // Take the exclusive path whenever the storage is uniquely owned.
        if Arc::get_mut(&mut self.data).is_some() {
            return DataMut::Unique(Arc::get_mut(&mut self.data).expect("uniqueness just checked"));
        }
        DataMut::Shared(self.data.as_ref())
    }

    /// Create a deep copy of the tensor data while preserving autograd history.
    #[inline]
    pub fn deep_clone(&self) -> Result<Self> {
        let data = Arc::new(self.data.as_ref().clone_data());
        let mut cloned = Tensor::new(
            data,
            self.shape.clone(),
            self.dtype,
            self.device,
            self.requires_grad,
        );

        if self.requires_grad {
            let grad_fn = Arc::new(CloneBackward {
                input_id: self.tensor_id,
            });
            cloned.set_grad_fn(Some(grad_fn.clone()));
            autograd::add_to_graph(&cloned, Some(grad_fn))?;
        }

        Ok(cloned)
    }

    /// Materialise the tensor into a contiguous layout.
    pub fn contiguous(&self) -> Result<Self> {
        if self.is_contiguous() && self.data.is_contiguous() {
            return Ok(self.clone());
        }

        if !self.device.is_cpu() {
            return Err(MinitensorError::invalid_operation(
                "contiguous currently supports only CPU tensors".to_string(),
            ));
        }

        let dtype = self.dtype;
        let device = self.device;
        let requires_grad = self.requires_grad;
        let shape = self.shape.dims().to_vec();
        let strides = self.strides.as_slice().to_vec();

        /// One dtype arm: gather the strided view into a fresh contiguous
        /// buffer (parallel above the map threshold, no zeroing pass).
        macro_rules! gather_arm {
            ($accessor:ident, $ty:ty, $label:literal) => {{
                let src = self.data.$accessor().ok_or_else(|| {
                    MinitensorError::invalid_operation(concat!(
                        "failed to access ",
                        $label,
                        " data for contiguous copy"
                    ))
                })?;
                TensorData::from_vec::<$ty>(
                    crate::ops::map::strided_gather(src, &shape, &strides),
                    dtype,
                    device,
                )
            }};
        }

        let output_data = match dtype {
            DataType::Float32 => gather_arm!(as_f32_slice, f32, "float32"),
            DataType::Float64 => gather_arm!(as_f64_slice, f64, "float64"),
            DataType::Int32 => gather_arm!(as_i32_slice, i32, "int32"),
            DataType::Int64 => gather_arm!(as_i64_slice, i64, "int64"),
            DataType::Bool => gather_arm!(as_bool_slice, bool, "bool"),
        };

        let mut output = Tensor::new(
            Arc::new(output_data),
            Shape::new(shape),
            dtype,
            device,
            requires_grad,
        );

        // The materialized copy passes gradients straight through to the
        // source tensor (identity backward).
        if requires_grad {
            let grad_fn = Arc::new(CloneBackward {
                input_id: self.tensor_id,
            });
            output.set_grad_fn(Some(grad_fn.clone()));
            autograd::add_to_graph(&output, Some(grad_fn))?;
        }

        Ok(output)
    }
    /// Set the gradient function for this tensor.
    ///
    /// While grad recording is disabled (`no_grad` scopes and the backward
    /// executor), attaching a gradient function is a no-op so operation
    /// outputs stay leaves; this mirrors the gating in
    /// [`autograd::add_to_graph`], keeping the tensor's metadata and the
    /// graph consistent. Clearing (`None`) is always honoured.
    #[inline(always)]
    pub fn set_grad_fn(&mut self, grad_fn: Option<Arc<dyn GradientFunction>>) {
        if grad_fn.is_some() && !autograd::is_grad_enabled() {
            return;
        }
        self.grad_fn = grad_fn;
    }

    /// Get the gradient function for this tensor
    #[inline(always)]
    pub fn grad_fn(&self) -> Option<&Arc<dyn GradientFunction>> {
        self.grad_fn.as_ref()
    }

    /// Enable gradient computation for this tensor
    #[inline(always)]
    pub fn requires_grad_(mut self, requires_grad: bool) -> Self {
        self.requires_grad = requires_grad;
        self
    }

    /// Assign a fresh tensor identifier and clear autograd metadata.
    #[inline(always)]
    pub(crate) fn refresh_autograd_metadata(&mut self) {
        self.tensor_id = TensorId::new();
        self.grad_fn = None;
        self.grad = None;
    }

    /// Get the gradient for this tensor
    #[inline(always)]
    pub fn grad(&self) -> Option<&Arc<Tensor>> {
        self.grad.as_ref()
    }

    /// Get mutable access to the gradient if uniquely owned
    #[inline(always)]
    pub fn grad_mut(&mut self) -> Option<&mut Tensor> {
        self.grad.as_mut().and_then(|g| std::sync::Arc::get_mut(g))
    }

    /// Set the gradient for this tensor
    #[inline(always)]
    pub fn set_grad(&mut self, grad: Option<Tensor>) {
        self.grad = grad.map(Arc::new);
    }

    /// Accumulate gradient for this tensor
    #[inline]
    pub fn accumulate_grad(&mut self, grad: Tensor) -> Result<()> {
        match &self.grad {
            Some(existing) => {
                let sum = add(existing.as_ref(), &grad)?;
                self.grad = Some(Arc::new(sum));
            }
            None => {
                self.grad = Some(Arc::new(grad));
            }
        }
        Ok(())
    }

    /// Clear the gradient for this tensor.
    ///
    /// Only this tensor's gradient is affected — both the copy stored on the
    /// tensor and its entry in the global gradient map. (Earlier versions
    /// wiped every gradient on the thread, so zeroing one tensor silently
    /// cleared unrelated models' gradients.)
    #[inline(always)]
    pub fn zero_grad(&mut self, set_to_none: bool) {
        autograd::clear_gradient(self);
        if set_to_none {
            self.grad = None;
            return;
        }

        // If gradient exists, zero it in place
        if let Some(g) = self.grad_mut() {
            match g.dtype() {
                DataType::Float32 => {
                    if let Some(slice) = g.data_mut().as_f32_slice_mut() {
                        slice.fill(0.0);
                    }
                }
                DataType::Float64 => {
                    if let Some(slice) = g.data_mut().as_f64_slice_mut() {
                        slice.fill(0.0);
                    }
                }
                DataType::Int32 => {
                    if let Some(slice) = g.data_mut().as_i32_slice_mut() {
                        slice.fill(0);
                    }
                }
                DataType::Int64 => {
                    if let Some(slice) = g.data_mut().as_i64_slice_mut() {
                        slice.fill(0);
                    }
                }
                DataType::Bool => {
                    if let Some(slice) = g.data_mut().as_bool_slice_mut() {
                        slice.fill(false);
                    }
                }
            }
        } else if self.requires_grad {
            // If gradient doesn't exist but is required, create a zero tensor
            let zero = Tensor::zeros(self.shape.clone(), self.dtype, self.device, false);
            self.grad = Some(Arc::new(zero));
        } else {
            self.grad = None;
        }
    }

    /// Check if this tensor has a gradient
    #[inline(always)]
    pub fn has_grad(&self) -> bool {
        self.grad.is_some()
    }

    /// Perform backward pass from this tensor
    pub fn backward(&self, gradient: Option<Tensor>) -> Result<()> {
        use crate::autograd;

        // If no gradient is provided, create a gradient of ones for scalar tensors
        let grad = match gradient {
            Some(g) => g,
            None => {
                if self.numel() != 1 {
                    return Err(MinitensorError::gradient_error(
                        "Gradient can only be implicitly created for scalar tensors",
                    ));
                }
                // Create a tensor of ones with the same shape as self
                Self::ones(self.shape.clone(), self.dtype, self.device, false)
            }
        };

        // Perform backward pass through the computation graph
        autograd::backward(self, Some(grad))
    }
}

impl Tensor {
    /// Create a view of this tensor with a new shape.
    ///
    /// The tensor must be contiguous: a view only reinterprets the existing
    /// buffer, and re-striding a non-contiguous tensor (e.g. the result of
    /// `expand`) would silently associate the new shape with storage that does
    /// not contain the tensor's logical elements. Use [`Self::reshape`] to get
    /// an automatic copy in that case.
    #[inline(always)]
    pub fn view(&self, new_shape: Shape) -> Result<Self> {
        if new_shape.numel() != self.numel() {
            return Err(MinitensorError::shape_mismatch(
                vec![self.numel()],
                vec![new_shape.numel()],
            ));
        }

        if !self.is_contiguous() {
            return Err(MinitensorError::invalid_operation(
                "view is not supported for non-contiguous tensors; call contiguous() or reshape() instead",
            ));
        }

        let mut tensor = self.clone();
        tensor.strides = Strides::from_shape(&new_shape);
        tensor.shape = new_shape;
        Ok(tensor)
    }

    /// Reshape the tensor to a new shape.
    ///
    /// Returns a zero-copy view when the tensor is contiguous and materialises
    /// a contiguous copy otherwise. Autograd-aware: a `ReshapeBackward` node is
    /// recorded when the tensor requires grad, so gradients flow back to the
    /// original shape. (A bare [`Self::view`] would keep the input's identity
    /// and grad function, silently routing the caller's gradient into the
    /// *pre-reshape* node — which is how the multi-dim `cross_entropy` wrapper,
    /// `flatten_all`, `squeeze`, and `unsqueeze` used to drop/misroute the
    /// backward pass. The backward pass itself runs under `NoGradGuard`, so this
    /// records nothing when called from inside a gradient kernel.)
    #[inline(always)]
    pub fn reshape(&self, new_shape: Shape) -> Result<Self> {
        crate::ops::shape_ops::reshape(self, new_shape)
    }

    /// Flatten the tensor into a one-dimensional view.
    /// This operation avoids data copies when possible.
    #[inline(always)]
    pub fn flatten_all(&self) -> Result<Self> {
        let len = self.numel();
        self.reshape(Shape::new(vec![len]))
    }

    /// Alias for [`flatten_all`](Self::flatten_all) for backward compatibility.
    #[inline(always)]
    pub fn ravel(&self) -> Result<Self> {
        self.flatten_all()
    }

    /// Transpose two dimensions of the tensor
    #[inline(always)]
    pub fn transpose(&self, dim0: isize, dim1: isize) -> Result<Self> {
        use crate::ops::linalg::transpose;
        transpose(self, dim0, dim1)
    }

    /// Permute tensor dimensions
    #[inline(always)]
    pub fn permute(&self, dims: Vec<isize>) -> Result<Self> {
        use crate::ops::shape_ops::permute;
        permute(self, dims)
    }
}

// ===== math operation methods =====
impl Tensor {
    /// Unary negation
    #[inline(always)]
    pub fn neg(&self) -> Result<Self> {
        use crate::ops::arithmetic::neg;
        neg(self)
    }

    /// Add two tensors element-wise
    #[inline(always)]
    pub fn add(&self, other: &Tensor) -> Result<Self> {
        use crate::ops::arithmetic::add;
        add(self, other)
    }

    /// Element-wise maximum
    #[inline(always)]
    pub fn maximum(&self, other: &Tensor) -> Result<Self> {
        use crate::ops::minmax::maximum;
        maximum(self, other)
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn minimum(&self, other: &Tensor) -> Result<Self> {
        use crate::ops::minmax::minimum;
        minimum(self, other)
    }

    /// Select elements from self or other based on a boolean condition tensor
    #[inline(always)]
    pub fn where_select(&self, condition: &Tensor, other: &Tensor) -> Result<Self> {
        use crate::ops::selection::where_op;
        where_op(condition, self, other)
    }

    /// Fill elements specified by `mask` with values from `value`.
    #[inline(always)]
    pub fn masked_fill(&self, mask: &Tensor, value: &Tensor) -> Result<Self> {
        crate::ops::selection::masked_fill(self, mask, value)
    }

    /// Fill elements specified by `mask` with a scalar.
    #[inline(always)]
    pub fn masked_fill_scalar(&self, mask: &Tensor, value: f64) -> Result<Self> {
        crate::ops::selection::masked_fill_scalar(self, mask, value)
    }

    /// Dot product between two 1D tensors
    #[inline(always)]
    pub fn dot(&self, other: &Tensor) -> Result<Self> {
        crate::ops::linalg::dot(self, other)
    }

    /// Matrix multiplication
    #[inline(always)]
    pub fn matmul(&self, other: &Tensor) -> Result<Self> {
        use crate::ops::linalg::matmul;
        matmul(self, other)
    }

    /// Batched matrix multiplication specialised for 3D tensors
    #[inline(always)]
    pub fn bmm(&self, other: &Tensor) -> Result<Self> {
        use crate::ops::linalg::bmm;
        bmm(self, other)
    }

    /// Upper triangular part of the tensor's last two dimensions
    #[inline(always)]
    pub fn triu(&self, diagonal: i64) -> Result<Self> {
        use crate::ops::linalg::triu;
        triu(self, diagonal)
    }

    /// Lower triangular part of the tensor's last two dimensions
    #[inline(always)]
    pub fn tril(&self, diagonal: i64) -> Result<Self> {
        use crate::ops::linalg::tril;
        tril(self, diagonal)
    }

    /// Extract a diagonal along two dimensions.
    #[inline(always)]
    pub fn diagonal(&self, offset: isize, dim1: isize, dim2: isize) -> Result<Self> {
        use crate::ops::linalg::diagonal;
        diagonal(self, offset, dim1, dim2)
    }

    /// Sum the diagonal elements along two dimensions.
    #[inline(always)]
    pub fn trace(&self, offset: isize, dim1: isize, dim2: isize) -> Result<Self> {
        use crate::ops::linalg::trace;
        trace(self, offset, dim1, dim2)
    }

    /// Sum reduction
    #[inline(always)]
    pub fn sum(&self, dim: Option<Vec<isize>>, keepdim: bool) -> Result<Self> {
        use crate::ops::reduction::sum;
        sum(self, dim, keepdim)
    }

    /// NaN-aware sum reduction
    #[inline(always)]
    pub fn nansum(&self, dim: Option<Vec<isize>>, keepdim: bool) -> Result<Self> {
        use crate::ops::reduction::nansum;
        nansum(self, dim, keepdim)
    }

    /// Log-sum-exp reduction
    #[inline(always)]
    pub fn logsumexp(&self, dim: Option<Vec<isize>>, keepdim: bool) -> Result<Self> {
        use crate::ops::reduction::logsumexp;
        logsumexp(self, dim, keepdim)
    }

    /// Product reduction
    #[inline(always)]
    pub fn prod(&self, dim: Option<Vec<isize>>, keepdim: bool) -> Result<Self> {
        use crate::ops::reduction::prod;
        prod(self, dim, keepdim)
    }

    /// Mean reduction
    #[inline(always)]
    pub fn mean(&self, dim: Option<Vec<isize>>, keepdim: bool) -> Result<Self> {
        use crate::ops::reduction::mean;
        mean(self, dim, keepdim)
    }

    /// NaN-aware mean reduction
    #[inline(always)]
    pub fn nanmean(&self, dim: Option<Vec<isize>>, keepdim: bool) -> Result<Self> {
        use crate::ops::reduction::nanmean;
        nanmean(self, dim, keepdim)
    }

    /// Logical all reduction
    #[inline(always)]
    pub fn all(&self, dim: Option<isize>, keepdim: bool) -> Result<Self> {
        use crate::ops::reduction::all;
        all(self, dim, keepdim)
    }

    /// Logical any reduction
    #[inline(always)]
    pub fn any(&self, dim: Option<isize>, keepdim: bool) -> Result<Self> {
        use crate::ops::reduction::any;
        any(self, dim, keepdim)
    }

    /// Cumulative sum along a dimension
    #[inline(always)]
    pub fn cumsum(&self, dim: isize) -> Result<Self> {
        use crate::ops::reduction::cumsum;
        cumsum(self, dim)
    }

    /// Cumulative product along a dimension
    #[inline(always)]
    pub fn cumprod(&self, dim: isize) -> Result<Self> {
        use crate::ops::reduction::cumprod;
        cumprod(self, dim)
    }

    /// Maximum value
    #[inline(always)]
    pub fn max(&self, dim: Option<isize>, keepdim: bool) -> Result<Self> {
        use crate::ops::reduction::max;
        max(self, dim, keepdim)
    }

    /// NaN-aware maximum value
    #[inline(always)]
    pub fn nanmax(&self, dim: Option<isize>, keepdim: bool) -> Result<Self> {
        use crate::ops::reduction::nanmax;
        nanmax(self, dim, keepdim)
    }

    /// Minimum value
    #[inline(always)]
    pub fn min(&self, dim: Option<isize>, keepdim: bool) -> Result<Self> {
        use crate::ops::reduction::min;
        min(self, dim, keepdim)
    }

    /// NaN-aware minimum value
    #[inline(always)]
    pub fn nanmin(&self, dim: Option<isize>, keepdim: bool) -> Result<Self> {
        use crate::ops::reduction::nanmin;
        nanmin(self, dim, keepdim)
    }

    /// Argument of maximum value
    #[inline(always)]
    pub fn argmax(&self, dim: Option<isize>, keepdim: bool) -> Result<Self> {
        use crate::ops::reduction::argmax;
        argmax(self, dim, keepdim)
    }

    /// Argument of minimum value
    #[inline(always)]
    pub fn argmin(&self, dim: Option<isize>, keepdim: bool) -> Result<Self> {
        use crate::ops::reduction::argmin;
        argmin(self, dim, keepdim)
    }

    /// Maximum values and their indices along a dimension
    #[inline(always)]
    pub fn max_with_indices(&self, dim: isize, keepdim: bool) -> Result<(Self, Self)> {
        use crate::ops::reduction::max_with_indices;
        max_with_indices(self, dim, keepdim)
    }

    /// NaN-aware maximum values and their indices along a dimension
    #[inline(always)]
    pub fn nanmax_with_indices(&self, dim: isize, keepdim: bool) -> Result<(Self, Self)> {
        use crate::ops::reduction::nanmax_with_indices;
        nanmax_with_indices(self, dim, keepdim)
    }

    /// Minimum values and their indices along a dimension
    #[inline(always)]
    pub fn min_with_indices(&self, dim: isize, keepdim: bool) -> Result<(Self, Self)> {
        use crate::ops::reduction::min_with_indices;
        min_with_indices(self, dim, keepdim)
    }

    /// NaN-aware minimum values and their indices along a dimension
    #[inline(always)]
    pub fn nanmin_with_indices(&self, dim: isize, keepdim: bool) -> Result<(Self, Self)> {
        use crate::ops::reduction::nanmin_with_indices;
        nanmin_with_indices(self, dim, keepdim)
    }

    /// Median value (optionally along a dimension)
    #[inline(always)]
    pub fn median(&self, dim: Option<isize>, keepdim: bool) -> Result<(Self, Option<Self>)> {
        use crate::ops::reduction::median;
        median(self, dim, keepdim)
    }

    /// Quantile reduction with configurable interpolation
    #[inline(always)]
    pub fn quantile(
        &self,
        q: f64,
        dim: Option<isize>,
        keepdim: bool,
        interpolation: QuantileInterpolation,
    ) -> Result<Self> {
        use crate::ops::reduction::quantile;
        quantile(self, q, dim, keepdim, interpolation)
    }

    /// Quantile reduction that ignores NaN values
    #[inline(always)]
    pub fn nanquantile(
        &self,
        q: f64,
        dim: Option<isize>,
        keepdim: bool,
        interpolation: QuantileInterpolation,
    ) -> Result<Self> {
        use crate::ops::reduction::nanquantile;
        nanquantile(self, q, dim, keepdim, interpolation)
    }

    /// Median reduction that ignores NaN values
    #[inline(always)]
    pub fn nanmedian(&self, dim: Option<isize>, keepdim: bool) -> Result<Self> {
        use crate::ops::reduction::nanmedian;
        nanmedian(self, dim, keepdim)
    }

    /// Batched quantile reduction for multiple probabilities at once
    #[inline(always)]
    pub fn quantiles(
        &self,
        qs: &[f64],
        dim: Option<isize>,
        keepdim: bool,
        interpolation: QuantileInterpolation,
    ) -> Result<Self> {
        use crate::ops::reduction::quantiles;
        quantiles(self, qs, dim, keepdim, interpolation)
    }

    /// Batched quantile reduction that ignores NaN values
    #[inline(always)]
    pub fn nanquantiles(
        &self,
        qs: &[f64],
        dim: Option<isize>,
        keepdim: bool,
        interpolation: QuantileInterpolation,
    ) -> Result<Self> {
        use crate::ops::reduction::nanquantiles;
        nanquantiles(self, qs, dim, keepdim, interpolation)
    }

    /// Top-k values and indices along a dimension
    #[inline(always)]
    pub fn topk(
        &self,
        k: usize,
        dim: Option<isize>,
        largest: bool,
        sorted: bool,
    ) -> Result<(Self, Self)> {
        use crate::ops::reduction::topk;
        topk(self, k, dim, largest, sorted)
    }

    /// Sort tensor values along a dimension
    #[inline(always)]
    pub fn sort(&self, dim: Option<isize>, descending: bool, stable: bool) -> Result<(Self, Self)> {
        use crate::ops::reduction::sort;
        sort(self, dim, descending, stable)
    }

    /// Indices that would sort the tensor along a dimension
    #[inline(always)]
    pub fn argsort(&self, dim: Option<isize>, descending: bool, stable: bool) -> Result<Self> {
        use crate::ops::reduction::argsort;
        argsort(self, dim, descending, stable)
    }

    /// Element-wise equality comparison
    #[inline(always)]
    pub fn eq(&self, other: &Tensor) -> Result<Self> {
        use crate::ops::comparison::eq;
        eq(self, other)
    }

    /// Element-wise inequality comparison
    pub fn ne(&self, other: &Tensor) -> Result<Self> {
        use crate::ops::comparison::ne;
        ne(self, other)
    }

    /// Element-wise less-than comparison
    #[inline(always)]
    pub fn lt(&self, other: &Tensor) -> Result<Self> {
        use crate::ops::comparison::lt;
        lt(self, other)
    }

    /// Element-wise less-than-or-equal comparison
    #[inline(always)]
    pub fn le(&self, other: &Tensor) -> Result<Self> {
        use crate::ops::comparison::le;
        le(self, other)
    }

    /// Element-wise greater-than comparison
    #[inline(always)]
    pub fn gt(&self, other: &Tensor) -> Result<Self> {
        use crate::ops::comparison::gt;
        gt(self, other)
    }

    /// Element-wise greater-than-or-equal comparison
    #[inline(always)]
    pub fn ge(&self, other: &Tensor) -> Result<Self> {
        use crate::ops::comparison::ge;
        ge(self, other)
    }

    /// Standard deviation
    #[inline(always)]
    pub fn std(&self, dim: Option<Vec<isize>>, keepdim: bool, unbiased: bool) -> Result<Self> {
        use crate::ops::reduction::std;
        std(self, dim, keepdim, unbiased)
    }

    /// Variance
    #[inline(always)]
    pub fn var(&self, dim: Option<Vec<isize>>, keepdim: bool, unbiased: bool) -> Result<Self> {
        use crate::ops::reduction::var;
        var(self, dim, keepdim, unbiased)
    }

    /// Exponential function
    #[inline(always)]
    pub fn exp(&self) -> Result<Self> {
        use crate::ops::activation::exp;
        exp(self)
    }

    /// Natural logarithm
    #[inline(always)]
    pub fn log(&self) -> Result<Self> {
        use crate::ops::activation::log;
        log(self)
    }

    /// log1p (log(1 + x))
    #[inline(always)]
    pub fn log1p(&self) -> Result<Self> {
        use crate::ops::activation::log1p;
        log1p(self)
    }

    /// expm1 (exp(x) - 1)
    #[inline(always)]
    pub fn expm1(&self) -> Result<Self> {
        use crate::ops::activation::expm1;
        expm1(self)
    }

    /// Sine function
    #[inline(always)]
    pub fn sin(&self) -> Result<Self> {
        use crate::ops::activation::sin;
        sin(self)
    }

    /// Cosine function
    #[inline(always)]
    pub fn cos(&self) -> Result<Self> {
        use crate::ops::activation::cos;
        cos(self)
    }

    /// Tangent function
    #[inline(always)]
    pub fn tan(&self) -> Result<Self> {
        use crate::ops::activation::tan;
        tan(self)
    }

    /// Inverse sine function
    #[inline(always)]
    pub fn asin(&self) -> Result<Self> {
        use crate::ops::activation::asin;
        asin(self)
    }

    /// Inverse cosine function
    #[inline(always)]
    pub fn acos(&self) -> Result<Self> {
        use crate::ops::activation::acos;
        acos(self)
    }

    /// Inverse tangent function
    #[inline(always)]
    pub fn atan(&self) -> Result<Self> {
        use crate::ops::activation::atan;
        atan(self)
    }

    /// Hyperbolic sine
    #[inline(always)]
    pub fn sinh(&self) -> Result<Self> {
        use crate::ops::activation::sinh;
        sinh(self)
    }

    /// Hyperbolic cosine
    #[inline(always)]
    pub fn cosh(&self) -> Result<Self> {
        use crate::ops::activation::cosh;
        cosh(self)
    }

    /// Inverse hyperbolic sine
    #[inline(always)]
    pub fn asinh(&self) -> Result<Self> {
        use crate::ops::activation::asinh;
        asinh(self)
    }

    /// Inverse hyperbolic cosine
    #[inline(always)]
    pub fn acosh(&self) -> Result<Self> {
        use crate::ops::activation::acosh;
        acosh(self)
    }

    /// Inverse hyperbolic tangent
    #[inline(always)]
    pub fn atanh(&self) -> Result<Self> {
        use crate::ops::activation::atanh;
        atanh(self)
    }

    /// Hyperbolic tangent
    #[inline(always)]
    pub fn tanh(&self) -> Result<Self> {
        use crate::ops::activation::tanh;
        tanh(self)
    }

    /// Sigmoid activation
    #[inline(always)]
    pub fn sigmoid(&self) -> Result<Self> {
        use crate::ops::activation::sigmoid;
        sigmoid(self)
    }

    /// Softplus activation
    #[inline(always)]
    pub fn softplus(&self, beta: f64, threshold: f64) -> Result<Self> {
        use crate::ops::activation::softplus;
        softplus(self, beta, threshold)
    }

    /// GELU activation
    #[inline(always)]
    pub fn gelu(&self, approximate: bool) -> Result<Self> {
        use crate::ops::activation::gelu;
        gelu(self, approximate)
    }

    /// ELU activation
    #[inline(always)]
    pub fn elu(&self, alpha: f64) -> Result<Self> {
        use crate::ops::activation::elu;
        elu(self, alpha)
    }

    /// SELU activation
    #[inline(always)]
    pub fn selu(&self) -> Result<Self> {
        use crate::ops::activation::selu;
        selu(self)
    }

    /// SiLU activation
    #[inline(always)]
    pub fn silu(&self) -> Result<Self> {
        use crate::ops::activation::silu;
        silu(self)
    }

    /// Softsign activation
    #[inline(always)]
    pub fn softsign(&self) -> Result<Self> {
        use crate::ops::activation::softsign;
        softsign(self)
    }

    /// ReLU activation
    #[inline(always)]
    pub fn relu(&self) -> Result<Self> {
        use crate::ops::activation::relu;
        relu(self)
    }

    /// Hardshrink activation
    #[inline(always)]
    pub fn hardshrink(&self, lambd: f64) -> Result<Self> {
        use crate::ops::activation::hardshrink;
        hardshrink(self, lambd)
    }

    /// Softmax activation
    #[inline(always)]
    pub fn softmax(&self, dim: Option<usize>) -> Result<Self> {
        use crate::ops::activation::softmax;
        softmax(self, dim)
    }

    /// Log-Softmax activation
    #[inline(always)]
    pub fn log_softmax(&self, dim: Option<usize>) -> Result<Self> {
        use crate::ops::activation::log_softmax;
        log_softmax(self, dim)
    }

    /// Masked Softmax activation
    #[inline(always)]
    pub fn masked_softmax(&self, mask: &Tensor, dim: Option<usize>) -> Result<Self> {
        use crate::ops::activation::masked_softmax;
        masked_softmax(self, mask, dim)
    }

    /// Masked Log-Softmax activation
    #[inline(always)]
    pub fn masked_log_softmax(&self, mask: &Tensor, dim: Option<usize>) -> Result<Self> {
        use crate::ops::activation::masked_log_softmax;
        masked_log_softmax(self, mask, dim)
    }
}

// ===== indexing and slicing methods =====
impl Tensor {
    /// Squeeze dimensions of size 1
    #[inline(always)]
    pub fn squeeze(&self) -> Result<Self> {
        let new_dims: Vec<usize> = self
            .shape
            .dims()
            .iter()
            .filter(|&&dim| dim != 1)
            .copied()
            .collect();

        let new_shape = Shape::new(new_dims);
        self.reshape(new_shape)
    }

    /// Squeeze specific dimension if it has size 1. Negative indices are supported.
    #[inline(always)]
    pub fn squeeze_dim(&self, dim: isize) -> Result<Self> {
        let ndim = self.ndim() as isize;
        let dim = if dim < 0 { dim + ndim } else { dim };

        if dim < 0 || dim >= ndim {
            return Err(MinitensorError::index_error(dim, 0, ndim as usize));
        }

        let dim = dim as usize;

        if self.shape.dims()[dim] != 1 {
            return Ok(self.clone());
        }

        let mut new_dims = self.shape.dims().to_vec();
        new_dims.remove(dim);
        let new_shape = Shape::new(new_dims);
        self.reshape(new_shape)
    }

    /// Add dimension of size 1. Negative indices are supported.
    #[inline(always)]
    pub fn unsqueeze(&self, dim: isize) -> Result<Self> {
        let ndim = self.ndim() as isize;
        let dim = if dim < 0 { dim + ndim + 1 } else { dim };

        if dim < 0 || dim > ndim {
            return Err(MinitensorError::index_error(dim, 0, (ndim + 1) as usize));
        }

        let dim = dim as usize;

        let mut new_dims = self.shape.dims().to_vec();
        new_dims.insert(dim, 1);
        let new_shape = Shape::new(new_dims);
        self.reshape(new_shape)
    }

    /// Expand tensor dimensions without allocating new memory
    #[inline(always)]
    pub fn expand(&self, dims: Vec<isize>) -> Result<Self> {
        let orig_dims = self.shape.dims();
        let orig_strides = self.strides.as_slice();
        let n_orig = orig_dims.len();
        let n_new = dims.len();

        if n_new < n_orig {
            return Err(MinitensorError::invalid_operation(
                "cannot expand to fewer dimensions".to_string(),
            ));
        }

        let mut new_dims = vec![0usize; n_new];
        let mut new_strides = vec![0usize; n_new];

        for i in 0..n_new {
            let size_spec = dims[n_new - 1 - i];
            if size_spec < -1 {
                return Err(MinitensorError::invalid_operation(
                    "invalid negative dimension".to_string(),
                ));
            }

            let orig_idx_opt = if i < n_orig {
                Some(n_orig - 1 - i)
            } else {
                None
            };
            let orig_dim = orig_idx_opt.map(|idx| orig_dims[idx]).unwrap_or(1);
            let orig_stride = orig_idx_opt.map(|idx| orig_strides[idx]).unwrap_or(0);

            let target = if size_spec == -1 {
                orig_dim
            } else {
                size_spec as usize
            };

            if let Some(idx) = orig_idx_opt {
                if target == orig_dim {
                    new_dims[n_new - 1 - i] = target;
                    new_strides[n_new - 1 - i] = orig_stride;
                } else if orig_dim == 1 && target > 0 {
                    new_dims[n_new - 1 - i] = target;
                    new_strides[n_new - 1 - i] = 0;
                } else {
                    return Err(MinitensorError::invalid_operation(format!(
                        "cannot expand dimension {} from {} to {}",
                        idx, orig_dim, target
                    )));
                }
            } else {
                if size_spec == -1 {
                    return Err(MinitensorError::invalid_operation(
                        "the size -1 is not allowed for a new leading dimension".to_string(),
                    ));
                }
                // New leading dimensions broadcast with stride 0.
                new_dims[n_new - 1 - i] = target;
                new_strides[n_new - 1 - i] = 0;
            }
        }

        let mut tensor = self.clone();
        tensor.refresh_autograd_metadata();
        tensor.shape = Shape::new(new_dims.clone());
        tensor.strides = Strides::new(new_strides);

        if tensor.requires_grad {
            let grad_fn = Arc::new(crate::autograd::ExpandBackward {
                input_shape: orig_dims.to_vec(),
                input_id: self.id(),
            });
            tensor.set_grad_fn(Some(grad_fn.clone()));
            autograd::add_to_graph(&tensor, Some(grad_fn))?;
        }

        Ok(tensor)
    }

    /// Repeat tensor according to `repeats` along each dimension
    #[inline(always)]
    pub fn repeat(&self, repeats: Vec<usize>) -> Result<Self> {
        crate::ops::shape_ops::repeat(self, &repeats)
    }

    /// Flatten tensor from `start_dim` to `end_dim`
    pub fn flatten(&self, start_dim: isize, end_dim: isize) -> Result<Self> {
        let ndim = self.ndim() as isize;

        let start = if start_dim < 0 {
            start_dim + ndim
        } else {
            start_dim
        };
        let end = if end_dim < 0 { end_dim + ndim } else { end_dim };

        if start < 0 || start >= ndim {
            return Err(MinitensorError::index_error(start, 0, ndim as usize));
        }
        if end < 0 || end >= ndim {
            return Err(MinitensorError::index_error(end, 0, ndim as usize));
        }
        if start > end {
            return Err(MinitensorError::invalid_argument(
                "start_dim must be less than or equal to end_dim",
            ));
        }

        self.flatten_range(start as usize, end as usize)
    }

    /// Flatten tensor from start_dim to end_dim
    #[inline(always)]
    pub fn flatten_range(&self, start_dim: usize, end_dim: usize) -> Result<Self> {
        if start_dim >= self.ndim() || end_dim >= self.ndim() || start_dim > end_dim {
            return Err(MinitensorError::invalid_argument(
                "Invalid dimension range for flatten",
            ));
        }

        let dims = self.shape.dims();
        let mut new_dims = Vec::new();

        // Add dimensions before start_dim
        new_dims.extend_from_slice(&dims[..start_dim]);

        // Compute flattened dimension size
        let flattened_size: usize = dims[start_dim..=end_dim].iter().product();
        new_dims.push(flattened_size);

        // Add dimensions after end_dim
        if end_dim + 1 < dims.len() {
            new_dims.extend_from_slice(&dims[end_dim + 1..]);
        }

        let new_shape = Shape::new(new_dims);
        self.reshape(new_shape)
    }
}

impl Tensor {
    /// Basic tensor indexing and slicing
    #[inline(always)]
    pub fn index(&self, indices: &[TensorIndex]) -> Result<Self> {
        if indices.len() > self.ndim() {
            return Err(MinitensorError::invalid_argument(
                "Too many indices for tensor",
            ));
        }

        let shape_dims = self.shape.dims();
        let strides = self.strides.as_slice();
        let mut offset = 0usize;
        let mut out_dims = Vec::new();
        let mut orig_dim_map = Vec::new();
        let mut starts = Vec::new();
        let mut steps: Vec<usize> = Vec::new();

        for i in 0..self.ndim() {
            let dim_size = shape_dims[i];
            let idx = indices.get(i).cloned().unwrap_or(TensorIndex::Slice {
                start: 0,
                end: dim_size,
                step: 1,
            });
            match idx {
                TensorIndex::Index(pos) => {
                    if pos >= dim_size {
                        return Err(MinitensorError::index_error(pos as isize, 0, dim_size));
                    }
                    offset += pos * strides[i];
                }
                TensorIndex::Slice { start, end, step } => {
                    if start > end || end > dim_size {
                        return Err(MinitensorError::index_error(end as isize, 0, dim_size));
                    }
                    let size = if end <= start {
                        0
                    } else {
                        (end - start).div_ceil(step)
                    };
                    out_dims.push(size);
                    orig_dim_map.push(i);
                    starts.push(start);
                    steps.push(step);
                }
            }
        }

        if out_dims.is_empty() {
            let mut result_data = TensorData::zeros_on_device(1, self.dtype, self.device);
            match self.dtype {
                DataType::Float32 => {
                    let input = self
                        .data
                        .as_f32_slice()
                        .ok_or_else(|| MinitensorError::internal_error("Expected f32 data"))?;
                    result_data.as_f32_slice_mut().unwrap()[0] = input[offset];
                }
                DataType::Float64 => {
                    let input = self
                        .data
                        .as_f64_slice()
                        .ok_or_else(|| MinitensorError::internal_error("Expected f64 data"))?;
                    result_data.as_f64_slice_mut().unwrap()[0] = input[offset];
                }
                DataType::Int32 => {
                    let input = self
                        .data
                        .as_i32_slice()
                        .ok_or_else(|| MinitensorError::internal_error("Expected i32 data"))?;
                    result_data.as_i32_slice_mut().unwrap()[0] = input[offset];
                }
                DataType::Int64 => {
                    let input = self
                        .data
                        .as_i64_slice()
                        .ok_or_else(|| MinitensorError::internal_error("Expected i64 data"))?;
                    result_data.as_i64_slice_mut().unwrap()[0] = input[offset];
                }
                DataType::Bool => {
                    let input = self
                        .data
                        .as_bool_slice()
                        .ok_or_else(|| MinitensorError::internal_error("Expected bool data"))?;
                    result_data.as_bool_slice_mut().unwrap()[0] = input[offset];
                }
            }
            let output = Tensor::new(
                Arc::new(result_data),
                Shape::scalar(),
                self.dtype,
                self.device,
                self.requires_grad,
            );
            return self.wrap_index_grad(
                output,
                offset,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            );
        }

        let out_shape = Shape::new(out_dims.clone());
        let out_strides = Strides::from_shape(&out_shape);
        let mut result_data =
            TensorData::zeros_on_device(out_shape.numel(), self.dtype, self.device);

        match self.dtype {
            DataType::Float32 => {
                let input = self
                    .data
                    .as_f32_slice()
                    .ok_or_else(|| MinitensorError::internal_error("Expected f32 data"))?;
                let output = result_data.as_f32_slice_mut().unwrap();
                for (idx, out_elem) in output.iter_mut().enumerate() {
                    let mut rem = idx;
                    let mut src_idx = offset;
                    for (j, &stride) in out_strides.as_slice().iter().enumerate() {
                        let coord = rem / stride;
                        rem %= stride;
                        let orig_dim = orig_dim_map[j];
                        let step = steps[j];
                        src_idx += (starts[j] + coord * step) * strides[orig_dim];
                    }
                    *out_elem = input[src_idx];
                }
            }
            DataType::Float64 => {
                let input = self
                    .data
                    .as_f64_slice()
                    .ok_or_else(|| MinitensorError::internal_error("Expected f64 data"))?;
                let output = result_data.as_f64_slice_mut().unwrap();
                for (idx, out_elem) in output.iter_mut().enumerate() {
                    let mut rem = idx;
                    let mut src_idx = offset;
                    for (j, &stride) in out_strides.as_slice().iter().enumerate() {
                        let coord = rem / stride;
                        rem %= stride;
                        let orig_dim = orig_dim_map[j];
                        let step = steps[j];
                        src_idx += (starts[j] + coord * step) * strides[orig_dim];
                    }
                    *out_elem = input[src_idx];
                }
            }
            DataType::Int32 => {
                let input = self
                    .data
                    .as_i32_slice()
                    .ok_or_else(|| MinitensorError::internal_error("Expected i32 data"))?;
                let output = result_data.as_i32_slice_mut().unwrap();
                for (idx, out_elem) in output.iter_mut().enumerate() {
                    let mut rem = idx;
                    let mut src_idx = offset;
                    for (j, &stride) in out_strides.as_slice().iter().enumerate() {
                        let coord = rem / stride;
                        rem %= stride;
                        let orig_dim = orig_dim_map[j];
                        let step = steps[j];
                        src_idx += (starts[j] + coord * step) * strides[orig_dim];
                    }
                    *out_elem = input[src_idx];
                }
            }
            DataType::Int64 => {
                let input = self
                    .data
                    .as_i64_slice()
                    .ok_or_else(|| MinitensorError::internal_error("Expected i64 data"))?;
                let output = result_data.as_i64_slice_mut().unwrap();
                for (idx, out_elem) in output.iter_mut().enumerate() {
                    let mut rem = idx;
                    let mut src_idx = offset;
                    for (j, &stride) in out_strides.as_slice().iter().enumerate() {
                        let coord = rem / stride;
                        rem %= stride;
                        let orig_dim = orig_dim_map[j];
                        let step = steps[j];
                        src_idx += (starts[j] + coord * step) * strides[orig_dim];
                    }
                    *out_elem = input[src_idx];
                }
            }
            DataType::Bool => {
                let input = self
                    .data
                    .as_bool_slice()
                    .ok_or_else(|| MinitensorError::internal_error("Expected bool data"))?;
                let output = result_data.as_bool_slice_mut().unwrap();
                for (idx, out_elem) in output.iter_mut().enumerate() {
                    let mut rem = idx;
                    let mut src_idx = offset;
                    for (j, &stride) in out_strides.as_slice().iter().enumerate() {
                        let coord = rem / stride;
                        rem %= stride;
                        let orig_dim = orig_dim_map[j];
                        let step = steps[j];
                        src_idx += (starts[j] + coord * step) * strides[orig_dim];
                    }
                    *out_elem = input[src_idx];
                }
            }
        }

        let output = Tensor::new(
            Arc::new(result_data),
            out_shape,
            self.dtype,
            self.device,
            self.requires_grad,
        );
        self.wrap_index_grad(output, offset, out_dims, orig_dim_map, starts, steps)
    }

    /// Attach an [`IndexBackward`] gradient function to a freshly indexed tensor.
    ///
    /// `out_dims` is empty for a scalar (fully integer-indexed) result. Gradient
    /// tracking is only wired for floating-point, contiguous inputs, which is
    /// always the case at the Python boundary where indexing is applied.
    fn wrap_index_grad(
        &self,
        output: Tensor,
        offset: usize,
        out_dims: Vec<usize>,
        orig_dim_map: Vec<usize>,
        starts: Vec<usize>,
        steps: Vec<usize>,
    ) -> Result<Tensor> {
        if !self.requires_grad || !self.dtype.is_float() || !self.is_contiguous() {
            return Ok(output);
        }
        let grad_fn = Arc::new(crate::autograd::IndexBackward {
            input_id: self.tensor_id,
            input_shape: self.shape.dims().to_vec(),
            input_strides: self.strides.as_slice().to_vec(),
            offset,
            out_dims,
            orig_dim_map,
            starts,
            steps,
        });
        let mut output = output;
        output.set_grad_fn(Some(grad_fn.clone()));
        autograd::add_to_graph(&output, Some(grad_fn))?;
        Ok(output)
    }

    /// Assign values to tensor slice
    #[inline(always)]
    pub fn index_assign(&mut self, indices: &[TensorIndex], value: &Tensor) -> Result<()> {
        if indices.len() > self.ndim() {
            return Err(MinitensorError::invalid_argument(
                "Too many indices for tensor",
            ));
        }

        let shape_dims = self.shape.dims();
        let strides = self.strides.as_slice();
        let mut offset = 0usize;
        let mut out_dims = Vec::new();
        let mut orig_dim_map = Vec::new();
        let mut starts = Vec::new();
        let mut steps: Vec<usize> = Vec::new();

        for i in 0..self.ndim() {
            let dim_size = shape_dims[i];
            let idx = indices.get(i).cloned().unwrap_or(TensorIndex::Slice {
                start: 0,
                end: dim_size,
                step: 1,
            });
            match idx {
                TensorIndex::Index(pos) => {
                    if pos >= dim_size {
                        return Err(MinitensorError::index_error(pos as isize, 0, dim_size));
                    }
                    offset += pos * strides[i];
                }
                TensorIndex::Slice { start, end, step } => {
                    if start > end || end > dim_size {
                        return Err(MinitensorError::index_error(end as isize, 0, dim_size));
                    }
                    let size = if end <= start {
                        0
                    } else {
                        (end - start).div_ceil(step)
                    };
                    out_dims.push(size);
                    orig_dim_map.push(i);
                    starts.push(start);
                    steps.push(step);
                }
            }
        }

        let out_shape = Shape::new(out_dims.clone());
        if value.numel() != out_shape.numel() && value.numel() != 1 {
            return Err(MinitensorError::invalid_argument(
                "Assigned value has incompatible shape",
            ));
        }

        let out_strides = Strides::from_shape(&out_shape);
        let data = if let Some(d) = Arc::get_mut(&mut self.data) {
            d
        } else {
            let cloned = self.data.clone_data();
            self.data = Arc::new(cloned);
            Arc::get_mut(&mut self.data).unwrap()
        };

        match self.dtype {
            DataType::Float32 => {
                let slice = data.as_f32_slice_mut().unwrap();
                let val_slice = value.data().as_f32_slice().unwrap();
                for (idx, &val) in val_slice.iter().cycle().take(out_shape.numel()).enumerate() {
                    let mut rem = idx;
                    let mut src_idx = offset;
                    for (j, &stride) in out_strides.as_slice().iter().enumerate() {
                        let coord = rem / stride;
                        rem %= stride;
                        let orig_dim = orig_dim_map[j];
                        let step = steps[j];
                        src_idx += (starts[j] + coord * step) * strides[orig_dim];
                    }
                    slice[src_idx] = val;
                }
            }
            DataType::Float64 => {
                let slice = data.as_f64_slice_mut().unwrap();
                let val_slice = value.data().as_f64_slice().unwrap();
                for (idx, &val) in val_slice.iter().cycle().take(out_shape.numel()).enumerate() {
                    let mut rem = idx;
                    let mut src_idx = offset;
                    for (j, &stride) in out_strides.as_slice().iter().enumerate() {
                        let coord = rem / stride;
                        rem %= stride;
                        let orig_dim = orig_dim_map[j];
                        let step = steps[j];
                        src_idx += (starts[j] + coord * step) * strides[orig_dim];
                    }
                    slice[src_idx] = val;
                }
            }
            DataType::Int32 => {
                let slice = data.as_i32_slice_mut().unwrap();
                let val_slice = value.data().as_i32_slice().unwrap();
                for (idx, &val) in val_slice.iter().cycle().take(out_shape.numel()).enumerate() {
                    let mut rem = idx;
                    let mut src_idx = offset;
                    for (j, &stride) in out_strides.as_slice().iter().enumerate() {
                        let coord = rem / stride;
                        rem %= stride;
                        let orig_dim = orig_dim_map[j];
                        let step = steps[j];
                        src_idx += (starts[j] + coord * step) * strides[orig_dim];
                    }
                    slice[src_idx] = val;
                }
            }
            DataType::Int64 => {
                let slice = data.as_i64_slice_mut().unwrap();
                let val_slice = value.data().as_i64_slice().unwrap();
                for (idx, &val) in val_slice.iter().cycle().take(out_shape.numel()).enumerate() {
                    let mut rem = idx;
                    let mut src_idx = offset;
                    for (j, &stride) in out_strides.as_slice().iter().enumerate() {
                        let coord = rem / stride;
                        rem %= stride;
                        let orig_dim = orig_dim_map[j];
                        let step = steps[j];
                        src_idx += (starts[j] + coord * step) * strides[orig_dim];
                    }
                    slice[src_idx] = val;
                }
            }
            DataType::Bool => {
                let slice = data.as_bool_slice_mut().unwrap();
                let val_slice = value.data().as_bool_slice().unwrap();
                for (idx, &val) in val_slice.iter().cycle().take(out_shape.numel()).enumerate() {
                    let mut rem = idx;
                    let mut src_idx = offset;
                    for (j, &stride) in out_strides.as_slice().iter().enumerate() {
                        let coord = rem / stride;
                        rem %= stride;
                        let orig_dim = orig_dim_map[j];
                        let step = steps[j];
                        src_idx += (starts[j] + coord * step) * strides[orig_dim];
                    }
                    slice[src_idx] = val;
                }
            }
        }
        Ok(())
    }
}

impl Tensor {
    /// Check if tensor contains NaN values
    #[inline(always)]
    pub fn has_nan(&self) -> bool {
        match self.dtype {
            DataType::Float32 => {
                if let Some(data) = self.data.as_f32_slice() {
                    data.iter().any(|&x| x.is_nan())
                } else {
                    false
                }
            }
            DataType::Float64 => {
                if let Some(data) = self.data.as_f64_slice() {
                    data.iter().any(|&x| x.is_nan())
                } else {
                    false
                }
            }
            _ => false, // Integer and boolean types cannot be NaN
        }
    }

    /// Check if tensor contains infinite values
    #[inline(always)]
    pub fn has_inf(&self) -> bool {
        match self.dtype {
            DataType::Float32 => {
                if let Some(data) = self.data.as_f32_slice() {
                    data.iter().any(|&x| x.is_infinite())
                } else {
                    false
                }
            }
            DataType::Float64 => {
                if let Some(data) = self.data.as_f64_slice() {
                    data.iter().any(|&x| x.is_infinite())
                } else {
                    false
                }
            }
            _ => false, // Integer and boolean types cannot be infinite
        }
    }

    /// Element-wise check for NaN values
    #[inline(always)]
    pub fn isnan(&self) -> Result<Tensor> {
        let mut output = TensorData::zeros_on_device(self.numel(), DataType::Bool, self.device);
        match self.dtype {
            DataType::Float32 => {
                let input = self
                    .data
                    .as_f32_slice()
                    .ok_or_else(|| MinitensorError::internal_error("Expected f32 data"))?;
                let out_slice = output
                    .as_bool_slice_mut()
                    .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;
                for (o, &x) in out_slice.iter_mut().zip(input.iter()) {
                    *o = x.is_nan();
                }
            }
            DataType::Float64 => {
                let input = self
                    .data
                    .as_f64_slice()
                    .ok_or_else(|| MinitensorError::internal_error("Expected f64 data"))?;
                let out_slice = output
                    .as_bool_slice_mut()
                    .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;
                for (o, &x) in out_slice.iter_mut().zip(input.iter()) {
                    *o = x.is_nan();
                }
            }
            _ => {
                // Non-floating types cannot be NaN; output already zero
            }
        }
        Ok(Tensor::new(
            Arc::new(output),
            self.shape.clone(),
            DataType::Bool,
            self.device,
            false,
        ))
    }

    /// Element-wise check for infinite values
    #[inline(always)]
    pub fn isinf(&self) -> Result<Tensor> {
        let mut output = TensorData::zeros_on_device(self.numel(), DataType::Bool, self.device);
        match self.dtype {
            DataType::Float32 => {
                let input = self
                    .data
                    .as_f32_slice()
                    .ok_or_else(|| MinitensorError::internal_error("Expected f32 data"))?;
                let out_slice = output
                    .as_bool_slice_mut()
                    .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;
                for (o, &x) in out_slice.iter_mut().zip(input.iter()) {
                    *o = x.is_infinite();
                }
            }
            DataType::Float64 => {
                let input = self
                    .data
                    .as_f64_slice()
                    .ok_or_else(|| MinitensorError::internal_error("Expected f64 data"))?;
                let out_slice = output
                    .as_bool_slice_mut()
                    .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;
                for (o, &x) in out_slice.iter_mut().zip(input.iter()) {
                    *o = x.is_infinite();
                }
            }
            _ => {
                // Non-floating types cannot be infinite; output remains false
            }
        }
        Ok(Tensor::new(
            Arc::new(output),
            self.shape.clone(),
            DataType::Bool,
            self.device,
            false,
        ))
    }

    /// Element-wise check for finite values
    #[inline(always)]
    pub fn isfinite(&self) -> Result<Tensor> {
        let mut output = TensorData::zeros_on_device(self.numel(), DataType::Bool, self.device);
        match self.dtype {
            DataType::Float32 => {
                let input = self
                    .data
                    .as_f32_slice()
                    .ok_or_else(|| MinitensorError::internal_error("Expected f32 data"))?;
                let out_slice = output
                    .as_bool_slice_mut()
                    .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;
                for (o, &x) in out_slice.iter_mut().zip(input.iter()) {
                    *o = x.is_finite();
                }
            }
            DataType::Float64 => {
                let input = self
                    .data
                    .as_f64_slice()
                    .ok_or_else(|| MinitensorError::internal_error("Expected f64 data"))?;
                let out_slice = output
                    .as_bool_slice_mut()
                    .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;
                for (o, &x) in out_slice.iter_mut().zip(input.iter()) {
                    *o = x.is_finite();
                }
            }
            _ => {
                // Integer and bool types are always finite
                let out_slice = output
                    .as_bool_slice_mut()
                    .ok_or_else(|| MinitensorError::internal_error("Failed to get bool slice"))?;
                for o in out_slice.iter_mut() {
                    *o = true;
                }
            }
        }
        Ok(Tensor::new(
            Arc::new(output),
            self.shape.clone(),
            DataType::Bool,
            self.device,
            false,
        ))
    }

    /// Get the maximum value in the tensor
    #[inline(always)]
    pub fn max_value(&self) -> Option<f64> {
        match self.dtype {
            DataType::Float32 => self
                .data
                .as_f32_slice()?
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|&x| x as f64),
            DataType::Float64 => self
                .data
                .as_f64_slice()?
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .copied(),
            DataType::Int32 => self.data.as_i32_slice()?.iter().max().map(|&x| x as f64),
            DataType::Int64 => self.data.as_i64_slice()?.iter().max().map(|&x| x as f64),
            DataType::Bool => self
                .data
                .as_bool_slice()?
                .iter()
                .max()
                .map(|&x| if x { 1.0 } else { 0.0 }),
        }
    }

    /// Get the minimum value in the tensor
    #[inline(always)]
    pub fn min_value(&self) -> Option<f64> {
        match self.dtype {
            DataType::Float32 => self
                .data
                .as_f32_slice()?
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|&x| x as f64),
            DataType::Float64 => self
                .data
                .as_f64_slice()?
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .copied(),
            DataType::Int32 => self.data.as_i32_slice()?.iter().min().map(|&x| x as f64),
            DataType::Int64 => self.data.as_i64_slice()?.iter().min().map(|&x| x as f64),
            DataType::Bool => self
                .data
                .as_bool_slice()?
                .iter()
                .min()
                .map(|&x| if x { 1.0 } else { 0.0 }),
        }
    }

    /// Get memory usage in bytes
    #[inline(always)]
    pub fn memory_usage_bytes(&self) -> usize {
        let element_size = match self.dtype {
            DataType::Float32 => 4,
            DataType::Float64 => 8,
            DataType::Int32 => 4,
            DataType::Int64 => 8,
            DataType::Bool => 1,
        };
        self.numel() * element_size
    }

    /// Get the stride information
    pub fn stride(&self) -> &Strides {
        &self.strides
    }

    /// Check if this tensor is a leaf node in the computation graph
    #[inline(always)]
    pub fn is_leaf(&self) -> bool {
        self.grad_fn.is_none()
    }
}

// ===== dtype cast, comparison helpers, and conversions =====
#[inline(always)]
fn allclose_f32(a: f32, b: f32, rtol: f32, atol: f32, equal_nan: bool) -> bool {
    if a == b {
        return true;
    }
    if equal_nan && a.is_nan() && b.is_nan() {
        return true;
    }
    if !a.is_finite() || !b.is_finite() {
        return false;
    }
    let diff = (a - b).abs();
    diff <= atol + rtol * b.abs()
}

#[inline(always)]
fn allclose_f64(a: f64, b: f64, rtol: f64, atol: f64, equal_nan: bool) -> bool {
    if a == b {
        return true;
    }
    if equal_nan && a.is_nan() && b.is_nan() {
        return true;
    }
    if !a.is_finite() || !b.is_finite() {
        return false;
    }
    let diff = (a - b).abs();
    diff <= atol + rtol * b.abs()
}

impl Tensor {
    /// Solve a linear system `AX = B` for `X` where `self` provides `A`.
    pub fn solve(&self, rhs: &Self) -> Result<Self> {
        use crate::ops::linalg::solve;
        solve(self, rhs)
    }

    /// Layer normalization
    #[inline(always)]
    pub fn layer_norm(
        &self,
        normalized_shape: &[usize],
        weight: Option<&Tensor>,
        bias: Option<&Tensor>,
        eps: f64,
    ) -> Result<Self> {
        use crate::ops::normalization::layer_norm;
        layer_norm(self, normalized_shape, weight, bias, eps)
    }

    /// Root-mean-square layer normalization (RMSNorm).
    pub fn rms_norm(
        &self,
        normalized_shape: &[usize],
        weight: Option<&Tensor>,
        eps: f64,
    ) -> Result<Self> {
        use crate::ops::normalization::rms_norm;
        rms_norm(self, normalized_shape, weight, eps)
    }

    /// Absolute value
    #[inline(always)]
    pub fn abs(&self) -> Result<Self> {
        use crate::ops::activation::abs;
        abs(self)
    }

    /// Element-wise sign (returns -1, 0, or 1 for each value).
    #[inline(always)]
    pub fn sign(&self) -> Result<Self> {
        use crate::ops::activation::sign;
        sign(self)
    }

    /// Clip tensor values to the provided range.
    #[inline(always)]
    pub fn clip(&self, min_val: Option<f64>, max_val: Option<f64>) -> Result<Self> {
        if let (Some(min), Some(max)) = (min_val, max_val)
            && min > max
        {
            return Err(MinitensorError::invalid_argument(format!(
                "clip minimum {min} cannot be greater than maximum {max}",
            )));
        }

        use crate::ops::activation::clip;
        clip(self, min_val, max_val)
    }

    /// Alias for [`Tensor::clip`].
    #[inline(always)]
    pub fn clamp(&self, min_val: Option<f64>, max_val: Option<f64>) -> Result<Self> {
        self.clip(min_val, max_val)
    }

    /// Clamp tensor values to be no smaller than `min_val`.
    #[inline(always)]
    pub fn clamp_min(&self, min_val: f64) -> Result<Self> {
        self.clip(Some(min_val), None)
    }

    /// Clamp tensor values to be no larger than `max_val`.
    #[inline(always)]
    pub fn clamp_max(&self, max_val: f64) -> Result<Self> {
        self.clip(None, Some(max_val))
    }

    /// Replace NaN with `nan`, positive infinity with `posinf` or dtype max,
    /// and negative infinity with `neginf` or dtype min.
    #[inline(always)]
    pub fn nan_to_num(&self, nan: f64, posinf: Option<f64>, neginf: Option<f64>) -> Result<Self> {
        use crate::ops::activation::nan_to_num;
        nan_to_num(self, nan, posinf, neginf)
    }

    /// Round tensor values to a specific number of decimal places.
    #[inline(always)]
    pub fn round(&self, decimals: i32) -> Result<Self> {
        use crate::ops::activation::round;
        round(self, decimals)
    }

    /// Floor tensor values element-wise.
    #[inline(always)]
    pub fn floor(&self) -> Result<Self> {
        use crate::ops::activation::floor;
        floor(self)
    }

    /// Ceil tensor values element-wise.
    #[inline(always)]
    pub fn ceil(&self) -> Result<Self> {
        use crate::ops::activation::ceil;
        ceil(self)
    }

    /// Square root
    #[inline(always)]
    pub fn sqrt(&self) -> Result<Self> {
        use crate::ops::activation::sqrt;
        sqrt(self)
    }

    pub fn rsqrt(&self) -> Result<Self> {
        use crate::ops::activation::rsqrt;
        rsqrt(self)
    }

    /// Element-wise reciprocal (1/x).
    #[inline(always)]
    pub fn reciprocal(&self) -> Result<Self> {
        use crate::ops::activation::reciprocal;
        reciprocal(self)
    }

    /// Raise tensor elements to a scalar power
    #[inline(always)]
    pub fn powf(&self, exponent: f64) -> Result<Self> {
        use crate::ops::activation::powf;
        powf(self, exponent)
    }

    /// Numerically stable logaddexp
    #[inline(always)]
    pub fn logaddexp(&self, other: &Tensor) -> Result<Self> {
        use crate::ops::activation::logaddexp;
        logaddexp(self, other)
    }

    /// Element-wise power with another tensor
    pub fn pow(&self, exponent: &Tensor) -> Result<Self> {
        use crate::ops::activation::pow;
        pow(self, exponent)
    }

    /// Move tensor to device
    #[inline(always)]
    pub fn to(&self, device: Device) -> Result<Self> {
        if self.device == device {
            return Ok(self.clone());
        }

        // For now, just clone the tensor with the new device
        // In a full implementation, we'd copy data between devices
        let mut new_tensor = self.clone();
        new_tensor.device = device;
        Ok(new_tensor)
    }

    /// Convert tensor to a different data type
    #[inline(always)]
    pub fn astype(&self, dtype: DataType) -> Result<Self> {
        if self.dtype == dtype {
            return Ok(self.clone());
        }

        // Non-contiguous storage (e.g. `expand` views) must be materialized
        // first so the stored element count matches the logical shape. The
        // previous zip-based fill silently zero-padded the tail in that case.
        if !(self.is_contiguous() && self.data.is_contiguous()) {
            return self.contiguous()?.astype(dtype);
        }

        // Cast `src` into a fresh buffer for the target dtype (no zeroing
        // pass). `cast_num!` covers numeric sources (`as` casts, `!= 0` for
        // bool targets); the bool source arm maps to 0/1 explicitly.
        macro_rules! cast_num {
            ($src:expr, $sty:ty) => {{
                let src = $src.ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get source slice from tensor data")
                })?;
                match dtype {
                    DataType::Float32 => TensorData::from_vec::<f32>(
                        unary_map_threshold(src, CAST_PAR_THRESHOLD, |v: $sty| v as f32),
                        dtype,
                        self.device,
                    ),
                    DataType::Float64 => TensorData::from_vec::<f64>(
                        unary_map_threshold(src, CAST_PAR_THRESHOLD, |v: $sty| v as f64),
                        dtype,
                        self.device,
                    ),
                    DataType::Int32 => TensorData::from_vec::<i32>(
                        unary_map_threshold(src, CAST_PAR_THRESHOLD, |v: $sty| v as i32),
                        dtype,
                        self.device,
                    ),
                    DataType::Int64 => TensorData::from_vec::<i64>(
                        unary_map_threshold(src, CAST_PAR_THRESHOLD, |v: $sty| v as i64),
                        dtype,
                        self.device,
                    ),
                    DataType::Bool => TensorData::from_vec::<bool>(
                        unary_map_threshold(src, CAST_PAR_THRESHOLD, |v: $sty| v != 0 as $sty),
                        dtype,
                        self.device,
                    ),
                }
            }};
        }

        let new_data =
            match self.dtype {
                DataType::Float32 => cast_num!(self.data.as_f32_slice(), f32),
                DataType::Float64 => cast_num!(self.data.as_f64_slice(), f64),
                DataType::Int32 => cast_num!(self.data.as_i32_slice(), i32),
                DataType::Int64 => cast_num!(self.data.as_i64_slice(), i64),
                DataType::Bool => {
                    let src = self.data.as_bool_slice().ok_or_else(|| {
                        MinitensorError::internal_error("Failed to get bool slice from tensor data")
                    })?;
                    match dtype {
                        DataType::Float32 => {
                            TensorData::from_vec::<f32>(
                                unary_map_threshold(src, CAST_PAR_THRESHOLD, |v: bool| {
                                    if v { 1.0 } else { 0.0 }
                                }),
                                dtype,
                                self.device,
                            )
                        }
                        DataType::Float64 => {
                            TensorData::from_vec::<f64>(
                                unary_map_threshold(src, CAST_PAR_THRESHOLD, |v: bool| {
                                    if v { 1.0 } else { 0.0 }
                                }),
                                dtype,
                                self.device,
                            )
                        }
                        DataType::Int32 => TensorData::from_vec::<i32>(
                            unary_map_threshold(
                                src,
                                CAST_PAR_THRESHOLD,
                                |v: bool| {
                                    if v { 1 } else { 0 }
                                },
                            ),
                            dtype,
                            self.device,
                        ),
                        DataType::Int64 => TensorData::from_vec::<i64>(
                            unary_map_threshold(
                                src,
                                CAST_PAR_THRESHOLD,
                                |v: bool| {
                                    if v { 1 } else { 0 }
                                },
                            ),
                            dtype,
                            self.device,
                        ),
                        DataType::Bool => unreachable!("same-dtype cast returns early"),
                    }
                }
            };

        Ok(Tensor::new(
            Arc::new(new_data),
            self.shape.clone(),
            dtype,
            self.device,
            self.requires_grad,
        ))
    }
}

impl Tensor {
    /// Copy data from ``source`` into this tensor in-place, preserving dtype and device.
    pub fn copy_(&mut self, source: &Tensor) -> Result<()> {
        if self.shape != *source.shape() {
            return Err(MinitensorError::invalid_argument(format!(
                "copy_ expected source with shape {:?}, but received {:?}",
                self.shape.dims(),
                source.shape().dims()
            )));
        }

        if !self.device.is_cpu() {
            return Err(MinitensorError::invalid_operation(
                "copy_ currently supports only CPU tensors".to_string(),
            ));
        }

        let mut prepared: Cow<'_, Tensor> = Cow::Borrowed(source);

        if prepared.dtype() != self.dtype {
            prepared = Cow::Owned(prepared.astype(self.dtype)?);
        }

        if prepared.device() != self.device {
            prepared = Cow::Owned(prepared.to(self.device)?);
        }

        if !prepared.is_contiguous() {
            prepared = Cow::Owned(prepared.contiguous()?);
        }

        if !self.is_contiguous() {
            return Err(MinitensorError::invalid_operation(
                "copy_ currently requires the destination tensor to be contiguous".to_string(),
            ));
        }

        let dtype = self.dtype;
        {
            let dst_data = self.data_mut();
            match dtype {
                DataType::Float32 => {
                    let dst = dst_data.as_f32_slice_mut().ok_or_else(|| {
                        MinitensorError::internal_error(
                            "failed to obtain mutable float32 slice for copy_".to_string(),
                        )
                    })?;
                    let src = prepared.data().as_f32_slice().ok_or_else(|| {
                        MinitensorError::internal_error(
                            "failed to access float32 source data for copy_".to_string(),
                        )
                    })?;
                    dst.copy_from_slice(src);
                }
                DataType::Float64 => {
                    let dst = dst_data.as_f64_slice_mut().ok_or_else(|| {
                        MinitensorError::internal_error(
                            "failed to obtain mutable float64 slice for copy_".to_string(),
                        )
                    })?;
                    let src = prepared.data().as_f64_slice().ok_or_else(|| {
                        MinitensorError::internal_error(
                            "failed to access float64 source data for copy_".to_string(),
                        )
                    })?;
                    dst.copy_from_slice(src);
                }
                DataType::Int32 => {
                    let dst = dst_data.as_i32_slice_mut().ok_or_else(|| {
                        MinitensorError::internal_error(
                            "failed to obtain mutable int32 slice for copy_".to_string(),
                        )
                    })?;
                    let src = prepared.data().as_i32_slice().ok_or_else(|| {
                        MinitensorError::internal_error(
                            "failed to access int32 source data for copy_".to_string(),
                        )
                    })?;
                    dst.copy_from_slice(src);
                }
                DataType::Int64 => {
                    let dst = dst_data.as_i64_slice_mut().ok_or_else(|| {
                        MinitensorError::internal_error(
                            "failed to obtain mutable int64 slice for copy_".to_string(),
                        )
                    })?;
                    let src = prepared.data().as_i64_slice().ok_or_else(|| {
                        MinitensorError::internal_error(
                            "failed to access int64 source data for copy_".to_string(),
                        )
                    })?;
                    dst.copy_from_slice(src);
                }
                DataType::Bool => {
                    let dst = dst_data.as_bool_slice_mut().ok_or_else(|| {
                        MinitensorError::internal_error(
                            "failed to obtain mutable bool slice for copy_".to_string(),
                        )
                    })?;
                    let src = prepared.data().as_bool_slice().ok_or_else(|| {
                        MinitensorError::internal_error(
                            "failed to access bool source data for copy_".to_string(),
                        )
                    })?;
                    dst.copy_from_slice(src);
                }
            }
        }

        self.refresh_autograd_metadata();
        if self.requires_grad {
            autograd::add_to_graph(self, None)?;
        }

        Ok(())
    }

    /// Fill the tensor in-place with ``value`` converted to the tensor dtype.
    pub fn fill_(&mut self, value: f64) -> Result<()> {
        if !self.device.is_cpu() {
            return Err(MinitensorError::invalid_operation(
                "fill_ currently supports only CPU tensors".to_string(),
            ));
        }

        if !self.is_contiguous() {
            return Err(MinitensorError::invalid_operation(
                "fill_ currently requires contiguous tensors".to_string(),
            ));
        }

        let dtype = self.dtype;
        {
            let data = self.data_mut();
            match dtype {
                DataType::Float32 => {
                    let slice = data.as_f32_slice_mut().ok_or_else(|| {
                        MinitensorError::internal_error(
                            "failed to obtain mutable float32 slice for fill_".to_string(),
                        )
                    })?;
                    slice.fill(value as f32);
                }
                DataType::Float64 => {
                    let slice = data.as_f64_slice_mut().ok_or_else(|| {
                        MinitensorError::internal_error(
                            "failed to obtain mutable float64 slice for fill_".to_string(),
                        )
                    })?;
                    slice.fill(value);
                }
                DataType::Int32 => {
                    let slice = data.as_i32_slice_mut().ok_or_else(|| {
                        MinitensorError::internal_error(
                            "failed to obtain mutable int32 slice for fill_".to_string(),
                        )
                    })?;
                    slice.fill(value as i32);
                }
                DataType::Int64 => {
                    let slice = data.as_i64_slice_mut().ok_or_else(|| {
                        MinitensorError::internal_error(
                            "failed to obtain mutable int64 slice for fill_".to_string(),
                        )
                    })?;
                    slice.fill(value as i64);
                }
                DataType::Bool => {
                    let slice = data.as_bool_slice_mut().ok_or_else(|| {
                        MinitensorError::internal_error(
                            "failed to obtain mutable bool slice for fill_".to_string(),
                        )
                    })?;
                    slice.fill(value != 0.0);
                }
            }
        }

        self.refresh_autograd_metadata();
        if self.requires_grad {
            autograd::add_to_graph(self, None)?;
        }

        Ok(())
    }
}

impl Tensor {
    /// Detach tensor from computation graph
    #[inline(always)]
    pub fn detach(&self) -> Self {
        let mut detached = self.clone();
        detached.requires_grad = false;
        detached.grad_fn = None;
        detached.grad = None;
        detached
    }

    /// Detach tensor from the computation graph in-place
    #[inline(always)]
    pub fn detach_inplace(&mut self) {
        self.requires_grad = false;
        self.refresh_autograd_metadata();
    }

    /// Check if tensors are approximately equal
    #[inline(always)]
    pub fn allclose(&self, other: &Tensor, rtol: f64, atol: f64) -> bool {
        self.allclose_with_equal_nan(other, rtol, atol, false)
    }

    /// Check if tensors are approximately equal, optionally treating NaNs at
    /// matching positions as equal.
    #[inline(always)]
    pub fn allclose_with_equal_nan(
        &self,
        other: &Tensor,
        rtol: f64,
        atol: f64,
        equal_nan: bool,
    ) -> bool {
        if self.shape != other.shape || self.dtype != other.dtype {
            return false;
        }

        // Fast path: byte-for-byte equality check for contiguous CPU tensors
        if (equal_nan || !self.dtype.is_float())
            && self.device.is_cpu()
            && other.device.is_cpu()
            && self.is_contiguous()
            && other.is_contiguous()
            && let (Some(a), Some(b)) = (self.data.as_bytes(), other.data.as_bytes())
            && a == b
        {
            return true;
        }

        let numel = self.numel();
        match self.dtype {
            DataType::Float32 => {
                if let (Some(self_data), Some(other_data)) =
                    (self.data.as_f32_slice(), other.data.as_f32_slice())
                {
                    if numel >= 1024 {
                        self_data
                            .par_iter()
                            .zip(other_data.par_iter())
                            .all(|(&a, &b)| allclose_f32(a, b, rtol as f32, atol as f32, equal_nan))
                    } else {
                        self_data
                            .iter()
                            .zip(other_data.iter())
                            .all(|(&a, &b)| allclose_f32(a, b, rtol as f32, atol as f32, equal_nan))
                    }
                } else {
                    false
                }
            }
            DataType::Float64 => {
                if let (Some(self_data), Some(other_data)) =
                    (self.data.as_f64_slice(), other.data.as_f64_slice())
                {
                    if numel >= 1024 {
                        self_data
                            .par_iter()
                            .zip(other_data.par_iter())
                            .all(|(&a, &b)| allclose_f64(a, b, rtol, atol, equal_nan))
                    } else {
                        self_data
                            .iter()
                            .zip(other_data.iter())
                            .all(|(&a, &b)| allclose_f64(a, b, rtol, atol, equal_nan))
                    }
                } else {
                    false
                }
            }
            _ => self.array_equal(other),
        }
    }

    /// Check if tensors are exactly equal
    #[inline(always)]
    pub fn array_equal(&self, other: &Tensor) -> bool {
        if self.shape != other.shape || self.dtype != other.dtype {
            return false;
        }

        // Fast path for contiguous CPU tensors using raw bytes comparison
        if self.device.is_cpu()
            && other.device.is_cpu()
            && self.is_contiguous()
            && other.is_contiguous()
            && let (Some(a), Some(b)) = (self.data.as_bytes(), other.data.as_bytes())
        {
            return a == b;
        }

        let numel = self.numel();
        match self.dtype {
            DataType::Float32 => {
                if let (Some(self_data), Some(other_data)) =
                    (self.data.as_f32_slice(), other.data.as_f32_slice())
                {
                    if numel >= 1024 {
                        self_data
                            .par_iter()
                            .zip(other_data.par_iter())
                            .all(|(&a, &b)| a == b)
                    } else {
                        self_data == other_data
                    }
                } else {
                    false
                }
            }
            DataType::Float64 => {
                if let (Some(self_data), Some(other_data)) =
                    (self.data.as_f64_slice(), other.data.as_f64_slice())
                {
                    if numel >= 1024 {
                        self_data
                            .par_iter()
                            .zip(other_data.par_iter())
                            .all(|(&a, &b)| a == b)
                    } else {
                        self_data == other_data
                    }
                } else {
                    false
                }
            }
            DataType::Int32 => {
                if let (Some(self_data), Some(other_data)) =
                    (self.data.as_i32_slice(), other.data.as_i32_slice())
                {
                    if numel >= 1024 {
                        self_data
                            .par_iter()
                            .zip(other_data.par_iter())
                            .all(|(&a, &b)| a == b)
                    } else {
                        self_data == other_data
                    }
                } else {
                    false
                }
            }
            DataType::Int64 => {
                if let (Some(self_data), Some(other_data)) =
                    (self.data.as_i64_slice(), other.data.as_i64_slice())
                {
                    if numel >= 1024 {
                        self_data
                            .par_iter()
                            .zip(other_data.par_iter())
                            .all(|(&a, &b)| a == b)
                    } else {
                        self_data == other_data
                    }
                } else {
                    false
                }
            }
            DataType::Bool => {
                if let (Some(self_data), Some(other_data)) =
                    (self.data.as_bool_slice(), other.data.as_bool_slice())
                {
                    if numel >= 1024 {
                        self_data
                            .par_iter()
                            .zip(other_data.par_iter())
                            .all(|(&a, &b)| a == b)
                    } else {
                        self_data == other_data
                    }
                } else {
                    false
                }
            }
        }
    }
}

// ===== formatting and tests =====
impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("dtype", &self.dtype)
            .field("device", &self.device)
            .field("requires_grad", &self.requires_grad)
            .field("tensor_id", &self.tensor_id)
            .field("has_grad_fn", &self.grad_fn.is_some())
            .field("has_grad", &self.grad.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::storage::TensorData;

    #[test]
    fn test_tensor_creation() {
        let shape = Shape::new(vec![2, 3]);
        let data = Arc::new(TensorData::zeros(shape.numel(), DataType::Float32));
        let tensor = Tensor::new(data, shape.clone(), DataType::Float32, Device::cpu(), false);

        assert_eq!(tensor.shape(), &shape);
        assert_eq!(tensor.dtype(), DataType::Float32);
        assert_eq!(tensor.device(), Device::cpu());
        assert!(!tensor.requires_grad());
        assert_eq!(tensor.ndim(), 2);
        assert_eq!(tensor.numel(), 6);
    }

    #[test]
    fn test_tensor_view() {
        let shape = Shape::new(vec![2, 3]);
        let data = Arc::new(TensorData::zeros(shape.numel(), DataType::Float32));
        let tensor = Tensor::new(data, shape, DataType::Float32, Device::cpu(), false);

        let new_shape = Shape::new(vec![3, 2]);
        let reshaped = tensor.view(new_shape.clone()).unwrap();
        assert_eq!(reshaped.shape(), &new_shape);
        assert_eq!(reshaped.numel(), 6);
    }

    #[test]
    fn test_tensor_zeros_and_ones() {
        let shape = Shape::new(vec![2, 3]);

        let zeros = Tensor::zeros(shape.clone(), DataType::Float32, Device::cpu(), false);
        assert_eq!(zeros.shape(), &shape);
        assert_eq!(zeros.dtype(), DataType::Float32);
        assert!(!zeros.requires_grad());

        let ones = Tensor::ones(shape.clone(), DataType::Float32, Device::cpu(), true);
        assert_eq!(ones.shape(), &shape);
        assert_eq!(ones.dtype(), DataType::Float32);
        assert!(ones.requires_grad());
    }

    #[test]
    fn test_gradient_management() {
        let shape = Shape::new(vec![2, 2]);
        let mut tensor = Tensor::zeros(shape.clone(), DataType::Float32, Device::cpu(), true);

        // Initially no gradient
        assert!(!tensor.has_grad());
        assert!(tensor.grad().is_none());

        // Set a gradient
        let grad = Tensor::ones(shape.clone(), DataType::Float32, Device::cpu(), false);
        tensor.set_grad(Some(grad));
        assert!(tensor.has_grad());
        assert!(tensor.grad().is_some());

        // Clear gradient (should zero it in place)
        tensor.zero_grad(false);
        assert!(tensor.has_grad());
        let expected = Tensor::zeros(shape.clone(), DataType::Float32, Device::cpu(), false);
        assert!(tensor.grad().unwrap().allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_gradient_accumulation() {
        let shape = Shape::new(vec![2, 2]);
        let mut tensor = Tensor::zeros(shape.clone(), DataType::Float32, Device::cpu(), true);

        let grad1 = Tensor::ones(shape.clone(), DataType::Float32, Device::cpu(), false);
        let grad2 = Tensor::ones(shape.clone(), DataType::Float32, Device::cpu(), false);

        // Accumulate first gradient
        tensor.accumulate_grad(grad1).unwrap();
        assert!(tensor.has_grad());

        // Accumulate second gradient (should replace for now)
        tensor.accumulate_grad(grad2).unwrap();
        assert!(tensor.has_grad());
    }

    #[test]
    fn test_backward_scalar_tensor() {
        let shape = Shape::new(vec![1]);
        let tensor = Tensor::ones(shape, DataType::Float32, Device::cpu(), true);

        // This should work for scalar tensors and produce a gradient
        let result = tensor.backward(None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_backward_non_scalar_error() {
        let tensor = Tensor::ones(Shape::new(vec![2]), DataType::Float32, Device::cpu(), true);
        let result = tensor.backward(None);
        assert!(result.is_err());
    }

    #[test]
    fn test_isnan_isinf_isfinite() {
        let data = vec![0.0f32, f32::NAN, f32::INFINITY, -5.0];
        let shape = Shape::new(vec![4]);
        let tensor = Tensor::new(
            Arc::new(TensorData::from_vec_f32(data.clone(), Device::cpu())),
            shape.clone(),
            DataType::Float32,
            Device::cpu(),
            false,
        );

        let isnan = tensor.isnan().unwrap();
        let isinf = tensor.isinf().unwrap();
        let isfinite = tensor.isfinite().unwrap();

        let isnan_data = isnan.data().as_bool_slice().unwrap();
        let isinf_data = isinf.data().as_bool_slice().unwrap();
        let isfinite_data = isfinite.data().as_bool_slice().unwrap();

        assert_eq!(isnan_data, &[false, true, false, false]);
        assert_eq!(isinf_data, &[false, false, true, false]);
        assert_eq!(isfinite_data, &[true, false, false, true]);
        assert_eq!(isnan.shape(), &shape);
    }

    #[test]
    fn test_clamp() {
        let data = vec![-2.0f32, -0.5, 0.5, 2.0];
        let shape = Shape::new(vec![4]);
        let tensor = Tensor::new(
            Arc::new(TensorData::from_vec_f32(data.clone(), Device::cpu())),
            shape.clone(),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let clamped = tensor.clamp(Some(-1.0), Some(1.0)).unwrap();
        let clamped_data = clamped.data().as_f32_slice().unwrap();
        assert_eq!(clamped_data, &[-1.0, -0.5, 0.5, 1.0]);
        assert_eq!(clamped.shape(), &shape);
    }

    #[test]
    fn test_astype() {
        let data = vec![1.5f32, -2.3];
        let shape = Shape::new(vec![2]);
        let tensor = Tensor::new(
            Arc::new(TensorData::from_vec_f32(data.clone(), Device::cpu())),
            shape.clone(),
            DataType::Float32,
            Device::cpu(),
            false,
        );

        let casted = tensor.astype(DataType::Float64).unwrap();
        let casted_data = casted.data().as_f64_slice().unwrap();
        assert!((casted_data[0] - 1.5).abs() < 1e-6);
        assert!((casted_data[1] - (-2.3)).abs() < 1e-6);
        assert_eq!(casted.shape(), &shape);

        let casted_int = tensor.astype(DataType::Int32).unwrap();
        let casted_int_data = casted_int.data().as_i32_slice().unwrap();
        assert_eq!(casted_int_data, &[1, -2]);
        assert_eq!(casted_int.shape(), &shape);

        let casted_bool = tensor.astype(DataType::Bool).unwrap();
        let casted_bool_data = casted_bool.data().as_bool_slice().unwrap();
        assert_eq!(casted_bool_data, &[true, true]);
    }

    #[test]
    fn test_astype_from_bool() {
        let data = vec![true, false, true];
        let shape = Shape::new(vec![3]);
        let tensor = Tensor::new(
            Arc::new(TensorData::from_vec_bool(data.clone(), Device::cpu())),
            shape.clone(),
            DataType::Bool,
            Device::cpu(),
            false,
        );

        let to_float = tensor.astype(DataType::Float32).unwrap();
        assert_eq!(to_float.data().as_f32_slice().unwrap(), &[1.0, 0.0, 1.0]);

        let to_int = tensor.astype(DataType::Int64).unwrap();
        assert_eq!(to_int.data().as_i64_slice().unwrap(), &[1, 0, 1]);
    }

    #[test]
    fn test_add_scalar_broadcasting() {
        let a = Tensor::ones(
            Shape::new(vec![2, 3]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let scalar = Tensor::ones(Shape::scalar(), DataType::Float32, Device::cpu(), false);
        let result = a.add(&scalar).unwrap();
        assert_eq!(result.data().as_f32_slice().unwrap(), &[2.0; 6]);
        assert_eq!(result.shape(), &Shape::new(vec![2, 3]));
    }

    #[test]
    fn test_add_incompatible_shapes_error() {
        let a = Tensor::ones(
            Shape::new(vec![2, 2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let b = Tensor::ones(
            Shape::new(vec![3, 1]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        assert!(a.add(&b).is_err());
    }

    #[test]
    fn test_view_shape_mismatch_error() {
        let shape = Shape::new(vec![2, 2]);
        let data = Arc::new(TensorData::zeros(shape.numel(), DataType::Float32));
        let tensor = Tensor::new(data, shape, DataType::Float32, Device::cpu(), false);
        let bad_shape = Shape::new(vec![3, 1]);
        assert!(tensor.view(bad_shape).is_err());
    }

    #[test]
    fn test_view_rejects_non_contiguous_tensor() {
        let data = TensorData::from_vec_f32(vec![1.0, 2.0, 3.0], Device::cpu());
        let tensor = Tensor::new(
            Arc::new(data),
            Shape::new(vec![1, 3]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let expanded = tensor.expand(vec![4, 3]).unwrap();
        assert!(!expanded.is_contiguous());
        // A raw view would silently pair the new shape with storage that only
        // holds 3 elements; it must be rejected.
        assert!(expanded.view(Shape::new(vec![12])).is_err());
    }

    #[test]
    fn test_contiguous_participates_in_autograd() {
        // `contiguous()` materializes a copy but must keep gradient flow to
        // the source (identity backward). Regression test: the producer-buffer
        // rewrite once dropped the CloneBackward attachment, which only the
        // Python differential suite caught.
        let data = TensorData::from_vec_f32(vec![1.0, 2.0, 3.0], Device::cpu());
        let tensor = Tensor::new(
            Arc::new(data),
            Shape::new(vec![1, 3]),
            DataType::Float32,
            Device::cpu(),
            true,
        );
        let expanded = tensor.expand(vec![2, 3]).unwrap();
        let materialized = expanded.contiguous().unwrap();
        assert!(materialized.requires_grad());
        assert!(materialized.grad_fn().is_some());

        let loss = crate::ops::reduction::sum(&materialized, None, false).unwrap();
        let grads = crate::autograd::backward_collect(&loss, None).unwrap();
        let grad = grads.get(&tensor.id()).expect("source tensor gradient");
        // Each source element is broadcast into 2 rows, so its gradient is 2.
        assert_eq!(grad.data().as_f32_slice().unwrap(), &[2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_contiguous_matches_strided_view_values() {
        // transpose().contiguous() must gather through the view's strides.
        let data = TensorData::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Device::cpu());
        let tensor = Tensor::new(
            Arc::new(data),
            Shape::new(vec![2, 3]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let transposed = tensor.transpose(0, 1).unwrap();
        assert_eq!(
            transposed.data().as_f32_slice().unwrap(),
            &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
    }

    #[test]
    fn test_reshape_materializes_non_contiguous_tensor() {
        let data = TensorData::from_vec_f32(vec![1.0, 2.0, 3.0], Device::cpu());
        let tensor = Tensor::new(
            Arc::new(data),
            Shape::new(vec![1, 3]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let expanded = tensor.expand(vec![4, 3]).unwrap();
        let reshaped = expanded.reshape(Shape::new(vec![12])).unwrap();
        assert_eq!(reshaped.shape().dims(), &[12]);
        // Storage must now really contain all 12 broadcast elements.
        assert_eq!(reshaped.data().numel(), 12);
        assert_eq!(
            reshaped.data().as_f32_slice().unwrap(),
            &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        );

        // The ops-layer reshape must materialise as well.
        let via_op = crate::ops::shape_ops::reshape(&expanded, Shape::new(vec![12])).unwrap();
        assert_eq!(via_op.data().numel(), 12);
        assert_eq!(
            via_op.data().as_f32_slice().unwrap(),
            &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn test_reshape_scalar_to_vector() {
        let scalar = Tensor::ones(Shape::scalar(), DataType::Float32, Device::cpu(), false);
        let reshaped = scalar.reshape(Shape::new(vec![1])).unwrap();
        assert_eq!(reshaped.shape().dims(), &[1]);
        assert_eq!(reshaped.data().as_f32_slice().unwrap(), &[1.0]);
    }

    #[test]
    fn test_transpose_basic() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::new(
            Arc::new(TensorData::from_vec_f32(data, Device::cpu())),
            Shape::new(vec![2, 3]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let transposed = tensor.transpose(0, 1).unwrap();
        assert_eq!(transposed.shape().dims(), &[3, 2]);
        assert_eq!(
            transposed.data().as_f32_slice().unwrap(),
            &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
    }

    #[test]
    fn test_transpose_out_of_bounds() {
        let tensor = Tensor::ones(
            Shape::new(vec![2, 2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        assert!(tensor.transpose(0, 2).is_err());
    }

    #[test]
    fn test_transpose_same_dim_noop() {
        let tensor = Tensor::ones(
            Shape::new(vec![2, 2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let transposed = tensor.transpose(1, 1).unwrap();
        assert_eq!(transposed.data().as_f32_slice().unwrap(), &[1.0; 4]);
        assert_eq!(transposed.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_astype_multiple_conversions() {
        let base = Tensor::new(
            Arc::new(TensorData::from_vec_f32(
                vec![1.5, -2.0, 0.0],
                Device::cpu(),
            )),
            Shape::new(vec![3]),
            DataType::Float32,
            Device::cpu(),
            false,
        );

        let as_i32 = base.astype(DataType::Int32).unwrap();
        assert_eq!(as_i32.data().as_i32_slice().unwrap(), &[1, -2, 0]);

        let as_bool = as_i32.astype(DataType::Bool).unwrap();
        assert_eq!(
            as_bool.data().as_bool_slice().unwrap(),
            &[true, true, false]
        );

        let as_f64 = as_bool.astype(DataType::Float64).unwrap();
        assert_eq!(as_f64.data().as_f64_slice().unwrap(), &[1.0, 1.0, 0.0]);
    }

    #[test]
    fn test_astype_parallel_large_buffer() {
        let size = 2048;
        let data: Vec<f32> = (0..size).map(|v| v as f32).collect();
        let tensor = Tensor::new(
            Arc::new(TensorData::from_vec_f32(data, Device::cpu())),
            Shape::new(vec![size]),
            DataType::Float32,
            Device::cpu(),
            false,
        );

        let converted = tensor.astype(DataType::Int32).unwrap();
        let expected: Vec<i32> = (0..size).map(|v| v as i32).collect();
        assert_eq!(
            converted.data().as_i32_slice().unwrap(),
            expected.as_slice()
        );
    }

    #[test]
    fn test_array_equal_fast_path() {
        let t1 = Tensor::new(
            Arc::new(TensorData::from_vec_f32(vec![1.0, 2.0, 3.0], Device::cpu())),
            Shape::new(vec![3]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let t2 = Tensor::new(
            Arc::new(TensorData::from_vec_f32(vec![1.0, 2.0, 3.0], Device::cpu())),
            Shape::new(vec![3]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        assert!(t1.array_equal(&t2));
        assert!(t1.allclose(&t2, 0.0, 0.0));
    }

    #[test]
    fn test_array_equal_mismatch() {
        let t1 = Tensor::new(
            Arc::new(TensorData::from_vec_f32(vec![1.0, 2.0], Device::cpu())),
            Shape::new(vec![2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let t2 = Tensor::new(
            Arc::new(TensorData::from_vec_f32(vec![1.0, 2.1], Device::cpu())),
            Shape::new(vec![2]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        assert!(!t1.array_equal(&t2));
        assert!(!t1.allclose(&t2, 1e-5, 1e-5));
    }

    #[test]
    fn test_array_equal_zero_sized() {
        let empty1 = Tensor::new(
            Arc::new(TensorData::from_vec_f32(vec![], Device::cpu())),
            Shape::new(vec![0]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let empty2 = Tensor::new(
            Arc::new(TensorData::from_vec_f32(vec![], Device::cpu())),
            Shape::new(vec![0]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        assert!(empty1.array_equal(&empty2));
    }

    #[test]
    fn test_deep_clone_independent_storage() {
        let shape = Shape::new(vec![2, 2]);
        let data = Arc::new(TensorData::from_vec_f32(
            vec![1.0, 2.0, 3.0, 4.0],
            Device::cpu(),
        ));
        let tensor = Tensor::new(data, shape.clone(), DataType::Float32, Device::cpu(), false);

        let mut cloned = tensor.deep_clone().unwrap();
        {
            let slice = cloned.data_mut().as_f32_slice_mut().unwrap();
            slice[0] = 42.0;
        }

        let original_slice = tensor.data().as_f32_slice().unwrap();
        assert_eq!(original_slice, &[1.0, 2.0, 3.0, 4.0]);
        let cloned_slice = cloned.data().as_f32_slice().unwrap();
        assert_eq!(cloned_slice, &[42.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_deep_clone_preserves_gradients() {
        let shape = Shape::new(vec![3]);
        let data = Arc::new(TensorData::from_vec_f32(
            vec![1.0, -2.0, 3.0],
            Device::cpu(),
        ));
        let mut tensor = Tensor::new(data, shape.clone(), DataType::Float32, Device::cpu(), true);
        tensor.zero_grad(true);

        let cloned = tensor.deep_clone().unwrap();
        assert!(cloned.requires_grad());

        let grad = Tensor::new(
            Arc::new(TensorData::from_vec_f32(
                vec![0.5, -1.0, 2.0],
                Device::cpu(),
            )),
            shape,
            DataType::Float32,
            Device::cpu(),
            false,
        );
        cloned.backward(Some(grad.clone())).unwrap();

        let accumulated = autograd::get_gradient(&tensor).expect("gradient should be set");
        assert!(accumulated.allclose(&grad, 1e-6, 1e-6));
    }

    #[test]
    fn test_contiguous_materialises_expanded_views() {
        let base = Tensor::new(
            Arc::new(TensorData::from_vec_f32(vec![1.0, 2.0], Device::cpu())),
            Shape::new(vec![2, 1]),
            DataType::Float32,
            Device::cpu(),
            false,
        );

        let expanded = base
            .expand(vec![2isize, 3isize])
            .expect("expand should succeed");
        assert!(!expanded.is_contiguous());

        let contiguous = expanded.contiguous().expect("contiguous should copy data");
        assert!(contiguous.is_contiguous());
        assert_eq!(contiguous.shape().dims(), &[2, 3]);
        let values = contiguous
            .data()
            .as_f32_slice()
            .expect("materialised data should be accessible")
            .to_vec();
        assert_eq!(values, vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    }
}

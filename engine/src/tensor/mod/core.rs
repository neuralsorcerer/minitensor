// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

pub use super::data::{DataMut, TensorData};
pub use super::dtype::DataType;
pub use super::shape::{Shape, Strides};

// Method impls split by concern. They are children of this module (rather
// than siblings) so they keep access to `Tensor`'s private fields.
#[path = "autograd.rs"]
mod autograd_methods;
#[path = "indexing.rs"]
mod indexing_methods;
#[path = "ops.rs"]
mod ops_methods;
#[path = "utils.rs"]
mod utils_methods;

use crate::{
    autograd::{self, CloneBackward, GradientFunction, TensorId},
    device::Device,
    error::{MinitensorError, Result},
    operations::{arithmetic::add, reduction::QuantileInterpolation},
};
use rayon::prelude::*;
use std::{borrow::Cow, sync::Arc};

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
                    crate::operations::map::strided_gather(src, &shape, &strides),
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
    /// a contiguous copy otherwise.
    #[inline(always)]
    pub fn reshape(&self, new_shape: Shape) -> Result<Self> {
        if self.is_contiguous() {
            self.view(new_shape)
        } else {
            self.contiguous()?.view(new_shape)
        }
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
        use crate::operations::linalg::transpose;
        transpose(self, dim0, dim1)
    }

    /// Permute tensor dimensions
    #[inline(always)]
    pub fn permute(&self, dims: Vec<isize>) -> Result<Self> {
        use crate::operations::shape_ops::permute;
        permute(self, dims)
    }
}

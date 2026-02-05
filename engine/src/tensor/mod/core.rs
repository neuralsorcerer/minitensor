// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

pub use data::TensorData;
pub use dtype::DataType;
pub use shape::{Shape, Strides};

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
            requires_grad,
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

    /// Get a mutable reference to the tensor data
    #[inline(always)]
    pub(crate) fn data_mut(&mut self) -> &mut TensorData {
        let needs_detach = self.grad_fn.is_some() || !self.requires_grad;
        if needs_detach {
            if Arc::get_mut(&mut self.data).is_none() {
                let cloned = self.data.as_ref().clone_data();
                self.data = Arc::new(cloned);
            }
            Arc::get_mut(&mut self.data).expect("Tensor data should be uniquely owned")
        } else {
            let ptr = Arc::as_ptr(&self.data) as *mut TensorData;
            unsafe { &mut *ptr }
        }
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

        let numel = self.numel();
        let dtype = self.dtype;
        let device = self.device;
        let requires_grad = self.requires_grad;
        let shape = self.shape.dims().to_vec();
        let strides = self.strides.as_slice().to_vec();

        let mut output_data = TensorData::uninitialized_on_device(numel, dtype, device);

        match dtype {
            DataType::Float32 => {
                let src = self.data.as_f32_slice().ok_or_else(|| {
                    MinitensorError::invalid_operation(
                        "failed to access float32 data for contiguous copy".to_string(),
                    )
                })?;
                let dst = output_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::invalid_operation(
                        "failed to access float32 storage for contiguous copy".to_string(),
                    )
                })?;
                copy_strided_to_contiguous(src, dst, &shape, &strides);
            }
            DataType::Float64 => {
                let src = self.data.as_f64_slice().ok_or_else(|| {
                    MinitensorError::invalid_operation(
                        "failed to access float64 data for contiguous copy".to_string(),
                    )
                })?;
                let dst = output_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::invalid_operation(
                        "failed to access float64 storage for contiguous copy".to_string(),
                    )
                })?;
                copy_strided_to_contiguous(src, dst, &shape, &strides);
            }
            DataType::Int32 => {
                let src = self.data.as_i32_slice().ok_or_else(|| {
                    MinitensorError::invalid_operation(
                        "failed to access int32 data for contiguous copy".to_string(),
                    )
                })?;
                let dst = output_data.as_i32_slice_mut().ok_or_else(|| {
                    MinitensorError::invalid_operation(
                        "failed to access int32 storage for contiguous copy".to_string(),
                    )
                })?;
                copy_strided_to_contiguous(src, dst, &shape, &strides);
            }
            DataType::Int64 => {
                let src = self.data.as_i64_slice().ok_or_else(|| {
                    MinitensorError::invalid_operation(
                        "failed to access int64 data for contiguous copy".to_string(),
                    )
                })?;
                let dst = output_data.as_i64_slice_mut().ok_or_else(|| {
                    MinitensorError::invalid_operation(
                        "failed to access int64 storage for contiguous copy".to_string(),
                    )
                })?;
                copy_strided_to_contiguous(src, dst, &shape, &strides);
            }
            DataType::Bool => {
                let src = self.data.as_bool_slice().ok_or_else(|| {
                    MinitensorError::invalid_operation(
                        "failed to access bool data for contiguous copy".to_string(),
                    )
                })?;
                let dst = output_data.as_bool_slice_mut().ok_or_else(|| {
                    MinitensorError::invalid_operation(
                        "failed to access bool storage for contiguous copy".to_string(),
                    )
                })?;
                copy_strided_to_contiguous(src, dst, &shape, &strides);
            }
        }

        let mut output = Tensor::new(
            Arc::new(output_data),
            self.shape.clone(),
            dtype,
            device,
            requires_grad,
        );

        if requires_grad {
            let grad_fn = Arc::new(CloneBackward {
                input_id: self.tensor_id,
            });
            output.set_grad_fn(Some(grad_fn.clone()));
            autograd::add_to_graph(&output, Some(grad_fn))?;
        }

        Ok(output)
    }
}

impl Tensor {
    /// Set the gradient function for this tensor
    #[inline(always)]
    pub fn set_grad_fn(&mut self, grad_fn: Option<Arc<dyn GradientFunction>>) {
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

    /// Clear the gradient for this tensor
    #[inline(always)]
    pub fn zero_grad(&mut self, set_to_none: bool) {
        autograd::zero_gradients();
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
        autograd::backward(self, Some(grad)).map(|_| ()) // Convert HashMap result to ()
    }
}

impl Tensor {
    /// Create a view of this tensor with a new shape
    #[inline(always)]
    pub fn view(&self, new_shape: Shape) -> Result<Self> {
        if new_shape.numel() != self.numel() {
            return Err(MinitensorError::shape_mismatch(
                vec![self.numel()],
                vec![new_shape.numel()],
            ));
        }

        let mut tensor = self.clone();
        tensor.strides = Strides::from_shape(&new_shape);
        tensor.shape = new_shape;
        Ok(tensor)
    }

    /// Reshape the tensor to a new shape
    #[inline(always)]
    pub fn reshape(&self, new_shape: Shape) -> Result<Self> {
        self.view(new_shape)
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

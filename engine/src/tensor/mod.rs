// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

pub mod data;
pub mod dtype;
pub mod shape;

pub use data::TensorData;
pub use dtype::DataType;
pub use shape::{Shape, Strides};

use crate::{
    autograd::{self, CloneBackward, GradientFunction, TensorId},
    device::Device,
    error::{MinitensorError, Result},
    operations::arithmetic::add,
};
use rayon::prelude::*;
use std::sync::Arc;

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

    /// Unary negation
    #[inline(always)]
    pub fn neg(&self) -> Result<Self> {
        use crate::operations::arithmetic::neg;
        neg(self)
    }

    /// Add two tensors element-wise
    #[inline(always)]
    pub fn add(&self, other: &Tensor) -> Result<Self> {
        use crate::operations::arithmetic::add;
        add(self, other)
    }

    /// Element-wise maximum
    #[inline(always)]
    pub fn maximum(&self, other: &Tensor) -> Result<Self> {
        use crate::operations::minmax::maximum;
        maximum(self, other)
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn minimum(&self, other: &Tensor) -> Result<Self> {
        use crate::operations::minmax::minimum;
        minimum(self, other)
    }

    /// Select elements from self or other based on a boolean condition tensor
    #[inline(always)]
    pub fn where_select(&self, condition: &Tensor, other: &Tensor) -> Result<Self> {
        use crate::operations::selection::where_op;
        where_op(condition, self, other)
    }

    /// Fill elements specified by `mask` with values from `value`.
    #[inline(always)]
    pub fn masked_fill(&self, mask: &Tensor, value: &Tensor) -> Result<Self> {
        crate::operations::selection::masked_fill(self, mask, value)
    }

    /// Fill elements specified by `mask` with a scalar.
    #[inline(always)]
    pub fn masked_fill_scalar(&self, mask: &Tensor, value: f64) -> Result<Self> {
        crate::operations::selection::masked_fill_scalar(self, mask, value)
    }

    /// Dot product between two 1D tensors
    #[inline(always)]
    pub fn dot(&self, other: &Tensor) -> Result<Self> {
        crate::operations::linalg::dot(self, other)
    }

    /// Matrix multiplication
    #[inline(always)]
    pub fn matmul(&self, other: &Tensor) -> Result<Self> {
        use crate::operations::linalg::matmul;
        matmul(self, other)
    }

    /// Upper triangular part of the tensor's last two dimensions
    #[inline(always)]
    pub fn triu(&self, diagonal: i64) -> Result<Self> {
        use crate::operations::linalg::triu;
        triu(self, diagonal)
    }

    /// Lower triangular part of the tensor's last two dimensions
    #[inline(always)]
    pub fn tril(&self, diagonal: i64) -> Result<Self> {
        use crate::operations::linalg::tril;
        tril(self, diagonal)
    }

    /// Extract a diagonal along two dimensions.
    #[inline(always)]
    pub fn diagonal(&self, offset: isize, dim1: isize, dim2: isize) -> Result<Self> {
        use crate::operations::linalg::diagonal;
        diagonal(self, offset, dim1, dim2)
    }

    /// Sum the diagonal elements along two dimensions.
    #[inline(always)]
    pub fn trace(&self, offset: isize, dim1: isize, dim2: isize) -> Result<Self> {
        use crate::operations::linalg::trace;
        trace(self, offset, dim1, dim2)
    }

    /// Sum reduction
    #[inline(always)]
    pub fn sum(&self, dim: Option<Vec<isize>>, keepdim: bool) -> Result<Self> {
        use crate::operations::reduction::sum;
        sum(self, dim, keepdim)
    }

    /// Log-sum-exp reduction
    #[inline(always)]
    pub fn logsumexp(&self, dim: Option<Vec<isize>>, keepdim: bool) -> Result<Self> {
        use crate::operations::reduction::logsumexp;
        logsumexp(self, dim, keepdim)
    }

    /// Product reduction
    #[inline(always)]
    pub fn prod(&self, dim: Option<Vec<isize>>, keepdim: bool) -> Result<Self> {
        use crate::operations::reduction::prod;
        prod(self, dim, keepdim)
    }

    /// Mean reduction
    #[inline(always)]
    pub fn mean(&self, dim: Option<Vec<isize>>, keepdim: bool) -> Result<Self> {
        use crate::operations::reduction::mean;
        mean(self, dim, keepdim)
    }

    /// Logical all reduction
    #[inline(always)]
    pub fn all(&self, dim: Option<isize>, keepdim: bool) -> Result<Self> {
        use crate::operations::reduction::all;
        all(self, dim, keepdim)
    }

    /// Logical any reduction
    #[inline(always)]
    pub fn any(&self, dim: Option<isize>, keepdim: bool) -> Result<Self> {
        use crate::operations::reduction::any;
        any(self, dim, keepdim)
    }

    /// Cumulative sum along a dimension
    #[inline(always)]
    pub fn cumsum(&self, dim: isize) -> Result<Self> {
        use crate::operations::reduction::cumsum;
        cumsum(self, dim)
    }

    /// Cumulative product along a dimension
    #[inline(always)]
    pub fn cumprod(&self, dim: isize) -> Result<Self> {
        use crate::operations::reduction::cumprod;
        cumprod(self, dim)
    }

    /// Maximum value
    #[inline(always)]
    pub fn max(&self, dim: Option<isize>, keepdim: bool) -> Result<Self> {
        use crate::operations::reduction::max;
        max(self, dim, keepdim)
    }

    /// Minimum value
    #[inline(always)]
    pub fn min(&self, dim: Option<isize>, keepdim: bool) -> Result<Self> {
        use crate::operations::reduction::min;
        min(self, dim, keepdim)
    }

    /// Argument of maximum value
    #[inline(always)]
    pub fn argmax(&self, dim: Option<isize>, keepdim: bool) -> Result<Self> {
        use crate::operations::reduction::argmax;
        argmax(self, dim, keepdim)
    }

    /// Argument of minimum value
    #[inline(always)]
    pub fn argmin(&self, dim: Option<isize>, keepdim: bool) -> Result<Self> {
        use crate::operations::reduction::argmin;
        argmin(self, dim, keepdim)
    }

    /// Median value (optionally along a dimension)
    #[inline(always)]
    pub fn median(&self, dim: Option<isize>, keepdim: bool) -> Result<(Self, Option<Self>)> {
        use crate::operations::reduction::median;
        median(self, dim, keepdim)
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
        use crate::operations::reduction::topk;
        topk(self, k, dim, largest, sorted)
    }

    /// Sort tensor values along a dimension
    #[inline(always)]
    pub fn sort(&self, dim: Option<isize>, descending: bool, stable: bool) -> Result<(Self, Self)> {
        use crate::operations::reduction::sort;
        sort(self, dim, descending, stable)
    }

    /// Indices that would sort the tensor along a dimension
    #[inline(always)]
    pub fn argsort(&self, dim: Option<isize>, descending: bool, stable: bool) -> Result<Self> {
        use crate::operations::reduction::argsort;
        argsort(self, dim, descending, stable)
    }

    /// Element-wise equality comparison
    #[inline(always)]
    pub fn eq(&self, other: &Tensor) -> Result<Self> {
        use crate::operations::comparison::eq;
        eq(self, other)
    }

    /// Element-wise inequality comparison
    pub fn ne(&self, other: &Tensor) -> Result<Self> {
        use crate::operations::comparison::ne;
        ne(self, other)
    }

    /// Element-wise less-than comparison
    #[inline(always)]
    pub fn lt(&self, other: &Tensor) -> Result<Self> {
        use crate::operations::comparison::lt;
        lt(self, other)
    }

    /// Element-wise less-than-or-equal comparison
    #[inline(always)]
    pub fn le(&self, other: &Tensor) -> Result<Self> {
        use crate::operations::comparison::le;
        le(self, other)
    }

    /// Element-wise greater-than comparison
    #[inline(always)]
    pub fn gt(&self, other: &Tensor) -> Result<Self> {
        use crate::operations::comparison::gt;
        gt(self, other)
    }

    /// Element-wise greater-than-or-equal comparison
    #[inline(always)]
    pub fn ge(&self, other: &Tensor) -> Result<Self> {
        use crate::operations::comparison::ge;
        ge(self, other)
    }

    /// Standard deviation
    #[inline(always)]
    pub fn std(&self, dim: Option<isize>, keepdim: bool, unbiased: bool) -> Result<Self> {
        use crate::operations::reduction::std;
        std(self, dim, keepdim, unbiased)
    }

    /// Variance
    #[inline(always)]
    pub fn var(&self, dim: Option<isize>, keepdim: bool, unbiased: bool) -> Result<Self> {
        use crate::operations::reduction::var;
        var(self, dim, keepdim, unbiased)
    }

    /// Exponential function
    #[inline(always)]
    pub fn exp(&self) -> Result<Self> {
        use crate::operations::activation::exp;
        exp(self)
    }

    /// Natural logarithm
    #[inline(always)]
    pub fn log(&self) -> Result<Self> {
        use crate::operations::activation::log;
        log(self)
    }

    /// log1p (log(1 + x))
    #[inline(always)]
    pub fn log1p(&self) -> Result<Self> {
        use crate::operations::activation::log1p;
        log1p(self)
    }

    /// expm1 (exp(x) - 1)
    #[inline(always)]
    pub fn expm1(&self) -> Result<Self> {
        use crate::operations::activation::expm1;
        expm1(self)
    }

    /// Sine function
    #[inline(always)]
    pub fn sin(&self) -> Result<Self> {
        use crate::operations::activation::sin;
        sin(self)
    }

    /// Cosine function
    #[inline(always)]
    pub fn cos(&self) -> Result<Self> {
        use crate::operations::activation::cos;
        cos(self)
    }

    /// Tangent function
    #[inline(always)]
    pub fn tan(&self) -> Result<Self> {
        use crate::operations::activation::tan;
        tan(self)
    }

    /// Hyperbolic tangent
    #[inline(always)]
    pub fn tanh(&self) -> Result<Self> {
        use crate::operations::activation::tanh;
        tanh(self)
    }

    /// Sigmoid activation
    #[inline(always)]
    pub fn sigmoid(&self) -> Result<Self> {
        use crate::operations::activation::sigmoid;
        sigmoid(self)
    }

    /// Softplus activation
    #[inline(always)]
    pub fn softplus(&self, beta: f64, threshold: f64) -> Result<Self> {
        use crate::operations::activation::softplus;
        softplus(self, beta, threshold)
    }

    /// GELU activation
    #[inline(always)]
    pub fn gelu(&self, approximate: bool) -> Result<Self> {
        use crate::operations::activation::gelu;
        gelu(self, approximate)
    }

    /// ELU activation
    #[inline(always)]
    pub fn elu(&self, alpha: f64) -> Result<Self> {
        use crate::operations::activation::elu;
        elu(self, alpha)
    }

    /// SELU activation
    #[inline(always)]
    pub fn selu(&self) -> Result<Self> {
        use crate::operations::activation::selu;
        selu(self)
    }

    /// SiLU activation
    #[inline(always)]
    pub fn silu(&self) -> Result<Self> {
        use crate::operations::activation::silu;
        silu(self)
    }

    /// Softsign activation
    #[inline(always)]
    pub fn softsign(&self) -> Result<Self> {
        use crate::operations::activation::softsign;
        softsign(self)
    }

    /// ReLU activation
    #[inline(always)]
    pub fn relu(&self) -> Result<Self> {
        use crate::operations::activation::relu;
        relu(self)
    }

    /// Hardshrink activation
    #[inline(always)]
    pub fn hardshrink(&self, lambd: f64) -> Result<Self> {
        use crate::operations::activation::hardshrink;
        hardshrink(self, lambd)
    }

    /// Softmax activation
    #[inline(always)]
    pub fn softmax(&self, dim: Option<usize>) -> Result<Self> {
        use crate::operations::activation::softmax;
        softmax(self, dim)
    }

    /// Log-Softmax activation
    #[inline(always)]
    pub fn log_softmax(&self, dim: Option<usize>) -> Result<Self> {
        use crate::operations::activation::log_softmax;
        log_softmax(self, dim)
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
        use crate::operations::normalization::layer_norm;
        layer_norm(self, normalized_shape, weight, bias, eps)
    }

    /// Absolute value
    #[inline(always)]
    pub fn abs(&self) -> Result<Self> {
        use crate::operations::activation::abs;
        abs(self)
    }

    /// Square root
    #[inline(always)]
    pub fn sqrt(&self) -> Result<Self> {
        use crate::operations::activation::sqrt;
        sqrt(self)
    }

    pub fn rsqrt(&self) -> Result<Self> {
        use crate::operations::activation::rsqrt;
        rsqrt(self)
    }

    /// Raise tensor elements to a scalar power
    #[inline(always)]
    pub fn powf(&self, exponent: f64) -> Result<Self> {
        use crate::operations::activation::powf;
        powf(self, exponent)
    }

    /// Numerically stable logaddexp
    #[inline(always)]
    pub fn logaddexp(&self, other: &Tensor) -> Result<Self> {
        use crate::operations::activation::logaddexp;
        logaddexp(self, other)
    }

    /// Element-wise power with another tensor
    pub fn pow(&self, exponent: &Tensor) -> Result<Self> {
        use crate::operations::activation::pow;
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

        let numel = self.numel();
        let mut new_data = TensorData::zeros_on_device(numel, dtype, self.device);

        // Helper macro to cast between slices using parallel iteration for large buffers
        macro_rules! cast {
            ($src:expr, $dst:expr, $conv:expr) => {{
                if numel >= 1024 {
                    $dst.par_iter_mut().zip($src.par_iter()).for_each(|(d, s)| {
                        *d = $conv(*s);
                    });
                } else {
                    for (d, &s) in $dst.iter_mut().zip($src.iter()) {
                        *d = $conv(s);
                    }
                }
            }};
        }

        match (self.dtype, dtype) {
            (DataType::Float32, DataType::Float64) => {
                let src = self.data.as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from tensor data")
                })?;
                let dst = new_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice from tensor data",
                    )
                })?;
                cast!(src, dst, |v: f32| v as f64);
            }
            (DataType::Float32, DataType::Int32) => {
                let src = self.data.as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from tensor data")
                })?;
                let dst = new_data.as_i32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable i32 slice from tensor data",
                    )
                })?;
                cast!(src, dst, |v: f32| v as i32);
            }
            (DataType::Float32, DataType::Int64) => {
                let src = self.data.as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from tensor data")
                })?;
                let dst = new_data.as_i64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable i64 slice from tensor data",
                    )
                })?;
                cast!(src, dst, |v: f32| v as i64);
            }
            (DataType::Float32, DataType::Bool) => {
                let src = self.data.as_f32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f32 slice from tensor data")
                })?;
                let dst = new_data.as_bool_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable bool slice from tensor data",
                    )
                })?;
                cast!(src, dst, |v: f32| v != 0.0);
            }
            (DataType::Float64, DataType::Float32) => {
                let src = self.data.as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from tensor data")
                })?;
                let dst = new_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice from tensor data",
                    )
                })?;
                cast!(src, dst, |v: f64| v as f32);
            }
            (DataType::Float64, DataType::Int32) => {
                let src = self.data.as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from tensor data")
                })?;
                let dst = new_data.as_i32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable i32 slice from tensor data",
                    )
                })?;
                cast!(src, dst, |v: f64| v as i32);
            }
            (DataType::Float64, DataType::Int64) => {
                let src = self.data.as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from tensor data")
                })?;
                let dst = new_data.as_i64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable i64 slice from tensor data",
                    )
                })?;
                cast!(src, dst, |v: f64| v as i64);
            }
            (DataType::Float64, DataType::Bool) => {
                let src = self.data.as_f64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get f64 slice from tensor data")
                })?;
                let dst = new_data.as_bool_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable bool slice from tensor data",
                    )
                })?;
                cast!(src, dst, |v: f64| v != 0.0);
            }
            (DataType::Int32, DataType::Float32) => {
                let src = self.data.as_i32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get i32 slice from tensor data")
                })?;
                let dst = new_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice from tensor data",
                    )
                })?;
                cast!(src, dst, |v: i32| v as f32);
            }
            (DataType::Int32, DataType::Float64) => {
                let src = self.data.as_i32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get i32 slice from tensor data")
                })?;
                let dst = new_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice from tensor data",
                    )
                })?;
                cast!(src, dst, |v: i32| v as f64);
            }
            (DataType::Int32, DataType::Int64) => {
                let src = self.data.as_i32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get i32 slice from tensor data")
                })?;
                let dst = new_data.as_i64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable i64 slice from tensor data",
                    )
                })?;
                cast!(src, dst, |v: i32| v as i64);
            }
            (DataType::Int32, DataType::Bool) => {
                let src = self.data.as_i32_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get i32 slice from tensor data")
                })?;
                let dst = new_data.as_bool_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable bool slice from tensor data",
                    )
                })?;
                cast!(src, dst, |v: i32| v != 0);
            }
            (DataType::Int64, DataType::Float32) => {
                let src = self.data.as_i64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get i64 slice from tensor data")
                })?;
                let dst = new_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice from tensor data",
                    )
                })?;
                cast!(src, dst, |v: i64| v as f32);
            }
            (DataType::Int64, DataType::Float64) => {
                let src = self.data.as_i64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get i64 slice from tensor data")
                })?;
                let dst = new_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice from tensor data",
                    )
                })?;
                cast!(src, dst, |v: i64| v as f64);
            }
            (DataType::Int64, DataType::Int32) => {
                let src = self.data.as_i64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get i64 slice from tensor data")
                })?;
                let dst = new_data.as_i32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable i32 slice from tensor data",
                    )
                })?;
                cast!(src, dst, |v: i64| v as i32);
            }
            (DataType::Int64, DataType::Bool) => {
                let src = self.data.as_i64_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get i64 slice from tensor data")
                })?;
                let dst = new_data.as_bool_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable bool slice from tensor data",
                    )
                })?;
                cast!(src, dst, |v: i64| v != 0);
            }
            (DataType::Bool, DataType::Float32) => {
                let src = self.data.as_bool_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get bool slice from tensor data")
                })?;
                let dst = new_data.as_f32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f32 slice from tensor data",
                    )
                })?;
                cast!(src, dst, |v: bool| if v { 1.0 } else { 0.0 });
            }
            (DataType::Bool, DataType::Float64) => {
                let src = self.data.as_bool_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get bool slice from tensor data")
                })?;
                let dst = new_data.as_f64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable f64 slice from tensor data",
                    )
                })?;
                cast!(src, dst, |v: bool| if v { 1.0 } else { 0.0 });
            }
            (DataType::Bool, DataType::Int32) => {
                let src = self.data.as_bool_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get bool slice from tensor data")
                })?;
                let dst = new_data.as_i32_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable i32 slice from tensor data",
                    )
                })?;
                cast!(src, dst, |v: bool| if v { 1 } else { 0 });
            }
            (DataType::Bool, DataType::Int64) => {
                let src = self.data.as_bool_slice().ok_or_else(|| {
                    MinitensorError::internal_error("Failed to get bool slice from tensor data")
                })?;
                let dst = new_data.as_i64_slice_mut().ok_or_else(|| {
                    MinitensorError::internal_error(
                        "Failed to get mutable i64 slice from tensor data",
                    )
                })?;
                cast!(src, dst, |v: bool| if v { 1 } else { 0 });
            }
            _ => unreachable!("Unhandled dtype conversion"),
        }

        Ok(Tensor::new(
            Arc::new(new_data),
            self.shape.clone(),
            dtype,
            self.device,
            self.requires_grad,
        ))
    }

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
        if self.shape != other.shape || self.dtype != other.dtype {
            return false;
        }

        // Fast path: byte-for-byte equality check for contiguous CPU tensors
        if self.device.is_cpu()
            && other.device.is_cpu()
            && self.is_contiguous()
            && other.is_contiguous()
        {
            if let (Some(a), Some(b)) = (self.data.as_bytes(), other.data.as_bytes()) {
                if a == b {
                    return true;
                }
            }
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
                            .all(|(&a, &b)| {
                                let diff = (a - b).abs();
                                diff <= atol as f32 + rtol as f32 * b.abs()
                            })
                    } else {
                        self_data.iter().zip(other_data.iter()).all(|(&a, &b)| {
                            let diff = (a - b).abs();
                            diff <= atol as f32 + rtol as f32 * b.abs()
                        })
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
                            .all(|(&a, &b)| {
                                let diff = (a - b).abs();
                                diff <= atol + rtol * b.abs()
                            })
                    } else {
                        self_data.iter().zip(other_data.iter()).all(|(&a, &b)| {
                            let diff = (a - b).abs();
                            diff <= atol + rtol * b.abs()
                        })
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
        {
            if let (Some(a), Some(b)) = (self.data.as_bytes(), other.data.as_bytes()) {
                return a == b;
            }
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
        self.view(new_shape)
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
        self.view(new_shape)
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
        self.view(new_shape)
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
                if target != 1 {
                    return Err(MinitensorError::invalid_operation(
                        "cannot introduce new leading dimension".to_string(),
                    ));
                }
                new_dims[n_new - 1 - i] = 1;
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
        crate::operations::shape_ops::repeat(self, &repeats)
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
        self.view(new_shape)
    }

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
            return Ok(Tensor::new(
                Arc::new(result_data),
                Shape::scalar(),
                self.dtype,
                self.device,
                self.requires_grad,
            ));
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

        Ok(Tensor::new(
            Arc::new(result_data),
            out_shape,
            self.dtype,
            self.device,
            self.requires_grad,
        ))
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

    /// Clamp tensor values between optional minimum and maximum
    #[inline(always)]
    pub fn clamp(&self, min: Option<f64>, max: Option<f64>) -> Result<Tensor> {
        if min.is_none() && max.is_none() {
            return Ok(self.clone());
        }

        let output = match self.dtype {
            DataType::Float32 => {
                let input = self
                    .data
                    .as_f32_slice()
                    .ok_or_else(|| MinitensorError::internal_error("Expected f32 data"))?;
                let mut vec = Vec::with_capacity(input.len());
                for &x in input {
                    let mut v = x;
                    if let Some(mn) = min {
                        v = v.max(mn as f32);
                    }
                    if let Some(mx) = max {
                        v = v.min(mx as f32);
                    }
                    vec.push(v);
                }
                TensorData::from_vec_f32(vec, self.device)
            }
            DataType::Float64 => {
                let input = self
                    .data
                    .as_f64_slice()
                    .ok_or_else(|| MinitensorError::internal_error("Expected f64 data"))?;
                let mut vec = Vec::with_capacity(input.len());
                for &x in input {
                    let mut v = x;
                    if let Some(mn) = min {
                        v = v.max(mn);
                    }
                    if let Some(mx) = max {
                        v = v.min(mx);
                    }
                    vec.push(v);
                }
                TensorData::from_vec_f64(vec, self.device)
            }
            DataType::Int32 => {
                let input = self
                    .data
                    .as_i32_slice()
                    .ok_or_else(|| MinitensorError::internal_error("Expected i32 data"))?;
                let mut vec = Vec::with_capacity(input.len());
                for &x in input {
                    let mut v = x;
                    if let Some(mn) = min {
                        v = v.max(mn as i32);
                    }
                    if let Some(mx) = max {
                        v = v.min(mx as i32);
                    }
                    vec.push(v);
                }
                TensorData::from_vec_i32(vec, self.device)
            }
            DataType::Int64 => {
                let input = self
                    .data
                    .as_i64_slice()
                    .ok_or_else(|| MinitensorError::internal_error("Expected i64 data"))?;
                let mut vec = Vec::with_capacity(input.len());
                for &x in input {
                    let mut v = x;
                    if let Some(mn) = min {
                        v = v.max(mn as i64);
                    }
                    if let Some(mx) = max {
                        v = v.min(mx as i64);
                    }
                    vec.push(v);
                }
                TensorData::from_vec_i64(vec, self.device)
            }
            DataType::Bool => {
                // Boolean values are already 0 or 1; clamping does nothing
                return Ok(self.clone());
            }
        };

        Ok(Tensor::new(
            Arc::new(output),
            self.shape.clone(),
            self.dtype,
            self.device,
            self.requires_grad,
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

fn copy_strided_to_contiguous<T: Copy>(
    src: &[T],
    dst: &mut [T],
    shape: &[usize],
    strides: &[usize],
) {
    if dst.is_empty() {
        return;
    }

    if shape.is_empty() {
        dst[0] = src[0];
        return;
    }

    let ndim = shape.len();
    let mut index = vec![0usize; ndim];

    for value in dst.iter_mut() {
        let mut offset = 0usize;
        for (&idx, &stride) in index.iter().zip(strides.iter()) {
            offset += idx * stride;
        }
        *value = src[offset];

        for dim in (0..ndim).rev() {
            index[dim] += 1;
            if index[dim] < shape[dim] {
                break;
            }
            index[dim] = 0;
        }
    }
}

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
    use crate::tensor::data::TensorData;

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

// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

pub mod data;
pub mod dtype;
pub mod shape;

pub use data::TensorData;
pub use dtype::DataType;
pub use shape::{Shape, Strides};

use crate::{
    autograd::{GradientFunction, TensorId},
    device::Device,
    error::{MinitensorError, Result},
    operations::arithmetic::add,
};
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

    /// Create a tensor filled with zeros
    pub fn zeros(shape: Shape, dtype: DataType, device: Device, requires_grad: bool) -> Self {
        let data = Arc::new(TensorData::zeros_on_device(shape.numel(), dtype, device));
        Self::new(data, shape, dtype, device, requires_grad)
    }

    /// Create a tensor filled with ones
    pub fn ones(shape: Shape, dtype: DataType, device: Device, requires_grad: bool) -> Self {
        let data = Arc::new(TensorData::ones_on_device(shape.numel(), dtype, device));
        Self::new(data, shape, dtype, device, requires_grad)
    }

    /// Get the tensor's shape
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Get the tensor's strides
    pub fn strides(&self) -> &Strides {
        &self.strides
    }

    /// Get the tensor's data type
    pub fn dtype(&self) -> DataType {
        self.dtype
    }

    /// Get the tensor's device
    pub fn device(&self) -> Device {
        self.device
    }

    /// Check if this tensor requires gradients
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Get the tensor's unique ID
    pub fn id(&self) -> TensorId {
        self.tensor_id
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }

    /// Get the total number of elements
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// Get the size of a specific dimension
    pub fn size(&self, dim: usize) -> Result<usize> {
        self.shape.size(dim)
    }

    /// Check if the tensor is contiguous in memory
    pub fn is_contiguous(&self) -> bool {
        self.strides.is_contiguous(&self.shape)
    }

    /// Get a reference to the tensor data
    pub fn data(&self) -> &Arc<TensorData> {
        &self.data
    }

    /// Set the gradient function for this tensor
    pub fn set_grad_fn(&mut self, grad_fn: Option<Arc<dyn GradientFunction>>) {
        self.grad_fn = grad_fn;
    }

    /// Get the gradient function for this tensor
    pub fn grad_fn(&self) -> Option<&Arc<dyn GradientFunction>> {
        self.grad_fn.as_ref()
    }

    /// Enable gradient computation for this tensor
    pub fn requires_grad_(mut self, requires_grad: bool) -> Self {
        self.requires_grad = requires_grad;
        self
    }

    /// Get the gradient for this tensor
    pub fn grad(&self) -> Option<&Arc<Tensor>> {
        self.grad.as_ref()
    }

    /// Set the gradient for this tensor
    pub fn set_grad(&mut self, grad: Option<Tensor>) {
        self.grad = grad.map(Arc::new);
    }

    /// Accumulate gradient for this tensor
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
    pub fn zero_grad(&mut self) {
        self.grad = None;
    }

    /// Check if this tensor has a gradient
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
    pub fn reshape(&self, new_shape: Shape) -> Result<Self> {
        self.view(new_shape)
    }

    /// Transpose two dimensions of the tensor
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Self> {
        use crate::operations::linalg::transpose;
        transpose(self, dim0, dim1)
    }

    /// Add two tensors element-wise
    pub fn add(&self, other: &Tensor) -> Result<Self> {
        use crate::operations::arithmetic::add;
        add(self, other)
    }

    /// Matrix multiplication
    pub fn matmul(&self, other: &Tensor) -> Result<Self> {
        use crate::operations::linalg::matmul;
        matmul(self, other)
    }

    /// Sum reduction
    pub fn sum(&self, dim: Option<Vec<usize>>, keepdim: bool) -> Result<Self> {
        use crate::operations::reduction::sum;
        sum(self, dim, keepdim)
    }

    /// Mean reduction
    pub fn mean(&self, dim: Option<Vec<usize>>, keepdim: bool) -> Result<Self> {
        use crate::operations::reduction::mean;
        mean(self, dim, keepdim)
    }

    /// Logical all reduction
    pub fn all(&self, dim: Option<usize>, keepdim: bool) -> Result<Self> {
        use crate::operations::reduction::all;
        all(self, dim, keepdim)
    }

    /// Logical any reduction
    pub fn any(&self, dim: Option<usize>, keepdim: bool) -> Result<Self> {
        use crate::operations::reduction::any;
        any(self, dim, keepdim)
    }

    /// Maximum value
    pub fn max(&self, dim: Option<usize>, keepdim: bool) -> Result<Self> {
        use crate::operations::reduction::max;
        max(self, dim, keepdim)
    }

    /// Minimum value
    pub fn min(&self, dim: Option<usize>, keepdim: bool) -> Result<Self> {
        use crate::operations::reduction::min;
        min(self, dim, keepdim)
    }

    /// Argument of maximum value
    pub fn argmax(&self, dim: Option<usize>, keepdim: bool) -> Result<Self> {
        use crate::operations::reduction::argmax;
        argmax(self, dim, keepdim)
    }

    /// Argument of minimum value
    pub fn argmin(&self, dim: Option<usize>, keepdim: bool) -> Result<Self> {
        use crate::operations::reduction::argmin;
        argmin(self, dim, keepdim)
    }

    /// Element-wise equality comparison
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
    pub fn lt(&self, other: &Tensor) -> Result<Self> {
        use crate::operations::comparison::lt;
        lt(self, other)
    }

    /// Element-wise less-than-or-equal comparison
    pub fn le(&self, other: &Tensor) -> Result<Self> {
        use crate::operations::comparison::le;
        le(self, other)
    }

    /// Element-wise greater-than comparison
    pub fn gt(&self, other: &Tensor) -> Result<Self> {
        use crate::operations::comparison::gt;
        gt(self, other)
    }

    /// Element-wise greater-than-or-equal comparison
    pub fn ge(&self, other: &Tensor) -> Result<Self> {
        use crate::operations::comparison::ge;
        ge(self, other)
    }

    /// Standard deviation
    pub fn std(&self, dim: Option<usize>, keepdim: bool) -> Result<Self> {
        use crate::operations::reduction::std;
        std(self, dim, keepdim)
    }

    /// Variance
    pub fn var(&self, dim: Option<usize>, keepdim: bool) -> Result<Self> {
        use crate::operations::reduction::var;
        var(self, dim, keepdim)
    }

    /// Exponential function
    pub fn exp(&self) -> Result<Self> {
        use crate::operations::activation::exp;
        exp(self)
    }

    /// Natural logarithm
    pub fn log(&self) -> Result<Self> {
        use crate::operations::activation::log;
        log(self)
    }

    /// Sine function
    pub fn sin(&self) -> Result<Self> {
        use crate::operations::activation::sin;
        sin(self)
    }

    /// Cosine function
    pub fn cos(&self) -> Result<Self> {
        use crate::operations::activation::cos;
        cos(self)
    }

    /// Tangent function
    pub fn tan(&self) -> Result<Self> {
        use crate::operations::activation::tan;
        tan(self)
    }

    /// Hyperbolic tangent
    pub fn tanh(&self) -> Result<Self> {
        use crate::operations::activation::tanh;
        tanh(self)
    }

    /// Sigmoid activation
    pub fn sigmoid(&self) -> Result<Self> {
        use crate::operations::activation::sigmoid;
        sigmoid(self)
    }

    /// ReLU activation
    pub fn relu(&self) -> Result<Self> {
        use crate::operations::activation::relu;
        relu(self)
    }

    /// Softmax activation
    pub fn softmax(&self, dim: Option<usize>) -> Result<Self> {
        use crate::operations::activation::softmax;
        softmax(self, dim)
    }

    /// Absolute value
    pub fn abs(&self) -> Result<Self> {
        use crate::operations::activation::abs;
        abs(self)
    }

    /// Square root
    pub fn sqrt(&self) -> Result<Self> {
        use crate::operations::activation::sqrt;
        sqrt(self)
    }

    /// Raise tensor elements to a scalar power
    pub fn powf(&self, exponent: f64) -> Result<Self> {
        use crate::operations::activation::powf;
        powf(self, exponent)
    }

    /// Element-wise power with another tensor
    pub fn pow(&self, exponent: &Tensor) -> Result<Self> {
        use crate::operations::activation::pow;
        pow(self, exponent)
    }

    /// Move tensor to device
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

    /// Detach tensor from computation graph
    pub fn detach(&self) -> Self {
        let mut detached = self.clone();
        detached.requires_grad = false;
        detached.grad_fn = None;
        detached.grad = None;
        detached
    }

    /// Check if tensors are approximately equal
    pub fn allclose(&self, other: &Tensor, rtol: f64, atol: f64) -> bool {
        if self.shape != other.shape || self.dtype != other.dtype {
            return false;
        }

        match self.dtype {
            DataType::Float32 => {
                if let (Some(self_data), Some(other_data)) =
                    (self.data.as_f32_slice(), other.data.as_f32_slice())
                {
                    self_data.iter().zip(other_data.iter()).all(|(&a, &b)| {
                        let diff = (a - b).abs();
                        diff <= atol as f32 + rtol as f32 * b.abs()
                    })
                } else {
                    false
                }
            }
            DataType::Float64 => {
                if let (Some(self_data), Some(other_data)) =
                    (self.data.as_f64_slice(), other.data.as_f64_slice())
                {
                    self_data.iter().zip(other_data.iter()).all(|(&a, &b)| {
                        let diff = (a - b).abs();
                        diff <= atol + rtol * b.abs()
                    })
                } else {
                    false
                }
            }
            _ => self.array_equal(other),
        }
    }

    /// Check if tensors are exactly equal
    pub fn array_equal(&self, other: &Tensor) -> bool {
        if self.shape != other.shape || self.dtype != other.dtype {
            return false;
        }

        match self.dtype {
            DataType::Float32 => {
                if let (Some(self_data), Some(other_data)) =
                    (self.data.as_f32_slice(), other.data.as_f32_slice())
                {
                    self_data == other_data
                } else {
                    false
                }
            }
            DataType::Float64 => {
                if let (Some(self_data), Some(other_data)) =
                    (self.data.as_f64_slice(), other.data.as_f64_slice())
                {
                    self_data == other_data
                } else {
                    false
                }
            }
            DataType::Int32 => {
                if let (Some(self_data), Some(other_data)) =
                    (self.data.as_i32_slice(), other.data.as_i32_slice())
                {
                    self_data == other_data
                } else {
                    false
                }
            }
            DataType::Int64 => {
                if let (Some(self_data), Some(other_data)) =
                    (self.data.as_i64_slice(), other.data.as_i64_slice())
                {
                    self_data == other_data
                } else {
                    false
                }
            }
            DataType::Bool => {
                if let (Some(self_data), Some(other_data)) =
                    (self.data.as_bool_slice(), other.data.as_bool_slice())
                {
                    self_data == other_data
                } else {
                    false
                }
            }
        }
    }

    /// Squeeze dimensions of size 1
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

    /// Squeeze specific dimension if it has size 1
    pub fn squeeze_dim(&self, dim: usize) -> Result<Self> {
        if dim >= self.ndim() {
            return Err(MinitensorError::index_error(dim as isize, 0, self.ndim()));
        }

        if self.shape.dims()[dim] != 1 {
            return Ok(self.clone());
        }

        let mut new_dims = self.shape.dims().to_vec();
        new_dims.remove(dim);
        let new_shape = Shape::new(new_dims);
        self.view(new_shape)
    }

    /// Add dimension of size 1
    pub fn unsqueeze(&self, dim: usize) -> Result<Self> {
        if dim > self.ndim() {
            return Err(MinitensorError::index_error(
                dim as isize,
                0,
                self.ndim() + 1,
            ));
        }

        let mut new_dims = self.shape.dims().to_vec();
        new_dims.insert(dim, 1);
        let new_shape = Shape::new(new_dims);
        self.view(new_shape)
    }

    /// Flatten tensor from start_dim to end_dim
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
                        ((end - start) + step - 1) / step
                    };
                    out_dims.push(size);
                    orig_dim_map.push(i);
                    starts.push(start);
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
                for idx in 0..out_shape.numel() {
                    let mut rem = idx;
                    let mut src_idx = offset;
                    for (j, &stride) in out_strides.as_slice().iter().enumerate() {
                        let coord = rem / stride;
                        rem %= stride;
                        let orig_dim = orig_dim_map[j];
                        src_idx += (starts[j] + coord) * strides[orig_dim];
                    }
                    output[idx] = input[src_idx];
                }
            }
            DataType::Float64 => {
                let input = self
                    .data
                    .as_f64_slice()
                    .ok_or_else(|| MinitensorError::internal_error("Expected f64 data"))?;
                let output = result_data.as_f64_slice_mut().unwrap();
                for idx in 0..out_shape.numel() {
                    let mut rem = idx;
                    let mut src_idx = offset;
                    for (j, &stride) in out_strides.as_slice().iter().enumerate() {
                        let coord = rem / stride;
                        rem %= stride;
                        let orig_dim = orig_dim_map[j];
                        src_idx += (starts[j] + coord) * strides[orig_dim];
                    }
                    output[idx] = input[src_idx];
                }
            }
            DataType::Int32 => {
                let input = self
                    .data
                    .as_i32_slice()
                    .ok_or_else(|| MinitensorError::internal_error("Expected i32 data"))?;
                let output = result_data.as_i32_slice_mut().unwrap();
                for idx in 0..out_shape.numel() {
                    let mut rem = idx;
                    let mut src_idx = offset;
                    for (j, &stride) in out_strides.as_slice().iter().enumerate() {
                        let coord = rem / stride;
                        rem %= stride;
                        let orig_dim = orig_dim_map[j];
                        src_idx += (starts[j] + coord) * strides[orig_dim];
                    }
                    output[idx] = input[src_idx];
                }
            }
            DataType::Int64 => {
                let input = self
                    .data
                    .as_i64_slice()
                    .ok_or_else(|| MinitensorError::internal_error("Expected i64 data"))?;
                let output = result_data.as_i64_slice_mut().unwrap();
                for idx in 0..out_shape.numel() {
                    let mut rem = idx;
                    let mut src_idx = offset;
                    for (j, &stride) in out_strides.as_slice().iter().enumerate() {
                        let coord = rem / stride;
                        rem %= stride;
                        let orig_dim = orig_dim_map[j];
                        src_idx += (starts[j] + coord) * strides[orig_dim];
                    }
                    output[idx] = input[src_idx];
                }
            }
            DataType::Bool => {
                let input = self
                    .data
                    .as_bool_slice()
                    .ok_or_else(|| MinitensorError::internal_error("Expected bool data"))?;
                let output = result_data.as_bool_slice_mut().unwrap();
                for idx in 0..out_shape.numel() {
                    let mut rem = idx;
                    let mut src_idx = offset;
                    for (j, &stride) in out_strides.as_slice().iter().enumerate() {
                        let coord = rem / stride;
                        rem %= stride;
                        let orig_dim = orig_dim_map[j];
                        src_idx += (starts[j] + coord) * strides[orig_dim];
                    }
                    output[idx] = input[src_idx];
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
                        ((end - start) + step - 1) / step
                    };
                    out_dims.push(size);
                    orig_dim_map.push(i);
                    starts.push(start);
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
                let scalar = val_slice.get(0).copied();
                for idx in 0..out_shape.numel() {
                    let mut rem = idx;
                    let mut src_idx = offset;
                    for (j, &stride) in out_strides.as_slice().iter().enumerate() {
                        let coord = rem / stride;
                        rem %= stride;
                        let orig_dim = orig_dim_map[j];
                        src_idx += (starts[j] + coord) * strides[orig_dim];
                    }
                    let val = if value.numel() == 1 {
                        scalar.unwrap()
                    } else {
                        val_slice[idx]
                    };
                    slice[src_idx] = val;
                }
            }
            DataType::Float64 => {
                let slice = data.as_f64_slice_mut().unwrap();
                let val_slice = value.data().as_f64_slice().unwrap();
                let scalar = val_slice.get(0).copied();
                for idx in 0..out_shape.numel() {
                    let mut rem = idx;
                    let mut src_idx = offset;
                    for (j, &stride) in out_strides.as_slice().iter().enumerate() {
                        let coord = rem / stride;
                        rem %= stride;
                        let orig_dim = orig_dim_map[j];
                        src_idx += (starts[j] + coord) * strides[orig_dim];
                    }
                    let val = if value.numel() == 1 {
                        scalar.unwrap()
                    } else {
                        val_slice[idx]
                    };
                    slice[src_idx] = val;
                }
            }
            DataType::Int32 => {
                let slice = data.as_i32_slice_mut().unwrap();
                let val_slice = value.data().as_i32_slice().unwrap();
                let scalar = val_slice.get(0).copied();
                for idx in 0..out_shape.numel() {
                    let mut rem = idx;
                    let mut src_idx = offset;
                    for (j, &stride) in out_strides.as_slice().iter().enumerate() {
                        let coord = rem / stride;
                        rem %= stride;
                        let orig_dim = orig_dim_map[j];
                        src_idx += (starts[j] + coord) * strides[orig_dim];
                    }
                    let val = if value.numel() == 1 {
                        scalar.unwrap()
                    } else {
                        val_slice[idx]
                    };
                    slice[src_idx] = val;
                }
            }
            DataType::Int64 => {
                let slice = data.as_i64_slice_mut().unwrap();
                let val_slice = value.data().as_i64_slice().unwrap();
                let scalar = val_slice.get(0).copied();
                for idx in 0..out_shape.numel() {
                    let mut rem = idx;
                    let mut src_idx = offset;
                    for (j, &stride) in out_strides.as_slice().iter().enumerate() {
                        let coord = rem / stride;
                        rem %= stride;
                        let orig_dim = orig_dim_map[j];
                        src_idx += (starts[j] + coord) * strides[orig_dim];
                    }
                    let val = if value.numel() == 1 {
                        scalar.unwrap()
                    } else {
                        val_slice[idx]
                    };
                    slice[src_idx] = val;
                }
            }
            DataType::Bool => {
                let slice = data.as_bool_slice_mut().unwrap();
                let val_slice = value.data().as_bool_slice().unwrap();
                let scalar = val_slice.get(0).copied();
                for idx in 0..out_shape.numel() {
                    let mut rem = idx;
                    let mut src_idx = offset;
                    for (j, &stride) in out_strides.as_slice().iter().enumerate() {
                        let coord = rem / stride;
                        rem %= stride;
                        let orig_dim = orig_dim_map[j];
                        src_idx += (starts[j] + coord) * strides[orig_dim];
                    }
                    let val = if value.numel() == 1 {
                        scalar.unwrap()
                    } else {
                        val_slice[idx]
                    };
                    slice[src_idx] = val;
                }
            }
        }
        Ok(())
    }

    /// Check if tensor contains NaN values
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
            DataType::Bool => {
                self.data
                    .as_bool_slice()?
                    .iter()
                    .max()
                    .map(|&x| if x { 1.0 } else { 0.0 })
            }
        }
    }

    /// Get the minimum value in the tensor
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
            DataType::Bool => {
                self.data
                    .as_bool_slice()?
                    .iter()
                    .min()
                    .map(|&x| if x { 1.0 } else { 0.0 })
            }
        }
    }

    /// Get memory usage in bytes
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
    pub fn is_leaf(&self) -> bool {
        self.grad_fn.is_none()
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

        // Clear gradient
        tensor.zero_grad();
        assert!(!tensor.has_grad());
        assert!(tensor.grad().is_none());
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
}

// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

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

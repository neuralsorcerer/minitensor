// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

#[pymethods]
impl PyTensor {
    #[classattr]
    fn __array_priority__() -> f64 {
        1000.0
    }

    /// Create a new tensor from Python data
    #[new]
    #[pyo3(signature = (data=None, dtype=None, device=None, requires_grad=false))]
    fn new(
        data: Option<&Bound<PyAny>>,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let dtype = dtype::resolve_dtype_arg(dtype)?;
        let device = device.map(|d| d.device()).unwrap_or_else(Device::cpu);
        let requires_grad = requires_grad.unwrap_or(false);

        if let Some(value) = data {
            let tensor = convert_python_data_to_tensor(value, dtype, device, requires_grad)?;
            Ok(Self::from_tensor(tensor))
        } else {
            let tensor = Tensor::empty(Shape::new(Vec::new()), dtype, device, requires_grad);
            Ok(Self::from_tensor(tensor))
        }
    }

    // Properties
    #[getter]
    pub fn shape(&self) -> ShapeSequence {
        ShapeSequence::from_dims(self.inner.shape().dims().to_vec())
    }

    pub fn shape_vec(&self) -> Vec<usize> {
        self.inner.shape().dims().to_vec()
    }

    #[getter]
    pub fn dtype(&self) -> String {
        dtype::dtype_to_python_string(self.inner.dtype()).to_string()
    }

    #[getter]
    fn device(&self) -> String {
        self.inner.device().to_string()
    }

    #[getter]
    fn _tensor(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }

    #[setter]
    #[allow(non_snake_case)]
    fn set__tensor(&mut self, value: &PyTensor) {
        self.inner = value.inner.clone();
    }

    #[getter]
    pub fn requires_grad(&self) -> bool {
        self.inner.requires_grad()
    }

    #[getter]
    fn is_leaf(&self) -> bool {
        self.inner.is_leaf()
    }

    #[getter]
    fn has_grad(&self) -> bool {
        if engine::autograd::get_gradient(&self.inner).is_some() {
            return true;
        }

        self.inner.has_grad() || self.inner.grad().is_some()
    }

    #[getter]
    fn grad(&self) -> PyResult<Option<Self>> {
        if let Some(grad) = engine::autograd::get_gradient(&self.inner) {
            return Ok(Some(Self::from_tensor(grad)));
        }

        if let Some(stored) = self.inner.grad() {
            return Ok(Some(Self::from_tensor((**stored).clone())));
        }

        Ok(None)
    }

    #[getter]
    fn size(&self) -> usize {
        self.inner.numel()
    }

    #[getter]
    fn itemsize(&self) -> usize {
        match self.inner.dtype() {
            DataType::Float32 | DataType::Int32 => 4,
            DataType::Float64 | DataType::Int64 => 8,
            DataType::Bool => 1,
        }
    }

    #[getter]
    fn nbytes(&self) -> usize {
        self.size() * self.itemsize()
    }

    /// Get memory usage in bytes
    fn memory_usage_bytes(&self) -> usize {
        self.inner.memory_usage_bytes()
    }

    #[getter]
    fn strides<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.strides().as_slice())
    }

    // Basic tensor info methods
    pub fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    fn numel(&self) -> usize {
        self.inner.numel()
    }

    fn is_contiguous(&self) -> bool {
        self.inner.is_contiguous()
    }

    // Tensor manipulation methods
    #[pyo3(signature = (*shape))]
    pub fn reshape(&self, shape: &Bound<PyTuple>) -> PyResult<Self> {
        let dims = normalize_variadic_isize_args(shape, "shape")?;
        let reshaped = engine::operations::reshape_with_inference(&self.inner, dims)
            .map_err(_convert_error)?;
        Ok(Self::from_tensor(reshaped))
    }

    #[pyo3(signature = (*shape))]
    pub fn view(&self, shape: &Bound<PyTuple>) -> PyResult<Self> {
        self.reshape(shape)
    }

    #[pyo3(signature = (dim0=0, dim1=1))]
    pub fn transpose(&self, dim0: Option<isize>, dim1: Option<isize>) -> PyResult<Self> {
        let dim0 = dim0.unwrap_or(0);
        let dim1 = dim1.unwrap_or(1);
        let result = self.inner.transpose(dim0, dim1).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (*dims))]
    pub fn permute(&self, dims: &Bound<PyTuple>) -> PyResult<Self> {
        let dims_vec = normalize_variadic_isize_args(dims, "dims")?;
        let result = self.inner.permute(dims_vec).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn movedim(&self, source: &Bound<PyAny>, destination: &Bound<PyAny>) -> PyResult<Self> {
        let src_vec: Vec<isize> = match source.extract::<isize>() {
            Ok(v) => vec![v],
            Err(_) => source.extract()?,
        };
        let dst_vec: Vec<isize> = match destination.extract::<isize>() {
            Ok(v) => vec![v],
            Err(_) => destination.extract()?,
        };
        let result = engine::operations::shape_ops::movedim(&self.inner, &src_vec, &dst_vec)
            .map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(name = "moveaxis")]
    #[pyo3(signature = (source, destination))]
    pub fn moveaxis_alias(
        &self,
        source: &Bound<PyAny>,
        destination: &Bound<PyAny>,
    ) -> PyResult<Self> {
        self.movedim(source, destination)
    }

    #[pyo3(name = "swapaxes")]
    #[pyo3(signature = (dim0, dim1))]
    pub fn swapaxes_alias(&self, dim0: isize, dim1: isize) -> PyResult<Self> {
        self.transpose(Some(dim0), Some(dim1))
    }

    #[pyo3(name = "swapdims")]
    #[pyo3(signature = (dim0, dim1))]
    pub fn swapdims_alias(&self, dim0: isize, dim1: isize) -> PyResult<Self> {
        self.transpose(Some(dim0), Some(dim1))
    }

    #[pyo3(signature = (dim=None))]
    pub fn squeeze(&self, dim: Option<isize>) -> PyResult<Self> {
        let result = if let Some(d) = dim {
            self.inner.squeeze_dim(d)
        } else {
            self.inner.squeeze()
        }
        .map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (dim))]
    pub fn unsqueeze(&self, dim: isize) -> PyResult<Self> {
        let result = self.inner.unsqueeze(dim).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (*dims))]
    pub fn expand(&self, dims: &Bound<PyTuple>) -> PyResult<Self> {
        let dims_vec = normalize_variadic_isize_args(dims, "shape")?;
        let result = self.inner.expand(dims_vec).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (*repeats))]
    pub fn repeat(&self, repeats: &Bound<PyTuple>) -> PyResult<Self> {
        let repeats_any = if repeats.len() == 1 {
            let first = repeats.get_item(0)?;
            if first.cast::<PySequence>().is_ok() {
                first.clone().into_any()
            } else {
                repeats.clone().into_any()
            }
        } else {
            repeats.clone().into_any()
        };
        let repeat_vec = normalize_repeat_spec(&repeats_any)?;
        let result = self.inner.repeat(repeat_vec).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn flip(&self, dims: &Bound<PyAny>) -> PyResult<Self> {
        let dims_vec = normalize_required_axes(dims, "dims")?;
        let result =
            engine::operations::shape_ops::flip(&self.inner, &dims_vec).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (shifts, dims=None))]
    pub fn roll(&self, shifts: &Bound<PyAny>, dims: Option<&Bound<PyAny>>) -> PyResult<Self> {
        let shift_vec = normalize_roll_shifts(shifts)?;
        let dims_vec = normalize_optional_axes(dims)?;
        let dims_ref = dims_vec.as_deref();
        let result = engine::operations::shape_ops::roll(&self.inner, &shift_vec, dims_ref)
            .map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (repeats, dim=None, output_size=None))]
    pub fn repeat_interleave(
        &self,
        repeats: &Bound<PyAny>,
        dim: Option<isize>,
        output_size: Option<usize>,
    ) -> PyResult<Self> {
        if let Ok(value) = repeats.extract::<usize>() {
            let result = engine::operations::shape_ops::repeat_interleave(
                &self.inner,
                RepeatInterleaveSpec::Scalar(value),
                dim,
                output_size,
            )
            .map_err(_convert_error)?;
            return Ok(Self::from_tensor(result));
        }

        if let Ok(seq) = repeats.extract::<Vec<i64>>() {
            let mut converted = Vec::with_capacity(seq.len());
            for value in seq {
                if value < 0 {
                    return Err(PyValueError::new_err(
                        "repeat_interleave: repeats must be non-negative integers",
                    ));
                }
                let value = usize::try_from(value).map_err(|_| {
                    PyValueError::new_err("repeat_interleave: repeat value exceeds platform limits")
                })?;
                converted.push(value);
            }
            let result = engine::operations::shape_ops::repeat_interleave(
                &self.inner,
                RepeatInterleaveSpec::Slice(&converted),
                dim,
                output_size,
            )
            .map_err(_convert_error)?;
            return Ok(Self::from_tensor(result));
        }

        if let Ok(py_tensor) = repeats.extract::<PyRef<PyTensor>>() {
            let result = engine::operations::shape_ops::repeat_interleave(
                &self.inner,
                RepeatInterleaveSpec::Tensor(py_tensor.tensor()),
                dim,
                output_size,
            )
            .map_err(_convert_error)?;
            return Ok(Self::from_tensor(result));
        }

        if let Ok(bound_attr) = repeats.getattr("_tensor")
            && let Ok(py_tensor) = bound_attr.extract::<PyRef<PyTensor>>()
        {
            let result = engine::operations::shape_ops::repeat_interleave(
                &self.inner,
                RepeatInterleaveSpec::Tensor(py_tensor.tensor()),
                dim,
                output_size,
            )
            .map_err(_convert_error)?;
            return Ok(Self::from_tensor(result));
        }

        Err(PyTypeError::new_err(
            "repeat_interleave: repeats must be an int, sequence of ints, or Tensor",
        ))
    }

    #[pyo3(signature = (dim, start, length))]
    pub fn narrow(&self, dim: isize, start: usize, length: usize) -> PyResult<Self> {
        let result = engine::operations::shape_ops::narrow(&self.inner, dim, start, length)
            .map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (start_dim=0, end_dim=-1))]
    pub fn flatten(&self, start_dim: isize, end_dim: isize) -> PyResult<Self> {
        let result = self
            .inner
            .flatten(start_dim, end_dim)
            .map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn ravel(&self) -> PyResult<Self> {
        self.flatten(0, -1)
    }

}

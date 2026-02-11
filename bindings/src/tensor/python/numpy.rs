// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

fn convert_tensor_to_numpy(tensor: &Tensor, py: Python, _force_copy: bool) -> PyResult<Py<PyAny>> {
    if tensor.device() != Device::cpu() {
        return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "Cannot convert GPU tensor to NumPy array. Use .cpu() first.",
        ));
    }

    let shape = tensor.shape().dims();
    let strides = tensor.strides().as_slice();
    let numel: usize = shape.iter().product();

    macro_rules! to_numpy {
        ($slice:expr, $ty:ty) => {{
            let data = $slice.ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to get tensor data")
            })?;
            let mut out = Vec::<$ty>::with_capacity(numel);
            let mut indices = vec![0usize; shape.len()];
            for _ in 0..numel {
                let mut offset = 0usize;
                for (idx, stride) in indices.iter().zip(strides) {
                    offset += idx * stride;
                }
                out.push(data[offset]);
                for axis in (0..indices.len()).rev() {
                    indices[axis] += 1;
                    if indices[axis] < shape[axis] {
                        break;
                    }
                    indices[axis] = 0;
                }
            }
            let array = PyArray::from_vec(py, out).reshape(shape)?;
            Ok(array.into_any().unbind())
        }};
    }

    let array: PyResult<Py<PyAny>> = match tensor.dtype() {
        DataType::Float32 => to_numpy!(tensor.data().as_f32_slice(), f32),
        DataType::Float64 => to_numpy!(tensor.data().as_f64_slice(), f64),
        DataType::Int32 => to_numpy!(tensor.data().as_i32_slice(), i32),
        DataType::Int64 => to_numpy!(tensor.data().as_i64_slice(), i64),
        DataType::Bool => to_numpy!(tensor.data().as_bool_slice(), bool),
    };

    array
}

fn convert_tensor_to_python_list(tensor: &Tensor, py: Python) -> PyResult<Py<PyAny>> {
    let shape: Vec<usize> = tensor.shape().dims().to_vec();
    match tensor.dtype() {
        DataType::Float32 => {
            let data = tensor.data().as_f32_slice().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to get f32 data")
            })?;
            nested_list_from_slice(py, data, &shape)
        }
        DataType::Float64 => {
            let data = tensor.data().as_f64_slice().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to get f64 data")
            })?;
            nested_list_from_slice(py, data, &shape)
        }
        DataType::Int32 => {
            let data = tensor.data().as_i32_slice().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to get i32 data")
            })?;
            nested_list_from_slice(py, data, &shape)
        }
        DataType::Int64 => {
            let data = tensor.data().as_i64_slice().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to get i64 data")
            })?;
            nested_list_from_slice(py, data, &shape)
        }
        DataType::Bool => {
            let data = tensor.data().as_bool_slice().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to get bool data")
            })?;
            nested_list_from_slice(py, data, &shape)
        }
    }
}

fn convert_tensor_to_python_scalar(tensor: &Tensor, py: Python) -> PyResult<Py<PyAny>> {
    if tensor.numel() != 1 {
        return Err(PyErr::new::<PyRuntimeError, _>(format!(
            "a Tensor with {} elements cannot be converted to Scalar",
            tensor.numel()
        )));
    }

    match tensor.dtype() {
        DataType::Float32 => {
            let data = tensor.data().as_f32_slice().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to get f32 data")
            })?;
            data[0].into_py_any(py)
        }
        DataType::Float64 => {
            let data = tensor.data().as_f64_slice().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to get f64 data")
            })?;
            data[0].into_py_any(py)
        }
        DataType::Int32 => {
            let data = tensor.data().as_i32_slice().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to get i32 data")
            })?;
            data[0].into_py_any(py)
        }
        DataType::Int64 => {
            let data = tensor.data().as_i64_slice().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to get i64 data")
            })?;
            data[0].into_py_any(py)
        }
        DataType::Bool => {
            let data = tensor.data().as_bool_slice().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to get bool data")
            })?;
            data[0].into_py_any(py)
        }
    }
}

fn nested_list_from_slice<'py, T>(
    py: Python<'py>,
    data: &[T],
    shape: &[usize],
) -> PyResult<Py<PyAny>>
where
    T: Copy + IntoPyObjectExt<'py>,
{
    if shape.is_empty() {
        if let Some(value) = data.first() {
            return (*value).into_py_any(py);
        }
        return PyList::empty(py).into_py_any(py);
    }

    if shape.len() == 1 {
        let mut elements: Vec<Py<PyAny>> = Vec::with_capacity(data.len());
        for value in data.iter().copied() {
            elements.push(value.into_py_any(py)?);
        }
        let list = PyList::new(py, elements)?;
        return list.into_py_any(py);
    }

    let chunk = shape[1..]
        .iter()
        .fold(1usize, |acc, &dim| acc.saturating_mul(dim));
    let mut parts: Vec<Py<PyAny>> = Vec::with_capacity(shape[0]);
    for index in 0..shape[0] {
        let start = index * chunk;
        let end = start + chunk;
        let slice = if start <= end && end <= data.len() {
            &data[start..end]
        } else {
            &[]
        };
        parts.push(nested_list_from_slice(py, slice, &shape[1..])?);
    }

    let list = PyList::new(py, parts)?;
    list.into_py_any(py)
}

fn create_random_tensor(
    shape: Shape,
    dtype: DataType,
    device: Device,
    requires_grad: bool,
    normal: bool,
) -> PyResult<Tensor> {
    let mut tensor_data = TensorData::uninitialized_on_device(shape.numel(), dtype, device);

    match dtype {
        DataType::Float32 => {
            if let Some(slice) = tensor_data.as_f32_slice_mut() {
                use rand::RngExt;
                random::with_rng(|rng| {
                    if normal {
                        use rand_distr::{Distribution, Normal};
                        let normal_dist = Normal::new(0.0f32, 1.0f32).unwrap();
                        for val in slice.iter_mut() {
                            *val = normal_dist.sample(rng);
                        }
                    } else {
                        for val in slice.iter_mut() {
                            *val = rng.random::<f32>();
                        }
                    }
                });
            }
        }
        DataType::Float64 => {
            if let Some(slice) = tensor_data.as_f64_slice_mut() {
                use rand::RngExt;
                random::with_rng(|rng| {
                    if normal {
                        use rand_distr::{Distribution, Normal};
                        let normal_dist = Normal::new(0.0f64, 1.0f64).unwrap();
                        for val in slice.iter_mut() {
                            *val = normal_dist.sample(rng);
                        }
                    } else {
                        for val in slice.iter_mut() {
                            *val = rng.random::<f64>();
                        }
                    }
                });
            }
        }
        DataType::Int32 => {
            if let Some(slice) = tensor_data.as_i32_slice_mut() {
                use rand::RngExt;
                random::with_rng(|rng| {
                    if normal {
                        use rand_distr::{Distribution, Normal};
                        let normal_dist = Normal::new(0.0f32, 1.0f32).unwrap();
                        for val in slice.iter_mut() {
                            *val = normal_dist.sample(rng) as i32;
                        }
                    } else {
                        for val in slice.iter_mut() {
                            *val = rng.random::<i32>();
                        }
                    }
                });
            }
        }
        DataType::Int64 => {
            if let Some(slice) = tensor_data.as_i64_slice_mut() {
                use rand::RngExt;
                random::with_rng(|rng| {
                    if normal {
                        use rand_distr::{Distribution, Normal};
                        let normal_dist = Normal::new(0.0f64, 1.0f64).unwrap();
                        for val in slice.iter_mut() {
                            *val = normal_dist.sample(rng) as i64;
                        }
                    } else {
                        for val in slice.iter_mut() {
                            *val = rng.random::<i64>();
                        }
                    }
                });
            }
        }
        DataType::Bool => {
            if let Some(slice) = tensor_data.as_bool_slice_mut() {
                use rand::RngExt;
                random::with_rng(|rng| {
                    for val in slice.iter_mut() {
                        *val = rng.random::<bool>();
                    }
                });
            }
        }
    }

    Ok(Tensor::new(
        Arc::new(tensor_data),
        shape,
        dtype,
        device,
        requires_grad,
    ))
}

enum FanInitKind {
    XavierUniform,
    XavierNormal,
    HeUniform,
    HeNormal,
    LecunUniform,
    LecunNormal,
}

impl FanInitKind {
    fn apply(
        &self,
        shape: Shape,
        dtype: DataType,
        device: Device,
        requires_grad: bool,
    ) -> Result<Tensor, MinitensorError> {
        match self {
            FanInitKind::XavierUniform => {
                nn::init::xavier_uniform_init(shape, dtype, device, requires_grad)
            }
            FanInitKind::XavierNormal => {
                nn::init::xavier_normal_init(shape, dtype, device, requires_grad)
            }
            FanInitKind::HeUniform => {
                nn::init::he_uniform_init(shape, dtype, device, requires_grad)
            }
            FanInitKind::HeNormal => nn::init::he_normal_init(shape, dtype, device, requires_grad),
            FanInitKind::LecunUniform => {
                nn::init::lecun_uniform_init(shape, dtype, device, requires_grad)
            }
            FanInitKind::LecunNormal => {
                nn::init::lecun_normal_init(shape, dtype, device, requires_grad)
            }
        }
    }
}

fn ensure_float_dtype(dtype: DataType, context: &str) -> PyResult<()> {
    match dtype {
        DataType::Float32 | DataType::Float64 => Ok(()),
        _ => Err(PyValueError::new_err(format!(
            "{context} only supports float32 or float64 dtypes",
        ))),
    }
}

fn ensure_valid_fan_shape(shape: &Shape, context: &str) -> PyResult<()> {
    if shape.dims().contains(&0) {
        Err(PyValueError::new_err(format!(
            "{context} requires all shape dimensions to be at least 1",
        )))
    } else {
        Ok(())
    }
}

fn create_fan_init_tensor(
    shape: Shape,
    dtype: DataType,
    device: Device,
    requires_grad: bool,
    kind: FanInitKind,
    context: &str,
) -> PyResult<Tensor> {
    ensure_float_dtype(dtype, context)?;
    ensure_valid_fan_shape(&shape, context)?;
    let tensor = kind.apply(shape, dtype, device, requires_grad);
    tensor.map_err(_convert_error)
}

fn create_uniform_tensor(
    shape: Shape,
    dtype: DataType,
    device: Device,
    requires_grad: bool,
    low: f64,
    high: f64,
) -> PyResult<Tensor> {
    if !low.is_finite() || !high.is_finite() {
        return Err(PyValueError::new_err(
            "uniform requires finite low and high values",
        ));
    }

    if high.partial_cmp(&low) != Some(Ordering::Greater) {
        return Err(PyValueError::new_err(
            "uniform requires high to be greater than low",
        ));
    }

    let tensor = nn::init::init_uniform(shape, low, high, dtype, device, requires_grad)
        .map_err(_convert_error)?;
    Ok(tensor)
}

#[allow(clippy::too_many_arguments)]
fn create_truncated_normal_tensor(
    shape: Shape,
    dtype: DataType,
    device: Device,
    requires_grad: bool,
    mean: f64,
    std: f64,
    lower: Option<f64>,
    upper: Option<f64>,
    context: &str,
) -> PyResult<Tensor> {
    ensure_float_dtype(dtype, context)?;

    if !mean.is_finite() {
        return Err(PyValueError::new_err(format!(
            "{context} requires a finite mean",
        )));
    }

    if !std.is_finite() || std <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "{context} requires std to be a positive finite value",
        )));
    }

    let default_lower = mean - 2.0 * std;
    let default_upper = mean + 2.0 * std;
    let lower = lower.unwrap_or(default_lower);
    let upper = upper.unwrap_or(default_upper);

    if lower.is_nan() || upper.is_nan() {
        return Err(PyValueError::new_err(format!(
            "{context} requires non-NaN bounds",
        )));
    }

    if upper.partial_cmp(&lower) != Some(Ordering::Greater) {
        return Err(PyValueError::new_err(format!(
            "{context} requires upper bound to be greater than lower bound",
        )));
    }

    let tensor = nn::init::truncated_normal_init(
        shape,
        mean,
        std,
        lower,
        upper,
        dtype,
        device,
        requires_grad,
    )
    .map_err(_convert_error)?;
    Ok(tensor)
}

fn prepare_new_tensor_from_existing(
    source: &Tensor,
    dtype: DataType,
    device: Device,
    requires_grad: bool,
) -> PyResult<Tensor> {
    let mut tensor = source.detach();

    if tensor.device() != device {
        tensor = tensor.to(device).map_err(_convert_error)?;
    }

    if tensor.dtype() != dtype {
        tensor = tensor.astype(dtype).map_err(_convert_error)?;
    }

    tensor = tensor.deep_clone().map_err(_convert_error)?;

    if requires_grad {
        tensor = tensor.requires_grad_(true);
    }

    Ok(tensor)
}

fn adapt_tensor_for_as_tensor(
    source: &Tensor,
    dtype: DataType,
    device: Device,
    requires_grad: bool,
    copy: bool,
) -> PyResult<Tensor> {
    if !copy
        && source.dtype() == dtype
        && source.device() == device
        && source.requires_grad() == requires_grad
    {
        return Ok(source.clone());
    }

    let mut tensor = if copy || (source.requires_grad() && !requires_grad) {
        source.detach()
    } else {
        source.clone()
    };

    if tensor.device() != device {
        tensor = tensor.to(device).map_err(_convert_error)?;
    }

    if tensor.dtype() != dtype {
        tensor = tensor.astype(dtype).map_err(_convert_error)?;
    }

    if copy {
        tensor = tensor.deep_clone().map_err(_convert_error)?;
    }

    if tensor.requires_grad() != requires_grad {
        tensor = tensor.requires_grad_(requires_grad);
    }

    Ok(tensor)
}

fn create_randint_tensor(
    shape: Shape,
    dtype: DataType,
    device: Device,
    requires_grad: bool,
    low: i64,
    high: i64,
) -> PyResult<Tensor> {
    let tensor_data = match dtype {
        DataType::Int32 => {
            let low_i32 = i32::try_from(low)
                .map_err(|_| PyValueError::new_err("low is out of range for dtype int32"))?;
            let high_i32 = i32::try_from(high)
                .map_err(|_| PyValueError::new_err("high is out of range for dtype int32"))?;
            if low_i32 >= high_i32 {
                return Err(PyValueError::new_err(
                    "randint requires that low < high after casting to int32",
                ));
            }
            let mut values = vec![0i32; shape.numel()];
            random::with_rng(|rng| {
                use rand::RngExt;
                for value in &mut values {
                    *value = rng.random_range(low_i32..high_i32);
                }
            });
            TensorData::from_vec_i32(values, device)
        }
        DataType::Int64 => {
            if high <= low {
                return Err(PyValueError::new_err("randint requires that low < high"));
            }
            let mut values = vec![0i64; shape.numel()];
            random::with_rng(|rng| {
                use rand::RngExt;
                for value in &mut values {
                    *value = rng.random_range(low..high);
                }
            });
            TensorData::from_vec_i64(values, device)
        }
        _ => {
            return Err(PyValueError::new_err(
                "randint only supports int32 or int64 dtypes",
            ));
        }
    };

    Ok(Tensor::new(
        Arc::new(tensor_data),
        shape,
        dtype,
        device,
        requires_grad,
    ))
}

fn create_randperm_tensor(
    n: usize,
    dtype: DataType,
    device: Device,
    requires_grad: bool,
) -> PyResult<Tensor> {
    let tensor_data = match dtype {
        DataType::Int32 => {
            let _ = i32::try_from(n).map_err(|_| {
                PyValueError::new_err("randperm with dtype int32 requires n <= i32::MAX")
            })?;
            let mut values = Vec::with_capacity(n);
            for idx in 0..n {
                values.push(i32::try_from(idx).map_err(|_| {
                    PyValueError::new_err("randperm with dtype int32 requires n <= i32::MAX")
                })?);
            }
            random::with_rng(|rng| {
                use rand::seq::SliceRandom;
                values.shuffle(rng);
            });
            TensorData::from_vec_i32(values, device)
        }
        DataType::Int64 => {
            let _ = i64::try_from(n).map_err(|_| {
                PyValueError::new_err("randperm with dtype int64 requires n <= i64::MAX")
            })?;
            let mut values = Vec::with_capacity(n);
            for idx in 0..n {
                values.push(idx as i64);
            }
            random::with_rng(|rng| {
                use rand::seq::SliceRandom;
                values.shuffle(rng);
            });
            TensorData::from_vec_i64(values, device)
        }
        _ => {
            return Err(PyValueError::new_err(
                "randperm only supports int32 or int64 dtypes",
            ));
        }
    };

    Ok(Tensor::new(
        Arc::new(tensor_data),
        Shape::new(vec![n]),
        dtype,
        device,
        requires_grad,
    ))
}

fn create_eye_tensor(
    n: usize,
    m: usize,
    dtype: DataType,
    device: Device,
    requires_grad: bool,
) -> PyResult<Tensor> {
    let shape = Shape::new(vec![n, m]);
    let mut tensor_data = TensorData::zeros_on_device(shape.numel(), dtype, device);

    match dtype {
        DataType::Float32 => {
            if let Some(slice) = tensor_data.as_f32_slice_mut() {
                for i in 0..n.min(m) {
                    slice[i * m + i] = 1.0;
                }
            }
        }
        DataType::Float64 => {
            if let Some(slice) = tensor_data.as_f64_slice_mut() {
                for i in 0..n.min(m) {
                    slice[i * m + i] = 1.0;
                }
            }
        }
        DataType::Int32 => {
            if let Some(slice) = tensor_data.as_i32_slice_mut() {
                for i in 0..n.min(m) {
                    slice[i * m + i] = 1;
                }
            }
        }
        DataType::Int64 => {
            if let Some(slice) = tensor_data.as_i64_slice_mut() {
                for i in 0..n.min(m) {
                    slice[i * m + i] = 1;
                }
            }
        }
        DataType::Bool => {
            if let Some(slice) = tensor_data.as_bool_slice_mut() {
                for i in 0..n.min(m) {
                    slice[i * m + i] = true;
                }
            }
        }
    }

    Ok(Tensor::new(
        Arc::new(tensor_data),
        shape,
        dtype,
        device,
        requires_grad,
    ))
}

fn create_full_tensor(
    shape: Vec<usize>,
    fill_value: f64,
    dtype: DataType,
    device: Device,
    requires_grad: bool,
) -> PyResult<Tensor> {
    let shape = Shape::new(shape);
    let mut tensor_data = TensorData::uninitialized_on_device(shape.numel(), dtype, device);

    match dtype {
        DataType::Float32 => {
            if let Some(slice) = tensor_data.as_f32_slice_mut() {
                slice.fill(fill_value as f32);
            }
        }
        DataType::Float64 => {
            if let Some(slice) = tensor_data.as_f64_slice_mut() {
                slice.fill(fill_value);
            }
        }
        DataType::Int32 => {
            if let Some(slice) = tensor_data.as_i32_slice_mut() {
                slice.fill(fill_value as i32);
            }
        }
        DataType::Int64 => {
            if let Some(slice) = tensor_data.as_i64_slice_mut() {
                slice.fill(fill_value as i64);
            }
        }
        DataType::Bool => {
            if let Some(slice) = tensor_data.as_bool_slice_mut() {
                slice.fill(fill_value != 0.0);
            }
        }
    }

    Ok(Tensor::new(
        Arc::new(tensor_data),
        shape,
        dtype,
        device,
        requires_grad,
    ))
}
fn create_arange_tensor(
    start: f64,
    end: f64,
    step: f64,
    dtype: DataType,
    device: Device,
    requires_grad: bool,
) -> PyResult<Tensor> {
    if step == 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Step cannot be zero",
        ));
    }

    let num_elements = ((end - start) / step).ceil() as usize;
    let shape = Shape::new(vec![num_elements]);
    let mut tensor_data = TensorData::uninitialized_on_device(shape.numel(), dtype, device);

    match dtype {
        DataType::Float32 => {
            if let Some(slice) = tensor_data.as_f32_slice_mut() {
                for (i, val) in slice.iter_mut().enumerate() {
                    *val = (start + i as f64 * step) as f32;
                }
            }
        }
        DataType::Float64 => {
            if let Some(slice) = tensor_data.as_f64_slice_mut() {
                for (i, val) in slice.iter_mut().enumerate() {
                    *val = start + i as f64 * step;
                }
            }
        }
        DataType::Int32 => {
            if let Some(slice) = tensor_data.as_i32_slice_mut() {
                for (i, val) in slice.iter_mut().enumerate() {
                    *val = (start + i as f64 * step) as i32;
                }
            }
        }
        DataType::Int64 => {
            if let Some(slice) = tensor_data.as_i64_slice_mut() {
                for (i, val) in slice.iter_mut().enumerate() {
                    *val = (start + i as f64 * step) as i64;
                }
            }
        }
        DataType::Bool => {
            if let Some(slice) = tensor_data.as_bool_slice_mut() {
                for (i, val) in slice.iter_mut().enumerate() {
                    *val = (start + i as f64 * step) != 0.0;
                }
            }
        }
    }

    Ok(Tensor::new(
        Arc::new(tensor_data),
        shape,
        dtype,
        device,
        requires_grad,
    ))
}

fn create_linspace_tensor(
    start: f64,
    end: f64,
    steps: usize,
    dtype: DataType,
    device: Device,
    requires_grad: bool,
) -> PyResult<Tensor> {
    if steps == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Number of steps must be positive",
        ));
    }

    let shape = Shape::new(vec![steps]);
    let mut tensor_data = TensorData::uninitialized_on_device(shape.numel(), dtype, device);
    let denom = if steps > 1 { (steps - 1) as f64 } else { 1.0 };
    let step = if steps > 1 {
        (end - start) / denom
    } else {
        0.0
    };

    match dtype {
        DataType::Float32 => {
            if let Some(slice) = tensor_data.as_f32_slice_mut() {
                for (i, val) in slice.iter_mut().enumerate() {
                    let value = if steps == 1 {
                        start
                    } else {
                        start + i as f64 * step
                    };
                    *val = value as f32;
                }
            }
        }
        DataType::Float64 => {
            if let Some(slice) = tensor_data.as_f64_slice_mut() {
                for (i, val) in slice.iter_mut().enumerate() {
                    let value = if steps == 1 {
                        start
                    } else {
                        start + i as f64 * step
                    };
                    *val = value;
                }
            }
        }
        DataType::Int32 => {
            if let Some(slice) = tensor_data.as_i32_slice_mut() {
                for (i, val) in slice.iter_mut().enumerate() {
                    let value = if steps == 1 {
                        start
                    } else {
                        start + i as f64 * step
                    };
                    *val = value.round() as i32;
                }
            }
        }
        DataType::Int64 => {
            if let Some(slice) = tensor_data.as_i64_slice_mut() {
                for (i, val) in slice.iter_mut().enumerate() {
                    let value = if steps == 1 {
                        start
                    } else {
                        start + i as f64 * step
                    };
                    *val = value.round() as i64;
                }
            }
        }
        DataType::Bool => {
            if let Some(slice) = tensor_data.as_bool_slice_mut() {
                for (i, val) in slice.iter_mut().enumerate() {
                    let value = if steps == 1 {
                        start
                    } else {
                        start + i as f64 * step
                    };
                    *val = value != 0.0;
                }
            }
        }
    }

    Ok(Tensor::new(
        Arc::new(tensor_data),
        shape,
        dtype,
        device,
        requires_grad,
    ))
}

fn create_logspace_tensor(
    start: f64,
    end: f64,
    steps: usize,
    base: f64,
    dtype: DataType,
    device: Device,
    requires_grad: bool,
) -> PyResult<Tensor> {
    if steps == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Number of steps must be positive",
        ));
    }

    if base <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Base must be positive",
        ));
    }

    let shape = Shape::new(vec![steps]);
    let mut tensor_data = TensorData::uninitialized_on_device(shape.numel(), dtype, device);
    let denom = if steps > 1 { (steps - 1) as f64 } else { 1.0 };
    let step = if steps > 1 {
        (end - start) / denom
    } else {
        0.0
    };

    match dtype {
        DataType::Float32 => {
            if let Some(slice) = tensor_data.as_f32_slice_mut() {
                for (i, val) in slice.iter_mut().enumerate() {
                    let exponent = if steps == 1 {
                        start
                    } else {
                        start + i as f64 * step
                    };
                    *val = base.powf(exponent) as f32;
                }
            }
        }
        DataType::Float64 => {
            if let Some(slice) = tensor_data.as_f64_slice_mut() {
                for (i, val) in slice.iter_mut().enumerate() {
                    let exponent = if steps == 1 {
                        start
                    } else {
                        start + i as f64 * step
                    };
                    *val = base.powf(exponent);
                }
            }
        }
        DataType::Int32 => {
            if let Some(slice) = tensor_data.as_i32_slice_mut() {
                for (i, val) in slice.iter_mut().enumerate() {
                    let exponent = if steps == 1 {
                        start
                    } else {
                        start + i as f64 * step
                    };
                    *val = base.powf(exponent).round() as i32;
                }
            }
        }
        DataType::Int64 => {
            if let Some(slice) = tensor_data.as_i64_slice_mut() {
                for (i, val) in slice.iter_mut().enumerate() {
                    let exponent = if steps == 1 {
                        start
                    } else {
                        start + i as f64 * step
                    };
                    *val = base.powf(exponent).round() as i64;
                }
            }
        }
        DataType::Bool => {
            if let Some(slice) = tensor_data.as_bool_slice_mut() {
                for (i, val) in slice.iter_mut().enumerate() {
                    let exponent = if steps == 1 {
                        start
                    } else {
                        start + i as f64 * step
                    };
                    *val = base.powf(exponent) != 0.0;
                }
            }
        }
    }

    Ok(Tensor::new(
        Arc::new(tensor_data),
        shape,
        dtype,
        device,
        requires_grad,
    ))
}

// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
pub(crate) fn convert_python_data_to_tensor(
    data: &Bound<PyAny>,
    dtype: DataType,
    device: Device,
    requires_grad: bool,
) -> PyResult<Tensor> {
    // First try NumPy array conversion for any supported dtype
    if let Ok(numpy_module) = PyModule::import(data.py(), "numpy")
        && let Ok(ndarray_type) = numpy_module.getattr("ndarray")
        && data.is_instance(&ndarray_type)?
    {
        let maybe_tensor = panic::catch_unwind(AssertUnwindSafe(|| {
            convert_numpy_to_tensor(data, requires_grad)
        }));

        match maybe_tensor {
            Ok(Ok(tensor)) => {
                let tensor = if tensor.dtype() != dtype {
                    tensor.astype(dtype).map_err(_convert_error)?
                } else {
                    tensor
                };
                return Ok(tensor);
            }
            Ok(Err(err)) => {
                return Err(err);
            }
            Err(_) => {
                // Fall back to the slower Python list conversion path
                // when the NumPy capsule isn't available.
            }
        }
    }

    // Handle Python lists and tuples by flattening values into scalar variants
    if let Ok(list) = data.cast::<PyList>() {
        let (shape, flat_data) = flatten_python_data(list)?;
        let (base_tensor, base_dtype) =
            tensor_from_flat_scalars(shape, flat_data, device, requires_grad)?;

        if base_dtype == dtype {
            return Ok(base_tensor);
        }

        return base_tensor.astype(dtype).map_err(_convert_error);
    }

    if let Ok(tuple) = data.cast::<PyTuple>() {
        let list = tuple.to_list();
        return convert_python_data_to_tensor(list.as_any(), dtype, device, requires_grad);
    }

    // Handle scalars
    if let Ok(value_bool) = data.extract::<bool>() {
        let shape = Shape::new(vec![]);
        let base_data = Arc::new(TensorData::from_vec_bool(vec![value_bool], device));
        let mut tensor = Tensor::new(base_data, shape, DataType::Bool, device, requires_grad);
        if dtype != DataType::Bool {
            tensor = tensor.astype(dtype).map_err(_convert_error)?;
        }
        return Ok(tensor);
    }

    if let Ok(value_int) = data.extract::<i64>() {
        let shape = Shape::new(vec![]);
        let base_data = Arc::new(TensorData::from_vec_i64(vec![value_int], device));
        let mut tensor = Tensor::new(base_data, shape, DataType::Int64, device, requires_grad);
        if dtype != DataType::Int64 {
            tensor = tensor.astype(dtype).map_err(_convert_error)?;
        }
        return Ok(tensor);
    }

    if let Ok(value_float) = data.extract::<f64>() {
        let shape = Shape::new(vec![]);
        let base_data = Arc::new(TensorData::from_vec_f64(vec![value_float], device));
        let mut tensor = Tensor::new(base_data, shape, DataType::Float64, device, requires_grad);
        if dtype != DataType::Float64 {
            tensor = tensor.astype(dtype).map_err(_convert_error)?;
        }
        return Ok(tensor);
    }

    let float_name = intern!(data.py(), "__float__");
    if data.hasattr(float_name)? {
        let method = data.getattr(float_name)?;
        if method.is_callable() {
            let float_obj = method.call0()?;
            let val = float_obj.extract::<f64>()?;
            let shape = Shape::new(vec![]);
            let base_data = Arc::new(TensorData::from_vec_f64(vec![val], device));
            let mut tensor =
                Tensor::new(base_data, shape, DataType::Float64, device, requires_grad);
            if dtype != DataType::Float64 {
                tensor = tensor.astype(dtype).map_err(_convert_error)?;
            }
            return Ok(tensor);
        }
    }

    Err(PyErr::new::<PyTypeError, _>(
        "Unsupported data type for tensor creation",
    ))
}

pub(crate) fn apply_binary_ufunc<F>(
    operands: &[Tensor],
    kind: BinaryOpKind,
    op: F,
) -> PyResult<Tensor>
where
    F: Fn(&Tensor, &Tensor) -> Result<Tensor, MinitensorError>,
{
    if operands.len() != 2 {
        return Err(PyValueError::new_err(
            "Binary ufuncs require exactly two operands",
        ));
    }

    let (lhs_cast, rhs_cast, _) =
        coerce_binary_operands(&operands[0], &operands[1], kind).map_err(_convert_error)?;

    let lhs_tensor = match lhs_cast {
        Cow::Borrowed(tensor) => tensor.clone(),
        Cow::Owned(tensor) => tensor,
    };
    let rhs_tensor = match rhs_cast {
        Cow::Borrowed(tensor) => tensor.clone(),
        Cow::Owned(tensor) => tensor,
    };

    op(&lhs_tensor, &rhs_tensor).map_err(_convert_error)
}

pub(crate) fn apply_unary_ufunc<F>(operands: &[Tensor], op: F) -> PyResult<Tensor>
where
    F: Fn(&Tensor) -> Result<Tensor, MinitensorError>,
{
    if operands.len() != 1 {
        return Err(PyValueError::new_err(
            "Unary ufuncs require exactly one operand",
        ));
    }

    let tensor = operands[0].clone();
    op(&tensor).map_err(_convert_error)
}

pub(crate) fn py_not_implemented(py: Python) -> PyResult<Py<PyAny>> {
    unsafe {
        Ok(
            pyo3::Bound::<pyo3::PyAny>::from_borrowed_ptr(py, pyo3::ffi::Py_NotImplemented())
                .unbind(),
        )
    }
}

pub(crate) fn parse_dtype_like(value: &Bound<PyAny>) -> PyResult<DataType> {
    if let Ok(name) = value.extract::<String>() {
        dtype::parse_dtype(&name)
    } else {
        Err(PyTypeError::new_err(
            "dtype must be specified as a string such as 'float32'",
        ))
    }
}

pub(crate) fn parse_device_like(value: &Bound<PyAny>) -> PyResult<Device> {
    if let Ok(device) = value.extract::<PyDevice>() {
        return Ok(device.device());
    }

    if let Ok(spec) = value.extract::<String>() {
        return Device::from_str(&spec).map_err(|err| {
            PyValueError::new_err(format!("Unsupported device specification '{spec}': {err}"))
        });
    }

    Err(PyTypeError::new_err(
        "device must be specified as a Device object or string like 'cpu' or 'cuda:0'",
    ))
}

pub(crate) fn ensure_backward_gradient_compatible(
    reference: &Tensor,
    gradient: &mut Tensor,
) -> PyResult<()> {
    let expected_shape = reference.shape().dims();
    let actual_shape = gradient.shape().dims();
    if expected_shape != actual_shape {
        return Err(PyRuntimeError::new_err(format!(
            "backward() expected gradient tensor with shape {:?}, but got {:?}",
            expected_shape, actual_shape
        )));
    }

    if gradient.device() != reference.device() {
        *gradient = gradient.to(reference.device()).map_err(_convert_error)?;
    }

    if gradient.dtype() != reference.dtype() {
        *gradient = gradient.astype(reference.dtype()).map_err(_convert_error)?;
    }

    if gradient.requires_grad() {
        *gradient = gradient.detach();
    }

    Ok(())
}

pub(crate) fn tensor_from_py_value(reference: &Tensor, value: &Bound<PyAny>) -> PyResult<Tensor> {
    if let Some(py_tensor) = extract_wrapped_pytensor(value) {
        return Ok(py_tensor.inner.clone());
    }

    if let Ok(numpy_module) = PyModule::import(value.py(), "numpy")
        && let Ok(ndarray_type) = numpy_module.getattr("ndarray")
        && value.is_instance(&ndarray_type)?
    {
        if let Ok(dtype_obj) = value.getattr("dtype") {
            let dtype_str = dtype_obj.str()?.to_str()?.to_ascii_lowercase();
            if let Ok(array_dtype) = dtype::parse_dtype(&dtype_str) {
                return convert_python_data_to_tensor(
                    value,
                    array_dtype,
                    reference.device(),
                    false,
                );
            }
        }
        return convert_python_data_to_tensor(value, reference.dtype(), reference.device(), false);
    }

    if let Ok(py_tensor) = PyTensor::from_python_value(value) {
        let mut tensor = py_tensor.inner;
        if tensor.device() != reference.device() {
            tensor = tensor.to(reference.device()).map_err(_convert_error)?;
        }

        let target_dtype = dtype::resolve_scalar_dtype(value, reference.dtype())
            .ok()
            .or_else(|| infer_python_value_dtype(value))
            .unwrap_or(reference.dtype());

        if tensor.dtype() != target_dtype {
            tensor = tensor.astype(target_dtype).map_err(_convert_error)?;
        }

        return Ok(tensor);
    }

    let index_name = intern!(value.py(), "__index__");
    if value.hasattr(index_name)? {
        let method = value.getattr(index_name)?;
        if method.is_callable() {
            let result = method.call0()?;
            if result.is_instance_of::<PyInt>() {
                let dtype = match dtype::resolve_scalar_dtype(value, reference.dtype()) {
                    Ok(dt) => dt,
                    Err(_) => reference.dtype(),
                };
                return convert_python_data_to_tensor(
                    result.as_any(),
                    dtype,
                    reference.device(),
                    false,
                );
            }
        }
    }

    let dtype = match dtype::resolve_scalar_dtype(value, reference.dtype()) {
        Ok(dt) => dt,
        Err(_) => infer_python_value_dtype(value).unwrap_or(reference.dtype()),
    };
    convert_python_data_to_tensor(value, dtype, reference.device(), false)
}

pub(crate) fn tensor_bool_from_py(value: &Bound<PyAny>, device: Device) -> PyResult<Tensor> {
    if let Some(py_tensor) = extract_wrapped_pytensor(value) {
        let mut tensor = py_tensor.inner.clone();
        if tensor.dtype() != DataType::Bool {
            return Err(PyTypeError::new_err("mask must be a bool tensor"));
        }
        if tensor.device() != device {
            tensor = tensor.to(device).map_err(_convert_error)?;
        }
        return Ok(tensor);
    }

    if let Ok(value_bool) = value.extract::<bool>() {
        let data = Arc::new(TensorData::from_vec_bool(vec![value_bool], device));
        return Ok(Tensor::new(
            data,
            Shape::new(vec![]),
            DataType::Bool,
            device,
            false,
        ));
    }

    convert_python_data_to_tensor(value, DataType::Bool, device, false)
}

fn promote_dtypes(a: DataType, b: DataType) -> DataType {
    use DataType::*;

    if a == b {
        return a;
    }

    match (a, b) {
        (Float64, _) | (_, Float64) => Float64,
        (Float32, _) | (_, Float32) => Float32,
        (Int64, _) | (_, Int64) => Int64,
        (Int32, _) | (_, Int32) => Int32,
        _ => Bool,
    }
}

pub(crate) fn infer_python_value_dtype(value: &Bound<PyAny>) -> Option<DataType> {
    if let Some(py_tensor) = extract_wrapped_pytensor(value) {
        return Some(py_tensor.inner.dtype());
    }

    if value.extract::<bool>().is_ok() {
        return Some(DataType::Bool);
    }

    if value.extract::<i64>().is_ok() {
        return Some(DataType::Int64);
    }

    if value.extract::<f64>().is_ok() {
        return Some(dtype::default_dtype());
    }

    if let Ok(numpy_module) = PyModule::import(value.py(), "numpy")
        && let Ok(ndarray_type) = numpy_module.getattr("ndarray")
        && let Ok(true) = value.is_instance(&ndarray_type)
        && let Ok(dtype_obj) = value.getattr("dtype")
        && let Ok(dtype_str) = dtype_obj.str()
        && let Ok(dtype) = dtype::parse_dtype(&dtype_str.to_str().ok()?.to_ascii_lowercase())
    {
        return Some(dtype);
    }

    if let Ok(list) = value.cast::<PyList>() {
        return infer_sequence_dtype(list.iter());
    }

    if let Ok(tuple) = value.cast::<PyTuple>() {
        return infer_sequence_dtype(tuple.iter());
    }

    None
}

fn infer_sequence_dtype<'py, I>(iter: I) -> Option<DataType>
where
    I: Iterator<Item = Bound<'py, PyAny>>,
{
    let mut dtype: Option<DataType> = None;
    for item in iter {
        let item_dtype = infer_python_value_dtype(&item)?;
        dtype = Some(match dtype {
            Some(current) => promote_dtypes(current, item_dtype),
            None => item_dtype,
        });
    }
    dtype
}

pub(crate) fn prepare_binary_operands_from_py(
    reference: &Tensor,
    other: &Bound<PyAny>,
    reverse: bool,
    kind: BinaryOpKind,
) -> PyResult<(Tensor, Tensor)> {
    let lhs_input = if reverse {
        tensor_from_py_value(reference, other)?
    } else {
        reference.clone()
    };

    let rhs_input = if reverse {
        reference.clone()
    } else {
        tensor_from_py_value(reference, other)?
    };

    let (lhs_cast, rhs_cast, _) =
        coerce_binary_operands(&lhs_input, &rhs_input, kind).map_err(_convert_error)?;
    let lhs_tensor = match lhs_cast {
        Cow::Borrowed(_) => lhs_input.clone(),
        Cow::Owned(tensor) => tensor,
    };
    let rhs_tensor = match rhs_cast {
        Cow::Borrowed(_) => rhs_input.clone(),
        Cow::Owned(tensor) => tensor,
    };

    Ok((lhs_tensor, rhs_tensor))
}

fn flatten_python_data(list: &Bound<PyList>) -> PyResult<(Vec<usize>, Vec<ScalarValue>)> {
    let mut shape = vec![list.len()];
    let mut flat_data = vec![];

    fn process_nested(
        item: &Bound<PyAny>,
        depth: usize,
        shape: &mut Vec<usize>,
        flat_data: &mut Vec<ScalarValue>,
    ) -> PyResult<()> {
        if let Ok(nested_list) = item.cast::<PyList>() {
            let length = nested_list.len();
            if depth >= shape.len() {
                shape.push(length);
            } else if shape[depth] != length {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Inconsistent nested sequence lengths",
                ));
            }
            for nested_item in nested_list.iter() {
                process_nested(&nested_item, depth + 1, shape, flat_data)?;
            }
            return Ok(());
        }

        if let Ok(nested_tuple) = item.cast::<PyTuple>() {
            let list = nested_tuple.to_list();
            let length = list.len();
            if depth >= shape.len() {
                shape.push(length);
            } else if shape[depth] != length {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Inconsistent nested sequence lengths",
                ));
            }
            for nested_item in list.iter() {
                process_nested(&nested_item, depth + 1, shape, flat_data)?;
            }
            return Ok(());
        }

        if let Ok(value_bool) = item.extract::<bool>() {
            flat_data.push(ScalarValue::Bool(value_bool));
            return Ok(());
        }

        if let Ok(value_int) = item.extract::<i64>() {
            flat_data.push(ScalarValue::Int(value_int));
            return Ok(());
        }

        let index_name = intern!(item.py(), "__index__");
        if item.hasattr(index_name)? {
            let method = item.getattr(index_name)?;
            if method.is_callable() {
                let result = method.call0()?;
                if result.is_instance_of::<PyInt>() {
                    let value = result.extract::<i64>()?;
                    flat_data.push(ScalarValue::Int(value));
                    return Ok(());
                }
            }
        }

        if let Ok(value_float) = item.extract::<f64>() {
            flat_data.push(ScalarValue::Float(value_float));
            return Ok(());
        }

        let float_name = intern!(item.py(), "__float__");
        if item.hasattr(float_name)? {
            let method = item.getattr(float_name)?;
            if method.is_callable() {
                let float_obj = method.call0()?;
                let value = float_obj.extract::<f64>()?;
                flat_data.push(ScalarValue::Float(value));
                return Ok(());
            }
        }

        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Unsupported scalar type in nested sequence",
        ))
    }

    for item in list.iter() {
        process_nested(&item, 1, &mut shape, &mut flat_data)?;
    }

    Ok((shape, flat_data))
}

#[derive(Clone, Copy)]
enum ScalarValue {
    Bool(bool),
    Int(i64),
    Float(f64),
}

impl ScalarValue {
    fn kind(&self) -> ScalarKind {
        match self {
            ScalarValue::Bool(_) => ScalarKind::Bool,
            ScalarValue::Int(_) => ScalarKind::Int,
            ScalarValue::Float(_) => ScalarKind::Float,
        }
    }

    fn to_bool(self) -> bool {
        match self {
            ScalarValue::Bool(value) => value,
            ScalarValue::Int(value) => value != 0,
            ScalarValue::Float(value) => value != 0.0,
        }
    }

    fn to_i64(self) -> i64 {
        match self {
            ScalarValue::Bool(value) => value as i64,
            ScalarValue::Int(value) => value,
            ScalarValue::Float(value) => value as i64,
        }
    }

    fn to_f64(self) -> f64 {
        match self {
            ScalarValue::Bool(value) => {
                if value {
                    1.0
                } else {
                    0.0
                }
            }
            ScalarValue::Int(value) => value as f64,
            ScalarValue::Float(value) => value,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ScalarKind {
    Bool,
    Int,
    Float,
}

impl ScalarKind {
    fn combine(self, other: ScalarKind) -> ScalarKind {
        use ScalarKind::*;
        match (self, other) {
            (Float, _) | (_, Float) => Float,
            (Int, _) | (_, Int) => Int,
            _ => Bool,
        }
    }
}

fn tensor_from_flat_scalars(
    shape: Vec<usize>,
    values: Vec<ScalarValue>,
    device: Device,
    requires_grad: bool,
) -> PyResult<(Tensor, DataType)> {
    let mut kind = ScalarKind::Bool;
    for value in &values {
        kind = kind.combine(value.kind());
    }

    let tensor = match kind {
        ScalarKind::Bool => {
            let data: Vec<bool> = values.into_iter().map(ScalarValue::to_bool).collect();
            Tensor::new(
                Arc::new(TensorData::from_vec_bool(data, device)),
                Shape::new(shape),
                DataType::Bool,
                device,
                requires_grad,
            )
        }
        ScalarKind::Int => {
            let data: Vec<i64> = values.into_iter().map(ScalarValue::to_i64).collect();
            Tensor::new(
                Arc::new(TensorData::from_vec_i64(data, device)),
                Shape::new(shape),
                DataType::Int64,
                device,
                requires_grad,
            )
        }
        ScalarKind::Float => {
            let data: Vec<f64> = values.into_iter().map(ScalarValue::to_f64).collect();
            Tensor::new(
                Arc::new(TensorData::from_vec_f64(data, device)),
                Shape::new(shape),
                DataType::Float64,
                device,
                requires_grad,
            )
        }
    };

    let dtype = tensor.dtype();
    Ok((tensor, dtype))
}
fn is_ellipsis(item: &Bound<PyAny>) -> bool {
    item.is_instance_of::<pyo3::types::PyEllipsis>()
}

/// Recursively find the first scalar leaf of a (possibly nested) list.
fn first_list_leaf<'py>(item: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    if let Ok(list) = item.cast::<PyList>() {
        match list.iter().next() {
            Some(inner) => first_list_leaf(&inner),
            None => Ok(None),
        }
    } else {
        Ok(Some(item.clone()))
    }
}

/// Select rows along dim 0 for integer fancy indexing, wrapping negative
/// indices like NumPy.
fn select_rows(reference: &Tensor, vals: &[i64]) -> PyResult<Tensor> {
    if reference.ndim() == 0 {
        return Err(PyIndexError::new_err("too many indices for a 0-d tensor"));
    }
    let dim0 = reference.shape().dims()[0] as i64;
    let mut idx = Vec::with_capacity(vals.len());
    for &v in vals {
        let r = if v < 0 { v + dim0 } else { v };
        if r < 0 || r >= dim0 {
            return Err(PyIndexError::new_err(format!(
                "index {v} is out of bounds for dimension 0 with size {dim0}"
            )));
        }
        idx.push(r as usize);
    }
    engine::ops::shape_ops::index_select(reference, 0, &idx).map_err(_convert_error)
}

/// Extract a boolean mask tensor from a `__getitem__`/`__setitem__` key when
/// the key is a bool tensor, a bool ndarray, or a (nested) list of bools.
pub(crate) fn try_bool_mask_key(key: &Bound<PyAny>) -> PyResult<Option<Tensor>> {
    if let Some(pt) = extract_wrapped_pytensor(key) {
        if pt.tensor().dtype() == DataType::Bool {
            return Ok(Some(pt.tensor().clone()));
        }
        return Ok(None);
    }
    if key.cast::<PyArrayDyn<bool>>().is_ok() {
        return Ok(Some(convert_numpy_to_tensor(key, false)?));
    }
    if key.cast::<PyList>().is_ok()
        && let Some(leaf) = first_list_leaf(key)?
        && leaf.is_instance_of::<pyo3::types::PyBool>()
    {
        let mask = convert_python_data_to_tensor(key, DataType::Bool, Device::cpu(), false)?;
        return Ok(Some(mask));
    }
    Ok(None)
}

/// NumPy-style fancy `__getitem__` forms: a boolean mask selects blocks along
/// the leading dimensions (`masked_index`); a 1-D integer tensor/ndarray/list
/// selects rows along dim 0 (negative indices wrap). Returns `Ok(None)` when
/// `key` is not a fancy index so basic indexing can proceed.
pub(crate) fn try_fancy_index_tensor(
    reference: &Tensor,
    key: &Bound<PyAny>,
) -> PyResult<Option<Tensor>> {
    if let Some(mask) = try_bool_mask_key(key)? {
        return engine::ops::selection::masked_index(reference, &mask)
            .map(Some)
            .map_err(_convert_error);
    }

    if let Some(pt) = extract_wrapped_pytensor(key) {
        let t = pt.tensor();
        if matches!(t.dtype(), DataType::Int32 | DataType::Int64) && t.ndim() == 1 {
            let t = t.contiguous().map_err(_convert_error)?;
            let vals: Vec<i64> = match t.dtype() {
                DataType::Int32 => t
                    .data()
                    .as_i32_slice()
                    .ok_or_else(|| PyRuntimeError::new_err("failed to read index tensor"))?
                    .iter()
                    .map(|&v| v as i64)
                    .collect(),
                _ => t
                    .data()
                    .as_i64_slice()
                    .ok_or_else(|| PyRuntimeError::new_err("failed to read index tensor"))?
                    .to_vec(),
            };
            return select_rows(reference, &vals).map(Some);
        }
        return Ok(None);
    }

    if let Ok(arr) = key.cast::<PyArrayDyn<i64>>() {
        let ro = arr.readonly();
        if ro.ndim() == 1 {
            return select_rows(reference, ro.as_slice()?).map(Some);
        }
        return Ok(None);
    }
    if let Ok(arr) = key.cast::<PyArrayDyn<i32>>() {
        let ro = arr.readonly();
        if ro.ndim() == 1 {
            let vals: Vec<i64> = ro.as_slice()?.iter().map(|&v| v as i64).collect();
            return select_rows(reference, &vals).map(Some);
        }
        return Ok(None);
    }

    if let Ok(list) = key.cast::<PyList>() {
        // Bool lists were handled above; a flat list of ints selects rows.
        // (An empty list selects zero rows, like NumPy.)
        let mut vals = Vec::with_capacity(list.len());
        for item in list.iter() {
            if item.is_instance_of::<pyo3::types::PyBool>() || item.cast::<PyList>().is_ok() {
                return Ok(None);
            }
            match item.extract::<i64>() {
                Ok(v) => vals.push(v),
                Err(_) => return Ok(None),
            }
        }
        return select_rows(reference, &vals).map(Some);
    }

    Ok(None)
}

fn full_slice(dim: usize) -> TensorIndex {
    TensorIndex::Slice {
        start: 0,
        end: dim,
        step: 1,
    }
}

fn parse_index(item: &Bound<PyAny>, dim_size: usize) -> PyResult<TensorIndex> {
    if let Ok(i) = item.extract::<isize>() {
        let mut idx = i;
        if idx < 0 {
            idx += dim_size as isize;
        }
        if idx < 0 || idx >= dim_size as isize {
            return Err(PyIndexError::new_err("Index out of bounds"));
        }
        Ok(TensorIndex::Index(idx as usize))
    } else if let Ok(slice) = item.cast::<PySlice>() {
        use std::convert::TryInto;

        let dim_size_isize: isize = dim_size
            .try_into()
            .map_err(|_| PyValueError::new_err("dim_size too large"))?;
        let indices = slice.indices(dim_size_isize)?;
        if indices.step <= 0 {
            return Err(PyIndexError::new_err("slice step must be positive"));
        }
        Ok(TensorIndex::Slice {
            start: indices.start.max(0) as usize,
            end: indices.stop.max(0) as usize,
            step: indices.step as usize,
        })
    } else if item.is_none() {
        Ok(TensorIndex::Slice {
            start: 0,
            end: dim_size,
            step: 1,
        })
    } else {
        Err(PyTypeError::new_err("Invalid index type"))
    }
}

/// Parse a `__getitem__` key into per-dimension indices plus the output-axis
/// positions where a size-1 axis must be inserted for `None`/`np.newaxis`.
///
/// Unlike [`parse_indices`], `None` does not consume an input dimension; it
/// inserts a new length-1 axis at the corresponding output position.
/// Integer indices drop their dimension, slices keep it.
pub(crate) fn parse_getitem_indices(
    key: &Bound<PyAny>,
    shape: &[usize],
) -> PyResult<(Vec<TensorIndex>, Vec<usize>)> {
    let items: Vec<Bound<PyAny>> = if let Ok(tup) = key.cast::<PyTuple>() {
        tup.iter().collect()
    } else {
        vec![key.clone()]
    };

    // `None` entries add axes and `...` expands to full slices, so only the
    // real entries count against the tensor rank.
    if items.iter().filter(|it| is_ellipsis(it)).count() > 1 {
        return Err(PyIndexError::new_err(
            "an index can only have a single ellipsis ('...')",
        ));
    }
    let real_count = items
        .iter()
        .filter(|it| !it.is_none() && !is_ellipsis(it))
        .count();
    if real_count > shape.len() {
        return Err(PyIndexError::new_err("Too many indices"));
    }

    let mut real_indices: Vec<TensorIndex> = Vec::with_capacity(shape.len());
    let mut newaxis_positions: Vec<usize> = Vec::new();
    let mut input_dim = 0usize;
    let mut out_dim = 0usize;

    for item in &items {
        if item.is_none() {
            newaxis_positions.push(out_dim);
            out_dim += 1;
            continue;
        }
        if is_ellipsis(item) {
            // `...` stands for every dimension not consumed by the real
            // entries, each taken in full.
            for _ in 0..shape.len() - real_count {
                real_indices.push(full_slice(shape[input_dim]));
                input_dim += 1;
                out_dim += 1;
            }
            continue;
        }
        let idx = parse_index(item, shape[input_dim])?;
        // Integer indices remove the dimension; slices keep it in the output.
        let keeps_dim = matches!(idx, TensorIndex::Slice { .. });
        real_indices.push(idx);
        input_dim += 1;
        if keeps_dim {
            out_dim += 1;
        }
    }

    // Any dimensions not addressed explicitly are taken in full.
    for &dim in &shape[input_dim..] {
        real_indices.push(full_slice(dim));
    }

    Ok((real_indices, newaxis_positions))
}

pub(crate) fn parse_indices(key: &Bound<PyAny>, shape: &[usize]) -> PyResult<Vec<TensorIndex>> {
    let items: Vec<Bound<PyAny>> = if let Ok(tup) = key.cast::<PyTuple>() {
        tup.iter().collect()
    } else {
        vec![key.clone()]
    };

    if items.iter().filter(|it| is_ellipsis(it)).count() > 1 {
        return Err(PyIndexError::new_err(
            "an index can only have a single ellipsis ('...')",
        ));
    }
    let real_count = items.iter().filter(|it| !is_ellipsis(it)).count();
    if real_count > shape.len() {
        return Err(PyIndexError::new_err("Too many indices"));
    }

    let mut result: Vec<TensorIndex> = Vec::with_capacity(shape.len());
    let mut input_dim = 0usize;
    for item in &items {
        if is_ellipsis(item) {
            for _ in 0..shape.len() - real_count {
                result.push(full_slice(shape[input_dim]));
                input_dim += 1;
            }
            continue;
        }
        result.push(parse_index(item, shape[input_dim])?);
        input_dim += 1;
    }
    for &dim in &shape[input_dim..] {
        result.push(full_slice(dim));
    }
    Ok(result)
}

pub(crate) fn convert_numpy_to_tensor(
    array: &Bound<PyAny>,
    requires_grad: bool,
) -> PyResult<Tensor> {
    if let Ok(array_f32) = array.cast::<PyArrayDyn<f32>>() {
        let readonly = array_f32.readonly();
        let shape = Shape::new(readonly.shape().to_vec());
        let data_vec: Vec<f32> = readonly.as_slice()?.to_vec();
        let tensor_data = Arc::new(TensorData::from_vec(
            data_vec,
            DataType::Float32,
            Device::cpu(),
        ));
        Ok(Tensor::new(
            tensor_data,
            shape,
            DataType::Float32,
            Device::cpu(),
            requires_grad,
        ))
    } else if let Ok(array_f64) = array.cast::<PyArrayDyn<f64>>() {
        let readonly = array_f64.readonly();
        let shape = Shape::new(readonly.shape().to_vec());
        let data_vec: Vec<f64> = readonly.as_slice()?.to_vec();
        let tensor_data = Arc::new(TensorData::from_vec(
            data_vec,
            DataType::Float64,
            Device::cpu(),
        ));
        Ok(Tensor::new(
            tensor_data,
            shape,
            DataType::Float64,
            Device::cpu(),
            requires_grad,
        ))
    } else if let Ok(array_i32) = array.cast::<PyArrayDyn<i32>>() {
        let readonly = array_i32.readonly();
        let shape = Shape::new(readonly.shape().to_vec());
        let data_vec: Vec<i32> = readonly.as_slice()?.to_vec();
        let tensor_data = Arc::new(TensorData::from_vec(
            data_vec,
            DataType::Int32,
            Device::cpu(),
        ));
        Ok(Tensor::new(
            tensor_data,
            shape,
            DataType::Int32,
            Device::cpu(),
            requires_grad,
        ))
    } else if let Ok(array_i64) = array.cast::<PyArrayDyn<i64>>() {
        let readonly = array_i64.readonly();
        let shape = Shape::new(readonly.shape().to_vec());
        let data_vec: Vec<i64> = readonly.as_slice()?.to_vec();
        let tensor_data = Arc::new(TensorData::from_vec(
            data_vec,
            DataType::Int64,
            Device::cpu(),
        ));
        Ok(Tensor::new(
            tensor_data,
            shape,
            DataType::Int64,
            Device::cpu(),
            requires_grad,
        ))
    } else if let Ok(array_bool) = array.cast::<PyArrayDyn<bool>>() {
        let readonly = array_bool.readonly();
        let shape = Shape::new(readonly.shape().to_vec());
        let data_vec: Vec<bool> = readonly.as_slice()?.to_vec();
        let tensor_data = Arc::new(TensorData::from_vec(
            data_vec,
            DataType::Bool,
            Device::cpu(),
        ));
        Ok(Tensor::new(
            tensor_data,
            shape,
            DataType::Bool,
            Device::cpu(),
            requires_grad,
        ))
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Unsupported NumPy array type",
        ))
    }
}

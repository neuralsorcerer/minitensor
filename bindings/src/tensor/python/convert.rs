// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
use numpy::{PyArrayDescrMethods, PyUntypedArray};
use pyo3::sync::PyOnceLock;
use pyo3::types::PyFloat;

/// Build a tensor from a Python object, then mark it trainable.
///
/// The order matters. Several of the paths below reach the requested dtype by
/// converting at some other one and casting, and a cast between float dtypes is
/// a differentiable operation. Carrying `requires_grad` into the conversion
/// therefore made the tensor handed back to the caller an *interior* node of a
/// graph whose input was a throwaway temporary -- so it was no longer a leaf,
/// and the first backward pass released its stored gradient along with the rest
/// of the subgraph. `mt.Tensor([1.0, 2.0], requires_grad=True)` accumulated
/// correctly once and then reported no gradient at all.
///
/// Marking it afterwards makes what a caller constructs a leaf every time,
/// whatever conversions it took to get there. The flag still respects
/// `no_grad`, matching what `Tensor::new` would have done with it.
/// The deepest nesting a sequence may have: NumPy's own limit on dimensions.
///
/// Every walk over a nested Python sequence checks it, because the walks
/// recurse (or step) once per level, and a list that contains itself has no
/// bottom: it overflowed the stack and took the interpreter with it.
const MAX_NESTING: usize = 64;

fn too_deeply_nested() -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
        "sequence is nested more than {MAX_NESTING} levels deep, or contains itself"
    ))
}

pub(crate) fn convert_python_data_to_tensor(
    data: &Bound<PyAny>,
    dtype: DataType,
    device: Device,
    requires_grad: bool,
) -> PyResult<Tensor> {
    let tensor = build_tensor_from_python(data, dtype, device, false)?;
    if requires_grad && engine::autograd::is_grad_enabled() {
        Ok(tensor.requires_grad_(true))
    } else {
        Ok(tensor)
    }
}

/// [`convert_python_data_to_tensor`] at `dtype`, or at the dtype `data`
/// infers to when there is none.
///
/// A list or tuple is read by NumPy once for both: inferring its dtype and
/// then converting it called `numpy.asarray` twice, half the 64us `as_tensor`
/// of a 1,000-element list took.
pub(crate) fn convert_python_data_inferring(
    data: &Bound<PyAny>,
    dtype: Option<DataType>,
    device: Device,
    requires_grad: bool,
) -> PyResult<Tensor> {
    if dtype.is_none()
        && (data.cast::<PyList>().is_ok() || data.cast::<PyTuple>().is_ok())
        && let Some(tensor) = sequence_via_numpy(data, None, device, false)
    {
        return Ok(if requires_grad && engine::autograd::is_grad_enabled() {
            tensor.requires_grad_(true)
        } else {
            tensor
        });
    }
    let dtype = dtype
        .or_else(|| infer_python_value_dtype(data))
        .unwrap_or_else(dtype::default_dtype);
    convert_python_data_to_tensor(data, dtype, device, requires_grad)
}

fn build_tensor_from_python(
    data: &Bound<PyAny>,
    dtype: DataType,
    device: Device,
    requires_grad: bool,
) -> PyResult<Tensor> {
    // First try NumPy array conversion for any supported dtype. A direct type
    // check, for the reason given in `tensor_from_py_value`: the Python
    // `isinstance` spelling costs a module import and an attribute lookup on
    // every call, and every scalar operand reaches here to be told no.
    if data.cast::<PyUntypedArray>().is_ok() {
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
        // NumPy is already a hard dependency of this extension, and it turns a
        // nested Python sequence into a contiguous typed buffer in C. Walking
        // the object graph here instead costs ~880ns per element against
        // `np.asarray`'s ~16ns -- 55x on a 20k list. Anything NumPy cannot
        // represent as a dtype this crate supports (ragged nesting, object
        // arrays, strings) returns `None` and falls through to the traversal
        // below, so its behaviour and error messages are unchanged.
        if let Some(tensor) = sequence_via_numpy(data, Some(dtype), device, requires_grad) {
            return Ok(tensor);
        }

        let (shape, flat_data, oversized) = flatten_python_data(list)?;
        if let Some(value) = oversized
            && dtype.is_int()
        {
            return Err(does_not_fit(value, dtype));
        }
        let (base_tensor, _) = tensor_from_flat_scalars(shape, flat_data, device, requires_grad)?;
        return python_values_as(base_tensor, dtype);
    }

    if let Ok(tuple) = data.cast::<PyTuple>() {
        let list = tuple.to_list();
        return build_tensor_from_python(list.as_any(), dtype, device, requires_grad);
    }

    // A range is a sequence of ints; inside a list it already converted.
    if data.is_instance_of::<pyo3::types::PyRange>() {
        let list = PyList::new(data.py(), data.try_iter()?.collect::<PyResult<Vec<_>>>()?)?;
        return build_tensor_from_python(list.as_any(), dtype, device, requires_grad);
    }

    // Handle scalars.
    //
    // Each extract is guarded by a test of the value's kind because `extract`
    // reports failure by *raising*: on a Python float, the bool and int
    // attempts each built and discarded a Python exception before the float
    // attempt got the value, so `x + 1.0` paid for two exceptions nothing ever
    // saw. The exact builtin types say their kind, and so does a NumPy
    // scalar's dtype -- `np.float64` subclasses `float`, so it used to try
    // every branch. Anything else -- a subclass, or an object that merely
    // converts -- still reaches every branch in the old order.
    let exact_bool = data.is_exact_instance_of::<PyBool>();
    let exact_int = data.is_exact_instance_of::<PyInt>();
    let exact_float = data.is_exact_instance_of::<PyFloat>();
    let numpy = if exact_bool || exact_int || exact_float {
        None
    } else {
        crate::dtype::numpy_scalar_dtype(data)
    };
    let inexact = !(exact_bool || exact_int || exact_float) && numpy.is_none();
    let (is_bool, is_int, is_float) = match numpy {
        Some(DataType::Bool) => (true, false, false),
        Some(DataType::Int32 | DataType::Int64) => (false, true, false),
        Some(DataType::Float32 | DataType::Float64) => (false, false, true),
        None => (
            exact_bool || inexact,
            exact_int || inexact,
            exact_float || inexact,
        ),
    };

    if is_bool && let Ok(value_bool) = data.extract::<bool>() {
        let shape = Shape::new(vec![]);
        let base_data = Arc::new(TensorData::from_vec_bool(vec![value_bool], device));
        let mut tensor = Tensor::new(base_data, shape, DataType::Bool, device, requires_grad);
        if dtype != DataType::Bool {
            tensor = tensor.astype(dtype).map_err(_convert_error)?;
        }
        return Ok(tensor);
    }

    if is_int && let Ok(value_int) = data.extract::<i64>() {
        let shape = Shape::new(vec![]);
        let base_data = Arc::new(TensorData::from_vec_i64(vec![value_int], device));
        let tensor = Tensor::new(base_data, shape, DataType::Int64, device, requires_grad);
        return python_values_as(tensor, dtype);
    }

    // An int that reached here is past int64. It can still be a float, but an
    // integer dtype cannot hold it, and converting through a float saturated
    // it to the dtype's bound.
    if dtype.is_int() && data.is_instance_of::<PyInt>() {
        return Err(does_not_fit(data.str()?, dtype));
    }

    if is_float && let Ok(value_float) = data.extract::<f64>() {
        let shape = Shape::new(vec![]);
        let base_data = Arc::new(TensorData::from_vec_f64(vec![value_float], device));
        let tensor = Tensor::new(base_data, shape, DataType::Float64, device, requires_grad);
        return python_values_as(tensor, dtype);
    }

    let float_name = intern!(data.py(), "__float__");
    if data.hasattr(float_name)? {
        let method = data.getattr(float_name)?;
        if method.is_callable() {
            let float_obj = method.call0()?;
            let val = float_obj.extract::<f64>()?;
            let shape = Shape::new(vec![]);
            let base_data = Arc::new(TensorData::from_vec_f64(vec![val], device));
            let tensor = Tensor::new(base_data, shape, DataType::Float64, device, requires_grad);
            return python_values_as(tensor, dtype);
        }
    }

    Err(PyErr::new::<PyTypeError, _>(format!(
        "Unsupported data type for tensor creation: {}",
        data.get_type().name()?
    )))
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
    Ok(py.NotImplemented())
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
        return crate::device::ensure_available(device.device());
    }

    if let Ok(spec) = value.extract::<String>() {
        let device = Device::from_str(&spec).map_err(|err| {
            PyValueError::new_err(format!("Unsupported device specification '{spec}': {err}"))
        })?;
        return crate::device::ensure_available(device);
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
    if let Some(py_tensor) = borrow_wrapped_tensor(value) {
        // One clone, from a borrow. Extracting the class by value would have
        // cloned it first and made this a clone of a clone, and a `Tensor`
        // clone allocates twice -- once for the shape, once for the strides.
        return Ok(py_tensor.inner.clone());
    }

    // A direct type check, not `isinstance(value, numpy.ndarray)`. The Python
    // spelling needed the module and its `ndarray` attribute looked up on
    // every call, and this runs for every operand that is not a tensor --
    // including every `x + 1.0`, where the answer is always no. `x + 1.0` cost
    // 2.5us against 0.47 for a tensor operand, nearly all of it here.
    if let Ok(array) = value.cast::<PyUntypedArray>() {
        // The array's own width, read off its dtype descriptor. A dtype NumPy
        // has and the engine does not falls back to the tensor's, which is
        // what the conversion would have to narrow to anyway.
        let array_dtype = crate::share::dtype_from_type_num(array.dtype().num())
            .unwrap_or_else(|| reference.dtype());
        return convert_python_data_to_tensor(value, array_dtype, reference.device(), false);
    }

    // The dtype is resolved *before* the tensor is built. Building first and
    // casting after routed every Python float through the default dtype: with
    // float32 as the default, `x * 0.1` on a float64 tensor multiplied by
    // 0.10000000149011612, because widening a float32 back to float64 cannot
    // recover the digits that dropped on the way in. The same applied to a list
    // of floats, and to a Python int past 2^24.
    let target_dtype = dtype::resolve_scalar_dtype(value, reference.dtype())
        .ok()
        .or_else(|| {
            // Not a scalar -- a list or tuple. Its inferred dtype is the one it
            // would get standing alone, at the configured default; here it is an
            // operand, so it takes the context's width by the same rule a lone
            // float does.
            infer_python_value_dtype(value)
                .map(|inferred| dtype::dtype_for_context(inferred, reference.dtype()))
        })
        .unwrap_or(reference.dtype());

    // Built at the resolved dtype, so only the device can still differ. A
    // wrapped tensor returned above, and asking again would repeat the
    // wrapper lookup on every scalar operand.
    if let Ok(mut tensor) = convert_python_data_to_tensor(value, target_dtype, Device::cpu(), false)
    {
        if tensor.device() != reference.device() {
            tensor = tensor.to(reference.device()).map_err(_convert_error)?;
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
    if let Some(py_tensor) = borrow_wrapped_tensor(value) {
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
    infer_dtype_at(value, 0)
}

/// [`infer_python_value_dtype`] at a nesting depth, which bounds the walk: a
/// list containing itself recursed until the stack overflowed. Past the bound
/// there is no answer, and the conversion that follows reports the nesting.
fn infer_dtype_at(value: &Bound<PyAny>, depth: usize) -> Option<DataType> {
    if depth > MAX_NESTING {
        return None;
    }
    if let Some(py_tensor) = borrow_wrapped_tensor(value) {
        return Some(py_tensor.inner.dtype());
    }

    // By type before by extraction. An extraction a value refuses raises and
    // formats an exception, three of them for a list before it was recognised
    // as one: `as_tensor` of three floats took 4.1us, of an array 5.1.
    if value.is_exact_instance_of::<PyBool>() {
        return Some(DataType::Bool);
    }
    if value.is_exact_instance_of::<PyFloat>() {
        return Some(dtype::default_dtype());
    }

    // An array answers with its own dtype, 0-d ones included: asked for a
    // Python scalar first, a 0-d float64 array came back float32, a 0-d bool
    // array 1.0 and a 0-d int32 one int64, where `from_numpy` kept all three.
    if let Ok(array) = value.cast::<PyUntypedArray>() {
        if let Some(dtype) = crate::share::dtype_from_type_num(array.dtype().num()) {
            return Some(dtype);
        }
        // A dtype the engine does not carry directly still has an answer when
        // it widens exactly (uint8 -> int32, and so on). Without this the
        // inference falls through to the default float dtype, and `as_tensor`
        // would disagree with `from_numpy` about the same array.
        if let Ok((kind, itemsize)) = numpy_dtype_parts(value)
            && let Some(widened) = widened_numpy_dtype(kind, itemsize)
            && let Ok(dtype) = dtype::parse_dtype(widened)
        {
            return Some(dtype);
        }
    }

    if value.cast::<PyList>().is_ok() || value.cast::<PyTuple>().is_ok() {
        // Same reasoning as `sequence_via_numpy`: NumPy determines the common
        // type of a nested sequence in C. Walking it here calls back into this
        // function once per element, and `as_tensor` then walks it a second
        // time to read the values -- so the naive path traversed a 20k list
        // twice at ~450ns an element.
        if let Some(dtype) = sequence_dtype_via_numpy(value) {
            return Some(dtype);
        }
        if let Ok(list) = value.cast::<PyList>() {
            return infer_sequence_dtype(list.iter(), depth + 1);
        }
        if let Ok(tuple) = value.cast::<PyTuple>() {
            return infer_sequence_dtype(tuple.iter(), depth + 1);
        }
    }

    // A NumPy scalar keeps its width too, as it does as an operand.
    if let Some(dtype) = crate::dtype::numpy_scalar_dtype(value) {
        return Some(dtype);
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

    None
}

/// The dtype a Python sequence infers to, via `numpy.asarray`.
///
/// Returns `None` for anything NumPy cannot type (ragged, object, strings) so
/// the caller falls back to the element-wise walk. The mapping is this
/// library's own: any float width becomes the configured default float dtype,
/// so `[1.0, 2.0]` infers `float32` whatever width `asarray` read it at.
fn sequence_dtype_via_numpy(value: &Bound<PyAny>) -> Option<DataType> {
    sequence_kind(&numpy_asarray(value)?)
}

/// The dtype of a sequence NumPy has read into `array`, by the rule
/// [`sequence_dtype_via_numpy`] states.
fn sequence_kind(array: &Bound<PyAny>) -> Option<DataType> {
    match array.cast::<PyUntypedArray>().ok()?.dtype().kind() {
        b'b' => Some(DataType::Bool),
        b'i' | b'u' => Some(DataType::Int64),
        b'f' => Some(dtype::default_dtype()),
        _ => None,
    }
}

/// `numpy.asarray`, looked up once for the two sequence paths that call it.
static ASARRAY: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

/// `numpy.asarray(value)`, or `None` where NumPy refuses it -- ragged
/// nesting, say -- or cannot be imported.
fn numpy_asarray<'py>(value: &Bound<'py, PyAny>) -> Option<Bound<'py, PyAny>> {
    let py = value.py();
    let asarray = match ASARRAY.get(py) {
        Some(asarray) => asarray,
        None => {
            let found = PyModule::import(py, "numpy")
                .ok()?
                .getattr(intern!(py, "asarray"))
                .ok()?
                .unbind();
            ASARRAY.get_or_init(py, || found)
        }
    };
    asarray.bind(py).call1((value,)).ok()
}

fn infer_sequence_dtype<'py, I>(iter: I, depth: usize) -> Option<DataType>
where
    I: Iterator<Item = Bound<'py, PyAny>>,
{
    let mut dtype: Option<DataType> = None;
    for item in iter {
        let item_dtype = infer_dtype_at(&item, depth)?;
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

    // Both `Cow`s are taken apart before either input is moved out, which is
    // what ends their borrows. Spelled the other way round -- matching on one
    // `Cow` and cloning the input in its `Borrowed` arm -- it cost a second
    // clone of each operand in the ordinary case where the two dtypes already
    // agree and the coercion had nothing to do. A `Tensor` clone allocates
    // twice, so that was four allocations per binary op that reached Python.
    let (lhs_cast, rhs_cast, _) =
        coerce_binary_operands(&lhs_input, &rhs_input, kind).map_err(_convert_error)?;
    let lhs_owned = match lhs_cast {
        Cow::Owned(tensor) => Some(tensor),
        Cow::Borrowed(_) => None,
    };
    let rhs_owned = match rhs_cast {
        Cow::Owned(tensor) => Some(tensor),
        Cow::Borrowed(_) => None,
    };

    Ok((
        lhs_owned.unwrap_or(lhs_input),
        rhs_owned.unwrap_or(rhs_input),
    ))
}

/// The shape and values of a nested sequence, and the first int in it too
/// large for int64 -- read as the float it rounds to, which only a float
/// dtype can hold.
fn flatten_python_data(
    list: &Bound<PyList>,
) -> PyResult<(Vec<usize>, Vec<ScalarValue>, Option<String>)> {
    let mut shape = vec![list.len()];
    let mut flat_data = vec![];

    fn ragged() -> PyErr {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("Inconsistent nested sequence lengths")
    }

    // `shape` holds one entry per level that has held a sequence, so a scalar
    // belongs exactly one level past its end, and a sequence belongs above the
    // level the first scalar was found at. A scalar beside a sequence broke
    // neither length check: `[[1, 2], 3]` read as shape [2, 2] over three
    // values.
    fn process_nested(
        item: &Bound<PyAny>,
        depth: usize,
        shape: &mut Vec<usize>,
        flat_data: &mut Vec<ScalarValue>,
        leaf_depth: &mut Option<usize>,
        oversized: &mut Option<String>,
    ) -> PyResult<()> {
        if depth > MAX_NESTING {
            return Err(too_deeply_nested());
        }
        let is_sequence = item.cast::<PyList>().is_ok() || item.cast::<PyTuple>().is_ok();
        if is_sequence {
            if leaf_depth.is_some_and(|leaf| depth >= leaf) {
                return Err(ragged());
            }
        } else if shape.len() != depth {
            return Err(ragged());
        } else {
            *leaf_depth = Some(depth);
        }
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
                process_nested(
                    &nested_item,
                    depth + 1,
                    shape,
                    flat_data,
                    leaf_depth,
                    oversized,
                )?;
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
                process_nested(
                    &nested_item,
                    depth + 1,
                    shape,
                    flat_data,
                    leaf_depth,
                    oversized,
                )?;
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

        if item.is_instance_of::<PyInt>() {
            flat_data.push(ScalarValue::Float(item.extract::<f64>()?));
            if oversized.is_none() {
                *oversized = Some(item.str()?.to_string());
            }
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

    let mut leaf_depth = None;
    let mut oversized = None;
    for item in list.iter() {
        process_nested(
            &item,
            1,
            &mut shape,
            &mut flat_data,
            &mut leaf_depth,
            &mut oversized,
        )?;
    }

    Ok((shape, flat_data, oversized))
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
    // A loop, not recursion: a deeply nested list overflowed the stack, and a
    // list containing itself never ended. The step bound turns the second
    // into an error rather than a hang.
    let mut current = item.clone();
    for _ in 0..=MAX_NESTING {
        let Ok(list) = current.cast::<PyList>() else {
            return Ok(Some(current));
        };
        let Some(inner) = list.iter().next() else {
            return Ok(None);
        };
        current = inner;
    }
    Err(too_deeply_nested())
}

/// Select rows along dim 0 for integer fancy indexing, wrapping negative
/// indices.
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
        idx.push(r);
    }
    engine::ops::shape_ops::index_select(reference, 0, &idx).map_err(_convert_error)
}

/// The values of an `int64` or `int32` NumPy array in row-major order, widened
/// to `i64`, with its shape; `None` for any other object.
///
/// Not `as_slice` alone. It hands back a Fortran-ordered buffer in memory
/// order, which is column-major, so `t[np.asfortranarray(idx)]` gathered the
/// transposed indices into the right shape -- wrong values, no error. And it
/// refuses a strided array outright, so `t[idx[::2]]` raised an error about
/// alignment. The slice is taken only when the array is C-contiguous, where
/// memory order is row-major order; otherwise the array's own iterator walks
/// it in logical order, strides and all, with no copy through Python.
fn numpy_integer_values(item: &Bound<PyAny>) -> PyResult<Option<(Vec<i64>, Vec<usize>)>> {
    macro_rules! read {
        ($ty:ty) => {
            if let Ok(array) = item.cast::<PyArrayDyn<$ty>>() {
                let shape = array.shape().to_vec();
                // SAFETY: read without rust-numpy's dynamic borrow, for the
                // reason `convert_numpy_to_tensor` gives: the bindings never
                // release the GIL, and nothing here calls back into Python.
                let values: Vec<i64> = if array.is_c_contiguous() {
                    unsafe { array.as_slice() }?
                        .iter()
                        .map(|&v| v as i64)
                        .collect()
                } else {
                    unsafe { array.as_array() }
                        .iter()
                        .map(|&v| v as i64)
                        .collect()
                };
                return Ok(Some((values, shape)));
            }
        };
    }
    read!(i64);
    read!(i32);
    Ok(None)
}

/// A list of positions with a bool among them: `[1, True]` read the bool as
/// position 1. Positions and flags are two different kinds of index, and a
/// list mixing them is neither.
fn bool_among_positions() -> PyErr {
    PyTypeError::new_err(
        "an index list mixes ints and bools: a list of ints names positions and a list of \
         bools is a mask, so a bool among positions is neither",
    )
}

/// Whether a nested list has a bool anywhere among its leaves.
fn holds_bool(item: &Bound<PyAny>) -> PyResult<bool> {
    if item.is_instance_of::<pyo3::types::PyBool>() {
        return Ok(true);
    }
    match item.cast::<PyList>() {
        Ok(list) => {
            for entry in list.iter() {
                if holds_bool(&entry)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        Err(_) => Ok(false),
    }
}

/// An integer index array from one entry of a subscript: its values, flattened,
/// and the shape they came in.
///
/// `None` when the entry is not an integer array -- an `int`, a `slice`, `None`
/// and `...` all mean something else and are handled by the basic path. A
/// boolean array is not one either: it is a mask, and masks select a different
/// number of elements than they contain.
pub(crate) fn integer_index_array(item: &Bound<PyAny>) -> PyResult<Option<(Vec<i64>, Vec<usize>)>> {
    // The basic entries by type, not by asking each for an integer: a slice
    // refuses by raising, and that was most of what `x[1:3]` cost. Anything
    // else that is an integer -- a NumPy one, an `__index__` -- is none of
    // the forms below and comes out as `None` at the end.
    if item.is_exact_instance_of::<PyInt>()
        || item.is_instance_of::<pyo3::types::PyBool>()
        || item.is_exact_instance_of::<PySlice>()
        || item.is_none()
        || is_ellipsis(item)
    {
        return Ok(None);
    }
    if let Some(pt) = borrow_wrapped_tensor(item) {
        let t = pt.tensor();
        if !matches!(t.dtype(), DataType::Int32 | DataType::Int64) || t.ndim() == 0 {
            return Ok(None);
        }
        let shape = t.shape().dims().to_vec();
        let t = t.contiguous().map_err(_convert_error)?;
        let values: Vec<i64> = match t.dtype() {
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
        return Ok(Some((values, shape)));
    }
    if let Some((values, shape)) = numpy_integer_values(item)? {
        return Ok((!shape.is_empty()).then_some((values, shape)));
    }
    if let Ok(list) = item.cast::<PyList>() {
        // A bool leaf means a mask, and anything that is not an integer is not
        // a subscript this path can answer for. An empty list indexes nothing,
        // which is a shape rather than a refusal.
        match first_list_leaf(item)? {
            // `[]` indexes nothing. `[[]]` has no leaf either, but it has a
            // shape, so it goes the same way as any other nested list.
            None if list.is_empty() => return Ok(Some((Vec::new(), vec![0]))),
            Some(leaf)
                if leaf.is_instance_of::<pyo3::types::PyBool>()
                    || leaf.extract::<i64>().is_err() =>
            {
                return Ok(None);
            }
            None | Some(_) => {}
        }
        // A flat list is the common case and is read straight off; a nested one
        // is an index array of two dimensions or more, and the tensor
        // conversion already knows how to measure it.
        let mut values = Vec::with_capacity(list.len());
        for entry in list.iter() {
            if entry.is_instance_of::<pyo3::types::PyBool>() {
                return Err(bool_among_positions());
            }
            match entry.extract::<i64>() {
                Ok(v) => values.push(v),
                Err(_) => {
                    if holds_bool(item)? {
                        return Err(bool_among_positions());
                    }
                    let nested =
                        convert_python_data_to_tensor(item, DataType::Int64, Device::cpu(), false)?;
                    let shape = nested.shape().dims().to_vec();
                    let values = nested
                        .data()
                        .as_i64_slice()
                        .ok_or_else(|| PyRuntimeError::new_err("failed to read index list"))?
                        .to_vec();
                    return Ok(Some((values, shape)));
                }
            }
        }
        let len = values.len();
        return Ok(Some((values, vec![len])));
    }
    Ok(None)
}

/// Wrap negative positions and bounds-check against an axis of `dim_size`.
fn resolve_index_values(values: &[i64], axis: usize, dim_size: usize) -> PyResult<Vec<i64>> {
    let extent = dim_size as i64;
    let mut resolved = Vec::with_capacity(values.len());
    for &v in values {
        let wrapped = if v < 0 { v + extent } else { v };
        if wrapped < 0 || wrapped >= extent {
            return Err(PyIndexError::new_err(format!(
                "index {v} is out of bounds for dimension {axis} with size {dim_size}"
            )));
        }
        resolved.push(wrapped);
    }
    Ok(resolved)
}

/// Extract a boolean mask tensor from a `__getitem__`/`__setitem__` key when
/// the key is a bool tensor, a bool ndarray, or a (nested) list of bools.
pub(crate) fn try_bool_mask_key(key: &Bound<PyAny>) -> PyResult<Option<Tensor>> {
    if let Some(pt) = borrow_wrapped_tensor(key) {
        if pt.tensor().dtype() == DataType::Bool {
            return Ok(Some(pt.inner.clone()));
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

/// Fancy `__getitem__` forms: a boolean mask selects blocks along
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

    if let Some(pt) = borrow_wrapped_tensor(key) {
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

    if let Some((values, shape)) = numpy_integer_values(key)? {
        if shape.len() == 1 {
            return select_rows(reference, &values).map(Some);
        }
        return Ok(None);
    }

    if let Ok(list) = key.cast::<PyList>() {
        // Bool lists were handled above; a flat list of ints selects rows.
        // (An empty list selects zero rows.)
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

/// One axis's worth of advanced index, as it was written.
///
/// A mask cannot be resolved where it is read: its length has to be checked
/// against the axis it lands on, and that axis is only known once the entries
/// before it have been walked.
enum AxisIndex {
    /// Positions, flattened, and the shape they came in.
    Positions(Vec<i64>, Vec<usize>),
    /// A mask's flags in row-major order, and its shape: one flag per position
    /// across the `shape.len()` axes it spans.
    Mask(Vec<bool>, Vec<usize>),
}

/// The advanced index in one entry of a subscript, if it is one. An `int`, a
/// `slice`, `None` and `...` are all basic and give `None` here.
fn axis_index(item: &Bound<PyAny>) -> PyResult<Option<AxisIndex>> {
    if let Some((values, shape)) = integer_index_array(item)? {
        return Ok(Some(AxisIndex::Positions(values, shape)));
    }
    // A 1-D mask picks positions along one axis, so it is the same job as a
    // list of those positions. A mask of higher rank does that job across the
    // axes it spans, taken as one: `x[:, m]` with a `(3, 4)` mask over a
    // `(2, 3, 4)` tensor is `x.reshape(2, 12)[:, m.flatten()]`.
    if let Some(mask) = try_bool_mask_key(item)?
        && mask.ndim() >= 1
    {
        let shape = mask.shape().dims().to_vec();
        let mask = mask.contiguous().map_err(_convert_error)?;
        let flags = mask
            .data()
            .as_bool_slice()
            .ok_or_else(|| PyRuntimeError::new_err("failed to read index mask"))?
            .to_vec();
        return Ok(Some(AxisIndex::Mask(flags, shape)));
    }
    Ok(None)
}

/// Where the one advanced index of a subscript lands, and which positions it
/// names.
struct AdvancedIndex {
    /// Which entry of the subscript it is.
    position: usize,
    /// The first input axis it indexes.
    input_axis: usize,
    /// The extents of the input axes it spans, starting at `input_axis`: one
    /// axis for an index array, as many as it has dimensions for a mask.
    /// `selected` counts positions across all of them in row-major order.
    spanned: Vec<usize>,
    /// Where that axis sits in the shape the subscript selects.
    output_axis: usize,
    /// The positions along it, resolved and in the order written.
    selected: Vec<i64>,
    /// The index's own shape, which occupies `index_shape.len()` axes of the
    /// selection starting at `output_axis`.
    index_shape: Vec<usize>,
}

/// The entries of a subscript. A bare key is a subscript of one entry.
fn subscript_items<'py>(key: &Bound<'py, PyAny>) -> Vec<Bound<'py, PyAny>> {
    match key.cast::<PyTuple>() {
        Ok(tuple) => tuple.iter().collect(),
        Err(_) => vec![key.clone()],
    }
}

/// Find the subscript's one advanced index and resolve it against `dims`.
///
/// `None` when every entry is basic, so the caller can take the ordinary path.
/// Two advanced indices are refused: they could mean pairing up elementwise or
/// an outer product, two different operations, and guessing would be wrong
/// quietly for whoever meant the other.
fn locate_advanced_index(
    items: &[Bound<PyAny>],
    dims: &[usize],
) -> PyResult<Option<AdvancedIndex>> {
    let mut found: Option<(usize, AxisIndex)> = None;
    for (position, item) in items.iter().enumerate() {
        if let Some(index) = axis_index(item)? {
            if found.is_some() {
                return Err(PyIndexError::new_err(
                    "only one index array is supported in a subscript; index one axis at a \
                     time, or use `gather` when the arrays are meant to pair up",
                ));
            }
            found = Some((position, index));
        }
    }
    let Some((position, index)) = found else {
        return Ok(None);
    };
    let span = match &index {
        AxisIndex::Positions(..) => 1,
        AxisIndex::Mask(_, shape) => shape.len(),
    };

    // Where the index sits among the input axes, and where that axis lands in
    // the output: an integer entry drops its axis, `None` adds one, `...`
    // stands for every axis the explicit entries do not consume.
    let explicit = items
        .iter()
        .filter(|it| !it.is_none() && !is_ellipsis(it))
        .count()
        + (span - 1);
    if explicit > dims.len() {
        return Err(PyIndexError::new_err(format!(
            "too many indices for tensor: it has {} dimension(s) but {explicit} were indexed",
            dims.len()
        )));
    }
    let filled = dims.len() - explicit;
    let (mut input_axis, mut output_axis) = (0usize, 0usize);
    for item in &items[..position] {
        if item.is_none() {
            output_axis += 1;
        } else if is_ellipsis(item) {
            input_axis += filled;
            output_axis += filled;
        } else if item.extract::<i64>().is_ok() && !item.is_instance_of::<pyo3::types::PyBool>() {
            input_axis += 1;
        } else {
            input_axis += 1;
            output_axis += 1;
        }
    }
    if input_axis + span > dims.len() {
        return Err(PyIndexError::new_err(format!(
            "too many indices for tensor: it has {} dimension(s)",
            dims.len()
        )));
    }

    let spanned = dims[input_axis..input_axis + span].to_vec();
    let (selected, index_shape) = match index {
        AxisIndex::Positions(values, shape) => (
            resolve_index_values(&values, input_axis, dims[input_axis])?,
            shape,
        ),
        AxisIndex::Mask(flags, shape) => {
            if span == 1 && flags.len() != dims[input_axis] {
                return Err(PyIndexError::new_err(format!(
                    "boolean index has {} element(s) but dimension {input_axis} has size {}",
                    flags.len(),
                    dims[input_axis]
                )));
            }
            if shape != spanned {
                return Err(PyIndexError::new_err(format!(
                    "boolean index of shape {shape:?} does not match dimensions {input_axis} \
                     to {} of the tensor, which have shape {spanned:?}",
                    input_axis + span - 1
                )));
            }
            let taken: Vec<i64> = flags
                .iter()
                .enumerate()
                .filter_map(|(i, &on)| on.then_some(i as i64))
                .collect();
            let count = taken.len();
            (taken, vec![count])
        }
    };

    Ok(Some(AdvancedIndex {
        position,
        input_axis,
        spanned,
        output_axis,
        selected,
        index_shape,
    }))
}

/// The same subscript with full slices where the advanced index was, one for
/// each axis it spans, so the basic path can take everything else.
fn basic_part<'py>(
    key: &Bound<'py, PyAny>,
    items: &[Bound<'py, PyAny>],
    found: &AdvancedIndex,
) -> PyResult<Bound<'py, PyTuple>> {
    let mut basic: Vec<Bound<PyAny>> = Vec::with_capacity(items.len() + found.spanned.len());
    for (i, item) in items.iter().enumerate() {
        if i == found.position {
            basic.extend(
                found
                    .spanned
                    .iter()
                    .map(|_| PySlice::full(key.py()).into_any()),
            );
        } else {
            basic.push(item.clone());
        }
    }
    PyTuple::new(key.py(), basic)
}

/// A subscript holding one advanced index among otherwise basic entries, as
/// `x[:, idx]`, `x[1:3, idx]`, `x[..., idx]` or `x[:, mask]`.
///
/// The whole-key forms -- a bare mask, or a bare index array standing for the
/// leading axis -- are handled before this. What is left is the far commoner
/// shape of the same idea: an index array somewhere other than the front, which
/// used to be a bare `TypeError: Invalid index type` even though `index_select`
/// does exactly this and was one call away.
///
/// With exactly one index array the answer does not depend on the order the two
/// kinds are applied in, and the array's axes stay where the array was. So
/// this applies the basic subscript with a full slice in the
/// array's place and selects along the axis that leaves.
pub(crate) fn try_single_array_index(
    reference: &Tensor,
    key: &Bound<PyAny>,
) -> PyResult<Option<Tensor>> {
    // A bare key reaching here is an index array of two dimensions or more:
    // the 1-D leading case never arrives, `try_fancy_index_tensor` takes it.
    let items = subscript_items(key);
    let dims = reference.shape().dims();
    let Some(found) = locate_advanced_index(&items, dims)? else {
        return Ok(None);
    };

    let basic_key = basic_part(key, &items, &found)?;
    let (indices, newaxis_positions) = parse_getitem_indices(&basic_key, dims)?;
    // The basic part is read only by the gather below, so where it selects
    // everything it is the tensor itself, shared rather than copied: `t[:, i]`
    // copied all of `t` first, 2.0ms for a 2048x2048 float32 matrix around a
    // 19us gather.
    let selects_all = indices.iter().zip(dims).all(|(index, &dim)| {
        matches!(index, TensorIndex::Slice { start: 0, end, step: 1 } if *end == dim)
    });
    let mut base = if selects_all {
        reference.clone()
    } else {
        reference.index(&indices).map_err(_convert_error)?
    };
    for &pos in &newaxis_positions {
        base = base.unsqueeze(pos as isize).map_err(_convert_error)?;
    }
    if found.spanned.len() > 1 {
        // A mask over several axes counts its positions across all of them,
        // so they become one axis to select along.
        let mut merged = base.shape().dims().to_vec();
        let span = found.output_axis..found.output_axis + found.spanned.len();
        merged.splice(span, [found.spanned.iter().product()]);
        base = base
            .reshape(engine::tensor::Shape::new(merged))
            .map_err(_convert_error)?;
    }

    let taken =
        engine::ops::shape_ops::index_select(&base, found.output_axis as isize, &found.selected)
            .map_err(_convert_error)?;
    if found.index_shape.len() == 1 {
        return Ok(Some(taken));
    }
    // A multi-dimensional index array puts its own shape where the axis was.
    let mut shape = taken.shape().dims().to_vec();
    shape.splice(
        found.output_axis..found.output_axis + 1,
        found.index_shape.iter().copied(),
    );
    taken
        .reshape(engine::tensor::Shape::new(shape))
        .map(Some)
        .map_err(_convert_error)
}

/// An assignment through one advanced index, resolved against the tensor it
/// will write into.
pub(crate) struct AxisAssign {
    /// The basic part of the subscript, one entry per input dimension, with the
    /// indexed axis still taken in full.
    indices: Vec<TensorIndex>,
    /// The first of those entries the advanced index replaces.
    input_axis: usize,
    /// The extents of the entries it replaces, from `input_axis` on: a
    /// position in `selected` counts across all of them in row-major order.
    spanned: Vec<usize>,
    /// The shape the subscript selects, which is what a value is lined up
    /// against.
    selection: Vec<usize>,
    /// Where the index's own axes end in that shape, so a value can be asked
    /// whether it reaches them at all.
    selection_span: usize,
    /// The same shape with the axes a newaxis added dropped and the index's
    /// axes collapsed into one: what one position's share is read from.
    flat: Vec<usize>,
    /// Where that one axis sits in `flat`.
    flat_axis: usize,
    /// The positions to write, in the order they were written.
    selected: Vec<i64>,
}

/// Plan `t[..., idx, ...] = value` if that is what the subscript is.
///
/// Resolving the plan needs only the target's shape, so it happens before the
/// mutable borrow the write itself takes -- which is also what lets
/// `t[:, idx] = t` extract its value without hitting an already-borrowed error.
pub(crate) fn plan_single_array_assign(
    key: &Bound<PyAny>,
    dims: &[usize],
) -> PyResult<Option<AxisAssign>> {
    let items = subscript_items(key);
    let Some(found) = locate_advanced_index(&items, dims)? else {
        return Ok(None);
    };

    let basic_key = basic_part(key, &items, &found)?;
    let (indices, newaxis_positions) = parse_getitem_indices(&basic_key, dims)?;
    let base: Vec<usize> = indices
        .iter()
        .filter_map(|index| match index {
            TensorIndex::Index(_) => None,
            TensorIndex::Slice { start, end, step } => {
                Some(end.saturating_sub(*start).div_ceil((*step).max(1)))
            }
        })
        .collect();

    // The value is lined up against the shape the subscript selects, so the
    // axes a newaxis adds have to be in that even though they name nothing to
    // write into. The shares are read from the same thing without them: they
    // are extent one, so dropping them is a reshape and nothing moves.
    let mut selection = base.clone();
    for &position in &newaxis_positions {
        selection.insert(position.min(selection.len()), 1);
    }
    let span = found.spanned.len();
    selection.splice(
        found.output_axis..found.output_axis + span,
        found.index_shape.iter().copied(),
    );
    let added_before = newaxis_positions
        .iter()
        .filter(|&&position| position < found.output_axis)
        .count();
    let flat_axis = found.output_axis - added_before;
    let mut flat = base;
    flat.splice(flat_axis..flat_axis + span, [found.selected.len()]);

    Ok(Some(AxisAssign {
        indices,
        input_axis: found.input_axis,
        spanned: found.spanned,
        selection_span: found.output_axis + found.index_shape.len(),
        selection,
        flat,
        flat_axis,
        selected: found.selected,
    }))
}

/// Write `value` into the positions a plan names, one position at a time.
///
/// Each position is an ordinary basic assignment, so the write goes through the
/// same shared storage the rest of `__setitem__` uses -- assigning to a
/// parameter still reaches the layer -- and a position named twice keeps the
/// last write.
pub(crate) fn apply_single_array_assign(
    target: &mut Tensor,
    plan: &AxisAssign,
    value: &Tensor,
) -> PyResult<()> {
    if plan.selected.is_empty() {
        return Ok(());
    }
    // The value is read between writes, so `t[:, [0, 1]] = t[:, [1, 0]]` needs
    // a copy to read from: by the second position the first is already gone.
    let mut value = value.deep_clone().map_err(_convert_error)?;
    // Leading axes of extent one need not be spelled out in the value, which is
    // what the basic path allows too.
    while value.ndim() > plan.selection.len() && value.shape().dims()[0] == 1 {
        let rest: Vec<usize> = value.shape().dims()[1..].to_vec();
        value = value
            .reshape(engine::tensor::Shape::new(rest))
            .map_err(_convert_error)?;
    }

    let mut dest = plan.indices.clone();
    if plan.selection.len().saturating_sub(value.ndim()) >= plan.selection_span {
        // Lined up from the right, the value stops short of the indexed axes,
        // so every position gets all of it -- `t[:, idx] = row`, and the scalar
        // case with it. Nothing has to be materialised for that.
        for &position in &plan.selected {
            plan.place(&mut dest, position);
            target.index_assign(&dest, &value).map_err(_convert_error)?;
        }
        return Ok(());
    }

    // Otherwise the value spans the indexed axes and each position takes its
    // own share of it. Broadcasting to the selection first means the shares can
    // be read straight off, and collapsing the index's axes into one makes a
    // position one step along it whatever shape the index came in.
    let broadcast = value
        .expand(plan.selection.iter().map(|&dim| dim as isize).collect())
        .and_then(|value| value.contiguous())
        .and_then(|value| value.reshape(engine::tensor::Shape::new(plan.flat.clone())))
        .map_err(|_| {
            // The expansion is an implementation detail; what the caller wrote
            // is a value and a subscript, so name those instead.
            PyValueError::new_err(format!(
                "cannot broadcast a value of shape {:?} into the selection of shape {:?}",
                value.shape().dims(),
                plan.selection
            ))
        })?;

    let mut share: Vec<TensorIndex> = plan.flat.iter().map(|&dim| full_slice(dim)).collect();
    for (step, &position) in plan.selected.iter().enumerate() {
        share[plan.flat_axis] = TensorIndex::Index(step);
        let part = broadcast.index(&share).map_err(_convert_error)?;
        plan.place(&mut dest, position);
        target.index_assign(&dest, &part).map_err(_convert_error)?;
    }
    Ok(())
}

impl AxisAssign {
    /// Point the replaced entries of `dest` at one selected position, which
    /// for a mask over several axes means one coordinate on each.
    fn place(&self, dest: &mut [TensorIndex], position: i64) {
        let mut rest = position as usize;
        for (offset, &extent) in self.spanned.iter().enumerate().rev() {
            dest[self.input_axis + offset] = TensorIndex::Index(rest % extent);
            rest /= extent;
        }
    }
}

fn full_slice(dim: usize) -> TensorIndex {
    TensorIndex::Slice {
        start: 0,
        end: dim,
        step: 1,
    }
}

/// Parse one entry of a subscript against the axis it addresses.
///
/// `axis` is carried purely for the error message: an out-of-range subscript
/// used to raise a bare "Index out of bounds", which told the caller neither
/// which axis was overrun nor how long it is — the two facts needed to fix the
/// call. The engine's `IndexError` already reports all three, so match it.
fn parse_index(item: &Bound<PyAny>, axis: usize, dim_size: usize) -> PyResult<TensorIndex> {
    // A bool converts to an integer, so `x[True]` was `x[1]` and `x[:, False]`
    // was `x[:, 0]`: a condition that happened to be a Python bool picked a
    // row without a word. A mask is a bool tensor or list; a lone bool is
    // neither a mask nor a position.
    if item.is_instance_of::<pyo3::types::PyBool>() {
        return Err(PyTypeError::new_err(format!(
            "a bool cannot index axis {axis}: it is not a position, and reading it as one \
             would pick {} without saying so. Index with an int to pick a position, or with a \
             bool tensor or list to select by a mask",
            if item.is_truthy()? {
                "position 1"
            } else {
                "position 0"
            }
        )));
    }
    // A slice or `None` is not asked for an integer first: the refusal is
    // raised and its message formatted, which cost more than the slice.
    let basic = item.is_exact_instance_of::<PySlice>() || item.is_none();
    if !basic && let Ok(i) = item.extract::<isize>() {
        let mut idx = i;
        if idx < 0 {
            idx += dim_size as isize;
        }
        if idx < 0 || idx >= dim_size as isize {
            return Err(PyIndexError::new_err(format!(
                "index {i} is out of bounds for dimension {axis} with size {dim_size}"
            )));
        }
        Ok(TensorIndex::Index(idx as usize))
    } else if let Ok(slice) = item.cast::<PySlice>() {
        use std::convert::TryInto;

        let dim_size_isize: isize = dim_size
            .try_into()
            .map_err(|_| PyValueError::new_err("dim_size too large"))?;
        let indices = slice.indices(dim_size_isize)?;
        if indices.step < 0 {
            // Reversing by subscript is not supported, but "slice step must
            // be positive" stated the rule and left
            // the caller to find the remedy. `flip` is the remedy, and a step
            // other than -1 needs the positive stride afterwards -- applied to
            // the *same* axis, so the leading colons have to be there.
            let stride = -indices.step;
            let follow_up = if stride == 1 {
                String::new()
            } else {
                format!("[{}::{stride}]", ":, ".repeat(axis))
            };
            return Err(PyIndexError::new_err(format!(
                "negative slice step is not supported: axis {axis} was indexed with a step of \
                 {step}. Reverse that axis instead -- `x.flip({axis}){follow_up}` selects the \
                 same elements in the same order.",
                step = indices.step
            )));
        }
        if indices.step == 0 {
            return Err(PyIndexError::new_err("slice step cannot be zero"));
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
        Err(PyTypeError::new_err(format!(
            "{} cannot index a tensor: an index is an int, a slice, None, ..., or an int or \
             bool tensor, array or list",
            item.get_type().fully_qualified_name()?
        )))
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
        return Err(PyIndexError::new_err(format!(
            "too many indices for tensor: it has {} dimension(s) but {real_count} were indexed",
            shape.len()
        )));
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
        let idx = parse_index(item, input_dim, shape[input_dim])?;
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
    // `None`/`np.newaxis` adds an axis to the selection rather than naming one
    // to write into, so it changes nothing about where the value lands and
    // does not count against the tensor's rank.
    let real_count = items
        .iter()
        .filter(|it| !it.is_none() && !is_ellipsis(it))
        .count();
    if real_count > shape.len() {
        return Err(PyIndexError::new_err(format!(
            "too many indices for tensor: it has {} dimension(s) but {real_count} were indexed",
            shape.len()
        )));
    }

    let mut result: Vec<TensorIndex> = Vec::with_capacity(shape.len());
    let mut input_dim = 0usize;
    for item in &items {
        if item.is_none() {
            continue;
        }
        if is_ellipsis(item) {
            for _ in 0..shape.len() - real_count {
                result.push(full_slice(shape[input_dim]));
                input_dim += 1;
            }
            continue;
        }
        result.push(parse_index(item, input_dim, shape[input_dim])?);
        input_dim += 1;
    }
    for &dim in &shape[input_dim..] {
        result.push(full_slice(dim));
    }
    Ok(result)
}

/// Build a tensor from a Python sequence by way of `numpy.asarray`.
///
/// Returns `None` whenever NumPy cannot produce a buffer this crate supports,
/// so the caller can fall back to the element-by-element traversal and keep its
/// exact errors. The requested `dtype` still governs the result: a list of
/// Python floats becomes the library's default float dtype, not NumPy's
/// float64, because the caller resolved that before calling -- or, with no
/// `dtype`, because [`sequence_kind`] reads it off the array that holds the
/// values.
fn sequence_via_numpy(
    data: &Bound<PyAny>,
    dtype: Option<DataType>,
    device: Device,
    requires_grad: bool,
) -> Option<Tensor> {
    if !device.is_cpu() {
        return None;
    }
    // Ragged input raises here in NumPy 2; that is a fall-through, not an
    // error, so the message the slow path produces is the one users see.
    let array = numpy_asarray(data)?;
    let dtype = match dtype {
        Some(dtype) => dtype,
        None => sequence_kind(&array)?,
    };
    // `convert_numpy_to_tensor` reaches into the array's buffer, which panics
    // rather than erroring if the capsule is unavailable -- the same guard the
    // ndarray branch above uses.
    let tensor = panic::catch_unwind(AssertUnwindSafe(|| {
        convert_numpy_to_tensor(&array, requires_grad)
    }))
    .ok()?
    .ok()?;

    // A value that does not fit is left to the walk below, which reports it.
    python_values_as(tensor, dtype).ok()
}

/// `tensor`, holding values read from Python, at `dtype`.
///
/// A Python number has no width of its own: it takes the dtype it is given,
/// and one that dtype cannot hold is refused rather than wrapped or clamped.
/// Narrowed to int32, 2**40 keeps its low 32 bits and becomes 0 -- a wrong
/// value with nothing to say so. A float bound for an integer dtype is
/// truncated, as any float-to-integer conversion is, but NaN, an infinity or
/// a value whose truncation is out of range has no integer to truncate to: the
/// cast saturated them to 0 and the dtype's bounds, so `fill_(inf)` on int32
/// wrote 2147483647.
fn python_values_as(tensor: Tensor, dtype: DataType) -> PyResult<Tensor> {
    if tensor.dtype() == dtype {
        return Ok(tensor);
    }
    if tensor.dtype() == DataType::Int64
        && dtype == DataType::Int32
        && let Some(values) = tensor.data().as_i64_slice()
        && let Some(value) = values.iter().find(|&&v| i32::try_from(v).is_err())
    {
        return Err(does_not_fit(value, DataType::Int32));
    }
    if tensor.dtype().is_float() && dtype.is_int() {
        match tensor.dtype() {
            DataType::Float64 => {
                for &value in tensor.data().as_f64_slice().unwrap_or(&[]) {
                    check_float_fits(value, dtype)?;
                }
            }
            _ => {
                for &value in tensor.data().as_f32_slice().unwrap_or(&[]) {
                    check_float_fits(f64::from(value), dtype)?;
                }
            }
        }
    }
    tensor.astype(dtype).map_err(_convert_error)
}

/// Refuse a float with no integer of `dtype` to truncate to: NaN, an infinity,
/// or one whose truncation is out of the dtype's range. Every other float
/// truncates, as any float-to-integer conversion does.
pub(crate) fn check_float_fits(value: f64, dtype: DataType) -> PyResult<()> {
    // The exclusive upper limit is a power of two, so it is exact as an f64
    // where `i64::MAX` is not.
    let (low, above) = match dtype {
        DataType::Int32 => (f64::from(i32::MIN), f64::from(i32::MAX) + 1.0),
        DataType::Int64 => (i64::MIN as f64, -(i64::MIN as f64)),
        _ => return Ok(()),
    };
    if value.is_finite() && (low..above).contains(&value.trunc()) {
        return Ok(());
    }
    let shown = if value.is_nan() {
        "nan".to_string()
    } else if value.is_infinite() {
        if value > 0.0 { "inf" } else { "-inf" }.to_string()
    } else {
        value.to_string()
    };
    Err(does_not_fit(shown, dtype))
}

/// The error for a Python int outside the range of the integer `dtype` it
/// was given.
pub(crate) fn does_not_fit(value: impl std::fmt::Display, dtype: DataType) -> PyErr {
    let (name, low, high) = match dtype {
        DataType::Int32 => ("int32", i64::from(i32::MIN), i64::from(i32::MAX)),
        _ => ("int64", i64::MIN, i64::MAX),
    };
    PyErr::new::<pyo3::exceptions::PyOverflowError, _>(format!(
        "{value} does not fit in {name} (range {low} to {high})"
    ))
}

/// The supported dtype a NumPy dtype widens to, if any.
///
/// The engine carries five dtypes; NumPy has many more. Rather than refuse the
/// rest, cast the ones that widen *exactly* -- every value round-trips, so the
/// conversion cannot change a number. `float16` fits in `float32`'s 24-bit
/// mantissa; `int8`/`int16`/`uint8`/`uint16` fit in `int32`; `uint32` fits in
/// `int64`.
///
/// `uint64` and `longdouble` are deliberately absent: values above
/// `i64::MAX`, and mantissas wider than `float64`'s, cannot survive the cast,
/// and silently rounding a user's data is worse than telling them to choose
/// the cast themselves.
fn widened_numpy_dtype(kind: u8, itemsize: usize) -> Option<&'static str> {
    match (kind, itemsize) {
        (b'f', 2) => Some("float32"),
        (b'i', 1) | (b'i', 2) => Some("int32"),
        (b'u', 1) | (b'u', 2) => Some("int32"),
        (b'u', 4) => Some("int64"),
        _ => None,
    }
}

/// Read a NumPy array's dtype as `(kind, itemsize)`, e.g. `(b'u', 1)` for
/// uint8.
///
/// Off the descriptor for an array. Asking the dtype object for the two
/// attributes, and building a `String` for the one-letter kind, was a third
/// of `from_numpy` on a small array -- every array operand pays it.
fn numpy_dtype_parts(array: &Bound<PyAny>) -> PyResult<(u8, usize)> {
    if let Ok(untyped) = array.cast::<PyUntypedArray>() {
        let descr = untyped.dtype();
        return Ok((descr.kind(), descr.itemsize()));
    }
    let dtype = array.getattr(intern!(array.py(), "dtype"))?;
    let kind: String = dtype.getattr(intern!(array.py(), "kind"))?.extract()?;
    let itemsize: usize = dtype.getattr(intern!(array.py(), "itemsize"))?.extract()?;
    Ok((kind.bytes().next().unwrap_or(b'?'), itemsize))
}

/// Cast a NumPy array to a dtype the engine supports, when that is exact.
///
/// Leaves supported dtypes untouched, and leaves the lossy ones alone too so
/// the caller reports them rather than rounding them.
fn widen_numpy_dtype<'py>(array: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let (kind, itemsize) = numpy_dtype_parts(array)?;
    match widened_numpy_dtype(kind, itemsize) {
        Some(target) => array.call_method1(intern!(array.py(), "astype"), (target,)),
        None => Ok(array.clone()),
    }
}

/// Force C-contiguous element order before any buffer is read.
///
/// `PyReadonlyArray::as_slice` accepts a Fortran-contiguous array -- there are
/// no gaps in it -- and hands back the buffer in column-major order, which the
/// callers below then pair with the row-major shape. That silently transposed
/// the data: `as_tensor(x.T)` returned a tensor of the right shape holding the
/// wrong values, with no error, for an input as ordinary as a transpose.
///
/// `np.ascontiguousarray` returns the same object when the array is already
/// C-contiguous, so the common path does not copy.
fn as_c_contiguous<'py>(array: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(untyped) = array.cast::<numpy::PyUntypedArray>()
        && untyped.is_c_contiguous()
    {
        return Ok(array.clone());
    }
    PyModule::import(array.py(), "numpy")?.call_method1("ascontiguousarray", (array,))
}

pub(crate) fn convert_numpy_to_tensor(
    array: &Bound<PyAny>,
    requires_grad: bool,
) -> PyResult<Tensor> {
    let array = &as_c_contiguous(&widen_numpy_dtype(array)?)?;
    // One arm per supported dtype, written once: they differed only in the
    // element type and the tag, and the copy is the same parallel one for all
    // of them -- 64MB of float32 took 63ms copied on one core and 25 split.
    //
    // The array is read without rust-numpy's dynamic borrow, whose
    // registration and release in a process-wide table were half of what
    // `convert_numpy_to_tensor` cost on a small array. It guards against a
    // conflicting borrow held in Rust, and there can be none: the bindings
    // never release the GIL, so no other Rust runs while this copy does, and
    // nothing between here and the end of the copy calls back into Python.
    macro_rules! from_array {
        ($ty:ty, $dtype:expr) => {
            if let Ok(typed) = array.cast::<PyArrayDyn<$ty>>() {
                let shape = Shape::new(typed.shape().to_vec());
                // SAFETY: see above; the array is C-contiguous
                // (`as_c_contiguous`), which `as_slice` checks as well.
                let values = unsafe { typed.as_slice() }?;
                let data =
                    TensorData::from_vec(TensorData::copied_slice(values), $dtype, Device::cpu());
                return Ok(Tensor::new(
                    Arc::new(data),
                    shape,
                    $dtype,
                    Device::cpu(),
                    requires_grad,
                ));
            }
        };
    }
    from_array!(f32, DataType::Float32);
    from_array!(f64, DataType::Float64);
    from_array!(i32, DataType::Int32);
    from_array!(i64, DataType::Int64);
    // Not through `from_array!`: that would read the array as `&[bool]`, and a
    // NumPy bool array can hold any byte -- `np.array([2], np.uint8)
    // .view(bool)` is an ordinary one, and prints `[ True]`. Every byte but 0
    // and 1 is undefined behavior as a Rust `bool`, so the bytes are read as
    // what they are and normalised the way NumPy reads them: not zero is true.
    if let Ok(typed) = array.cast::<PyArrayDyn<bool>>() {
        let shape = Shape::new(typed.shape().to_vec());
        let len = typed.len();
        let bytes: &[u8] = if len == 0 {
            &[]
        } else {
            // SAFETY: the array is C-contiguous (`as_c_contiguous` above) with
            // `len` one-byte elements, nothing writes it while the GIL is held
            // (see `from_array!`), and any byte is a valid `u8`.
            unsafe { std::slice::from_raw_parts(typed.data().cast::<u8>(), len) }
        };
        let values: Vec<bool> = bytes.iter().map(|&byte| byte != 0).collect();
        let data = TensorData::from_vec(values, DataType::Bool, Device::cpu());
        return Ok(Tensor::new(
            Arc::new(data),
            shape,
            DataType::Bool,
            Device::cpu(),
            requires_grad,
        ));
    }
    {
        let described = numpy_dtype_parts(array)
            .map(|(kind, size)| format!("{}{}", kind as char, size * 8))
            .unwrap_or_else(|_| "unknown".to_string());
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
            "Unsupported NumPy dtype '{described}'. Supported dtypes are \
             float32, float64, int32, int64 and bool; float16, int8, int16, \
             uint8, uint16 and uint32 are widened automatically. Cast \
             explicitly (for example `.astype('int64')`) to choose how values \
             that do not fit should be handled."
        )))
    }
}

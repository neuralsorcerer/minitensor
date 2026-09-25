// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
fn convert_dimension(value: isize, arg_name: &str) -> PyResult<usize> {
    if value < 0 {
        return Err(PyValueError::new_err(format!(
            "{arg_name} must contain non-negative integers",
        )));
    }

    usize::try_from(value).map_err(|_| {
        PyValueError::new_err(format!("{arg_name} value is too large for this platform",))
    })
}

fn convert_dimensions(values: Vec<isize>, arg_name: &str) -> PyResult<Vec<usize>> {
    let mut dims = Vec::with_capacity(values.len());
    for value in values {
        dims.push(convert_dimension(value, arg_name)?);
    }
    Ok(dims)
}

pub(crate) fn normalize_variadic_isize_args(
    tuple: &Bound<PyTuple>,
    arg_name: &str,
) -> PyResult<Vec<isize>> {
    // `(((3,),),)` means `(3,)`. The singleton tuples are peeled in a loop:
    // peeling them by recursion overflowed the stack, and crashed the
    // interpreter, on a hundred thousand of them.
    let peeled = peel_singleton_tuples(tuple)?;
    let tuple = &peeled;
    if tuple.is_empty() {
        return Ok(Vec::new());
    }

    if tuple.len() == 1 {
        let first = tuple.get_item(0)?;

        if let Ok(list) = first.cast::<PyList>() {
            let mut dims = Vec::with_capacity(list.len());
            for item in list.iter() {
                dims.push(item.extract::<isize>()?);
            }
            return Ok(dims);
        }

        if let Ok(shape_sequence) = first.extract::<ShapeSequence>() {
            return convert_usize_list_to_isize(shape_sequence.to_list(), arg_name);
        }

        if let Ok(values) = first.extract::<Vec<isize>>() {
            return Ok(values);
        }

        if let Ok(values) = first.extract::<Vec<usize>>() {
            return convert_usize_list_to_isize(values, arg_name);
        }

        if let Ok(value) = first.extract::<isize>() {
            return Ok(vec![value]);
        }

        if let Ok(value) = first.extract::<usize>() {
            return Ok(vec![convert_usize_to_isize(value, arg_name)?]);
        }
    }

    let mut dims = Vec::with_capacity(tuple.len());
    for item in tuple.iter() {
        dims.push(item.extract::<isize>()?);
    }
    Ok(dims)
}

fn convert_usize_list_to_isize(values: Vec<usize>, arg_name: &str) -> PyResult<Vec<isize>> {
    let mut converted = Vec::with_capacity(values.len());
    for value in values {
        converted.push(convert_usize_to_isize(value, arg_name)?);
    }
    Ok(converted)
}

fn convert_usize_to_isize(value: usize, arg_name: &str) -> PyResult<isize> {
    isize::try_from(value).map_err(|_| {
        PyValueError::new_err(format!(
            "{arg_name} dimension {value} is too large for this platform"
        ))
    })
}

/// Reject a shape whose dimensions multiply past `usize` before it reaches the
/// engine.
///
/// Each dimension is already checked on its own, but nothing checked the
/// product, and `Shape::numel` computes that with `checked_mul` and *panics* on
/// overflow -- deliberately, since a wrapped element count would under-allocate
/// storage that indexing code still trusts. That panic is meant as the last
/// line of defence, not as what a caller sees: `mt.zeros(2**32, 2**32)` reached
/// it straight from Python. `reshape` already validates its own dimensions this
/// way and returns an error; this gives every other shape argument the same
/// treatment, and leaves the panic underneath for anything that slips past.
fn reject_overflowing_shape(dims: Vec<usize>, arg_name: &str) -> PyResult<Vec<usize>> {
    if dims
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .is_none()
    {
        return Err(PyValueError::new_err(format!(
            "{arg_name} {dims:?} has more elements than this platform can represent"
        )));
    }
    Ok(dims)
}

/// A byte count at a scale a reader can hold in their head.
///
/// The requests this refuses are often absurd -- `mt.zeros([10**18])` asks for
/// 3.5 EiB -- and "3725290298.5 GiB" does not read as a number, which is the
/// moment a caller stops reading the message and starts guessing.
fn human_bytes(bytes: usize) -> String {
    const UNITS: [&str; 7] = ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

/// Refuse an allocation this machine cannot make, instead of aborting on it.
///
/// [`reject_overflowing_shape`] covers the element *count*; this covers the
/// bytes. A count well inside `usize` can still ask for more memory than
/// exists -- `mt.zeros([10**18])` is 4 exabytes in the default dtype -- and
/// that request did not raise. Rust's `Vec` allocation calls
/// `handle_alloc_error` when the allocator says no, which **aborts the
/// process**: not a Python exception, not even a `PanicException`, just the
/// interpreter gone and with it whatever was in it. One mistyped exponent in a
/// notebook took the kernel down.
///
/// `TensorData::is_allocatable_size` is that question, and its note explains
/// why it takes two forms rather than one. This used to ask `try_reserve`
/// here, in a second copy of the same policy, and the copy is what let the two
/// drift: the engine's grew a physical-memory ceiling after macOS was found to
/// grant reservations it then kills the process over, and this one would not
/// have. One decision, two messages -- a Python caller gets `MemoryError` and
/// a byte count it can read, which is not what the engine's `Result` says.
pub(crate) fn reject_unallocatable(numel: usize, dtype: DataType, what: &str) -> PyResult<()> {
    let Some(bytes) = numel.checked_mul(dtype.size_bytes()) else {
        return Err(PyValueError::new_err(format!(
            "{what} of {numel} {dtype:?} elements is larger than this platform can address"
        )));
    };

    if !TensorData::is_allocatable_size(bytes) {
        return Err(PyMemoryError::new_err(format!(
            "{what} of {numel} {dtype:?} elements needs {}, which this machine cannot hold",
            human_bytes(bytes)
        )));
    }
    Ok(())
}

pub(crate) fn parse_shape_tuple(shape: &Bound<PyTuple>, arg_name: &str) -> PyResult<Vec<usize>> {
    reject_overflowing_shape(parse_shape_tuple_dims(shape, arg_name)?, arg_name)
}

/// `tuple` with every enclosing singleton tuple removed.
fn peel_singleton_tuples<'py>(tuple: &Bound<'py, PyTuple>) -> PyResult<Bound<'py, PyTuple>> {
    let mut current = tuple.clone();
    while current.len() == 1 {
        match current.get_item(0)?.cast_into::<PyTuple>() {
            Ok(inner) => current = inner,
            Err(_) => break,
        }
    }
    Ok(current)
}

fn parse_shape_tuple_dims(shape: &Bound<PyTuple>, arg_name: &str) -> PyResult<Vec<usize>> {
    // `(((3,),),)` means `(3,)`. The singleton tuples are peeled in a loop:
    // peeling them by recursion overflowed the stack, and crashed the
    // interpreter, on a hundred thousand of them.
    let peeled = peel_singleton_tuples(shape)?;
    let shape = &peeled;
    if shape.is_empty() {
        return Ok(Vec::new());
    }

    if shape.len() == 1 {
        let first = shape.get_item(0)?;
        if let Ok(list) = first.cast::<PyList>() {
            let mut dims = Vec::with_capacity(list.len());
            for item in list.iter() {
                let value: isize = item.extract()?;
                dims.push(convert_dimension(value, arg_name)?);
            }
            return Ok(dims);
        }
        if let Ok(shape_seq) = first.extract::<ShapeSequence>() {
            return Ok(shape_seq.to_list());
        }
        if let Ok(values) = first.extract::<Vec<isize>>() {
            return convert_dimensions(values, arg_name);
        }
        if let Ok(value) = first.extract::<isize>() {
            return Ok(vec![convert_dimension(value, arg_name)?]);
        }
    }

    let mut dims = Vec::with_capacity(shape.len());
    for item in shape.iter() {
        let value: isize = item.extract()?;
        dims.push(convert_dimension(value, arg_name)?);
    }
    Ok(dims)
}

pub(crate) fn parse_shape_like(obj: &Bound<PyAny>, arg_name: &str) -> PyResult<Vec<usize>> {
    reject_overflowing_shape(parse_shape_like_dims(obj, arg_name)?, arg_name)
}

fn parse_shape_like_dims(obj: &Bound<PyAny>, arg_name: &str) -> PyResult<Vec<usize>> {
    if let Ok(tuple) = obj.cast::<PyTuple>() {
        return parse_shape_tuple_dims(tuple, arg_name);
    }

    if let Ok(list) = obj.cast::<PyList>() {
        let mut dims = Vec::with_capacity(list.len());
        for item in list.iter() {
            let value: isize = item.extract()?;
            dims.push(convert_dimension(value, arg_name)?);
        }
        return Ok(dims);
    }

    if let Ok(shape_seq) = obj.extract::<ShapeSequence>() {
        return Ok(shape_seq.to_list());
    }

    if let Ok(values) = obj.extract::<Vec<isize>>() {
        return convert_dimensions(values, arg_name);
    }

    if let Ok(value) = obj.extract::<isize>() {
        return Ok(vec![convert_dimension(value, arg_name)?]);
    }

    Err(PyTypeError::new_err(format!(
        "{arg_name} must be an int or sequence of ints",
    )))
}

pub(crate) fn normalize_roll_shifts(shifts: &Bound<PyAny>) -> PyResult<Vec<isize>> {
    normalize_required_axes(shifts, "shifts")
}

pub(crate) fn normalize_required_axes<'py>(
    dim: &'py Bound<'py, PyAny>,
    name: &str,
) -> PyResult<Vec<isize>> {
    match normalize_optional_axes(Some(dim))? {
        Some(values) => Ok(values),
        None => Err(PyTypeError::new_err(format!(
            "{} must be an int or a sequence of ints",
            name
        ))),
    }
}

/// The single axis a reduction that reports an index takes.
///
/// `max(dim)`, `median(dim)`, `argmax(dim)` and the rest of the index-reporting
/// reductions name a position, and a position is a position along one axis, so
/// they cannot take a list of them. That is the whole difference between `max`
/// and `amax`, and between `median` and `nanmedian`, both of which reduce as
/// many axes as they are given.
///
/// Passing several used to reach pyo3's `isize` conversion and come back as
/// "'list' object cannot be interpreted as an integer", which says what failed
/// but nothing about why one axis is wanted or where to go for several.
pub(crate) fn normalize_indexed_axis(
    dim: Option<&Bound<PyAny>>,
    op: &str,
    instead: Option<&str>,
) -> PyResult<Option<isize>> {
    match normalize_optional_axes(dim)? {
        None => Ok(None),
        Some(axes) if axes.len() == 1 => Ok(Some(axes[0])),
        Some(axes) => Err(PyTypeError::new_err(format!(
            "{op} reports an index, which names a position along one axis, so \
             it takes a single dim -- got {}.{}",
            axes.len(),
            instead.map(|line| format!(" {line}")).unwrap_or_default()
        ))),
    }
}

pub(crate) fn normalize_optional_axes(dim: Option<&Bound<PyAny>>) -> PyResult<Option<Vec<isize>>> {
    let Some(obj) = dim else {
        return Ok(None);
    };

    if obj.is_none() {
        return Ok(None);
    }

    if is_bool_axis(obj)? {
        return Err(PyTypeError::new_err(
            "dim must be an int or a sequence of ints",
        ));
    }

    if let Ok(value) = obj.extract::<isize>() {
        return Ok(Some(vec![value]));
    }

    if obj.is_instance_of::<PyString>() {
        return Err(PyTypeError::new_err(
            "dim must be an int or a sequence of ints",
        ));
    }

    if let Ok(sequence) = obj.cast::<PySequence>() {
        let length = sequence.len()?;
        let mut axes = Vec::with_capacity(length);
        for index in 0..length {
            let item = sequence.get_item(index)?;
            if is_bool_axis(&item)? {
                return Err(PyTypeError::new_err(
                    "dim must be an int or a sequence of ints",
                ));
            }
            let value: isize = item.extract()?;
            axes.push(value);
        }
        return Ok(Some(axes));
    }

    Err(PyTypeError::new_err(
        "dim must be an int or a sequence of ints",
    ))
}

fn is_bool_axis(obj: &Bound<PyAny>) -> PyResult<bool> {
    if obj.is_instance_of::<PyBool>() {
        return Ok(true);
    }

    // `None` records that numpy is unavailable, which is also worth caching:
    // the previous `get_or_try_init` did not remember failures, so every call
    // on a numpy-less install retried the import.
    static NUMPY_BOOL_TYPE: OnceLock<Option<Py<PyAny>>> = OnceLock::new();
    let py = obj.py();
    let numpy_bool = NUMPY_BOOL_TYPE.get_or_init(|| {
        PyModule::import(py, "numpy")
            .and_then(|numpy| numpy.getattr("bool_"))
            .map(|bool_obj| bool_obj.unbind())
            .ok()
    });
    if let Some(numpy_bool) = numpy_bool
        && obj.is_instance(numpy_bool.bind(py))?
    {
        return Ok(true);
    }

    Ok(false)
}

pub(crate) fn normalize_repeat_spec(repeats: &Bound<PyAny>) -> PyResult<Vec<usize>> {
    if repeats.is_instance_of::<PyString>() {
        return Ok(vec![extract_repeat_element(repeats)?]);
    }

    if let Ok(sequence) = repeats.extract::<Vec<i64>>() {
        let mut values = Vec::with_capacity(sequence.len());
        for repeat in sequence {
            if repeat < 0 {
                return Err(PyValueError::new_err(
                    "repeat expects non-negative integers",
                ));
            }
            values.push(repeat as usize);
        }
        return Ok(values);
    }

    Ok(vec![extract_repeat_element(repeats)?])
}

fn extract_repeat_element(value: &Bound<PyAny>) -> PyResult<usize> {
    let repeat: i64 = value.extract()?;
    if repeat < 0 {
        Err(PyValueError::new_err(
            "repeat expects non-negative integers",
        ))
    } else {
        Ok(repeat as usize)
    }
}

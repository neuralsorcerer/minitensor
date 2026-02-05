// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

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

fn normalize_variadic_isize_args(tuple: &Bound<PyTuple>, arg_name: &str) -> PyResult<Vec<isize>> {
    if tuple.is_empty() {
        return Ok(Vec::new());
    }

    if tuple.len() == 1 {
        let first = tuple.get_item(0)?;

        if let Ok(nested) = first.cast::<PyTuple>() {
            return normalize_variadic_isize_args(nested, arg_name);
        }

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

fn parse_shape_tuple(shape: &Bound<PyTuple>, arg_name: &str) -> PyResult<Vec<usize>> {
    if shape.is_empty() {
        return Ok(Vec::new());
    }

    if shape.len() == 1 {
        let first = shape.get_item(0)?;
        if let Ok(tuple) = first.cast::<PyTuple>() {
            return parse_shape_tuple(tuple, arg_name);
        }
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

fn parse_shape_like(obj: &Bound<PyAny>, arg_name: &str) -> PyResult<Vec<usize>> {
    if let Ok(tuple) = obj.cast::<PyTuple>() {
        return parse_shape_tuple(tuple, arg_name);
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

fn normalize_roll_shifts(shifts: &Bound<PyAny>) -> PyResult<Vec<isize>> {
    normalize_required_axes(shifts, "shifts")
}

fn normalize_required_axes<'py>(dim: &'py Bound<'py, PyAny>, name: &str) -> PyResult<Vec<isize>> {
    match normalize_optional_axes(Some(dim))? {
        Some(values) => Ok(values),
        None => Err(PyTypeError::new_err(format!(
            "{} must be an int or a sequence of ints",
            name
        ))),
    }
}

fn normalize_optional_axes(dim: Option<&Bound<PyAny>>) -> PyResult<Option<Vec<isize>>> {
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

    static NUMPY_BOOL_TYPE: OnceCell<Py<PyAny>> = OnceCell::new();
    let py = obj.py();
    if let Ok(numpy_bool) = NUMPY_BOOL_TYPE.get_or_try_init(|| -> PyResult<Py<PyAny>> {
        let numpy = PyModule::import(py, "numpy")?;
        let bool_obj = numpy.getattr("bool_")?;
        Ok(bool_obj.unbind())
    }) && obj.is_instance(numpy_bool.bind(py))?
    {
        return Ok(true);
    }

    Ok(false)
}

fn normalize_repeat_spec(repeats: &Bound<PyAny>) -> PyResult<Vec<usize>> {
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

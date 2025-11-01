// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use engine::DataType;
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyFloat, PyInt, PyModule};

static DEFAULT_DTYPE: Lazy<RwLock<DataType>> = Lazy::new(|| RwLock::new(DataType::Float32));

fn dtype_from_str(name: &str) -> Option<DataType> {
    match name.to_ascii_lowercase().as_str() {
        "float32" | "f32" => Some(DataType::Float32),
        "float64" | "f64" => Some(DataType::Float64),
        "int32" | "i32" => Some(DataType::Int32),
        "int64" | "i64" => Some(DataType::Int64),
        "bool" | "boolean" => Some(DataType::Bool),
        _ => None,
    }
}

fn dtype_to_str(dtype: DataType) -> &'static str {
    match dtype {
        DataType::Float32 => "float32",
        DataType::Float64 => "float64",
        DataType::Int32 => "int32",
        DataType::Int64 => "int64",
        DataType::Bool => "bool",
    }
}

pub fn parse_dtype(name: &str) -> PyResult<DataType> {
    dtype_from_str(name).ok_or_else(|| {
        PyValueError::new_err(format!(
            "Unsupported dtype '{name}'. Expected one of float32, float64, int32, int64, bool"
        ))
    })
}

pub fn resolve_dtype_arg(arg: Option<&str>) -> PyResult<DataType> {
    match arg {
        Some(name) => parse_dtype(name),
        None => Ok(default_dtype()),
    }
}

pub fn default_dtype() -> DataType {
    *DEFAULT_DTYPE.read()
}

pub fn default_float_dtype() -> DataType {
    match default_dtype() {
        DataType::Float64 => DataType::Float64,
        _ => DataType::Float32,
    }
}

pub fn set_default_dtype(name: &str) -> PyResult<()> {
    let dtype = parse_dtype(name)?;
    *DEFAULT_DTYPE.write() = dtype;
    Ok(())
}

pub fn get_default_dtype() -> String {
    dtype_to_str(default_dtype()).to_string()
}

fn numpy_scalar_dtype(value: &Bound<'_, PyAny>) -> PyResult<Option<DataType>> {
    let py = value.py();
    let numpy = match PyModule::import(py, "numpy") {
        Ok(module) => module,
        Err(_) => return Ok(None),
    };

    let generic = match numpy.getattr("generic") {
        Ok(generic) => generic,
        Err(_) => return Ok(None),
    };

    if !value.is_instance(&generic)? {
        return Ok(None);
    }

    let dtype_obj = value.getattr("dtype")?;
    let dtype_str = dtype_obj.str()?.to_str()?.to_ascii_lowercase();
    Ok(dtype_from_str(&dtype_str))
}

pub fn resolve_scalar_dtype(value: &Bound<'_, PyAny>, context: DataType) -> PyResult<DataType> {
    if let Some(dtype) = numpy_scalar_dtype(value)? {
        return Ok(dtype);
    }

    if value.is_instance_of::<PyBool>() {
        return Ok(DataType::Bool);
    }

    if value.is_instance_of::<PyFloat>() {
        return Ok(if context == DataType::Float64 {
            DataType::Float64
        } else {
            default_float_dtype()
        });
    }

    if value.is_instance_of::<PyInt>() {
        return Ok(match context {
            DataType::Int32 | DataType::Int64 | DataType::Float32 | DataType::Float64 => context,
            _ => DataType::Int64,
        });
    }

    let index_name = intern!(value.py(), "__index__");
    if value.hasattr(index_name)? {
        let method = value.getattr(index_name)?;
        if method.is_callable() {
            let result = method.call0()?;
            if result.is_instance_of::<PyInt>() {
                // Ensure the returned value can be represented as a concrete integer.
                let _ = result.extract::<i64>()?;
                return Ok(match context {
                    DataType::Int32 | DataType::Int64 | DataType::Float32 | DataType::Float64 => {
                        context
                    }
                    _ => DataType::Int64,
                });
            }
        }
    }

    let float_name = intern!(value.py(), "__float__");
    if value.hasattr(float_name)? {
        let float_attr = value.getattr(float_name)?;
        if float_attr.is_callable() {
            return Ok(if context == DataType::Float64 {
                DataType::Float64
            } else {
                default_float_dtype()
            });
        }
    }

    Err(PyValueError::new_err(
        "Unsupported scalar type for tensor operation",
    ))
}

pub fn dtype_to_python_string(dtype: DataType) -> &'static str {
    dtype_to_str(dtype)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn resolve_scalar_dtype_respects_index_like_objects() {
        Python::attach(|py| -> PyResult<()> {
            let code = CString::new(
                "class IndexLike:\n    def __init__(self, value):\n        self.value = value\n    def __index__(self):\n        return self.value\n",
            )
            .unwrap();
            let filename = CString::new("<dtype_tests>").unwrap();
            let module_name = CString::new("dtype_helpers").unwrap();
            let module = PyModule::from_code(
                py,
                code.as_c_str(),
                filename.as_c_str(),
                module_name.as_c_str(),
            )?;
            let index_like = module.getattr("IndexLike")?.call1((7,))?;
            let dtype = resolve_scalar_dtype(&index_like, DataType::Int32)?;
            assert_eq!(dtype, DataType::Int32);
            Ok(())
        })
        .unwrap();
    }
}

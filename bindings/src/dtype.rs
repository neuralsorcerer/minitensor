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
use pyo3::types::{PyAny, PyBool, PyFloat, PyInt};

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

fn is_numpy_module(module_name: &str) -> bool {
    module_name == "numpy" || module_name.starts_with("numpy.")
}

fn integer_like_dtype_for_context(context: DataType) -> DataType {
    match context {
        DataType::Int32 | DataType::Int64 | DataType::Float32 | DataType::Float64 => context,
        _ => DataType::Int64,
    }
}

fn numpy_scalar_dtype(value: &Bound<'_, PyAny>) -> PyResult<Option<DataType>> {
    let is_numpy_type = match value.get_type().module() {
        Ok(module) => match module.to_str() {
            Ok(module_name) => is_numpy_module(module_name),
            Err(_) => false,
        },
        Err(_) => false,
    };

    if !is_numpy_type {
        return Ok(None);
    }

    let dtype_name = intern!(value.py(), "dtype");
    let dtype_obj = match value.getattr(dtype_name) {
        Ok(dtype) => dtype,
        Err(_) => return Ok(None),
    };

    let dtype_str = match dtype_obj.str() {
        Ok(dtype_text) => match dtype_text.to_str() {
            Ok(text) => text.to_ascii_lowercase(),
            Err(_) => return Ok(None),
        },
        Err(_) => return Ok(None),
    };
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
        return Ok(integer_like_dtype_for_context(context));
    }

    let index_name = intern!(value.py(), "__index__");
    if value.hasattr(index_name)? {
        let method = value.getattr(index_name)?;
        if method.is_callable() {
            let result = method.call0()?;
            if result.is_instance_of::<PyInt>() {
                // Ensure the returned value can be represented as a concrete integer.
                let _ = result.extract::<i64>()?;
                return Ok(integer_like_dtype_for_context(context));
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
    use pyo3::types::PyModule;
    use std::ffi::CString;

    fn module_from_code<'py>(
        py: Python<'py>,
        code: &str,
        module_name: &str,
    ) -> PyResult<Bound<'py, PyModule>> {
        let code = CString::new(code).expect("embedded test code should not contain NUL bytes");
        let filename =
            CString::new("<dtype_tests>").expect("static filename should not contain NUL bytes");
        let module_name =
            CString::new(module_name).expect("module name should not contain NUL bytes");
        PyModule::from_code(
            py,
            code.as_c_str(),
            filename.as_c_str(),
            module_name.as_c_str(),
        )
    }

    #[test]
    fn resolve_scalar_dtype_builtin_scalar_paths() {
        Python::attach(|py| -> PyResult<()> {
            let module = module_from_code(
                py,
                r#"BOOL_VALUE = True
FLOAT_VALUE = 1.25
INT_VALUE = 7
"#,
                "dtype_helpers_builtin_scalars",
            )?;

            let bool_obj = module.getattr("BOOL_VALUE")?;
            assert_eq!(
                resolve_scalar_dtype(&bool_obj, DataType::Float32)?,
                DataType::Bool
            );

            let float_obj = module.getattr("FLOAT_VALUE")?;
            assert_eq!(
                resolve_scalar_dtype(&float_obj, DataType::Float64)?,
                DataType::Float64
            );

            let int_obj = module.getattr("INT_VALUE")?;
            assert_eq!(
                resolve_scalar_dtype(&int_obj, DataType::Int32)?,
                DataType::Int32
            );
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn set_default_dtype_rejects_invalid_name() {
        let err = set_default_dtype("bad_dtype").unwrap_err();
        assert!(err.to_string().contains("Unsupported dtype 'bad_dtype'"));
    }

    #[test]
    fn dtype_to_python_string_covers_all_variants() {
        assert_eq!(dtype_to_python_string(DataType::Int64), "int64");
        assert_eq!(dtype_to_python_string(DataType::Bool), "bool");
    }

    #[test]
    fn resolve_scalar_dtype_index_and_float_method_edge_cases() {
        Python::attach(|py| -> PyResult<()> {
            let module = module_from_code(
                py,
                r#"class IndexRaises:
    def __index__(self):
        raise RuntimeError('index failure')

class IndexReturnsFloat:
    def __index__(self):
        return 1.5
    def __float__(self):
        return 1.5

class FloatNotCallable:
    __float__ = 1
"#,
                "dtype_helpers_method_edge_cases",
            )?;

            let index_raises = module.getattr("IndexRaises")?.call0()?;
            assert!(resolve_scalar_dtype(&index_raises, DataType::Float32).is_err());

            let index_float = module.getattr("IndexReturnsFloat")?.call0()?;
            assert_eq!(
                resolve_scalar_dtype(&index_float, DataType::Float32)?,
                DataType::Float32
            );

            let float_not_callable = module.getattr("FloatNotCallable")?.call0()?;
            assert!(resolve_scalar_dtype(&float_not_callable, DataType::Float32).is_err());
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn numpy_scalar_dtype_fast_path_and_failures() {
        Python::attach(|py| -> PyResult<()> {
            let helper = module_from_code(
                py,
                r#"def make_numpy_scalar(dtype_text='float64', raise_str=False):
    class DType:
        def __str__(self):
            if raise_str:
                raise RuntimeError('dtype str failed')
            return dtype_text

    class Scalar:
        __module__ = 'numpy'
        @property
        def dtype(self):
            return DType()

    return Scalar()


def make_bad_module_name_scalar():
    class Scalar:
        __module__ = '\udcff'
        @property
        def dtype(self):
            class DType:
                def __str__(self):
                    return 'float32'
            return DType()
    return Scalar()
"#,
                "dtype_helpers_numpy_paths",
            )?;

            let scalar = helper.getattr("make_numpy_scalar")?.call0()?;
            assert_eq!(numpy_scalar_dtype(&scalar)?, Some(DataType::Float64));

            let unknown_dtype = helper
                .getattr("make_numpy_scalar")?
                .call1(("complex128", false))?;
            assert_eq!(numpy_scalar_dtype(&unknown_dtype)?, None);

            let bad_dtype_str = helper
                .getattr("make_numpy_scalar")?
                .call1(("float64", true))?;
            assert_eq!(numpy_scalar_dtype(&bad_dtype_str)?, None);

            let bad_module = helper.getattr("make_bad_module_name_scalar")?.call0()?;
            assert_eq!(numpy_scalar_dtype(&bad_module)?, None);

            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn parse_and_resolve_dtype_arg_cover_success_and_error_paths() {
        assert_eq!(parse_dtype("f32").unwrap(), DataType::Float32);
        assert_eq!(parse_dtype("FLOAT64").unwrap(), DataType::Float64);
        assert!(parse_dtype("bad_dtype").is_err());

        set_default_dtype("float32").unwrap();
        assert_eq!(resolve_dtype_arg(None).unwrap(), DataType::Float32);
        assert_eq!(resolve_dtype_arg(Some("int64")).unwrap(), DataType::Int64);
    }

    #[test]
    fn default_dtype_round_trip_and_python_string() {
        set_default_dtype("float32").unwrap();
        assert_eq!(default_dtype(), DataType::Float32);
        assert_eq!(default_float_dtype(), DataType::Float32);
        assert_eq!(get_default_dtype(), "float32");

        set_default_dtype("bool").unwrap();
        assert_eq!(default_dtype(), DataType::Bool);
        assert_eq!(default_float_dtype(), DataType::Float32);
        assert_eq!(dtype_to_python_string(DataType::Int32), "int32");

        set_default_dtype("float64").unwrap();
        assert_eq!(default_dtype(), DataType::Float64);
        assert_eq!(default_float_dtype(), DataType::Float64);
        assert_eq!(get_default_dtype(), "float64");

        set_default_dtype("float32").unwrap();
    }

    #[test]
    fn resolve_scalar_dtype_integer_like_context_fallback() {
        Python::attach(|py| -> PyResult<()> {
            let code = r#"class FlagEnumLike:
    def __index__(self):
        return 7
"#;
            let module = module_from_code(py, code, "dtype_helpers_int_context")?;
            let index_like = module.getattr("FlagEnumLike")?.call0()?;

            assert_eq!(
                resolve_scalar_dtype(&index_like, DataType::Bool)?,
                DataType::Int64
            );
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn resolve_scalar_dtype_unsupported_scalar_errors() {
        Python::attach(|py| -> PyResult<()> {
            let code = r#"class NotScalar:
    pass
"#;
            let module = module_from_code(py, code, "dtype_helpers_not_scalar")?;
            let value = module.getattr("NotScalar")?.call0()?;
            let err = resolve_scalar_dtype(&value, DataType::Float32).unwrap_err();
            assert!(err.to_string().contains("Unsupported scalar type"));
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn resolve_scalar_dtype_dtype_str_errors_fall_back() {
        Python::attach(|py| -> PyResult<()> {
            let module = module_from_code(
                py,
                r#"class BadDTypeText:
    __module__ = 'numpy'
    class _DType:
        def __str__(self):
            raise RuntimeError('dtype str failed')
    @property
    def dtype(self):
        return self._DType()
    def __float__(self):
        return 1.5
"#,
                "dtype_helpers_bad_dtype_text",
            )?;

            let value = module.getattr("BadDTypeText")?.call0()?;
            let resolved = resolve_scalar_dtype(&value, DataType::Float32)?;
            assert_eq!(resolved, DataType::Float32);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn is_numpy_module_checks_exact_prefix() {
        assert!(is_numpy_module("numpy"));
        assert!(is_numpy_module("numpy.random"));
        assert!(!is_numpy_module("numpyx"));
        assert!(!is_numpy_module("other.numpy"));
    }

    #[test]
    fn resolve_scalar_dtype_respects_index_like_objects() {
        Python::attach(|py| -> PyResult<()> {
            let code = r#"class IndexLike:
    def __init__(self, value):
        self.value = value
    def __index__(self):
        return self.value
"#;
            let module = module_from_code(py, code, "dtype_helpers")?;
            let index_like = module.getattr("IndexLike")?.call1((7,))?;
            let dtype = resolve_scalar_dtype(&index_like, DataType::Int32)?;
            assert_eq!(dtype, DataType::Int32);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn resolve_scalar_dtype_non_numpy_module_name_does_not_trigger_numpy_path() {
        Python::attach(|py| -> PyResult<()> {
            let code = r#"class FakeNumpyScalar:
    __module__ = 'numpyx'
    def __float__(self):
        return 1.25
"#;
            let module = module_from_code(py, code, "dtype_helpers_module_name")?;

            let value = module.getattr("FakeNumpyScalar")?.call0()?;
            let resolved = resolve_scalar_dtype(&value, DataType::Float32)?;
            assert_eq!(resolved, DataType::Float32);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn resolve_scalar_dtype_numpy_module_with_dtype_uses_numpy_fast_path() {
        Python::attach(|py| -> PyResult<()> {
            let code = r#"class PretendNumpyType:
    __module__ = 'numpy'
    @property
    def dtype(self):
        class DType:
            def __str__(self):
                return 'float64'
        return DType()
"#;
            let module = module_from_code(py, code, "dtype_helpers_numpy_fallback")?;

            let value = module.getattr("PretendNumpyType")?.call0()?;
            let resolved = resolve_scalar_dtype(&value, DataType::Float64)?;
            assert_eq!(resolved, DataType::Float64);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn resolve_scalar_dtype_numpy_module_without_dtype_falls_back() {
        Python::attach(|py| -> PyResult<()> {
            let helpers = module_from_code(
                py,
                r#"class PretendNumpyNoDtype:
    __module__ = 'numpy'
    def __float__(self):
        return 1.0
"#,
                "dtype_helpers_numpy_no_dtype",
            )?;
            let value = helpers.getattr("PretendNumpyNoDtype")?.call0()?;
            let resolved = resolve_scalar_dtype(&value, DataType::Float32)?;
            assert_eq!(resolved, DataType::Float32);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn resolve_scalar_dtype_dtype_lookup_errors_fall_back() {
        Python::attach(|py| -> PyResult<()> {
            let module = module_from_code(
                py,
                r#"class PretendNumpyBrokenDtype:
    __module__ = 'numpy'
    def __getattribute__(self, name):
        if name == 'dtype':
            raise RuntimeError('dtype access failed')
        return object.__getattribute__(self, name)
    def __float__(self):
        return 4.0
"#,
                "dtype_helpers_broken_dtype",
            )?;

            let value = module.getattr("PretendNumpyBrokenDtype")?.call0()?;
            let resolved = resolve_scalar_dtype(&value, DataType::Float32)?;
            assert_eq!(resolved, DataType::Float32);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn resolve_scalar_dtype_handles_module_lookup_errors() {
        Python::attach(|py| -> PyResult<()> {
            let code = r#"class Meta(type):
    def __getattribute__(cls, name):
        if name == '__module__':
            raise RuntimeError('module lookup failed')
        return super().__getattribute__(name)

class FloatLike(metaclass=Meta):
    def __float__(self):
        return 2.0
"#;
            let module = module_from_code(py, code, "dtype_helpers_module_error")?;

            let value = module.getattr("FloatLike")?.call0()?;
            let resolved = resolve_scalar_dtype(&value, DataType::Float64)?;
            assert_eq!(resolved, DataType::Float64);
            Ok(())
        })
        .unwrap();
    }
}

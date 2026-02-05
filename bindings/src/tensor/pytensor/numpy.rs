// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

#[pymethods]
impl PyTensor {
    // NumPy conversion methods
    fn numpy(&self, py: Python) -> PyResult<Py<PyAny>> {
        convert_tensor_to_numpy(&self.inner, py, false)
    }

    fn numpy_copy(&self, py: Python) -> PyResult<Py<PyAny>> {
        convert_tensor_to_numpy(&self.inner, py, true)
    }

    #[pyo3(signature = (dtype=None))]
    fn __array__(&self, py: Python, dtype: Option<&Bound<PyAny>>) -> PyResult<Py<PyAny>> {
        let array = self.numpy(py)?;
        if let Some(dtype_obj) = dtype {
            let array_bound = array.bind(py);
            let kwargs = PyDict::new(py);
            kwargs.set_item(intern!(py, "copy"), false)?;
            let casted =
                array_bound.call_method(intern!(py, "astype"), (dtype_obj,), Some(&kwargs))?;
            Ok(casted.into())
        } else {
            Ok(array)
        }
    }

    #[pyo3(signature = (ufunc, method, *inputs, **kwargs))]
    fn __array_ufunc__(
        &self,
        py: Python,
        ufunc: &Bound<PyAny>,
        method: &str,
        inputs: &Bound<PyTuple>,
        kwargs: Option<&Bound<PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        if method != "__call__" {
            return py_not_implemented(py);
        }

        if let Some(mapping) = kwargs
            && let Some(out) = mapping.get_item("out")?
            && !out.is_none()
        {
            return py_not_implemented(py);
        }

        let mut operands: Vec<Tensor> = Vec::with_capacity(inputs.len());
        for value in inputs.iter() {
            match tensor_from_py_value(&self.inner, &value) {
                Ok(tensor) => operands.push(tensor),
                Err(_) => return py_not_implemented(py),
            }
        }

        let Some(name_obj) = ufunc.getattr(intern!(py, "__name__")).ok() else {
            return py_not_implemented(py);
        };
        let name = name_obj.str()?.to_str()?.to_ascii_lowercase();

        let result = match (name.as_str(), operands.len()) {
            ("add", 2) => {
                apply_binary_ufunc(&operands, BinaryOpKind::Add, |lhs, rhs| lhs.add(rhs))?
            }
            ("subtract", 2) => apply_binary_ufunc(&operands, BinaryOpKind::Sub, |lhs, rhs| {
                engine::operations::arithmetic::sub(lhs, rhs)
            })?,
            ("multiply", 2) => apply_binary_ufunc(&operands, BinaryOpKind::Mul, |lhs, rhs| {
                engine::operations::arithmetic::mul(lhs, rhs)
            })?,
            ("true_divide", 2) | ("divide", 2) => {
                apply_binary_ufunc(&operands, BinaryOpKind::Div, |lhs, rhs| {
                    engine::operations::arithmetic::div(lhs, rhs)
                })?
            }
            ("power", 2) => {
                apply_binary_ufunc(&operands, BinaryOpKind::Mul, |lhs, rhs| lhs.pow(rhs))?
            }
            ("maximum", 2) => apply_binary_ufunc(&operands, BinaryOpKind::Maximum, |lhs, rhs| {
                lhs.maximum(rhs)
            })?,
            ("minimum", 2) => apply_binary_ufunc(&operands, BinaryOpKind::Minimum, |lhs, rhs| {
                lhs.minimum(rhs)
            })?,
            ("negative", 1) => apply_unary_ufunc(&operands, |tensor| {
                engine::operations::arithmetic::neg(tensor)
            })?,
            ("absolute", 1) | ("abs", 1) => apply_unary_ufunc(&operands, |tensor| tensor.abs())?,
            ("exp", 1) => apply_unary_ufunc(&operands, |tensor| tensor.exp())?,
            ("log", 1) => apply_unary_ufunc(&operands, |tensor| tensor.log())?,
            ("sin", 1) => apply_unary_ufunc(&operands, |tensor| tensor.sin())?,
            ("cos", 1) => apply_unary_ufunc(&operands, |tensor| tensor.cos())?,
            ("tan", 1) => apply_unary_ufunc(&operands, |tensor| tensor.tan())?,
            ("sqrt", 1) => apply_unary_ufunc(&operands, |tensor| tensor.sqrt())?,
            _ => return py_not_implemented(py),
        };

        let py_tensor = Py::new(py, PyTensor::from_tensor(result))?;
        Ok(py_tensor.into_any())
    }

    fn tolist(&self) -> PyResult<Py<PyAny>> {
        if self.inner.ndim() == 0 {
            Python::attach(|py| convert_tensor_to_python_scalar(&self.inner, py))
        } else {
            Python::attach(|py| convert_tensor_to_python_list(&self.inner, py))
        }
    }

    fn item(&self) -> PyResult<Py<PyAny>> {
        Python::attach(|py| convert_tensor_to_python_scalar(&self.inner, py))
    }

}

// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;

/// The NumPy type string for a dtype, in this machine's byte order.
///
/// Single-byte types take `|`, which says the question does not arise; every
/// other type has to name the order it is actually stored in, and a tensor is
/// stored in the host's.
fn typestr(dtype: DataType) -> &'static str {
    let big = cfg!(target_endian = "big");
    match dtype {
        DataType::Float32 => {
            if big {
                ">f4"
            } else {
                "<f4"
            }
        }
        DataType::Float64 => {
            if big {
                ">f8"
            } else {
                "<f8"
            }
        }
        DataType::Int32 => {
            if big {
                ">i4"
            } else {
                "<i4"
            }
        }
        DataType::Int64 => {
            if big {
                ">i8"
            } else {
                "<i8"
            }
        }
        DataType::Bool => "|b1",
    }
}

#[pymethods]
impl PyTensor {
    /// The buffer itself, for a NumPy that would rather not be handed a copy.
    ///
    /// `numpy.asarray(tensor)` goes through this and comes back pointing at
    /// the tensor's own memory: no copy, no allocation, whatever the size. The
    /// engine keeps every tensor contiguous and row-major, which is exactly
    /// what an array header describes, so there is nothing to rearrange on the
    /// way out. NumPy holds a reference to the tensor for as long as the array
    /// lives, so the buffer cannot be freed underneath it.
    ///
    /// **Read-only**, and not as a nicety. Several tensors can share one
    /// buffer -- `detach`, `reshape`, `astype` to the dtype it already has --
    /// and each of them is supposed to have its own values; a NumPy array
    /// writing into that buffer would change all of them at once. Ask for
    /// `numpy.array(tensor)` or `tensor.numpy()` when you want something to
    /// write into: both copy.
    ///
    /// An in-place operation on the tensor is the other direction of the same
    /// sharing. Writing to a buffer this tensor holds alone writes through the
    /// array; writing to one it shares copies first, and the array is left on
    /// the old buffer, still valid and still holding the values it had. So an
    /// exported array is a stable read of the values as they were, and a
    /// program that both exports and writes in place should not depend on
    /// which of the two it gets.
    ///
    /// A tensor that is not on the CPU has no host buffer to point at, so it
    /// does not have this attribute at all -- which is the protocol's way of
    /// saying so, and leaves `numpy.asarray` to fall through to `__array__`
    /// and its message about calling `.cpu()` first.
    #[getter]
    fn __array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        if self.inner.device() != Device::cpu() {
            return Err(PyErr::new::<pyo3::exceptions::PyAttributeError, _>(
                "__array_interface__",
            ));
        }

        let data = self.inner.data();
        let address = match self.inner.dtype() {
            DataType::Float32 => data.as_f32_slice().map(|s| s.as_ptr() as usize),
            DataType::Float64 => data.as_f64_slice().map(|s| s.as_ptr() as usize),
            DataType::Int32 => data.as_i32_slice().map(|s| s.as_ptr() as usize),
            DataType::Int64 => data.as_i64_slice().map(|s| s.as_ptr() as usize),
            DataType::Bool => data.as_bool_slice().map(|s| s.as_ptr() as usize),
        }
        .ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "tensor buffer does not match its dtype",
            )
        })?;

        let interface = PyDict::new(py);
        interface.set_item(
            intern!(py, "shape"),
            PyTuple::new(py, self.inner.shape().dims())?,
        )?;
        interface.set_item(intern!(py, "typestr"), typestr(self.inner.dtype()))?;
        // `true` is the read-only flag, and `strides` is left out entirely:
        // absent means C-contiguous, which every tensor is.
        interface.set_item(intern!(py, "data"), (address, true))?;
        interface.set_item(intern!(py, "version"), 3)?;
        Ok(interface)
    }

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
                engine::ops::arithmetic::sub(lhs, rhs)
            })?,
            ("multiply", 2) => apply_binary_ufunc(&operands, BinaryOpKind::Mul, |lhs, rhs| {
                engine::ops::arithmetic::mul(lhs, rhs)
            })?,
            ("true_divide", 2) | ("divide", 2) => {
                apply_binary_ufunc(&operands, BinaryOpKind::Div, |lhs, rhs| {
                    engine::ops::arithmetic::div(lhs, rhs)
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
            ("negative", 1) => apply_unary_ufunc(&operands, engine::ops::arithmetic::neg)?,
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

    pub(crate) fn tolist(&self) -> PyResult<Py<PyAny>> {
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

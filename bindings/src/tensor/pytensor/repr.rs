// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
#[pymethods]
impl PyTensor {
    // String representations
    fn __repr__(&self) -> String {
        format!(
            "Tensor(shape={:?}, dtype={}, device={}, requires_grad={})",
            self.inner.shape().dims(),
            self.dtype(),
            self.device(),
            // Spelled the way Python spells a boolean, since this is read in
            // Python: it used to print `false`.
            if self.inner.requires_grad() {
                "True"
            } else {
                "False"
            }
        )
    }

    fn __str__(&self) -> String {
        if self.inner.numel() <= 100 {
            match self.tolist() {
                Ok(data) => Python::attach(|py| format!("tensor({})", data.bind(py))),
                Err(_) => self.__repr__(),
            }
        } else {
            self.__repr__()
        }
    }

    fn __len__(&self) -> PyResult<usize> {
        if self.inner.ndim() == 0 {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "len() of unsized object",
            ))
        } else {
            Ok(self.inner.shape().dims()[0])
        }
    }

    fn __bool__(&self) -> PyResult<bool> {
        if self.inner.numel() != 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                if self.inner.numel() == 0 {
                    "The truth value of an empty tensor is ambiguous"
                } else {
                    "The truth value of a tensor with more than one element is ambiguous"
                },
            ));
        }

        match self.inner.dtype() {
            DataType::Float32 => {
                let data = self.inner.data().as_f32_slice().ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to get f32 data")
                })?;
                Ok(data[0] != 0.0)
            }
            DataType::Float64 => {
                let data = self.inner.data().as_f64_slice().ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to get f64 data")
                })?;
                Ok(data[0] != 0.0)
            }
            DataType::Int32 => {
                let data = self.inner.data().as_i32_slice().ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to get i32 data")
                })?;
                Ok(data[0] != 0)
            }
            DataType::Int64 => {
                let data = self.inner.data().as_i64_slice().ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to get i64 data")
                })?;
                Ok(data[0] != 0)
            }
            DataType::Bool => {
                let data = self.inner.data().as_bool_slice().ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to get bool data")
                })?;
                Ok(data[0])
            }
        }
    }

    fn __abs__(&self) -> PyResult<Self> {
        let result = self.inner.abs().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    fn __pos__(&self) -> Self {
        // `+t` is the identity; the clone shares storage via Arc and returns
        // the input values unchanged.
        self.clone()
    }

    fn __float__(&self) -> PyResult<f64> {
        let value = self.scalar_as_f64()?;
        Ok(value)
    }

    /// The value as a Python int: exact for an integer tensor, truncated
    /// toward zero for a float one, as `int()` truncates a float.
    ///
    /// Every dtype used to pass through an f64, so an int64 above 2^53 came
    /// back rounded -- `int(t)` for 2^60 + 1 gave 2^60 -- and a float
    /// outside the i64 range saturated instead of growing.
    fn __int__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.rounded_to_int(py, "__int__")
    }

    /// `math.floor(t)`, as a Python int. Without it `math.floor` went through
    /// `float()`, which rounds an int64 past 2^53: 2^62 + 1 came back as 2^62.
    fn __floor__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.rounded_to_int(py, "__floor__")
    }

    /// `math.ceil(t)`, as a Python int; exact for an integer tensor.
    fn __ceil__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.rounded_to_int(py, "__ceil__")
    }

    /// `math.trunc(t)`, as a Python int; exact for an integer tensor. It was
    /// missing outright, where `floor` and `ceil` at least had `float()`.
    fn __trunc__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.rounded_to_int(py, "__trunc__")
    }

    /// The value as an index: an integer or bool tensor of one element, so
    /// it can subscript a list or size a `range`. A float tensor is refused,
    /// as a float is.
    fn __index__(&self) -> PyResult<i64> {
        self.scalar_as_i64()?.ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "only an integer tensor can be used as an index, not {}",
                self.inner.dtype()
            ))
        })
    }

    /// Iterate over the first axis.
    ///
    /// Without this, iteration fell back to subscripting 0, 1, 2, ... until
    /// an IndexError -- which a 0-d tensor raises at once, so
    /// `for x in scalar` ran zero times rather than refusing, while `len`
    /// refused.
    fn __iter__(slf: &Bound<'_, Self>) -> PyResult<TensorRows> {
        Self::rows(slf, false)
    }

    /// Iterate over the first axis from the last row back. `len` and
    /// subscripting were both there, but a class defining its subscript the
    /// way this one does is not a sequence to `reversed`, which refused it.
    fn __reversed__(slf: &Bound<'_, Self>) -> PyResult<TensorRows> {
        Self::rows(slf, true)
    }

    /// `round(t)` and `round(t, n)`: the elementwise rounding `round`
    /// already gives, to `n` decimals.
    #[pyo3(signature = (ndigits=None))]
    fn __round__(&self, ndigits: Option<i32>) -> PyResult<Self> {
        self.round(ndigits.unwrap_or(0))
    }

    /// `divmod(t, x)`: the pair `(t // x, t % x)`.
    fn __divmod__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(slf.py(), [slf.floor_div(other)?, slf.rem(other)?])
    }

    /// `divmod(x, t)`: the pair `(x // t, x % t)`.
    fn __rdivmod__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(slf.py(), [other.floor_div(slf)?, other.rem(slf)?])
    }

    /// Format a one-element tensor as its value, so `f"{loss:.4f}"` works;
    /// any tensor formats with an empty spec, as `str` does.
    fn __format__(slf: &Bound<'_, Self>, spec: &str) -> PyResult<String> {
        if spec.is_empty() {
            return Ok(slf.str()?.to_string());
        }
        if slf.borrow().inner.numel() != 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "a format spec ({spec:?}) applies to a one-element tensor, and this one has {} elements",
                slf.borrow().inner.numel()
            )));
        }
        slf.call_method0("item")?
            .call_method1("__format__", (spec,))?
            .extract()
    }

    fn __getitem__(&self, key: &Bound<PyAny>) -> PyResult<Self> {
        // Fancy forms first: boolean masks select blocks along
        // the leading dims, 1-D integer keys select rows along dim 0.
        if let Some(result) = try_fancy_index_tensor(&self.inner, key)? {
            return Ok(Self::from_tensor(result));
        }
        // An index array somewhere other than the front, mixed with slices,
        // integers and `...` -- `x[:, idx]` and its relatives.
        if let Some(result) = try_single_array_index(&self.inner, key)? {
            return Ok(Self::from_tensor(result));
        }
        let (indices, newaxis_positions) = parse_getitem_indices(key, self.inner.shape().dims())?;
        let mut result = self.inner.index(&indices).map_err(_convert_error)?;
        for &pos in &newaxis_positions {
            result = result.unsqueeze(pos as isize).map_err(_convert_error)?;
        }
        Ok(Self::from_tensor(result))
    }

    /// Assign `value` to the positions `key` selects.
    ///
    /// Into a tensor that is part of a recorded computation, or with a value
    /// that is, the assignment is recorded: this tensor becomes a new one
    /// holding the assigned values, and the gradient reaching it goes to the
    /// value where the value was written and to the old tensor everywhere else.
    /// See [`engine::autograd::records_assignment`] for what is written in
    /// place instead.
    fn __setitem__(
        slf: &Bound<'_, Self>,
        key: &Bound<PyAny>,
        value: &Bound<PyAny>,
    ) -> PyResult<()> {
        // Resolve everything that needs to read `self` or `value` BEFORE
        // taking the mutable borrow: with a `&mut self` receiver, a
        // self-referential assignment like `t[mask] = t` would hit an
        // "already mutably borrowed" error when extracting the value.
        let (dtype, device, in_dims) = {
            let this = slf.borrow();
            (
                this.inner.dtype(),
                this.inner.device(),
                this.inner.shape().dims().to_vec(),
            )
        };
        let assignment = Assignment::resolve(key, &in_dims)?;
        let value = if let Ok(t) = value.extract::<PyTensor>() {
            t.inner
        } else {
            convert_python_data_to_tensor(value, dtype, device, false)?
        };

        let mut this = slf.borrow_mut();
        // The write itself never follows the value's graph: either the
        // assignment is recorded as a whole below, or it is not recorded.
        let detached;
        let written = if value.requires_grad() {
            detached = value.detach();
            &detached
        } else {
            &value
        };
        if !engine::autograd::records_assignment(&this.inner, value.requires_grad()) {
            return assignment.apply(&mut this.inner, written);
        }
        let (mut probe, elements) = engine::autograd::assignment_probe(&this.inner, &value);
        assignment.apply(&mut probe, &elements)?;
        let mut result = this.inner.detach();
        assignment.apply(&mut result, written)?;
        this.inner = engine::autograd::record_assignment(&this.inner, &value, &probe, result)
            .map_err(_convert_error)?;
        Ok(())
    }
}

/// Where `t[key] = value` writes, resolved from the key and the target's shape
/// alone, so that it can be applied to more than one tensor.
enum Assignment {
    /// A boolean mask over the leading dimensions; `value` may be a scalar or
    /// anything that broadcasts to the selection shape.
    Mask(Tensor),
    /// An index array or a mask somewhere in the subscript (`t[:, idx] = v`):
    /// each position it names is an ordinary basic assignment, so the value is
    /// lined up with the whole selection first and each position takes its
    /// share.
    Axis(AxisAssign),
    /// Integers, slices and `...`.
    Basic(Vec<TensorIndex>),
}

impl Assignment {
    fn resolve(key: &Bound<PyAny>, dims: &[usize]) -> PyResult<Self> {
        if let Some(mask) = try_bool_mask_key(key)? {
            let m_dims = mask.shape().dims();
            if m_dims.len() > dims.len() || dims[..m_dims.len()] != *m_dims {
                return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                    "boolean index mask shape {:?} must match the leading dimensions of tensor shape {:?}",
                    m_dims, dims
                )));
            }
            return Ok(Self::Mask(mask));
        }
        if let Some(plan) = plan_single_array_assign(key, dims)? {
            return Ok(Self::Axis(plan));
        }
        Ok(Self::Basic(parse_indices(key, dims)?))
    }

    /// Write `value` into `target`, refusing a value of another dtype.
    ///
    /// A value with a dtype of its own keeps it, and a disagreement with the
    /// destination is refused rather than resolved; a Python value carries no
    /// dtype and was already converted to the destination's. An integer or
    /// slice key refused, but a mask cast the value instead, so one assignment
    /// followed two rules depending on how its positions were named.
    fn apply(&self, target: &mut Tensor, value: &Tensor) -> PyResult<()> {
        if value.dtype() != target.dtype() {
            return Err(_convert_error(engine::MinitensorError::type_mismatch(
                format!("{:?}", target.dtype()),
                format!("{:?}", value.dtype()),
            )));
        }
        match self {
            Self::Mask(mask) => engine::ops::selection::masked_index_assign(target, mask, value)
                .map_err(_convert_error),
            Self::Axis(plan) => apply_single_array_assign(target, plan, value),
            Self::Basic(indices) => target.index_assign(indices, value).map_err(_convert_error),
        }
    }
}

/// A lazy iterator over a tensor's first axis, each row read with the same
/// subscript `t[i]` a caller would write.
#[pyclass(name = "TensorRows", module = "minitensor._core")]
pub struct TensorRows {
    tensor: Py<PyAny>,
    next: usize,
    len: usize,
    backwards: bool,
}

#[pymethods]
impl TensorRows {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        if self.next >= self.len {
            return Ok(None);
        }
        let position = if self.backwards {
            self.len - 1 - self.next
        } else {
            self.next
        };
        let row = self.tensor.bind(py).get_item(position)?;
        self.next += 1;
        Ok(Some(row.unbind()))
    }
}

impl PyTensor {
    fn rows(slf: &Bound<'_, Self>, backwards: bool) -> PyResult<TensorRows> {
        let this = slf.borrow();
        if this.inner.ndim() == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "iteration over a 0-d tensor",
            ));
        }
        Ok(TensorRows {
            tensor: slf.clone().into_any().unbind(),
            next: 0,
            len: this.inner.shape().dims()[0],
            backwards,
        })
    }

    /// The one element as a Python int: itself for an integer or bool
    /// tensor, so no f64 rounds it, and a float one rounded by the float
    /// method of the same name.
    fn rounded_to_int(&self, py: Python<'_>, method: &str) -> PyResult<Py<PyAny>> {
        if let Some(value) = self.scalar_as_i64()? {
            return Ok(value.into_pyobject(py)?.into_any().unbind());
        }
        let value = self.scalar_as_f64()?;
        Ok(pyo3::types::PyFloat::new(py, value)
            .call_method0(method)?
            .unbind())
    }

    /// The value of a one-element integer or bool tensor, exactly; `None`
    /// for a float one. Mirrors `__bool__`'s single-element requirement.
    fn scalar_as_i64(&self) -> PyResult<Option<i64>> {
        if self.inner.numel() != 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "only one element tensors can be converted to Python scalars",
            ));
        }
        let err =
            || PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to access tensor data");
        Ok(match self.inner.dtype() {
            DataType::Int32 => Some(self.inner.data().as_i32_slice().ok_or_else(err)?[0] as i64),
            DataType::Int64 => Some(self.inner.data().as_i64_slice().ok_or_else(err)?[0]),
            DataType::Bool => Some(self.inner.data().as_bool_slice().ok_or_else(err)?[0] as i64),
            DataType::Float32 | DataType::Float64 => None,
        })
    }

    /// Extract the value of a one-element tensor as f64 for `__float__` /
    /// `__int__`. Mirrors `__bool__`'s single-element requirement.
    fn scalar_as_f64(&self) -> PyResult<f64> {
        if self.inner.numel() != 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "only one element tensors can be converted to Python scalars",
            ));
        }
        let err =
            || PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Failed to access tensor data");
        match self.inner.dtype() {
            DataType::Float32 => Ok(self.inner.data().as_f32_slice().ok_or_else(err)?[0] as f64),
            DataType::Float64 => Ok(self.inner.data().as_f64_slice().ok_or_else(err)?[0]),
            DataType::Int32 => Ok(self.inner.data().as_i32_slice().ok_or_else(err)?[0] as f64),
            DataType::Int64 => Ok(self.inner.data().as_i64_slice().ok_or_else(err)?[0] as f64),
            DataType::Bool => Ok(if self.inner.data().as_bool_slice().ok_or_else(err)?[0] {
                1.0
            } else {
                0.0
            }),
        }
    }
}

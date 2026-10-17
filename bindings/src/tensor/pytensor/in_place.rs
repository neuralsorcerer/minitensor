// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;

/// `self op= other` as a write into `self`, rather than a new tensor bound to
/// the name.
///
/// Without the in-place dunders Python fell back to `x = x op y`: the name
/// moved to a new tensor and nothing else saw the change. A parameter handle
/// shares its layer's storage, so `for p in model.parameters(): p -= lr *
/// p.grad` read as an update and left the model exactly as it was; `y = x;
/// y += 1` left `x` alone; and the "in-place" result could change dtype or grow
/// by broadcasting.
///
/// The operator runs as it does out of place, and its result is written into
/// `self` following the rules every write here follows: a parameter leaf is
/// updated in place, a tensor computed from others records the write for
/// autograd, and a tensor a pending backward still needs is refused. The write
/// keeps `self`'s dtype and shape, and a result that would change either is
/// refused by name -- an int64 tensor cannot hold `x / 2`, and `x += row`
/// cannot turn a row into a matrix -- where the out-of-place form gives the new
/// tensor it describes.
///
/// The dunders return nothing, and the interpreter binds the name back to
/// `self`, the tensor just written.
fn in_place<'py>(
    slf: Bound<'py, PyTensor>,
    other: &Bound<'py, PyAny>,
    symbol: &str,
    operator: &str,
) -> PyResult<Bound<'py, PyTensor>> {
    let (dtype, shape, leaf) = {
        let this = slf.borrow();
        let inner = &this.inner;
        (
            inner.dtype(),
            inner.shape().dims().to_vec(),
            inner.requires_grad() && inner.grad_fn().is_none(),
        )
    };
    let result = {
        // A leaf that requires a gradient is written and not recorded, which
        // is how a parameter is set -- so its new value is computed untracked
        // too. Tracked, `p -= lr * p.grad` built a graph holding `p` and then
        // refused to write into it as a tensor "a pending backward still
        // needs". A backward really pending from before still holds `p`, and
        // is still protected by the write.
        let _untracked = leaf.then(engine::autograd::NoGradGuard::new);
        slf.call_method1(operator, (other,))?
    };
    let result = result.cast_into::<PyTensor>().map_err(PyErr::from)?;
    let (result_dtype, result_shape) = {
        let value = result.borrow();
        (value.inner.dtype(), value.inner.shape().dims().to_vec())
    };
    if result_dtype != dtype {
        return Err(PyTypeError::new_err(format!(
            "`{symbol}=` writes into the {} tensor, and the result is {}, which it \
             cannot hold; write `x = x {symbol} y` for a new tensor, or cast the \
             operands first",
            dtype::dtype_to_python_string(dtype),
            dtype::dtype_to_python_string(result_dtype),
        )));
    }
    if result_shape != shape {
        return Err(PyValueError::new_err(format!(
            "`{symbol}=` writes into a tensor of shape {shape:?}, and the result has \
             shape {result_shape:?}; write `x = x {symbol} y` for a new tensor"
        )));
    }
    // The result is a fresh tensor nothing else refers to when this holds the
    // only reference to it, and then its storage can be taken rather than
    // copied: what is left behind is never seen again.
    // SAFETY: `result` is a live object this function holds a reference to,
    // so reading its reference count reads valid memory.
    let sole = unsafe { pyo3::ffi::Py_REFCNT(result.as_ptr()) } == 1;
    let value = if sole {
        let mut taken = result.borrow_mut();
        let placeholder = Tensor::empty(Shape::new(vec![0]), dtype, taken.inner.device(), false);
        std::mem::replace(&mut taken.inner, placeholder)
    } else {
        result.borrow().inner.clone()
    };
    {
        let mut this = slf.borrow_mut();
        this.inner.adopt_or_copy_(value).map_err(_convert_error)?;
        register_leaf_tensor(&this.inner);
    }
    Ok(slf)
}

#[pymethods]
impl PyTensor {
    fn __iadd__<'py>(slf: Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<()> {
        in_place(slf, other, "+", "__add__").map(drop)
    }

    fn __isub__<'py>(slf: Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<()> {
        in_place(slf, other, "-", "__sub__").map(drop)
    }

    fn __imul__<'py>(slf: Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<()> {
        in_place(slf, other, "*", "__mul__").map(drop)
    }

    fn __itruediv__<'py>(slf: Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<()> {
        in_place(slf, other, "/", "__truediv__").map(drop)
    }

    fn __ifloordiv__<'py>(slf: Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<()> {
        in_place(slf, other, "//", "__floordiv__").map(drop)
    }

    fn __imod__<'py>(slf: Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<()> {
        in_place(slf, other, "%", "__mod__").map(drop)
    }

    fn __ipow__<'py>(
        slf: Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
        _modulo: &Bound<'py, PyAny>,
    ) -> PyResult<()> {
        in_place(slf, other, "**", "__pow__").map(drop)
    }

    fn __imatmul__<'py>(slf: Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<()> {
        in_place(slf, other, "@", "__matmul__").map(drop)
    }

    fn __iand__<'py>(slf: Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<()> {
        in_place(slf, other, "&", "__and__").map(drop)
    }

    fn __ior__<'py>(slf: Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<()> {
        in_place(slf, other, "|", "__or__").map(drop)
    }

    fn __ixor__<'py>(slf: Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<()> {
        in_place(slf, other, "^", "__xor__").map(drop)
    }

    fn __ilshift__<'py>(slf: Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<()> {
        in_place(slf, other, "<<", "__lshift__").map(drop)
    }

    fn __irshift__<'py>(slf: Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<()> {
        in_place(slf, other, ">>", "__rshift__").map(drop)
    }
}

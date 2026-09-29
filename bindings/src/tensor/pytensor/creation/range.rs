// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
#[pymethods]
impl PyTensor {
    /// Evenly spaced values over `[start, end)` with the given step.
    #[staticmethod]
    #[pyo3(signature = (start, end=None, step=None, dtype=None, device=None, requires_grad=false))]
    fn arange(
        start: &Bound<PyAny>,
        end: Option<&Bound<PyAny>>,
        step: Option<&Bound<PyAny>>,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let dtype = dtype::resolve_dtype_arg(dtype)?;
        let device = resolve_device(device)?;
        let requires_grad = requires_grad.unwrap_or(false);

        if !dtype.is_float()
            && let Some(tensor) = arange_exact_int(start, end, step, dtype, device)?
        {
            return Self::created(tensor, requires_grad);
        }
        let real = |value: &Bound<PyAny>, name: &str| extract_real_scalar(value, name);
        let start = real(start, "start")?;
        let end = end.map(|value| real(value, "end")).transpose()?;
        let step = step
            .map(|value| real(value, "step"))
            .transpose()?
            .unwrap_or(1.0);

        let (start, end) = match end {
            Some(value) => (start, value),
            None => (0.0, start),
        };

        // The count the engine will derive, checked before it allocates it.
        if step != 0.0 && step.is_finite() && start.is_finite() && end.is_finite() {
            let span = ((end - start) / step).ceil();
            if span.is_finite() && span > 0.0 {
                reject_unallocatable(span as usize, dtype, "arange")?;
            }
        }

        let tensor = create_arange_tensor(start, end, step, dtype, device, requires_grad)?;
        Self::created(tensor, requires_grad)
    }

    /// `steps` values evenly spaced over `[start, end]`, inclusive of both.
    #[staticmethod]
    #[pyo3(signature = (start, end, steps, dtype=None, device=None, requires_grad=false))]
    fn linspace(
        start: f64,
        end: f64,
        steps: usize,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        if steps == 0 {
            return Err(PyValueError::new_err("steps must be greater than zero"));
        }

        let dtype = dtype::resolve_dtype_arg(dtype)?;
        let device = resolve_device(device)?;
        let requires_grad = requires_grad.unwrap_or(false);

        reject_unallocatable(steps, dtype, "linspace")?;
        let tensor = create_linspace_tensor(start, end, steps, dtype, device, requires_grad)?;
        Self::created(tensor, requires_grad)
    }

    /// `steps` values evenly spaced on a log scale between `base ** start` and `base ** end`.
    #[staticmethod]
    #[pyo3(signature = (start, end, steps, base=None, dtype=None, device=None, requires_grad=false))]
    fn logspace(
        start: f64,
        end: f64,
        steps: usize,
        base: Option<f64>,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        if steps == 0 {
            return Err(PyValueError::new_err("steps must be greater than zero"));
        }

        let dtype = dtype::resolve_dtype_arg(dtype)?;
        let device = resolve_device(device)?;
        let requires_grad = requires_grad.unwrap_or(false);
        let base = base.unwrap_or(10.0);

        reject_unallocatable(steps, dtype, "logspace")?;
        let tensor = create_logspace_tensor(start, end, steps, base, dtype, device, requires_grad)?;
        Self::created(tensor, requires_grad)
    }

    /// A tensor holding a copy of a NumPy array's data.
    #[staticmethod]
    #[pyo3(signature = (array, requires_grad=false))]
    fn from_numpy(array: &Bound<PyAny>, requires_grad: bool) -> PyResult<Self> {
        let tensor = convert_numpy_to_tensor(array, requires_grad)?;
        Self::created(tensor, requires_grad)
    }

    /// A tensor over a NumPy array's own memory, with no copy.
    ///
    /// The tensor holds a reference to `array`, so the buffer outlives it, and
    /// the two see each other's writes. That is the point of asking, and also
    /// the hazard: mutating the array changes what every tensor derived from
    /// it reads, including operands a backward pass has saved. ``from_numpy``
    /// is the one that copies, and is what you want unless you have a reason.
    ///
    /// The array must be C-contiguous, in native byte order, aligned, and of a
    /// dtype minitensor stores. Anything else raises rather than quietly
    /// copying: a caller who asked to share should find out when they did not
    /// get it, not discover it later through a write that went nowhere.
    #[staticmethod]
    #[pyo3(signature = (array, requires_grad=false))]
    fn from_numpy_shared(array: &Bound<PyAny>, requires_grad: bool) -> PyResult<Self> {
        Self::created(
            crate::share::shared_pytensor(array, requires_grad)?
                .tensor()
                .clone(),
            requires_grad,
        )
    }
}

/// `arange` over an integer dtype with Python-int bounds, counted exactly.
///
/// The bounds otherwise pass through an f64, where integers past 2^53 are
/// spaced more than one apart: `arange(2**60 + 1, 2**60 + 3)` rounded both
/// ends to the same value and came back empty. `None` when a bound is not a
/// Python int, leaving the general path to answer.
fn arange_exact_int(
    start: &Bound<PyAny>,
    end: Option<&Bound<PyAny>>,
    step: Option<&Bound<PyAny>>,
    dtype: DataType,
    device: Device,
) -> PyResult<Option<Tensor>> {
    // A Python int past int64 has no integer-dtype answer; through a float it
    // saturated to the dtype's bound.
    let exact = |value: &Bound<PyAny>| -> PyResult<Option<i64>> {
        match exact_python_int(value) {
            Some(value) => Ok(Some(value)),
            None if dtype.is_int() && value.is_instance_of::<pyo3::types::PyInt>() => {
                Err(does_not_fit(value.str()?, dtype))
            }
            None => Ok(None),
        }
    };
    let Some(first) = exact(start)? else {
        return Ok(None);
    };
    let (start, end) = match end {
        None => (0, first),
        Some(value) => match exact(value)? {
            Some(end) => (first, end),
            None => return Ok(None),
        },
    };
    let step = match step {
        None => 1,
        Some(value) => match exact(value)? {
            Some(step) => step,
            None => return Ok(None),
        },
    };
    if step == 0 {
        return Err(PyValueError::new_err("Step cannot be zero"));
    }
    // The count is ceil((end - start) / step), in i128 so the span of two
    // int64 bounds cannot overflow.
    let span = end as i128 - start as i128;
    let step_wide = step as i128;
    let count = if (span > 0) == (step_wide > 0) && span != 0 {
        (span + step_wide - step_wide.signum()) / step_wide
    } else {
        0
    };
    let count = usize::try_from(count).map_err(|_| {
        PyValueError::new_err(format!(
            "arange from {start} to {end} by {step} is too long"
        ))
    })?;
    reject_unallocatable(count, dtype, "arange")?;
    // The values run monotonically from the first to the last, so those two
    // decide whether an int32 range holds them all; one that cannot is
    // refused rather than wrapped.
    if dtype == DataType::Int32 && count > 0 {
        let last = start as i128 + (count as i128 - 1) * step_wide;
        for value in [start as i128, last] {
            if i32::try_from(value).is_err() {
                return Err(does_not_fit(value, dtype));
            }
        }
    }
    let data = TensorData::from_index_i64(count, dtype, device, |index| {
        start.wrapping_add((index as i64).wrapping_mul(step))
    });
    Ok(Some(Tensor::new(
        Arc::new(data),
        Shape::new(vec![count]),
        dtype,
        device,
        false,
    )))
}

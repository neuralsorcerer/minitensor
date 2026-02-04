// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

#[pymethods]
impl PyTensor {
    #[staticmethod]
    #[pyo3(signature = (start, end=None, step=1.0, dtype=None, device=None, requires_grad=false))]
    fn arange(
        start: f64,
        end: Option<f64>,
        step: f64,
        dtype: Option<&str>,
        device: Option<&PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<Self> {
        let dtype = dtype::resolve_dtype_arg(dtype)?;
        let device = device.map(|d| d.device()).unwrap_or_else(Device::cpu);
        let requires_grad = requires_grad.unwrap_or(false);

        let (start, end) = match end {
            Some(value) => (start, value),
            None => (0.0, start),
        };

        let tensor = create_arange_tensor(start, end, step, dtype, device, requires_grad)?;
        Ok(Self::from_tensor(tensor))
    }

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
        let device = device.map(|d| d.device()).unwrap_or_else(Device::cpu);
        let requires_grad = requires_grad.unwrap_or(false);

        let tensor = create_linspace_tensor(start, end, steps, dtype, device, requires_grad)?;
        Ok(Self::from_tensor(tensor))
    }

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
        let device = device.map(|d| d.device()).unwrap_or_else(Device::cpu);
        let requires_grad = requires_grad.unwrap_or(false);
        let base = base.unwrap_or(10.0);

        let tensor = create_logspace_tensor(start, end, steps, base, dtype, device, requires_grad)?;
        Ok(Self::from_tensor(tensor))
    }

    #[staticmethod]
    #[pyo3(signature = (array, requires_grad=false))]
    fn from_numpy(array: &Bound<PyAny>, requires_grad: bool) -> PyResult<Self> {
        let tensor = convert_numpy_to_tensor(array, requires_grad)?;
        Ok(Self::from_tensor(tensor))
    }

    #[staticmethod]
    #[pyo3(signature = (array, requires_grad=false))]
    fn from_numpy_shared(array: &Bound<PyAny>, requires_grad: bool) -> PyResult<Self> {
        // For now, same as from_numpy - true zero-copy would require more complex memory management
        Self::from_numpy(array, requires_grad)
    }

}

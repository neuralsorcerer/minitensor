// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

#[pymethods]
impl PyTensor {
    // Reduction operations
    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn sum(&self, dim: Option<&Bound<PyAny>>, keepdim: Option<bool>) -> PyResult<Self> {
        let keepdim = keepdim.unwrap_or(false);
        let dims = normalize_optional_axes(dim)?;
        let result = self.inner.sum(dims, keepdim).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn nansum(&self, dim: Option<&Bound<PyAny>>, keepdim: Option<bool>) -> PyResult<Self> {
        let keepdim = keepdim.unwrap_or(false);
        let dims = normalize_optional_axes(dim)?;
        let result = self.inner.nansum(dims, keepdim).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn logsumexp(&self, dim: Option<&Bound<PyAny>>, keepdim: Option<bool>) -> PyResult<Self> {
        let keepdim = keepdim.unwrap_or(false);
        let dims = normalize_optional_axes(dim)?;
        match self.inner.logsumexp(dims, keepdim) {
            Ok(result) => Ok(Self::from_tensor(result)),
            Err(err @ MinitensorError::InvalidOperation { .. }) => {
                Err(PyRuntimeError::new_err(err.detailed_message()))
            }
            Err(err) => Err(_convert_error(err)),
        }
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn prod(&self, dim: Option<&Bound<PyAny>>, keepdim: Option<bool>) -> PyResult<Self> {
        let keepdim = keepdim.unwrap_or(false);
        let dims = normalize_optional_axes(dim)?;
        let result = self.inner.prod(dims, keepdim).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn mean(&self, dim: Option<&Bound<PyAny>>, keepdim: Option<bool>) -> PyResult<Self> {
        let keepdim = keepdim.unwrap_or(false);
        let dims = normalize_optional_axes(dim)?;
        let result = self.inner.mean(dims, keepdim).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn nanmean(&self, dim: Option<&Bound<PyAny>>, keepdim: Option<bool>) -> PyResult<Self> {
        let keepdim = keepdim.unwrap_or(false);
        let dims = normalize_optional_axes(dim)?;
        let result = self.inner.nanmean(dims, keepdim).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn all(&self, dim: Option<isize>, keepdim: Option<bool>) -> PyResult<Self> {
        let keepdim = keepdim.unwrap_or(false);
        let result = self.inner.all(dim, keepdim).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn any(&self, dim: Option<isize>, keepdim: Option<bool>) -> PyResult<Self> {
        let keepdim = keepdim.unwrap_or(false);
        let result = self.inner.any(dim, keepdim).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (dim))]
    pub fn cumsum(&self, dim: isize) -> PyResult<Self> {
        let result = self.inner.cumsum(dim).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (dim))]
    pub fn cumprod(&self, dim: isize) -> PyResult<Self> {
        let result = self.inner.cumprod(dim).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn max<'py>(
        &self,
        py: Python<'py>,
        dim: Option<isize>,
        keepdim: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let keepdim = keepdim.unwrap_or(false);
        if let Some(dim) = dim {
            let (values, indices) = self
                .inner
                .max_with_indices(dim, keepdim)
                .map_err(_convert_error)?;
            let values = Py::new(py, PyTensor::from_tensor(values))?.into_any();
            let indices = Py::new(py, PyTensor::from_tensor(indices))?.into_any();
            let tuple = PyTuple::new(py, [values, indices])?;
            Ok(tuple.into_any().unbind())
        } else {
            Ok(Py::new(py, self.max_values(None, keepdim)?)?.into_any())
        }
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn nanmax<'py>(
        &self,
        py: Python<'py>,
        dim: Option<isize>,
        keepdim: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let keepdim = keepdim.unwrap_or(false);
        if let Some(dim) = dim {
            let (values, indices) = self
                .inner
                .nanmax_with_indices(dim, keepdim)
                .map_err(_convert_error)?;
            let values = Py::new(py, PyTensor::from_tensor(values))?.into_any();
            let indices = Py::new(py, PyTensor::from_tensor(indices))?.into_any();
            let tuple = PyTuple::new(py, [values, indices])?;
            Ok(tuple.into_any().unbind())
        } else {
            Ok(Py::new(py, self.nanmax_values(None, keepdim)?)?.into_any())
        }
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn min<'py>(
        &self,
        py: Python<'py>,
        dim: Option<isize>,
        keepdim: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let keepdim = keepdim.unwrap_or(false);
        if let Some(dim) = dim {
            let (values, indices) = self
                .inner
                .min_with_indices(dim, keepdim)
                .map_err(_convert_error)?;
            let values = Py::new(py, PyTensor::from_tensor(values))?.into_any();
            let indices = Py::new(py, PyTensor::from_tensor(indices))?.into_any();
            let tuple = PyTuple::new(py, [values, indices])?;
            Ok(tuple.into_any().unbind())
        } else {
            Ok(Py::new(py, self.min_values(None, keepdim)?)?.into_any())
        }
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn nanmin<'py>(
        &self,
        py: Python<'py>,
        dim: Option<isize>,
        keepdim: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let keepdim = keepdim.unwrap_or(false);
        if let Some(dim) = dim {
            let (values, indices) = self
                .inner
                .nanmin_with_indices(dim, keepdim)
                .map_err(_convert_error)?;
            let values = Py::new(py, PyTensor::from_tensor(values))?.into_any();
            let indices = Py::new(py, PyTensor::from_tensor(indices))?.into_any();
            let tuple = PyTuple::new(py, [values, indices])?;
            Ok(tuple.into_any().unbind())
        } else {
            Ok(Py::new(py, self.nanmin_values(None, keepdim)?)?.into_any())
        }
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn median<'py>(
        &self,
        py: Python<'py>,
        dim: Option<isize>,
        keepdim: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let keepdim = keepdim.unwrap_or(false);
        let (values, indices) = self.median_with_indices(dim, keepdim)?;
        if dim.is_some() {
            let indices = indices.ok_or_else(|| {
                PyRuntimeError::new_err("median returned no indices for the requested dimension")
            })?;
            let values = Py::new(py, values)?.into_any();
            let indices = Py::new(py, indices)?.into_any();
            let tuple = PyTuple::new(py, [values, indices])?;
            Ok(tuple.into_any().unbind())
        } else {
            Ok(Py::new(py, values)?.into_any())
        }
    }

    #[pyo3(signature = (q, dim=None, keepdim=false, interpolation="linear"))]
    pub fn quantile(
        &self,
        q: &Bound<PyAny>,
        dim: Option<isize>,
        keepdim: Option<bool>,
        interpolation: Option<&str>,
    ) -> PyResult<Self> {
        let keepdim = keepdim.unwrap_or(false);
        let interpolation = parse_quantile_interpolation(interpolation)?;
        match parse_quantile_arg(q)? {
            QuantileArg::Scalar(prob) => {
                let result = self
                    .inner
                    .quantile(prob, dim, keepdim, interpolation)
                    .map_err(_convert_error)?;
                Ok(Self::from_tensor(result))
            }
            QuantileArg::Multiple(qs) => {
                let result = self
                    .inner
                    .quantiles(&qs, dim, keepdim, interpolation)
                    .map_err(_convert_error)?;
                Ok(Self::from_tensor(result))
            }
        }
    }

    #[pyo3(signature = (q, dim=None, keepdim=false, interpolation="linear"))]
    pub fn nanquantile(
        &self,
        q: &Bound<PyAny>,
        dim: Option<isize>,
        keepdim: Option<bool>,
        interpolation: Option<&str>,
    ) -> PyResult<Self> {
        let keepdim = keepdim.unwrap_or(false);
        let interpolation = parse_quantile_interpolation(interpolation)?;
        match parse_quantile_arg(q)? {
            QuantileArg::Scalar(prob) => {
                let result = self
                    .inner
                    .nanquantile(prob, dim, keepdim, interpolation)
                    .map_err(_convert_error)?;
                Ok(Self::from_tensor(result))
            }
            QuantileArg::Multiple(qs) => {
                let result = self
                    .inner
                    .nanquantiles(&qs, dim, keepdim, interpolation)
                    .map_err(_convert_error)?;
                Ok(Self::from_tensor(result))
            }
        }
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn argmax(&self, dim: Option<isize>, keepdim: Option<bool>) -> PyResult<Self> {
        let keepdim = keepdim.unwrap_or(false);
        let result = self.inner.argmax(dim, keepdim).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn argmin(&self, dim: Option<isize>, keepdim: Option<bool>) -> PyResult<Self> {
        let keepdim = keepdim.unwrap_or(false);
        let result = self.inner.argmin(dim, keepdim).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (k, dim=None, largest=true, sorted=true))]
    pub fn topk(
        &self,
        k: usize,
        dim: Option<isize>,
        largest: Option<bool>,
        sorted: Option<bool>,
    ) -> PyResult<(Self, Self)> {
        let largest = largest.unwrap_or(true);
        let sorted = sorted.unwrap_or(true);
        match self.inner.topk(k, dim, largest, sorted) {
            Ok((values, indices)) => Ok((Self::from_tensor(values), Self::from_tensor(indices))),
            Err(err @ MinitensorError::InvalidArgument { .. }) => {
                Err(PyRuntimeError::new_err(err.detailed_message()))
            }
            Err(err) => Err(_convert_error(err)),
        }
    }

    #[pyo3(signature = (dim=None, descending=false, stable=false))]
    pub fn sort(
        &self,
        dim: Option<isize>,
        descending: Option<bool>,
        stable: Option<bool>,
    ) -> PyResult<(Self, Self)> {
        let descending = descending.unwrap_or(false);
        let stable = stable.unwrap_or(false);
        match self.inner.sort(dim, descending, stable) {
            Ok((values, indices)) => Ok((Self::from_tensor(values), Self::from_tensor(indices))),
            Err(err @ MinitensorError::InvalidArgument { .. }) => {
                Err(PyRuntimeError::new_err(err.detailed_message()))
            }
            Err(err) => Err(_convert_error(err)),
        }
    }

    #[pyo3(signature = (dim=None, descending=false, stable=false))]
    pub fn argsort(
        &self,
        dim: Option<isize>,
        descending: Option<bool>,
        stable: Option<bool>,
    ) -> PyResult<Self> {
        let descending = descending.unwrap_or(false);
        let stable = stable.unwrap_or(false);
        match self.inner.argsort(dim, descending, stable) {
            Ok(indices) => Ok(Self::from_tensor(indices)),
            Err(err @ MinitensorError::InvalidArgument { .. }) => {
                Err(PyRuntimeError::new_err(err.detailed_message()))
            }
            Err(err) => Err(_convert_error(err)),
        }
    }

    #[pyo3(signature = (dim=None, unbiased=true, keepdim=false))]
    pub fn std(
        &self,
        dim: Option<isize>,
        unbiased: Option<bool>,
        keepdim: Option<bool>,
    ) -> PyResult<Self> {
        let keepdim = keepdim.unwrap_or(false);
        let unbiased = unbiased.unwrap_or(true);
        let result = self
            .inner
            .std(dim, keepdim, unbiased)
            .map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (dim=None, unbiased=true, keepdim=false))]
    pub fn var(
        &self,
        dim: Option<isize>,
        unbiased: Option<bool>,
        keepdim: Option<bool>,
    ) -> PyResult<Self> {
        let keepdim = keepdim.unwrap_or(false);
        let unbiased = unbiased.unwrap_or(true);
        let result = self
            .inner
            .var(dim, keepdim, unbiased)
            .map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

}

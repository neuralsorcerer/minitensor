// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
#[pymethods]
impl PyTensor {
    /// Sum over `dim`, or over every element when `dim` is omitted.
    // Reduction operations
    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn sum(&self, dim: Option<&Bound<PyAny>>, keepdim: Option<bool>) -> PyResult<Self> {
        let keepdim = keepdim.unwrap_or(false);
        let dims = normalize_optional_axes(dim)?;
        let result = self.inner.sum(dims, keepdim).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Like `sum`, treating NaN as zero. An all-NaN slice sums to 0.
    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn nansum(&self, dim: Option<&Bound<PyAny>>, keepdim: Option<bool>) -> PyResult<Self> {
        let keepdim = keepdim.unwrap_or(false);
        let dims = normalize_optional_axes(dim)?;
        let result = self.inner.nansum(dims, keepdim).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Vector `p`-norm over `dim`, or over every element when `dim` is omitted.
    #[pyo3(signature = (p=None, dim=None, keepdim=false))]
    pub fn norm(
        &self,
        p: Option<&Bound<PyAny>>,
        dim: Option<&Bound<PyAny>>,
        keepdim: Option<bool>,
    ) -> PyResult<Self> {
        let keepdim = keepdim.unwrap_or(false);
        let order = parse_norm_order(p)?;
        let dims = normalize_optional_axes(dim)?;
        let result = self
            .inner
            .norm(order, dims, keepdim)
            .map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// `log(sum(exp(x)))` over `dim`, shifted by the maximum so large values do not overflow.
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

    /// Product over `dim`, or over every element when `dim` is omitted.
    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn prod(&self, dim: Option<&Bound<PyAny>>, keepdim: Option<bool>) -> PyResult<Self> {
        let keepdim = keepdim.unwrap_or(false);
        let dims = normalize_optional_axes(dim)?;
        let result = self.inner.prod(dims, keepdim).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Arithmetic mean over `dim`, or over every element when `dim` is omitted.
    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn mean(&self, dim: Option<&Bound<PyAny>>, keepdim: Option<bool>) -> PyResult<Self> {
        let keepdim = keepdim.unwrap_or(false);
        let dims = normalize_optional_axes(dim)?;
        let result = self.inner.mean(dims, keepdim).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Like `mean`, ignoring NaN. A slice that is entirely NaN gives NaN.
    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn nanmean(&self, dim: Option<&Bound<PyAny>>, keepdim: Option<bool>) -> PyResult<Self> {
        let keepdim = keepdim.unwrap_or(false);
        let dims = normalize_optional_axes(dim)?;
        let result = self.inner.nanmean(dims, keepdim).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Whether every element is true (or non-zero) over `dim`.
    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn all(&self, dim: Option<isize>, keepdim: Option<bool>) -> PyResult<Self> {
        let keepdim = keepdim.unwrap_or(false);
        let result = self.inner.all(dim, keepdim).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Whether any element is true (or non-zero) over `dim`.
    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn any(&self, dim: Option<isize>, keepdim: Option<bool>) -> PyResult<Self> {
        let keepdim = keepdim.unwrap_or(false);
        let result = self.inner.any(dim, keepdim).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Running sum along `dim`, keeping the input's shape.
    #[pyo3(signature = (dim))]
    pub fn cumsum(&self, dim: isize) -> PyResult<Self> {
        let result = self.inner.cumsum(dim).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Running product along `dim`, keeping the input's shape.
    #[pyo3(signature = (dim))]
    pub fn cumprod(&self, dim: isize) -> PyResult<Self> {
        let result = self.inner.cumprod(dim).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Largest element over `dim`, values only.
    ///
    /// `max(dim=...)` also reports where the maximum was, and tracking that
    /// index is most of the work: it defeats the vectorized fold that the plain
    /// comparison compiles to. On a 2048x1024 float32 matrix the values-only
    /// reduction takes 0.121ms against 0.895ms for the pair, so a caller who
    /// discards the indices pays 7.4x for them.
    ///
    /// Named for NumPy and PyTorch, which both spell this `amax`.
    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn amax(&self, dim: Option<isize>, keepdim: Option<bool>) -> PyResult<Self> {
        self.max_values(dim, keepdim.unwrap_or(false))
    }

    /// Smallest element over `dim`, values only. See [`Self::amax`].
    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn amin(&self, dim: Option<isize>, keepdim: Option<bool>) -> PyResult<Self> {
        self.min_values(dim, keepdim.unwrap_or(false))
    }

    /// Like `amax`, ignoring NaN. A slice that is all NaN reduces to NaN.
    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn nanamax(&self, dim: Option<isize>, keepdim: Option<bool>) -> PyResult<Self> {
        self.nanmax_values(dim, keepdim.unwrap_or(false))
    }

    /// Like `amin`, ignoring NaN. A slice that is all NaN reduces to NaN.
    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn nanamin(&self, dim: Option<isize>, keepdim: Option<bool>) -> PyResult<Self> {
        self.nanmin_values(dim, keepdim.unwrap_or(false))
    }

    /// Largest element over `dim`; with a `dim` it returns the values and their indices.
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

    /// Like `max`, ignoring NaN.
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

    /// Smallest element over `dim`; with a `dim` it returns the values and their indices.
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

    /// Like `min`, ignoring NaN.
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

    /// Middle element over `dim`. For an even count this is the lower of the two, not their average -- use `quantile(0.5)` for the interpolated definition.
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

    /// Like `median`, ignoring NaN.
    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn nanmedian(&self, dim: Option<isize>, keepdim: Option<bool>) -> PyResult<Self> {
        let keepdim = keepdim.unwrap_or(false);
        let result = self.inner.nanmedian(dim, keepdim).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// The `q`-th quantile over `dim`, interpolating between neighbouring elements.
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

    /// Like `quantile`, ignoring NaN.
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

    /// Index of the largest element over `dim`. Ties go to the first occurrence.
    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn argmax(&self, dim: Option<isize>, keepdim: Option<bool>) -> PyResult<Self> {
        let keepdim = keepdim.unwrap_or(false);
        let result = self.inner.argmax(dim, keepdim).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Index of the smallest element over `dim`. Ties go to the first occurrence.
    #[pyo3(signature = (dim=None, keepdim=false))]
    pub fn argmin(&self, dim: Option<isize>, keepdim: Option<bool>) -> PyResult<Self> {
        let keepdim = keepdim.unwrap_or(false);
        let result = self.inner.argmin(dim, keepdim).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// The `k` largest elements along `dim`, with their indices. Pass `largest=False` for the smallest.
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

    /// Sort along `dim`, returning the sorted values and the indices that produced them.
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

    /// The indices that would sort along `dim`.
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

    /// Standard deviation over `dim`. `unbiased` applies Bessel's correction.
    #[pyo3(signature = (dim=None, unbiased=true, keepdim=false))]
    pub fn std(
        &self,
        dim: Option<&Bound<PyAny>>,
        unbiased: Option<bool>,
        keepdim: Option<bool>,
    ) -> PyResult<Self> {
        let keepdim = keepdim.unwrap_or(false);
        let unbiased = unbiased.unwrap_or(true);
        let dims = normalize_optional_axes(dim)?;
        let result = self
            .inner
            .std(dims, keepdim, unbiased)
            .map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Variance over `dim`. `unbiased` applies Bessel's correction.
    #[pyo3(signature = (dim=None, unbiased=true, keepdim=false))]
    pub fn var(
        &self,
        dim: Option<&Bound<PyAny>>,
        unbiased: Option<bool>,
        keepdim: Option<bool>,
    ) -> PyResult<Self> {
        let keepdim = keepdim.unwrap_or(false);
        let unbiased = unbiased.unwrap_or(true);
        let dims = normalize_optional_axes(dim)?;
        let result = self
            .inner
            .var(dims, keepdim, unbiased)
            .map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }
}

/// Interpret the `p` argument of `norm`.
///
/// Accepts a number, or the string `"fro"`. Frobenius is the 2-norm of the
/// flattened tensor, so it is the same computation — it is accepted because
/// that is the name the matrix case goes by and callers reach for it.
pub(crate) fn parse_norm_order(p: Option<&Bound<PyAny>>) -> PyResult<f64> {
    let Some(value) = p else {
        return Ok(2.0);
    };
    if let Ok(name) = value.extract::<String>() {
        return match name.as_str() {
            "fro" => Ok(2.0),
            other => Err(PyValueError::new_err(format!(
                "unsupported norm order '{other}'; expected a number or 'fro'"
            ))),
        };
    }
    value
        .extract::<f64>()
        .map_err(|_| PyTypeError::new_err("norm order p must be a number or 'fro'"))
}

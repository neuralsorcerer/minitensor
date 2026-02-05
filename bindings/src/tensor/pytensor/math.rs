// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

#[pymethods]
impl PyTensor {
    // Mathematical functions
    fn abs(&self) -> PyResult<Self> {
        let result = self.inner.abs().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    fn sqrt(&self) -> PyResult<Self> {
        let result = self.inner.sqrt().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn rsqrt(&self) -> PyResult<Self> {
        let result = self.inner.rsqrt().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    fn pow(&self, exponent: &Bound<PyAny>) -> PyResult<Self> {
        if let Ok(exp_tensor) = exponent.extract::<PyTensor>() {
            let result = self.inner.pow(&exp_tensor.inner).map_err(_convert_error)?;
            return Ok(Self::from_tensor(result));
        }

        if let Ok(exp) = exponent.extract::<f64>() {
            let result = self.inner.powf(exp).map_err(_convert_error)?;
            return Ok(Self::from_tensor(result));
        }

        let exp_tensor = tensor_from_py_value(&self.inner, exponent)?;
        let result = self.inner.pow(&exp_tensor).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    fn exp(&self) -> PyResult<Self> {
        let result = self.inner.exp().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    fn log(&self) -> PyResult<Self> {
        let result = self.inner.log().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn log1p(&self) -> PyResult<Self> {
        let result = self.inner.log1p().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn expm1(&self) -> PyResult<Self> {
        let result = self.inner.expm1().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn sin(&self) -> PyResult<Self> {
        let result = self.inner.sin().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn cos(&self) -> PyResult<Self> {
        let result = self.inner.cos().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn tan(&self) -> PyResult<Self> {
        let result = self.inner.tan().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn asin(&self) -> PyResult<Self> {
        let result = self.inner.asin().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn acos(&self) -> PyResult<Self> {
        let result = self.inner.acos().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn atan(&self) -> PyResult<Self> {
        let result = self.inner.atan().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn sinh(&self) -> PyResult<Self> {
        let result = self.inner.sinh().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn cosh(&self) -> PyResult<Self> {
        let result = self.inner.cosh().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn asinh(&self) -> PyResult<Self> {
        let result = self.inner.asinh().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn acosh(&self) -> PyResult<Self> {
        let result = self.inner.acosh().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn atanh(&self) -> PyResult<Self> {
        let result = self.inner.atanh().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    fn isnan(&self) -> PyResult<Self> {
        let result = self.inner.isnan().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    fn isinf(&self) -> PyResult<Self> {
        let result = self.inner.isinf().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    fn isfinite(&self) -> PyResult<Self> {
        let result = self.inner.isfinite().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    fn __pow__(&self, exponent: &Bound<PyAny>, _mod: Option<&Bound<PyAny>>) -> PyResult<Self> {
        self.pow(exponent)
    }

    fn __rpow__(&self, base: &Bound<PyAny>, _mod: Option<&Bound<PyAny>>) -> PyResult<Self> {
        let base_tensor = tensor_from_py_value(&self.inner, base)?;
        let result = base_tensor.pow(&self.inner).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn relu(&self) -> PyResult<Self> {
        let result = self.inner.relu().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn hardshrink(&self, lambd: Option<f64>) -> PyResult<Self> {
        let result = self
            .inner
            .hardshrink(lambd.unwrap_or(0.5))
            .map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (dim=None))]
    pub fn softmax(&self, dim: Option<isize>) -> PyResult<Self> {
        let resolved_dim = match dim {
            Some(dim) => {
                let ndim = self.inner.ndim() as isize;
                let dim = if dim < 0 { dim + ndim } else { dim };
                if dim < 0 || dim >= ndim {
                    return Err(PyIndexError::new_err(format!(
                        "Dimension out of range (expected to be in range of [-{ndim}, {ndim}), but got {dim})"
                    )));
                }
                Some(dim as usize)
            }
            None => None,
        };

        let result = self.inner.softmax(resolved_dim).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (dim=None))]
    pub fn log_softmax(&self, dim: Option<isize>) -> PyResult<Self> {
        let resolved_dim = match dim {
            Some(dim) => {
                let ndim = self.inner.ndim() as isize;
                let dim = if dim < 0 { dim + ndim } else { dim };
                if dim < 0 || dim >= ndim {
                    return Err(PyIndexError::new_err(format!(
                        "Dimension out of range (expected to be in range of [-{ndim}, {ndim}), but got {dim})"
                    )));
                }
                Some(dim as usize)
            }
            None => None,
        };

        let result = self
            .inner
            .log_softmax(resolved_dim)
            .map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (mask, dim=None))]
    pub fn masked_softmax(&self, mask: &Bound<PyAny>, dim: Option<isize>) -> PyResult<Self> {
        let mask_tensor = tensor_from_py_value(&self.inner, mask)?;
        let resolved_dim = match dim {
            Some(dim) => {
                let ndim = self.inner.ndim() as isize;
                let dim = if dim < 0 { dim + ndim } else { dim };
                if dim < 0 || dim >= ndim {
                    return Err(PyIndexError::new_err(format!(
                        "Dimension out of range (expected to be in range of [-{ndim}, {ndim}), but got {dim})"
                    )));
                }
                Some(dim as usize)
            }
            None => None,
        };

        let result = self
            .inner
            .masked_softmax(&mask_tensor, resolved_dim)
            .map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (mask, dim=None))]
    pub fn masked_log_softmax(&self, mask: &Bound<PyAny>, dim: Option<isize>) -> PyResult<Self> {
        let mask_tensor = tensor_from_py_value(&self.inner, mask)?;
        let resolved_dim = match dim {
            Some(dim) => {
                let ndim = self.inner.ndim() as isize;
                let dim = if dim < 0 { dim + ndim } else { dim };
                if dim < 0 || dim >= ndim {
                    return Err(PyIndexError::new_err(format!(
                        "Dimension out of range (expected to be in range of [-{ndim}, {ndim}), but got {dim})"
                    )));
                }
                Some(dim as usize)
            }
            None => None,
        };

        let result = self
            .inner
            .masked_log_softmax(&mask_tensor, resolved_dim)
            .map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (normalized_shape, weight=None, bias=None, eps=1e-5))]
    pub fn layer_norm(
        &self,
        normalized_shape: Vec<usize>,
        weight: Option<&PyTensor>,
        bias: Option<&PyTensor>,
        eps: Option<f64>,
    ) -> PyResult<Self> {
        if normalized_shape.is_empty() {
            return Err(PyValueError::new_err(
                "layer_norm requires normalized_shape to contain at least one dimension",
            ));
        }

        let weight_inner = weight.map(|w| &w.inner);
        let bias_inner = bias.map(|b| &b.inner);
        let result = self
            .inner
            .layer_norm(
                &normalized_shape,
                weight_inner,
                bias_inner,
                eps.unwrap_or(1e-5),
            )
            .map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn gelu(&self, approximate: Option<&str>) -> PyResult<Self> {
        let approx_mode = approximate.unwrap_or("none");
        let approximate = if approx_mode.eq_ignore_ascii_case("none") {
            false
        } else if approx_mode.eq_ignore_ascii_case("tanh") {
            true
        } else {
            return Err(PyValueError::new_err(
                "approximate must be 'none' or 'tanh' for gelu",
            ));
        };

        let result = self.inner.gelu(approximate).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn sigmoid(&self) -> PyResult<Self> {
        let result = self.inner.sigmoid().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn softplus(&self, beta: Option<f64>, threshold: Option<f64>) -> PyResult<Self> {
        let result = self
            .inner
            .softplus(beta.unwrap_or(1.0), threshold.unwrap_or(20.0))
            .map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn elu(&self, alpha: Option<f64>) -> PyResult<Self> {
        let result = self
            .inner
            .elu(alpha.unwrap_or(1.0))
            .map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn selu(&self) -> PyResult<Self> {
        let result = self.inner.selu().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn silu(&self) -> PyResult<Self> {
        let result = self.inner.silu().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn softsign(&self) -> PyResult<Self> {
        let result = self.inner.softsign().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn tanh(&self) -> PyResult<Self> {
        let result = self.inner.tanh().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

}

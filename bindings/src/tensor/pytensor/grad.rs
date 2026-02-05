// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

#[pymethods]
impl PyTensor {
    // Gradient operations
    #[pyo3(signature = (gradient=None, retain_graph=false, create_graph=false))]
    fn backward(
        &self,
        gradient: Option<&Bound<PyAny>>,
        retain_graph: bool,
        create_graph: bool,
    ) -> PyResult<()> {
        if create_graph {
            return Err(PyNotImplementedError::new_err(
                "create_graph=True is not supported; all computations execute in the Rust backend",
            ));
        }

        if !self.requires_grad() && self.is_leaf() {
            return Err(PyRuntimeError::new_err(
                "element 0 of tensors does not require grad and does not have a grad_fn",
            ));
        }

        if !retain_graph && engine::autograd::is_graph_consumed() {
            return Err(PyRuntimeError::new_err(
                "Computation graph has been freed. Re-run the forward pass or call backward(retain_graph=True).",
            ));
        }

        let grad_tensor = if let Some(value) = gradient {
            if value.is_none() {
                None
            } else if let Ok(py_tensor) = value.extract::<PyTensor>() {
                let mut tensor = py_tensor.inner.clone();
                ensure_backward_gradient_compatible(&self.inner, &mut tensor)?;
                Some(tensor)
            } else {
                let mut tensor = tensor_from_py_value(&self.inner, value)?;
                ensure_backward_gradient_compatible(&self.inner, &mut tensor)?;
                Some(tensor)
            }
        } else {
            None
        };

        self.inner.backward(grad_tensor).map_err(_convert_error)?;

        if !retain_graph {
            engine::autograd::mark_graph_consumed();
        }

        Ok(())
    }

    pub fn requires_grad_(&mut self, requires_grad: bool) -> PyResult<()> {
        self.inner = self.inner.clone().requires_grad_(requires_grad);
        Ok(())
    }

    #[pyo3(signature = (source, *, non_blocking=false))]
    fn copy_<'py>(
        mut slf: PyRefMut<'py, Self>,
        source: &Bound<PyAny>,
        non_blocking: Option<bool>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        if non_blocking.unwrap_or(false) {
            return Err(PyNotImplementedError::new_err(
                "non_blocking copy_ is not implemented",
            ));
        }

        let reference = PyTensor::from_python_value(source)?;
        slf.inner
            .copy_(reference.tensor())
            .map_err(_convert_error)?;
        register_leaf_tensor(&slf.inner);
        Ok(slf)
    }

    fn fill_<'py>(
        mut slf: PyRefMut<'py, Self>,
        value: &Bound<PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let fill_value = extract_real_scalar(value, "value")?;
        slf.inner.fill_(fill_value).map_err(_convert_error)?;
        register_leaf_tensor(&slf.inner);
        Ok(slf)
    }

    #[pyo3(signature = (set_to_none=false))]
    fn zero_grad(&mut self, set_to_none: bool) {
        self.inner.zero_grad(set_to_none);
    }

}

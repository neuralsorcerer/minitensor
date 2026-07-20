// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
#[pymethods]
impl PyTensor {
    #[pyo3(signature = (other, rtol=None, atol=None, equal_nan=false))]
    pub fn isclose(
        &self,
        other: &Bound<PyAny>,
        rtol: Option<f64>,
        atol: Option<f64>,
        equal_nan: bool,
    ) -> PyResult<Self> {
        let rtol = rtol.unwrap_or(1e-5);
        let atol = atol.unwrap_or(1e-8);
        let (lhs, rhs) =
            prepare_binary_operands_from_py(&self.inner, other, false, BinaryOpKind::Add)?;
        let result = engine::ops::comparison::isclose(&lhs, &rhs, rtol, atol, equal_nan)
            .map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }
}

// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
#[pymethods]
impl PyTensor {
    // Comparison operators as Python dunder methods
    fn __eq__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.eq_from_py(other)
    }

    fn __ne__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.ne_from_py(other)
    }

    fn __lt__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.lt_from_py(other)
    }

    fn __le__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.le_from_py(other)
    }

    fn __gt__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.gt_from_py(other)
    }

    fn __ge__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.ge_from_py(other)
    }

    pub fn matmul(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_tensor = tensor_from_py_value(&self.inner, other)?;
        let result = self.inner.matmul(&other_tensor).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    fn __matmul__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.matmul(other)
    }

    fn __rmatmul__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_tensor = tensor_from_py_value(&self.inner, other)?;
        let result = other_tensor.matmul(&self.inner).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn solve(&self, rhs: &Bound<PyAny>) -> PyResult<Self> {
        let rhs_tensor = tensor_from_py_value(&self.inner, rhs)?;
        let result = self.inner.solve(&rhs_tensor).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Determinant of each square matrix. Overflows for large matrices -- see `slogdet`.
    pub fn det(&self) -> PyResult<Self> {
        Ok(Self::from_tensor(self.inner.det().map_err(_convert_error)?))
    }

    /// `(sign, logabsdet)` such that `sign * exp(logabsdet)` is the determinant. A singular matrix gives sign 0 and `-inf`.
    pub fn slogdet(&self) -> PyResult<(Self, Self)> {
        let (sign, logabsdet) = self.inner.slogdet().map_err(_convert_error)?;
        Ok((Self::from_tensor(sign), Self::from_tensor(logabsdet)))
    }

    /// Inverse of each square matrix, solved against the identity.
    pub fn inv(&self) -> PyResult<Self> {
        Ok(Self::from_tensor(self.inner.inv().map_err(_convert_error)?))
    }

    /// Cholesky factor `L` with `A = L @ L.T`, or `U` with `A = U.T @ U` when `upper=True`. Only the lower triangle is read; the matrix must be positive definite.
    #[pyo3(signature = (upper=false))]
    pub fn cholesky(&self, upper: bool) -> PyResult<Self> {
        Ok(Self::from_tensor(
            self.inner.cholesky(upper).map_err(_convert_error)?,
        ))
    }

    /// `(Q, R)` with `A = Q @ R`, `Q` orthonormal and `R` upper triangular. `mode="reduced"` gives the thin factors, `mode="complete"` a full square `Q`.
    #[pyo3(signature = (mode="reduced"))]
    pub fn qr(&self, mode: &str) -> PyResult<(Self, Self)> {
        let parsed = engine::ops::linalg::QrMode::from_name(mode).map_err(_convert_error)?;
        let (q, r) = self.inner.qr(parsed).map_err(_convert_error)?;
        Ok((Self::from_tensor(q), Self::from_tensor(r)))
    }

    /// The packed `LU` of each matrix, with its row exchanges as zero-based `int64`. `L` is unit lower triangular and lives strictly below the diagonal; `U` is on and above it. Returned detached -- the differentiable ways to ask about these matrices are `solve`, `det`, `slogdet` and `inv`, which share this factorisation.
    pub fn lu_factor(&self) -> PyResult<(Self, Self)> {
        let (factors, pivots) = self.inner.lu_factor().map_err(_convert_error)?;
        Ok((Self::from_tensor(factors), Self::from_tensor(pivots)))
    }

    /// `(P, L, U)` for each matrix, with `A = P @ L @ U`. Built from the packed form rather than computed separately.
    pub fn lu(&self) -> PyResult<(Self, Self, Self)> {
        let (p, l, u) = self.inner.lu().map_err(_convert_error)?;
        Ok((
            Self::from_tensor(p),
            Self::from_tensor(l),
            Self::from_tensor(u),
        ))
    }

    /// Solve `A X = other` from the factorisation `lu_factor` produced, without factorising again.
    pub fn lu_solve(&self, pivots: &Bound<PyAny>, other: &Bound<PyAny>) -> PyResult<Self> {
        let exchanges = tensor_from_py_value(&self.inner, pivots)?;
        let rhs = tensor_from_py_value(&self.inner, other)?;
        Ok(Self::from_tensor(
            self.inner
                .lu_solve(&exchanges, &rhs)
                .map_err(_convert_error)?,
        ))
    }

    /// Solve `self X = other` for triangular `self`. Only the named triangle is read. `left=False` solves `X self = other` instead; `unitriangular=True` treats the diagonal as ones.
    #[pyo3(signature = (other, upper=false, left=true, unitriangular=false))]
    pub fn solve_triangular(
        &self,
        other: &Bound<PyAny>,
        upper: bool,
        left: bool,
        unitriangular: bool,
    ) -> PyResult<Self> {
        let rhs = tensor_from_py_value(&self.inner, other)?;
        Ok(Self::from_tensor(
            self.inner
                .solve_triangular(&rhs, upper, left, unitriangular)
                .map_err(_convert_error)?,
        ))
    }

    /// Solve `A X = self` given the Cholesky factor of `A` rather than `A`. Two triangular solves and nothing else.
    #[pyo3(signature = (factor, upper=false))]
    pub fn cholesky_solve(&self, factor: &Bound<PyAny>, upper: bool) -> PyResult<Self> {
        let l = tensor_from_py_value(&self.inner, factor)?;
        Ok(Self::from_tensor(
            self.inner
                .cholesky_solve(&l, upper)
                .map_err(_convert_error)?,
        ))
    }

    /// The Moore-Penrose pseudo-inverse. `rcond` is relative to the largest singular value; `None` uses `max(m, n) * eps`.
    #[pyo3(signature = (rcond=None))]
    pub fn pinv(&self, rcond: Option<f64>) -> PyResult<Self> {
        Ok(Self::from_tensor(
            self.inner.pinv(rcond).map_err(_convert_error)?,
        ))
    }

    /// The number of singular values above the tolerance, as `int64`.
    #[pyo3(signature = (tol=None))]
    pub fn matrix_rank(&self, tol: Option<f64>) -> PyResult<Self> {
        Ok(Self::from_tensor(
            self.inner.matrix_rank(tol).map_err(_convert_error)?,
        ))
    }

    /// The 2-norm condition number: the largest singular value over the smallest.
    pub fn cond(&self) -> PyResult<Self> {
        Ok(Self::from_tensor(
            self.inner.cond().map_err(_convert_error)?,
        ))
    }

    /// The least-squares solution of `self @ x = other`, of smallest norm when there are many.
    #[pyo3(signature = (other, rcond=None))]
    pub fn lstsq(&self, other: &Bound<PyAny>, rcond: Option<f64>) -> PyResult<Self> {
        let rhs = tensor_from_py_value(&self.inner, other)?;
        Ok(Self::from_tensor(
            self.inner.lstsq(&rhs, rcond).map_err(_convert_error)?,
        ))
    }

    /// `self` raised to an integer matrix power. Zero gives the identity; a negative power inverts first.
    pub fn matrix_power(&self, power: i64) -> PyResult<Self> {
        Ok(Self::from_tensor(
            self.inner.matrix_power(power).map_err(_convert_error)?,
        ))
    }

    /// A tensor whose diagonal is `self`, the inverse of `diagonal`.
    #[pyo3(signature = (offset=0, dim1=-2, dim2=-1))]
    pub fn diag_embed(&self, offset: isize, dim1: isize, dim2: isize) -> PyResult<Self> {
        Ok(Self::from_tensor(
            self.inner
                .diag_embed(offset, dim1, dim2)
                .map_err(_convert_error)?,
        ))
    }

    /// A matrix from a vector, or the diagonal of a matrix.
    #[pyo3(signature = (offset=0))]
    pub fn diag(&self, offset: isize) -> PyResult<Self> {
        Ok(Self::from_tensor(
            self.inner.diag(offset).map_err(_convert_error)?,
        ))
    }

    /// `(U, s, Vh)` for a matrix, with `A = U @ diag(s) @ Vh` and `s` descending. Columns of `U` and rows of `Vh` are determined up to sign.
    #[pyo3(signature = (full_matrices=true))]
    pub fn svd(&self, full_matrices: bool) -> PyResult<(Self, Self, Self)> {
        let (u, s, vt) = self.inner.svd(full_matrices).map_err(_convert_error)?;
        Ok((
            Self::from_tensor(u),
            Self::from_tensor(s),
            Self::from_tensor(vt),
        ))
    }

    /// The singular values of a matrix, descending.
    pub fn svdvals(&self) -> PyResult<Self> {
        Ok(Self::from_tensor(
            self.inner.svdvals().map_err(_convert_error)?,
        ))
    }

    /// `(w, V)` for a symmetric matrix, with `w` ascending and `A @ V == V @ diag(w)`. Only the lower triangle is read. Eigenvectors are determined up to sign.
    pub fn eigh(&self) -> PyResult<(Self, Self)> {
        let (values, vectors) = self.inner.eigh().map_err(_convert_error)?;
        Ok((Self::from_tensor(values), Self::from_tensor(vectors)))
    }

    /// The eigenvalues of a symmetric matrix, ascending. Cheaper than `eigh` when the vectors are not wanted.
    pub fn eigvalsh(&self) -> PyResult<Self> {
        Ok(Self::from_tensor(
            self.inner.eigvalsh().map_err(_convert_error)?,
        ))
    }

    pub fn bmm(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_tensor = tensor_from_py_value(&self.inner, other)?;
        let result = self.inner.bmm(&other_tensor).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn dot(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_tensor = tensor_from_py_value(&self.inner, other)?;
        let result = self.inner.dot(&other_tensor).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (diagonal=0))]
    pub fn triu(&self, diagonal: i64) -> PyResult<Self> {
        let result = self.inner.triu(diagonal).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (diagonal=0))]
    pub fn tril(&self, diagonal: i64) -> PyResult<Self> {
        let result = self.inner.tril(diagonal).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (offset=0, dim1=-2, dim2=-1))]
    pub fn diagonal(&self, offset: isize, dim1: isize, dim2: isize) -> PyResult<Self> {
        let result = self
            .inner
            .diagonal(offset, dim1, dim2)
            .map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (offset=0, dim1=-2, dim2=-1))]
    pub fn trace(&self, offset: isize, dim1: isize, dim2: isize) -> PyResult<Self> {
        let result = self
            .inner
            .trace(offset, dim1, dim2)
            .map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(name = "where")]
    pub fn where_method(&self, condition: &Bound<PyAny>, other: &Bound<PyAny>) -> PyResult<Self> {
        let device = self.inner.device();
        let condition_tensor = tensor_bool_from_py(condition, device)?;

        let other_input = tensor_from_py_value(&self.inner, other)?;
        let (input_cast, other_cast, _) =
            coerce_binary_operands(&self.inner, &other_input, BinaryOpKind::Add)
                .map_err(_convert_error)?;

        let input_tensor = match input_cast {
            Cow::Borrowed(_) => self.inner.clone(),
            Cow::Owned(tensor) => tensor,
        };
        let other_tensor = match other_cast {
            Cow::Borrowed(_) => other_input,
            Cow::Owned(tensor) => tensor,
        };

        let result = input_tensor
            .where_select(&condition_tensor, &other_tensor)
            .map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn masked_fill(&self, mask: &Bound<PyAny>, value: &Bound<PyAny>) -> PyResult<Self> {
        let device = self.inner.device();
        let mask_tensor = tensor_bool_from_py(mask, device)?;

        let mut tensor_value = tensor_from_py_value(&self.inner, value).map_err(|_| {
            PyTypeError::new_err("masked_fill value must be a Tensor or numeric scalar")
        })?;

        if tensor_value.device() != device {
            tensor_value = tensor_value.to(device).map_err(_convert_error)?;
        }

        let (input_cast, value_cast, _) =
            coerce_binary_operands(&self.inner, &tensor_value, BinaryOpKind::Add)
                .map_err(_convert_error)?;

        let input_tensor = match input_cast {
            Cow::Borrowed(_) => self.inner.clone(),
            Cow::Owned(tensor) => tensor,
        };
        let value_tensor = match value_cast {
            Cow::Borrowed(_) => tensor_value,
            Cow::Owned(tensor) => tensor,
        };

        let result = input_tensor
            .masked_fill(&mask_tensor, &value_tensor)
            .map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    #[pyo3(signature = (other, axis=None))]
    pub fn cross(&self, other: &Bound<PyAny>, axis: Option<i32>) -> PyResult<Self> {
        let py = other.py();

        let maybe_tensor = if let Ok(tensor) = other.extract::<PyTensor>() {
            Some(tensor)
        } else if let Ok(attr) = other.getattr(intern!(py, "_tensor")) {
            attr.extract::<PyTensor>().ok()
        } else {
            None
        };

        let other_tensor = if let Some(tensor) = maybe_tensor {
            tensor
        } else {
            let dtype = self.inner.dtype();
            let device = self.inner.device();
            let converted = convert_python_data_to_tensor(other, dtype, device, false)?;
            PyTensor::from_tensor(converted)
        };

        cross_impl(self, &other_tensor, axis)
    }

    pub fn maximum(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let (lhs, rhs) =
            prepare_binary_operands_from_py(&self.inner, other, false, BinaryOpKind::Maximum)?;
        let result = lhs.maximum(&rhs).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn minimum(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let (lhs, rhs) =
            prepare_binary_operands_from_py(&self.inner, other, false, BinaryOpKind::Minimum)?;
        let result = lhs.minimum(&rhs).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn logaddexp(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let (lhs, rhs) =
            prepare_binary_operands_from_py(&self.inner, other, false, BinaryOpKind::Add)?;
        let result = lhs.logaddexp(&rhs).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    pub fn _coerce_binary_operands(
        &self,
        other: &PyTensor,
        op: &str,
    ) -> PyResult<(PyTensor, PyTensor)> {
        let op_kind = match op {
            "__add__" | "add" | "logaddexp" => BinaryOpKind::Add,
            "__sub__" | "sub" => BinaryOpKind::Sub,
            "__mul__" | "mul" => BinaryOpKind::Mul,
            "__truediv__" | "div" => BinaryOpKind::Div,
            "__floordiv__" | "floor_divide" => BinaryOpKind::FloorDiv,
            "__mod__" | "remainder" => BinaryOpKind::Rem,
            "maximum" => BinaryOpKind::Maximum,
            "minimum" => BinaryOpKind::Minimum,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported binary operation for dtype coercion: {op}"
                )));
            }
        };

        let (lhs_cast, rhs_cast, _) =
            coerce_binary_operands(self.tensor(), other.tensor(), op_kind)
                .map_err(_convert_error)?;

        let lhs_tensor = match lhs_cast {
            Cow::Borrowed(_) => self.inner.clone(),
            Cow::Owned(tensor) => tensor,
        };
        let rhs_tensor = match rhs_cast {
            Cow::Borrowed(_) => other.inner.clone(),
            Cow::Owned(tensor) => tensor,
        };

        Ok((
            PyTensor::from_tensor(lhs_tensor),
            PyTensor::from_tensor(rhs_tensor),
        ))
    }
}

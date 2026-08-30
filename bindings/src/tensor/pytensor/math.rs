// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
#[pymethods]
impl PyTensor {
    // Mathematical functions
    /// Whether any element is NaN, stopping at the first one found.
    ///
    /// `isnan(x).any()` answers the same question but builds an N-element
    /// boolean tensor to do it, and reads every element even when the first is
    /// already NaN. Integer and boolean tensors are always False.
    pub fn has_nan(&self) -> bool {
        self.inner.has_nan()
    }

    /// Whether any element is positive or negative infinity, stopping at the
    /// first one found. See [`Self::has_nan`].
    pub fn has_inf(&self) -> bool {
        self.inner.has_inf()
    }

    /// Element-wise absolute value.
    pub fn abs(&self) -> PyResult<Self> {
        let result = self.inner.abs().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Element-wise square root. Negative inputs give NaN.
    pub fn sqrt(&self) -> PyResult<Self> {
        let result = self.inner.sqrt().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Element-wise `1 / sqrt(x)`, computed without the intermediate root.
    pub fn rsqrt(&self) -> PyResult<Self> {
        let result = self.inner.rsqrt().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Raise each element to `exponent`, which may be a scalar or a broadcastable tensor.
    pub fn pow(&self, exponent: &Bound<PyAny>) -> PyResult<Self> {
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

    /// Element-wise `e ** x`.
    pub fn exp(&self) -> PyResult<Self> {
        let result = self.inner.exp().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Element-wise natural logarithm. Zero gives `-inf`, negatives give NaN.
    pub fn log(&self) -> PyResult<Self> {
        let result = self.inner.log().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Element-wise base-2 logarithm.
    pub fn log2(&self) -> PyResult<Self> {
        let result = self.inner.log2().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Element-wise base-10 logarithm.
    pub fn log10(&self) -> PyResult<Self> {
        let result = self.inner.log10().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Element-wise error function.
    pub fn erf(&self) -> PyResult<Self> {
        let result = self.inner.erf().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Element-wise complementary error function, `1 - erf(x)`, accurate in the tails where that subtraction would cancel.
    pub fn erfc(&self) -> PyResult<Self> {
        let result = self.inner.erfc().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Element-wise inverse error function on `[-1, 1]`, infinite at the endpoints and NaN outside them.
    pub fn erfinv(&self) -> PyResult<Self> {
        let result = self.inner.erfinv().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Element-wise `2 ** x`, from the hardware's base-2 exponential rather than `exp(x * ln 2)`.
    pub fn exp2(&self) -> PyResult<Self> {
        let result = self.inner.exp2().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Element-wise `sin(pi * x) / (pi * x)`, taken as 1 at zero.
    pub fn sinc(&self) -> PyResult<Self> {
        let result = self.inner.sinc().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Element-wise `log |gamma(x)|`, which stays finite where `gamma` itself overflows.
    pub fn lgamma(&self) -> PyResult<Self> {
        let result = self.inner.lgamma().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Element-wise digamma, the derivative of `lgamma`.
    pub fn digamma(&self) -> PyResult<Self> {
        let result = self.inner.digamma().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Element-wise `log(x / (1 - x))`, the inverse of `sigmoid`. `eps` clamps the input into `[eps, 1 - eps]` first.
    #[pyo3(signature = (eps=None))]
    pub fn logit(&self, eps: Option<f64>) -> PyResult<Self> {
        let result = self.inner.logit(eps).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// The `order`-th derivative of `digamma`. Order 0 is `digamma` itself and order 1 is the trigamma function.
    pub fn polygamma(&self, order: i64) -> PyResult<Self> {
        let result = self.inner.polygamma(order).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Modified Bessel function of the first kind, order zero.
    pub fn i0(&self) -> PyResult<Self> {
        let result = self.inner.i0().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Modified Bessel function of the first kind, order one.
    pub fn i1(&self) -> PyResult<Self> {
        let result = self.inner.i1().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// `exp(-|x|) * i0(x)`, which stays under one where `i0` overflows.
    pub fn i0e(&self) -> PyResult<Self> {
        let result = self.inner.i0e().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// `exp(-|x|) * i1(x)`.
    pub fn i1e(&self) -> PyResult<Self> {
        let result = self.inner.i1e().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Element-wise `log(1 + x)`, accurate for small `x` where `log(1 + x)` would cancel.
    pub fn log1p(&self) -> PyResult<Self> {
        let result = self.inner.log1p().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Element-wise `exp(x) - 1`, accurate for small `x` where the subtraction would cancel.
    pub fn expm1(&self) -> PyResult<Self> {
        let result = self.inner.expm1().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Element-wise sine, taking radians.
    pub fn sin(&self) -> PyResult<Self> {
        let result = self.inner.sin().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Element-wise cosine, taking radians.
    pub fn cos(&self) -> PyResult<Self> {
        let result = self.inner.cos().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Element-wise tangent, taking radians.
    pub fn tan(&self) -> PyResult<Self> {
        let result = self.inner.tan().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Element-wise inverse sine, returning radians in `[-pi/2, pi/2]`. Inputs outside `[-1, 1]` give NaN.
    pub fn asin(&self) -> PyResult<Self> {
        let result = self.inner.asin().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Element-wise inverse cosine, returning radians in `[0, pi]`. Inputs outside `[-1, 1]` give NaN.
    pub fn acos(&self) -> PyResult<Self> {
        let result = self.inner.acos().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Element-wise inverse tangent, returning radians in `(-pi/2, pi/2)`.
    pub fn atan(&self) -> PyResult<Self> {
        let result = self.inner.atan().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Element-wise hyperbolic sine.
    pub fn sinh(&self) -> PyResult<Self> {
        let result = self.inner.sinh().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Element-wise hyperbolic cosine.
    pub fn cosh(&self) -> PyResult<Self> {
        let result = self.inner.cosh().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Element-wise inverse hyperbolic sine.
    pub fn asinh(&self) -> PyResult<Self> {
        let result = self.inner.asinh().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Element-wise inverse hyperbolic cosine. Inputs below 1 give NaN.
    pub fn acosh(&self) -> PyResult<Self> {
        let result = self.inner.acosh().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Element-wise inverse hyperbolic tangent. Inputs outside `(-1, 1)` give NaN or infinity.
    pub fn atanh(&self) -> PyResult<Self> {
        let result = self.inner.atanh().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// The angle of the point `(other, input)` from the positive x-axis, in `(-pi, pi]`, keeping the quadrant that `atan(input / other)` loses.
    pub fn atan2(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let (lhs, rhs) =
            prepare_binary_operands_from_py(&self.inner, other, false, BinaryOpKind::Div)?;
        let result = lhs.atan2(&rhs).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// `sqrt(input^2 + other^2)`, computed without forming either square, so it answers where the squares would overflow.
    pub fn hypot(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let (lhs, rhs) =
            prepare_binary_operands_from_py(&self.inner, other, false, BinaryOpKind::Div)?;
        let result = lhs.hypot(&rhs).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// The unit step of `input`: 0 below zero, 1 above it, and `other` at exactly zero.
    pub fn heaviside(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let (lhs, rhs) =
            prepare_binary_operands_from_py(&self.inner, other, false, BinaryOpKind::Div)?;
        let result = lhs.heaviside(&rhs).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// The next representable value after each element, in the direction of `other`.
    pub fn nextafter(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let (lhs, rhs) =
            prepare_binary_operands_from_py(&self.inner, other, false, BinaryOpKind::Div)?;
        let result = lhs.nextafter(&rhs).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// The magnitude of `input` with the sign of `other`, signed zeros included.
    pub fn copysign(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let (lhs, rhs) =
            prepare_binary_operands_from_py(&self.inner, other, false, BinaryOpKind::Div)?;
        let result = lhs.copysign(&rhs).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// `input * log(other)`, taken as `0` wherever `input` is zero rather than as the `0 * -inf` the plain product would give.
    pub fn xlogy(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let (lhs, rhs) =
            prepare_binary_operands_from_py(&self.inner, other, false, BinaryOpKind::Div)?;
        let result = lhs.xlogy(&rhs).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Replace NaN with `nan` and the infinities with `posinf`/`neginf`, defaulting to the dtype's finite extremes.
    #[pyo3(signature = (nan=0.0, posinf=None, neginf=None))]
    pub fn nan_to_num(&self, nan: f64, posinf: Option<f64>, neginf: Option<f64>) -> PyResult<Self> {
        let result = self
            .inner
            .nan_to_num(nan, posinf, neginf)
            .map_err(_convert_error)?;
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

    /// Element-wise `max(x, 0)`.
    pub fn relu(&self) -> PyResult<Self> {
        let result = self.inner.relu().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// `sigmoid` replaced by three straight lines: 0 below -3, 1 above 3, `x/6 + 1/2` between.
    pub fn hardsigmoid(&self) -> PyResult<Self> {
        let result = self.inner.hardsigmoid().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// `x * hardsigmoid(x)`: `silu` with the exponential replaced by three straight lines.
    pub fn hardswish(&self) -> PyResult<Self> {
        let result = self.inner.hardswish().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// `x - tanh(x)`, what `tanh` leaves behind.
    pub fn tanhshrink(&self) -> PyResult<Self> {
        let result = self.inner.tanhshrink().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// `x * tanh(softplus(x))`: smooth, non-monotonic, and keeps a small negative tail.
    pub fn mish(&self) -> PyResult<Self> {
        let result = self.inner.mish().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// `log(sigmoid(x))`, evaluated as `-softplus(-x)` so it stays exact where the direct form underflows.
    pub fn logsigmoid(&self) -> PyResult<Self> {
        let result = self.inner.logsigmoid().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// `hardtanh` on `[0, 6]`: the clipped ReLU that quantized networks use.
    pub fn relu6(&self) -> PyResult<Self> {
        let result = self.inner.relu6().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// `x` clamped to `[min_val, max_val]`, with no gradient outside them.
    #[pyo3(signature = (min_val=-1.0, max_val=1.0))]
    pub fn hardtanh(&self, min_val: f64, max_val: f64) -> PyResult<Self> {
        let result = self
            .inner
            .hardtanh(min_val, max_val)
            .map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// `x` where it exceeds `threshold`, `value` everywhere else.
    pub fn threshold(&self, threshold: f64, value: f64) -> PyResult<Self> {
        let result = self
            .inner
            .threshold(threshold, value)
            .map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Shrink each element towards zero by `lambd`, flattening `[-lambd, lambd]`.
    #[pyo3(signature = (lambd=0.5))]
    pub fn softshrink(&self, lambd: f64) -> PyResult<Self> {
        let result = self.inner.softshrink(lambd).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// `elu` rescaled so its slope is continuous at zero for every `alpha`.
    #[pyo3(signature = (alpha=1.0))]
    pub fn celu(&self, alpha: f64) -> PyResult<Self> {
        let result = self.inner.celu(alpha).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Normalize along `dim` so the values are positive, sum to 1, and favour the smallest element.
    #[pyo3(signature = (dim=None))]
    pub fn softmin(&self, dim: Option<isize>) -> PyResult<Self> {
        let resolved_dim = dim
            .map(|dim| engine::ops::normalize_dim(dim, self.inner.ndim()))
            .transpose()
            .map_err(_convert_error)?;
        let result = self.inner.softmin(resolved_dim).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Zero out values with magnitude below `lambd`, leaving the rest unchanged.
    #[pyo3(signature = (lambd=None))]
    pub fn hardshrink(&self, lambd: Option<f64>) -> PyResult<Self> {
        let result = self
            .inner
            .hardshrink(lambd.unwrap_or(0.5))
            .map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Normalize along `dim` so the values are positive and sum to 1. Shifted by the row maximum, so large inputs do not overflow.
    #[pyo3(signature = (dim=None))]
    pub fn softmax(&self, dim: Option<isize>) -> PyResult<Self> {
        let resolved_dim = dim
            .map(|dim| engine::ops::normalize_dim(dim, self.inner.ndim()))
            .transpose()
            .map_err(_convert_error)?;

        let result = self.inner.softmax(resolved_dim).map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Logarithm of `softmax`, computed directly rather than as `log(softmax(x))`, which underflows for confident rows.
    #[pyo3(signature = (dim=None))]
    pub fn log_softmax(&self, dim: Option<isize>) -> PyResult<Self> {
        let resolved_dim = dim
            .map(|dim| engine::ops::normalize_dim(dim, self.inner.ndim()))
            .transpose()
            .map_err(_convert_error)?;

        let result = self
            .inner
            .log_softmax(resolved_dim)
            .map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// `softmax` over the positions `mask` leaves alone: a true entry is excluded from the max and the sum -- not zeroed after normalizing -- and comes out 0.
    #[pyo3(signature = (mask, dim=None))]
    pub fn masked_softmax(&self, mask: &Bound<PyAny>, dim: Option<isize>) -> PyResult<Self> {
        let mask_tensor = tensor_from_py_value(&self.inner, mask)?;
        let resolved_dim = dim
            .map(|dim| engine::ops::normalize_dim(dim, self.inner.ndim()))
            .transpose()
            .map_err(_convert_error)?;

        let result = self
            .inner
            .masked_softmax(&mask_tensor, resolved_dim)
            .map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// `log_softmax` over the positions `mask` selects. See `masked_softmax`.
    #[pyo3(signature = (mask, dim=None))]
    pub fn masked_log_softmax(&self, mask: &Bound<PyAny>, dim: Option<isize>) -> PyResult<Self> {
        let mask_tensor = tensor_from_py_value(&self.inner, mask)?;
        let resolved_dim = dim
            .map(|dim| engine::ops::normalize_dim(dim, self.inner.ndim()))
            .transpose()
            .map_err(_convert_error)?;

        let result = self
            .inner
            .masked_log_softmax(&mask_tensor, resolved_dim)
            .map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Normalize over the trailing `normalized_shape` dimensions using that slice's own mean and variance, then scale and shift.
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

    /// Root-mean-square layer normalization (RMSNorm).
    pub fn rms_norm(
        &self,
        normalized_shape: Vec<usize>,
        weight: Option<&PyTensor>,
        eps: Option<f64>,
    ) -> PyResult<Self> {
        if normalized_shape.is_empty() {
            return Err(PyValueError::new_err(
                "rms_norm requires normalized_shape to contain at least one dimension",
            ));
        }
        let weight_inner = weight.map(|w| &w.inner);
        let result = self
            .inner
            .rms_norm(&normalized_shape, weight_inner, eps.unwrap_or(1e-6))
            .map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Gaussian Error Linear Unit, `x * Phi(x)`. Pass `approximate="tanh"` for the tanh approximation.
    #[pyo3(signature = (approximate=None))]
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

    /// Element-wise `1 / (1 + exp(-x))`, evaluated so that large-magnitude inputs saturate instead of producing NaN.
    pub fn sigmoid(&self) -> PyResult<Self> {
        let result = self.inner.sigmoid().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Element-wise `log(1 + exp(beta * x)) / beta`, falling back to the linear `x` above `threshold`.
    #[pyo3(signature = (beta=None, threshold=None))]
    pub fn softplus(&self, beta: Option<f64>, threshold: Option<f64>) -> PyResult<Self> {
        let result = self
            .inner
            .softplus(beta.unwrap_or(1.0), threshold.unwrap_or(20.0))
            .map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Exponential Linear Unit: `x` where positive, `alpha * (exp(x) - 1)` elsewhere.
    #[pyo3(signature = (alpha=None))]
    pub fn elu(&self, alpha: Option<f64>) -> PyResult<Self> {
        let result = self
            .inner
            .elu(alpha.unwrap_or(1.0))
            .map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// `x` where positive, `negative_slope * x` elsewhere.
    #[pyo3(signature = (negative_slope=None))]
    pub fn leaky_relu(&self, negative_slope: Option<f64>) -> PyResult<Self> {
        let result = self
            .inner
            .leaky_relu(negative_slope.unwrap_or(0.01))
            .map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Scaled Exponential Linear Unit, with the fixed constants that make it self-normalizing.
    pub fn selu(&self) -> PyResult<Self> {
        let result = self.inner.selu().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Sigmoid Linear Unit (Swish), `x * sigmoid(x)`.
    pub fn silu(&self) -> PyResult<Self> {
        let result = self.inner.silu().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Element-wise `x / (1 + abs(x))`.
    pub fn softsign(&self) -> PyResult<Self> {
        let result = self.inner.softsign().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }

    /// Element-wise hyperbolic tangent.
    pub fn tanh(&self) -> PyResult<Self> {
        let result = self.inner.tanh().map_err(_convert_error)?;
        Ok(Self::from_tensor(result))
    }
}

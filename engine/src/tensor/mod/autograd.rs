// Copyright (c) 2026 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

impl Tensor {
    /// Unary negation
    #[inline(always)]
    pub fn neg(&self) -> Result<Self> {
        use crate::operations::arithmetic::neg;
        neg(self)
    }

    /// Add two tensors element-wise
    #[inline(always)]
    pub fn add(&self, other: &Tensor) -> Result<Self> {
        use crate::operations::arithmetic::add;
        add(self, other)
    }

    /// Element-wise maximum
    #[inline(always)]
    pub fn maximum(&self, other: &Tensor) -> Result<Self> {
        use crate::operations::minmax::maximum;
        maximum(self, other)
    }

    /// Element-wise minimum
    #[inline(always)]
    pub fn minimum(&self, other: &Tensor) -> Result<Self> {
        use crate::operations::minmax::minimum;
        minimum(self, other)
    }

    /// Select elements from self or other based on a boolean condition tensor
    #[inline(always)]
    pub fn where_select(&self, condition: &Tensor, other: &Tensor) -> Result<Self> {
        use crate::operations::selection::where_op;
        where_op(condition, self, other)
    }

    /// Fill elements specified by `mask` with values from `value`.
    #[inline(always)]
    pub fn masked_fill(&self, mask: &Tensor, value: &Tensor) -> Result<Self> {
        crate::operations::selection::masked_fill(self, mask, value)
    }

    /// Fill elements specified by `mask` with a scalar.
    #[inline(always)]
    pub fn masked_fill_scalar(&self, mask: &Tensor, value: f64) -> Result<Self> {
        crate::operations::selection::masked_fill_scalar(self, mask, value)
    }

    /// Dot product between two 1D tensors
    #[inline(always)]
    pub fn dot(&self, other: &Tensor) -> Result<Self> {
        crate::operations::linalg::dot(self, other)
    }

    /// Matrix multiplication
    #[inline(always)]
    pub fn matmul(&self, other: &Tensor) -> Result<Self> {
        use crate::operations::linalg::matmul;
        matmul(self, other)
    }

    /// Batched matrix multiplication specialised for 3D tensors
    #[inline(always)]
    pub fn bmm(&self, other: &Tensor) -> Result<Self> {
        use crate::operations::linalg::bmm;
        bmm(self, other)
    }

    /// Upper triangular part of the tensor's last two dimensions
    #[inline(always)]
    pub fn triu(&self, diagonal: i64) -> Result<Self> {
        use crate::operations::linalg::triu;
        triu(self, diagonal)
    }

    /// Lower triangular part of the tensor's last two dimensions
    #[inline(always)]
    pub fn tril(&self, diagonal: i64) -> Result<Self> {
        use crate::operations::linalg::tril;
        tril(self, diagonal)
    }

    /// Extract a diagonal along two dimensions.
    #[inline(always)]
    pub fn diagonal(&self, offset: isize, dim1: isize, dim2: isize) -> Result<Self> {
        use crate::operations::linalg::diagonal;
        diagonal(self, offset, dim1, dim2)
    }

    /// Sum the diagonal elements along two dimensions.
    #[inline(always)]
    pub fn trace(&self, offset: isize, dim1: isize, dim2: isize) -> Result<Self> {
        use crate::operations::linalg::trace;
        trace(self, offset, dim1, dim2)
    }

    /// Sum reduction
    #[inline(always)]
    pub fn sum(&self, dim: Option<Vec<isize>>, keepdim: bool) -> Result<Self> {
        use crate::operations::reduction::sum;
        sum(self, dim, keepdim)
    }

    /// NaN-aware sum reduction
    #[inline(always)]
    pub fn nansum(&self, dim: Option<Vec<isize>>, keepdim: bool) -> Result<Self> {
        use crate::operations::reduction::nansum;
        nansum(self, dim, keepdim)
    }

    /// Log-sum-exp reduction
    #[inline(always)]
    pub fn logsumexp(&self, dim: Option<Vec<isize>>, keepdim: bool) -> Result<Self> {
        use crate::operations::reduction::logsumexp;
        logsumexp(self, dim, keepdim)
    }

    /// Product reduction
    #[inline(always)]
    pub fn prod(&self, dim: Option<Vec<isize>>, keepdim: bool) -> Result<Self> {
        use crate::operations::reduction::prod;
        prod(self, dim, keepdim)
    }

    /// Mean reduction
    #[inline(always)]
    pub fn mean(&self, dim: Option<Vec<isize>>, keepdim: bool) -> Result<Self> {
        use crate::operations::reduction::mean;
        mean(self, dim, keepdim)
    }

    /// NaN-aware mean reduction
    #[inline(always)]
    pub fn nanmean(&self, dim: Option<Vec<isize>>, keepdim: bool) -> Result<Self> {
        use crate::operations::reduction::nanmean;
        nanmean(self, dim, keepdim)
    }

    /// Logical all reduction
    #[inline(always)]
    pub fn all(&self, dim: Option<isize>, keepdim: bool) -> Result<Self> {
        use crate::operations::reduction::all;
        all(self, dim, keepdim)
    }

    /// Logical any reduction
    #[inline(always)]
    pub fn any(&self, dim: Option<isize>, keepdim: bool) -> Result<Self> {
        use crate::operations::reduction::any;
        any(self, dim, keepdim)
    }

    /// Cumulative sum along a dimension
    #[inline(always)]
    pub fn cumsum(&self, dim: isize) -> Result<Self> {
        use crate::operations::reduction::cumsum;
        cumsum(self, dim)
    }

    /// Cumulative product along a dimension
    #[inline(always)]
    pub fn cumprod(&self, dim: isize) -> Result<Self> {
        use crate::operations::reduction::cumprod;
        cumprod(self, dim)
    }

    /// Maximum value
    #[inline(always)]
    pub fn max(&self, dim: Option<isize>, keepdim: bool) -> Result<Self> {
        use crate::operations::reduction::max;
        max(self, dim, keepdim)
    }

    /// NaN-aware maximum value
    #[inline(always)]
    pub fn nanmax(&self, dim: Option<isize>, keepdim: bool) -> Result<Self> {
        use crate::operations::reduction::nanmax;
        nanmax(self, dim, keepdim)
    }

    /// Minimum value
    #[inline(always)]
    pub fn min(&self, dim: Option<isize>, keepdim: bool) -> Result<Self> {
        use crate::operations::reduction::min;
        min(self, dim, keepdim)
    }

    /// NaN-aware minimum value
    #[inline(always)]
    pub fn nanmin(&self, dim: Option<isize>, keepdim: bool) -> Result<Self> {
        use crate::operations::reduction::nanmin;
        nanmin(self, dim, keepdim)
    }

    /// Argument of maximum value
    #[inline(always)]
    pub fn argmax(&self, dim: Option<isize>, keepdim: bool) -> Result<Self> {
        use crate::operations::reduction::argmax;
        argmax(self, dim, keepdim)
    }

    /// Argument of minimum value
    #[inline(always)]
    pub fn argmin(&self, dim: Option<isize>, keepdim: bool) -> Result<Self> {
        use crate::operations::reduction::argmin;
        argmin(self, dim, keepdim)
    }

    /// Maximum values and their indices along a dimension
    #[inline(always)]
    pub fn max_with_indices(&self, dim: isize, keepdim: bool) -> Result<(Self, Self)> {
        use crate::operations::reduction::max_with_indices;
        max_with_indices(self, dim, keepdim)
    }

    /// NaN-aware maximum values and their indices along a dimension
    #[inline(always)]
    pub fn nanmax_with_indices(&self, dim: isize, keepdim: bool) -> Result<(Self, Self)> {
        use crate::operations::reduction::nanmax_with_indices;
        nanmax_with_indices(self, dim, keepdim)
    }

    /// Minimum values and their indices along a dimension
    #[inline(always)]
    pub fn min_with_indices(&self, dim: isize, keepdim: bool) -> Result<(Self, Self)> {
        use crate::operations::reduction::min_with_indices;
        min_with_indices(self, dim, keepdim)
    }

    /// NaN-aware minimum values and their indices along a dimension
    #[inline(always)]
    pub fn nanmin_with_indices(&self, dim: isize, keepdim: bool) -> Result<(Self, Self)> {
        use crate::operations::reduction::nanmin_with_indices;
        nanmin_with_indices(self, dim, keepdim)
    }

    /// Median value (optionally along a dimension)
    #[inline(always)]
    pub fn median(&self, dim: Option<isize>, keepdim: bool) -> Result<(Self, Option<Self>)> {
        use crate::operations::reduction::median;
        median(self, dim, keepdim)
    }

    /// Quantile reduction with configurable interpolation
    #[inline(always)]
    pub fn quantile(
        &self,
        q: f64,
        dim: Option<isize>,
        keepdim: bool,
        interpolation: QuantileInterpolation,
    ) -> Result<Self> {
        use crate::operations::reduction::quantile;
        quantile(self, q, dim, keepdim, interpolation)
    }

    /// Quantile reduction that ignores NaN values
    #[inline(always)]
    pub fn nanquantile(
        &self,
        q: f64,
        dim: Option<isize>,
        keepdim: bool,
        interpolation: QuantileInterpolation,
    ) -> Result<Self> {
        use crate::operations::reduction::nanquantile;
        nanquantile(self, q, dim, keepdim, interpolation)
    }

    /// Batched quantile reduction for multiple probabilities at once
    #[inline(always)]
    pub fn quantiles(
        &self,
        qs: &[f64],
        dim: Option<isize>,
        keepdim: bool,
        interpolation: QuantileInterpolation,
    ) -> Result<Self> {
        use crate::operations::reduction::quantiles;
        quantiles(self, qs, dim, keepdim, interpolation)
    }

    /// Batched quantile reduction that ignores NaN values
    #[inline(always)]
    pub fn nanquantiles(
        &self,
        qs: &[f64],
        dim: Option<isize>,
        keepdim: bool,
        interpolation: QuantileInterpolation,
    ) -> Result<Self> {
        use crate::operations::reduction::nanquantiles;
        nanquantiles(self, qs, dim, keepdim, interpolation)
    }

    /// Top-k values and indices along a dimension
    #[inline(always)]
    pub fn topk(
        &self,
        k: usize,
        dim: Option<isize>,
        largest: bool,
        sorted: bool,
    ) -> Result<(Self, Self)> {
        use crate::operations::reduction::topk;
        topk(self, k, dim, largest, sorted)
    }

    /// Sort tensor values along a dimension
    #[inline(always)]
    pub fn sort(&self, dim: Option<isize>, descending: bool, stable: bool) -> Result<(Self, Self)> {
        use crate::operations::reduction::sort;
        sort(self, dim, descending, stable)
    }

    /// Indices that would sort the tensor along a dimension
    #[inline(always)]
    pub fn argsort(&self, dim: Option<isize>, descending: bool, stable: bool) -> Result<Self> {
        use crate::operations::reduction::argsort;
        argsort(self, dim, descending, stable)
    }

    /// Element-wise equality comparison
    #[inline(always)]
    pub fn eq(&self, other: &Tensor) -> Result<Self> {
        use crate::operations::comparison::eq;
        eq(self, other)
    }

    /// Element-wise inequality comparison
    pub fn ne(&self, other: &Tensor) -> Result<Self> {
        use crate::operations::comparison::ne;
        ne(self, other)
    }

    /// Element-wise less-than comparison
    #[inline(always)]
    pub fn lt(&self, other: &Tensor) -> Result<Self> {
        use crate::operations::comparison::lt;
        lt(self, other)
    }

    /// Element-wise less-than-or-equal comparison
    #[inline(always)]
    pub fn le(&self, other: &Tensor) -> Result<Self> {
        use crate::operations::comparison::le;
        le(self, other)
    }

    /// Element-wise greater-than comparison
    #[inline(always)]
    pub fn gt(&self, other: &Tensor) -> Result<Self> {
        use crate::operations::comparison::gt;
        gt(self, other)
    }

    /// Element-wise greater-than-or-equal comparison
    #[inline(always)]
    pub fn ge(&self, other: &Tensor) -> Result<Self> {
        use crate::operations::comparison::ge;
        ge(self, other)
    }

    /// Standard deviation
    #[inline(always)]
    pub fn std(&self, dim: Option<isize>, keepdim: bool, unbiased: bool) -> Result<Self> {
        use crate::operations::reduction::std;
        std(self, dim, keepdim, unbiased)
    }

    /// Variance
    #[inline(always)]
    pub fn var(&self, dim: Option<isize>, keepdim: bool, unbiased: bool) -> Result<Self> {
        use crate::operations::reduction::var;
        var(self, dim, keepdim, unbiased)
    }

    /// Exponential function
    #[inline(always)]
    pub fn exp(&self) -> Result<Self> {
        use crate::operations::activation::exp;
        exp(self)
    }

    /// Natural logarithm
    #[inline(always)]
    pub fn log(&self) -> Result<Self> {
        use crate::operations::activation::log;
        log(self)
    }

    /// log1p (log(1 + x))
    #[inline(always)]
    pub fn log1p(&self) -> Result<Self> {
        use crate::operations::activation::log1p;
        log1p(self)
    }

    /// expm1 (exp(x) - 1)
    #[inline(always)]
    pub fn expm1(&self) -> Result<Self> {
        use crate::operations::activation::expm1;
        expm1(self)
    }

    /// Sine function
    #[inline(always)]
    pub fn sin(&self) -> Result<Self> {
        use crate::operations::activation::sin;
        sin(self)
    }

    /// Cosine function
    #[inline(always)]
    pub fn cos(&self) -> Result<Self> {
        use crate::operations::activation::cos;
        cos(self)
    }

    /// Tangent function
    #[inline(always)]
    pub fn tan(&self) -> Result<Self> {
        use crate::operations::activation::tan;
        tan(self)
    }

    /// Inverse sine function
    #[inline(always)]
    pub fn asin(&self) -> Result<Self> {
        use crate::operations::activation::asin;
        asin(self)
    }

    /// Inverse cosine function
    #[inline(always)]
    pub fn acos(&self) -> Result<Self> {
        use crate::operations::activation::acos;
        acos(self)
    }

    /// Inverse tangent function
    #[inline(always)]
    pub fn atan(&self) -> Result<Self> {
        use crate::operations::activation::atan;
        atan(self)
    }

    /// Hyperbolic sine
    #[inline(always)]
    pub fn sinh(&self) -> Result<Self> {
        use crate::operations::activation::sinh;
        sinh(self)
    }

    /// Hyperbolic cosine
    #[inline(always)]
    pub fn cosh(&self) -> Result<Self> {
        use crate::operations::activation::cosh;
        cosh(self)
    }

    /// Inverse hyperbolic sine
    #[inline(always)]
    pub fn asinh(&self) -> Result<Self> {
        use crate::operations::activation::asinh;
        asinh(self)
    }

    /// Inverse hyperbolic cosine
    #[inline(always)]
    pub fn acosh(&self) -> Result<Self> {
        use crate::operations::activation::acosh;
        acosh(self)
    }

    /// Inverse hyperbolic tangent
    #[inline(always)]
    pub fn atanh(&self) -> Result<Self> {
        use crate::operations::activation::atanh;
        atanh(self)
    }

    /// Hyperbolic tangent
    #[inline(always)]
    pub fn tanh(&self) -> Result<Self> {
        use crate::operations::activation::tanh;
        tanh(self)
    }

    /// Sigmoid activation
    #[inline(always)]
    pub fn sigmoid(&self) -> Result<Self> {
        use crate::operations::activation::sigmoid;
        sigmoid(self)
    }

    /// Softplus activation
    #[inline(always)]
    pub fn softplus(&self, beta: f64, threshold: f64) -> Result<Self> {
        use crate::operations::activation::softplus;
        softplus(self, beta, threshold)
    }

    /// GELU activation
    #[inline(always)]
    pub fn gelu(&self, approximate: bool) -> Result<Self> {
        use crate::operations::activation::gelu;
        gelu(self, approximate)
    }

    /// ELU activation
    #[inline(always)]
    pub fn elu(&self, alpha: f64) -> Result<Self> {
        use crate::operations::activation::elu;
        elu(self, alpha)
    }

    /// SELU activation
    #[inline(always)]
    pub fn selu(&self) -> Result<Self> {
        use crate::operations::activation::selu;
        selu(self)
    }

    /// SiLU activation
    #[inline(always)]
    pub fn silu(&self) -> Result<Self> {
        use crate::operations::activation::silu;
        silu(self)
    }

    /// Softsign activation
    #[inline(always)]
    pub fn softsign(&self) -> Result<Self> {
        use crate::operations::activation::softsign;
        softsign(self)
    }

    /// ReLU activation
    #[inline(always)]
    pub fn relu(&self) -> Result<Self> {
        use crate::operations::activation::relu;
        relu(self)
    }

    /// Hardshrink activation
    #[inline(always)]
    pub fn hardshrink(&self, lambd: f64) -> Result<Self> {
        use crate::operations::activation::hardshrink;
        hardshrink(self, lambd)
    }

    /// Softmax activation
    #[inline(always)]
    pub fn softmax(&self, dim: Option<usize>) -> Result<Self> {
        use crate::operations::activation::softmax;
        softmax(self, dim)
    }

    /// Log-Softmax activation
    #[inline(always)]
    pub fn log_softmax(&self, dim: Option<usize>) -> Result<Self> {
        use crate::operations::activation::log_softmax;
        log_softmax(self, dim)
    }

    /// Masked Softmax activation
    #[inline(always)]
    pub fn masked_softmax(&self, mask: &Tensor, dim: Option<usize>) -> Result<Self> {
        use crate::operations::activation::masked_softmax;
        masked_softmax(self, mask, dim)
    }

    /// Masked Log-Softmax activation
    #[inline(always)]
    pub fn masked_log_softmax(&self, mask: &Tensor, dim: Option<usize>) -> Result<Self> {
        use crate::operations::activation::masked_log_softmax;
        masked_log_softmax(self, mask, dim)
    }
}

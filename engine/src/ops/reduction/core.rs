// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
use crate::autograd::with_grad_fn;
use crate::ops::order::{float_key32, float_key64};

use crate::{
    autograd::{MedianBackward, QuantileBackward},
    error::{MinitensorError, Result},
    ops::map::unary_map,
    tensor::{DataType, Shape, Tensor, TensorData},
};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::sync::Arc;

/// Interpolation modes supported by the quantile reduction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantileInterpolation {
    Linear,
    Lower,
    Higher,
    Midpoint,
    Nearest,
}

impl QuantileInterpolation {
    #[inline(always)]
    pub(crate) fn interpolate(self, lower: f64, upper: f64, weight: f64) -> f64 {
        match self {
            QuantileInterpolation::Linear => lower + (upper - lower) * weight,
            QuantileInterpolation::Lower => lower,
            QuantileInterpolation::Higher => upper,
            QuantileInterpolation::Midpoint => 0.5 * (lower + upper),
            QuantileInterpolation::Nearest => {
                // Index-aware callers should route through `nearest_index_with_tie_even`
                // for tie-to-even behavior. This fallback remains
                // deterministic for non-indexed interpolation use-cases.
                if weight < 0.5 { lower } else { upper }
            }
        }
    }
}

/// Reduce several axes at once, by bringing them together and reducing the one
/// axis they become.
///
/// `reduce` takes a tensor and one axis and reduces it away. Everything around
/// that call is shape work on `permute` and `reshape`, both of which carry
/// gradients, so a reduction written for one axis reaches several without a
/// second backward being written for it -- and the gradient it gets is the
/// one-axis gradient over the whole group, which is what reducing those axes
/// together means.
///
/// The axes are gathered rather than reduced one after another because those
/// are two different reductions. Folding `amax` axis by axis splits a tie
/// within each axis and then again between the partial winners, so
/// `[[3, 3], [3, 1]]` reduced over both axes sends a quarter, a quarter and a
/// half to three elements that are equally the maximum. Gathering them first
/// gives each a third, which is what an even split over the ties means and
/// what PyTorch's `amax` gives for the same tensor.
///
/// Nothing is copied when the axes are already at the end in order, which is
/// the common case (`dim=(-2, -1)`): the permutation is then the identity and
/// the reshape is a view.
pub(crate) fn reduce_gathered_dims<F>(
    tensor: &Tensor,
    dims: &[usize],
    keepdim: bool,
    reduce: F,
) -> Result<Tensor>
where
    F: FnOnce(&Tensor, usize) -> Result<Tensor>,
{
    let shape = tensor.shape().dims().to_vec();
    let kept: Vec<usize> = (0..tensor.ndim()).filter(|d| !dims.contains(d)).collect();

    let order: Vec<isize> = kept
        .iter()
        .chain(dims.iter())
        .map(|&d| d as isize)
        .collect();
    let moved = crate::ops::shape_ops::permute(tensor, order)?;

    let mut gathered: Vec<usize> = kept.iter().map(|&d| shape[d]).collect();
    gathered.push(dims.iter().map(|&d| shape[d]).product());
    let grouped = crate::ops::shape_ops::reshape(&moved, Shape::new(gathered))?;

    let reduced = reduce(&grouped, kept.len())?;
    if !keepdim {
        // `kept` is ascending, so the surviving axes are already in the order
        // the input had them in.
        return Ok(reduced);
    }
    let mut with_ones = shape;
    for &d in dims {
        with_ones[d] = 1;
    }
    // A reduction may report more than one value per slice -- `quantiles`
    // puts a `q` axis in front -- and those axes are the reduction's own, not
    // the input's, so they are carried across rather than rebuilt.
    let reported = reduced.ndim() - kept.len();
    let mut target: Vec<usize> = reduced.shape().dims()[..reported].to_vec();
    target.extend(with_ones);
    crate::ops::shape_ops::reshape(&reduced, Shape::new(target))
}

/// Let a reduction written for one axis take a list of them.
///
/// `axis_op` is the one-axis reduction, called with the `dim` it would have
/// been called with before. A list of several axes is answered by gathering
/// them (see [`reduce_gathered_dims`]) and reducing the one axis they become,
/// so the reduction itself does not learn about lists and its gradient
/// arrives through the shape ops that did the gathering.
///
/// A rank-zero tensor is passed straight through, because each of these ops
/// has its own reading of `dim` there -- `quantile` takes 0 or -1 on a scalar
/// and `amax` refuses both -- and normalising first would answer for them.
pub(crate) fn reduce_over_dims<F>(
    tensor: &Tensor,
    dim: Option<Vec<isize>>,
    keepdim: bool,
    axis_op: F,
) -> Result<Tensor>
where
    F: Fn(&Tensor, Option<isize>, bool) -> Result<Tensor>,
{
    if tensor.ndim() == 0 {
        return match dim.as_deref() {
            None => axis_op(tensor, None, keepdim),
            Some([d]) => axis_op(tensor, Some(*d), keepdim),
            Some(_) => Err(MinitensorError::invalid_argument(
                "a 0-dimensional tensor has no axes to reduce over",
            )),
        };
    }
    match normalize_reduction_dims(dim, tensor.ndim())?.as_deref() {
        None => axis_op(tensor, None, keepdim),
        Some([d]) => axis_op(tensor, Some(*d as isize), keepdim),
        Some(dims) => reduce_gathered_dims(tensor, dims, keepdim, |grouped, axis| {
            axis_op(grouped, Some(axis as isize), false)
        }),
    }
}

/// Resolve a reduction's `dim` list: negatives counted from the end, sorted,
/// duplicates dropped.
///
/// Sorting matters beyond tidiness -- callers that remove axes from the output
/// shape walk the list in reverse, which only removes the right axes if it is
/// ascending. Deduplicating matters for the same reason: `sum(dim=[0, 0])`
/// would otherwise drop two axes.
pub(crate) fn normalize_reduction_dims(
    dims: Option<Vec<isize>>,
    ndim: usize,
) -> Result<Option<Vec<usize>>> {
    Ok(match dims {
        Some(dims) => {
            let mut normalized = Vec::with_capacity(dims.len());
            for d in dims {
                normalized.push(normalize_dim(d, ndim)?);
            }
            normalized.sort_unstable();
            normalized.dedup();
            // Naming every axis is the same request as naming none of them,
            // and the two took different routes: the whole-tensor reductions
            // are a single pass, where reducing one axis at a time is a pass
            // per axis with a full-size intermediate between them. `sum([0])`
            // of a four-million-element vector cost 0.86ms against 0.33 for
            // `sum()` of the same tensor, and `norm`, which always names its
            // axes, paid it every time.
            //
            // A rank-zero tensor is left alone: it has no axes, so an empty
            // list there means "reduce nothing" rather than "reduce all".
            if ndim > 0 && normalized.len() == ndim {
                None
            } else {
                Some(normalized)
            }
        }
        None => None,
    })
}

pub(crate) fn non_nan_mask(tensor: &Tensor) -> Result<Tensor> {
    // Plain element map, so it goes through the shared `unary_map` pipeline
    // rather than a hand-written `par_iter_mut().zip(..)` — which handed rayon
    // one work item per element for a single `is_nan` test.
    let mask: Vec<bool> = match tensor.dtype() {
        DataType::Float32 => {
            let data = tensor
                .data()
                .as_f32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;
            unary_map(data, |v: f32| !v.is_nan())
        }
        DataType::Float64 => {
            let data = tensor
                .data()
                .as_f64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;
            unary_map(data, |v: f64| !v.is_nan())
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "nan reductions are only supported for floating point tensors",
            ));
        }
    };

    Ok(Tensor::new(
        Arc::new(TensorData::from_vec_bool(mask, tensor.device())),
        tensor.shape().clone(),
        DataType::Bool,
        tensor.device(),
        false,
    ))
}

/// Ascending and descending order over `(position, value)` pairs, for the
/// selections that carry an index alongside each value.
///
/// The comparison is by [`float_key32`] rather than by the three-way float
/// comparison it reads like: the key's integer order *is* the float order,
/// with every NaN folded above the numbers and the two zeros folded together,
/// so the branches and the NaN test come out of the inner loop and go into
/// three arithmetic instructions. `select_nth_unstable_by` over two million
/// float32 costs 31ms with the branching form and 10 with this one.
///
/// The tie-break by position is still there and still ascending in both
/// directions, so equal values keep their input order and the descending
/// answer is the mirror of the ascending one rather than its reverse.
pub(crate) fn cmp_f32_desc(a: &(usize, f32), b: &(usize, f32)) -> Ordering {
    // Descending puts NaN first, mirroring where ascending puts it last, and
    // ties still fall back to the *ascending* index: the answer is the mirror
    // of the ascending one, not its reverse.
    float_key32(b.1)
        .cmp(&float_key32(a.1))
        .then_with(|| a.0.cmp(&b.0))
}

pub(crate) fn cmp_f32_asc(a: &(usize, f32), b: &(usize, f32)) -> Ordering {
    float_key32(a.1)
        .cmp(&float_key32(b.1))
        .then_with(|| a.0.cmp(&b.0))
}

pub(crate) fn cmp_f64_desc(a: &(usize, f64), b: &(usize, f64)) -> Ordering {
    float_key64(b.1)
        .cmp(&float_key64(a.1))
        .then_with(|| a.0.cmp(&b.0))
}

pub(crate) fn cmp_f64_asc(a: &(usize, f64), b: &(usize, f64)) -> Ordering {
    float_key64(a.1)
        .cmp(&float_key64(b.1))
        .then_with(|| a.0.cmp(&b.0))
}

/// The same order, for a scan that asks about it once per element.
///
/// A bounded top-`k` compares every element against a heap root that is
/// already better than nearly all of them, so the answer is the same almost
/// every time and the branch predicts. One float comparison costs less there
/// than building two keys, and the key form measured 11.3ms against 6.4 on the
/// top hundred of two million float32 -- the mirror of the selection result,
/// where the branches never predict and the keys win threefold.
///
/// Not a second definition of the order: two floats that are neither equal nor
/// NaN compare the same way as their keys do, by construction, and every other
/// case falls through to [`cmp_f32_asc`], which *is* the definition.
pub(crate) fn scan_cmp_f32_asc(a: &(usize, f32), b: &(usize, f32)) -> Ordering {
    if a.1 < b.1 {
        Ordering::Less
    } else if a.1 > b.1 {
        Ordering::Greater
    } else {
        cmp_f32_asc(a, b)
    }
}

/// [`scan_cmp_f32_asc`] the other way round.
pub(crate) fn scan_cmp_f32_desc(a: &(usize, f32), b: &(usize, f32)) -> Ordering {
    if a.1 > b.1 {
        Ordering::Less
    } else if a.1 < b.1 {
        Ordering::Greater
    } else {
        cmp_f32_desc(a, b)
    }
}

/// [`scan_cmp_f32_asc`] for double precision.
pub(crate) fn scan_cmp_f64_asc(a: &(usize, f64), b: &(usize, f64)) -> Ordering {
    if a.1 < b.1 {
        Ordering::Less
    } else if a.1 > b.1 {
        Ordering::Greater
    } else {
        cmp_f64_asc(a, b)
    }
}

/// [`scan_cmp_f32_desc`] for double precision.
pub(crate) fn scan_cmp_f64_desc(a: &(usize, f64), b: &(usize, f64)) -> Ordering {
    if a.1 > b.1 {
        Ordering::Less
    } else if a.1 < b.1 {
        Ordering::Greater
    } else {
        cmp_f64_desc(a, b)
    }
}

pub(crate) fn cmp_i32_desc(a: &(usize, i32), b: &(usize, i32)) -> Ordering {
    match b.1.cmp(&a.1) {
        Ordering::Equal => a.0.cmp(&b.0),
        order => order,
    }
}

pub(crate) fn cmp_i32_asc(a: &(usize, i32), b: &(usize, i32)) -> Ordering {
    match a.1.cmp(&b.1) {
        Ordering::Equal => a.0.cmp(&b.0),
        order => order,
    }
}

pub(crate) fn cmp_i64_desc(a: &(usize, i64), b: &(usize, i64)) -> Ordering {
    match b.1.cmp(&a.1) {
        Ordering::Equal => a.0.cmp(&b.0),
        order => order,
    }
}

pub(crate) fn cmp_i64_asc(a: &(usize, i64), b: &(usize, i64)) -> Ordering {
    match a.1.cmp(&b.1) {
        Ordering::Equal => a.0.cmp(&b.0),
        order => order,
    }
}

pub(crate) fn cmp_bool_desc(a: &(usize, bool), b: &(usize, bool)) -> Ordering {
    match (a.1, b.1) {
        (true, true) | (false, false) => a.0.cmp(&b.0),
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
    }
}

pub(crate) fn cmp_bool_asc(a: &(usize, bool), b: &(usize, bool)) -> Ordering {
    match (a.1, b.1) {
        (true, true) | (false, false) => a.0.cmp(&b.0),
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
    }
}

/// Reject an empty reduction for an operation that has no identity element.
///
/// `sum` and `prod` can answer for an empty input because they have identities
/// (0 and 1). `max`, `median` and friends cannot, and a sentinel is worse than
/// no answer: the int64 `max` of nothing is `i64::MIN`, which is also a
/// perfectly ordinary value, so the caller cannot tell the two apart. Raising
/// is the only honest answer.
///
/// `op` names the operation so the message points at what the caller actually
/// called -- this used to be hardcoded to `median`, which is how `nanquantile`
/// came to report a `median()` error.
pub(crate) fn ensure_non_empty(numel: usize, op: &str) -> Result<()> {
    if numel == 0 {
        Err(MinitensorError::invalid_argument(format!(
            "{op}() does not support empty tensors"
        )))
    } else {
        Ok(())
    }
}

/// Median of the tensor, optionally along one dimension.
///
/// For an even number of elements this returns the **lower** of the two middle
/// values; it does *not* average them. That is what lets the reduction also
/// report the index of the element it selected (returned as the second tuple
/// element when `dim` is given), since an averaged midpoint belongs to no
/// element. Use `quantile(0.5)` for the interpolated definition.
///
/// A `NaN` anywhere in a reduced slice makes that slice's median `NaN`, and
/// the index then names the first `NaN` in the slice. The index names the
/// element returned in every case -- gathering along `dim` gives the values
/// back -- which is the property the whole convention exists to provide.
pub fn median(
    tensor: &Tensor,
    dim: Option<isize>,
    keepdim: bool,
) -> Result<(Tensor, Option<Tensor>)> {
    ensure_non_empty(tensor.numel(), "median")?;

    if tensor.ndim() == 0 {
        return Ok((tensor.clone(), None));
    }

    let (values, indices, norm_dim) = match dim {
        None => {
            let (values, indices) = median_all(tensor)?;
            (values, indices, None)
        }
        Some(dim_value) => {
            let axis = if tensor.ndim() == 0 {
                if dim_value == 0 || dim_value == -1 {
                    0
                } else {
                    return Err(MinitensorError::dim_out_of_range(dim_value, 1));
                }
            } else {
                normalize_dim(dim_value, tensor.ndim())?
            };
            let (values, indices) = median_along_dim(tensor, axis, keepdim)?;
            (values, Some(indices), Some(axis))
        }
    };
    let values = attach_median_grad(values, tensor, norm_dim, keepdim, false, indices.as_ref())?;
    Ok((values, indices))
}

/// Attach a [`MedianBackward`] gradient to a median value reduction.
///
/// `selected` is the index tensor when the caller reported one, which makes the
/// gradient go to that element rather than being split among everything equal
/// to the median. See [`MedianBackward`].
fn attach_median_grad(
    values: Tensor,
    input: &Tensor,
    dim: Option<usize>,
    keepdim: bool,
    nan_aware: bool,
    selected: Option<&Tensor>,
) -> Result<Tensor> {
    if !input.requires_grad() || !input.dtype().is_float() {
        return Ok(values);
    }
    let grad_fn = Arc::new(MedianBackward {
        input_id: input.id(),
        input: input.detach(),
        dim,
        keepdim,
        nan_aware,
        selected: selected.map(|t| t.detach()),
    });
    with_grad_fn(values, grad_fn)
}

/// The `q`-th quantile, over one axis, several, or the whole tensor.
///
/// Several axes are gathered into one and reduced together, which is the only
/// reading a quantile has over more than one axis -- the q-th value of the
/// group, not a quantile of quantiles. NumPy's `axis` tuple means the same.
pub fn quantile(
    tensor: &Tensor,
    q: f64,
    dim: Option<Vec<isize>>,
    keepdim: bool,
    interpolation: QuantileInterpolation,
) -> Result<Tensor> {
    // An integer argument widens rather than being refused, the rule `mean`
    // already follows: a quantile of integers is an interpolated real number,
    // and NumPy answers one. See `ops::util::widen_integer_input`.
    if let Some(widened) = crate::ops::util::widen_integer_input(tensor)? {
        return quantile(&widened, q, dim, keepdim, interpolation);
    }
    reduce_over_dims(tensor, dim, keepdim, |t, axis, keep| {
        quantile_axis(t, q, axis, keep, interpolation)
    })
}

/// Several quantiles in one pass. See [`quantile`].
pub fn quantiles(
    tensor: &Tensor,
    qs: &[f64],
    dim: Option<Vec<isize>>,
    keepdim: bool,
    interpolation: QuantileInterpolation,
) -> Result<Tensor> {
    // An integer argument widens rather than being refused, the rule `mean`
    // already follows: a quantile of integers is an interpolated real number,
    // and NumPy answers one. See `ops::util::widen_integer_input`.
    if let Some(widened) = crate::ops::util::widen_integer_input(tensor)? {
        return quantiles(&widened, qs, dim, keepdim, interpolation);
    }
    if let Some(stacked) = quantiles_with_grad(tensor, qs, &dim, keepdim, interpolation, false)? {
        return Ok(stacked);
    }
    reduce_over_dims(tensor, dim, keepdim, |t, axis, keep| {
        quantiles_axis(t, qs, axis, keep, interpolation)
    })
}

/// The batched quantile kernels sort once and read every probability out of
/// the one ordering, which is the whole reason they exist -- and none of them
/// reports a gradient. The result came back with `requires_grad` set and no
/// `grad_fn`, so it looked tracked, `backward()` walked past it as if it were
/// a leaf, and `quantile(x, [0.1, 0.9]).sum().backward()` left `x.grad` as
/// `None`: no error, no gradient, which is how a multi-quantile loss trains
/// nothing and says nothing.
///
/// When a gradient is wanted, each probability therefore goes through the
/// single-probability reduction, which has one, and the results are stacked
/// along the axis the batched kernel would have put them on. That is `k`
/// sorts instead of one, paid only by a caller who is going to back-propagate
/// through it; inference keeps the single pass.
fn quantiles_with_grad(
    tensor: &Tensor,
    qs: &[f64],
    dim: &Option<Vec<isize>>,
    keepdim: bool,
    interpolation: QuantileInterpolation,
    nan_aware: bool,
) -> Result<Option<Tensor>> {
    if qs.is_empty()
        || !tensor.dtype().is_float()
        || !tensor.requires_grad()
        || !crate::autograd::is_grad_enabled()
    {
        return Ok(None);
    }

    let mut parts = Vec::with_capacity(qs.len());
    for &q in qs {
        let part = if nan_aware {
            nanquantile(tensor, q, dim.clone(), keepdim, interpolation)?
        } else {
            quantile(tensor, q, dim.clone(), keepdim, interpolation)?
        };
        parts.push(crate::ops::shape_ops::unsqueeze(&part, 0)?);
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Ok(Some(crate::ops::shape_ops::concatenate(&refs, 0)?))
}

/// Like [`quantile`], ignoring NaN.
pub fn nanquantile(
    tensor: &Tensor,
    q: f64,
    dim: Option<Vec<isize>>,
    keepdim: bool,
    interpolation: QuantileInterpolation,
) -> Result<Tensor> {
    // An integer argument widens rather than being refused, the rule `mean`
    // already follows: a quantile of integers is an interpolated real number,
    // and NumPy answers one. See `ops::util::widen_integer_input`.
    if let Some(widened) = crate::ops::util::widen_integer_input(tensor)? {
        return nanquantile(&widened, q, dim, keepdim, interpolation);
    }
    reduce_over_dims(tensor, dim, keepdim, |t, axis, keep| {
        nanquantile_axis(t, q, axis, keep, interpolation)
    })
}

/// Like [`quantiles`], ignoring NaN.
pub fn nanquantiles(
    tensor: &Tensor,
    qs: &[f64],
    dim: Option<Vec<isize>>,
    keepdim: bool,
    interpolation: QuantileInterpolation,
) -> Result<Tensor> {
    // An integer argument widens rather than being refused, the rule `mean`
    // already follows: a quantile of integers is an interpolated real number,
    // and NumPy answers one. See `ops::util::widen_integer_input`.
    if let Some(widened) = crate::ops::util::widen_integer_input(tensor)? {
        return nanquantiles(&widened, qs, dim, keepdim, interpolation);
    }
    if let Some(stacked) = quantiles_with_grad(tensor, qs, &dim, keepdim, interpolation, true)? {
        return Ok(stacked);
    }
    reduce_over_dims(tensor, dim, keepdim, |t, axis, keep| {
        nanquantiles_axis(t, qs, axis, keep, interpolation)
    })
}

/// The median over the non-NaN values, over one axis, several, or the whole
/// tensor.
///
/// `median` cannot take a list and this can, for the reason the two differ at
/// all: `median` reports the index of the element it selected, and an index
/// names a position along a single axis.
///
/// Like [`median`], an even count selects the lower of the two middle values
/// rather than averaging them; `nanquantile(0.5)` is the interpolated
/// equivalent.
pub fn nanmedian(tensor: &Tensor, dim: Option<Vec<isize>>, keepdim: bool) -> Result<Tensor> {
    reduce_over_dims(tensor, dim, keepdim, nanmedian_axis)
}

fn quantile_axis(
    tensor: &Tensor,
    q: f64,
    dim: Option<isize>,
    keepdim: bool,
    interpolation: QuantileInterpolation,
) -> Result<Tensor> {
    ensure_non_empty(tensor.numel(), "quantile")?;

    validate_quantile_value(q)?;
    ensure_floating_point_dtype(tensor.dtype())?;

    let (output, norm_dim) = match dim {
        None => (quantile_all(tensor, q, keepdim, interpolation)?, None),
        Some(dim_value) => {
            if tensor.ndim() == 0 {
                if dim_value == 0 || dim_value == -1 {
                    (quantile_all(tensor, q, keepdim, interpolation)?, None)
                } else {
                    return Err(MinitensorError::dim_out_of_range(dim_value, 1));
                }
            } else {
                let axis = normalize_dim(dim_value, tensor.ndim())?;
                (
                    quantile_along_dim(tensor, axis, keepdim, q, interpolation)?,
                    Some(axis),
                )
            }
        }
    };

    attach_quantile_grad(output, tensor, norm_dim, q, interpolation, false)
}

/// Attach a [`QuantileBackward`] gradient to a quantile value reduction.
fn attach_quantile_grad(
    output: Tensor,
    input: &Tensor,
    dim: Option<usize>,
    q: f64,
    interpolation: QuantileInterpolation,
    nan_aware: bool,
) -> Result<Tensor> {
    if !input.requires_grad() || !input.dtype().is_float() {
        return Ok(output);
    }
    let grad_fn = Arc::new(QuantileBackward {
        input_id: input.id(),
        input: input.detach(),
        dim,
        q,
        interpolation,
        nan_aware,
    });
    with_grad_fn(output, grad_fn)
}

fn quantiles_axis(
    tensor: &Tensor,
    qs: &[f64],
    dim: Option<isize>,
    keepdim: bool,
    interpolation: QuantileInterpolation,
) -> Result<Tensor> {
    ensure_non_empty(tensor.numel(), "quantile")?;

    if qs.is_empty() {
        return Err(MinitensorError::invalid_argument(
            "quantile() expected at least one probability value".to_string(),
        ));
    }

    for &q in qs {
        validate_quantile_value(q)?;
    }

    ensure_floating_point_dtype(tensor.dtype())?;

    match dim {
        None => quantiles_all(tensor, qs, keepdim, interpolation),
        Some(dim_value) => {
            if tensor.ndim() == 0 {
                if dim_value == 0 || dim_value == -1 {
                    return quantiles_all(tensor, qs, keepdim, interpolation);
                }
                return Err(MinitensorError::dim_out_of_range(dim_value, 1));
            }

            let axis = normalize_dim(dim_value, tensor.ndim())?;
            quantiles_along_dim(tensor, axis, qs, keepdim, interpolation)
        }
    }
}

fn nanquantile_axis(
    tensor: &Tensor,
    q: f64,
    dim: Option<isize>,
    keepdim: bool,
    interpolation: QuantileInterpolation,
) -> Result<Tensor> {
    ensure_non_empty(tensor.numel(), "nanquantile")?;

    validate_quantile_value(q)?;
    ensure_floating_point_dtype(tensor.dtype())?;

    let (output, norm_dim) = match dim {
        None => (nanquantile_all(tensor, q, keepdim, interpolation)?, None),
        Some(dim_value) => {
            if tensor.ndim() == 0 {
                if dim_value == 0 || dim_value == -1 {
                    (nanquantile_all(tensor, q, keepdim, interpolation)?, None)
                } else {
                    return Err(MinitensorError::dim_out_of_range(dim_value, 1));
                }
            } else {
                let axis = normalize_dim(dim_value, tensor.ndim())?;
                (
                    nanquantile_along_dim(tensor, axis, keepdim, q, interpolation)?,
                    Some(axis),
                )
            }
        }
    };
    attach_quantile_grad(output, tensor, norm_dim, q, interpolation, true)
}

/// Compute the median while ignoring NaN values.
///
/// Like [`median`], an even count selects the lower of the two middle values
/// rather than averaging them; `nanquantile(0.5)` is the interpolated
/// equivalent.
fn nanmedian_axis(tensor: &Tensor, dim: Option<isize>, keepdim: bool) -> Result<Tensor> {
    // An integer holds no NaN, so this is `median` -- the same delegation
    // `nanmax` and `nanprod` make, rather than a refusal. The median of
    // integers is an integer, so this one widens nothing.
    if !tensor.dtype().is_float() {
        return Ok(median(tensor, dim, keepdim)?.0);
    }
    ensure_floating_point_dtype_for(tensor.dtype(), "nanmedian")?;

    let (values, norm_dim) = match dim {
        None => (nanmedian_all(tensor, keepdim)?, None),
        Some(dim_value) => {
            if tensor.ndim() == 0 {
                if dim_value == 0 || dim_value == -1 {
                    (nanmedian_all(tensor, keepdim)?, None)
                } else {
                    return Err(MinitensorError::dim_out_of_range(dim_value, 1));
                }
            } else {
                let axis = normalize_dim(dim_value, tensor.ndim())?;
                (nanmedian_along_dim(tensor, axis, keepdim)?, Some(axis))
            }
        }
    };
    attach_median_grad(values, tensor, norm_dim, keepdim, true, None)
}

fn nanquantiles_axis(
    tensor: &Tensor,
    qs: &[f64],
    dim: Option<isize>,
    keepdim: bool,
    interpolation: QuantileInterpolation,
) -> Result<Tensor> {
    ensure_non_empty(tensor.numel(), "nanquantile")?;

    if qs.is_empty() {
        return Err(MinitensorError::invalid_argument(
            "nanquantile() expected at least one probability value".to_string(),
        ));
    }

    for &q in qs {
        validate_quantile_value(q)?;
    }

    ensure_floating_point_dtype(tensor.dtype())?;

    match dim {
        None => nanquantiles_all(tensor, qs, keepdim, interpolation),
        Some(dim_value) => {
            if tensor.ndim() == 0 {
                if dim_value == 0 || dim_value == -1 {
                    return nanquantiles_all(tensor, qs, keepdim, interpolation);
                }
                return Err(MinitensorError::dim_out_of_range(dim_value, 1));
            }

            let axis = normalize_dim(dim_value, tensor.ndim())?;
            nanquantiles_along_dim(tensor, axis, qs, keepdim, interpolation)
        }
    }
}

fn validate_quantile_value(q: f64) -> Result<()> {
    if !q.is_finite() {
        return Err(MinitensorError::invalid_argument(
            "quantile() requires a finite probability in [0, 1]".to_string(),
        ));
    }
    if !(0.0..=1.0).contains(&q) {
        return Err(MinitensorError::invalid_argument(format!(
            "quantile() expected q in [0, 1], got {q}",
        )));
    }
    Ok(())
}

fn ensure_floating_point_dtype(dtype: DataType) -> Result<()> {
    ensure_floating_point_dtype_for(dtype, "quantile")
}

fn ensure_floating_point_dtype_for(dtype: DataType, operation: &str) -> Result<()> {
    match dtype {
        DataType::Float32 | DataType::Float64 => Ok(()),
        _ => Err(MinitensorError::invalid_operation(format!(
            "{operation}() currently supports only floating point tensors"
        ))),
    }
}

// A single-element reduction slice trivially equals its only value; when that
// value is NaN it flows straight through. An all-NaN slice returns NaN rather
// than erroring -- there is no non-NaN element to report.

pub(crate) fn fill_quantiles_all_single_f32(value: f32, values: &mut [f32]) {
    if value.is_nan() {
        values.fill(f32::NAN);
    } else {
        values.fill(value);
    }
}

pub(crate) fn fill_quantiles_all_single_f64(value: f64, values: &mut [f64]) {
    if value.is_nan() {
        values.fill(f64::NAN);
    } else {
        values.fill(value);
    }
}

pub(crate) fn fill_nanquantiles_all_single_f32(value: f32, values: &mut [f32]) {
    // A single-element tensor's quantile is its value for every q; NaN flows
    // through, since an all-NaN input has no value to report.
    values.fill(value);
}

pub(crate) fn fill_nanquantiles_all_single_f64(value: f64, values: &mut [f64]) {
    values.fill(value);
}

#[derive(Clone, Copy)]
pub(crate) struct QuantilePosition {
    pub(crate) lower_idx: usize,
    pub(crate) upper_idx: usize,
    pub(crate) nearest_idx: usize,
    pub(crate) weight: f64,
}

pub(crate) fn quantile_positions_for_len(len: usize, qs: &[f64]) -> Vec<QuantilePosition> {
    let mut positions = Vec::with_capacity(qs.len());
    for &q in qs {
        positions.push(quantile_position_for_len_q(len, q));
    }
    positions
}

#[inline(always)]
pub(crate) fn quantile_position_for_len_q(len: usize, q: f64) -> QuantilePosition {
    if len <= 1 {
        return QuantilePosition {
            lower_idx: 0,
            upper_idx: 0,
            nearest_idx: 0,
            weight: 0.0,
        };
    }

    let max_index = (len - 1) as f64;
    let pos = (q * max_index).clamp(0.0, max_index);
    let lower_idx = pos.floor() as usize;
    let upper_idx = pos.ceil() as usize;
    let weight = (pos - lower_idx as f64).clamp(0.0, 1.0);
    let nearest_idx = nearest_index_with_tie_even(lower_idx, upper_idx, weight);

    QuantilePosition {
        lower_idx,
        upper_idx,
        nearest_idx,
        weight,
    }
}

#[inline(always)]
fn nearest_index_with_tie_even(lower_idx: usize, upper_idx: usize, weight: f64) -> usize {
    debug_assert!((0.0..=1.0).contains(&weight));

    if upper_idx <= lower_idx {
        debug_assert_eq!(
            upper_idx, lower_idx,
            "nearest_index_with_tie_even requires upper_idx >= lower_idx"
        );
        return lower_idx;
    }

    debug_assert_eq!(upper_idx, lower_idx + 1);

    if weight < 0.5 {
        lower_idx
    } else if weight > 0.5 {
        upper_idx
    } else {
        lower_idx + (lower_idx & 1)
    }
}

fn quantile_all(
    tensor: &Tensor,
    q: f64,
    keepdim: bool,
    interpolation: QuantileInterpolation,
) -> Result<Tensor> {
    if tensor.ndim() == 0 {
        return Ok(tensor.clone());
    }

    let mut result_data = TensorData::zeros_on_device(1, tensor.dtype(), tensor.device());

    match tensor.dtype() {
        DataType::Float32 => {
            let data = tensor
                .data()
                .as_f32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;
            let mut values = Vec::with_capacity(data.len());
            for &value in data {
                if value.is_nan() {
                    result_data.as_f32_slice_mut().ok_or_else(|| {
                        MinitensorError::internal_error("Failed to get mutable f32 slice")
                    })?[0] = f32::NAN;
                    let result_shape = if keepdim {
                        Shape::new(vec![1; tensor.ndim()])
                    } else {
                        Shape::scalar()
                    };
                    return Ok(Tensor::new(
                        Arc::new(result_data),
                        result_shape,
                        tensor.dtype(),
                        tensor.device(),
                        tensor.requires_grad(),
                    ));
                }
                values.push(value);
            }
            let quant = quantile_from_unsorted(&mut values, q, interpolation);
            result_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice")
            })?[0] = quant;
        }
        DataType::Float64 => {
            let data = tensor
                .data()
                .as_f64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;
            let mut values = Vec::with_capacity(data.len());
            for &value in data {
                if value.is_nan() {
                    result_data.as_f64_slice_mut().ok_or_else(|| {
                        MinitensorError::internal_error("Failed to get mutable f64 slice")
                    })?[0] = f64::NAN;
                    let result_shape = if keepdim {
                        Shape::new(vec![1; tensor.ndim()])
                    } else {
                        Shape::scalar()
                    };
                    return Ok(Tensor::new(
                        Arc::new(result_data),
                        result_shape,
                        tensor.dtype(),
                        tensor.device(),
                        tensor.requires_grad(),
                    ));
                }
                values.push(value);
            }
            let quant = quantile_from_unsorted(&mut values, q, interpolation);
            result_data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice")
            })?[0] = quant;
        }
        _ => unreachable!("dtype validated"),
    }

    let result_shape = if keepdim {
        Shape::new(vec![1; tensor.ndim()])
    } else {
        Shape::scalar()
    };

    Ok(Tensor::new(
        Arc::new(result_data),
        result_shape,
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    ))
}

fn nanquantile_all(
    tensor: &Tensor,
    q: f64,
    keepdim: bool,
    interpolation: QuantileInterpolation,
) -> Result<Tensor> {
    // A 0-d tensor is its own quantile; a NaN scalar returns NaN.
    if tensor.ndim() == 0 {
        return Ok(tensor.clone());
    }

    let mut result_data = TensorData::zeros_on_device(1, tensor.dtype(), tensor.device());

    // An all-NaN input yields NaN rather than an error.
    match tensor.dtype() {
        DataType::Float32 => {
            let data = tensor
                .data()
                .as_f32_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f32 slice"))?;
            let mut values: Vec<f32> = data.iter().copied().filter(|v| !v.is_nan()).collect();
            let quant = if values.is_empty() {
                f32::NAN
            } else {
                quantile_from_unsorted(&mut values, q, interpolation)
            };
            result_data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f32 slice")
            })?[0] = quant;
        }
        DataType::Float64 => {
            let data = tensor
                .data()
                .as_f64_slice()
                .ok_or_else(|| MinitensorError::internal_error("Failed to get f64 slice"))?;
            let mut values: Vec<f64> = data.iter().copied().filter(|v| !v.is_nan()).collect();
            let quant = if values.is_empty() {
                f64::NAN
            } else {
                quantile_from_unsorted(&mut values, q, interpolation)
            };
            result_data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get mutable f64 slice")
            })?[0] = quant;
        }
        _ => unreachable!("dtype validated"),
    }

    let result_shape = if keepdim {
        Shape::new(vec![1; tensor.ndim()])
    } else {
        Shape::scalar()
    };

    Ok(Tensor::new(
        Arc::new(result_data),
        result_shape,
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    ))
}

/// The total ordering on floats, which `num_traits::Float` does not expose.
///
/// Quantile selection orders by `total_cmp` rather than `partial_cmp` so that
/// NaN has a defined position instead of making the comparator inconsistent —
/// `select_nth_unstable_by` with an inconsistent comparator may panic or return
/// an arbitrary element. Callers filter NaN out beforehand; this keeps the
/// selection well-defined regardless.
pub(crate) trait TotalCmp: num_traits::Float + Copy {
    fn total_order(&self, other: &Self) -> Ordering;
}

impl TotalCmp for f32 {
    #[inline]
    fn total_order(&self, other: &Self) -> Ordering {
        self.total_cmp(other)
    }
}

impl TotalCmp for f64 {
    #[inline]
    fn total_order(&self, other: &Self) -> Ordering {
        self.total_cmp(other)
    }
}

/// The element of rank `idx`, found by quickselect. Reorders `values`.
fn select_rank<T: TotalCmp>(values: &mut [T], idx: usize) -> T {
    let (_, pivot, _) = values.select_nth_unstable_by(idx, |a, b| a.total_order(b));
    *pivot
}

/// The elements of ranks `lower_idx` and `upper_idx`.
///
/// Selecting the upper rank first partitions everything below it, so the second
/// selection only has to search that prefix.
fn select_rank_pair<T: TotalCmp>(values: &mut [T], lower_idx: usize, upper_idx: usize) -> (T, T) {
    if lower_idx == upper_idx {
        let value = select_rank(values, lower_idx);
        return (value, value);
    }

    let upper = select_rank(values, upper_idx);
    let lower = select_rank(&mut values[..upper_idx], lower_idx);
    (lower, upper)
}

/// Puts every rank in `ranks` at its own index, so reading `values[r]` for any
/// `r` in `ranks` gives the element of that rank.
///
/// `select_nth_unstable_by` partitions around the rank it finds, so selecting
/// the *middle* rank first leaves every lower rank inside the prefix and every
/// higher one inside the suffix. Recursing that way costs `O(n log k)`, where
/// selecting each rank over the whole slice would cost `O(n k)`.
///
/// `ranks` must be ascending, without duplicates, and inside `values`.
fn select_ranks<T: TotalCmp>(values: &mut [T], offset: usize, ranks: &[usize]) {
    if ranks.is_empty() {
        return;
    }
    let mid = ranks.len() / 2;
    // `ranks[mid]` is a rank of the whole slice; `offset` is where this
    // subslice starts in it, so the local index is the difference.
    let local = ranks[mid] - offset;
    debug_assert!(local < values.len());
    select_rank(values, local);
    select_ranks(&mut values[..local], offset, &ranks[..mid]);
    select_ranks(
        &mut values[local + 1..],
        offset + local + 1,
        &ranks[mid + 1..],
    );
}

/// The ranks `quantile_from_sorted_position` will actually read.
///
/// Only `Linear` and `Midpoint` look at two; the other three read one apiece,
/// so asking for all of them would select up to three times the work.
fn ranks_for_positions(
    positions: &[QuantilePosition],
    interpolation: QuantileInterpolation,
) -> Vec<usize> {
    let mut ranks = Vec::with_capacity(positions.len() * 2);
    for position in positions {
        match interpolation {
            QuantileInterpolation::Lower => ranks.push(position.lower_idx),
            QuantileInterpolation::Higher => ranks.push(position.upper_idx),
            QuantileInterpolation::Nearest => ranks.push(position.nearest_idx),
            QuantileInterpolation::Linear | QuantileInterpolation::Midpoint => {
                ranks.push(position.lower_idx);
                ranks.push(position.upper_idx);
            }
        }
    }
    ranks.sort_unstable();
    ranks.dedup();
    ranks
}

/// Above this many distinct ranks, ordering the whole slice beats selecting
/// them one at a time.
///
/// Measured over a million float32: one quickselect costs 3.7ms and a serial
/// sort 31, so selection is ahead while `log2(k)` stays under about eight.
/// The crossover is flat enough either side that a constant is as good as a
/// formula, and being wrong by a factor of two here costs a few milliseconds
/// rather than a wrong answer.
const QUANTILE_SELECT_LIMIT: usize = 64;

/// Arranges `values` so `quantile_from_sorted_position` can read every entry
/// of `positions`, by whichever of selection and sorting is cheaper.
///
/// A full sort is the obvious way and usually the wrong one: `percentile` with
/// two quantiles read 0.12x of NumPy because it sorted a million elements to
/// answer two questions, where NumPy selects.
///
/// `parallel` says whether a sort here may use the thread pool. It is false
/// when the caller is already inside it with one row per worker, where a
/// nested parallel sort would fight its own siblings for the same cores.
pub(crate) fn order_for_quantiles<T: TotalCmp + Send>(
    values: &mut [T],
    positions: &[QuantilePosition],
    interpolation: QuantileInterpolation,
    parallel: bool,
) {
    let ranks = ranks_for_positions(positions, interpolation);
    if ranks.len() <= QUANTILE_SELECT_LIMIT {
        select_ranks(values, 0, &ranks);
    } else if parallel {
        values.par_sort_unstable_by(|a, b| a.total_order(b));
    } else {
        // Unstable rather than stable: the order is total and equal elements
        // are read by index alone, so there is nothing for stability to
        // preserve -- and the stable sort allocates a second buffer.
        values.sort_unstable_by(|a, b| a.total_order(b));
    }
}

/// The `q`-th quantile of an unsorted slice, via quickselect rather than a full
/// sort: only the one or two order statistics that bracket the requested
/// position are needed, which is `O(n)` instead of `O(n log n)`.
///
/// Reorders `values`.
pub(crate) fn quantile_from_unsorted<T: TotalCmp>(
    values: &mut [T],
    q: f64,
    interpolation: QuantileInterpolation,
) -> T {
    if values.len() == 1 {
        return values[0];
    }

    let position = quantile_position_for_len_q(values.len(), q);
    if position.lower_idx == position.upper_idx {
        return select_rank(values, position.lower_idx);
    }

    match interpolation {
        QuantileInterpolation::Lower => select_rank(values, position.lower_idx),
        QuantileInterpolation::Higher => select_rank(values, position.upper_idx),
        QuantileInterpolation::Nearest => select_rank(values, position.nearest_idx),
        QuantileInterpolation::Linear | QuantileInterpolation::Midpoint => {
            let (lower, upper) = select_rank_pair(values, position.lower_idx, position.upper_idx);
            let interpolated = interpolation.interpolate(
                lower.to_f64().unwrap_or(f64::NAN),
                upper.to_f64().unwrap_or(f64::NAN),
                position.weight,
            );
            T::from(interpolated).unwrap_or_else(T::nan)
        }
    }
}

/// Read the `q`-th quantile out of an already-sorted slice.
pub(crate) fn quantile_from_sorted_position<T: TotalCmp>(
    values: &[T],
    position: &QuantilePosition,
    interpolation: QuantileInterpolation,
) -> T {
    match interpolation {
        QuantileInterpolation::Lower => values[position.lower_idx],
        QuantileInterpolation::Higher => values[position.upper_idx],
        QuantileInterpolation::Nearest => values[position.nearest_idx],
        QuantileInterpolation::Linear | QuantileInterpolation::Midpoint => {
            let lower = values[position.lower_idx].to_f64().unwrap_or(f64::NAN);
            let upper = values[position.upper_idx].to_f64().unwrap_or(f64::NAN);
            T::from(interpolation.interpolate(lower, upper, position.weight)).unwrap_or_else(T::nan)
        }
    }
}

#[cfg(test)]
mod core_tests {
    use super::*;
    use crate::Device;

    #[test]
    fn test_quantile_interpolation_modes() {
        let lower = 1.0;
        let upper = 3.0;
        let weight = 0.25;

        assert_eq!(
            QuantileInterpolation::Linear.interpolate(lower, upper, weight),
            1.5
        );
        assert_eq!(
            QuantileInterpolation::Lower.interpolate(lower, upper, weight),
            lower
        );
        assert_eq!(
            QuantileInterpolation::Higher.interpolate(lower, upper, weight),
            upper
        );
        assert_eq!(
            QuantileInterpolation::Midpoint.interpolate(lower, upper, weight),
            2.0
        );
        assert_eq!(
            QuantileInterpolation::Nearest.interpolate(lower, upper, 0.49),
            lower
        );
        assert_eq!(
            QuantileInterpolation::Nearest.interpolate(lower, upper, 0.5),
            upper
        );
    }

    #[test]
    fn test_normalize_reduction_dims_sorts_dedups_and_supports_negative_dims() {
        let dims = Some(vec![2, -1, 0, 2, -3]);
        let normalized = normalize_reduction_dims(dims, 3).unwrap();
        assert_eq!(normalized, Some(vec![0, 2]));
        assert_eq!(normalize_reduction_dims(None, 3).unwrap(), None);
    }

    #[test]
    fn test_normalize_reduction_dims_rejects_out_of_range() {
        assert!(normalize_reduction_dims(Some(vec![3]), 3).is_err());
        assert!(normalize_reduction_dims(Some(vec![-4]), 3).is_err());
    }

    #[test]
    fn test_non_nan_mask_for_float_tensors_and_invalid_dtype() {
        let f32_tensor = Tensor::new(
            Arc::new(TensorData::from_vec_f32(
                vec![1.0, f32::NAN, -2.0],
                Device::cpu(),
            )),
            Shape::new(vec![3]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let mask = non_nan_mask(&f32_tensor).unwrap();
        assert_eq!(mask.dtype(), DataType::Bool);
        assert_eq!(mask.data().as_bool_slice().unwrap(), &[true, false, true],);

        let f64_tensor = Tensor::new(
            Arc::new(TensorData::from_vec_f64(
                vec![f64::NAN, 4.0, 0.0],
                Device::cpu(),
            )),
            Shape::new(vec![3]),
            DataType::Float64,
            Device::cpu(),
            false,
        );
        let f64_mask = non_nan_mask(&f64_tensor).unwrap();
        assert_eq!(
            f64_mask.data().as_bool_slice().unwrap(),
            &[false, true, true],
        );

        let bool_tensor = Tensor::new(
            Arc::new(TensorData::from_vec_bool(vec![true, false], Device::cpu())),
            Shape::new(vec![2]),
            DataType::Bool,
            Device::cpu(),
            false,
        );
        assert!(non_nan_mask(&bool_tensor).is_err());
    }

    #[test]
    fn test_comparator_tie_breaking_and_nan_ordering() {
        let mut f32_desc = [(1, 2.0f32), (0, 2.0), (2, f32::NAN)];
        f32_desc.sort_by(cmp_f32_desc);
        assert!(f32_desc[0].1.is_nan());
        assert_eq!(f32_desc[0].0, 2);
        assert_eq!(
            f32_desc[1..].iter().map(|(i, _)| *i).collect::<Vec<_>>(),
            vec![0, 1]
        );

        let mut f32_asc = [(1, 2.0f32), (0, 2.0), (2, f32::NAN)];
        f32_asc.sort_by(cmp_f32_asc);
        assert!(f32_asc[2].1.is_nan());
        assert_eq!(f32_asc[2].0, 2);
        assert_eq!(
            f32_asc[..2].iter().map(|(i, _)| *i).collect::<Vec<_>>(),
            vec![0, 1]
        );

        let mut f64_desc = [(1, 2.0f64), (0, 2.0), (2, f64::NAN)];
        f64_desc.sort_by(cmp_f64_desc);
        assert!(f64_desc[0].1.is_nan());
        assert_eq!(f64_desc[0].0, 2);
        assert_eq!(
            f64_desc[1..].iter().map(|(i, _)| *i).collect::<Vec<_>>(),
            vec![0, 1]
        );

        let mut f64_asc = [(1, 2.0f64), (0, 2.0), (2, f64::NAN)];
        f64_asc.sort_by(cmp_f64_asc);
        assert!(f64_asc[2].1.is_nan());
        assert_eq!(f64_asc[2].0, 2);
        assert_eq!(
            f64_asc[..2].iter().map(|(i, _)| *i).collect::<Vec<_>>(),
            vec![0, 1]
        );

        let mut i32_desc = vec![(1, 7_i32), (0, 7_i32), (2, 1_i32)];
        i32_desc.sort_by(cmp_i32_desc);
        assert_eq!(i32_desc, vec![(0, 7), (1, 7), (2, 1)]);

        let mut i32_asc = vec![(1, 7_i32), (0, 7_i32), (2, 1_i32)];
        i32_asc.sort_by(cmp_i32_asc);
        assert_eq!(i32_asc, vec![(2, 1), (0, 7), (1, 7)]);

        let mut i64_desc = vec![(1, 7_i64), (0, 7_i64), (2, 1_i64)];
        i64_desc.sort_by(cmp_i64_desc);
        assert_eq!(i64_desc, vec![(0, 7), (1, 7), (2, 1)]);

        let mut i64_asc = vec![(1, 7_i64), (0, 7_i64), (2, 1_i64)];
        i64_asc.sort_by(cmp_i64_asc);
        assert_eq!(i64_asc, vec![(2, 1), (0, 7), (1, 7)]);

        let mut bool_desc = vec![(1, true), (0, true), (2, false)];
        bool_desc.sort_by(cmp_bool_desc);
        assert_eq!(bool_desc, vec![(0, true), (1, true), (2, false)]);

        let mut bool_asc = vec![(1, true), (0, true), (2, false)];
        bool_asc.sort_by(cmp_bool_asc);
        assert_eq!(bool_asc, vec![(2, false), (0, true), (1, true)]);
    }

    #[test]
    fn test_ensure_non_empty_guard() {
        assert!(ensure_non_empty(0, "median").is_err());
        assert!(ensure_non_empty(1, "median").is_ok());
    }
}

#[cfg(test)]
mod comparator_tests {
    use super::*;

    /// Every interesting float32 bit pattern, and a few ordinary values: the
    /// two zeros, both infinities, the subnormal boundary, quiet and signalling
    /// NaN of both signs.
    fn f32_corners() -> Vec<f32> {
        let mut values = vec![
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            3.5,
            -3.5,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::from_bits(0x8000_0001),
            f32::MAX,
            f32::MIN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];
        for bits in [0x7fc0_0000u32, 0xffc0_0000, 0x7f80_0001, 0xffff_ffff] {
            values.push(f32::from_bits(bits));
        }
        values
    }

    fn f64_corners() -> Vec<f64> {
        f32_corners().iter().map(|&v| v as f64).collect()
    }

    /// The scan comparators take a fast path for the ordinary case and fall
    /// through to the key comparators for everything else, so they can only
    /// stay in step if the fast path really does agree. Check every pair.
    #[test]
    fn the_scan_and_key_comparators_agree_everywhere() {
        for (i, &left) in f32_corners().iter().enumerate() {
            for (j, &right) in f32_corners().iter().enumerate() {
                let a = (i, left);
                let b = (j, right);
                assert_eq!(
                    scan_cmp_f32_asc(&a, &b),
                    cmp_f32_asc(&a, &b),
                    "ascending disagreed on {left} against {right}"
                );
                assert_eq!(
                    scan_cmp_f32_desc(&a, &b),
                    cmp_f32_desc(&a, &b),
                    "descending disagreed on {left} against {right}"
                );
            }
        }
        for (i, &left) in f64_corners().iter().enumerate() {
            for (j, &right) in f64_corners().iter().enumerate() {
                let a = (i, left);
                let b = (j, right);
                assert_eq!(scan_cmp_f64_asc(&a, &b), cmp_f64_asc(&a, &b));
                assert_eq!(scan_cmp_f64_desc(&a, &b), cmp_f64_desc(&a, &b));
            }
        }
    }

    /// The order itself, stated where it can be read: NaN last ascending and
    /// first descending, the two zeros equal, ties by position in both
    /// directions.
    #[test]
    fn the_order_is_the_documented_one() {
        let nan = (0usize, f32::NAN);
        let negative_nan = (1usize, -f32::NAN);
        let number = (2usize, 5.0f32);
        assert_eq!(cmp_f32_asc(&nan, &number), Ordering::Greater);
        assert_eq!(cmp_f32_asc(&negative_nan, &number), Ordering::Greater);
        assert_eq!(cmp_f32_desc(&nan, &number), Ordering::Less);
        assert_eq!(cmp_f32_desc(&negative_nan, &number), Ordering::Less);

        // Two NaNs are equal, so the position decides -- ascending, both ways.
        assert_eq!(cmp_f32_asc(&nan, &negative_nan), Ordering::Less);
        assert_eq!(cmp_f32_desc(&nan, &negative_nan), Ordering::Less);

        let minus_zero = (3usize, -0.0f32);
        let plus_zero = (4usize, 0.0f32);
        assert_eq!(cmp_f32_asc(&minus_zero, &plus_zero), Ordering::Less);
        assert_eq!(cmp_f32_desc(&minus_zero, &plus_zero), Ordering::Less);
    }
}

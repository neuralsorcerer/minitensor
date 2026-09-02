// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! The named activation units, on one skeleton.
//!
//! Each of these is the same shape: a value kernel, a chain-rule kernel, a
//! dtype dispatch and a gradient node. Only the first two say anything about
//! the op, so only those are written per unit; the rest is shared, and the
//! derivative sits next to the function it differentiates.
//!
//! Scalar parameters travel as a fixed-size array so one signature covers
//! `mish` (which takes none), `celu` (one) and `hardtanh` (two). They are
//! converted to the working width once per call rather than per element.

use super::*;
use crate::{
    autograd::{ActivationUnitBackward, with_grad_fn},
    error::{MinitensorError, Result},
    ops::map::{outputs_per_task, par_out_chunks, unary_map},
    ops::util::NegLogSigmoid,
    tensor::{DataType, Tensor, TensorData},
};
use std::sync::Arc;

/// Up to two scalar parameters, in the working width.
pub type UnitParams<T> = [T; 2];

/// An activation's value kernel, in both float widths.
pub type UnitKernel = (
    fn(f32, UnitParams<f32>) -> f32,
    fn(f64, UnitParams<f64>) -> f64,
);

/// An activation's chain rule, `(input, grad_out, params) -> grad_in`, in both
/// float widths.
pub type UnitGradKernel = (
    fn(f32, f32, UnitParams<f32>) -> f32,
    fn(f64, f64, UnitParams<f64>) -> f64,
);

/// Defines one [`UnitKernel`] from a single body, instantiated at both widths.
/// The body has to typecheck as `f32` and as `f64`, so it may not name either.
macro_rules! unit_kernel {
    ($(#[$meta:meta])* $name:ident, |$x:pat_param, $p:pat_param| $body:expr) => {
        $(#[$meta])*
        const $name: UnitKernel = {
            #[inline(always)]
            fn narrow($x: f32, $p: UnitParams<f32>) -> f32 {
                $body
            }
            #[inline(always)]
            fn wide($x: f64, $p: UnitParams<f64>) -> f64 {
                $body
            }
            (narrow, wide)
        };
    };
}

/// [`unit_kernel!`] for the chain rule, which also takes the incoming gradient.
macro_rules! unit_grad_kernel {
    ($(#[$meta:meta])* $name:ident, |$x:pat_param, $g:pat_param, $p:pat_param| $body:expr) => {
        $(#[$meta])*
        const $name: UnitGradKernel = {
            #[inline(always)]
            fn narrow($x: f32, $g: f32, $p: UnitParams<f32>) -> f32 {
                $body
            }
            #[inline(always)]
            fn wide($x: f64, $g: f64, $p: UnitParams<f64>) -> f64 {
                $body
            }
            (narrow, wide)
        };
    };
}

// The two macros above are the whole per-op cost of this skeleton, so the
// special functions in `ops::special` reach for them rather than restating it.
pub(crate) use {unit_grad_kernel, unit_kernel};

/// `1 / (1 + exp(-x))`, evaluated on whichever side of zero keeps the
/// exponential from overflowing. Several units below need it, and none of them
/// can call the tensor-level `sigmoid` from inside an element kernel.
macro_rules! stable_sigmoid {
    ($x:expr) => {{
        let x = $x;
        if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            let e = x.exp();
            e / (1.0 + e)
        }
    }};
}

/// `log(1 + exp(x))`, likewise. Past the cutoff `exp(x)` overflows while
/// `log1p(exp(x))` has already converged on `x` to the last bit -- at x = 20
/// the difference is 2e-9, which is below the float32 epsilon at that
/// magnitude and 12 orders below float64's answer.
macro_rules! stable_softplus {
    ($x:expr) => {{
        let x = $x;
        if x > 20.0 { x } else { x.exp().ln_1p() }
    }};
}

// --- hardtanh, and relu6 on top of it --------------------------------------

unit_kernel!(
    /// `x` clamped to `[p[0], p[1]]`. NaN survives, as it does for `clamp`.
    HARDTANH, |x, p| if x < p[0] {
        p[0]
    } else if x > p[1] {
        p[1]
    } else {
        x
    }
);
unit_grad_kernel!(
    /// Flat outside the bounds, so the gradient is the incoming one strictly
    /// inside them and zero elsewhere -- the endpoints included, where the
    /// one-sided derivatives disagree.
    HARDTANH_D, |x, g, p| if x > p[0] && x < p[1] { g } else { 0.0 }
);

// --- hardsigmoid and hardswish ---------------------------------------------

unit_kernel!(
    /// The piecewise-linear stand-in for `sigmoid`: `0` below -3, `1` above 3,
    /// `x/6 + 1/2` between. No exponential, which is the point of it.
    HARDSIGMOID, |x, _p| if x <= -3.0 {
        0.0
    } else if x >= 3.0 {
        1.0
    } else {
        x / 6.0 + 0.5
    }
);
unit_grad_kernel!(
    /// `1/6` on the sloped segment, zero on the flats.
    HARDSIGMOID_D, |x, g, _p| if x > -3.0 && x < 3.0 {
        g / 6.0
    } else {
        0.0
    }
);

unit_kernel!(
    /// `x * hardsigmoid(x)`: the piecewise-linear stand-in for `silu`.
    HARDSWISH, |x, _p| if x <= -3.0 {
        0.0
    } else if x >= 3.0 {
        x
    } else {
        x * (x + 3.0) / 6.0
    }
);
unit_grad_kernel!(
    /// `d/dx x(x+3)/6 = (2x+3)/6` on the quadratic segment; `0` and `1` on the
    /// flats either side.
    HARDSWISH_D, |x, g, _p| if x <= -3.0 {
        0.0
    } else if x >= 3.0 {
        g
    } else {
        g * (2.0 * x + 3.0) / 6.0
    }
);

// --- threshold and softshrink ----------------------------------------------

unit_kernel!(
    /// `x` where it exceeds `p[0]`, the replacement `p[1]` elsewhere.
    THRESHOLD, |x, p| if x > p[0] { x } else { p[1] }
);
unit_grad_kernel!(
    /// The replacement does not depend on `x`, so nothing flows through it.
    THRESHOLD_D, |x, g, p| if x > p[0] { g } else { 0.0 }
);

unit_kernel!(
    /// Shrinks towards zero by `p[0]`, and flattens the band `[-p[0], p[0]]`.
    /// Unlike `hardshrink` this is continuous: it subtracts the threshold
    /// rather than leaving the value where it was.
    SOFTSHRINK, |x, p| if x > p[0] {
        x - p[0]
    } else if x < -p[0] {
        x + p[0]
    } else {
        0.0
    }
);
unit_grad_kernel!(
    /// A unit slope outside the band, zero inside it.
    SOFTSHRINK_D, |x, g, p| if x > p[0] || x < -p[0] {
        g
    } else {
        0.0
    }
);

// --- tanhshrink, mish, celu, logsigmoid ------------------------------------

unit_kernel!(
    /// `x - tanh(x)`: what `tanh` leaves behind, which is `x^3/3` near zero.
    TANHSHRINK, |x, _p| x - x.tanh()
);
unit_grad_kernel!(
    /// `1 - sech^2(x) = tanh^2(x)`, written as the square so no second
    /// transcendental is needed.
    TANHSHRINK_D, |x, g, _p| {
        let t = x.tanh();
        g * t * t
    }
);

unit_kernel!(
    /// `x * tanh(softplus(x))`. Smooth, non-monotonic, and self-regularising:
    /// it keeps a small negative tail instead of clipping it to zero.
    MISH, |x, _p| x * stable_softplus!(x).tanh()
);
unit_grad_kernel!(
    /// `tanh(sp) + x * sech^2(sp) * sigmoid(x)`, where `sp = softplus(x)` and
    /// `d(sp)/dx = sigmoid(x)`. `sech^2` is written as `1 - tanh^2`.
    MISH_D, |x, g, _p| {
        let t = stable_softplus!(x).tanh();
        g * (t + x * (1.0 - t * t) * stable_sigmoid!(x))
    }
);

unit_kernel!(
    /// `max(0, x) + min(0, alpha * (exp(x / alpha) - 1))`, with `alpha` in
    /// `p[0]`. Unlike `elu` it is continuously differentiable at zero for
    /// every `alpha`, since the exponential is scaled to meet slope 1 there.
    CELU, |x, p| if x > 0.0 {
        x
    } else {
        p[0] * (x / p[0]).exp_m1()
    }
);
unit_grad_kernel!(
    /// `1` above zero and `exp(x / alpha)` below, which agree at zero -- that
    /// is what distinguishes `celu` from `elu`.
    CELU_D, |x, g, p| if x > 0.0 {
        g
    } else {
        g * (x / p[0]).exp()
    }
);

// `logsigmoid`'s value kernel is not declared here. `-softplus(-x)` has a
// vectorized implementation, so the function computes its own values through
// `NegLogSigmoid` rather than through the scalar unit machinery, and a
// constant restating the same formula would be a second definition that
// nothing evaluates. The gradient has no such kernel and stays.
unit_grad_kernel!(
    /// `d/dx log(sigmoid(x)) = 1 - sigmoid(x) = sigmoid(-x)`.
    LOGSIGMOID_D, |x, g, _p| g * stable_sigmoid!(-x)
);

/// Applies one value kernel over a float tensor.
fn unit_forward_data(
    tensor: &Tensor,
    name: &str,
    kernel: UnitKernel,
    params: UnitParams<f64>,
) -> Result<TensorData> {
    match tensor.dtype() {
        DataType::Float32 => {
            let input = tensor.data().as_f32_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f32 slice from input tensor")
            })?;
            let narrow = [params[0] as f32, params[1] as f32];
            let op = kernel.0;
            Ok(TensorData::from_vec::<f32>(
                unary_map(input, move |v| op(v, narrow)),
                DataType::Float32,
                tensor.device(),
            ))
        }
        DataType::Float64 => {
            let input = tensor.data().as_f64_slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get f64 slice from input tensor")
            })?;
            let op = kernel.1;
            Ok(TensorData::from_vec::<f64>(
                unary_map(input, move |v| op(v, params)),
                DataType::Float64,
                tensor.device(),
            ))
        }
        other => Err(MinitensorError::invalid_operation(format!(
            "{name} is only supported for floating point tensors, got {other}"
        ))),
    }
}

/// The shared body: run the value kernel, and record the chain rule if the
/// input wants a gradient.
pub(crate) fn unary_unit(
    tensor: &Tensor,
    name: &'static str,
    kernel: UnitKernel,
    grad_kernel: UnitGradKernel,
    params: UnitParams<f64>,
) -> Result<Tensor> {
    let output_data = unit_forward_data(tensor, name, kernel, params)?;
    unary_unit_from_data(tensor, name, output_data, grad_kernel, params)
}

/// The half of [`unary_unit`] after the values exist: wrap them in a tensor
/// and record the chain rule.
///
/// Split out for the units whose forward has a vectorized kernel and whose
/// backward does not, so they can compute their own values without also
/// restating how a unit's gradient is attached.
pub(crate) fn unary_unit_from_data(
    tensor: &Tensor,
    name: &'static str,
    output_data: TensorData,
    grad_kernel: UnitGradKernel,
    params: UnitParams<f64>,
) -> Result<Tensor> {
    let output = Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        tensor.requires_grad(),
    );

    if output.requires_grad() {
        return with_grad_fn(
            output,
            Arc::new(ActivationUnitBackward {
                input_id: tensor.id(),
                input: tensor.detach(),
                name,
                grad_kernel,
                params,
            }),
        );
    }

    Ok(output)
}

/// Declares a unit that takes no parameters.
macro_rules! plain_unit {
    ($name:ident, $kernel:ident, $grad:ident, $doc:literal) => {
        #[doc = $doc]
        pub fn $name(tensor: &Tensor) -> Result<Tensor> {
            unary_unit(tensor, stringify!($name), $kernel, $grad, [0.0; 2])
        }
    };
}

plain_unit!(
    hardsigmoid,
    HARDSIGMOID,
    HARDSIGMOID_D,
    "`sigmoid` replaced by three straight lines: 0 below -3, 1 above 3, `x/6 + 1/2` between."
);
plain_unit!(
    hardswish,
    HARDSWISH,
    HARDSWISH_D,
    "`x * hardsigmoid(x)`: `silu` with the exponential replaced by three straight lines."
);
plain_unit!(
    tanhshrink,
    TANHSHRINK,
    TANHSHRINK_D,
    "`x - tanh(x)`, what `tanh` leaves behind."
);
plain_unit!(
    mish,
    MISH,
    MISH_D,
    "`x * tanh(softplus(x))`: smooth, non-monotonic, and keeps a small negative tail."
);
/// `log(sigmoid(x))`, evaluated as `-softplus(-x)` so it stays exact where the
/// direct form underflows.
///
/// Written out rather than declared with `plain_unit!` because `-softplus(-x)`
/// has a vectorized kernel and the scalar rearrangement it replaces cost two
/// `libm` calls an element. Only the value kernel changes: the gradient is
/// still the unit one, attached the way every other unit attaches it.
pub fn logsigmoid(tensor: &Tensor) -> Result<Tensor> {
    macro_rules! values_for {
        ($ty:ty, $slice:ident, $dtype:expr) => {{
            let input = tensor.data().$slice().ok_or_else(|| {
                MinitensorError::internal_error("Failed to get slice from input tensor")
            })?;
            let mut values = vec![<$ty>::default(); input.len()];
            par_out_chunks(&mut values, outputs_per_task(1), &|offset, out_block| {
                let block = &input[offset..offset + out_block.len()];
                <$ty>::neg_log_sigmoid_into(block, out_block);
                // The kernel computes `-log(sigmoid(x))`, the form binary
                // cross-entropy wants; this is the other sign of it. The pass
                // is over a block still in cache and vectorizes on its own.
                for o in out_block.iter_mut() {
                    *o = -*o;
                }
            });
            TensorData::from_vec::<$ty>(values, $dtype, tensor.device())
        }};
    }

    let output_data = match tensor.dtype() {
        DataType::Float32 => values_for!(f32, as_f32_slice, DataType::Float32),
        DataType::Float64 => values_for!(f64, as_f64_slice, DataType::Float64),
        other => {
            return Err(MinitensorError::invalid_operation(format!(
                "logsigmoid is only supported for floating point tensors, got {other}"
            )));
        }
    };
    unary_unit_from_data(tensor, "logsigmoid", output_data, LOGSIGMOID_D, [0.0; 2])
}

/// `x` clamped to `[min_val, max_val]`, with no gradient outside them.
pub fn hardtanh(tensor: &Tensor, min_val: f64, max_val: f64) -> Result<Tensor> {
    // Spelled out rather than as `!(min_val <= max_val)`: NaN bounds compare
    // false either way round, and a clamp against them has no meaning.
    if min_val.is_nan() || max_val.is_nan() || min_val > max_val {
        return Err(MinitensorError::invalid_argument(format!(
            "hardtanh requires min_val <= max_val, got {min_val} and {max_val}"
        )));
    }
    unary_unit(tensor, "hardtanh", HARDTANH, HARDTANH_D, [min_val, max_val])
}

/// `hardtanh` on `[0, 6]`: the clipped ReLU that quantized networks use.
pub fn relu6(tensor: &Tensor) -> Result<Tensor> {
    // Named separately because it is what the literature and every other
    // library call it, but there is nothing else to it.
    unary_unit(tensor, "relu6", HARDTANH, HARDTANH_D, [0.0, 6.0])
}

/// `x` where it exceeds `threshold`, `value` everywhere else.
pub fn threshold(tensor: &Tensor, threshold: f64, value: f64) -> Result<Tensor> {
    unary_unit(
        tensor,
        "threshold",
        THRESHOLD,
        THRESHOLD_D,
        [threshold, value],
    )
}

/// Shrinks each element towards zero by `lambd`, flattening `[-lambd, lambd]`.
pub fn softshrink(tensor: &Tensor, lambd: f64) -> Result<Tensor> {
    if lambd.is_nan() || lambd < 0.0 {
        return Err(MinitensorError::invalid_argument(format!(
            "softshrink requires lambd to be non-negative, got {lambd}"
        )));
    }
    unary_unit(tensor, "softshrink", SOFTSHRINK, SOFTSHRINK_D, [lambd, 0.0])
}

/// `max(0, x) + min(0, alpha * (exp(x / alpha) - 1))`: `elu` rescaled so the
/// slope is continuous at zero for every `alpha`.
pub fn celu(tensor: &Tensor, alpha: f64) -> Result<Tensor> {
    if alpha == 0.0 || alpha.is_nan() {
        return Err(MinitensorError::invalid_argument(format!(
            "celu requires a non-zero alpha, got {alpha}"
        )));
    }
    unary_unit(tensor, "celu", CELU, CELU_D, [alpha, 0.0])
}

/// `softmax` of the negated input: the distribution that favours the smallest
/// element instead of the largest.
pub fn softmin(tensor: &Tensor, dim: Option<usize>) -> Result<Tensor> {
    // Composed rather than given its own kernel: negation is exact in floating
    // point, so this is the same distribution to the last bit, and it inherits
    // the shift-by-the-maximum that keeps `softmax` from overflowing.
    softmax(&crate::ops::arithmetic::neg(tensor)?, dim)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{autograd::backward_collect, device::Device, tensor::Shape};

    fn f64_tensor(data: Vec<f64>) -> Tensor {
        let len = data.len();
        Tensor::new(
            Arc::new(TensorData::from_vec_f64(data, Device::cpu())),
            Shape::new(vec![len]),
            DataType::Float64,
            Device::cpu(),
            false,
        )
    }

    fn wide(tensor: &Tensor) -> Vec<f64> {
        tensor.data().as_f64_slice().unwrap().to_vec()
    }

    /// A spread that reaches every branch of every unit here: both flats of
    /// the hard units, the sloped middles, the shrink bands, and enough
    /// magnitude either way to overflow a naive `exp`.
    const SAMPLE: [f64; 13] = [
        -40.0, -6.0, -3.0, -2.5, -1.0, -0.25, 0.0, 0.25, 1.0, 2.5, 3.0, 6.0, 40.0,
    ];

    /// One unit under test: its name and the call with its parameters bound.
    type BoundUnit = (&'static str, fn(&Tensor) -> Result<Tensor>);

    /// One unit's reference: its name and the scalar it has to agree with.
    type Reference = (&'static str, Box<dyn Fn(f64) -> f64>);

    /// Every unit as a closure of one tensor, with its scalar parameters bound.
    fn units() -> Vec<BoundUnit> {
        vec![
            ("hardsigmoid", hardsigmoid),
            ("hardswish", hardswish),
            ("tanhshrink", tanhshrink),
            ("mish", mish),
            ("logsigmoid", logsigmoid),
            ("relu6", relu6),
            ("hardtanh", |t| hardtanh(t, -1.0, 1.0)),
            ("threshold", |t| threshold(t, 0.5, -1.0)),
            ("softshrink", |t| softshrink(t, 0.5)),
            ("celu", |t| celu(t, 1.5)),
        ]
    }

    #[test]
    fn forward_values_match_their_definitions() {
        let sigmoid = |x: f64| 1.0 / (1.0 + (-x).exp());
        let softplus = |x: f64| x.exp().ln_1p();
        let expected: Vec<Reference> = vec![
            (
                "hardsigmoid",
                Box::new(|x: f64| (x / 6.0 + 0.5).clamp(0.0, 1.0)),
            ),
            (
                "hardswish",
                Box::new(move |x: f64| x * (x / 6.0 + 0.5).clamp(0.0, 1.0)),
            ),
            ("tanhshrink", Box::new(|x: f64| x - x.tanh())),
            ("relu6", Box::new(|x: f64| x.clamp(0.0, 6.0))),
            ("hardtanh", Box::new(|x: f64| x.clamp(-1.0, 1.0))),
            (
                "threshold",
                Box::new(|x: f64| if x > 0.5 { x } else { -1.0 }),
            ),
            (
                "softshrink",
                Box::new(|x: f64| {
                    if x > 0.5 {
                        x - 0.5
                    } else if x < -0.5 {
                        x + 0.5
                    } else {
                        0.0
                    }
                }),
            ),
            (
                "celu",
                Box::new(|x: f64| {
                    if x > 0.0 {
                        x
                    } else {
                        1.5 * ((x / 1.5).exp() - 1.0)
                    }
                }),
            ),
        ];

        let input = f64_tensor(SAMPLE.to_vec());
        for (name, op) in units() {
            let Some((_, reference)) = expected.iter().find(|(n, _)| *n == name) else {
                continue;
            };
            let got = wide(&op(&input).unwrap());
            for (i, &x) in SAMPLE.iter().enumerate() {
                let want = reference(x);
                assert!(
                    (got[i] - want).abs() <= 1e-12 * want.abs().max(1.0),
                    "{name}({x}) = {}, want {want}",
                    got[i]
                );
            }
        }

        // `mish` and `logsigmoid` are checked against forms that only hold
        // where they do not overflow, so their extremes are checked separately.
        let got = wide(&mish(&input).unwrap());
        for (i, &x) in SAMPLE.iter().enumerate() {
            if x.abs() <= 20.0 {
                let want = x * softplus(x).tanh();
                assert!((got[i] - want).abs() <= 1e-12 * want.abs().max(1.0));
            }
        }
        let got = wide(&logsigmoid(&input).unwrap());
        for (i, &x) in SAMPLE.iter().enumerate() {
            if x.abs() <= 20.0 {
                let want = sigmoid(x).ln();
                assert!((got[i] - want).abs() <= 1e-12 * want.abs().max(1.0));
            }
        }
    }

    #[test]
    fn the_saturating_units_stay_finite_where_a_naive_form_overflows() {
        // exp(710) is infinity in f64, so `log(sigmoid(x))` and
        // `x * tanh(log(1 + exp(x)))` written directly give NaN at both ends.
        let extreme = f64_tensor(vec![-800.0, 800.0]);

        let got = wide(&logsigmoid(&extreme).unwrap());
        assert_eq!(got[0], -800.0, "logsigmoid must converge on x, not -inf");
        // At the other end sigmoid has already rounded to 1, so its log has
        // underflowed to zero -- but it must be a zero, not the NaN that
        // `log(1/(1 + exp(-800)))` gives.
        assert_eq!(got[1], 0.0, "logsigmoid({}) = {}", 800.0, got[1]);

        let got = wide(&mish(&extreme).unwrap());
        assert!(got[0].abs() < 1e-300, "mish({}) = {}", -800.0, got[0]);
        assert_eq!(got[1], 800.0, "mish must converge on x for large x");
    }

    #[test]
    fn gradients_match_central_differences() {
        // Off every kink: the hard units break at -3, 0, 3 and their bounds,
        // `threshold` at 0.5, `softshrink` at +/-0.5, `celu` at 0.
        let sample = [-4.5, -2.0, -0.75, 0.75, 2.0, 4.5];
        let eps = 1e-6;

        for (name, op) in units() {
            let input = f64_tensor(sample.to_vec()).requires_grad_(true);
            let out = op(&input).unwrap();
            let seed = Tensor::ones(out.shape().clone(), out.dtype(), out.device(), false);
            let grads = backward_collect(&out, Some(seed)).unwrap();
            let analytic = wide(grads.get(&input.id()).unwrap());

            for (i, &x) in sample.iter().enumerate() {
                let mut up = sample.to_vec();
                let mut down = sample.to_vec();
                up[i] = x + eps;
                down[i] = x - eps;
                let numeric = (wide(&op(&f64_tensor(up)).unwrap())[i]
                    - wide(&op(&f64_tensor(down)).unwrap())[i])
                    / (2.0 * eps);
                assert!(
                    (analytic[i] - numeric).abs() <= 1e-5 * (1.0 + numeric.abs()),
                    "{name}' at {x}: analytic {}, numeric {numeric}",
                    analytic[i]
                );
            }
        }
    }

    #[test]
    fn the_flat_regions_pass_no_gradient() {
        // Inside a flat segment the derivative is exactly zero, which a
        // finite difference cannot distinguish from "very small".
        let cases: Vec<(BoundUnit, Vec<f64>)> = vec![
            (("hardsigmoid", hardsigmoid), vec![-5.0, 5.0]),
            (("hardswish", hardswish), vec![-5.0]),
            (("relu6", relu6), vec![-1.0, 7.0]),
            (
                ("softshrink", |t| softshrink(t, 0.5)),
                vec![-0.25, 0.0, 0.25],
            ),
            (("threshold", |t| threshold(t, 0.5, -1.0)), vec![0.0, 0.5]),
        ];

        for ((name, op), values) in cases {
            let n = values.len();
            let input = f64_tensor(values).requires_grad_(true);
            let out = op(&input).unwrap();
            let seed = Tensor::ones(out.shape().clone(), out.dtype(), out.device(), false);
            let grads = backward_collect(&out, Some(seed)).unwrap();
            assert_eq!(
                wide(grads.get(&input.id()).unwrap()),
                vec![0.0; n],
                "{name} leaked a gradient through a flat segment"
            );
        }
    }

    #[test]
    fn relu6_is_hardtanh_on_its_interval() {
        let input = f64_tensor(SAMPLE.to_vec());
        assert_eq!(
            wide(&relu6(&input).unwrap()),
            wide(&hardtanh(&input, 0.0, 6.0).unwrap())
        );
    }

    #[test]
    fn softmin_is_softmax_of_the_negation() {
        let input = f64_tensor(vec![1.0, 2.0, 3.0]);
        let got = wide(&softmin(&input, Some(0)).unwrap());

        // It sums to one and ranks the smallest element highest, which is the
        // whole difference from `softmax`.
        assert!((got.iter().sum::<f64>() - 1.0).abs() < 1e-15);
        assert!(got[0] > got[1] && got[1] > got[2]);

        let negated = f64_tensor(vec![-1.0, -2.0, -3.0]);
        assert_eq!(got, wide(&softmax(&negated, Some(0)).unwrap()));
    }

    #[test]
    fn parameters_are_validated_and_non_float_dtypes_rejected() {
        let input = f64_tensor(vec![1.0]);
        assert!(hardtanh(&input, 1.0, -1.0).is_err());
        assert!(softshrink(&input, -0.5).is_err());
        assert!(celu(&input, 0.0).is_err());
        // The degenerate but well-defined bounds are accepted.
        assert_eq!(wide(&hardtanh(&input, 2.0, 2.0).unwrap()), vec![2.0]);
        assert_eq!(wide(&softshrink(&input, 0.0).unwrap()), vec![1.0]);

        let ints = Tensor::new(
            Arc::new(TensorData::from_vec_i64(vec![1, 2], Device::cpu())),
            Shape::new(vec![2]),
            DataType::Int64,
            Device::cpu(),
            false,
        );
        for (_, op) in units() {
            assert!(op(&ints).is_err());
        }
    }

    #[test]
    fn empty_tensors_come_back_empty() {
        let empty = f64_tensor(vec![]);
        for (name, op) in units() {
            let out = op(&empty).unwrap();
            assert_eq!(out.shape().dims(), &[0], "{name}");
            assert!(wide(&out).is_empty(), "{name}");
        }
    }
}

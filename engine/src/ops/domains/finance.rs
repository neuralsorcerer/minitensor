// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Black-Scholes, as one kernel, with the Greeks as its gradient.
//!
//! This is the case where automatic differentiation and the domain want exactly
//! the same thing. The Greeks *are* the partial derivatives of the option
//! price: delta is `dV/dS`, vega is `dV/dsigma`, rho is `dV/dr`, theta is
//! `-dV/dT`. A desk computes them constantly, and the usual way to get them
//! from a tensor library is to write the price as a chain of twenty operations
//! and let the tape differentiate it — which works, and costs twenty
//! intermediates forward and twenty gradient kernels back, for quantities that
//! have been in closed form since 1973.
//!
//! So the forward is one pass — `d1`, `d2`, two normal CDFs and a discount
//! factor per element, nothing materialised in between — and the backward
//! writes the Greeks down directly. Both directions allocate their output and
//! nothing else.
//!
//! ```text
//!     d1 = [ln(S/K) + (r + sigma^2/2) T] / (sigma sqrt(T))
//!     d2 = d1 - sigma sqrt(T)
//!     call = S Phi(d1) - K e^{-rT} Phi(d2)
//!     put  = K e^{-rT} Phi(-d2) - S Phi(-d1)
//! ```
//!
//! # The degenerate corner
//!
//! At `sigma sqrt(T) = 0` — an expired option, or a zero-volatility
//! assumption — `d1` is `+-inf` and the formula above is `0 * inf`. The limit
//! is the intrinsic value, `max(S - K e^{-rT}, 0)` for a call, and that is what
//! this returns. Its derivative is a step: delta is 1 or 0 depending on which
//! side of the strike the spot is, vega is 0, and at exactly `S = K e^{-rT}`
//! there is no derivative at all. A subgradient of 0 is returned there, which
//! is the same convention `relu` uses at the origin and for the same reason:
//! the alternative is a NaN that propagates through an entire portfolio.

use crate::{
    autograd::{GradientFunction, TensorId, with_grad_fn},
    error::{MinitensorError, Result},
    ops::gaussian::normal_cdf,
    ops::map::{PAR_CHUNK, par_out_chunks},
    tensor::{DataType, Shape, Tensor, TensorData},
};
use rustc_hash::FxHashMap;
use std::sync::Arc;

/// Which side of the contract is being priced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptionKind {
    Call,
    Put,
}

impl OptionKind {
    /// `+1` for a call, `-1` for a put.
    ///
    /// Both formulas are the same expression under this sign — put-call parity
    /// in the form that lets one kernel price both, rather than two kernels
    /// that must be kept in step.
    #[inline]
    fn sign(self) -> f64 {
        match self {
            Self::Call => 1.0,
            Self::Put => -1.0,
        }
    }
}

/// The standard normal density, for vega and gamma.
#[inline]
fn normal_pdf(x: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// Everything one option's price and Greeks are built from, computed once.
///
/// Forward and backward both need `d1`, `d2`, the discount factor and the
/// two CDFs; deriving them in one place is what keeps the two directions from
/// drifting apart, and what makes the degenerate case a single branch rather
/// than one per Greek.
#[derive(Clone, Copy)]
struct Quote {
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    time: f64,
    /// `e^{-rT}`.
    discount: f64,
    /// `d1`; infinite in the degenerate branch, which is how
    /// [`Quote::is_degenerate`] recognises it.
    d1: f64,
    /// `Phi(sign * d1)`.
    nd1: f64,
    /// `Phi(sign * d2)`.
    nd2: f64,
    sign: f64,
}

impl Quote {
    fn new(spot: f64, strike: f64, rate: f64, vol: f64, time: f64, kind: OptionKind) -> Self {
        let sign = kind.sign();
        let discount = (-rate * time).exp();
        let spread = if time > 0.0 && vol > 0.0 {
            vol * time.sqrt()
        } else {
            0.0
        };
        let (d1, d2) = if spread > 0.0 && spot > 0.0 && strike > 0.0 {
            let d1 = ((spot / strike).ln() + (rate + 0.5 * vol * vol) * time) / spread;
            (d1, d1 - spread)
        } else {
            (f64::INFINITY, f64::INFINITY)
        };
        let (nd1, nd2) = if d1.is_finite() {
            (
                normal_cdf(sign * d1, 0.0, 1.0),
                normal_cdf(sign * d2, 0.0, 1.0),
            )
        } else {
            (0.0, 0.0)
        };
        Self {
            spot,
            strike,
            rate,
            vol,
            time,
            discount,
            d1,
            nd1,
            nd2,
            sign,
        }
    }

    /// Whether the closed form applies, or the intrinsic-value limit does.
    #[inline]
    fn is_degenerate(&self) -> bool {
        !self.d1.is_finite()
    }

    fn price(&self) -> f64 {
        if self.is_degenerate() {
            let intrinsic = self.sign * (self.spot - self.strike * self.discount);
            return intrinsic.max(0.0);
        }
        self.sign * (self.spot * self.nd1 - self.strike * self.discount * self.nd2)
    }

    /// `dV/dS`.
    fn delta(&self) -> f64 {
        if self.is_degenerate() {
            let intrinsic = self.sign * (self.spot - self.strike * self.discount);
            return if intrinsic > 0.0 { self.sign } else { 0.0 };
        }
        self.sign * self.nd1
    }

    /// `dV/dK`.
    fn dual_delta(&self) -> f64 {
        if self.is_degenerate() {
            let intrinsic = self.sign * (self.spot - self.strike * self.discount);
            return if intrinsic > 0.0 {
                -self.sign * self.discount
            } else {
                0.0
            };
        }
        -self.sign * self.discount * self.nd2
    }

    /// `dV/dsigma`. The same for a call and a put — parity differs by a
    /// forward contract, which does not depend on volatility.
    fn vega(&self) -> f64 {
        if self.is_degenerate() {
            return 0.0;
        }
        self.spot * normal_pdf(self.d1) * self.time.sqrt()
    }

    /// `dV/dr`.
    fn rho(&self) -> f64 {
        if self.is_degenerate() {
            let intrinsic = self.sign * (self.spot - self.strike * self.discount);
            return if intrinsic > 0.0 {
                self.sign * self.strike * self.time * self.discount
            } else {
                0.0
            };
        }
        self.sign * self.strike * self.time * self.discount * self.nd2
    }

    /// `dV/dT` — note the sign: theta as a desk quotes it is `-dV/dT`, the
    /// decay per unit of *elapsed* time, and this is the derivative with
    /// respect to time *remaining*, which is what the tape needs.
    fn d_time(&self) -> f64 {
        if self.is_degenerate() {
            let intrinsic = self.sign * (self.spot - self.strike * self.discount);
            return if intrinsic > 0.0 {
                self.sign * self.strike * self.rate * self.discount
            } else {
                0.0
            };
        }
        let carry = self.spot * normal_pdf(self.d1) * self.vol / (2.0 * self.time.sqrt());
        carry + self.sign * self.rate * self.strike * self.discount * self.nd2
    }
}

/// Which input a Greek differentiates against, in argument order.
const INPUT_COUNT: usize = 5;

/// Price European options elementwise.
///
/// All five inputs must have the same shape and dtype; broadcasting is the
/// caller's to arrange (with `expand`), because guessing which of five operands
/// is meant to be the scalar is how a portfolio silently gets priced against
/// the wrong strike.
pub fn black_scholes(
    spot: &Tensor,
    strike: &Tensor,
    rate: &Tensor,
    vol: &Tensor,
    time: &Tensor,
    kind: OptionKind,
) -> Result<Tensor> {
    let inputs = [spot, strike, rate, vol, time];
    let names = ["spot", "strike", "rate", "vol", "time"];
    for (tensor, name) in inputs.iter().zip(names) {
        if tensor.shape().dims() != spot.shape().dims() {
            return Err(MinitensorError::shape_mismatch_with_context(
                spot.shape().dims().to_vec(),
                tensor.shape().dims().to_vec(),
                format!("black_scholes: {name} must have the same shape as spot"),
            ));
        }
        if tensor.dtype() != spot.dtype() {
            return Err(MinitensorError::type_mismatch(
                format!("{:?}", spot.dtype()),
                format!("{:?}", tensor.dtype()),
            ));
        }
    }

    macro_rules! price {
        ($ty:ty, $slice:ident, $from_vec:path) => {{
            let read = |t: &Tensor| -> Result<Vec<$ty>> {
                Ok(t.data()
                    .$slice()
                    .ok_or_else(|| {
                        MinitensorError::internal_error("black_scholes: dtype mismatch")
                    })?
                    .to_vec())
            };
            let s = read(spot)?;
            let k = read(strike)?;
            let r = read(rate)?;
            let v = read(vol)?;
            let t = read(time)?;
            let mut out = vec![<$ty>::default(); s.len()];
            par_out_chunks(&mut out, PAR_CHUNK, &|start, block| {
                for (offset, slot) in block.iter_mut().enumerate() {
                    let i = start + offset;
                    *slot = Quote::new(
                        s[i] as f64,
                        k[i] as f64,
                        r[i] as f64,
                        v[i] as f64,
                        t[i] as f64,
                        kind,
                    )
                    .price() as $ty;
                }
            });
            $from_vec(out, spot.device())
        }};
    }

    let data = match spot.dtype() {
        DataType::Float32 => price!(f32, as_f32_slice, TensorData::from_vec_f32),
        DataType::Float64 => price!(f64, as_f64_slice, TensorData::from_vec_f64),
        other => {
            return Err(MinitensorError::invalid_operation(format!(
                "black_scholes is defined for float32 and float64 tensors, not {other:?}"
            )));
        }
    };

    let requires_grad =
        crate::autograd::is_grad_enabled() && inputs.iter().any(|t| t.requires_grad());
    let output = Tensor::new(
        Arc::new(data),
        Shape::new(spot.shape().dims().to_vec()),
        spot.dtype(),
        spot.device(),
        requires_grad,
    );

    if !requires_grad {
        return Ok(output);
    }

    let mut input_ids = Vec::with_capacity(INPUT_COUNT);
    let mut wanted = [false; INPUT_COUNT];
    for (index, tensor) in inputs.iter().enumerate() {
        wanted[index] = tensor.requires_grad();
        input_ids.push(tensor.id());
    }

    with_grad_fn(
        output,
        Arc::new(BlackScholesBackward {
            saved: inputs.map(|t| t.detach()),
            input_ids,
            wanted,
            kind,
        }),
    )
}

/// The Greeks, as the tape's backward pass.
///
/// One walk over the elements produces all five partials at once, because they
/// share `d1`, `d2` and the discount factor. Contrast the composed version:
/// each of the five would be a separate chain of gradient kernels, and `d1`
/// would be recomputed in every one of them.
struct BlackScholesBackward {
    saved: [Tensor; INPUT_COUNT],
    input_ids: Vec<TensorId>,
    wanted: [bool; INPUT_COUNT],
    kind: OptionKind,
}

impl GradientFunction for BlackScholesBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        gradients.reserve(INPUT_COUNT);

        macro_rules! greeks {
            ($ty:ty, $slice:ident, $from_vec:path) => {{
                fn read<'a>(
                    t: &'a Tensor,
                    get: fn(&'a TensorData) -> Option<&'a [$ty]>,
                ) -> Result<&'a [$ty]> {
                    get(t.data()).ok_or_else(|| {
                        MinitensorError::internal_error("black_scholes backward: dtype mismatch")
                    })
                }
                let g = read(grad_output, TensorData::$slice)?;
                let s = read(&self.saved[0], TensorData::$slice)?;
                let k = read(&self.saved[1], TensorData::$slice)?;
                let r = read(&self.saved[2], TensorData::$slice)?;
                let v = read(&self.saved[3], TensorData::$slice)?;
                let t = read(&self.saved[4], TensorData::$slice)?;

                for slot in 0..INPUT_COUNT {
                    if !self.wanted[slot] {
                        continue;
                    }
                    let mut out = vec![<$ty>::default(); g.len()];
                    par_out_chunks(&mut out, PAR_CHUNK, &|start, block| {
                        for (offset, cell) in block.iter_mut().enumerate() {
                            let i = start + offset;
                            let quote = Quote::new(
                                s[i] as f64,
                                k[i] as f64,
                                r[i] as f64,
                                v[i] as f64,
                                t[i] as f64,
                                self.kind,
                            );
                            let partial = match slot {
                                0 => quote.delta(),
                                1 => quote.dual_delta(),
                                2 => quote.rho(),
                                3 => quote.vega(),
                                _ => quote.d_time(),
                            };
                            *cell = (g[i] as f64 * partial) as $ty;
                        }
                    });
                    let tensor = Tensor::new(
                        Arc::new($from_vec(out, grad_output.device())),
                        Shape::new(grad_output.shape().dims().to_vec()),
                        grad_output.dtype(),
                        grad_output.device(),
                        false,
                    );
                    gradients.insert(self.input_ids[slot], tensor);
                }
            }};
        }

        match grad_output.dtype() {
            DataType::Float32 => greeks!(f32, as_f32_slice, TensorData::from_vec_f32),
            DataType::Float64 => greeks!(f64, as_f64_slice, TensorData::from_vec_f64),
            other => {
                return Err(MinitensorError::internal_error(format!(
                    "black_scholes backward reached dtype {other:?}"
                )));
            }
        }

        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }

    fn name(&self) -> &'static str {
        "BlackScholesBackward"
    }
}

/// The volatility that reproduces an observed price, by Newton's method on vega.
///
/// Inverting Black-Scholes has no closed form, so this iterates. Vega is the
/// derivative Newton needs and it is already computed alongside the price, so
/// each step costs one evaluation. Vega vanishes for deep in- and
/// out-of-the-money options, which is where Newton diverges; the iteration is
/// bracketed by a bisection fallback so a step that leaves the bracket is
/// replaced by its midpoint rather than shooting off.
///
/// Returns NaN where no volatility reproduces the price — a quote below
/// intrinsic value, or above the spot — because there is no answer to return
/// and a clamped bound would look like one.
pub fn implied_volatility(
    price: &Tensor,
    spot: &Tensor,
    strike: &Tensor,
    rate: &Tensor,
    time: &Tensor,
    kind: OptionKind,
    tolerance: f64,
    max_iterations: usize,
) -> Result<Tensor> {
    for tensor in [spot, strike, rate, time] {
        if tensor.shape().dims() != price.shape().dims() {
            return Err(MinitensorError::shape_mismatch(
                price.shape().dims().to_vec(),
                tensor.shape().dims().to_vec(),
            ));
        }
    }

    macro_rules! solve {
        ($ty:ty, $slice:ident, $from_vec:path) => {{
            fn read<'a>(
                t: &'a Tensor,
                get: fn(&'a TensorData) -> Option<&'a [$ty]>,
            ) -> Result<&'a [$ty]> {
                get(t.data()).ok_or_else(|| {
                    MinitensorError::internal_error("implied_volatility: dtype mismatch")
                })
            }
            let p = read(price, TensorData::$slice)?;
            let s = read(spot, TensorData::$slice)?;
            let k = read(strike, TensorData::$slice)?;
            let r = read(rate, TensorData::$slice)?;
            let t = read(time, TensorData::$slice)?;
            let mut out = vec![<$ty>::default(); p.len()];
            par_out_chunks(&mut out, PAR_CHUNK, &|start, block| {
                for (offset, slot) in block.iter_mut().enumerate() {
                    let i = start + offset;
                    *slot = solve_one(
                        p[i] as f64,
                        s[i] as f64,
                        k[i] as f64,
                        r[i] as f64,
                        t[i] as f64,
                        kind,
                        tolerance,
                        max_iterations,
                    ) as $ty;
                }
            });
            $from_vec(out, price.device())
        }};
    }

    let data = match price.dtype() {
        DataType::Float32 => solve!(f32, as_f32_slice, TensorData::from_vec_f32),
        DataType::Float64 => solve!(f64, as_f64_slice, TensorData::from_vec_f64),
        other => {
            return Err(MinitensorError::invalid_operation(format!(
                "implied_volatility is defined for float32 and float64 tensors, not {other:?}"
            )));
        }
    };

    // Not differentiable through the iteration, and it does not need to be:
    // by the implicit function theorem `dsigma/dprice = 1 / vega`, so a caller
    // who wants that gradient gets it exactly by evaluating `black_scholes` at
    // the recovered volatility — which is a recorded operation. Taping the
    // Newton steps instead would differentiate the solver, not the solution.
    Ok(Tensor::new(
        Arc::new(data),
        Shape::new(price.shape().dims().to_vec()),
        price.dtype(),
        price.device(),
        false,
    ))
}

/// One Newton solve, bracketed. See [`implied_volatility`].
fn solve_one(
    target: f64,
    spot: f64,
    strike: f64,
    rate: f64,
    time: f64,
    kind: OptionKind,
    tolerance: f64,
    max_iterations: usize,
) -> f64 {
    if !(target.is_finite() && spot > 0.0 && strike > 0.0 && time > 0.0) {
        return f64::NAN;
    }
    // No volatility can price below intrinsic or above the spot (for a call);
    // outside the bracket there is nothing to find.
    let mut low = 1e-9;
    let mut high = 10.0;
    let price_at = |vol: f64| Quote::new(spot, strike, rate, vol, time, kind).price();
    if target < price_at(low) - tolerance || target > price_at(high) + tolerance {
        return f64::NAN;
    }

    let mut vol = 0.2_f64; // The conventional starting guess.
    for _ in 0..max_iterations {
        let quote = Quote::new(spot, strike, rate, vol, time, kind);
        let error = quote.price() - target;
        if error.abs() <= tolerance {
            return vol;
        }
        if error > 0.0 {
            high = vol
        } else {
            low = vol
        }

        let vega = quote.vega();
        let step = if vega > 1e-12 {
            vol - error / vega
        } else {
            f64::NAN
        };
        // A Newton step that leaves the bracket is not an improvement; the
        // midpoint always is, and keeps the guaranteed convergence bisection
        // has and Newton does not.
        vol = if step.is_finite() && step > low && step < high {
            step
        } else {
            0.5 * (low + high)
        };
    }
    vol
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar(value: f64) -> Tensor {
        Tensor::new(
            Arc::new(TensorData::from_vec_f64(vec![value], crate::Device::cpu())),
            Shape::new(vec![1]),
            DataType::Float64,
            crate::Device::cpu(),
            false,
        )
    }

    fn price_of(s: f64, k: f64, r: f64, v: f64, t: f64, kind: OptionKind) -> f64 {
        black_scholes(
            &scalar(s),
            &scalar(k),
            &scalar(r),
            &scalar(v),
            &scalar(t),
            kind,
        )
        .unwrap()
        .data()
        .as_f64_slice()
        .unwrap()[0]
    }

    /// The textbook worked example: S=100, K=100, r=5%, sigma=20%, T=1.
    /// Call 10.4506, put 5.5735 — values every reference agrees on.
    #[test]
    fn matches_the_textbook_example() {
        let call = price_of(100.0, 100.0, 0.05, 0.2, 1.0, OptionKind::Call);
        let put = price_of(100.0, 100.0, 0.05, 0.2, 1.0, OptionKind::Put);
        assert!((call - 10.450_583_572).abs() < 1e-8, "call {call}");
        assert!((put - 5.573_526_022).abs() < 1e-8, "put {put}");
    }

    /// Put-call parity: `C - P = S - K e^{-rT}`, exactly, for any inputs.
    #[test]
    fn respects_put_call_parity() {
        for &(s, k, r, v, t) in &[
            (100.0, 90.0, 0.03, 0.25, 0.5),
            (42.0, 55.0, 0.01, 0.9, 2.0),
            (7.0, 7.0, 0.0, 0.05, 0.25),
        ] {
            let call = price_of(s, k, r, v, t, OptionKind::Call);
            let put = price_of(s, k, r, v, t, OptionKind::Put);
            let parity = s - k * (-r * t).exp();
            assert!(
                (call - put - parity).abs() < 1e-10,
                "C-P = {} but S-Ke^-rT = {parity}",
                call - put
            );
        }
    }

    #[test]
    fn an_expired_option_is_worth_its_intrinsic_value() {
        assert!((price_of(120.0, 100.0, 0.05, 0.2, 0.0, OptionKind::Call) - 20.0).abs() < 1e-12);
        assert_eq!(price_of(80.0, 100.0, 0.05, 0.2, 0.0, OptionKind::Call), 0.0);
        assert!((price_of(80.0, 100.0, 0.0, 0.2, 0.0, OptionKind::Put) - 20.0).abs() < 1e-12);
    }

    #[test]
    fn zero_volatility_is_the_discounted_forward() {
        let price = price_of(100.0, 90.0, 0.05, 0.0, 1.0, OptionKind::Call);
        let expected = 100.0 - 90.0 * (-0.05_f64).exp();
        assert!((price - expected).abs() < 1e-12, "{price} != {expected}");
    }

    /// Every Greek against a central difference of the price. This is what
    /// makes the closed forms trustworthy: they are checked against the
    /// function they claim to differentiate, not against a table.
    #[test]
    fn the_greeks_are_the_derivatives_they_claim_to_be() {
        let base = (100.0, 95.0, 0.04, 0.3, 0.75);
        for kind in [OptionKind::Call, OptionKind::Put] {
            let quote = Quote::new(base.0, base.1, base.2, base.3, base.4, kind);
            let h = 1e-5;
            let bump = |slot: usize, delta: f64| {
                let (mut s, mut k, mut r, mut v, mut t) = base;
                match slot {
                    0 => s += delta,
                    1 => k += delta,
                    2 => r += delta,
                    3 => v += delta,
                    _ => t += delta,
                }
                price_of(s, k, r, v, t, kind)
            };
            let analytic = [
                quote.delta(),
                quote.dual_delta(),
                quote.rho(),
                quote.vega(),
                quote.d_time(),
            ];
            for (slot, &want) in analytic.iter().enumerate() {
                let numeric = (bump(slot, h) - bump(slot, -h)) / (2.0 * h);
                assert!(
                    (numeric - want).abs() < 1e-5 * want.abs().max(1.0),
                    "{kind:?} partial {slot}: analytic {want}, numeric {numeric}"
                );
            }
        }
    }

    #[test]
    fn implied_volatility_recovers_the_volatility_that_made_the_price() {
        for &vol in &[0.05, 0.2, 0.75, 2.0] {
            let price = price_of(100.0, 110.0, 0.03, vol, 1.5, OptionKind::Call);
            let recovered = implied_volatility(
                &scalar(price),
                &scalar(100.0),
                &scalar(110.0),
                &scalar(0.03),
                &scalar(1.5),
                OptionKind::Call,
                1e-12,
                100,
            )
            .unwrap()
            .data()
            .as_f64_slice()
            .unwrap()[0];
            assert!((recovered - vol).abs() < 1e-6, "{recovered} != {vol}");
        }
    }

    #[test]
    fn an_unattainable_price_has_no_implied_volatility() {
        // Above the spot: no volatility prices a call there.
        let recovered = implied_volatility(
            &scalar(150.0),
            &scalar(100.0),
            &scalar(100.0),
            &scalar(0.0),
            &scalar(1.0),
            OptionKind::Call,
            1e-10,
            100,
        )
        .unwrap();
        assert!(recovered.data().as_f64_slice().unwrap()[0].is_nan());
    }

    #[test]
    fn mismatched_shapes_are_rejected() {
        let wide = Tensor::new(
            Arc::new(TensorData::from_vec_f64(
                vec![1.0, 2.0],
                crate::Device::cpu(),
            )),
            Shape::new(vec![2]),
            DataType::Float64,
            crate::Device::cpu(),
            false,
        );
        assert!(
            black_scholes(
                &scalar(1.0),
                &wide,
                &scalar(0.0),
                &scalar(0.2),
                &scalar(1.0),
                OptionKind::Call
            )
            .is_err()
        );
    }
}

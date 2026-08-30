// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Element-wise binary functions whose answer is a float.
//!
//! `atan2`, `hypot`, `copysign` and `xlogy` differ only in three kernels each
//! -- the function and its two partial derivatives -- so that is all each one
//! writes here. The promotion, the broadcast, the graph bookkeeping and the
//! backward pass are shared, and the derivative of an op sits next to the op.
//!
//! Promotion follows `/`: the answer is a float even when both operands are
//! integers, and float64 only when an operand already is.

use crate::{
    autograd::{FloatBinaryBackward, with_grad_fn},
    error::{MinitensorError, Result},
    ops::binary::{BinaryOpKind, coerce_and_broadcast},
    ops::kernels::broadcast_binary_arm,
    tensor::{DataType, Shape, Tensor, TensorData},
};
use std::sync::Arc;

/// One element-wise binary function, in both float widths.
///
/// A pair of function pointers rather than a generic, so that a single value
/// can stand for a forward kernel in one place and for a partial derivative in
/// [`FloatBinaryBackward`] in another.
pub type FloatBinaryKernel = (fn(f32, f32) -> f32, fn(f64, f64) -> f64);

/// Defines one [`FloatBinaryKernel`] from a single body, instantiated at both
/// widths. The body has to typecheck as `f32` and as `f64`, which rules out
/// naming either type inside it -- write `a.hypot(b)`, not `f64::hypot(a, b)`.
macro_rules! float_pair {
    ($(#[$meta:meta])* $name:ident, |$a:pat_param, $b:pat_param| $body:expr) => {
        $(#[$meta])*
        const $name: FloatBinaryKernel = {
            #[inline(always)]
            fn narrow($a: f32, $b: f32) -> f32 {
                $body
            }
            #[inline(always)]
            fn wide($a: f64, $b: f64) -> f64 {
                $body
            }
            (narrow, wide)
        };
    };
}

// --- atan2 -----------------------------------------------------------------

float_pair!(
    /// The angle in `(-pi, pi]` from the positive x-axis to `(x, y)`, which
    /// `(y / x).atan()` cannot give: it loses the quadrant, and divides by
    /// zero on the y-axis.
    ATAN2, |y, x| y.atan2(x)
);
float_pair!(
    /// `d/dy atan2(y, x) = x / (x^2 + y^2)`, grouped through `hypot` so the
    /// sum of squares cannot overflow for operands past the square root of
    /// the dtype's range.
    ATAN2_D_Y, |y, x| {
        let h = y.hypot(x);
        (x / h) / h
    }
);
float_pair!(
    /// `d/dx atan2(y, x) = -y / (x^2 + y^2)`, grouped as in `ATAN2_D_Y`.
    ATAN2_D_X, |y, x| {
        let h = y.hypot(x);
        (-y / h) / h
    }
);

// --- hypot -----------------------------------------------------------------

float_pair!(
    /// `sqrt(x^2 + y^2)` without forming either square, so it answers for
    /// operands whose squares would overflow or flush to zero.
    HYPOT, |x, y| x.hypot(y)
);
float_pair!(
    /// `d/dx hypot(x, y) = x / hypot(x, y)`. Undefined at the origin, where
    /// the surface has a cone point, and NaN says so.
    HYPOT_D_X, |x, y| x / x.hypot(y)
);
float_pair!(
    /// `d/dy hypot(x, y) = y / hypot(x, y)`.
    HYPOT_D_Y, |x, y| y / x.hypot(y)
);

// --- copysign --------------------------------------------------------------

float_pair!(
    /// The magnitude of `x` with the sign of `y`, signed zeros and NaN
    /// included -- `copysign(1, -0.0)` is `-1`.
    COPYSIGN, |x, y| x.copysign(y)
);
float_pair!(
    /// `d/dx copysign(x, y)` is `+1` where the sign is kept and `-1` where it
    /// is flipped. `signum` reads the sign bit, so `-0.0` counts as negative.
    COPYSIGN_D_X, |x, y| x.signum() * y.signum()
);
float_pair!(
    /// `copysign` does not vary with the *magnitude* of `y`, only with its
    /// sign bit, which no derivative can see.
    COPYSIGN_D_Y, |_x, _y| 0.0
);

// --- xlogy -----------------------------------------------------------------

float_pair!(
    /// `x * log(y)`, defined as `0` wherever `x` is zero.
    ///
    /// That is the limit the entropy and cross-entropy formulas need: the
    /// plain product gives `0 * -inf = NaN` at `x = 0, y = 0`, which is the
    /// case those formulas hit most. A NaN `y` still poisons the result, since
    /// then there is no limit to take.
    XLOGY, |x, y| {
        if x == 0.0 && !y.is_nan() {
            0.0
        } else {
            x * y.ln()
        }
    }
);
float_pair!(
    /// `d/dx (x log y) = log y`.
    XLOGY_D_X, |_x, y| y.ln()
);
float_pair!(
    /// `d/dy (x log y) = x / y`.
    XLOGY_D_Y, |x, y| x / y
);

// --- heaviside -------------------------------------------------------------

float_pair!(
    /// The step: `0` below zero, `1` above it, and `other` exactly at it --
    /// which is the whole reason it takes a second operand, since that value
    /// is the one convention never agrees on.
    ///
    /// A NaN input stays NaN; it is on neither side of the step.
    HEAVISIDE, |x, at_zero| {
        if x < 0.0 {
            0.0
        } else if x > 0.0 {
            1.0
        } else if x == 0.0 {
            at_zero
        } else {
            x
        }
    }
);
float_pair!(
    /// The step is flat wherever it is defined, so the gradient with respect
    /// to the input is zero everywhere the derivative exists -- and the
    /// jump at zero has no finite derivative to report.
    HEAVISIDE_D_X, |_x, _at_zero| 0.0
);
float_pair!(
    /// The second operand is the value taken at exactly zero, so it reaches
    /// the output there and nowhere else.
    HEAVISIDE_D_AT_ZERO, |x, _at_zero| if x == 0.0 { 1.0 } else { 0.0 }
);

// --- nextafter -------------------------------------------------------------

/// Defines `nextafter` at one width. The bit manipulation names the type on
/// every line, so this cannot go through [`float_pair!`], which requires a
/// body that typechecks at both.
macro_rules! next_after_fn {
    ($name:ident, $ty:ty) => {
        #[inline(always)]
        fn $name(from: $ty, towards: $ty) -> $ty {
            if from.is_nan() || towards.is_nan() {
                return <$ty>::NAN;
            }
            if from == towards {
                // Includes `0.0` against `-0.0`, where the answer is the
                // destination rather than a step in either direction.
                return towards;
            }
            if from == 0.0 {
                // Neither zero has a neighbour one bit away in the direction
                // asked for; the smallest subnormal does.
                let smallest = <$ty>::from_bits(1);
                return if towards > 0.0 { smallest } else { -smallest };
            }
            // Consecutive representable values are consecutive bit patterns
            // within a sign, and the pattern grows away from zero. So a step
            // towards a larger value adds one bit above zero and subtracts one
            // below it.
            let bits = from.to_bits();
            let stepped = if (from < towards) == (from > 0.0) {
                bits + 1
            } else {
                bits - 1
            };
            <$ty>::from_bits(stepped)
        }
    };
}

next_after_fn!(next_after_f32, f32);
next_after_fn!(next_after_f64, f64);

/// One representable value from `from` in the direction of `towards`.
const NEXTAFTER: FloatBinaryKernel = (next_after_f32, next_after_f64);
float_pair!(
    /// `nextafter(x, y)` differs from `x` by a single ulp, so as a
    /// real-valued function it is the identity and this is its slope. The
    /// step itself is below anything a derivative can see.
    NEXTAFTER_D_FROM, |_x, _y| 1.0
);
float_pair!(
    /// Only the *direction* of the second operand reaches the answer, and no
    /// derivative can see a direction.
    NEXTAFTER_D_TOWARDS, |_x, _y| 0.0
);

/// The shared body: promote, broadcast, run `forward`, and record `partials`
/// if either operand wants a gradient.
fn float_binary(
    lhs: &Tensor,
    rhs: &Tensor,
    forward: FloatBinaryKernel,
    partials: [FloatBinaryKernel; 2],
) -> Result<Tensor> {
    let (lhs_cast, rhs_cast, dtype, output_shape) =
        coerce_and_broadcast(lhs, rhs, BinaryOpKind::Div)?;
    let lhs_tensor = lhs_cast.into_owned();
    let rhs_tensor = rhs_cast.into_owned();

    let output_data = float_binary_data(&lhs_tensor, &rhs_tensor, dtype, &output_shape, forward)?;

    let requires_grad = lhs.requires_grad() || rhs.requires_grad();
    let output = Tensor::new(
        Arc::new(output_data),
        output_shape,
        dtype,
        lhs.device(),
        requires_grad,
    );

    if requires_grad {
        return with_grad_fn(
            output,
            Arc::new(FloatBinaryBackward {
                lhs: lhs_tensor.detach(),
                rhs: rhs_tensor.detach(),
                input_ids: [lhs.id(), rhs.id()],
                input_shapes: [lhs.shape().dims().to_vec(), rhs.shape().dims().to_vec()],
                input_requires_grad: [lhs.requires_grad(), rhs.requires_grad()],
                partials,
            }),
        );
    }

    Ok(output)
}

/// Applies `kernel` element-wise over two operands already promoted to
/// `dtype`, with broadcasting.
pub(crate) fn float_binary_data(
    lhs: &Tensor,
    rhs: &Tensor,
    dtype: DataType,
    output_shape: &Shape,
    kernel: FloatBinaryKernel,
) -> Result<TensorData> {
    Ok(match dtype {
        DataType::Float32 => {
            broadcast_binary_arm!(lhs, rhs, output_shape, as_f32_slice, "f32", kernel.0)
        }
        DataType::Float64 => {
            broadcast_binary_arm!(lhs, rhs, output_shape, as_f64_slice, "f64", kernel.1)
        }
        other => {
            return Err(MinitensorError::internal_error(format!(
                "float binary kernel reached with dtype {other}, which promotion cannot produce"
            )));
        }
    })
}

/// [`float_binary_data`] wrapped as a tensor, for the backward pass to
/// evaluate one partial derivative in a single walk over the operands.
pub(crate) fn float_binary_tensor(
    lhs: &Tensor,
    rhs: &Tensor,
    kernel: FloatBinaryKernel,
) -> Result<Tensor> {
    let output_shape = lhs.shape().broadcast_with(rhs.shape())?;
    let data = float_binary_data(lhs, rhs, lhs.dtype(), &output_shape, kernel)?;
    Ok(Tensor::new(
        Arc::new(data),
        output_shape,
        lhs.dtype(),
        lhs.device(),
        false,
    ))
}

/// Declares one public op from its forward kernel and its two partials.
macro_rules! float_binary_op {
    ($name:ident, $forward:ident, $d_lhs:ident, $d_rhs:ident, $doc:literal) => {
        #[doc = $doc]
        pub fn $name(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
            float_binary(lhs, rhs, $forward, [$d_lhs, $d_rhs])
        }
    };
}

float_binary_op!(
    atan2,
    ATAN2,
    ATAN2_D_Y,
    ATAN2_D_X,
    "The angle of the point `(other, input)` from the positive x-axis, in \
     `(-pi, pi]`, keeping the quadrant that `atan(input / other)` loses."
);
float_binary_op!(
    hypot,
    HYPOT,
    HYPOT_D_X,
    HYPOT_D_Y,
    "`sqrt(input^2 + other^2)`, computed without forming either square, so it \
     answers where the squares would overflow."
);
float_binary_op!(
    copysign,
    COPYSIGN,
    COPYSIGN_D_X,
    COPYSIGN_D_Y,
    "The magnitude of `input` with the sign of `other`, signed zeros included."
);
float_binary_op!(
    xlogy,
    XLOGY,
    XLOGY_D_X,
    XLOGY_D_Y,
    "`input * log(other)`, taken as `0` wherever `input` is zero rather than \
     as the `0 * -inf` the plain product would give."
);
float_binary_op!(
    heaviside,
    HEAVISIDE,
    HEAVISIDE_D_X,
    HEAVISIDE_D_AT_ZERO,
    "The unit step of `input`, taking the value `other` at exactly zero."
);
float_binary_op!(
    nextafter,
    NEXTAFTER,
    NEXTAFTER_D_FROM,
    NEXTAFTER_D_TOWARDS,
    "The next representable value after `input` in the direction of `other`."
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::vector;
    use crate::{autograd::backward_collect, device::Device, tensor::TensorData};

    fn f64_tensor(data: Vec<f64>) -> Tensor {
        vector(data)
    }

    fn i64_tensor(data: Vec<i64>) -> Tensor {
        vector(data)
    }

    fn wide(tensor: &Tensor) -> Vec<f64> {
        tensor.data().as_f64_slice().unwrap().to_vec()
    }

    /// Central differences of `op` with respect to one operand.
    fn numeric_grad(
        op: fn(&Tensor, &Tensor) -> Result<Tensor>,
        lhs: &[f64],
        rhs: &[f64],
        which: usize,
    ) -> Vec<f64> {
        let eps = 1e-6;
        let mut grads = vec![0.0; lhs.len()];
        for i in 0..lhs.len() {
            let mut plus = [lhs.to_vec(), rhs.to_vec()];
            let mut minus = [lhs.to_vec(), rhs.to_vec()];
            plus[which][i] += eps;
            minus[which][i] -= eps;
            let up = wide(&op(&f64_tensor(plus[0].clone()), &f64_tensor(plus[1].clone())).unwrap());
            let down =
                wide(&op(&f64_tensor(minus[0].clone()), &f64_tensor(minus[1].clone())).unwrap());
            grads[i] = (up[i] - down[i]) / (2.0 * eps);
        }
        grads
    }

    fn analytic_grads(
        op: fn(&Tensor, &Tensor) -> Result<Tensor>,
        lhs: &[f64],
        rhs: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        let a = f64_tensor(lhs.to_vec()).requires_grad_(true);
        let b = f64_tensor(rhs.to_vec()).requires_grad_(true);
        let out = op(&a, &b).unwrap();
        let seed = Tensor::ones(out.shape().clone(), out.dtype(), out.device(), false);
        let grads = backward_collect(&out, Some(seed)).unwrap();
        (
            wide(grads.get(&a.id()).unwrap()),
            wide(grads.get(&b.id()).unwrap()),
        )
    }

    #[test]
    fn atan2_covers_every_quadrant_and_the_axes() {
        let y = vec![1.0, 1.0, -1.0, -1.0, 0.0, -0.0, 1.0, -1.0];
        let x = vec![1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 0.0, 0.0];
        let got = wide(&atan2(&f64_tensor(y.clone()), &f64_tensor(x.clone())).unwrap());
        for (i, (&yi, &xi)) in y.iter().zip(&x).enumerate() {
            assert_eq!(got[i], yi.atan2(xi), "atan2({yi}, {xi})");
        }
        // The quadrant is the point of the op: atan(y/x) collapses the second
        // and fourth quadrants onto the first and third.
        assert!(got[1] > 0.0 && got[2] < 0.0);
    }

    #[test]
    fn hypot_survives_operands_whose_squares_overflow() {
        let big = 1e200_f64;
        let got = wide(&hypot(&f64_tensor(vec![big, 3.0]), &f64_tensor(vec![big, 4.0])).unwrap());
        assert!(got[0].is_finite(), "hypot overflowed: {}", got[0]);
        assert!((got[0] - big * std::f64::consts::SQRT_2).abs() < big * 1e-15);
        assert_eq!(got[1], 5.0);

        // ...and operands whose squares would flush to zero.
        let tiny = 1e-200_f64;
        let got = wide(&hypot(&f64_tensor(vec![tiny]), &f64_tensor(vec![tiny])).unwrap());
        assert!(got[0] > 0.0, "hypot underflowed to {}", got[0]);
    }

    #[test]
    fn copysign_reads_the_sign_bit_of_zero() {
        let x = vec![3.0, 3.0, -3.0, -3.0, 0.0];
        let y = vec![1.0, -0.0, 0.0, -1.0, -1.0];
        let got = wide(&copysign(&f64_tensor(x.clone()), &f64_tensor(y.clone())).unwrap());
        assert_eq!(got[..4], [3.0, -3.0, 3.0, -3.0]);
        assert!(got[4].is_sign_negative() && got[4] == 0.0);
    }

    #[test]
    fn xlogy_is_zero_where_the_plain_product_is_nan() {
        let x = vec![0.0, 0.0, 0.0, 2.0, 2.0];
        let y = vec![0.0, f64::INFINITY, 1.0, std::f64::consts::E, 0.0];
        let got = wide(&xlogy(&f64_tensor(x), &f64_tensor(y)).unwrap());
        assert_eq!(got[0], 0.0, "0 * log(0) must be the limit, not NaN");
        assert_eq!(got[1], 0.0);
        assert_eq!(got[2], 0.0);
        assert!((got[3] - 2.0).abs() < 1e-15);
        assert_eq!(got[4], f64::NEG_INFINITY);

        // A NaN second operand has no limit to take, so it survives.
        let got = wide(&xlogy(&f64_tensor(vec![0.0]), &f64_tensor(vec![f64::NAN])).unwrap());
        assert!(got[0].is_nan());
    }

    #[test]
    fn gradients_match_central_differences() {
        // Each op is sampled away from its own singular set: no origin for
        // `atan2` and `hypot`, and a strictly positive second operand for
        // `xlogy`, whose derivative is the logarithm of it.
        let lhs = [0.7, -1.3, 2.4, -0.4];
        let signed = [1.1, 0.6, -2.2, 3.5];
        let positive = [1.1, 0.6, 2.2, 3.5];

        for (name, op, rhs) in [
            (
                "atan2",
                atan2 as fn(&Tensor, &Tensor) -> Result<Tensor>,
                signed,
            ),
            ("hypot", hypot, signed),
            ("xlogy", xlogy, positive),
        ] {
            let (d_lhs, d_rhs) = analytic_grads(op, &lhs, &rhs);
            for (which, analytic) in [(0, d_lhs), (1, d_rhs)] {
                let numeric = numeric_grad(op, &lhs, &rhs, which);
                for (i, (&a, &n)) in analytic.iter().zip(&numeric).enumerate() {
                    assert!(
                        (a - n).abs() <= 1e-5 * (1.0 + n.abs()),
                        "{name} d/d{which} [{i}]: analytic {a}, numeric {n}"
                    );
                }
            }
        }

        // `copysign` moves with the *sign* of its second operand, which no
        // central difference can see, so its derivative there is zero
        // everywhere it exists and only the first operand's is checkable.
        let (d_x, d_y) = analytic_grads(copysign, &lhs, &signed);
        assert_eq!(d_y, vec![0.0; 4]);
        for (i, &g) in d_x.iter().enumerate() {
            let expected = lhs[i].signum() * signed[i].signum();
            assert_eq!(g, expected, "copysign d/dx [{i}]");
        }
    }

    #[test]
    fn integer_operands_promote_to_float_and_broadcast() {
        let y = i64_tensor(vec![3, 4]);
        let x = Tensor::new(
            Arc::new(TensorData::from_vec_i64(vec![4], Device::cpu())),
            Shape::new(vec![1]),
            DataType::Int64,
            Device::cpu(),
            false,
        );
        let got = hypot(&y, &x).unwrap();
        // `/` promotes two integers to float32, and so do these.
        assert_eq!(got.dtype(), DataType::Float32);
        assert_eq!(got.data().as_f32_slice().unwrap(), &[5.0, 5.656854]);

        // A float64 operand pulls the result up, as it does for `/`.
        let got = hypot(&y, &f64_tensor(vec![4.0])).unwrap();
        assert_eq!(got.dtype(), DataType::Float64);
        assert_eq!(got.shape().dims(), &[2]);
    }

    #[test]
    fn a_frozen_operand_gets_no_gradient() {
        let a = f64_tensor(vec![1.0, 2.0]).requires_grad_(true);
        let b = f64_tensor(vec![3.0, 4.0]);
        let out = atan2(&a, &b).unwrap();
        let seed = Tensor::ones(out.shape().clone(), out.dtype(), out.device(), false);
        let grads = backward_collect(&out, Some(seed)).unwrap();
        assert!(grads.contains_key(&a.id()));
        assert!(!grads.contains_key(&b.id()));
    }

    #[test]
    fn empty_and_mismatched_shapes() {
        let empty = f64_tensor(vec![]);
        assert_eq!(hypot(&empty, &empty).unwrap().shape().dims(), &[0]);

        let a = f64_tensor(vec![1.0, 2.0, 3.0]);
        let b = f64_tensor(vec![1.0, 2.0]);
        assert!(atan2(&a, &b).is_err());
    }

    #[test]
    fn heaviside_steps_at_zero_and_takes_its_second_operand_there() {
        let x = f64_tensor(vec![-2.0, -0.0, 0.0, 3.5, f64::NAN]);
        let at_zero = f64_tensor(vec![0.5, 0.5, 0.5, 0.5, 0.5]);
        let got = wide(&heaviside(&x, &at_zero).unwrap());
        assert_eq!(got[0], 0.0);
        // Negative zero is still zero, so it takes the given value too.
        assert_eq!(got[1], 0.5);
        assert_eq!(got[2], 0.5);
        assert_eq!(got[3], 1.0);
        assert!(got[4].is_nan(), "a NaN is on neither side of the step");
    }

    #[test]
    fn the_heaviside_gradient_reaches_only_the_value_at_zero() {
        let x = f64_tensor(vec![-1.0, 0.0, 1.0]).requires_grad_(true);
        let at_zero = f64_tensor(vec![0.25, 0.25, 0.25]).requires_grad_(true);
        let out = heaviside(&x, &at_zero).unwrap();
        let seed = Tensor::ones(out.shape().clone(), out.dtype(), out.device(), false);
        let grads = backward_collect(&out, Some(seed)).unwrap();
        // Flat wherever it is defined.
        assert_eq!(wide(grads.get(&x.id()).unwrap()), vec![0.0; 3]);
        // And the second operand is the output at exactly zero, nowhere else.
        assert_eq!(wide(grads.get(&at_zero.id()).unwrap()), vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn nextafter_moves_exactly_one_representable_value() {
        let from = f64_tensor(vec![1.0, 1.0, -1.0, -1.0]);
        let towards = f64_tensor(vec![2.0, 0.0, 0.0, -2.0]);
        let got = wide(&nextafter(&from, &towards).unwrap());

        // One ulp up and down from 1, and the same either side of -1.
        assert_eq!(got[0], f64::from_bits(1.0f64.to_bits() + 1));
        assert_eq!(got[1], f64::from_bits(1.0f64.to_bits() - 1));
        assert_eq!(got[2], -f64::from_bits(1.0f64.to_bits() - 1));
        assert_eq!(got[3], -f64::from_bits(1.0f64.to_bits() + 1));

        // Every step is strictly in the direction asked for, and nothing sits
        // between the answer and where it started.
        assert!(got[0] > 1.0 && got[1] < 1.0);
        assert!(got[2] > -1.0 && got[3] < -1.0);
    }

    #[test]
    fn nextafter_handles_the_values_with_no_neighbour_of_their_own() {
        let from = f64_tensor(vec![0.0, -0.0, 0.0, f64::MAX, f64::INFINITY, 3.0, f64::NAN]);
        let towards = f64_tensor(vec![1.0, 1.0, -1.0, f64::INFINITY, 0.0, 3.0, 1.0]);
        let got = wide(&nextafter(&from, &towards).unwrap());

        // Neither zero has a neighbour a bit away; the smallest subnormal does.
        assert_eq!(got[0], f64::from_bits(1));
        assert_eq!(got[1], f64::from_bits(1));
        assert_eq!(got[2], -f64::from_bits(1));
        // Past the largest finite value there is only infinity, and back from
        // infinity there is only the largest finite value.
        assert_eq!(got[3], f64::INFINITY);
        assert_eq!(got[4], f64::MAX);
        // Already there, so it stays.
        assert_eq!(got[5], 3.0);
        assert!(got[6].is_nan());
    }

    #[test]
    fn nextafter_agrees_with_itself_at_float32() {
        let from = Tensor::new(
            Arc::new(TensorData::from_vec_f32(
                vec![1.0, -1.0, 0.0],
                Device::cpu(),
            )),
            Shape::new(vec![3]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let towards = Tensor::new(
            Arc::new(TensorData::from_vec_f32(
                vec![2.0, -2.0, 1.0],
                Device::cpu(),
            )),
            Shape::new(vec![3]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let got = nextafter(&from, &towards).unwrap();
        assert_eq!(got.dtype(), DataType::Float32);
        let values = got.data().as_f32_slice().unwrap();
        assert_eq!(values[0], f32::from_bits(1.0f32.to_bits() + 1));
        assert_eq!(values[1], -f32::from_bits(1.0f32.to_bits() + 1));
        assert_eq!(values[2], f32::from_bits(1));
    }

    #[test]
    fn the_nextafter_gradient_is_the_identity_in_its_first_operand() {
        let from = f64_tensor(vec![1.0, 2.0]).requires_grad_(true);
        let towards = f64_tensor(vec![5.0, 5.0]).requires_grad_(true);
        let out = nextafter(&from, &towards).unwrap();
        let seed = Tensor::ones(out.shape().clone(), out.dtype(), out.device(), false);
        let grads = backward_collect(&out, Some(seed)).unwrap();
        // One ulp away from `from`, so as a real function this is the identity.
        assert_eq!(wide(grads.get(&from.id()).unwrap()), vec![1.0, 1.0]);
        // Only the direction of the second operand reaches the answer.
        assert_eq!(wide(grads.get(&towards.id()).unwrap()), vec![0.0, 0.0]);
    }
}

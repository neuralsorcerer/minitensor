// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Bit-level and truth-value element-wise operations.
//!
//! Two families live here because one is written in terms of the other. The
//! bitwise ops (`&`, `|`, `^`, `~`, `<<`, `>>`) act on the promoted integer or
//! boolean dtype and keep it. The logical ops accept any dtype, reduce each
//! operand to a truth value (`x != 0`, so NaN is true and -0.0 is false, as in
//! NumPy and PyTorch) and hand the resulting booleans to the bitwise op with
//! the same truth table -- so `logical_and` adds no kernel of its own.
//!
//! None of these are differentiable: their outputs are integers or booleans,
//! which no gradient flows through.

use crate::{
    error::{MinitensorError, Result},
    ops::binary::{BinaryOpKind, coerce_and_broadcast},
    ops::kernels::broadcast_binary_arm,
    ops::map::unary_map,
    tensor::{DataType, Tensor, TensorData},
};
use std::sync::Arc;

/// Reports a float reaching a kernel that promotion should already have
/// rejected. Reachable only if the promotion table and this file disagree.
fn unexpected_float(op: &str, dtype: DataType) -> MinitensorError {
    MinitensorError::internal_error(format!(
        "{op} kernel reached with dtype {dtype}, which promotion should have rejected"
    ))
}

/// `&`, `|` and `^`, over the promoted dtype. `$op` is applied at all three
/// integral dtypes, so it must be written generically (`|a, b| a $op b`).
macro_rules! bitwise_op {
    ($name:ident, $op:tt, $doc:literal) => {
        #[doc = $doc]
        pub fn $name(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
            let (lhs_cast, rhs_cast, dtype, output_shape) =
                coerce_and_broadcast(lhs, rhs, BinaryOpKind::Bitwise)?;
            let lhs_ref = lhs_cast.as_ref();
            let rhs_ref = rhs_cast.as_ref();

            let output_data = match dtype {
                DataType::Bool => broadcast_binary_arm!(
                    lhs_ref,
                    rhs_ref,
                    &output_shape,
                    as_bool_slice,
                    "bool",
                    |a, b| a $op b
                ),
                DataType::Int32 => broadcast_binary_arm!(
                    lhs_ref,
                    rhs_ref,
                    &output_shape,
                    as_i32_slice,
                    "i32",
                    |a, b| a $op b
                ),
                DataType::Int64 => broadcast_binary_arm!(
                    lhs_ref,
                    rhs_ref,
                    &output_shape,
                    as_i64_slice,
                    "i64",
                    |a, b| a $op b
                ),
                float => return Err(unexpected_float(stringify!($name), float)),
            };

            Ok(Tensor::new(
                Arc::new(output_data),
                output_shape,
                dtype,
                lhs.device(),
                false,
            ))
        }
    };
}

bitwise_op!(
    bitwise_and,
    &,
    "Element-wise bitwise AND, and logical AND for booleans."
);
bitwise_op!(
    bitwise_or,
    |,
    "Element-wise bitwise OR, and logical OR for booleans."
);
bitwise_op!(
    bitwise_xor,
    ^,
    "Element-wise bitwise XOR, and logical XOR for booleans."
);

/// Element-wise bitwise NOT (`~`): logical NOT for bool tensors, two's
/// complement NOT for integer tensors, rejected for floats.
pub fn bitwise_not(tensor: &Tensor) -> Result<Tensor> {
    /// Applies `!` for one dtype into a fresh buffer, parallel above
    /// `PAR_THRESHOLD`.
    macro_rules! not_arm {
        ($accessor:ident, $dtype:ident, $tyname:literal) => {{
            let input = tensor.data().$accessor().ok_or_else(|| {
                MinitensorError::internal_error(concat!(
                    "Failed to get ",
                    $tyname,
                    " slice from tensor"
                ))
            })?;
            TensorData::from_vec(unary_map(input, |i| !i), DataType::$dtype, tensor.device())
        }};
    }

    let output_data = match tensor.dtype() {
        DataType::Bool => not_arm!(as_bool_slice, Bool, "bool"),
        DataType::Int32 => not_arm!(as_i32_slice, Int32, "i32"),
        DataType::Int64 => not_arm!(as_i64_slice, Int64, "i64"),
        DataType::Float32 | DataType::Float64 => {
            return Err(MinitensorError::invalid_operation(
                "Bitwise NOT only supported for boolean and integer tensors",
            ));
        }
    };

    Ok(Tensor::new(
        Arc::new(output_data),
        tensor.shape().clone(),
        tensor.dtype(),
        tensor.device(),
        false,
    ))
}

/// Generates the shift pair for one integer width.
///
/// Rust's `<<`/`>>` are undefined past the operand's width -- a panic in a
/// build with overflow checks, a silently masked count without them, which is
/// the same tensor answering two ways depending on how the caller compiled.
/// These take the limit of the operation instead: everything has been shifted
/// out, leaving zero, or for an arithmetic right shift of a negative value, the
/// sign bit smeared to -1. Negative counts never arrive (see
/// [`check_shift_amounts`]).
macro_rules! shift_fns {
    ($shl:ident, $shr:ident, $ty:ty) => {
        #[inline(always)]
        fn $shl(value: $ty, amount: $ty) -> $ty {
            if amount >= <$ty>::BITS as $ty {
                0
            } else {
                value.wrapping_shl(amount as u32)
            }
        }

        #[inline(always)]
        fn $shr(value: $ty, amount: $ty) -> $ty {
            if amount >= <$ty>::BITS as $ty {
                // Arithmetic shift: negatives converge on -1, not 0.
                if value < 0 { -1 } else { 0 }
            } else {
                value.wrapping_shr(amount as u32)
            }
        }
    };
}

shift_fns!(shl_i32, shr_i32, i32);
shift_fns!(shl_i64, shr_i64, i64);

/// Rejects negative shift counts up front, so no element has to carry a
/// nonsense answer.
///
/// This walks the *un-broadcast* right-hand buffer, which is at most the size
/// of the output and usually far smaller, and it reads memory the kernel is
/// about to read anyway.
fn check_shift_amounts(amounts: &Tensor, op: &str) -> Result<()> {
    let negative = match amounts.dtype() {
        DataType::Int32 => amounts
            .data()
            .as_i32_slice()
            .ok_or_else(|| {
                MinitensorError::internal_error("Failed to get i32 slice from rhs tensor")
            })?
            .iter()
            .any(|&v| v < 0),
        DataType::Int64 => amounts
            .data()
            .as_i64_slice()
            .ok_or_else(|| {
                MinitensorError::internal_error("Failed to get i64 slice from rhs tensor")
            })?
            .iter()
            .any(|&v| v < 0),
        // A boolean shift count is 0 or 1, and a float never gets this far.
        _ => false,
    };

    if negative {
        return Err(MinitensorError::invalid_operation(format!(
            "{op} requires non-negative shift counts"
        )));
    }
    Ok(())
}

/// One element-wise op over the promoted *integer* dtype -- no boolean arm,
/// because neither shifting a truth value nor taking its divisor means
/// anything.
///
/// The four-argument form runs `$check` over the right-hand operand first, for
/// the ops that have a count they can refuse up front.
macro_rules! integer_op {
    ($name:ident, $i32_fn:ident, $i64_fn:ident, $doc:literal) => {
        integer_op!($name, $i32_fn, $i64_fn, $doc, |_rhs, _name| Ok(()));
    };
    ($name:ident, $i32_fn:ident, $i64_fn:ident, $doc:literal, $check:expr) => {
        #[doc = $doc]
        pub fn $name(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
            let (lhs_cast, rhs_cast, dtype, output_shape) =
                coerce_and_broadcast(lhs, rhs, BinaryOpKind::Shift)?;
            let lhs_ref = lhs_cast.as_ref();
            let rhs_ref = rhs_cast.as_ref();
            let check: fn(&Tensor, &str) -> Result<()> = $check;
            check(rhs_ref, stringify!($name))?;

            let output_data = match dtype {
                DataType::Int32 => broadcast_binary_arm!(
                    lhs_ref,
                    rhs_ref,
                    &output_shape,
                    as_i32_slice,
                    "i32",
                    $i32_fn
                ),
                DataType::Int64 => broadcast_binary_arm!(
                    lhs_ref,
                    rhs_ref,
                    &output_shape,
                    as_i64_slice,
                    "i64",
                    $i64_fn
                ),
                other => return Err(unexpected_float(stringify!($name), other)),
            };

            Ok(Tensor::new(
                Arc::new(output_data),
                output_shape,
                dtype,
                lhs.device(),
                false,
            ))
        }
    };
}

integer_op!(
    bitwise_left_shift,
    shl_i32,
    shl_i64,
    "Element-wise left shift. Counts at or past the dtype's width give 0.",
    check_shift_amounts
);
integer_op!(
    bitwise_right_shift,
    shr_i32,
    shr_i64,
    "Element-wise arithmetic right shift, preserving sign. Counts at or past \
     the dtype's width give 0 for non-negative values and -1 for negative ones.",
    check_shift_amounts
);

/// The greatest common divisor and the least common multiple, at both integer
/// widths.
///
/// Euclid's algorithm on the magnitudes, which is what makes the answer
/// non-negative for negative operands: a common divisor of `-12` and `8` is a
/// common divisor of `12` and `8`, and the convention every library follows is
/// to report the positive one.
macro_rules! divisor_fns {
    ($gcd:ident, $lcm:ident, $ty:ty) => {
        #[inline(always)]
        fn $gcd(a: $ty, b: $ty) -> $ty {
            // `unsigned_abs` rather than `abs`, so the most negative value --
            // whose magnitude has no representation as a signed integer --
            // does not overflow on the way in.
            let mut x = a.unsigned_abs();
            let mut y = b.unsigned_abs();
            while y != 0 {
                let remainder = x % y;
                x = y;
                y = remainder;
            }
            // The one magnitude that cannot come back is the most negative
            // value's, and it can only survive here as `gcd(MIN, 0)`; the
            // saturating cast reports the largest representable divisor rather
            // than wrapping to a negative one.
            if x > <$ty>::MAX as _ {
                <$ty>::MAX
            } else {
                x as $ty
            }
        }

        #[inline(always)]
        fn $lcm(a: $ty, b: $ty) -> $ty {
            let divisor = $gcd(a, b);
            if divisor == 0 {
                // Every multiple of zero is zero, so zero is the least of them.
                return 0;
            }
            // Divide before multiplying: the product of two operands can leave
            // the dtype even when their multiple does not.
            (a / divisor).wrapping_mul(b).wrapping_abs()
        }
    };
}

divisor_fns!(gcd_i32, lcm_i32, i32);
divisor_fns!(gcd_i64, lcm_i64, i64);

integer_op!(
    gcd,
    gcd_i32,
    gcd_i64,
    "Element-wise greatest common divisor, always non-negative. `gcd(x, 0)` \
     is the magnitude of `x`, since every integer divides zero."
);
integer_op!(
    lcm,
    lcm_i32,
    lcm_i64,
    "Element-wise least common multiple, always non-negative. `lcm(x, 0)` is \
     0, since zero is the least of the multiples of zero."
);

/// Reduces a tensor of any dtype to the truth values the logical ops operate
/// on: `x != 0`, which is exactly what a cast to `bool` already means here.
fn as_truth_values(tensor: &Tensor) -> Result<Tensor> {
    tensor.astype(DataType::Bool)
}

/// `logical_and`, `logical_or` and `logical_xor`: the bitwise op of the same
/// name applied to both operands' truth values.
macro_rules! logical_op {
    ($name:ident, $bitwise:ident, $doc:literal) => {
        #[doc = $doc]
        pub fn $name(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
            // The device check belongs before the casts so a mismatch is
            // reported as one rather than as a failure to allocate on the wrong
            // device.
            if lhs.device() != rhs.device() {
                return Err(MinitensorError::device_mismatch(
                    format!("{:?}", lhs.device()),
                    format!("{:?}", rhs.device()),
                ));
            }
            $bitwise(&as_truth_values(lhs)?, &as_truth_values(rhs)?)
        }
    };
}

logical_op!(
    logical_and,
    bitwise_and,
    "Element-wise logical AND over truth values, giving a boolean tensor."
);
logical_op!(
    logical_or,
    bitwise_or,
    "Element-wise logical OR over truth values, giving a boolean tensor."
);
logical_op!(
    logical_xor,
    bitwise_xor,
    "Element-wise logical XOR over truth values, giving a boolean tensor."
);

/// Element-wise logical NOT over truth values, giving a boolean tensor.
///
/// Unlike [`bitwise_not`] this accepts floats, because it asks whether each
/// element is zero rather than what its bits are.
pub fn logical_not(tensor: &Tensor) -> Result<Tensor> {
    bitwise_not(&as_truth_values(tensor)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        device::Device,
        tensor::{Shape, TensorData},
    };

    fn bool_tensor(data: Vec<bool>, shape: Vec<usize>) -> Tensor {
        Tensor::new(
            Arc::new(TensorData::from_vec_bool(data, Device::cpu())),
            Shape::new(shape),
            DataType::Bool,
            Device::cpu(),
            false,
        )
    }

    fn i32_tensor(data: Vec<i32>) -> Tensor {
        let len = data.len();
        Tensor::new(
            Arc::new(TensorData::from_vec_i32(data, Device::cpu())),
            Shape::new(vec![len]),
            DataType::Int32,
            Device::cpu(),
            false,
        )
    }

    fn i64_tensor(data: Vec<i64>) -> Tensor {
        let len = data.len();
        Tensor::new(
            Arc::new(TensorData::from_vec_i64(data, Device::cpu())),
            Shape::new(vec![len]),
            DataType::Int64,
            Device::cpu(),
            false,
        )
    }

    fn f32_tensor(data: Vec<f32>) -> Tensor {
        let len = data.len();
        Tensor::new(
            Arc::new(TensorData::from_vec_f32(data, Device::cpu())),
            Shape::new(vec![len]),
            DataType::Float32,
            Device::cpu(),
            false,
        )
    }

    fn bools(tensor: &Tensor) -> Vec<bool> {
        tensor.data().as_bool_slice().unwrap().to_vec()
    }

    fn i32s(tensor: &Tensor) -> Vec<i32> {
        tensor.data().as_i32_slice().unwrap().to_vec()
    }

    fn i64s(tensor: &Tensor) -> Vec<i64> {
        tensor.data().as_i64_slice().unwrap().to_vec()
    }

    #[test]
    fn test_bitwise_truth_tables_over_booleans() {
        let a = bool_tensor(vec![false, false, true, true], vec![4]);
        let b = bool_tensor(vec![false, true, false, true], vec![4]);

        assert_eq!(
            bools(&bitwise_and(&a, &b).unwrap()),
            vec![false, false, false, true]
        );
        assert_eq!(
            bools(&bitwise_or(&a, &b).unwrap()),
            vec![false, true, true, true]
        );
        assert_eq!(
            bools(&bitwise_xor(&a, &b).unwrap()),
            vec![false, true, true, false]
        );
        assert_eq!(
            bools(&bitwise_not(&a).unwrap()),
            vec![true, true, false, false]
        );
    }

    #[test]
    fn test_bitwise_over_integers_including_negatives() {
        let a = i32_tensor(vec![0b1100, -1, 5]);
        let b = i32_tensor(vec![0b1010, 0b0110, -1]);

        assert_eq!(i32s(&bitwise_and(&a, &b).unwrap()), vec![0b1000, 0b0110, 5]);
        assert_eq!(i32s(&bitwise_or(&a, &b).unwrap()), vec![0b1110, -1, -1]);
        assert_eq!(
            i32s(&bitwise_xor(&a, &b).unwrap()),
            vec![0b0110, !0b0110, -6]
        );
    }

    #[test]
    fn test_bitwise_promotes_and_broadcasts() {
        // A boolean paired with an integer promotes to that integer dtype, and
        // the shapes broadcast like every other binary op.
        let mask = bool_tensor(vec![true, false], vec![2, 1]);
        let values = i64_tensor(vec![7, 8]);
        let result = bitwise_and(&mask, &values).unwrap();

        assert_eq!(result.dtype(), DataType::Int64);
        assert_eq!(result.shape().dims(), &[2, 2]);
        assert_eq!(i64s(&result), vec![1, 0, 0, 0]);
    }

    #[test]
    fn test_bitwise_rejects_floats() {
        let a = f32_tensor(vec![1.0, 2.0]);
        let b = i32_tensor(vec![1, 2]);
        for result in [
            bitwise_and(&a, &b),
            bitwise_or(&a, &b),
            bitwise_xor(&b, &a),
            bitwise_left_shift(&a, &b),
            bitwise_right_shift(&b, &a),
        ] {
            assert!(result.is_err());
        }
        assert!(bitwise_not(&a).is_err());
    }

    #[test]
    fn test_shifts_agree_with_multiplication_and_division() {
        let values = i32_tensor(vec![1, 3, -3, 0]);
        let by_two = i32_tensor(vec![2, 2, 2, 2]);

        assert_eq!(
            i32s(&bitwise_left_shift(&values, &by_two).unwrap()),
            vec![4, 12, -12, 0]
        );
        // An arithmetic right shift floors, so -3 >> 2 is -1, not 0.
        assert_eq!(
            i32s(&bitwise_right_shift(&values, &by_two).unwrap()),
            vec![0, 0, -1, 0]
        );
    }

    #[test]
    fn test_shift_counts_at_and_past_the_dtype_width() {
        // Rust's own `<<` is undefined here: a panic under overflow checks, a
        // masked count without them. Both widths must converge instead.
        let values = i32_tensor(vec![1, -1, 6]);
        let wide = i32_tensor(vec![32, 32, 99]);
        assert_eq!(
            i32s(&bitwise_left_shift(&values, &wide).unwrap()),
            vec![0, 0, 0]
        );
        assert_eq!(
            i32s(&bitwise_right_shift(&values, &wide).unwrap()),
            vec![0, -1, 0]
        );

        let values64 = i64_tensor(vec![1, -1]);
        let wide64 = i64_tensor(vec![64, 64]);
        assert_eq!(
            i64s(&bitwise_left_shift(&values64, &wide64).unwrap()),
            vec![0, 0]
        );
        assert_eq!(
            i64s(&bitwise_right_shift(&values64, &wide64).unwrap()),
            vec![0, -1]
        );
        // 63 is the last count that still moves a bit rather than clearing it.
        let edge = i64_tensor(vec![1, 1]);
        let sixty_three = i64_tensor(vec![63, 63]);
        assert_eq!(
            i64s(&bitwise_left_shift(&edge, &sixty_three).unwrap()),
            vec![i64::MIN, i64::MIN]
        );
    }

    #[test]
    fn test_negative_shift_counts_are_rejected() {
        let values = i32_tensor(vec![1, 2]);
        let counts = i32_tensor(vec![1, -1]);
        assert!(bitwise_left_shift(&values, &counts).is_err());
        assert!(bitwise_right_shift(&values, &counts).is_err());
    }

    #[test]
    fn test_shift_rejects_two_booleans_but_allows_a_mixed_pair() {
        let mask = bool_tensor(vec![true, false], vec![2]);
        assert!(bitwise_left_shift(&mask, &mask).is_err());

        let counts = i64_tensor(vec![3, 3]);
        let shifted = bitwise_left_shift(&mask, &counts).unwrap();
        assert_eq!(shifted.dtype(), DataType::Int64);
        assert_eq!(i64s(&shifted), vec![8, 0]);
    }

    #[test]
    fn test_logical_ops_reduce_any_dtype_to_truth_values() {
        // Every non-zero float is true, NaN included; -0.0 is false.
        let a = f32_tensor(vec![0.0, -0.0, 2.5, f32::NAN]);
        let b = f32_tensor(vec![1.0, 0.0, 0.0, 1.0]);

        assert_eq!(
            bools(&logical_and(&a, &b).unwrap()),
            vec![false, false, false, true]
        );
        assert_eq!(
            bools(&logical_or(&a, &b).unwrap()),
            vec![true, false, true, true]
        );
        assert_eq!(
            bools(&logical_xor(&a, &b).unwrap()),
            vec![true, false, true, false]
        );
        assert_eq!(
            bools(&logical_not(&a).unwrap()),
            vec![true, true, false, false]
        );
    }

    #[test]
    fn test_logical_ops_broadcast_and_mix_dtypes() {
        let mask = bool_tensor(vec![true, false], vec![2, 1]);
        let values = i32_tensor(vec![0, 5]);
        let result = logical_and(&mask, &values).unwrap();

        assert_eq!(result.dtype(), DataType::Bool);
        assert_eq!(result.shape().dims(), &[2, 2]);
        assert_eq!(bools(&result), vec![false, true, false, false]);
    }

    #[test]
    fn test_logical_and_bitwise_agree_on_booleans() {
        // The logical ops are defined as the bitwise ones over truth values, so
        // on boolean input the two must be indistinguishable.
        let a = bool_tensor(vec![false, false, true, true], vec![4]);
        let b = bool_tensor(vec![false, true, false, true], vec![4]);

        assert_eq!(
            bools(&logical_and(&a, &b).unwrap()),
            bools(&bitwise_and(&a, &b).unwrap())
        );
        assert_eq!(
            bools(&logical_or(&a, &b).unwrap()),
            bools(&bitwise_or(&a, &b).unwrap())
        );
        assert_eq!(
            bools(&logical_xor(&a, &b).unwrap()),
            bools(&bitwise_xor(&a, &b).unwrap())
        );
        assert_eq!(
            bools(&logical_not(&a).unwrap()),
            bools(&bitwise_not(&a).unwrap())
        );
    }

    #[test]
    fn test_empty_and_mismatched_shapes() {
        let empty = i32_tensor(vec![]);
        let result = bitwise_and(&empty, &empty).unwrap();
        assert_eq!(result.shape().dims(), &[0]);
        assert!(i32s(&result).is_empty());

        let a = i32_tensor(vec![1, 2, 3]);
        let b = i32_tensor(vec![1, 2]);
        assert!(bitwise_or(&a, &b).is_err());
        assert!(logical_or(&a, &b).is_err());
    }

    #[test]
    fn gcd_and_lcm_match_euclid_at_both_widths() {
        // Every sign pairing, plus the zeros, at i64 and i32.
        let pairs: [(i64, i64); 10] = [
            (12, 8),
            (-12, 8),
            (12, -8),
            (-12, -8),
            (0, 5),
            (5, 0),
            (0, 0),
            (17, 5),
            (270, 192),
            (1, 1),
        ];
        let expected_gcd = [4i64, 4, 4, 4, 5, 5, 0, 1, 6, 1];
        let expected_lcm = [24i64, 24, 24, 24, 0, 0, 0, 85, 8640, 1];

        let lhs: Vec<i64> = pairs.iter().map(|p| p.0).collect();
        let rhs: Vec<i64> = pairs.iter().map(|p| p.1).collect();
        let a = i64_tensor(lhs.clone());
        let b = i64_tensor(rhs.clone());
        assert_eq!(
            gcd(&a, &b).unwrap().data().as_i64_slice().unwrap(),
            &expected_gcd
        );
        assert_eq!(
            lcm(&a, &b).unwrap().data().as_i64_slice().unwrap(),
            &expected_lcm
        );

        let narrow_lhs: Vec<i32> = lhs.iter().map(|&v| v as i32).collect();
        let narrow_rhs: Vec<i32> = rhs.iter().map(|&v| v as i32).collect();
        let a = i32_tensor(narrow_lhs);
        let b = i32_tensor(narrow_rhs);
        let expected: Vec<i32> = expected_gcd.iter().map(|&v| v as i32).collect();
        assert_eq!(
            gcd(&a, &b).unwrap().data().as_i32_slice().unwrap(),
            &expected[..]
        );
    }

    #[test]
    fn gcd_survives_the_value_with_no_positive_magnitude() {
        // `i64::MIN.abs()` overflows, so the magnitudes are taken unsigned.
        let a = i64_tensor(vec![i64::MIN, i64::MIN, i64::MIN]);
        let b = i64_tensor(vec![0, 2, i64::MIN]);
        let got = gcd(&a, &b).unwrap();
        let values = got.data().as_i64_slice().unwrap();
        // gcd(MIN, 0) is |MIN|, which no i64 can hold; the largest one is
        // reported instead of a wrapped negative.
        assert_eq!(values[0], i64::MAX);
        assert_eq!(values[1], 2);
        assert_eq!(values[2], i64::MAX);
        assert!(
            values.iter().all(|&v| v >= 0),
            "a divisor is never negative"
        );
    }

    #[test]
    fn gcd_and_lcm_broadcast_and_promote_like_the_shifts() {
        let a = i32_tensor(vec![12, 18]);
        let b = i64_tensor(vec![8]);
        let got = gcd(&a, &b).unwrap();
        assert_eq!(got.dtype(), DataType::Int64);
        assert_eq!(got.data().as_i64_slice().unwrap(), &[4, 2]);
    }

    #[test]
    fn gcd_and_lcm_refuse_floats_and_booleans() {
        let floats = Tensor::new(
            Arc::new(TensorData::from_vec_f64(vec![1.0, 2.0], Device::cpu())),
            Shape::new(vec![2]),
            DataType::Float64,
            Device::cpu(),
            false,
        );
        let booleans = Tensor::new(
            Arc::new(TensorData::from_vec_bool(vec![true, false], Device::cpu())),
            Shape::new(vec![2]),
            DataType::Bool,
            Device::cpu(),
            false,
        );
        assert!(gcd(&floats, &floats).is_err());
        assert!(lcm(&floats, &floats).is_err());
        // Two truth values have no divisors, the same reason they have no bits
        // to shift.
        assert!(gcd(&booleans, &booleans).is_err());
    }

    #[test]
    fn lcm_divides_before_multiplying() {
        // The product of these two leaves i64; their least common multiple
        // does not, and only the ordering of the arithmetic keeps it.
        let a = i64_tensor(vec![4_000_000_000]);
        let b = i64_tensor(vec![6_000_000_000]);
        assert_eq!(
            lcm(&a, &b).unwrap().data().as_i64_slice().unwrap(),
            &[12_000_000_000]
        );
    }
}

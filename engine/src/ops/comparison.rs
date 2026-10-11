// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    error::{MinitensorError, Result},
    ops::binary::{BinaryOpKind, coerce_binary_operands},
    ops::kernels::{broadcast_binary_arm, broadcast_binary_data},
    tensor::{DataType, Shape, Tensor, TensorData},
};
use std::cmp::Ordering;
use std::sync::Arc;

/// Dispatches a comparison over the promoted dtype, producing a boolean
/// tensor. The five arms differ only in which slice accessor they name; the
/// broadcasting walk and the `TensorData` wrapping come from
/// [`crate::ops::kernels::broadcast_binary_arm`], which this file used to
/// carry its own bool-specialized copy of.
macro_rules! cmp_dispatch {
    ($lhs:expr, $rhs:expr, $dtype:expr, $out_shape:expr, $op:expr) => {{
        let lhs = $lhs;
        let rhs = $rhs;
        let out_shape = $out_shape;
        match $dtype {
            DataType::Float32 => {
                broadcast_binary_arm!(lhs, rhs, out_shape, as_f32_slice, "f32", $op)
            }
            DataType::Float64 => {
                broadcast_binary_arm!(lhs, rhs, out_shape, as_f64_slice, "f64", $op)
            }
            DataType::Int32 => {
                broadcast_binary_arm!(lhs, rhs, out_shape, as_i32_slice, "i32", $op)
            }
            DataType::Int64 => {
                broadcast_binary_arm!(lhs, rhs, out_shape, as_i64_slice, "i64", $op)
            }
            DataType::Bool => {
                broadcast_binary_arm!(lhs, rhs, out_shape, as_bool_slice, "bool", $op)
            }
        }
    }};
}

/// How an integer compares with a float, exactly; `None` when the float is NaN.
///
/// Promoting the pair to a common float first is what answers wrongly: an
/// int32 `20_000_001` and `20_000_000.0` meet in float32 as one number and
/// compare equal, and an int64 above 2^53 does the same in float64. Every
/// float at or beyond `+-2^63` lies outside the int64 range, and every one
/// inside it has a floor that is an int64 exactly -- so the integer is compared
/// with that floor, and only a tie needs to know whether the float had a
/// fraction above it.
#[inline(always)]
fn int_against_float(int: i64, float: f64) -> Option<Ordering> {
    // 2^63, which is exactly representable; every int64 is below it.
    const BOUND: f64 = 9_223_372_036_854_775_808.0;
    if float.is_nan() {
        return None;
    }
    if float >= BOUND {
        return Some(Ordering::Less);
    }
    if float < -BOUND {
        return Some(Ordering::Greater);
    }
    let floor = float.floor();
    Some(match int.cmp(&(floor as i64)) {
        Ordering::Equal if float != floor => Ordering::Less,
        other => other,
    })
}

/// A comparison between an integer tensor and a float one, made exactly; see
/// [`int_against_float`]. `None` when the operands are not such a pair.
///
/// `holds` says whether an ordering satisfies the comparison, read with the
/// operands in their original order; a NaN satisfies only `!=`.
fn mixed_comparison(
    lhs: &Tensor,
    rhs: &Tensor,
    holds: fn(Ordering) -> bool,
    nan_holds: bool,
) -> Result<Option<(TensorData, Shape)>> {
    let is_int = |t: &Tensor| matches!(t.dtype(), DataType::Int32 | DataType::Int64);
    let is_float = |t: &Tensor| matches!(t.dtype(), DataType::Float32 | DataType::Float64);
    let (int_side, float_side, int_on_left) = if is_int(lhs) && is_float(rhs) {
        (lhs, rhs, true)
    } else if is_float(lhs) && is_int(rhs) {
        (rhs, lhs, false)
    } else {
        return Ok(None);
    };
    // Both widenings are exact: every int32 is an int64, every float32 a float64.
    let ints = int_side.astype(DataType::Int64)?;
    let floats = float_side.astype(DataType::Float64)?;
    let int_data = ints.data().as_i64_slice().ok_or_else(|| {
        MinitensorError::internal_error("comparison: integer operand did not widen to int64")
    })?;
    let float_data = floats.data().as_f64_slice().ok_or_else(|| {
        MinitensorError::internal_error("comparison: float operand did not widen to float64")
    })?;
    let output_shape = lhs.shape().broadcast_with(rhs.shape())?;
    let answer = move |ordering: Option<Ordering>| match ordering {
        Some(ordering) => holds(ordering),
        None => nan_holds,
    };
    let data = if int_on_left {
        broadcast_binary_data(
            &ints,
            &floats,
            int_data,
            float_data,
            &output_shape,
            |a, b| answer(int_against_float(a, b)),
        )?
    } else {
        broadcast_binary_data(
            &floats,
            &ints,
            float_data,
            int_data,
            &output_shape,
            |b, a| answer(int_against_float(a, b).map(Ordering::reverse)),
        )?
    };
    Ok(Some((data, output_shape)))
}

macro_rules! cmp_op {
    ($fn_name:ident, $op:tt) => {
        pub fn $fn_name(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
            if lhs.device() != rhs.device() {
                return Err(MinitensorError::device_mismatch(
                    format!("{:?}", lhs.device()),
                    format!("{:?}", rhs.device()),
                ));
            }
            if let Some((data, shape)) = mixed_comparison(
                lhs,
                rhs,
                |ordering| ordering $op Ordering::Equal,
                stringify!($op) == "!=",
            )? {
                return Ok(Tensor::new(
                    Arc::new(data),
                    shape,
                    DataType::Bool,
                    lhs.device(),
                    false,
                ));
            }
            let (lhs_cast, rhs_cast, common_dtype) =
                coerce_binary_operands(lhs, rhs, BinaryOpKind::Add)?;

            let lhs_ref = lhs_cast.as_ref();
            let rhs_ref = rhs_cast.as_ref();

            let output_shape = lhs_ref.shape().broadcast_with(rhs_ref.shape())?;
            let output_data =
                cmp_dispatch!(lhs_ref, rhs_ref, common_dtype, &output_shape, |a, b| a $op b);

            Ok(Tensor::new(
                Arc::new(output_data),
                output_shape,
                DataType::Bool,
                lhs.device(),
                false,
            ))
        }
    };
}

#[inline(always)]
fn isclose_f32(a: f32, b: f32, rtol: f32, atol: f32, equal_nan: bool) -> bool {
    if a == b {
        return true;
    }
    if equal_nan && a.is_nan() && b.is_nan() {
        return true;
    }
    if !a.is_finite() || !b.is_finite() {
        return false;
    }
    (a - b).abs() <= atol + rtol * b.abs()
}

#[inline(always)]
fn isclose_f64(a: f64, b: f64, rtol: f64, atol: f64, equal_nan: bool) -> bool {
    if a == b {
        return true;
    }
    if equal_nan && a.is_nan() && b.is_nan() {
        return true;
    }
    if !a.is_finite() || !b.is_finite() {
        return false;
    }
    (a - b).abs() <= atol + rtol * b.abs()
}

pub fn isclose(
    lhs: &Tensor,
    rhs: &Tensor,
    rtol: f64,
    atol: f64,
    equal_nan: bool,
) -> Result<Tensor> {
    if !rtol.is_finite() || !atol.is_finite() || rtol < 0.0 || atol < 0.0 {
        return Err(MinitensorError::invalid_operation(
            "rtol and atol must be non-negative, finite values",
        ));
    }
    if lhs.device() != rhs.device() {
        return Err(MinitensorError::device_mismatch(
            format!("{:?}", lhs.device()),
            format!("{:?}", rhs.device()),
        ));
    }

    // An integer against a float is measured in float64, which holds every
    // int32 and every float32 exactly. Their common dtype is the float's, and
    // in float32 an int32 `20_000_001` was within no tolerance at all of
    // `20_000_000.0` -- it had become it.
    let integer = |t: &Tensor| matches!(t.dtype(), DataType::Int32 | DataType::Int64);
    let float = |t: &Tensor| matches!(t.dtype(), DataType::Float32 | DataType::Float64);
    if (integer(lhs) && float(rhs)) || (float(lhs) && integer(rhs)) {
        return isclose(
            &lhs.astype(DataType::Float64)?,
            &rhs.astype(DataType::Float64)?,
            rtol,
            atol,
            equal_nan,
        );
    }
    let (lhs_cast, rhs_cast, common_dtype) = coerce_binary_operands(lhs, rhs, BinaryOpKind::Add)?;
    let lhs_ref = lhs_cast.as_ref();
    let rhs_ref = rhs_cast.as_ref();
    let output_shape = lhs_ref.shape().broadcast_with(rhs_ref.shape())?;
    // `rtol`/`atol` only mean anything for the float dtypes; an exact-integer or
    // boolean comparison is what "close" reduces to elsewhere.
    let output_data = match common_dtype {
        DataType::Float32 => broadcast_binary_arm!(
            lhs_ref,
            rhs_ref,
            &output_shape,
            as_f32_slice,
            "f32",
            |a, b| isclose_f32(a, b, rtol as f32, atol as f32, equal_nan)
        ),
        DataType::Float64 => broadcast_binary_arm!(
            lhs_ref,
            rhs_ref,
            &output_shape,
            as_f64_slice,
            "f64",
            |a, b| isclose_f64(a, b, rtol, atol, equal_nan)
        ),
        dtype => cmp_dispatch!(lhs_ref, rhs_ref, dtype, &output_shape, |a, b| a == b),
    };

    Ok(Tensor::new(
        Arc::new(output_data),
        output_shape,
        DataType::Bool,
        lhs.device(),
        false,
    ))
}

cmp_op!(eq, ==);
cmp_op!(ne, !=);
// Ordered comparisons apply to booleans as well, with `false < true`. Refusing
// them was inconsistent with everything around it: `eq`/`ne` accepted booleans,
// `minimum`/`maximum` already ordered them, and a boolean compared against a
// number promoted and compared fine -- only two boolean operands were rejected.
cmp_op!(lt, <);
cmp_op!(le, <=);
cmp_op!(gt, >);
cmp_op!(ge, >=);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        device::Device,
        tensor::{DataType, Shape, TensorData},
    };

    fn tensor_from_vec_f32(data: Vec<f32>) -> Tensor {
        let shape = Shape::new(vec![data.len()]);
        let data = TensorData::from_vec_f32(data, Device::cpu());
        Tensor::new(
            Arc::new(data),
            shape,
            DataType::Float32,
            Device::cpu(),
            false,
        )
    }

    fn tensor_from_vec_f32_shape(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        let shape = Shape::new(shape);
        let data = TensorData::from_vec_f32(data, Device::cpu());
        Tensor::new(
            Arc::new(data),
            shape,
            DataType::Float32,
            Device::cpu(),
            false,
        )
    }

    fn tensor_from_vec_i32(data: Vec<i32>) -> Tensor {
        let shape = Shape::new(vec![data.len()]);
        let data = TensorData::from_vec_i32(data, Device::cpu());
        Tensor::new(Arc::new(data), shape, DataType::Int32, Device::cpu(), false)
    }

    fn tensor_from_vec_bool(data: Vec<bool>) -> Tensor {
        let shape = Shape::new(vec![data.len()]);
        let data = TensorData::from_vec_bool(data, Device::cpu());
        Tensor::new(Arc::new(data), shape, DataType::Bool, Device::cpu(), false)
    }

    #[test]
    fn test_eq_basic() {
        let a = tensor_from_vec_f32(vec![1.0, 2.0, 3.0]);
        let b = tensor_from_vec_f32(vec![1.0, 0.0, 3.0]);
        let result = eq(&a, &b).unwrap();
        let slice = result.data().as_bool_slice().unwrap();
        assert_eq!(slice, &[true, false, true]);
    }

    #[test]
    fn test_eq_broadcasting() {
        let a = tensor_from_vec_f32_shape(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = tensor_from_vec_f32_shape(vec![1.0, 4.0], vec![1, 2]);
        let result = eq(&a, &b).unwrap();
        let slice = result.data().as_bool_slice().unwrap();
        assert_eq!(slice, &[true, false, false, true]);
        assert_eq!(result.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_ordered_comparisons_order_booleans() {
        // `false < true` is the natural order. `eq`/`ne` already
        // accepted booleans and `minimum`/`maximum` already ordered them, so
        // rejecting `lt` here was the odd one out.
        let a = tensor_from_vec_bool(vec![true, false, true, false]);
        let b = tensor_from_vec_bool(vec![false, true, true, false]);

        assert_eq!(
            lt(&a, &b).unwrap().data().as_bool_slice().unwrap(),
            &[false, true, false, false]
        );
        assert_eq!(
            le(&a, &b).unwrap().data().as_bool_slice().unwrap(),
            &[false, true, true, true]
        );
        assert_eq!(
            gt(&a, &b).unwrap().data().as_bool_slice().unwrap(),
            &[true, false, false, false]
        );
        assert_eq!(
            ge(&a, &b).unwrap().data().as_bool_slice().unwrap(),
            &[true, false, true, true]
        );
    }

    #[test]
    fn test_gt_shape_mismatch_error() {
        let a = tensor_from_vec_f32(vec![1.0, 2.0, 3.0]);
        let b = tensor_from_vec_f32(vec![1.0, 2.0]);
        assert!(gt(&a, &b).is_err());
    }

    /// The ordering against exact arithmetic, at the edges of both ranges:
    /// either side of an integer, the float64 rounding gap past 2^53, the
    /// int64 limits and the floats just outside them, and the non-finite.
    #[test]
    fn int_against_float_is_exact() {
        use Ordering::*;
        let cases: &[(i64, f64, Option<Ordering>)] = &[
            (3, 3.0, Some(Equal)),
            (3, 3.5, Some(Less)),
            (4, 3.5, Some(Greater)),
            (-3, -3.5, Some(Greater)),
            (-4, -3.5, Some(Less)),
            (0, -0.0, Some(Equal)),
            ((1 << 53) + 1, (1u64 << 53) as f64, Some(Greater)),
            (1 << 53, (1u64 << 53) as f64, Some(Equal)),
            (i64::MAX, 9_223_372_036_854_775_808.0, Some(Less)),
            (i64::MIN, -9_223_372_036_854_775_808.0, Some(Equal)),
            (i64::MIN, -9_223_372_036_854_777_856.0, Some(Greater)),
            (i64::MAX, 9.2e18, Some(Greater)),
            (0, f64::INFINITY, Some(Less)),
            (0, f64::NEG_INFINITY, Some(Greater)),
            (0, f64::NAN, None),
        ];
        for &(int, float, expected) in cases {
            assert_eq!(
                int_against_float(int, float),
                expected,
                "{int} against {float}"
            );
        }
    }

    #[test]
    fn mixed_comparisons_do_not_round_through_a_common_float() {
        let ints = Tensor::new(
            Arc::new(TensorData::from_vec_i32(vec![20_000_001], Device::cpu())),
            Shape::new(vec![1]),
            DataType::Int32,
            Device::cpu(),
            false,
        );
        let floats = tensor_from_vec_f32(vec![20_000_000.0]);
        let truth = |t: Tensor| t.data().as_bool_slice().unwrap().to_vec();
        assert_eq!(truth(eq(&ints, &floats).unwrap()), vec![false]);
        assert_eq!(truth(gt(&ints, &floats).unwrap()), vec![true]);
        assert_eq!(truth(lt(&floats, &ints).unwrap()), vec![true]);
        assert_eq!(truth(ne(&floats, &ints).unwrap()), vec![true]);
        assert_eq!(
            truth(isclose(&ints, &floats, 0.0, 0.0, false).unwrap()),
            vec![false]
        );
    }

    #[test]
    fn test_eq_promotes_mixed_dtypes() {
        let a = tensor_from_vec_f32(vec![1.0, 2.0]);
        let b = tensor_from_vec_i32(vec![1, 3]);
        let result = eq(&a, &b).unwrap();
        let slice = result.data().as_bool_slice().unwrap();
        assert_eq!(slice, &[true, false]);
    }

    #[test]
    fn test_lt_promotes_bool_with_integers() {
        let a = tensor_from_vec_bool(vec![true, false]);
        let b = tensor_from_vec_i32(vec![2, -1]);
        let result = lt(&a, &b).unwrap();
        let slice = result.data().as_bool_slice().unwrap();
        assert_eq!(slice, &[true, false]);
    }
}

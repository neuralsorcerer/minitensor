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
/// operands in their original order, and `in_f64` is the same comparison of
/// two float64s; a NaN satisfies only `!=`.
fn mixed_comparison(
    lhs: &Tensor,
    rhs: &Tensor,
    holds: impl Fn(Ordering) -> bool + Copy + Send + Sync,
    in_f64: impl Fn(f64, f64) -> bool + Copy + Send + Sync,
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
    let output_shape = lhs.shape().broadcast_with(rhs.shape())?;
    let missing = || MinitensorError::internal_error("comparison: operand slice missing");
    let answer = move |ordering: Option<Ordering>| match ordering {
        Some(ordering) => holds(ordering),
        None => nan_holds,
    };
    // Each pair is widened as it is read rather than both tensors copied
    // first. A float is a float64 exactly, and so is every int32 and every
    // int64 within 2^53, and those pairs compare exactly as float64s -- with
    // the plain operator, which vectorizes. Only an int64 past 2^53 needs the
    // floor.
    macro_rules! compare_as {
        ($int_slice:ident, $float_slice:ident, $exact:expr) => {{
            let ints = int_side.data().$int_slice().ok_or_else(missing)?;
            let floats = float_side.data().$float_slice().ok_or_else(missing)?;
            let exact = $exact;
            if int_on_left {
                broadcast_binary_data(int_side, float_side, ints, floats, &output_shape, |a, b| {
                    let b = f64::from(b);
                    match exact(a) {
                        Some(a) => in_f64(a, b),
                        None => answer(int_against_float(a.into(), b)),
                    }
                })?
            } else {
                broadcast_binary_data(float_side, int_side, floats, ints, &output_shape, |b, a| {
                    let b = f64::from(b);
                    match exact(a) {
                        Some(a) => in_f64(b, a),
                        None => answer(int_against_float(a.into(), b).map(Ordering::reverse)),
                    }
                })?
            }
        }};
    }
    let int32 = |a: i32| Some(f64::from(a));
    let int64 = |a: i64| (a.unsigned_abs() <= 1 << 53).then_some(a as f64);
    let data = match (int_side.dtype(), float_side.dtype()) {
        (DataType::Int32, DataType::Float32) => compare_as!(as_i32_slice, as_f32_slice, int32),
        (DataType::Int32, _) => compare_as!(as_i32_slice, as_f64_slice, int32),
        (_, DataType::Float32) => compare_as!(as_i64_slice, as_f32_slice, int64),
        _ => compare_as!(as_i64_slice, as_f64_slice, int64),
    };
    Ok(Some((data, output_shape)))
}

/// The six comparisons, as values, so one can be turned into another.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Relation {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl Relation {
    /// The relation `b ? a` that holds exactly when `a ? b` does.
    fn swapped(self) -> Self {
        match self {
            Relation::Lt => Relation::Gt,
            Relation::Le => Relation::Ge,
            Relation::Gt => Relation::Lt,
            Relation::Ge => Relation::Le,
            same => same,
        }
    }

    /// Whether `x ? value` holds of an `x` above the value.
    fn holds_above(self) -> bool {
        matches!(self, Relation::Ne | Relation::Gt | Relation::Ge)
    }

    /// Whether `x ? value` holds of an `x` below the value.
    fn holds_below(self) -> bool {
        matches!(self, Relation::Ne | Relation::Lt | Relation::Le)
    }
}

/// A single number, read from a one-element tensor.
#[derive(Clone, Copy)]
enum Number {
    Integer(i64),
    Float(f64),
}

/// What `x ? value` comes to for every `x` of a tensor's dtype: one answer for
/// all of them, or the same question about a number of that dtype.
enum Posed {
    Constant(bool),
    Against(Relation, Number),
}

/// `x ? value` for a float `x` of the dtype `F`, posed against a float of that
/// dtype, exactly.
///
/// A value the dtype holds is itself. One it does not lies strictly between
/// two neighbouring floats `below < value < above`, and then no `x` equals it,
/// every `x` below it is at most `below`, and every one above it at least
/// `above` -- which `x < above` and `x > below` say. A NaN is neither below nor
/// above anything, and the floats asked about are not NaN, so that carries
/// over.
fn posed_for_float<F: Copy>(
    relation: Relation,
    rounded: F,
    order: Ordering,
    next_down: impl Fn(F) -> F,
    next_up: impl Fn(F) -> F,
    widen: impl Fn(F) -> f64,
) -> Posed {
    // `order` is how `rounded` compares with the value.
    let (below, above) = match order {
        Ordering::Equal => return Posed::Against(relation, Number::Float(widen(rounded))),
        Ordering::Greater => (next_down(rounded), rounded),
        Ordering::Less => (rounded, next_up(rounded)),
    };
    match relation {
        Relation::Eq => Posed::Constant(false),
        Relation::Ne => Posed::Constant(true),
        Relation::Lt | Relation::Le => Posed::Against(Relation::Lt, Number::Float(widen(above))),
        Relation::Gt | Relation::Ge => Posed::Against(Relation::Gt, Number::Float(widen(below))),
    }
}

/// `x ? value` for an integer `x` between `min` and `max`, posed against an
/// integer in that range, exactly.
fn posed_for_integer(relation: Relation, value: Number, min: i64, max: i64) -> Posed {
    // 2^63, exactly; every int64 is below it.
    const BOUND: f64 = 9_223_372_036_854_775_808.0;
    let (relation, threshold) = match value {
        Number::Integer(n) => (relation, n),
        Number::Float(f) if f.is_nan() => return Posed::Constant(relation == Relation::Ne),
        Number::Float(f) if f >= BOUND => return Posed::Constant(relation.holds_below()),
        Number::Float(f) if f < -BOUND => return Posed::Constant(relation.holds_above()),
        Number::Float(f) => {
            // Inside the int64 range a float's floor is an int64 exactly.
            let floor = f.floor();
            if floor == f {
                (relation, floor as i64)
            } else {
                // Strictly between `floor` and `floor + 1`, which no integer is.
                match relation {
                    Relation::Eq => return Posed::Constant(false),
                    Relation::Ne => return Posed::Constant(true),
                    Relation::Lt | Relation::Le => (Relation::Le, floor as i64),
                    Relation::Gt | Relation::Ge => (Relation::Gt, floor as i64),
                }
            }
        }
    };
    // Every `x` is below a threshold past the top of its range, and above one
    // past the bottom.
    if threshold > max {
        Posed::Constant(relation.holds_below())
    } else if threshold < min {
        Posed::Constant(relation.holds_above())
    } else {
        Posed::Against(relation, Number::Integer(threshold))
    }
}

/// A comparison of a tensor with one number of another dtype, made at the
/// tensor's own dtype; `None` when the operands are not such a pair.
///
/// The exact comparison of an integer with a float, below, widens both whole
/// tensors to int64 and float64 and orders every pair through a floor. Against
/// one number that is all avoidable: the question each element is asked can
/// be posed once, exactly, about a number of the tensor's dtype -- or answered
/// for all of them -- and the tensor compared as it is. `x > 0` on a float32
/// tensor, which the literal makes an integer comparison, was forty times the
/// cost of `x > 0.0`; and an int32 tensor against any Python int widened to
/// int64 first.
fn scalar_comparison(lhs: &Tensor, rhs: &Tensor, relation: Relation) -> Result<Option<Tensor>> {
    let (tensor, single, relation) = if rhs.numel() == 1 {
        (lhs, rhs, relation)
    } else if lhs.numel() == 1 {
        (rhs, lhs, relation.swapped())
    } else {
        return Ok(None);
    };
    if tensor.dtype() == single.dtype() {
        return Ok(None);
    }
    let value = match single.dtype() {
        DataType::Int32 => single
            .data()
            .as_i32_slice()
            .map(|v| Number::Integer(i64::from(v[0]))),
        DataType::Int64 => single.data().as_i64_slice().map(|v| Number::Integer(v[0])),
        DataType::Float32 => single
            .data()
            .as_f32_slice()
            .map(|v| Number::Float(f64::from(v[0]))),
        DataType::Float64 => single.data().as_f64_slice().map(|v| Number::Float(v[0])),
        DataType::Bool => return Ok(None),
    };
    let Some(value) = value else {
        return Ok(None);
    };
    let posed = match (tensor.dtype(), value) {
        (DataType::Float32, Number::Integer(n)) => {
            // To the nearest float32, which every int64 is within range of.
            let rounded = n as f32;
            let order =
                int_against_float(n, f64::from(rounded)).map_or(Ordering::Equal, Ordering::reverse);
            posed_for_float(
                relation,
                rounded,
                order,
                f32::next_down,
                f32::next_up,
                f64::from,
            )
        }
        (DataType::Float32, Number::Float(f)) => {
            if f.is_nan() {
                Posed::Constant(relation == Relation::Ne)
            } else {
                // Rounded to nearest, past the largest float32 to infinity, so
                // a neighbour can be infinite -- and `x < inf` is still exactly
                // `x <= f32::MAX`.
                let rounded = f as f32;
                let order = f64::from(rounded).total_cmp(&f);
                posed_for_float(
                    relation,
                    rounded,
                    order,
                    f32::next_down,
                    f32::next_up,
                    f64::from,
                )
            }
        }
        (DataType::Float64, Number::Integer(n)) => {
            let rounded = n as f64;
            let order = int_against_float(n, rounded).map_or(Ordering::Equal, Ordering::reverse);
            posed_for_float(
                relation,
                rounded,
                order,
                f64::next_down,
                f64::next_up,
                |v| v,
            )
        }
        (DataType::Int32, value) => {
            posed_for_integer(relation, value, i64::from(i32::MIN), i64::from(i32::MAX))
        }
        (DataType::Int64, value @ Number::Float(_)) => {
            posed_for_integer(relation, value, i64::MIN, i64::MAX)
        }
        _ => return Ok(None),
    };
    let output_shape = lhs.shape().broadcast_with(rhs.shape())?;
    let device = tensor.device();
    let (relation, number) = match posed {
        Posed::Constant(answer) => {
            let data = TensorData::from_vec_bool(vec![answer; output_shape.numel()], device);
            return Ok(Some(Tensor::new(
                Arc::new(data),
                output_shape,
                DataType::Bool,
                device,
                false,
            )));
        }
        Posed::Against(relation, number) => (relation, number),
    };
    // The number at the tensor's dtype, in the single operand's own shape so
    // that broadcasting comes out as it would have.
    let data = match (tensor.dtype(), number) {
        (DataType::Float32, Number::Float(f)) => TensorData::from_vec_f32(vec![f as f32], device),
        (DataType::Float64, Number::Float(f)) => TensorData::from_vec_f64(vec![f], device),
        (DataType::Int32, Number::Integer(n)) => TensorData::from_vec_i32(vec![n as i32], device),
        (DataType::Int64, Number::Integer(n)) => TensorData::from_vec_i64(vec![n], device),
        _ => {
            return Err(MinitensorError::internal_error(
                "comparison: a scalar was posed in a dtype other than its tensor's",
            ));
        }
    };
    let other = Tensor::new(
        Arc::new(data),
        single.shape().clone(),
        tensor.dtype(),
        device,
        false,
    );
    let compare = match relation {
        Relation::Eq => eq,
        Relation::Ne => ne,
        Relation::Lt => lt,
        Relation::Le => le,
        Relation::Gt => gt,
        Relation::Ge => ge,
    };
    let result = compare(tensor, &other)?;
    // Broadcasting is symmetric in the dims, but a one-element operand on the
    // left can still add leading ones; the result takes the shape the pair
    // broadcasts to, which reshaping a contiguous result to is free.
    if result.shape() != &output_shape {
        return Ok(Some(result.reshape(output_shape)?));
    }
    Ok(Some(result))
}

macro_rules! cmp_op {
    ($fn_name:ident, $op:tt, $relation:expr) => {
        pub fn $fn_name(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
            if lhs.device() != rhs.device() {
                return Err(MinitensorError::device_mismatch(
                    format!("{:?}", lhs.device()),
                    format!("{:?}", rhs.device()),
                ));
            }
            if let Some(result) = scalar_comparison(lhs, rhs, $relation)? {
                return Ok(result);
            }
            if let Some((data, shape)) = mixed_comparison(
                lhs,
                rhs,
                |ordering| ordering $op Ordering::Equal,
                |a: f64, b: f64| a $op b,
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

cmp_op!(eq, ==, Relation::Eq);
cmp_op!(ne, !=, Relation::Ne);
// Ordered comparisons apply to booleans as well, with `false < true`. Refusing
// them was inconsistent with everything around it: `eq`/`ne` accepted booleans,
// `minimum`/`maximum` already ordered them, and a boolean compared against a
// number promoted and compared fine -- only two boolean operands were rejected.
cmp_op!(lt, <, Relation::Lt);
cmp_op!(le, <=, Relation::Le);
cmp_op!(gt, >, Relation::Gt);
cmp_op!(ge, >=, Relation::Ge);

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

// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! The trapezoidal integral, each rounded once; see
//! `arithmetic::exact_sum::trapezoid_of`.

use std::sync::Arc;

use crate::{
    error::{MinitensorError, Result},
    ops::arithmetic::exact_sum::{
        Trapezoid, trapezoid_at, trapezoid_spaced, trapezoid_spaced_columns,
    },
    tensor::{DataType, Shape, Tensor, TensorData},
};

/// Slices integrated by one task, about this many samples between them.
const TASK: usize = 1 << 14;

/// The trapezoidal integral of `values` along `axis`, at `positions` --
/// shaped as `values`, or one a sample along `axis` -- or `step` apart
/// without them, each the float nearest the exact integral: `values` and
/// `positions` share a float dtype, and `step` is read at its width. The
/// result drops `axis`. No gradient: `trapezoid` gives it.
pub fn trapezoid_integral(
    values: &Tensor,
    positions: Option<&Tensor>,
    step: f64,
    axis: usize,
) -> Result<Tensor> {
    let dtype = values.dtype();
    if !matches!(dtype, DataType::Float32 | DataType::Float64) || axis >= values.ndim() {
        return Err(MinitensorError::invalid_operation(format!(
            "trapezoid_integral takes a float tensor and an axis of it, got {dtype:?} and axis \
             {axis} of {} dimensions",
            values.ndim()
        )));
    }
    let dims = values.shape().dims();
    let len = dims[axis];
    if let Some(positions) = positions {
        let shared = positions.ndim() == 1 && positions.numel() == len;
        if positions.dtype() != dtype || !(shared || positions.shape() == values.shape()) {
            return Err(MinitensorError::invalid_operation(format!(
                "trapezoid_integral takes positions of the values' dtype, shaped as them or one \
                 along the axis, got {:?} of {:?} beside {dtype:?} of {dims:?}",
                positions.dtype(),
                positions.shape().dims()
            )));
        }
    }
    let values = values.contiguous()?;
    let positions = positions
        .map(|positions| positions.contiguous())
        .transpose()?;
    let inner = dims[axis + 1..].iter().product::<usize>();
    let mut kept = dims.to_vec();
    kept.remove(axis);
    let count = kept.iter().product::<usize>();
    let missing = || MinitensorError::internal_error("trapezoid_integral: a float slice");
    let mut data = TensorData::uninitialized_on_device(count, dtype, values.device());
    if dtype == DataType::Float32 {
        let y = values.data().as_f32_slice().ok_or_else(missing)?;
        let x = positions
            .as_ref()
            .map(|x| x.data().as_f32_slice().ok_or_else(missing))
            .transpose()?;
        let out = data.as_f32_slice_mut().ok_or_else(missing)?;
        integrate(y, x, step as f32, out, len, inner);
    } else {
        let y = values.data().as_f64_slice().ok_or_else(missing)?;
        let x = positions
            .as_ref()
            .map(|x| x.data().as_f64_slice().ok_or_else(missing))
            .transpose()?;
        let out = data.as_f64_slice_mut().ok_or_else(missing)?;
        integrate(y, x, step, out, len, inner);
    }
    Ok(Tensor::new(
        Arc::new(data),
        Shape::new(kept),
        dtype,
        values.device(),
        false,
    ))
}

/// [`trapezoid_integral`] of `y` laid out `[outer, len, inner]` along the
/// middle axis into `out`, laid out `[outer, inner]`, with `x` shaped as `y`
/// or one a sample.
///
/// Evenly spaced and strided, the columns are carried a band at a time, row
/// by row. Otherwise a few long runs each spread across the pool and many
/// runs share it, whole runs to a task, each gathered where it is not side by
/// side.
fn integrate<T: Trapezoid>(
    y: &[T],
    x: Option<&[T]>,
    step: T,
    out: &mut [T],
    len: usize,
    inner: usize,
) {
    if out.is_empty() {
        return;
    }
    if len < 2 {
        out.fill(T::narrow(0.0));
        return;
    }
    if x.is_none() && inner > 1 {
        trapezoid_spaced_columns(y, len, inner, step, out);
        return;
    }
    let shared = x.is_some_and(|x| x.len() == len && y.len() != len);
    let one = |slice: usize, gathered: &mut (Vec<T>, Vec<T>)| {
        let base = slice / inner * len * inner + slice % inner;
        let gather = |from: &[T], into: &mut Vec<T>| {
            into.clear();
            into.extend((0..len).map(|k| from[base + k * inner]));
        };
        let (samples, places) = gathered;
        let row = if inner == 1 {
            &y[base..base + len]
        } else {
            gather(y, samples);
            &samples[..]
        };
        match x {
            None => trapezoid_spaced(row, step),
            Some(x) if shared => trapezoid_at(row, x),
            Some(x) if inner == 1 => trapezoid_at(row, &x[base..base + len]),
            Some(x) => {
                gather(x, places);
                trapezoid_at(row, places)
            }
        }
    };
    let slices = out.len();
    if slices < 4 || y.len() < crate::ops::map::PAR_THRESHOLD / 8 {
        // Too little to share, or a few runs, each spread across the pool by
        // the sums inside where it is long.
        let mut gathered = (Vec::new(), Vec::new());
        for (slice, slot) in out.iter_mut().enumerate() {
            *slot = one(slice, &mut gathered);
        }
        return;
    }
    let per_task = (TASK / len).max(1);
    let parts = crate::parallel::Parts::of(out);
    let task = |index: usize| {
        let mut gathered = (Vec::new(), Vec::new());
        for slice in index * per_task..((index + 1) * per_task).min(slices) {
            let value = one(slice, &mut gathered);
            // SAFETY: each slice's integral has its own place inside `out`.
            unsafe { *parts.start().add(slice) = value };
        }
    };
    crate::parallel::for_each_task(slices.div_ceil(per_task), &task);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;

    fn tensor_f64(data: Vec<f64>, shape: Vec<usize>) -> Tensor {
        Tensor::new(
            Arc::new(TensorData::from_vec_f64(data, Device::cpu())),
            Shape::new(shape),
            DataType::Float64,
            Device::cpu(),
            false,
        )
    }

    fn tensor_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        Tensor::new(
            Arc::new(TensorData::from_vec_f32(data, Device::cpu())),
            Shape::new(shape),
            DataType::Float32,
            Device::cpu(),
            false,
        )
    }

    fn values_f64(tensor: &Tensor) -> Vec<f64> {
        tensor.data().as_f64_slice().unwrap().to_vec()
    }

    #[test]
    fn spaced_and_positioned_integrals_round_once() {
        // Samples of `2^-53` beside ones: a plain sum drops them, the exact
        // integral `1 + 2^-52` keeps them.
        let tiny = 2f64.powi(-53);
        let y = tensor_f64(vec![1.0, tiny, tiny, 1.0], vec![4]);
        let got = trapezoid_integral(&y, None, 1.0, 0).unwrap();
        assert_eq!(values_f64(&got), vec![1.0 + 2f64.powi(-52)]);
        let x = tensor_f64(vec![0.0, 1.0, 2.0, 3.0], vec![4]);
        let got = trapezoid_integral(&y, Some(&x), 1.0, 0).unwrap();
        assert_eq!(values_f64(&got), vec![1.0 + 2f64.powi(-52)]);
        // `1 + 3 · 2^-53` lies halfway, and goes to the even neighbour.
        let y = tensor_f64(vec![1.0, tiny, tiny, tiny, 1.0], vec![5]);
        let got = trapezoid_integral(&y, None, 1.0, 0).unwrap();
        assert_eq!(values_f64(&got), vec![1.0 + 2f64.powi(-51)]);
        // Float32 samples halfway between two float32s round to even.
        let y = tensor_f32(vec![1.0, 2f32.powi(-24), 0.0], vec![3]);
        let got = trapezoid_integral(&y, None, 2.0, 0).unwrap();
        assert_eq!(got.data().as_f32_slice().unwrap(), &[1.0 + 2f32.powi(-23)]);
    }

    #[test]
    fn columns_and_rows_agree() {
        let data: Vec<f64> = (0..24).map(|i| f64::from(i * i % 7) - 2.5).collect();
        let matrix = tensor_f64(data.clone(), vec![4, 6]);
        let columns = trapezoid_integral(&matrix, None, 0.5, 0).unwrap();
        let mut transposed = vec![0.0; 24];
        for row in 0..4 {
            for column in 0..6 {
                transposed[column * 4 + row] = data[row * 6 + column];
            }
        }
        let rows = trapezoid_integral(&tensor_f64(transposed, vec![6, 4]), None, 0.5, 1).unwrap();
        assert_eq!(columns.shape().dims(), &[6]);
        assert_eq!(values_f64(&columns), values_f64(&rows));
        // A coordinate shared by every row.
        let grid = tensor_f64(vec![0.0, 1.0, 3.0, 4.0, 8.0, 9.0], vec![6]);
        let got = trapezoid_integral(&matrix, Some(&grid), 1.0, 1).unwrap();
        let want: Vec<f64> = (0..4)
            .map(|row| {
                let r = &data[row * 6..row * 6 + 6];
                let g = [0.0, 1.0, 3.0, 4.0, 8.0, 9.0];
                (0..5)
                    .map(|i| (g[i + 1] - g[i]) * (r[i] + r[i + 1]) / 2.0)
                    .sum()
            })
            .collect();
        assert_eq!(values_f64(&got), want);
    }

    #[test]
    fn products_past_the_range_and_specials() {
        // Products of positions and samples past the largest float, of an
        // integral well inside the range: the width is exact, and so is the
        // integral.
        let first = f64::MAX / 4.0;
        let last = first * (1.0 + 2f64.powi(-50));
        let x = tensor_f64(vec![first, last], vec![2]);
        let y = tensor_f64(vec![8.0, 8.0], vec![2]);
        let got = trapezoid_integral(&y, Some(&x), 1.0, 0).unwrap();
        assert_eq!(values_f64(&got), vec![(last - first) * 8.0]);
        let y = tensor_f64(vec![1.0, f64::INFINITY, 2.0], vec![3]);
        let got = trapezoid_integral(&y, None, 1.0, 0).unwrap();
        assert_eq!(values_f64(&got), vec![f64::INFINITY]);
        let y = tensor_f64(vec![1.0, f64::INFINITY, f64::NEG_INFINITY], vec![3]);
        assert!(values_f64(&trapezoid_integral(&y, None, 1.0, 0).unwrap())[0].is_nan());
        let y = tensor_f64(vec![0.0; 6], vec![2, 3]);
        let got = trapezoid_integral(&y, None, f64::NAN, 1).unwrap();
        assert!(values_f64(&got).iter().all(|value| value.is_nan()));
        let got = trapezoid_integral(&y, None, 3.0, 0).unwrap();
        assert_eq!(values_f64(&got), vec![0.0; 3]);
    }

    #[test]
    fn refuses_what_it_cannot_integrate() {
        let y = tensor_f64(vec![1.0, 2.0, 3.0], vec![3]);
        assert!(trapezoid_integral(&y, None, 1.0, 1).is_err());
        let x = tensor_f64(vec![0.0, 1.0], vec![2]);
        assert!(trapezoid_integral(&y, Some(&x), 1.0, 0).is_err());
        let x = tensor_f32(vec![0.0, 1.0, 2.0], vec![3]);
        assert!(trapezoid_integral(&y, Some(&x), 1.0, 0).is_err());
        let ints = Tensor::new(
            Arc::new(TensorData::from_vec_i64(vec![1, 2, 3], Device::cpu())),
            Shape::new(vec![3]),
            DataType::Int64,
            Device::cpu(),
            false,
        );
        assert!(trapezoid_integral(&ints, None, 1.0, 0).is_err());
    }
}

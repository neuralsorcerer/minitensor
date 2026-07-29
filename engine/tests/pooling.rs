// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! 2-D pooling, checked against explicit per-window references.

use std::sync::Arc;

use engine::autograd;
use engine::ops::pooling::{avg_pool2d, max_pool2d};
use engine::ops::reduction;
use engine::tensor::Shape;
use engine::{DataType, Device, Tensor, TensorData};

fn tensor(data: Vec<f32>, dims: Vec<usize>, requires_grad: bool) -> Tensor {
    Tensor::new(
        Arc::new(TensorData::from_vec_f32(data, Device::cpu())),
        Shape::new(dims),
        DataType::Float32,
        Device::cpu(),
        requires_grad,
    )
}

fn ramp(n: usize, c: usize, h: usize, w: usize) -> Vec<f32> {
    (0..n * c * h * w)
        .map(|i| ((i * 37) % 23) as f32 - 11.0)
        .collect()
}

/// Independent reference: walk every window explicitly.
fn reference_pool(
    data: &[f32],
    (n, c, h, w): (usize, usize, usize, usize),
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    max: bool,
    count_include_pad: bool,
) -> (Vec<f32>, usize, usize) {
    let out_h = (h + 2 * padding.0 - kernel.0) / stride.0 + 1;
    let out_w = (w + 2 * padding.1 - kernel.1) / stride.1 + 1;
    let mut out = Vec::with_capacity(n * c * out_h * out_w);
    for plane in 0..n * c {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let mut best = f32::NEG_INFINITY;
                let mut total = 0.0f32;
                let mut counted = 0usize;
                for ky in 0..kernel.0 {
                    let ih = oh * stride.0 + ky;
                    if ih < padding.0 || ih >= h + padding.0 {
                        continue;
                    }
                    for kx in 0..kernel.1 {
                        let iw = ow * stride.1 + kx;
                        if iw < padding.1 || iw >= w + padding.1 {
                            continue;
                        }
                        let v = data[plane * h * w + (ih - padding.0) * w + (iw - padding.1)];
                        best = best.max(v);
                        total += v;
                        counted += 1;
                    }
                }
                out.push(if max {
                    best
                } else {
                    let divisor = if count_include_pad {
                        kernel.0 * kernel.1
                    } else {
                        counted
                    };
                    total / divisor as f32
                });
            }
        }
    }
    (out, out_h, out_w)
}

#[test]
fn test_pooling_matches_an_explicit_reference() {
    let shapes = [(1usize, 1usize, 4usize, 4usize), (2, 3, 7, 5), (1, 2, 8, 8)];
    let configs = [
        ((2usize, 2usize), (2usize, 2usize), (0usize, 0usize)),
        ((3, 3), (1, 1), (1, 1)),
        ((2, 2), (1, 1), (0, 0)),
        ((3, 2), (2, 1), (1, 0)),
    ];

    for &(n, c, h, w) in &shapes {
        let data = ramp(n, c, h, w);
        let input = tensor(data.clone(), vec![n, c, h, w], false);
        for &(kernel, stride, padding) in &configs {
            if h + 2 * padding.0 < kernel.0 || w + 2 * padding.1 < kernel.1 {
                continue;
            }

            let (want, out_h, out_w) =
                reference_pool(&data, (n, c, h, w), kernel, stride, padding, true, false);
            let got = max_pool2d(&input, kernel, stride, padding).unwrap();
            assert_eq!(got.shape().dims(), &[n, c, out_h, out_w]);
            assert_eq!(
                got.data().as_f32_slice().unwrap(),
                want.as_slice(),
                "max_pool2d {n}x{c}x{h}x{w} k{kernel:?} s{stride:?} p{padding:?}"
            );

            for count_include_pad in [true, false] {
                let (want, out_h, out_w) = reference_pool(
                    &data,
                    (n, c, h, w),
                    kernel,
                    stride,
                    padding,
                    false,
                    count_include_pad,
                );
                let got = avg_pool2d(&input, kernel, stride, padding, count_include_pad).unwrap();
                assert_eq!(got.shape().dims(), &[n, c, out_h, out_w]);
                for (i, (&g, &e)) in got
                    .data()
                    .as_f32_slice()
                    .unwrap()
                    .iter()
                    .zip(want.iter())
                    .enumerate()
                {
                    assert!(
                        (g - e).abs() <= 1e-5 * e.abs().max(1.0),
                        "avg_pool2d {n}x{c}x{h}x{w} k{kernel:?} s{stride:?} p{padding:?} \
                         pad_counted={count_include_pad} at {i}: {g} vs {e}"
                    );
                }
            }
        }
    }
    let _ = autograd::clear_graph();
}

/// Central differences on the summed output, which is what a pooling layer
/// feeding a loss actually produces.
///
/// This is the noisy side of the comparison, not the reference: the estimate
/// subtracts two f32 sums of tens of terms and divides by `2 * step`, which
/// leaves an error floor around 1e-3 no matter how exact the analytic gradient
/// is. The tolerance below is set to that floor.
fn finite_difference_grad(
    data: &[f32],
    dims: (usize, usize, usize, usize),
    f: impl Fn(&Tensor) -> Tensor,
) -> Vec<f32> {
    let (n, c, h, w) = dims;
    let step = 2e-2f32;
    let mut grad = vec![0f32; data.len()];
    for i in 0..data.len() {
        let mut plus = data.to_vec();
        let mut minus = data.to_vec();
        plus[i] += step;
        minus[i] -= step;
        let sum_of = |v: Vec<f32>| -> f32 {
            let t = tensor(v, vec![n, c, h, w], false);
            let out = f(&t);
            out.data().as_f32_slice().unwrap().iter().sum()
        };
        grad[i] = (sum_of(plus) - sum_of(minus)) / (2.0 * step);
    }
    grad
}

#[test]
fn test_pooling_gradients_match_central_differences() {
    let (n, c, h, w) = (1usize, 2usize, 5usize, 5usize);
    // Distinct values so no window has a tied maximum, where the subgradient is
    // a choice rather than a derivative.
    let data: Vec<f32> = (0..n * c * h * w).map(|i| i as f32 * 0.37 - 4.0).collect();

    for &(kernel, stride, padding) in &[
        ((2usize, 2usize), (2usize, 2usize), (0usize, 0usize)),
        ((3, 3), (2, 2), (1, 1)),
        ((2, 2), (1, 1), (0, 0)),
    ] {
        for max in [true, false] {
            let input = tensor(data.clone(), vec![n, c, h, w], true);
            let out = if max {
                max_pool2d(&input, kernel, stride, padding).unwrap()
            } else {
                avg_pool2d(&input, kernel, stride, padding, false).unwrap()
            };
            let loss = reduction::sum(&out, None, false).unwrap();
            loss.backward(None).unwrap();
            let analytic = autograd::get_gradient(&input).expect("input gradient");
            let analytic = analytic.data().as_f32_slice().unwrap().to_vec();

            let numeric = finite_difference_grad(&data, (n, c, h, w), |t| {
                if max {
                    max_pool2d(t, kernel, stride, padding).unwrap()
                } else {
                    avg_pool2d(t, kernel, stride, padding, false).unwrap()
                }
            });

            for (i, (&a, &e)) in analytic.iter().zip(numeric.iter()).enumerate() {
                assert!(
                    (a - e).abs() <= 5e-3 * e.abs().max(1.0),
                    "{} k{kernel:?} s{stride:?} p{padding:?} grad[{i}]: {a} vs {e}",
                    if max { "max_pool2d" } else { "avg_pool2d" }
                );
            }
            let _ = autograd::clear_graph();
        }
    }
}

#[test]
fn test_max_pool_gradient_accumulates_on_overlapping_windows() {
    // With stride < kernel the windows overlap, so one input element can win
    // several of them and must receive the sum of their gradients.
    let input = tensor(
        vec![1.0, 2.0, 3.0, 4.0, 9.0, 5.0, 6.0, 7.0, 8.0],
        vec![1, 1, 3, 3],
        true,
    );
    // The 9.0 at the centre is the maximum of every 2x2 window.
    let out = max_pool2d(&input, (2, 2), (1, 1), (0, 0)).unwrap();
    assert_eq!(out.data().as_f32_slice().unwrap(), &[9.0, 9.0, 9.0, 9.0]);

    let loss = reduction::sum(&out, None, false).unwrap();
    loss.backward(None).unwrap();
    let grad = autograd::get_gradient(&input).unwrap();
    let grad = grad.data().as_f32_slice().unwrap();
    assert_eq!(grad, &[0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0]);
    let _ = autograd::clear_graph();
}

#[test]
fn test_pooling_rejects_invalid_geometry() {
    let input = tensor(vec![0.0; 16], vec![1, 1, 4, 4], false);
    // Window larger than the padded input.
    assert!(max_pool2d(&input, (5, 5), (1, 1), (0, 0)).is_err());
    // Padding beyond half the window pools pure padding.
    assert!(max_pool2d(&input, (2, 2), (1, 1), (2, 2)).is_err());
    // Zero stride would never advance.
    assert!(max_pool2d(&input, (2, 2), (0, 1), (0, 0)).is_err());
    // Rank must be 4.
    let flat = tensor(vec![0.0; 16], vec![16], false);
    assert!(max_pool2d(&flat, (2, 2), (2, 2), (0, 0)).is_err());
}

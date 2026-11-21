// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! The elementwise half of a GRU step, as one operation.
//!
//! The companion to [`crate::nn::lstm_cell`], and the same trade: the matmul
//! that projects the hidden state stays an ordinary operation, so the
//! gradients to `w_hh`, `b_hh` and the previous hidden state still come from
//! the graph. What is fused is the twelve elementwise operations after it,
//! which composed cost twelve retained intermediates per timestep.
//!
//! # Why this takes two gate buffers and the LSTM takes one
//!
//! An LSTM adds its two projections before doing anything else, so a fused
//! cell can take the sum. A GRU cannot: the reset gate multiplies the
//! *projected hidden* contribution to the candidate and not the input's, so
//! the two halves have to arrive separately and stay separate until `n`.
//!
//! That is also the detail GRU implementations most often get wrong. With the
//! bias inside the product, as it is here and in cuDNN, `r` scales `b_hn` too.
//! `n = tanh(W_in x + b_in + r * (W_hn h + b_hn))` is not
//! `tanh(W_in x + b_in + W_hn (r * h) + b_hn)`, and the difference is large
//! rather than a rounding artefact.
//!
//! # What this changes numerically, and what it does not
//!
//! The forward is **bit-for-bit** what the composed form produced. Every
//! rounding happens in the same order on the same values: float32 goes through
//! the same `F32Kernel` the elementwise `sigmoid` and `tanh` call, float64
//! through the same scalar fallbacks, and the final blend is written
//! `(1 - z) * n + z * h` rather than the algebraically equal
//! `n + z * (h - n)`.
//!
//! That last one is load-bearing and is the reason the composed form
//! allocated a tensor of ones. `sigmoid` reaches exactly `1.0` in float32 at a
//! logit of about 17, and there `(1 - z) * n + z * h` returns `h` bit-for-bit
//! while `n + z * (h - n)` misses it in roughly a third of cases by up to
//! 5e-7. A saturated update gate is exactly how a GRU carries state across a
//! long sequence, so that error would be injected at every step of the one
//! path that is supposed to be lossless. Fusing removes the `ones` allocation
//! without touching the arithmetic that made it necessary.
//!
//! The backward is not, for the same reason as the LSTM's: its slopes are
//! taken of the cell's inputs, where the composed form took `r (1 - r)`,
//! `z (1 - z)` and `1 - n^2` of the activations it had rounded and lost them
//! as the gates saturated -- `1 - z`, the share the candidate gets, along with
//! them. Every value and slope is formed again in float64 from the two
//! projections, their sums taken there too, and each gradient rounded once:
//! for float32 the float nearest the exact one, for float64 within a few
//! ulps, more where a logit is large. `tests/nn/test_recurrent_cell_slopes.py`
//! holds them to exact values, and
//! `tests/nn/test_recurrent_parameter_gradients.py` checks every parameter of
//! four GRU configurations against central differences.

use crate::{
    autograd::{GradientFunction, TensorId, with_grad_fn},
    error::{MinitensorError, Result},
    nn::cell::{
        Scaled, band_width, finite_or_zero, logistic_near, logistic_pair, logistic_scaled,
        par_row_bands, sigmoid_block_f64, tanh_and_slope_near, tanh_block_f64, tanh_pair,
        tanh_scaled,
    },
    nn::lstm_cell::{product, slack},
    ops::arithmetic::exact_sum::Output,
    ops::arithmetic::{extended::Pair, rounded_once::two_sum},
    ops::map::{
        EXPENSIVE_PAR_THRESHOLD, SIMD_PAR_THRESHOLD, VECTOR_F32_PAR_THRESHOLD, build_vec,
        par_out_chunks, unary_map_blocks_threshold,
    },
    tensor::{DataType, Shape, Tensor, TensorData},
};
use rustc_hash::FxHashMap;
use std::{mem::MaybeUninit, sync::Arc};

/// Gate order within the `3 * hidden` axis, matching the stored weights.
///
/// Reset and update being adjacent is what lets the forward activate them with
/// one `sigmoid` call over the whole batch; the candidate is separate because
/// it is a `tanh` and because the reset gate reaches it first.
const RESET: usize = 0;
const UPDATE: usize = 1;
const CANDIDATE: usize = 2;

/// What the backward needs: the cell's three inputs as they were given.
///
/// The slopes are taken of the two projections, whose sums the gates' logits
/// are -- formed in float64, rather than rounded to the dtype first -- and not
/// of the activations the forward rounded, so those are not kept. The hidden projection's
/// candidate block is among them, which `n` cannot be run backwards to
/// recover.
struct Saved {
    from_input: Tensor,
    from_hidden: Tensor,
    hidden_before: Tensor,
}

/// One GRU step's elementwise part.
///
/// `from_input` and `from_hidden` are both `(batch, 3 * hidden)` and carry
/// their own biases; `hidden_before` is `(batch, hidden)`.
pub fn gru_cell(
    from_input: &Tensor,
    from_hidden: &Tensor,
    hidden_before: &Tensor,
) -> Result<Tensor> {
    let dims = from_input.shape().dims();
    if dims.len() != 2 || !dims[1].is_multiple_of(3) {
        return Err(MinitensorError::invalid_operation(
            "gru_cell expects projections shaped [batch, 3 * hidden]",
        ));
    }
    let batch = dims[0];
    let hidden = dims[1] / 3;

    if from_hidden.shape().dims() != dims {
        return Err(MinitensorError::invalid_operation(format!(
            "gru_cell: hidden projection is {:?}, expected {dims:?}",
            from_hidden.shape().dims()
        )));
    }
    if hidden_before.shape().dims() != [batch, hidden] {
        return Err(MinitensorError::invalid_operation(format!(
            "gru_cell: hidden state is {:?}, expected [{batch}, {hidden}]",
            hidden_before.shape().dims()
        )));
    }
    let dtype = from_input.dtype();
    if from_hidden.dtype() != dtype || hidden_before.dtype() != dtype {
        return Err(MinitensorError::invalid_operation(
            "gru_cell: all three inputs must share a dtype",
        ));
    }

    let requires_grad =
        from_input.requires_grad() || from_hidden.requires_grad() || hidden_before.requires_grad();
    let narrow_shape = Shape::new(vec![batch, hidden]);
    let device = from_input.device();

    macro_rules! run {
        ($ty:ty, $slice:ident, $par:expr, $sigmoid_block:expr, $tanh_block:expr) => {{
            let input_gates = from_input.data().$slice().ok_or_else(|| {
                MinitensorError::internal_error("gru_cell: input projection dtype")
            })?;
            let hidden_gates = from_hidden.data().$slice().ok_or_else(|| {
                MinitensorError::internal_error("gru_cell: hidden projection dtype")
            })?;
            let previous = hidden_before
                .data()
                .$slice()
                .ok_or_else(|| MinitensorError::internal_error("gru_cell: hidden state dtype"))?;
            let sigmoid_block = $sigmoid_block;
            let tanh_block = $tanh_block;

            // `r` and `z` are adjacent in both projections, so their sums are
            // one contiguous `2 * hidden` block per row. Gathered into a buffer
            // whose rows *are* contiguous, the whole batch then takes a single
            // `sigmoid` call rather than one per row.
            // SAFETY: the rows tile the output and each is written whole.
            let pre_gates: Vec<$ty> = unsafe {
                build_vec(batch * 2 * hidden, |spare| {
                    if hidden == 0 {
                        return;
                    }
                    let width = band_width(spare.len(), 2 * hidden, SIMD_PAR_THRESHOLD);
                    par_out_chunks(spare, width, &|start, band| {
                        for (r, out_row) in band.chunks_exact_mut(2 * hidden).enumerate() {
                            let wide = (start + r * 2 * hidden) / 2 * 3;
                            let from_x = &input_gates[wide..][..2 * hidden];
                            let from_h = &hidden_gates[wide..][..2 * hidden];
                            for j in 0..2 * hidden {
                                out_row[j] = MaybeUninit::new(from_x[j] + from_h[j]);
                            }
                        }
                    });
                })
            };
            // SAFETY: `sigmoid_block` initializes every element it is given.
            let gates: Vec<$ty> =
                unsafe { unary_map_blocks_threshold(&pre_gates, $par, sigmoid_block) };
            drop(pre_gates);

            // n = tanh(from_input_n + r * from_hidden_n), gathered contiguous so
            // the `tanh` is one call for the batch. The reset gate scales the
            // hidden projection's candidate block and nothing else.
            // SAFETY: as above.
            let pre_candidate: Vec<$ty> = unsafe {
                build_vec(batch * hidden, |spare| {
                    if hidden == 0 {
                        return;
                    }
                    let width = band_width(spare.len(), hidden, SIMD_PAR_THRESHOLD);
                    par_out_chunks(spare, width, &|start, band| {
                        for (r, out_row) in band.chunks_exact_mut(hidden).enumerate() {
                            let narrow = start + r * hidden;
                            let from_x = &input_gates[narrow * 3 + CANDIDATE * hidden..][..hidden];
                            let reset = &gates[narrow * 2 + RESET * hidden..][..hidden];
                            let projected =
                                &hidden_gates[narrow * 3 + CANDIDATE * hidden..][..hidden];
                            for j in 0..hidden {
                                out_row[j] = MaybeUninit::new(from_x[j] + reset[j] * projected[j]);
                            }
                        }
                    });
                })
            };
            // SAFETY: `tanh_block` initializes every element it is given.
            let candidate: Vec<$ty> =
                unsafe { unary_map_blocks_threshold(&pre_candidate, $par, tanh_block) };
            drop(pre_candidate);

            // h' = (1 - z) * n + z * h. Four roundings in this order; see the
            // note at the top of this file on why the shorter form is wrong.
            // SAFETY: as above.
            let hidden_after: Vec<$ty> = unsafe {
                build_vec(batch * hidden, |spare| {
                    if hidden == 0 {
                        return;
                    }
                    let width = band_width(spare.len(), hidden, SIMD_PAR_THRESHOLD);
                    par_out_chunks(spare, width, &|start, band| {
                        for (r, out_row) in band.chunks_exact_mut(hidden).enumerate() {
                            let narrow = start + r * hidden;
                            let update = &gates[narrow * 2 + UPDATE * hidden..][..hidden];
                            let n = &candidate[narrow..][..hidden];
                            let prev = &previous[narrow..][..hidden];
                            for j in 0..hidden {
                                out_row[j] = MaybeUninit::new(
                                    (1.0 - update[j]) * n[j] + update[j] * prev[j],
                                );
                            }
                        }
                    });
                })
            };

            TensorData::from_vec::<$ty>(hidden_after, dtype, device)
        }};
    }

    let hidden_after = match dtype {
        DataType::Float32 => {
            let kernel = crate::ops::simd::F32Kernel::select();
            run!(
                f32,
                as_f32_slice,
                VECTOR_F32_PAR_THRESHOLD,
                |src: &[f32], dst: &mut [MaybeUninit<f32>]| kernel.sigmoid(src, dst),
                |src: &[f32], dst: &mut [MaybeUninit<f32>]| kernel.tanh(src, dst)
            )
        }
        DataType::Float64 => run!(
            f64,
            as_f64_slice,
            EXPENSIVE_PAR_THRESHOLD,
            sigmoid_block_f64,
            tanh_block_f64
        ),
        other => {
            return Err(MinitensorError::invalid_operation(format!(
                "gru_cell requires a floating point dtype, got {other:?}"
            )));
        }
    };

    let new_h = Tensor::new(
        Arc::new(hidden_after),
        narrow_shape,
        dtype,
        device,
        requires_grad,
    );

    if !requires_grad {
        return Ok(new_h);
    }

    let saved = Saved {
        from_input: from_input.detach(),
        from_hidden: from_hidden.detach(),
        hidden_before: hidden_before.detach(),
    };

    with_grad_fn(
        new_h,
        Arc::new(GruCellBackward {
            saved,
            hidden,
            ids: [from_input.id(), from_hidden.id(), hidden_before.id()],
            wanted: [
                from_input.requires_grad(),
                from_hidden.requires_grad(),
                hidden_before.requires_grad(),
            ],
        }),
    )
}

/// The backward for [`gru_cell`].
struct GruCellBackward {
    saved: Saved,
    hidden: usize,
    ids: [TensorId; 3],
    /// Whether each input wanted a gradient at all.
    wanted: [bool; 3],
}

impl GradientFunction for GruCellBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let hidden = self.hidden;
        let batch = self.saved.hidden_before.shape().dims()[0];
        let dtype = grad_output.dtype();
        let device = grad_output.device();

        macro_rules! run {
            ($ty:ty, $slice:ident) => {{
                let incoming = grad_output.data().$slice().ok_or_else(|| {
                    MinitensorError::internal_error("gru_cell backward: gradient dtype")
                })?;
                let from_input = self.saved.from_input.data().$slice().ok_or_else(|| {
                    MinitensorError::internal_error("gru_cell backward: saved input projection")
                })?;
                let from_hidden = self.saved.from_hidden.data().$slice().ok_or_else(|| {
                    MinitensorError::internal_error("gru_cell backward: saved hidden projection")
                })?;
                let previous = self.saved.hidden_before.data().$slice().ok_or_else(|| {
                    MinitensorError::internal_error("gru_cell backward: saved hidden state")
                })?;
                let [d_input, d_hidden, d_previous] = backward_rows::<$ty>(
                    from_input,
                    from_hidden,
                    previous,
                    incoming,
                    hidden,
                    self.wanted,
                );
                (
                    TensorData::from_vec::<$ty>(d_input, dtype, device),
                    TensorData::from_vec::<$ty>(d_hidden, dtype, device),
                    TensorData::from_vec::<$ty>(d_previous, dtype, device),
                )
            }};
        }

        let (d_input, d_hidden, d_previous) = match dtype {
            DataType::Float32 => run!(f32, as_f32_slice),
            DataType::Float64 => run!(f64, as_f64_slice),
            other => {
                return Err(MinitensorError::invalid_operation(format!(
                    "gru_cell backward requires a floating point dtype, got {other:?}"
                )));
            }
        };

        let wide = Shape::new(vec![batch, 3 * hidden]);
        let mut gradients = FxHashMap::default();
        if self.wanted[0] {
            gradients.insert(
                self.ids[0],
                Tensor::new(Arc::new(d_input), wide.clone(), dtype, device, false),
            );
        }
        if self.wanted[1] {
            gradients.insert(
                self.ids[1],
                Tensor::new(Arc::new(d_hidden), wide, dtype, device, false),
            );
        }
        if self.wanted[2] {
            gradients.insert(
                self.ids[2],
                Tensor::new(
                    Arc::new(d_previous),
                    Shape::new(vec![batch, hidden]),
                    dtype,
                    device,
                    false,
                ),
            );
        }
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.ids
    }
}

/// The gradients of [`gru_cell`] -- of the input projection, of the hidden
/// projection and of the hidden state before, each empty unless `wanted` --
/// for the incoming gradient `incoming` of the new hidden state.
///
/// Both projections receive the same gradient through `r` and `z`, whose
/// logits are plain sums of the two; they part in the candidate's block,
/// where the input's is `dn` and the hidden's `dn * r`. One pass writes all
/// three, a row of each at a time, so that every element's values and slopes
/// -- a few exponentials -- are formed once.
fn backward_rows<T: Output>(
    from_input: &[T],
    from_hidden: &[T],
    previous: &[T],
    incoming: &[T],
    hidden: usize,
    wanted: [bool; 3],
) -> [Vec<T>; 3] {
    let batch = previous.len() / hidden.max(1);
    let wide_row = 3 * hidden;
    let wide = T::DIGITS > 24;
    let widths = [
        if wanted[0] { wide_row } else { 0 },
        if wanted[1] { wide_row } else { 0 },
        if wanted[2] { hidden } else { 0 },
    ];
    let mut outs = widths.map(|width| Vec::with_capacity(batch * width));
    {
        let [d_input, d_hidden, d_previous] = &mut outs;
        par_row_bands(
            [
                &mut d_input.spare_capacity_mut()[..batch * widths[0]],
                &mut d_hidden.spare_capacity_mut()[..batch * widths[1]],
                &mut d_previous.spare_capacity_mut()[..batch * widths[2]],
            ],
            widths,
            batch,
            EXPENSIVE_PAR_THRESHOLD,
            &|first, count, [input_rows, hidden_rows, previous_rows]| {
                for row in 0..count {
                    let at = first + row;
                    let a = &from_input[at * wide_row..][..wide_row];
                    let b = &from_hidden[at * wide_row..][..wide_row];
                    let before = &previous[at * hidden..][..hidden];
                    let grad = &incoming[at * hidden..][..hidden];
                    for j in 0..hidden {
                        let pick =
                            |p: &[T]| [RESET, UPDATE, CANDIDATE].map(|g| p[g * hidden + j].widen());
                        let (d, d_from_hidden_n, d_before) = unit_backward(
                            pick(a),
                            pick(b),
                            before[j].widen(),
                            grad[j].widen(),
                            wide,
                        );
                        let out = row * wide_row + j;
                        if wanted[0] {
                            for (g, d) in [RESET, UPDATE, CANDIDATE].into_iter().zip(d) {
                                input_rows[out + g * hidden].write(T::narrow(d));
                            }
                        }
                        if wanted[1] {
                            hidden_rows[out + RESET * hidden].write(T::narrow(d[RESET]));
                            hidden_rows[out + UPDATE * hidden].write(T::narrow(d[UPDATE]));
                            hidden_rows[out + CANDIDATE * hidden].write(T::narrow(d_from_hidden_n));
                        }
                        if wanted[2] {
                            previous_rows[row * hidden + j].write(T::narrow(d_before));
                        }
                    }
                }
            },
        );
    }
    for (out, width) in outs.iter_mut().zip(widths) {
        // SAFETY: every row of a wanted output was written whole above, and an
        // output nobody wants is empty.
        unsafe { out.set_len(batch * width) };
    }
    outs
}

/// One hidden unit's backward, in float64: from its two projections `a` and
/// `b` (input and hidden, each `[r, z, n]`), its hidden state before and the
/// incoming gradient, the gradients of the input projection's three logits,
/// of the hidden projection's candidate logit, and of the state before.
///
/// `h' = (1 - z) n + z h` with `n = tanh(a_n + r b_n)`, so `dn` is
/// `grad (1 - z) tanh'`, `dr = dn b_n r'`, `dz = grad (h - n) z'`, the hidden
/// projection's candidate logit gets `dn r`, and the state before `grad z`
/// through the blend alone -- the rest of its influence is through the
/// projection, which is the matmul's business and not this node's. `1 - z` is
/// `sigmoid(-x_z)` itself, not a difference.
///
/// The logits are taken as pairs, as their slopes move by up to the logit's
/// size times whatever it is out by: `a + b` with its rounding beside it, and
/// `a_n + r b_n` with the product's -- and where `r b_n` is beyond `slack` in
/// size (see [`slack`]), so that an ulp of a float64
/// `r` is many of the logit's last places, `r` carried as a pair too. Without
/// them float64 logits of some fifty left the gradients nearly a hundred ulps
/// out. Where the state before is near `n`, `h - n` is small beside what a
/// float64 `n` is out by; there `n` is carried as a pair, and `r` with it.
#[inline(always)]
fn unit_backward(
    a: [f64; 3],
    b: [f64; 3],
    before: f64,
    grad: f64,
    wide: bool,
) -> ([f64; 3], f64, f64) {
    let slack = slack(wide);
    // A float32 pair's sum is exact in float64 but where their exponents lie
    // 29 apart, and then off by less than a float32 gradient shows.
    let sum = |x: f64, y: f64| {
        if wide {
            let (s, e) = two_sum(x, y);
            (s, finite_or_zero(e))
        } else {
            (x + y, 0.0)
        }
    };
    let (x_r, e_r) = sum(a[RESET], b[RESET]);
    let (x_z, e_z) = sum(a[UPDATE], b[UPDATE]);
    let (r, _, dr) = logistic_near(x_r, e_r);
    let (z, not_z, dz) = logistic_near(x_z, e_z);
    let scaled = product(r, b[CANDIDATE], wide);
    let mut x_n = Pair(a[CANDIDATE], 0.0).add(scaled);
    x_n.1 = finite_or_zero(x_n.1);
    let (mut n, mut dn_slope) = tanh_and_slope_near(x_n.0, x_n.1);
    let mut gap = before - n;
    let cancels = gap.abs() < n.abs() / (2.0 * slack);
    if cancels || scaled.0.abs() > slack {
        let scaled = logistic_pair(Pair(x_r, e_r)).mul::<false>(Pair(b[CANDIDATE], 0.0));
        x_n = Pair(a[CANDIDATE], 0.0).add(scaled);
        x_n.1 = finite_or_zero(x_n.1);
        (n, dn_slope) = tanh_and_slope_near(x_n.0, x_n.1);
        gap = before - n;
    }
    if cancels {
        gap = Pair(before, 0.0).add(tanh_pair(x_n).neg()).0;
    }
    if wide && (x_r.abs() > 708.0 || x_z.abs() > 708.0 || x_n.0.abs() > 354.0) {
        return far_unit_backward(Pair(x_r, e_r), Pair(x_z, e_z), x_n, b[CANDIDATE], gap, grad);
    }
    let dn = grad * (not_z * dn_slope);
    (
        [dn * (b[CANDIDATE] * dr), grad * (gap * dz), dn],
        dn * r,
        grad * z,
    )
}

/// [`unit_backward`] where a value or slope lies below float64's normal range
/// -- a logit past 708, or a candidate logit past 354 -- and float64 holds it
/// only to its last few bits, while the other factors may bring a gradient
/// back up where every bit shows: each factor kept as a [`Scaled`], and each
/// gradient rounded once.
#[cold]
#[inline(never)]
fn far_unit_backward(
    x_r: Pair,
    x_z: Pair,
    x_n: Pair,
    b_n: f64,
    gap: f64,
    grad: f64,
) -> ([f64; 3], f64, f64) {
    let [r, _, dr] = logistic_scaled(x_r.0, x_r.1);
    let [z, not_z, dz] = logistic_scaled(x_z.0, x_z.1);
    let [_, dn_slope] = tanh_scaled(x_n.0, x_n.1);
    let grad = Scaled::of(grad);
    let dn = grad.times(not_z.times(dn_slope));
    (
        [
            dn.times(Scaled::of(b_n).times(dr)).value(),
            grad.times(Scaled::of(gap).times(dz)).value(),
            dn.value(),
        ],
        dn.times(r).value(),
        grad.times(z).value(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;

    fn tensor_f64(values: Vec<f64>, shape: Vec<usize>) -> Tensor {
        Tensor::new(
            Arc::new(TensorData::from_vec_f64(values, Device::cpu())),
            Shape::new(shape),
            DataType::Float64,
            Device::cpu(),
            false,
        )
    }

    fn tensor_f32(values: Vec<f32>, shape: Vec<usize>) -> Tensor {
        Tensor::new(
            Arc::new(TensorData::from_vec_f32(values, Device::cpu())),
            Shape::new(shape),
            DataType::Float32,
            Device::cpu(),
            false,
        )
    }

    /// The forward against the definition rather than against itself.
    ///
    /// The central-difference tests check the *derivative*, computed from this
    /// same forward, so a consistently wrong forward would satisfy them. This
    /// works two rows of arbitrary values through `r, z, n` by hand -- with the
    /// reset gate scaling the hidden projection's candidate block and nothing
    /// else, which is the part a GRU implementation gets wrong.
    #[test]
    fn one_step_matches_the_definition() {
        let hidden = 2;
        // Per row: [r0 r1 z0 z1 n0 n1].
        let from_x = vec![
            0.5, -1.0, 2.0, 0.25, -0.75, 1.5, //
            -2.0, 0.8, 1.25, -0.5, 0.6, -1.75,
        ];
        let from_h = vec![
            0.2, 0.6, -1.5, 0.75, 1.0, -0.4, //
            0.9, -1.2, 0.3, 2.0, -0.8, 0.45,
        ];
        let previous = vec![0.3, -0.7, 1.1, -0.2];

        let got = gru_cell(
            &tensor_f64(from_x.clone(), vec![2, 3 * hidden]),
            &tensor_f64(from_h.clone(), vec![2, 3 * hidden]),
            &tensor_f64(previous.clone(), vec![2, hidden]),
        )
        .expect("well-formed inputs");

        let sigmoid = |x: f64| 1.0 / (1.0 + (-x).exp());
        for b in 0..2 {
            for j in 0..hidden {
                let row = b * 3 * hidden;
                let r = sigmoid(from_x[row + j] + from_h[row + j]);
                let z = sigmoid(from_x[row + hidden + j] + from_h[row + hidden + j]);
                // The reset gate scales only the hidden projection's block.
                let n = (from_x[row + 2 * hidden + j] + r * from_h[row + 2 * hidden + j]).tanh();
                let expected = (1.0 - z) * n + z * previous[b * hidden + j];

                let at = b * hidden + j;
                let actual = got.data().as_f64_slice().unwrap()[at];
                assert!(
                    (actual - expected).abs() < 1e-12,
                    "hidden[{b}][{j}]: {actual} vs {expected}"
                );
            }
        }
    }

    /// A saturated update gate must pass the previous state through unchanged,
    /// bit for bit.
    ///
    /// This is the property the composed form allocated a tensor of ones to
    /// keep, and nothing tested it. `sigmoid` reaches exactly `1.0` in float32
    /// at a logit of about 17, and there `(1 - z) * n + z * h` returns `h`
    /// exactly while the algebraically equal `n + z * (h - n)` does not. A
    /// saturated `z` is how a GRU carries state across a long sequence, so the
    /// error would enter at every step of the one path that is lossless.
    #[test]
    fn a_saturated_update_gate_passes_the_state_through_exactly() {
        let hidden = 4;
        let previous: Vec<f32> = vec![0.37, -2.5, 1.0e-7, 12345.678];
        let mut from_x = vec![0.0f32; 3 * hidden];
        // Drive `z` to exactly 1.0 and leave `n` somewhere well away from `h`.
        for j in 0..hidden {
            from_x[UPDATE * hidden + j] = 40.0;
            from_x[CANDIDATE * hidden + j] = 0.9;
        }

        let got = gru_cell(
            &tensor_f32(from_x, vec![1, 3 * hidden]),
            &tensor_f32(vec![0.0; 3 * hidden], vec![1, 3 * hidden]),
            &tensor_f32(previous.clone(), vec![1, hidden]),
        )
        .unwrap();

        assert_eq!(
            got.data().as_f32_slice().unwrap(),
            previous.as_slice(),
            "a saturated update gate did not pass the state through unchanged"
        );
    }

    /// Bands are cut on row boundaries because which gate an element belongs
    /// to is decided by where it sits within its row. A band off by one would
    /// not crash; it would activate the wrong elements with the wrong
    /// function, and only for batches large enough to be split.
    ///
    /// The reference runs the same rows in quarters, each small enough that
    /// every row-banded pass stays sequential.
    #[test]
    fn the_parallel_split_changes_nothing() {
        let hidden = 24; // not a multiple of any SIMD width
        // The fewest rows that take the whole batch past the threshold; a
        // quarter's widest pass, three gates wide, then stays under it.
        let quarter = SIMD_PAR_THRESHOLD / (4 * hidden) + 1;
        let batch = quarter * 4;
        let from_x: Vec<f32> = (0..batch * 3 * hidden)
            .map(|i| ((i % 37) as f32 - 18.0) / 6.0)
            .collect();
        let from_h: Vec<f32> = (0..batch * 3 * hidden)
            .map(|i| ((i % 23) as f32 - 11.0) / 5.0)
            .collect();
        let previous: Vec<f32> = (0..batch * hidden)
            .map(|i| ((i % 19) as f32 - 9.0) / 4.0)
            .collect();

        assert!(
            batch * hidden >= SIMD_PAR_THRESHOLD && quarter * 3 * hidden < SIMD_PAR_THRESHOLD,
            "the whole batch must cross the threshold and each quarter must not"
        );
        let together = gru_cell(
            &tensor_f32(from_x.clone(), vec![batch, 3 * hidden]),
            &tensor_f32(from_h.clone(), vec![batch, 3 * hidden]),
            &tensor_f32(previous.clone(), vec![batch, hidden]),
        )
        .unwrap();

        for part in 0..4 {
            let wide = part * quarter * 3 * hidden;
            let narrow = part * quarter * hidden;
            let alone = gru_cell(
                &tensor_f32(
                    from_x[wide..][..quarter * 3 * hidden].to_vec(),
                    vec![quarter, 3 * hidden],
                ),
                &tensor_f32(
                    from_h[wide..][..quarter * 3 * hidden].to_vec(),
                    vec![quarter, 3 * hidden],
                ),
                &tensor_f32(
                    previous[narrow..][..quarter * hidden].to_vec(),
                    vec![quarter, hidden],
                ),
            )
            .unwrap();
            assert_eq!(
                &together.data().as_f32_slice().unwrap()[narrow..][..quarter * hidden],
                alone.data().as_f32_slice().unwrap(),
                "quarter {part}"
            );
        }
    }

    /// `chunks_exact(0)` panics, and both degenerate shapes reach it.
    #[test]
    fn degenerate_shapes_produce_empty_outputs_rather_than_panicking() {
        for (batch, hidden) in [(0usize, 3usize), (4, 0), (0, 0)] {
            let got = gru_cell(
                &tensor_f64(vec![0.0; batch * 3 * hidden], vec![batch, 3 * hidden]),
                &tensor_f64(vec![0.0; batch * 3 * hidden], vec![batch, 3 * hidden]),
                &tensor_f64(vec![0.0; batch * hidden], vec![batch, hidden]),
            )
            .unwrap_or_else(|e| panic!("batch={batch} hidden={hidden}: {e}"));
            assert_eq!(got.shape().dims(), &[batch, hidden]);
        }
    }

    #[test]
    fn malformed_inputs_are_refused_rather_than_misread() {
        let wide = |v: Vec<f64>, d: Vec<usize>| tensor_f64(v, d);
        // Not two-dimensional.
        assert!(
            gru_cell(
                &wide(vec![0.0; 6], vec![6]),
                &wide(vec![0.0; 6], vec![1, 6]),
                &wide(vec![0.0; 2], vec![1, 2])
            )
            .is_err()
        );
        // Gate axis is not a multiple of three, so there is no gate layout.
        assert!(
            gru_cell(
                &wide(vec![0.0; 4], vec![1, 4]),
                &wide(vec![0.0; 4], vec![1, 4]),
                &wide(vec![0.0; 1], vec![1, 1])
            )
            .is_err()
        );
        // The two projections disagree.
        assert!(
            gru_cell(
                &wide(vec![0.0; 6], vec![1, 6]),
                &wide(vec![0.0; 12], vec![2, 6]),
                &wide(vec![0.0; 2], vec![1, 2])
            )
            .is_err()
        );
        // The hidden state does not match the implied batch and width.
        assert!(
            gru_cell(
                &wide(vec![0.0; 6], vec![1, 6]),
                &wide(vec![0.0; 6], vec![1, 6]),
                &wide(vec![0.0; 3], vec![1, 3])
            )
            .is_err()
        );
        // Two dtypes.
        assert!(
            gru_cell(
                &wide(vec![0.0; 6], vec![1, 6]),
                &wide(vec![0.0; 6], vec![1, 6]),
                &tensor_f32(vec![0.0; 2], vec![1, 2])
            )
            .is_err()
        );
        // An integer dtype has no derivative and no `tanh`.
        let ints = |n: usize, d: Vec<usize>| {
            Tensor::new(
                Arc::new(TensorData::from_vec::<i64>(
                    vec![0i64; n],
                    DataType::Int64,
                    Device::cpu(),
                )),
                Shape::new(d),
                DataType::Int64,
                Device::cpu(),
                false,
            )
        };
        assert!(
            gru_cell(
                &ints(6, vec![1, 6]),
                &ints(6, vec![1, 6]),
                &ints(2, vec![1, 2])
            )
            .is_err()
        );
    }
}

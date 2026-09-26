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
//! The backward is not bit-identical, for the same reason as the LSTM's: it is
//! one expression per gate where the composed form was a chain of separate
//! products. `tests/nn/test_recurrent_parameter_gradients.py` value-checks
//! every parameter of four GRU configurations against central differences at
//! float64.

use crate::{
    autograd::{GradientFunction, TensorId, with_grad_fn},
    error::{MinitensorError, Result},
    nn::cell::{band_width, sigmoid_block_f64, tanh_block_f64},
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

/// What the backward needs.
///
/// `gates` holds the activated `r` and `z` in a `(batch, 2 * hidden)` layout,
/// `candidate` holds `n`, and `candidate_hidden` holds the projected hidden
/// contribution to the candidate *before* the reset gate scaled it -- which
/// `n` cannot be run backwards to recover.
struct Saved {
    gates: Tensor,
    candidate: Tensor,
    candidate_hidden: Tensor,
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

            // The reset gate scales the hidden projection's candidate block and
            // nothing else. That block is kept, because `n` cannot be run
            // backwards to recover it.
            // SAFETY: the rows tile the output and each is written whole.
            let candidate_hidden: Vec<$ty> = unsafe {
                build_vec(batch * hidden, |spare| {
                    if hidden == 0 {
                        return;
                    }
                    let width = band_width(spare.len(), hidden, SIMD_PAR_THRESHOLD);
                    par_out_chunks(spare, width, &|start, band| {
                        for (r, out_row) in band.chunks_exact_mut(hidden).enumerate() {
                            let narrow = start + r * hidden;
                            let block = &hidden_gates[narrow * 3 + CANDIDATE * hidden..][..hidden];
                            for j in 0..hidden {
                                out_row[j] = MaybeUninit::new(block[j]);
                            }
                        }
                    });
                })
            };

            // n = tanh(from_input_n + r * candidate_hidden), gathered contiguous
            // so the `tanh` is one call for the batch.
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
                            let projected = &candidate_hidden[narrow..][..hidden];
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

            (
                TensorData::from_vec::<$ty>(gates, dtype, device),
                TensorData::from_vec::<$ty>(candidate, dtype, device),
                TensorData::from_vec::<$ty>(candidate_hidden, dtype, device),
                TensorData::from_vec::<$ty>(hidden_after, dtype, device),
            )
        }};
    }

    let (gates, candidate, candidate_hidden, hidden_after) = match dtype {
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
        narrow_shape.clone(),
        dtype,
        device,
        requires_grad,
    );

    if !requires_grad {
        return Ok(new_h);
    }

    let saved = Saved {
        gates: Tensor::new(
            Arc::new(gates),
            Shape::new(vec![batch, 2 * hidden]),
            dtype,
            device,
            false,
        ),
        candidate: Tensor::new(
            Arc::new(candidate),
            narrow_shape.clone(),
            dtype,
            device,
            false,
        ),
        candidate_hidden: Tensor::new(
            Arc::new(candidate_hidden),
            narrow_shape,
            dtype,
            device,
            false,
        ),
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
        let batch = self.saved.candidate.shape().dims()[0];
        let dtype = grad_output.dtype();
        let device = grad_output.device();
        let wanted = self.wanted;

        macro_rules! run {
            ($ty:ty, $slice:ident) => {{
                let incoming = grad_output.data().$slice().ok_or_else(|| {
                    MinitensorError::internal_error("gru_cell backward: gradient dtype")
                })?;
                let gates = self.saved.gates.data().$slice().ok_or_else(|| {
                    MinitensorError::internal_error("gru_cell backward: saved gates")
                })?;
                let candidate = self.saved.candidate.data().$slice().ok_or_else(|| {
                    MinitensorError::internal_error("gru_cell backward: saved candidate")
                })?;
                let projected = self.saved.candidate_hidden.data().$slice().ok_or_else(|| {
                    MinitensorError::internal_error("gru_cell backward: saved projection")
                })?;
                let previous = self.saved.hidden_before.data().$slice().ok_or_else(|| {
                    MinitensorError::internal_error("gru_cell backward: saved hidden state")
                })?;

                // Both projections receive *the same* gradient through `r`
                // and through `z`, whose pre-activations are plain sums of the
                // two. They part company in one block of the three: the
                // candidate's, where the input's term is `dq` and the hidden's
                // is `dq * r`. So the hidden's buffer is built *from* the
                // input's -- two thirds of it a copy -- rather than by
                // deriving both sigmoids a second time.
                //
                // Row slices rather than indexing, for the reason the first
                // version of this file is worth remembering: written as a
                // closure taking `(row, column)` and indexing five buffers per
                // element, nothing vectorized, and the backward was 59% slower
                // than the composed form at T=16 while still winning at T=256.
                // Cache traffic is what fusing buys back; it does not also pay
                // for scalar arithmetic.
                // SAFETY: the three blocks tile each row and every element of
                // each is written; the rows tile the output.
                let build_input = || -> Vec<$ty> {
                    unsafe {
                        build_vec(batch * 3 * hidden, |spare| {
                            if hidden == 0 {
                                return;
                            }
                            let width = band_width(spare.len(), 3 * hidden, SIMD_PAR_THRESHOLD);
                            par_out_chunks(spare, width, &|start, band| {
                                for (row, row_out) in band.chunks_exact_mut(3 * hidden).enumerate()
                                {
                                    let narrow = (start + row * 3 * hidden) / 3;
                                    let r = &gates[narrow * 2 + RESET * hidden..][..hidden];
                                    let z = &gates[narrow * 2 + UPDATE * hidden..][..hidden];
                                    let n = &candidate[narrow..][..hidden];
                                    let g = &incoming[narrow..][..hidden];
                                    let prev = &previous[narrow..][..hidden];
                                    let projected_row = &projected[narrow..][..hidden];
                                    let (d_reset, rest) = row_out.split_at_mut(hidden);
                                    let (d_update, d_candidate) = rest.split_at_mut(hidden);

                                    for j in 0..hidden {
                                        // h' = (1 - z) * n + z * h, then back
                                        // through n = tanh(ni + r * nh).
                                        let d_pre_n = g[j] * (1.0 - z[j]) * (1.0 - n[j] * n[j]);
                                        d_reset[j] = MaybeUninit::new(
                                            d_pre_n * projected_row[j] * r[j] * (1.0 - r[j]),
                                        );
                                        d_update[j] = MaybeUninit::new(
                                            g[j] * (prev[j] - n[j]) * z[j] * (1.0 - z[j]),
                                        );
                                        d_candidate[j] = MaybeUninit::new(d_pre_n);
                                    }
                                }
                            });
                        })
                    }
                };

                // SAFETY: as above.
                let build_hidden = |input: &[$ty]| -> Vec<$ty> {
                    unsafe {
                        build_vec(batch * 3 * hidden, |spare| {
                            if hidden == 0 {
                                return;
                            }
                            let width = band_width(spare.len(), 3 * hidden, SIMD_PAR_THRESHOLD);
                            par_out_chunks(spare, width, &|start, band| {
                                for (row, row_out) in band.chunks_exact_mut(3 * hidden).enumerate()
                                {
                                    let wide = start + row * 3 * hidden;
                                    let source = &input[wide..][..3 * hidden];
                                    let r = &gates[wide / 3 * 2 + RESET * hidden..][..hidden];
                                    let (shared, d_candidate) = row_out.split_at_mut(2 * hidden);
                                    for j in 0..2 * hidden {
                                        shared[j] = MaybeUninit::new(source[j]);
                                    }
                                    for j in 0..hidden {
                                        d_candidate[j] =
                                            MaybeUninit::new(source[2 * hidden + j] * r[j]);
                                    }
                                }
                            });
                        })
                    }
                };

                // When only the hidden projection wants a gradient the input's
                // is still built, as the source of the two thirds they share.
                // That case does not arise in a trained layer, where both come
                // from the same weights.
                let d_input = if wanted[0] || wanted[1] {
                    build_input()
                } else {
                    Vec::new()
                };
                let d_hidden = if wanted[1] {
                    build_hidden(&d_input)
                } else {
                    Vec::new()
                };
                let d_input = if wanted[0] { d_input } else { Vec::new() };

                // dL/dh comes back through the blend alone: `z * g`. The rest
                // of the previous state's influence is through the projection,
                // which is the matmul's business and not this node's.
                // SAFETY: the rows tile the output and each is written whole.
                let d_previous: Vec<$ty> = if wanted[2] {
                    unsafe {
                        build_vec(batch * hidden, |spare| {
                            if hidden == 0 {
                                return;
                            }
                            let width = band_width(spare.len(), hidden, SIMD_PAR_THRESHOLD);
                            par_out_chunks(spare, width, &|start, band| {
                                for (r, out_row) in band.chunks_exact_mut(hidden).enumerate() {
                                    let narrow = start + r * hidden;
                                    let update = &gates[narrow * 2 + UPDATE * hidden..][..hidden];
                                    let g = &incoming[narrow..][..hidden];
                                    for j in 0..hidden {
                                        out_row[j] = MaybeUninit::new(g[j] * update[j]);
                                    }
                                }
                            });
                        })
                    }
                } else {
                    Vec::new()
                };

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
        if wanted[0] {
            gradients.insert(
                self.ids[0],
                Tensor::new(Arc::new(d_input), wide.clone(), dtype, device, false),
            );
        }
        if wanted[1] {
            gradients.insert(
                self.ids[1],
                Tensor::new(Arc::new(d_hidden), wide, dtype, device, false),
            );
        }
        if wanted[2] {
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

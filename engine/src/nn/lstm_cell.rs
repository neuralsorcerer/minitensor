// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! The elementwise half of an LSTM step, as one operation.
//!
//! Composed from ordinary tensor operations, a step is four gate slices, four
//! activations, two products, a sum, a `tanh` and a final product -- thirteen
//! intermediate tensors, every one of them retained for the backward pass.
//! Measured on `hidden=128, batch=8`, that is what makes the backward grow
//! fourfold per timestep between `T=16` and `T=256`: the retained activations
//! leave cache. See the note on [`crate::nn::Recurrent`].
//!
//! This is the part worth fusing and the only part fused here. The matmul that
//! produces `gates` stays an ordinary operation, so the gradients to `w_hh`,
//! `b_hh` and the previous hidden state are still derived by the graph rather
//! than by hand -- which is where a recurrent backward usually goes wrong, and
//! is not a risk worth taking for arithmetic the existing machinery already
//! gets right.
//!
//! What is left to derive is one elementwise pass, and it is written out in
//! [`LstmCellBackward::backward`].
//!
//! # Why this is not one scalar loop
//!
//! The first version of this file was exactly that, and it lost: a fused
//! forward that computes `sigmoid` and `tanh` an element at a time is slower
//! than thirteen composed operations, because each of those thirteen runs
//! [`crate::ops::simd`]'s vectorized kernels. Fusing saves allocations and
//! graph nodes; it does not save enough to pay for giving up SIMD on the
//! transcendentals, which are most of the arithmetic here.
//!
//! So the forward is staged rather than fused elementwise. Gates are activated
//! in blocks -- input and forget are adjacent in the layout, so they take one
//! `sigmoid` call per row rather than two -- the new cell state is one
//! vectorizable product-and-sum, and its `tanh` is a single contiguous call
//! over the whole `(batch, hidden)` buffer.
//!
//! Every buffer is built once, into uninitialized capacity: the composed path
//! zeroed each intermediate and then overwrote it.
//!
//! # What this changes numerically, and what it does not
//!
//! The forward is **bit-for-bit** what the composed form produced. Float32 goes
//! through the same `F32Kernel` the elementwise `sigmoid` and `tanh` call, in
//! the same operand order (`f * c_prev + i * g`, then `o * tanh(c)`); float64
//! has no vectorized transcendental in either form and uses the same
//! `stable_sigmoid_f64`. Verified across four float32 shapes -- including a
//! `hidden` that is not a multiple of the SIMD width, and a bidirectional
//! two-layer stack -- and one float64 shape.
//!
//! The **backward is not**, and cannot be: it is one expression per gate where
//! the composed form was a chain of separate products, so the roundings fall in
//! different places. Measured over those same shapes the gradients agree to
//! 4e-6 absolute at float32 and 1e-14 relative at float64, which is
//! re-association and nothing else. What holds them to the true derivative
//! rather than to each other is `tests/nn/test_recurrent_parameter_gradients.py`,
//! which value-checks every parameter of eight configurations of these layers,
//! four of them LSTM, against central differences at float64.

use crate::{
    autograd::{GradientFunction, TensorId, with_grad_fn},
    error::{MinitensorError, Result},
    ops::{
        map::{
            EXPENSIVE_PAR_THRESHOLD, SIMD_PAR_CHUNK, SIMD_PAR_THRESHOLD, VECTOR_F32_PAR_THRESHOLD,
            build_vec, par_out_chunks, unary_map_blocks_threshold,
        },
        util::stable_sigmoid_f64,
    },
    tensor::{DataType, Shape, Tensor, TensorData},
};
use rustc_hash::FxHashMap;
use std::{mem::MaybeUninit, sync::Arc};

/// Gate order within the `4 * hidden` axis, matching the stored weights.
///
/// Input and forget being adjacent is what lets the forward activate them in
/// one call; everything here indexes by these names rather than relying on it.
const INPUT: usize = 0;
const FORGET: usize = 1;
const CANDIDATE: usize = 2;
const OUTPUT: usize = 3;

/// The chunk width to hand [`par_out_chunks`] for `total` elements laid out in
/// rows of `row_width`.
///
/// Below `threshold` the whole buffer is one chunk, which `par_out_chunks`
/// runs inline without reaching rayon at all. That is the case that matters:
/// a recurrent step's tensors are small by construction, and forking a pool
/// once per timestep would cost more than the arithmetic it splits. The
/// thresholds are the ones the composed operations used, so the point at which
/// this starts using every core is the point at which they did.
///
/// Above it the bands are whole rows. Which gate an element belongs to is
/// decided by where it sits within its row, so a split anywhere else would
/// hand a task half of one gate and half of the next.
#[inline]
fn band_width(total: usize, row_width: usize, threshold: usize) -> usize {
    if total < threshold || row_width == 0 {
        return total;
    }
    row_width * (SIMD_PAR_CHUNK / row_width).max(1)
}

/// What the backward needs, kept as one buffer rather than four tensors.
///
/// `activations` holds the *activated* gates in the same `(batch, 4 * hidden)`
/// layout as the input, so the backward reads them with the same indexing.
struct Saved {
    activations: Tensor,
    tanh_c: Tensor,
    cell_before: Tensor,
}

/// One LSTM step's elementwise part: gates and the previous cell state in,
/// the new hidden and cell states out.
///
/// `gates` is `(batch, 4 * hidden)` and already carries both projections and
/// both biases; `cell_before` is `(batch, hidden)`.
pub fn lstm_cell(gates: &Tensor, cell_before: &Tensor) -> Result<(Tensor, Tensor)> {
    let dims = gates.shape().dims();
    if dims.len() != 2 || !dims[1].is_multiple_of(4) {
        return Err(MinitensorError::invalid_operation(
            "lstm_cell expects gates shaped [batch, 4 * hidden]",
        ));
    }
    let batch = dims[0];
    let hidden = dims[1] / 4;

    if cell_before.shape().dims() != [batch, hidden] {
        return Err(MinitensorError::invalid_operation(format!(
            "lstm_cell: cell state is {:?}, expected [{batch}, {hidden}]",
            cell_before.shape().dims()
        )));
    }
    if gates.dtype() != cell_before.dtype() {
        return Err(MinitensorError::invalid_operation(
            "lstm_cell: gates and cell state must share a dtype",
        ));
    }

    let requires_grad = gates.requires_grad() || cell_before.requires_grad();
    let wide = Shape::new(vec![batch, 4 * hidden]);
    let narrow_shape = Shape::new(vec![batch, hidden]);
    let device = gates.device();
    let dtype = gates.dtype();

    macro_rules! run {
        ($ty:ty, $slice:ident, $par:expr, $sigmoid_block:expr, $tanh_block:expr) => {{
            let gate_values = gates.data().$slice().ok_or_else(|| {
                MinitensorError::internal_error("lstm_cell: gates are not the declared dtype")
            })?;
            let previous = cell_before.data().$slice().ok_or_else(|| {
                MinitensorError::internal_error("lstm_cell: cell state is not the declared dtype")
            })?;
            let sigmoid_block = $sigmoid_block;
            let tanh_block = $tanh_block;
            let wide_row = 4 * hidden;

            // Gate activations. `INPUT` and `FORGET` are adjacent, so the two
            // sigmoids over them are one call over `2 * hidden`.
            // SAFETY: the three block calls tile each row exactly and each
            // initializes every element it is given; the bands tile the output.
            let activations: Vec<$ty> = unsafe {
                build_vec(batch * wide_row, |spare| {
                    if hidden == 0 {
                        return;
                    }
                    let width = band_width(spare.len(), wide_row, $par);
                    par_out_chunks(spare, width, &|start, band| {
                        for (r, row_out) in band.chunks_exact_mut(wide_row).enumerate() {
                            let row_in = &gate_values[start + r * wide_row..][..wide_row];
                            let (gated, rest) = row_out.split_at_mut(2 * hidden);
                            let (candidate, output) = rest.split_at_mut(hidden);
                            sigmoid_block(&row_in[INPUT * hidden..(FORGET + 1) * hidden], gated);
                            tanh_block(
                                &row_in[CANDIDATE * hidden..(CANDIDATE + 1) * hidden],
                                candidate,
                            );
                            sigmoid_block(&row_in[OUTPUT * hidden..], output);
                        }
                    });
                })
            };

            // c = f * c_prev + i * g. Written as one expression rather than
            // `add(mul(..), mul(..))`, which is the same three roundings in the
            // same order -- Rust does not contract a multiply and an add into
            // an FMA on its own.
            // SAFETY: every element of every row is written, and the rows tile.
            let cell_after: Vec<$ty> = unsafe {
                build_vec(batch * hidden, |spare| {
                    if hidden == 0 {
                        return;
                    }
                    let width = band_width(spare.len(), hidden, SIMD_PAR_THRESHOLD);
                    par_out_chunks(spare, width, &|start, band| {
                        for (r, out_row) in band.chunks_exact_mut(hidden).enumerate() {
                            let narrow = start + r * hidden;
                            let act = &activations[narrow * 4..][..wide_row];
                            let i = &act[INPUT * hidden..(INPUT + 1) * hidden];
                            let f = &act[FORGET * hidden..(FORGET + 1) * hidden];
                            let g = &act[CANDIDATE * hidden..(CANDIDATE + 1) * hidden];
                            let prev = &previous[narrow..][..hidden];
                            for j in 0..hidden {
                                out_row[j] = MaybeUninit::new(f[j] * prev[j] + i[j] * g[j]);
                            }
                        }
                    });
                })
            };

            // One contiguous call over the whole new cell state.
            // SAFETY: `tanh_block` initializes every element it is given.
            let tanh_cell: Vec<$ty> = unsafe {
                unary_map_blocks_threshold(&cell_after, $par, |src, dst| tanh_block(src, dst))
            };

            // h = o * tanh(c)
            // SAFETY: as for `cell_after`.
            let hidden_after: Vec<$ty> = unsafe {
                build_vec(batch * hidden, |spare| {
                    if hidden == 0 {
                        return;
                    }
                    let width = band_width(spare.len(), hidden, SIMD_PAR_THRESHOLD);
                    par_out_chunks(spare, width, &|start, band| {
                        for (r, out_row) in band.chunks_exact_mut(hidden).enumerate() {
                            let narrow = start + r * hidden;
                            let o = &activations[narrow * 4 + OUTPUT * hidden..][..hidden];
                            let tanh_row = &tanh_cell[narrow..][..hidden];
                            for j in 0..hidden {
                                out_row[j] = MaybeUninit::new(o[j] * tanh_row[j]);
                            }
                        }
                    });
                })
            };

            (
                TensorData::from_vec::<$ty>(activations, dtype, device),
                TensorData::from_vec::<$ty>(cell_after, dtype, device),
                TensorData::from_vec::<$ty>(tanh_cell, dtype, device),
                TensorData::from_vec::<$ty>(hidden_after, dtype, device),
            )
        }};
    }

    let (activations, cell_after, tanh_cell, hidden_after) = match dtype {
        DataType::Float32 => {
            // Resolved once per step rather than per block, as the elementwise
            // kernels do. These are the same kernels the composed path called.
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
            |src: &[f64], dst: &mut [MaybeUninit<f64>]| {
                for (out, &x) in dst.iter_mut().zip(src) {
                    out.write(stable_sigmoid_f64(x));
                }
            },
            |src: &[f64], dst: &mut [MaybeUninit<f64>]| {
                for (out, &x) in dst.iter_mut().zip(src) {
                    out.write(x.tanh());
                }
            }
        ),
        other => {
            return Err(MinitensorError::invalid_operation(format!(
                "lstm_cell requires a floating point dtype, got {other:?}"
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
    let new_c = Tensor::new(
        Arc::new(cell_after),
        narrow_shape.clone(),
        dtype,
        device,
        requires_grad,
    );

    if !requires_grad {
        return Ok((new_h, new_c));
    }

    let saved = Arc::new(Saved {
        activations: Tensor::new(Arc::new(activations), wide, dtype, device, false),
        tanh_c: Tensor::new(Arc::new(tanh_cell), narrow_shape, dtype, device, false),
        cell_before: cell_before.detach(),
    });

    let ids = [gates.id(), cell_before.id()];
    let make = |from_hidden: bool| {
        Arc::new(LstmCellBackward {
            saved: Arc::clone(&saved),
            from_hidden,
            hidden,
            ids,
            wanted: [gates.requires_grad(), cell_before.requires_grad()],
        }) as Arc<dyn GradientFunction>
    };

    // One node per output, the way `qr` does it. Everything below `dc` is
    // linear in `dc`, so the two nodes' contributions sum to the same
    // gradient a single combined node would have produced.
    let new_h = with_grad_fn(new_h, make(true))?;
    let new_c = with_grad_fn(new_c, make(false))?;
    Ok((new_h, new_c))
}

/// The backward for one output of [`lstm_cell`].
struct LstmCellBackward {
    saved: Arc<Saved>,
    /// Whether this node stands for the hidden state or the cell state.
    from_hidden: bool,
    hidden: usize,
    ids: [TensorId; 2],
    /// Whether each input wanted a gradient at all.
    wanted: [bool; 2],
}

impl GradientFunction for LstmCellBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let hidden = self.hidden;
        let dims = self.saved.tanh_c.shape().dims();
        let batch = dims[0];
        let dtype = grad_output.dtype();
        let device = grad_output.device();
        let from_hidden = self.from_hidden;
        let wanted = self.wanted;

        macro_rules! run {
            ($ty:ty, $slice:ident) => {{
                let incoming = grad_output.data().$slice().ok_or_else(|| {
                    MinitensorError::internal_error("lstm_cell backward: gradient dtype")
                })?;
                let activations = self.saved.activations.data().$slice().ok_or_else(|| {
                    MinitensorError::internal_error("lstm_cell backward: saved gates")
                })?;
                let tanh_c = self.saved.tanh_c.data().$slice().ok_or_else(|| {
                    MinitensorError::internal_error("lstm_cell backward: saved tanh")
                })?;
                let cell_before = self.saved.cell_before.data().$slice().ok_or_else(|| {
                    MinitensorError::internal_error("lstm_cell backward: saved cell state")
                })?;
                let wide_row = 4 * hidden;

                // `h = o * tanh(c)` sends gradient to `o` and to `c`; `c`
                // receives it directly. Each node carries only its own output's
                // term, and `from_hidden` is the same for every element, so the
                // branch is outside the inner loops.
                //
                // Both passes recompute it. They have different row widths and
                // so cannot be partitioned together, and four operations per
                // element is cheaper than the buffer that would carry them.
                let grad_c_at = |o: $ty, tc: $ty, incoming: $ty| -> $ty {
                    if from_hidden {
                        incoming * o * (1.0 - tc * tc)
                    } else {
                        incoming
                    }
                };

                // dL/d(c_prev): `c = f * c_prev + i * g`. Skipped outright when
                // the previous cell state is a constant, which is every
                // sequence's first step.
                // SAFETY: every element of every row is written, rows tile.
                let d_cell: Vec<$ty> = if wanted[1] {
                    unsafe {
                        build_vec(batch * hidden, |spare| {
                            if hidden == 0 {
                                return;
                            }
                            let width = band_width(spare.len(), hidden, SIMD_PAR_THRESHOLD);
                            par_out_chunks(spare, width, &|start, band| {
                                for (r, out_row) in band.chunks_exact_mut(hidden).enumerate() {
                                    let narrow = start + r * hidden;
                                    let act = &activations[narrow * 4..][..wide_row];
                                    let f = &act[FORGET * hidden..(FORGET + 1) * hidden];
                                    let o = &act[OUTPUT * hidden..];
                                    let tc = &tanh_c[narrow..][..hidden];
                                    let inc = &incoming[narrow..][..hidden];
                                    for j in 0..hidden {
                                        out_row[j] =
                                            MaybeUninit::new(grad_c_at(o[j], tc[j], inc[j]) * f[j]);
                                    }
                                }
                            });
                        })
                    }
                } else {
                    Vec::new()
                };

                // dL/d(gates), back through each gate's own activation.
                // SAFETY: the four blocks tile each row and every element of
                // each is written; the rows tile the output.
                let d_gates: Vec<$ty> = if wanted[0] {
                    unsafe {
                        build_vec(batch * wide_row, |spare| {
                            if hidden == 0 {
                                return;
                            }
                            let width = band_width(spare.len(), wide_row, SIMD_PAR_THRESHOLD);
                            par_out_chunks(spare, width, &|start, band| {
                                for (r, row_out) in band.chunks_exact_mut(wide_row).enumerate() {
                                    let wide = start + r * wide_row;
                                    let narrow = wide / 4;
                                    let act = &activations[wide..][..wide_row];
                                    let i = &act[INPUT * hidden..(INPUT + 1) * hidden];
                                    let f = &act[FORGET * hidden..(FORGET + 1) * hidden];
                                    let g = &act[CANDIDATE * hidden..(CANDIDATE + 1) * hidden];
                                    let o = &act[OUTPUT * hidden..];
                                    let tc = &tanh_c[narrow..][..hidden];
                                    let before = &cell_before[narrow..][..hidden];
                                    let inc = &incoming[narrow..][..hidden];

                                    let (d_i, rest) = row_out.split_at_mut(hidden);
                                    let (d_f, rest) = rest.split_at_mut(hidden);
                                    let (d_g, d_o) = rest.split_at_mut(hidden);

                                    for j in 0..hidden {
                                        let grad_c = grad_c_at(o[j], tc[j], inc[j]);
                                        let grad_o = if from_hidden { inc[j] * tc[j] } else { 0.0 };

                                        d_i[j] =
                                            MaybeUninit::new(grad_c * g[j] * i[j] * (1.0 - i[j]));
                                        d_f[j] = MaybeUninit::new(
                                            grad_c * before[j] * f[j] * (1.0 - f[j]),
                                        );
                                        d_g[j] =
                                            MaybeUninit::new(grad_c * i[j] * (1.0 - g[j] * g[j]));
                                        d_o[j] = MaybeUninit::new(grad_o * o[j] * (1.0 - o[j]));
                                    }
                                }
                            });
                        })
                    }
                } else {
                    Vec::new()
                };

                (
                    TensorData::from_vec::<$ty>(d_gates, dtype, device),
                    TensorData::from_vec::<$ty>(d_cell, dtype, device),
                )
            }};
        }

        let (d_gates, d_cell) = match dtype {
            DataType::Float32 => run!(f32, as_f32_slice),
            DataType::Float64 => run!(f64, as_f64_slice),
            other => {
                return Err(MinitensorError::invalid_operation(format!(
                    "lstm_cell backward requires a floating point dtype, got {other:?}"
                )));
            }
        };

        let mut gradients = FxHashMap::default();
        if wanted[0] {
            gradients.insert(
                self.ids[0],
                Tensor::new(
                    Arc::new(d_gates),
                    Shape::new(vec![batch, 4 * hidden]),
                    dtype,
                    device,
                    false,
                ),
            );
        }
        if wanted[1] {
            gradients.insert(
                self.ids[1],
                Tensor::new(
                    Arc::new(d_cell),
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

    /// The forward, against the definition rather than against itself.
    ///
    /// `test_recurrent_parameter_gradients.py` checks the *derivative* to a
    /// central difference, which a forward that is consistently wrong would
    /// still satisfy. This is the other half: two rows of arbitrary gate
    /// values, worked through `i, f, g, o` by hand.
    #[test]
    fn one_step_matches_the_definition() {
        let hidden = 2;
        // Row 0 gates, then row 1: [i0 i1 f0 f1 g0 g1 o0 o1].
        let gate_values = vec![
            0.5, -1.0, 2.0, 0.25, -0.75, 1.5, 0.1, -0.3, //
            -2.0, 0.8, 1.25, -0.5, 0.6, -1.75, 2.5, 0.0,
        ];
        let previous = vec![0.3, -0.7, 1.1, -0.2];

        let (h, c) = lstm_cell(
            &tensor_f64(gate_values.clone(), vec![2, 4 * hidden]),
            &tensor_f64(previous.clone(), vec![2, hidden]),
        )
        .expect("well-formed inputs");

        let sigmoid = |x: f64| 1.0 / (1.0 + (-x).exp());
        for b in 0..2 {
            for j in 0..hidden {
                let row = b * 4 * hidden;
                let i = sigmoid(gate_values[row + j]);
                let f = sigmoid(gate_values[row + hidden + j]);
                let g = gate_values[row + 2 * hidden + j].tanh();
                let o = sigmoid(gate_values[row + 3 * hidden + j]);
                let expected_c = f * previous[b * hidden + j] + i * g;
                let expected_h = o * expected_c.tanh();

                let at = b * hidden + j;
                let got_c = c.data().as_f64_slice().unwrap()[at];
                let got_h = h.data().as_f64_slice().unwrap()[at];
                assert!(
                    (got_c - expected_c).abs() < 1e-12,
                    "cell[{b}][{j}]: {got_c} vs {expected_c}"
                );
                assert!(
                    (got_h - expected_h).abs() < 1e-12,
                    "hidden[{b}][{j}]: {got_h} vs {expected_h}"
                );
            }
        }
    }

    /// A batch wide enough to cross [`band_width`]'s threshold has to produce
    /// what the same batch produces below it. The bands are cut on row
    /// boundaries for that reason, and a split anywhere else would hand a task
    /// half of one gate and half of the next -- which would not fail loudly,
    /// it would activate the wrong elements with the wrong function.
    #[test]
    fn the_parallel_split_changes_nothing() {
        let hidden = 24; // not a multiple of any SIMD width
        let batch = 512; // 512 * 96 = 49_152 elements, past the threshold
        let gates: Vec<f32> = (0..batch * 4 * hidden)
            .map(|i| ((i % 37) as f32 - 18.0) / 6.0)
            .collect();
        let previous: Vec<f32> = (0..batch * hidden)
            .map(|i| ((i % 19) as f32 - 9.0) / 4.0)
            .collect();

        assert!(
            batch * 4 * hidden >= VECTOR_F32_PAR_THRESHOLD,
            "this test is only meaningful above the threshold"
        );
        let (h, c) = lstm_cell(
            &tensor_f32(gates.clone(), vec![batch, 4 * hidden]),
            &tensor_f32(previous.clone(), vec![batch, hidden]),
        )
        .unwrap();

        // The same rows, one at a time, which never reaches the parallel path.
        for b in 0..batch {
            let row = &gates[b * 4 * hidden..][..4 * hidden];
            let prev = &previous[b * hidden..][..hidden];
            let (row_h, row_c) = lstm_cell(
                &tensor_f32(row.to_vec(), vec![1, 4 * hidden]),
                &tensor_f32(prev.to_vec(), vec![1, hidden]),
            )
            .unwrap();
            assert_eq!(
                &h.data().as_f32_slice().unwrap()[b * hidden..][..hidden],
                row_h.data().as_f32_slice().unwrap(),
                "hidden state of row {b}"
            );
            assert_eq!(
                &c.data().as_f32_slice().unwrap()[b * hidden..][..hidden],
                row_c.data().as_f32_slice().unwrap(),
                "cell state of row {b}"
            );
        }
    }

    /// `chunks_exact(0)` panics, and both degenerate shapes reach it.
    #[test]
    fn degenerate_shapes_produce_empty_outputs_rather_than_panicking() {
        for (batch, hidden) in [(0usize, 3usize), (4, 0), (0, 0)] {
            let (h, c) = lstm_cell(
                &tensor_f64(vec![0.0; batch * 4 * hidden], vec![batch, 4 * hidden]),
                &tensor_f64(vec![0.0; batch * hidden], vec![batch, hidden]),
            )
            .unwrap_or_else(|e| panic!("batch={batch} hidden={hidden}: {e}"));
            assert_eq!(h.shape().dims(), &[batch, hidden]);
            assert_eq!(c.shape().dims(), &[batch, hidden]);
        }
    }

    #[test]
    fn malformed_inputs_are_refused_rather_than_misread() {
        // Not two-dimensional.
        assert!(
            lstm_cell(
                &tensor_f64(vec![0.0; 8], vec![8]),
                &tensor_f64(vec![0.0; 2], vec![2])
            )
            .is_err()
        );
        // Gate axis is not a multiple of four, so there is no gate layout.
        assert!(
            lstm_cell(
                &tensor_f64(vec![0.0; 6], vec![1, 6]),
                &tensor_f64(vec![0.0; 1], vec![1, 1])
            )
            .is_err()
        );
        // Cell state does not match the batch and hidden width implied above.
        assert!(
            lstm_cell(
                &tensor_f64(vec![0.0; 8], vec![1, 8]),
                &tensor_f64(vec![0.0; 3], vec![1, 3])
            )
            .is_err()
        );
        // Two dtypes.
        assert!(
            lstm_cell(
                &tensor_f64(vec![0.0; 8], vec![1, 8]),
                &tensor_f32(vec![0.0; 2], vec![1, 2])
            )
            .is_err()
        );
        // An integer dtype has no derivative and no `tanh`.
        let ints = Tensor::new(
            Arc::new(TensorData::from_vec::<i64>(
                vec![0i64; 8],
                DataType::Int64,
                Device::cpu(),
            )),
            Shape::new(vec![1, 8]),
            DataType::Int64,
            Device::cpu(),
            false,
        );
        let int_cell = Tensor::new(
            Arc::new(TensorData::from_vec::<i64>(
                vec![0i64; 2],
                DataType::Int64,
                Device::cpu(),
            )),
            Shape::new(vec![1, 2]),
            DataType::Int64,
            Device::cpu(),
            false,
        );
        assert!(lstm_cell(&ints, &int_cell).is_err());
    }
}

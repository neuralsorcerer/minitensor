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
//! The **backward is not**. Its slopes are taken of the cell's inputs -- the
//! gate logits and the cell state before -- where the composed form took them
//! of the activations it had rounded: `i (1 - i)`, `1 - g^2`, `1 - tanh(c)^2`.
//! Each of those is a difference of numbers near one once its gate saturates,
//! losing digits on the way there and zero from a logit of 17 in float32 (of
//! 9 for a `tanh`) while the slope is still a float; with ordinary logits the
//! gradients were up to a million ulps out. Every value and slope is now
//! formed again in float64 from `e^-|x|` or `e^-2|x|` (see
//! [`crate::nn::cell::logistic`]), `c` from them, and each gradient is their
//! product rounded once: for float32 the float nearest the exact one, for
//! float64 within a few ulps -- more where the cell state is large, since
//! `tanh'(c)` moves by `2|c|` times whatever `c` is out by. Each of the two
//! outputs' nodes rounds its own part, and the graph adds the two.
//! `tests/nn/test_recurrent_cell_slopes.py` holds them to exact values, and
//! `tests/nn/test_recurrent_parameter_gradients.py` checks every parameter of
//! eight configurations of these layers against central differences.

use crate::{
    autograd::{GradientFunction, TensorId, with_grad_fn},
    error::{MinitensorError, Result},
    nn::cell::{
        band_width, finite_or_zero, logistic, logistic_pair, logistic_scaled, par_row_bands,
        product_pair, sigmoid_block_f64, tanh_and_slope, tanh_and_slope_near, tanh_block_f64,
        tanh_pair, tanh_scaled,
    },
    ops::arithmetic::exact_sum::Output,
    ops::arithmetic::extended::{Pair, Scaled},
    ops::map::{
        EXPENSIVE_PAR_THRESHOLD, SIMD_PAR_THRESHOLD, VECTOR_F32_PAR_THRESHOLD, build_vec,
        par_out_chunks, unary_map_blocks_threshold,
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
// The forward activates `INPUT..=FORGET` as one block of `2 * hidden` and
// `OUTPUT` as the tail of the row, and the float64 block kernels fill their
// output by zipping it with their input -- so a reordering here would leave
// elements unwritten rather than fail. Checked at compile time instead.
const _: () = assert!(FORGET == INPUT + 1 && CANDIDATE == FORGET + 1 && OUTPUT == 3);

/// What the backward needs: the cell's two inputs as they were given.
///
/// The slopes are taken of the gate logits and the cell state before, not of
/// the activations the forward rounded, so those are not kept: the backward
/// forms every value again, in float64.
struct Saved {
    gates: Tensor,
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
                TensorData::from_vec::<$ty>(cell_after, dtype, device),
                TensorData::from_vec::<$ty>(hidden_after, dtype, device),
            )
        }};
    }

    let (cell_after, hidden_after) = match dtype {
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
            sigmoid_block_f64,
            tanh_block_f64
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
        narrow_shape,
        dtype,
        device,
        requires_grad,
    );

    if !requires_grad {
        return Ok((new_h, new_c));
    }

    let saved = Arc::new(Saved {
        gates: gates.detach(),
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
    // linear in `dc`, so the two nodes' contributions sum to the gradient a
    // single combined node would produce, each part rounded on its own.
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
        let batch = self.saved.cell_before.shape().dims()[0];
        let dtype = grad_output.dtype();
        let device = grad_output.device();

        macro_rules! run {
            ($ty:ty, $slice:ident) => {{
                let incoming = grad_output.data().$slice().ok_or_else(|| {
                    MinitensorError::internal_error("lstm_cell backward: gradient dtype")
                })?;
                let gates = self.saved.gates.data().$slice().ok_or_else(|| {
                    MinitensorError::internal_error("lstm_cell backward: saved gates")
                })?;
                let cell_before = self.saved.cell_before.data().$slice().ok_or_else(|| {
                    MinitensorError::internal_error("lstm_cell backward: saved cell state")
                })?;
                let (d_gates, d_cell) = backward_rows::<$ty>(
                    gates,
                    cell_before,
                    incoming,
                    hidden,
                    self.from_hidden,
                    self.wanted,
                );
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
        if self.wanted[0] {
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
        if self.wanted[1] {
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

/// How far past one an ulp of a float64 term may be magnified before a
/// gradient in `T` shows it, and so before a term is carried as a pair
/// instead: not at all for float64 (`wide`), and sixteen-fold for float32,
/// whose gradients round 29 bits above float64's last place -- which a term
/// out by sixteen of those still leaves on the same side of every halfway
/// point but for one case in some millions. For the same reason a float32
/// gradient's products and sums need not carry their own roundings beside
/// them.
pub(crate) fn slack(wide: bool) -> f64 {
    if wide { 1.0 } else { 16.0 }
}

/// `x * y`, with its rounding beside it where `wide` says the gradient's
/// precision would show it.
#[inline(always)]
pub(crate) fn product(x: f64, y: f64, wide: bool) -> Pair {
    if wide {
        product_pair(x, y)
    } else {
        Pair(x * y, 0.0)
    }
}

/// The gradients of one output of [`lstm_cell`] -- of the gate logits and of
/// the cell state before, each empty unless `wanted` -- for the incoming
/// gradient `incoming` of the hidden state (`from_hidden`) or the cell state.
///
/// One pass writes both, a row of each at a time: every element of them
/// needs the same values and slopes, which cost a few exponentials each, so
/// they are formed once per element rather than once per output.
fn backward_rows<T: Output>(
    gates: &[T],
    cell_before: &[T],
    incoming: &[T],
    hidden: usize,
    from_hidden: bool,
    wanted: [bool; 2],
) -> (Vec<T>, Vec<T>) {
    let batch = cell_before.len() / hidden.max(1);
    let wide_row = 4 * hidden;
    let wide = T::DIGITS > 24;
    let widths = [
        if wanted[0] { wide_row } else { 0 },
        if wanted[1] { hidden } else { 0 },
    ];
    let mut d_gates = Vec::with_capacity(batch * widths[0]);
    let mut d_cell = Vec::with_capacity(batch * widths[1]);
    par_row_bands(
        [
            &mut d_gates.spare_capacity_mut()[..batch * widths[0]],
            &mut d_cell.spare_capacity_mut()[..batch * widths[1]],
        ],
        widths,
        batch,
        EXPENSIVE_PAR_THRESHOLD,
        &|first, count, [gate_rows, cell_rows]| {
            for row in 0..count {
                let at = first + row;
                let logits = &gates[at * wide_row..][..wide_row];
                let before = &cell_before[at * hidden..][..hidden];
                let grad = &incoming[at * hidden..][..hidden];
                for j in 0..hidden {
                    let x =
                        [INPUT, FORGET, CANDIDATE, OUTPUT].map(|g| logits[g * hidden + j].widen());
                    let (d_x, d_before) =
                        unit_backward(x, before[j].widen(), grad[j].widen(), from_hidden, wide);
                    if wanted[0] {
                        for (g, d) in [INPUT, FORGET, CANDIDATE, OUTPUT].into_iter().zip(d_x) {
                            gate_rows[row * wide_row + g * hidden + j].write(T::narrow(d));
                        }
                    }
                    if wanted[1] {
                        cell_rows[row * hidden + j].write(T::narrow(d_before));
                    }
                }
            }
        },
    );
    // SAFETY: every row of a wanted output was written whole above, and an
    // output nobody wants is empty.
    unsafe {
        d_gates.set_len(batch * widths[0]);
        d_cell.set_len(batch * widths[1]);
    }
    (d_gates, d_cell)
}

/// One hidden unit's backward, in float64: from its gate logits `x`, its cell
/// state before and the incoming gradient, the gradients of the four logits
/// and of the cell state before.
///
/// `c = f c_before + i g` and `h = o tanh(c)`, so `dc` is the incoming
/// gradient for the cell state's node and `grad o tanh'(c)` for the hidden
/// state's; then `di = dc g i'`, `df = dc c_before f'`, `dg = dc i g'`,
/// `do = grad tanh(c) o'` for the hidden state alone, and `dc_before = dc f`.
/// The bounded factors are multiplied first, so that a product leaves the
/// range only where the gradient itself does.
///
/// `tanh'(c)` moves by `2|c|` times whatever `c` is out by, so `c` is taken
/// as a pair: both products' roundings beside it, and -- where `f c_before`
/// is beyond [`slack`] in size, so that an ulp of a float64 `f` is many of
/// `c`'s last places -- `f` carried as a pair too. Without it a float64 cell
/// state of 30 left the slope some 50 ulps out. Where the two products
/// cancel, `c` is small beside what float64 values of `f`, `i` and `g` are out
/// by, and so is `tanh(c)`: there all three are carried as pairs.
#[inline(always)]
fn unit_backward(
    x: [f64; 4],
    before: f64,
    grad: f64,
    from_hidden: bool,
    wide: bool,
) -> ([f64; 4], f64) {
    let slack = slack(wide);
    let (i, _, di) = logistic(x[INPUT]);
    let (f, _, df) = logistic(x[FORGET]);
    let (g, dg) = tanh_and_slope(x[CANDIDATE]);
    let mut c = Pair(0.0, 0.0);
    let (dc, d_o) = if from_hidden {
        let (o, _, slope_o) = logistic(x[OUTPUT]);
        let kept = product(f, before, wide);
        let fresh = product(i, g, wide);
        c = kept.add(fresh);
        let large = kept.0.abs() > slack;
        let cancels = c.0.abs() < (kept.0.abs() + fresh.0.abs()) / (2.0 * slack);
        if large || cancels {
            let fc = logistic_pair(Pair(x[FORGET], 0.0)).mul::<false>(Pair(before, 0.0));
            c = if cancels {
                let i = logistic_pair(Pair(x[INPUT], 0.0));
                fc.add(i.mul::<false>(tanh_pair(Pair(x[CANDIDATE], 0.0))))
            } else {
                fc.add(fresh)
            };
        }
        c.1 = finite_or_zero(c.1);
        let (tanh_c, slope_c) = tanh_and_slope_near(c.0, c.1);
        (grad * (o * slope_c), grad * (tanh_c * slope_o))
    } else {
        // The cell state does not depend on the output gate at all.
        (grad, 0.0)
    };
    let far = |v: f64, limit: f64| v.abs() > limit;
    if wide
        && (far(x[INPUT], 708.0)
            || far(x[FORGET], 708.0)
            || far(x[CANDIDATE], 354.0)
            || (from_hidden && (far(x[OUTPUT], 708.0) || far(c.0, 354.0))))
    {
        return far_unit_backward(x, before, grad, from_hidden, c);
    }
    (
        [dc * (g * di), dc * (before * df), dc * (i * dg), d_o],
        dc * f,
    )
}

/// [`unit_backward`] where a value or slope lies below float64's normal range
/// -- a logit past 708, or a cell state or candidate logit past 354 -- and
/// float64 holds it only to its last few bits, while the other factors may
/// bring a gradient back up where every bit shows: each factor kept as a
/// [`Scaled`], and each gradient rounded once.
#[cold]
#[inline(never)]
fn far_unit_backward(
    x: [f64; 4],
    before: f64,
    grad: f64,
    from_hidden: bool,
    c: Pair,
) -> ([f64; 4], f64) {
    let [i, _, di] = logistic_scaled(x[INPUT], 0.0);
    let [f, _, df] = logistic_scaled(x[FORGET], 0.0);
    let [g, dg] = tanh_scaled(x[CANDIDATE], 0.0);
    let grad = Scaled::of(grad);
    let (dc, d_o) = if from_hidden {
        let [o, _, slope_o] = logistic_scaled(x[OUTPUT], 0.0);
        let [tanh_c, slope_c] = tanh_scaled(c.0, c.1);
        (
            grad.times(o.times(slope_c)),
            grad.times(tanh_c.times(slope_o)).value(),
        )
    } else {
        (grad, 0.0)
    };
    (
        [
            dc.times(g.times(di)).value(),
            dc.times(Scaled::of(before).times(df)).value(),
            dc.times(i.times(dg)).value(),
            d_o,
        ],
        dc.times(f).value(),
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

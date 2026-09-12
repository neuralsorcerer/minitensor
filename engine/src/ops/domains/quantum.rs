// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Quantum-inspired state-vector kernels: gate application, measurement, tracing.
//!
//! A state vector over `q` qubits is `2^q` complex amplitudes. Applying a
//! single-qubit gate touches every amplitude exactly once, in pairs: the pair
//! that differ only in bit `t` of their index. That is a *butterfly* — the same
//! access pattern as one radix-2 stage of an FFT — and it is expressible with
//! elementwise tensor operations only by reshaping to `(2^{q-t-1}, 2, 2^t)`,
//! slicing, multiplying, and concatenating, which allocates several full copies
//! of the state to move numbers between two halves of one buffer.
//!
//! Written directly it is one pass with no intermediate at all, and it
//! parallelises perfectly: the pairs are disjoint, so every block of the output
//! is independent.
//!
//! # Representing complex numbers without a complex dtype
//!
//! The engine stores real tensors, so a state is `(..., 2^q, 2)`, with the last
//! axis holding `(real, imaginary)`. Interleaved rather than split into two
//! planes, because both parts of an amplitude are always used together: a
//! complex multiply reads all four inputs, and interleaving puts them in the
//! same cache line.
//!
//! Leading axes are a batch, so a whole ensemble of states evolves in one call.
//!
//! # Why these differentiate cleanly
//!
//! Gate application is linear in the state, so its derivative is the adjoint
//! gate applied to the incoming gradient — the same butterfly, run with
//! `U^dagger`. Measurement and tracing are quadratic (`|amplitude|^2`), so
//! theirs is `2 * conj(amplitude)` weighted by the incoming gradient. Both are
//! one pass, and neither needs the forward output saved.

use crate::{
    autograd::{GradientFunction, TensorId, with_grad_fn},
    error::{MinitensorError, Result},
    ops::map::{PAR_CHUNK, par_out_chunks},
    tensor::{DataType, Shape, Tensor, TensorData},
};
use rustc_hash::FxHashMap;
use std::sync::Arc;

/// A `2x2` complex gate, as eight reals: `[[a, b], [c, d]]` row-major, each
/// entry `(re, im)`.
pub type Gate1 = [f64; 8];

/// The Hadamard gate, the one every example starts with.
pub const HADAMARD: Gate1 = {
    const H: f64 = std::f64::consts::FRAC_1_SQRT_2;
    [H, 0.0, H, 0.0, H, 0.0, -H, 0.0]
};

/// Pauli-X: the quantum NOT.
pub const PAULI_X: Gate1 = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0];

/// Pauli-Z: a phase flip on `|1>`.
pub const PAULI_Z: Gate1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0];

/// The conjugate transpose of a gate, which is the gate its backward pass runs.
fn adjoint(gate: &Gate1) -> Gate1 {
    let [ar, ai, br, bi, cr, ci, dr, di] = *gate;
    // Transpose swaps b and c; conjugation negates every imaginary part.
    [ar, -ai, cr, -ci, br, -bi, dr, -di]
}

/// Check a state's shape and return `(batch, dim, qubits)`.
fn state_layout(state: &Tensor, op: &str) -> Result<(usize, usize, usize)> {
    let dims = state.shape().dims();
    if dims.len() < 2 || *dims.last().unwrap() != 2 {
        return Err(MinitensorError::invalid_operation(format!(
            "{op} expects a state shaped (..., 2^q, 2) with the last axis holding \
             (real, imaginary); got {dims:?}"
        )));
    }
    let dim = dims[dims.len() - 2];
    if dim == 0 || !dim.is_power_of_two() {
        return Err(MinitensorError::invalid_operation(format!(
            "{op} expects 2^q amplitudes, got {dim}"
        )));
    }
    let batch: usize = dims[..dims.len() - 2].iter().product();
    Ok((batch, dim, dim.trailing_zeros() as usize))
}

/// Apply a single-qubit gate to `target`, counting qubits from the least
/// significant bit of the amplitude index.
///
/// `state` is `(..., 2^q, 2)`; the result has the same shape.
pub fn apply_gate_1q(state: &Tensor, gate: &Gate1, target: usize) -> Result<Tensor> {
    let (batch, dim, qubits) = state_layout(state, "apply_gate_1q")?;
    if target >= qubits {
        return Err(MinitensorError::invalid_argument(format!(
            "apply_gate_1q: qubit {target} is out of range for a {qubits}-qubit state"
        )));
    }

    macro_rules! run {
        ($ty:ty, $slice:ident, $from_vec:path) => {{
            let input = state.data().$slice().ok_or_else(dtype_mismatch)?;
            let mut out = vec![<$ty>::default(); input.len()];
            butterfly::<$ty>(input, &mut out, batch, dim, target, gate);
            $from_vec(out, state.device())
        }};
    }

    let data = match state.dtype() {
        DataType::Float32 => run!(f32, as_f32_slice, TensorData::from_vec_f32),
        DataType::Float64 => run!(f64, as_f64_slice, TensorData::from_vec_f64),
        other => return Err(float_only("apply_gate_1q", other)),
    };

    let requires_grad = crate::autograd::is_grad_enabled() && state.requires_grad();
    let output = Tensor::new(
        Arc::new(data),
        Shape::new(state.shape().dims().to_vec()),
        state.dtype(),
        state.device(),
        requires_grad,
    );
    if !requires_grad {
        return Ok(output);
    }
    with_grad_fn(
        output,
        Arc::new(GateBackward {
            adjoint: adjoint(gate),
            target,
            batch,
            dim,
            input_id: state.id(),
        }),
    )
}

/// One butterfly pass: for every index pair differing in bit `target`, replace
/// `(a0, a1)` by `(g00 a0 + g01 a1, g10 a0 + g11 a1)`.
///
/// The output is split into blocks of whole `stride`-sized runs so that each
/// rayon task owns a contiguous span and every pair it needs lies inside it.
/// `stride` is `2^target`: indices differing only in bit `target` are exactly
/// `stride` apart, so the pairs within a `2 * stride` window are
/// `(i, i + stride)` for the first half of the window.
fn butterfly<T>(input: &[T], out: &mut [T], batch: usize, dim: usize, target: usize, gate: &Gate1)
where
    T: Copy + Default + Send + Sync + Into<f64> + FromF64,
{
    let [g00r, g00i, g01r, g01i, g10r, g10i, g11r, g11i] = *gate;
    let stride = 1usize << target;
    let window = stride * 2;
    // Two reals per amplitude, and a task should carry at least a cache line's
    // worth of windows.
    let per_state = dim * 2;
    let windows_per_task = (PAR_CHUNK / window).max(1);
    let chunk = (windows_per_task * window * 2).min(per_state).max(2);

    par_out_chunks(out, chunk, &|start, block| {
        // `start` counts reals; recover which state and which offset in it.
        for offset in (0..block.len()).step_by(2) {
            let flat = start + offset;
            let state_index = flat / per_state;
            if state_index >= batch {
                break;
            }
            let within = (flat % per_state) / 2;
            // Only the lower member of each pair drives a step; the upper is
            // written at the same time.
            if within & stride != 0 {
                continue;
            }
            let base = state_index * per_state;
            let lo = base + within * 2;
            let hi = base + (within + stride) * 2;

            let a0r: f64 = input[lo].into();
            let a0i: f64 = input[lo + 1].into();
            let a1r: f64 = input[hi].into();
            let a1i: f64 = input[hi + 1].into();

            let out_lo_r = g00r * a0r - g00i * a0i + g01r * a1r - g01i * a1i;
            let out_lo_i = g00r * a0i + g00i * a0r + g01r * a1i + g01i * a1r;
            let out_hi_r = g10r * a0r - g10i * a0i + g11r * a1r - g11i * a1i;
            let out_hi_i = g10r * a0i + g10i * a0r + g11r * a1i + g11i * a1r;

            // Both members of the pair are inside this block: `chunk` is a
            // whole number of `2 * stride` windows, and `within` is in the
            // lower half of its window.
            let block_base = flat - start;
            block[block_base] = T::from_f64(out_lo_r);
            block[block_base + 1] = T::from_f64(out_lo_i);
            block[block_base + stride * 2] = T::from_f64(out_hi_r);
            block[block_base + stride * 2 + 1] = T::from_f64(out_hi_i);
        }
    });
}

/// Narrowing conversion from the `f64` the gate arithmetic runs in.
pub trait FromF64: Sized {
    fn from_f64(value: f64) -> Self;
}
impl FromF64 for f32 {
    #[inline]
    fn from_f64(value: f64) -> Self {
        value as f32
    }
}
impl FromF64 for f64 {
    #[inline]
    fn from_f64(value: f64) -> Self {
        value
    }
}

/// Born-rule probabilities: `|amplitude|^2` per basis state.
///
/// `state` is `(..., 2^q, 2)`; the result is `(..., 2^q)`.
pub fn probabilities(state: &Tensor) -> Result<Tensor> {
    let (batch, dim, _) = state_layout(state, "probabilities")?;
    let mut out_dims = state.shape().dims().to_vec();
    out_dims.pop();

    macro_rules! run {
        ($ty:ty, $slice:ident, $from_vec:path) => {{
            let input = state.data().$slice().ok_or_else(dtype_mismatch)?;
            let mut out = vec![<$ty>::default(); batch * dim];
            par_out_chunks(&mut out, PAR_CHUNK, &|start, block| {
                for (offset, slot) in block.iter_mut().enumerate() {
                    let i = (start + offset) * 2;
                    *slot = input[i] * input[i] + input[i + 1] * input[i + 1];
                }
            });
            $from_vec(out, state.device())
        }};
    }

    let data = match state.dtype() {
        DataType::Float32 => run!(f32, as_f32_slice, TensorData::from_vec_f32),
        DataType::Float64 => run!(f64, as_f64_slice, TensorData::from_vec_f64),
        other => return Err(float_only("probabilities", other)),
    };

    let requires_grad = crate::autograd::is_grad_enabled() && state.requires_grad();
    let output = Tensor::new(
        Arc::new(data),
        Shape::new(out_dims),
        state.dtype(),
        state.device(),
        requires_grad,
    );
    if !requires_grad {
        return Ok(output);
    }
    with_grad_fn(
        output,
        Arc::new(ProbabilitiesBackward {
            state: state.detach(),
            input_id: state.id(),
        }),
    )
}

/// Marginal probabilities of the first `keep` qubits — a prefix trace.
///
/// Tracing out the remaining qubits of a *pure* state means summing
/// `|amplitude|^2` over every configuration of the traced qubits. With the
/// prefix held in the high bits of the index, the amplitudes contributing to
/// one prefix outcome are one contiguous run, so this is a strided sum with
/// perfect locality — the reason to trace a prefix rather than an arbitrary
/// subset, and the reason this is one kernel rather than a reshape and a
/// reduction.
///
/// `state` is `(..., 2^q, 2)`; the result is `(..., 2^keep)`.
pub fn prefix_trace(state: &Tensor, keep: usize) -> Result<Tensor> {
    let (batch, dim, qubits) = state_layout(state, "prefix_trace")?;
    if keep > qubits {
        return Err(MinitensorError::invalid_argument(format!(
            "prefix_trace: cannot keep {keep} qubits of a {qubits}-qubit state"
        )));
    }
    let kept = 1usize << keep;
    let traced = dim / kept;

    let per_state = probabilities(state)?;
    // Sum each contiguous run of `traced` probabilities. Recorded through the
    // existing reduction, so the derivative of the trace is the derivative of
    // `probabilities` broadcast back over the run -- which is correct and is
    // one fewer hand-written backward to keep in step.
    let mut grouped_dims = state.shape().dims()[..state.shape().dims().len() - 2].to_vec();
    grouped_dims.push(kept);
    grouped_dims.push(traced);
    let grouped = crate::ops::shape_ops::reshape(&per_state, Shape::new(grouped_dims))?;
    let _ = batch;
    crate::ops::reduction::sum(&grouped, Some(vec![-1]), false)
}

/// `<Z_t>`: the expectation of Pauli-Z on one qubit.
///
/// `+1` when the qubit is certainly `|0>`, `-1` when certainly `|1>`, and the
/// probability difference in between. `state` is `(..., 2^q, 2)`; the result
/// has the batch shape.
pub fn expect_z(state: &Tensor, target: usize) -> Result<Tensor> {
    let (_, dim, qubits) = state_layout(state, "expect_z")?;
    if target >= qubits {
        return Err(MinitensorError::invalid_argument(format!(
            "expect_z: qubit {target} is out of range for a {qubits}-qubit state"
        )));
    }

    // `sign[i] = +1` when bit `target` of `i` is clear, `-1` when set. Built
    // as a tensor and multiplied in, so the whole thing rides the existing
    // autograd path: `probabilities` is differentiable and the rest is a
    // weighted sum.
    let signs: Vec<f64> = (0..dim)
        .map(|index| {
            if index & (1 << target) == 0 {
                1.0
            } else {
                -1.0
            }
        })
        .collect();
    let signs = match state.dtype() {
        DataType::Float32 => {
            TensorData::from_vec_f32(signs.iter().map(|&v| v as f32).collect(), state.device())
        }
        DataType::Float64 => TensorData::from_vec_f64(signs, state.device()),
        other => return Err(float_only("expect_z", other)),
    };
    let signs = Tensor::new(
        Arc::new(signs),
        Shape::new(vec![dim]),
        state.dtype(),
        state.device(),
        false,
    );

    let probs = probabilities(state)?;
    let weighted = crate::ops::arithmetic::mul(&probs, &signs)?;
    crate::ops::reduction::sum(&weighted, Some(vec![-1]), false)
}

fn dtype_mismatch() -> MinitensorError {
    MinitensorError::internal_error("quantum: tensor data did not match its dtype")
}

fn float_only(op: &str, dtype: DataType) -> MinitensorError {
    MinitensorError::invalid_operation(format!(
        "{op} is defined for float32 and float64 tensors, not {dtype:?}"
    ))
}

/// The adjoint butterfly.
///
/// Gate application is `psi' = U psi`, linear, so `dL/dpsi = U^dagger dL/dpsi'`
/// — the same kernel with the conjugate transpose. Nothing from the forward
/// pass needs saving, which is why a long circuit costs no memory beyond its
/// states.
struct GateBackward {
    adjoint: Gate1,
    target: usize,
    batch: usize,
    dim: usize,
    input_id: TensorId,
}

impl GradientFunction for GateBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        macro_rules! run {
            ($ty:ty, $slice:ident, $from_vec:path) => {{
                let g = grad_output.data().$slice().ok_or_else(dtype_mismatch)?;
                let mut out = vec![<$ty>::default(); g.len()];
                butterfly::<$ty>(
                    g,
                    &mut out,
                    self.batch,
                    self.dim,
                    self.target,
                    &self.adjoint,
                );
                $from_vec(out, grad_output.device())
            }};
        }
        let data = match grad_output.dtype() {
            DataType::Float32 => run!(f32, as_f32_slice, TensorData::from_vec_f32),
            DataType::Float64 => run!(f64, as_f64_slice, TensorData::from_vec_f64),
            other => return Err(float_only("apply_gate_1q backward", other)),
        };
        let mut gradients = FxHashMap::default();
        gradients.insert(
            self.input_id,
            Tensor::new(
                Arc::new(data),
                Shape::new(grad_output.shape().dims().to_vec()),
                grad_output.dtype(),
                grad_output.device(),
                false,
            ),
        );
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn name(&self) -> &'static str {
        "QuantumGateBackward"
    }
}

/// `d|a|^2/da = 2a`, on both the real and the imaginary part.
struct ProbabilitiesBackward {
    state: Tensor,
    input_id: TensorId,
}

impl GradientFunction for ProbabilitiesBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        macro_rules! run {
            ($ty:ty, $slice:ident, $from_vec:path) => {{
                let g = grad_output.data().$slice().ok_or_else(dtype_mismatch)?;
                let s = self.state.data().$slice().ok_or_else(dtype_mismatch)?;
                let mut out = vec![<$ty>::default(); s.len()];
                par_out_chunks(&mut out, PAR_CHUNK, &|start, block| {
                    for (offset, slot) in block.iter_mut().enumerate() {
                        let i = start + offset;
                        // Both components of amplitude `i / 2` scale by the
                        // same incoming gradient.
                        *slot = <$ty as FromF64>::from_f64(2.0) * g[i / 2] * s[i];
                    }
                });
                $from_vec(out, grad_output.device())
            }};
        }
        let data = match grad_output.dtype() {
            DataType::Float32 => run!(f32, as_f32_slice, TensorData::from_vec_f32),
            DataType::Float64 => run!(f64, as_f64_slice, TensorData::from_vec_f64),
            other => return Err(float_only("probabilities backward", other)),
        };
        let mut gradients = FxHashMap::default();
        gradients.insert(
            self.input_id,
            Tensor::new(
                Arc::new(data),
                Shape::new(self.state.shape().dims().to_vec()),
                self.state.dtype(),
                self.state.device(),
                false,
            ),
        );
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn name(&self) -> &'static str {
        "QuantumProbabilitiesBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A state from `(re, im)` pairs.
    fn state(amplitudes: &[(f64, f64)]) -> Tensor {
        let flat: Vec<f64> = amplitudes.iter().flat_map(|&(r, i)| [r, i]).collect();
        Tensor::new(
            Arc::new(TensorData::from_vec_f64(flat, crate::Device::cpu())),
            Shape::new(vec![amplitudes.len(), 2]),
            DataType::Float64,
            crate::Device::cpu(),
            false,
        )
    }

    fn amplitudes(tensor: &Tensor) -> Vec<(f64, f64)> {
        tensor
            .data()
            .as_f64_slice()
            .unwrap()
            .chunks(2)
            .map(|p| (p[0], p[1]))
            .collect()
    }

    fn close(got: &[(f64, f64)], want: &[(f64, f64)]) {
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want) {
            assert!(
                (g.0 - w.0).abs() < 1e-12 && (g.1 - w.1).abs() < 1e-12,
                "{got:?} != {want:?}"
            );
        }
    }

    #[test]
    fn hadamard_makes_an_equal_superposition() {
        let zero = state(&[(1.0, 0.0), (0.0, 0.0)]);
        let out = apply_gate_1q(&zero, &HADAMARD, 0).unwrap();
        let h = std::f64::consts::FRAC_1_SQRT_2;
        close(&amplitudes(&out), &[(h, 0.0), (h, 0.0)]);
    }

    #[test]
    fn hadamard_is_its_own_inverse() {
        let start = state(&[(0.6, 0.0), (0.0, 0.8)]);
        let once = apply_gate_1q(&start, &HADAMARD, 0).unwrap();
        let twice = apply_gate_1q(&once, &HADAMARD, 0).unwrap();
        close(&amplitudes(&twice), &amplitudes(&start));
    }

    #[test]
    fn pauli_x_swaps_the_basis_states() {
        let s = state(&[(1.0, 2.0), (3.0, 4.0)]);
        let out = apply_gate_1q(&s, &PAULI_X, 0).unwrap();
        close(&amplitudes(&out), &[(3.0, 4.0), (1.0, 2.0)]);
    }

    /// The point of the strided butterfly: acting on a higher qubit pairs
    /// amplitudes that are far apart in the buffer.
    #[test]
    fn a_higher_qubit_pairs_distant_amplitudes() {
        // Three qubits: |000>..|111>, index bit 2 is the high qubit.
        let mut amps = vec![(0.0, 0.0); 8];
        amps[0] = (1.0, 0.0);
        let s = state(&amps);
        let out = apply_gate_1q(&s, &PAULI_X, 2).unwrap();
        // X on qubit 2 maps |000> (index 0) to |100> (index 4).
        let got = amplitudes(&out);
        assert!((got[4].0 - 1.0).abs() < 1e-12, "{got:?}");
        assert!(got[0].0.abs() < 1e-12, "{got:?}");
    }

    #[test]
    fn a_gate_preserves_the_norm() {
        let s = state(&[(0.3, 0.1), (0.5, -0.2), (0.1, 0.7), (-0.3, 0.15)]);
        let before: f64 = amplitudes(&s).iter().map(|a| a.0 * a.0 + a.1 * a.1).sum();
        let out = apply_gate_1q(&s, &HADAMARD, 1).unwrap();
        let after: f64 = amplitudes(&out).iter().map(|a| a.0 * a.0 + a.1 * a.1).sum();
        assert!((before - after).abs() < 1e-12, "{before} != {after}");
    }

    #[test]
    fn complex_phases_multiply_correctly() {
        // S gate: diag(1, i). Applied to |1> it should introduce exactly i.
        let s_gate: Gate1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let one = state(&[(0.0, 0.0), (1.0, 0.0)]);
        let out = apply_gate_1q(&one, &s_gate, 0).unwrap();
        close(&amplitudes(&out), &[(0.0, 0.0), (0.0, 1.0)]);
    }

    #[test]
    fn probabilities_follow_the_born_rule() {
        let s = state(&[(0.6, 0.0), (0.0, 0.8)]);
        let p = probabilities(&s).unwrap();
        let values = p.data().as_f64_slice().unwrap();
        assert!((values[0] - 0.36).abs() < 1e-12);
        assert!((values[1] - 0.64).abs() < 1e-12);
    }

    #[test]
    fn expect_z_is_plus_one_on_the_zero_state() {
        let zero = state(&[(1.0, 0.0), (0.0, 0.0)]);
        let z = expect_z(&zero, 0).unwrap();
        assert!((z.data().as_f64_slice().unwrap()[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn expect_z_is_zero_on_a_superposition() {
        let zero = state(&[(1.0, 0.0), (0.0, 0.0)]);
        let plus = apply_gate_1q(&zero, &HADAMARD, 0).unwrap();
        let z = expect_z(&plus, 0).unwrap();
        assert!(z.data().as_f64_slice().unwrap()[0].abs() < 1e-12);
    }

    #[test]
    fn prefix_trace_marginalises_the_low_qubits() {
        // Two qubits, all weight on |10> (index 2): tracing to the high qubit
        // leaves probability 0 on prefix 0 and 1 on prefix 1.
        let s = state(&[(0.0, 0.0), (0.0, 0.0), (1.0, 0.0), (0.0, 0.0)]);
        let marginal = prefix_trace(&s, 1).unwrap();
        let values = marginal.data().as_f64_slice().unwrap();
        assert_eq!(values.len(), 2);
        assert!(
            values[0].abs() < 1e-12 && (values[1] - 1.0).abs() < 1e-12,
            "{values:?}"
        );
    }

    #[test]
    fn prefix_trace_of_everything_is_the_probabilities() {
        let s = state(&[(0.5, 0.5), (0.5, -0.5), (0.0, 0.0), (0.0, 0.0)]);
        let full = prefix_trace(&s, 2).unwrap();
        let probs = probabilities(&s).unwrap();
        let a = full.data().as_f64_slice().unwrap();
        let b = probs.data().as_f64_slice().unwrap();
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < 1e-12);
        }
    }

    #[test]
    fn a_batch_of_states_evolves_independently() {
        let flat: Vec<f64> = vec![
            1.0, 0.0, 0.0, 0.0, // |0>
            0.0, 0.0, 1.0, 0.0, // |1>
        ];
        let batched = Tensor::new(
            Arc::new(TensorData::from_vec_f64(flat, crate::Device::cpu())),
            Shape::new(vec![2, 2, 2]),
            DataType::Float64,
            crate::Device::cpu(),
            false,
        );
        let out = apply_gate_1q(&batched, &PAULI_X, 0).unwrap();
        let got = out.data().as_f64_slice().unwrap();
        // Each state's amplitudes swapped, and no state touched the other.
        assert!(
            (got[0]).abs() < 1e-12 && (got[2] - 1.0).abs() < 1e-12,
            "{got:?}"
        );
        assert!(
            (got[4] - 1.0).abs() < 1e-12 && (got[6]).abs() < 1e-12,
            "{got:?}"
        );
    }

    #[test]
    fn bad_shapes_and_qubits_are_rejected() {
        let s = state(&[(1.0, 0.0), (0.0, 0.0)]);
        assert!(apply_gate_1q(&s, &HADAMARD, 1).is_err());
        assert!(expect_z(&s, 5).is_err());
        assert!(prefix_trace(&s, 3).is_err());

        let three = Tensor::new(
            Arc::new(TensorData::from_vec_f64(vec![0.0; 6], crate::Device::cpu())),
            Shape::new(vec![3, 2]),
            DataType::Float64,
            crate::Device::cpu(),
            false,
        );
        assert!(
            apply_gate_1q(&three, &HADAMARD, 0).is_err(),
            "3 is not a power of two"
        );
    }
}

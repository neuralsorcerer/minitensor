// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::{Layer, init::InitMethod};
use std::collections::HashMap;

use crate::{
    device::Device,
    error::{MinitensorError, Result},
    ops::{
        arithmetic::{add, mul, sub},
        linalg::matmul,
        shape_ops::{concatenate, narrow},
    },
    tensor::{DataType, Shape, Tensor},
};

/// Which recurrent cell a [`Recurrent`] stack runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellKind {
    /// Long Short-Term Memory (Hochreiter & Schmidhuber, 1997): carries a
    /// separate cell state alongside the hidden state.
    Lstm,
    /// Gated Recurrent Unit (Cho et al., 2014): one state, one fewer gate.
    Gru,
}

impl CellKind {
    /// How many gate blocks the weight matrices are split into.
    fn gates(self) -> usize {
        match self {
            CellKind::Lstm => 4,
            CellKind::Gru => 3,
        }
    }

    fn name(self) -> &'static str {
        match self {
            CellKind::Lstm => "LSTM",
            CellKind::Gru => "GRU",
        }
    }
}

/// The weights of one layer in the stack.
#[derive(Clone)]
struct LayerWeights {
    /// `[gates * hidden_size, layer_input_size]`
    w_ih: Tensor,
    /// `[gates * hidden_size, hidden_size]`
    w_hh: Tensor,
    b_ih: Option<Tensor>,
    b_hh: Option<Tensor>,
}

/// A stack of recurrent layers, shared by [`LSTM`] and [`GRU`].
///
/// The cells are built from the ordinary autograd-aware tensor operations
/// (`matmul`, `sigmoid`, `tanh`, elementwise arithmetic) rather than a fused
/// kernel with a hand-written backward. A recurrent backward pass has to
/// accumulate through every timestep, and writing that by hand is where these
/// layers usually go wrong; composing means the existing graph derives it, and
/// it is correct by construction. The cost is the intermediate tensors each step
/// allocates — a fused kernel would be faster, and is the obvious later
/// optimisation.
///
/// Within that composed form, the work that does *not* depend on the recurrence
/// is hoisted out of the timestep loop: both weight transposes are taken once
/// per direction, and each layer projects its whole input sequence in a single
/// matmul. See [`Recurrent::forward_with_state`]. What remains per step is the
/// hidden matmul and the gate arithmetic, which are sequential by definition.
#[derive(Clone)]
pub struct Recurrent {
    kind: CellKind,
    layers: Vec<LayerWeights>,
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    batch_first: bool,
    bias: bool,
    bidirectional: bool,
}

impl Recurrent {
    /// Build a stack of `num_layers` cells.
    ///
    /// Every parameter is drawn from `U(-1/sqrt(hidden_size), 1/sqrt(hidden_size))`,
    /// including the biases — the usual convention for recurrent layers, and
    /// unlike the zero biases the feedforward layers here start from.
    pub fn new(
        kind: CellKind,
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        bias: bool,
        batch_first: bool,
        bidirectional: bool,
        device: Device,
        dtype: DataType,
    ) -> Result<Self> {
        if input_size == 0 || hidden_size == 0 {
            return Err(MinitensorError::invalid_argument(format!(
                "{} requires input_size and hidden_size to be non-zero",
                kind.name()
            )));
        }
        if num_layers == 0 {
            return Err(MinitensorError::invalid_argument(format!(
                "{} requires at least one layer",
                kind.name()
            )));
        }
        if !dtype.is_float() {
            return Err(MinitensorError::invalid_argument(format!(
                "{} parameters must have a floating point dtype",
                kind.name()
            )));
        }

        let bound = 1.0 / (hidden_size as f64).sqrt();
        let uniform = InitMethod::Uniform {
            a: -bound,
            b: bound,
        };
        let gates = kind.gates();

        let directions = if bidirectional { 2 } else { 1 };
        let mut layers = Vec::with_capacity(num_layers * directions);
        for layer in 0..num_layers {
            // Only the first layer sees the raw input; the rest consume the
            // layer below, which is twice as wide when bidirectional because
            // the two directions are concatenated.
            let layer_input = if layer == 0 {
                input_size
            } else {
                hidden_size * directions
            };
            // Directions are adjacent within a layer, so the flat order is
            // layer0-forward, layer0-reverse, layer1-forward, ... matching the
            // `*_l{k}` and `*_l{k}_reverse` names.
            for _ in 0..directions {
                let make =
                    |shape: Vec<usize>| uniform.init_tensor(Shape::new(shape), dtype, device, true);
                layers.push(LayerWeights {
                    w_ih: make(vec![gates * hidden_size, layer_input])?,
                    w_hh: make(vec![gates * hidden_size, hidden_size])?,
                    b_ih: if bias {
                        Some(make(vec![gates * hidden_size])?)
                    } else {
                        None
                    },
                    b_hh: if bias {
                        Some(make(vec![gates * hidden_size])?)
                    } else {
                        None
                    },
                });
            }
        }

        Ok(Self {
            kind,
            layers,
            input_size,
            hidden_size,
            num_layers,
            batch_first,
            bias,
            bidirectional,
        })
    }

    /// Which cell this stack runs.
    pub fn kind(&self) -> CellKind {
        self.kind
    }

    /// Width of each input vector.
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Width of the hidden state.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Number of stacked layers.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Whether inputs are `(batch, seq, feature)` rather than `(seq, batch, feature)`.
    pub fn batch_first(&self) -> bool {
        self.batch_first
    }

    /// Whether the layers carry additive biases.
    pub fn has_bias(&self) -> bool {
        self.bias
    }

    /// Whether each layer also runs a second pass over the reversed sequence.
    pub fn bidirectional(&self) -> bool {
        self.bidirectional
    }

    /// `l{k}` or `l{k}_reverse` for the flat layer index `index`.
    fn suffix_for(index: usize, directions: usize) -> String {
        let layer = index / directions;
        if directions == 2 && index % 2 == 1 {
            format!("l{layer}_reverse")
        } else {
            format!("l{layer}")
        }
    }

    /// [`Self::suffix_for`] using this stack's direction count.
    fn parameter_suffix(&self, index: usize) -> String {
        Self::suffix_for(index, if self.bidirectional { 2 } else { 1 })
    }

    /// 2 when bidirectional, 1 otherwise.
    pub fn num_directions(&self) -> usize {
        if self.bidirectional { 2 } else { 1 }
    }

    /// Width of the output feature axis: both directions are concatenated.
    pub fn output_size(&self) -> usize {
        self.hidden_size * self.num_directions()
    }

    /// `x @ Wt + b`, the affine map each gate block shares.
    ///
    /// Takes the weight already transposed. The transpose materialises a copy,
    /// and these weights are reused at every timestep, so it is hoisted out of
    /// the recurrence by the caller rather than repeated here.
    fn affine(x: &Tensor, weight_t: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
        let projected = matmul(x, weight_t)?;
        match bias {
            Some(b) => add(&projected, b),
            None => Ok(projected),
        }
    }

    /// Slice gate block `index` out of a `[batch, gates * hidden]` tensor.
    fn gate(&self, gates: &Tensor, index: usize) -> Result<Tensor> {
        narrow(gates, 1, index * self.hidden_size, self.hidden_size)
    }

    /// One LSTM step: returns the new `(h, c)`.
    ///
    /// `from_input` is this timestep's slice of the input projection, which the
    /// caller computed for the whole sequence at once.
    fn lstm_step(
        &self,
        w_hh_t: &Tensor,
        b_hh: Option<&Tensor>,
        from_input: &Tensor,
        h: &Tensor,
        c: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        // The four gates share one matmul; they are only separated afterwards.
        let from_hidden = Self::affine(h, w_hh_t, b_hh)?;
        let gates = add(from_input, &from_hidden)?;

        // Block order is i, f, g, o — the layout the stored weights use, so a
        // state dict transfers without permuting.
        let input_gate = self.gate(&gates, 0)?.sigmoid()?;
        let forget_gate = self.gate(&gates, 1)?.sigmoid()?;
        let candidate = self.gate(&gates, 2)?.tanh()?;
        let output_gate = self.gate(&gates, 3)?.sigmoid()?;

        let new_c = add(&mul(&forget_gate, c)?, &mul(&input_gate, &candidate)?)?;
        let new_h = mul(&output_gate, &new_c.tanh()?)?;
        Ok((new_h, new_c))
    }

    /// One GRU step: returns the new `h`.
    ///
    /// `from_input` is this timestep's slice of the input projection, which the
    /// caller computed for the whole sequence at once.
    fn gru_step(
        &self,
        w_hh_t: &Tensor,
        b_hh: Option<&Tensor>,
        from_input: &Tensor,
        h: &Tensor,
    ) -> Result<Tensor> {
        let from_hidden = Self::affine(h, w_hh_t, b_hh)?;

        // Block order is r, z, n.
        let reset = add(&self.gate(from_input, 0)?, &self.gate(&from_hidden, 0)?)?.sigmoid()?;
        let update = add(&self.gate(from_input, 1)?, &self.gate(&from_hidden, 1)?)?.sigmoid()?;

        // The reset gate multiplies the *projected* hidden contribution, not the
        // hidden state before its matmul. The two are not equivalent — with the
        // bias inside the product, as it is here, `r` also scales `b_hn` — and
        // this is the detail GRU implementations most often get wrong. This
        // matches cuDNN.
        let gated_hidden = mul(&reset, &self.gate(&from_hidden, 2)?)?;
        let candidate = add(&self.gate(from_input, 2)?, &gated_hidden)?.tanh()?;

        // h' = (1 - z) * n + z * h
        //
        // Written this way on purpose. The algebraically equal `n + z * (h - n)`
        // needs no `ones` tensor and one fewer elementwise op, which makes it a
        // tempting optimisation — but it is not equal in floating point at the
        // saturated update gate. `sigmoid` reaches exactly 1.0 in f32 by a logit
        // of about 17, and there `(1 - z) * n + z * h` yields `h` bit-for-bit
        // while `n + z * (h - n)` misses it in roughly a third of cases, by up
        // to 5e-7. A saturated `z` is exactly how a GRU carries state across a
        // long sequence, so that error would be injected at every step of the
        // one path that is supposed to be lossless, and accumulate over the
        // sequence. The allocation buys exact pass-through; keep it.
        let ones = Tensor::ones(
            update.shape().clone(),
            update.dtype(),
            update.device(),
            false,
        );
        let keep = sub(&ones, &update)?;
        add(&mul(&keep, &candidate)?, &mul(&update, h)?)
    }

    /// Run the stack, returning the output sequence and the final states.
    ///
    /// `input` is `(seq, batch, input_size)`, or `(batch, seq, input_size)` when
    /// `batch_first`. `state` supplies the initial hidden (and, for LSTM, cell)
    /// states shaped `(num_layers, batch, hidden_size)`; zeros are used when it
    /// is omitted. The returned cell state is `None` for GRU.
    pub fn forward_with_state(
        &self,
        input: &Tensor,
        state: Option<(&Tensor, Option<&Tensor>)>,
    ) -> Result<(Tensor, Tensor, Option<Tensor>)> {
        if input.ndim() != 3 {
            return Err(MinitensorError::invalid_argument(format!(
                "{} expects a 3-D input, got {}-D",
                self.kind.name(),
                input.ndim()
            )));
        }

        // Work internally in (seq, batch, feature) whatever the caller used.
        let source = if self.batch_first {
            input.transpose(0, 1)?
        } else {
            input.clone()
        };
        let dims = source.shape().dims().to_vec();
        let (seq_len, batch) = (dims[0], dims[1]);

        if dims[2] != self.input_size {
            return Err(MinitensorError::invalid_argument(format!(
                "{} expects input feature size {}, got {}",
                self.kind.name(),
                self.input_size,
                dims[2]
            )));
        }
        if seq_len == 0 {
            return Err(MinitensorError::invalid_argument(format!(
                "{} requires a non-empty sequence",
                self.kind.name()
            )));
        }

        let state_shape = Shape::new(vec![
            self.num_layers * self.num_directions(),
            batch,
            self.hidden_size,
        ]);
        let check_state = |t: &Tensor, what: &str| -> Result<()> {
            if t.shape() != &state_shape {
                return Err(MinitensorError::shape_mismatch(
                    state_shape.dims().to_vec(),
                    t.shape().dims().to_vec(),
                ));
            }
            if t.dtype() != input.dtype() {
                return Err(MinitensorError::type_mismatch(
                    format!("{:?}", t.dtype()),
                    format!("{:?}", input.dtype()),
                ));
            }
            let _ = what;
            Ok(())
        };

        let zero_state =
            || Tensor::zeros(state_shape.clone(), input.dtype(), input.device(), false);
        let (h0, c0) = match state {
            Some((h, c)) => {
                check_state(h, "hidden")?;
                match (self.kind, c) {
                    (CellKind::Lstm, Some(c)) => {
                        check_state(c, "cell")?;
                        (h.clone(), Some(c.clone()))
                    }
                    (CellKind::Lstm, None) => (h.clone(), Some(zero_state())),
                    (CellKind::Gru, None) => (h.clone(), None),
                    (CellKind::Gru, Some(_)) => {
                        return Err(MinitensorError::invalid_argument(
                            "GRU has no cell state; pass only a hidden state",
                        ));
                    }
                }
            }
            None => (
                zero_state(),
                match self.kind {
                    CellKind::Lstm => Some(zero_state()),
                    CellKind::Gru => None,
                },
            ),
        };

        let per_layer = |state: &Tensor, layer: usize| -> Result<Tensor> {
            narrow(state, 0, layer, 1)?.reshape(Shape::new(vec![batch, self.hidden_size]))
        };

        let directions = self.num_directions();
        let mut final_h = Vec::with_capacity(self.num_layers * directions);
        let mut final_c = Vec::with_capacity(self.num_layers * directions);

        // Carried as one `(seq, batch, feature)` tensor rather than a timestep
        // list, so each layer can project its whole input in a single matmul.
        let mut layer_input = source;
        let mut layer_feature = self.input_size;

        for layer in 0..self.num_layers {
            // The input path does not depend on the recurrence, so it does not
            // have to be walked one timestep at a time: `seq` products of
            // `(batch, feature) x (feature, gates * hidden)` are one product of
            // `(seq * batch, feature)` instead. Only the hidden path is
            // genuinely sequential. The per-step slices are then rows of the
            // result, which is why the flattened layout is `(seq * batch, _)`
            // and not `(batch * seq, _)`.
            //
            // This holds the whole sequence's projection live at once, where
            // the per-step form could drop each one after its step. That costs
            // nothing while building a graph -- every step's projection is
            // retained for the backward pass either way -- and under `no_grad`
            // it peaks at `gates` times the output sequence this call already
            // allocates, so the bound stays proportional to what the caller
            // asked for.
            let flat = layer_input
                .contiguous()?
                .reshape(Shape::new(vec![seq_len * batch, layer_feature]))?;

            let mut per_direction: Vec<Vec<Tensor>> = Vec::with_capacity(directions);

            for direction in 0..directions {
                let index = layer * directions + direction;
                let weights = &self.layers[index];

                // Both weight transposes materialise a copy, so they are done
                // once per direction instead of once per timestep.
                let w_ih_t = weights.w_ih.transpose(0, 1)?;
                let w_hh_t = weights.w_hh.transpose(0, 1)?;
                let projected = Self::affine(&flat, &w_ih_t, weights.b_ih.as_ref())?;

                let mut h = per_layer(&h0, index)?;
                let mut c = match &c0 {
                    Some(c0) => Some(per_layer(c0, index)?),
                    None => None,
                };

                // Direction 1 consumes the sequence from the end.
                let order: Vec<usize> = if direction == 0 {
                    (0..seq_len).collect()
                } else {
                    (0..seq_len).rev().collect()
                };

                let mut outputs = Vec::with_capacity(seq_len);
                for &t in &order {
                    let x = narrow(&projected, 0, t * batch, batch)?;
                    match self.kind {
                        CellKind::Lstm => {
                            let cell = c.as_ref().expect("LSTM always carries a cell state");
                            let (new_h, new_c) =
                                self.lstm_step(&w_hh_t, weights.b_hh.as_ref(), &x, &h, cell)?;
                            h = new_h;
                            c = Some(new_c);
                        }
                        CellKind::Gru => {
                            h = self.gru_step(&w_hh_t, weights.b_hh.as_ref(), &x, &h)?
                        }
                    }
                    outputs.push(h.clone());
                }

                // The reverse pass produced its states last-to-first; put them
                // back on the input's timeline before they are concatenated,
                // or every output would be paired with the wrong timestep.
                if direction == 1 {
                    outputs.reverse();
                }

                final_h.push(h.reshape(Shape::new(vec![1, batch, self.hidden_size]))?);
                if let Some(c) = c {
                    final_c.push(c.reshape(Shape::new(vec![1, batch, self.hidden_size]))?);
                }
                per_direction.push(outputs);
            }

            // The next layer consumes this one's states; when bidirectional the
            // two directions are joined along the feature axis at each step.
            let steps: Vec<Tensor> = if directions == 1 {
                per_direction.pop().expect("one direction")
            } else {
                (0..seq_len)
                    .map(|t| {
                        let parts: Vec<&Tensor> = per_direction.iter().map(|d| &d[t]).collect();
                        concatenate(&parts, 1)
                    })
                    .collect::<Result<_>>()?
            };

            // Restack into `(seq, batch, feature)` for the next layer to
            // project in one go; after the last layer this is the output.
            layer_feature = self.output_size();
            let stacked: Vec<Tensor> = steps
                .iter()
                .map(|h| h.reshape(Shape::new(vec![1, batch, layer_feature])))
                .collect::<Result<_>>()?;
            let refs: Vec<&Tensor> = stacked.iter().collect();
            layer_input = concatenate(&refs, 0)?;
        }

        let output = if self.batch_first {
            layer_input.transpose(0, 1)?
        } else {
            layer_input
        };

        let h_refs: Vec<&Tensor> = final_h.iter().collect();
        let h_n = concatenate(&h_refs, 0)?;
        let c_n = if final_c.is_empty() {
            None
        } else {
            let c_refs: Vec<&Tensor> = final_c.iter().collect();
            Some(concatenate(&c_refs, 0)?)
        };

        Ok((output, h_n, c_n))
    }
}

impl Layer for Recurrent {
    /// Names are `weight_ih_l{k}`, `weight_hh_l{k}`, `bias_ih_l{k}`,
    /// `bias_hh_l{k}`, with `_reverse` appended for the backward direction.
    ///
    /// The flat layer order is layer0-forward, layer0-reverse, layer1-forward,
    /// ... so the index and direction fall straight out of it. Naming these
    /// matters more here than elsewhere: a stacked bidirectional LSTM has up to
    /// `4 * num_layers * directions` parameters, several of which share a
    /// shape, so a positional checkpoint is impossible to check by eye.
    fn named_parameters(&self) -> HashMap<String, &Tensor> {
        let mut named = HashMap::with_capacity(self.layers.len() * 4);
        for (index, layer) in self.layers.iter().enumerate() {
            let suffix = self.parameter_suffix(index);
            named.insert(format!("weight_ih_{suffix}"), &layer.w_ih);
            named.insert(format!("weight_hh_{suffix}"), &layer.w_hh);
            if let Some(b) = &layer.b_ih {
                named.insert(format!("bias_ih_{suffix}"), b);
            }
            if let Some(b) = &layer.b_hh {
                named.insert(format!("bias_hh_{suffix}"), b);
            }
        }
        named
    }

    /// Mutable counterpart of [`Self::named_parameters`].
    fn named_parameters_mut(&mut self) -> HashMap<String, &mut Tensor> {
        let directions = if self.bidirectional { 2 } else { 1 };
        let mut named = HashMap::with_capacity(self.layers.len() * 4);
        for (index, layer) in self.layers.iter_mut().enumerate() {
            let suffix = Self::suffix_for(index, directions);
            named.insert(format!("weight_ih_{suffix}"), &mut layer.w_ih);
            named.insert(format!("weight_hh_{suffix}"), &mut layer.w_hh);
            if let Some(b) = &mut layer.b_ih {
                named.insert(format!("bias_ih_{suffix}"), b);
            }
            if let Some(b) = &mut layer.b_hh {
                named.insert(format!("bias_hh_{suffix}"), b);
            }
        }
        named
    }

    /// Run the stack from zero state and return only the output sequence. Use
    /// [`Recurrent::forward_with_state`] when the final states are needed.
    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        let (output, _, _) = self.forward_with_state(input, None)?;
        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::with_capacity(self.num_layers * 4);
        for layer in &self.layers {
            params.push(&layer.w_ih);
            params.push(&layer.w_hh);
            if let Some(b) = &layer.b_ih {
                params.push(b);
            }
            if let Some(b) = &layer.b_hh {
                params.push(b);
            }
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::with_capacity(self.num_layers * 4);
        for layer in &mut self.layers {
            params.push(&mut layer.w_ih);
            params.push(&mut layer.w_hh);
            if let Some(b) = &mut layer.b_ih {
                params.push(b);
            }
            if let Some(b) = &mut layer.b_hh {
                params.push(b);
            }
        }
        params
    }
}

/// Long Short-Term Memory recurrent layer.
///
/// Inputs are `(seq, batch, input_size)`, or `(batch, seq, input_size)` when
/// `batch_first`. See [`Recurrent`] for the shared machinery.
pub type LSTM = Recurrent;

/// Gated Recurrent Unit layer. See [`Recurrent`].
pub type GRU = Recurrent;

impl Recurrent {
    /// Construct an LSTM stack.
    #[allow(clippy::too_many_arguments)]
    pub fn lstm(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        bias: bool,
        batch_first: bool,
        bidirectional: bool,
        device: Device,
        dtype: DataType,
    ) -> Result<Self> {
        Self::new(
            CellKind::Lstm,
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            bidirectional,
            device,
            dtype,
        )
    }

    /// Construct a GRU stack.
    #[allow(clippy::too_many_arguments)]
    pub fn gru(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        bias: bool,
        batch_first: bool,
        bidirectional: bool,
        device: Device,
        dtype: DataType,
    ) -> Result<Self> {
        Self::new(
            CellKind::Gru,
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            bidirectional,
            device,
            dtype,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorData;
    use std::sync::Arc;

    /// A tensor of `shape` filled with `value`; the length follows from the
    /// shape rather than being spelled out at each call site.
    fn filled(value: f64, shape: Vec<usize>) -> Tensor {
        let numel: usize = shape.iter().product();
        Tensor::new(
            Arc::new(TensorData::from_vec_f64(vec![value; numel], Device::cpu())),
            Shape::new(shape),
            DataType::Float64,
            Device::cpu(),
            false,
        )
    }

    fn build(kind: CellKind, input: usize, hidden: usize, layers: usize) -> Recurrent {
        build_with(kind, input, hidden, layers, false, false)
    }

    fn build_with(
        kind: CellKind,
        input: usize,
        hidden: usize,
        layers: usize,
        batch_first: bool,
        bidirectional: bool,
    ) -> Recurrent {
        Recurrent::new(
            kind,
            input,
            hidden,
            layers,
            true,
            batch_first,
            bidirectional,
            Device::cpu(),
            DataType::Float64,
        )
        .unwrap()
    }

    #[test]
    fn test_parameter_shapes_follow_the_gate_count() {
        // Every gate block shares one matrix, so the first axis is gates*hidden.
        // Only the first layer sees the raw input; the rest consume hidden.
        for (kind, gates) in [(CellKind::Lstm, 4), (CellKind::Gru, 3)] {
            let layer = build(kind, 3, 5, 2);
            let params = layer.parameters();
            assert_eq!(params.len(), 8, "two layers of four tensors");
            assert_eq!(params[0].shape().dims(), &[gates * 5, 3]);
            assert_eq!(params[1].shape().dims(), &[gates * 5, 5]);
            assert_eq!(params[2].shape().dims(), &[gates * 5]);
            assert_eq!(params[3].shape().dims(), &[gates * 5]);
            // Second layer's input is the first layer's hidden state.
            assert_eq!(params[4].shape().dims(), &[gates * 5, 5]);
        }
    }

    #[test]
    fn test_output_and_state_shapes() {
        for kind in [CellKind::Lstm, CellKind::Gru] {
            let layer = build(kind, 3, 5, 2);
            let x = filled(0.0, vec![4, 2, 3]);
            let (output, h_n, c_n) = layer.forward_with_state(&x, None).unwrap();
            assert_eq!(output.shape().dims(), &[4, 2, 5]);
            assert_eq!(h_n.shape().dims(), &[2, 2, 5]);
            match kind {
                CellKind::Lstm => assert_eq!(c_n.unwrap().shape().dims(), &[2, 2, 5]),
                CellKind::Gru => assert!(c_n.is_none(), "GRU has no cell state"),
            }
        }
    }

    #[test]
    fn test_batch_first_swaps_the_leading_axes() {
        for kind in [CellKind::Lstm, CellKind::Gru] {
            let layer = build_with(kind, 3, 5, 1, true, false);
            // (batch, seq, feature) in, (batch, seq, hidden) out.
            let x = filled(0.0, vec![2, 4, 3]);
            let (output, h_n, _) = layer.forward_with_state(&x, None).unwrap();
            assert_eq!(output.shape().dims(), &[2, 4, 5]);
            // The state is always (layers, batch, hidden), whatever batch_first is.
            assert_eq!(h_n.shape().dims(), &[1, 2, 5]);
        }
    }

    #[test]
    fn test_zero_input_and_state_without_bias_stays_at_zero() {
        // With no bias every gate pre-activation is zero. For LSTM the output
        // gate is 1/2 but tanh(c) is 0, so h stays 0; for GRU the candidate is
        // tanh(0) = 0 and h = (1 - 1/2)*0 + (1/2)*0 = 0. A cell that mixed up a
        // gate would break this.
        for kind in [CellKind::Lstm, CellKind::Gru] {
            let layer = Recurrent::new(
                kind,
                3,
                4,
                1,
                false,
                false,
                false,
                Device::cpu(),
                DataType::Float64,
            )
            .unwrap();
            assert_eq!(layer.parameters().len(), 2, "no bias tensors");
            let x = filled(0.0, vec![2, 1, 3]);
            let (output, _, _) = layer.forward_with_state(&x, None).unwrap();
            for v in output.data().as_f64_slice().unwrap() {
                assert_eq!(*v, 0.0);
            }
        }
    }

    #[test]
    fn test_gradients_reach_every_parameter() {
        // The interesting part is that gradient accumulates back through every
        // timestep; a layer whose loop dropped the chain would leave some of
        // these unset.
        for kind in [CellKind::Lstm, CellKind::Gru] {
            let mut layer = build(kind, 2, 3, 2);
            let x = filled(0.5, vec![3, 2, 2]);
            let output = layer.forward(&x).unwrap();
            let seed = filled(1.0, vec![3, 2, 3]);
            let grads = crate::autograd::backward_collect(&output, Some(seed)).unwrap();
            for (index, param) in layer.parameters().iter().enumerate() {
                assert!(
                    grads.contains_key(&param.id()),
                    "{:?} parameter {index} received no gradient",
                    kind
                );
            }
        }
    }

    #[test]
    fn test_rejects_malformed_input_and_configuration() {
        let layer = build(CellKind::Lstm, 3, 4, 1);
        // 2-D input
        assert!(
            layer
                .forward_with_state(&filled(0.0, vec![2, 3]), None)
                .is_err()
        );
        // Wrong feature width
        assert!(
            layer
                .forward_with_state(&filled(0.0, vec![2, 2, 9]), None)
                .is_err()
        );
        // Empty sequence
        assert!(
            layer
                .forward_with_state(&filled(0.0, vec![0, 2, 3]), None)
                .is_err()
        );
        // Hidden state of the wrong shape
        let x = filled(0.0, vec![2, 2, 3]);
        let bad_state = filled(0.0, vec![1, 2, 9]);
        assert!(
            layer
                .forward_with_state(&x, Some((&bad_state, None)))
                .is_err()
        );

        // Degenerate configurations
        let make = |input, hidden, layers| {
            Recurrent::new(
                CellKind::Gru,
                input,
                hidden,
                layers,
                true,
                false,
                false,
                Device::cpu(),
                DataType::Float64,
            )
        };
        assert!(make(0, 4, 1).is_err());
        assert!(make(3, 0, 1).is_err());
        assert!(make(3, 4, 0).is_err());
        // Integer parameters make no sense for a differentiable layer.
        assert!(
            Recurrent::new(
                CellKind::Gru,
                3,
                4,
                1,
                true,
                false,
                false,
                Device::cpu(),
                DataType::Int64
            )
            .is_err()
        );
    }

    #[test]
    fn test_gru_rejects_a_cell_state() {
        let layer = build(CellKind::Gru, 3, 4, 1);
        let x = filled(0.0, vec![2, 2, 3]);
        let state = filled(0.0, vec![1, 2, 4]);
        assert!(
            layer
                .forward_with_state(&x, Some((&state, Some(&state))))
                .is_err()
        );
    }

    #[test]
    fn test_bidirectional_doubles_the_weights_and_the_output_width() {
        for kind in [CellKind::Lstm, CellKind::Gru] {
            let layer = build_with(kind, 3, 5, 2, false, true);
            assert!(layer.bidirectional());
            assert_eq!(layer.num_directions(), 2);
            assert_eq!(layer.output_size(), 10);

            let params = layer.parameters();
            assert_eq!(params.len(), 16, "four tensors per direction per layer");
            // Directions are adjacent, so layer k begins at index 8k. The first
            // layer sees the raw input; the second sees both directions of the
            // first, hence twice the hidden width.
            assert_eq!(params[0].shape().dims()[1], 3);
            assert_eq!(params[4].shape().dims()[1], 3);
            assert_eq!(params[8].shape().dims()[1], 10);
            assert_eq!(params[12].shape().dims()[1], 10);

            let x = filled(0.0, vec![4, 2, 3]);
            let (output, h_n, _) = layer.forward_with_state(&x, None).unwrap();
            assert_eq!(output.shape().dims(), &[4, 2, 10]);
            // One state row per direction per layer.
            assert_eq!(h_n.shape().dims(), &[4, 2, 5]);
        }
    }

    #[test]
    fn test_bidirectional_gradients_reach_both_directions() {
        for kind in [CellKind::Lstm, CellKind::Gru] {
            let mut layer = build_with(kind, 2, 3, 2, false, true);
            let x = filled(0.5, vec![3, 2, 2]);
            let output = layer.forward(&x).unwrap();
            let seed = filled(1.0, vec![3, 2, 6]);
            let grads = crate::autograd::backward_collect(&output, Some(seed)).unwrap();
            for (index, param) in layer.parameters().iter().enumerate() {
                assert!(
                    grads.contains_key(&param.id()),
                    "{:?} parameter {index} received no gradient",
                    kind
                );
            }
        }
    }

    #[test]
    fn test_bidirectional_state_must_cover_every_direction() {
        let layer = build_with(CellKind::Gru, 3, 4, 1, false, true);
        let x = filled(0.0, vec![5, 2, 3]);
        // One row is a unidirectional state; two directions need two.
        assert!(
            layer
                .forward_with_state(&x, Some((&filled(0.0, vec![1, 2, 4]), None)))
                .is_err()
        );
        assert!(
            layer
                .forward_with_state(&x, Some((&filled(0.0, vec![2, 2, 4]), None)))
                .is_ok()
        );
    }

    #[test]
    fn test_gru_carries_state_untouched_when_the_update_gate_saturates() {
        // A saturated update gate is how a GRU holds state over a long
        // sequence, so the pass-through must be exact rather than merely close:
        // any error here is injected once per timestep and accumulates.
        let mut layer = build(CellKind::Gru, 2, 4, 1);
        {
            // Drive the z block of b_ih hard positive so sigmoid saturates.
            let mut params = layer.parameters_mut();
            let b_ih = params.get_mut(2).expect("b_ih");
            let slice = b_ih.data_mut().as_f64_slice_mut().unwrap();
            for value in slice.iter_mut().skip(4).take(4) {
                *value = 60.0;
            }
        }

        let batch = 3;
        let h0 = Tensor::ones(
            Shape::new(vec![1, batch, 4]),
            DataType::Float64,
            Device::cpu(),
            false,
        );
        let x = filled(0.25, vec![7, batch, 2]);
        let (output, _, _) = layer.forward_with_state(&x, Some((&h0, None))).unwrap();

        // Every timestep must reproduce the initial state exactly.
        for value in output.data().as_f64_slice().unwrap() {
            assert_eq!(*value, 1.0, "state drifted through a saturated gate");
        }
    }
}

// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::{
    Layer,
    init::{InitMethod, init_bias, init_parameter},
};
use crate::nn::layer::{FeatureAxis, check_feature_dim};
use crate::{
    device::Device,
    error::{MinitensorError, Result},
    ops::{attention::scaled_dot_product_attention, linalg::linear, shape_ops::reshape},
    tensor::{DataType, Shape, Tensor},
};
use std::collections::HashMap;

/// Multi-head attention (Vaswani et al., 2017) — the block that makes up a
/// Transformer layer.
///
/// Projects queries, keys and values with learned matrices, splits each into
/// `num_heads` independent subspaces, runs scaled dot-product attention in every
/// head in parallel, then concatenates the heads and applies an output
/// projection. Running several lower-dimensional heads lets the block attend to
/// different relationships at once for the same cost as one full-width head.
///
/// Inputs are batch-first, shape `(batch, seq, embed_dim)`. [`Layer::forward`]
/// performs *self*-attention (queries, keys and values all derive from the same
/// input), which is the shape a Transformer encoder/decoder block needs; use
/// [`MultiheadAttention::forward_qkv`] for cross-attention, where keys and
/// values come from a different sequence.
///
/// `is_causal` masks each query from attending to later positions, making the
/// block autoregressive.
#[derive(Clone)]
pub struct MultiheadAttention {
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    out_proj: Tensor,
    q_bias: Option<Tensor>,
    k_bias: Option<Tensor>,
    v_bias: Option<Tensor>,
    out_bias: Option<Tensor>,
    embed_dim: usize,
    num_heads: usize,
    head_dim: usize,
    is_causal: bool,
}

impl MultiheadAttention {
    /// Create a new multi-head attention block.
    ///
    /// # Arguments
    /// * `embed_dim` - Model width; must be divisible by `num_heads`.
    /// * `num_heads` - Number of parallel attention heads.
    /// * `bias` - Whether the four projections learn additive biases.
    /// * `is_causal` - Apply an autoregressive mask in `forward`.
    /// * `device` - Device to place the parameters on.
    /// * `dtype` - Floating point data type for the parameters.
    pub fn new(
        embed_dim: usize,
        num_heads: usize,
        bias: bool,
        is_causal: bool,
        device: Device,
        dtype: DataType,
    ) -> Result<Self> {
        if embed_dim == 0 || num_heads == 0 {
            return Err(MinitensorError::invalid_argument(
                "MultiheadAttention requires embed_dim and num_heads to be non-zero",
            ));
        }
        if !embed_dim.is_multiple_of(num_heads) {
            return Err(MinitensorError::invalid_argument(format!(
                "embed_dim {} must be divisible by num_heads {}",
                embed_dim, num_heads
            )));
        }
        if !dtype.is_float() {
            return Err(MinitensorError::invalid_argument(
                "MultiheadAttention parameters must have a floating point dtype",
            ));
        }

        let shape = || Shape::new(vec![embed_dim, embed_dim]);
        let proj = |method: InitMethod| init_parameter(shape(), method, dtype, device);
        let make_bias = || -> Result<Option<Tensor>> {
            if bias {
                Ok(Some(init_bias(Shape::new(vec![embed_dim]), dtype, device)?))
            } else {
                Ok(None)
            }
        };

        Ok(Self {
            q_proj: proj(InitMethod::XavierUniform)?,
            k_proj: proj(InitMethod::XavierUniform)?,
            v_proj: proj(InitMethod::XavierUniform)?,
            out_proj: proj(InitMethod::XavierUniform)?,
            q_bias: make_bias()?,
            k_bias: make_bias()?,
            v_bias: make_bias()?,
            out_bias: make_bias()?,
            embed_dim,
            num_heads,
            head_dim: embed_dim / num_heads,
            is_causal,
        })
    }

    /// Model width.
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// Number of attention heads.
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Width of each head (`embed_dim / num_heads`).
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Whether `forward` applies an autoregressive mask.
    pub fn is_causal(&self) -> bool {
        self.is_causal
    }

    /// Apply one `[embed_dim, embed_dim]` projection: `x @ W^T + b`.
    ///
    /// Four of these run per forward pass, so taking the weight transposed by
    /// stride rather than by copy saves four full projection matrices on the
    /// way in and their gradients' worth again on the way back.
    fn project(&self, x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
        linear(x, weight, bias)
    }

    /// `(batch, seq, embed_dim)` -> `(batch, heads, seq, head_dim)`.
    fn split_heads(&self, x: &Tensor, batch: usize, seq: usize) -> Result<Tensor> {
        let split = reshape(
            x,
            Shape::new(vec![batch, seq, self.num_heads, self.head_dim]),
        )?;
        split.transpose(1, 2)
    }

    /// `(batch, heads, seq, head_dim)` -> `(batch, seq, embed_dim)`.
    fn merge_heads(&self, x: &Tensor, batch: usize, seq: usize) -> Result<Tensor> {
        let merged = x.transpose(1, 2)?;
        reshape(&merged, Shape::new(vec![batch, seq, self.embed_dim]))
    }

    /// Validate a `(batch, seq, embed_dim)` operand and return `(batch, seq)`.
    fn check_input(&self, x: &Tensor, name: &str) -> Result<(usize, usize)> {
        let dims = x.shape().dims();
        if dims.len() != 3 {
            return Err(MinitensorError::invalid_operation(format!(
                "MultiheadAttention expects {} with shape (batch, seq, embed_dim), got {:?}",
                name, dims
            )));
        }
        check_feature_dim(
            "MultiheadAttention",
            "embed_dim",
            self.embed_dim,
            x,
            FeatureAxis::Last,
        )?;
        Ok((dims[0], dims[1]))
    }

    /// General attention over separate query, key and value sequences — this is
    /// cross-attention when `key`/`value` come from a different sequence than
    /// `query`, and self-attention when all three are the same tensor.
    ///
    /// `key` and `value` must share a sequence length and batch size; `query`
    /// may have its own sequence length. `attn_mask` is broadcastable to the
    /// per-head scores `(batch, heads, query_seq, key_seq)`; a float mask is
    /// added to the scores and a bool mask keeps `true` positions. Passing a
    /// mask together with causal masking is rejected.
    pub fn forward_qkv(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attn_mask: Option<&Tensor>,
        is_causal: bool,
    ) -> Result<Tensor> {
        let (batch, q_seq) = self.check_input(query, "query")?;
        let (k_batch, k_seq) = self.check_input(key, "key")?;
        let (v_batch, v_seq) = self.check_input(value, "value")?;

        if k_batch != batch || v_batch != batch {
            return Err(MinitensorError::invalid_operation(format!(
                "MultiheadAttention requires a common batch size, got query {}, key {}, value {}",
                batch, k_batch, v_batch
            )));
        }
        if k_seq != v_seq {
            return Err(MinitensorError::invalid_operation(format!(
                "MultiheadAttention requires key and value to share a sequence length, got {} and {}",
                k_seq, v_seq
            )));
        }

        let q = self.project(query, &self.q_proj, self.q_bias.as_ref())?;
        let k = self.project(key, &self.k_proj, self.k_bias.as_ref())?;
        let v = self.project(value, &self.v_proj, self.v_bias.as_ref())?;

        let q = self.split_heads(&q, batch, q_seq)?;
        let k = self.split_heads(&k, batch, k_seq)?;
        let v = self.split_heads(&v, batch, v_seq)?;

        // Attention runs over the trailing (seq, head_dim) axes; the batch and
        // head axes broadcast through matmul inside the primitive.
        let attended = scaled_dot_product_attention(&q, &k, &v, attn_mask, is_causal, None)?;

        let merged = self.merge_heads(&attended, batch, q_seq)?;
        self.project(&merged, &self.out_proj, self.out_bias.as_ref())
    }
}

impl Layer for MultiheadAttention {
    /// Named parameters for serialization.
    fn named_parameters(&self) -> HashMap<String, &Tensor> {
        let mut params = HashMap::new();
        params.insert("q_proj".to_string(), &self.q_proj);
        params.insert("k_proj".to_string(), &self.k_proj);
        params.insert("v_proj".to_string(), &self.v_proj);
        params.insert("out_proj".to_string(), &self.out_proj);
        for (name, bias) in [
            ("q_bias", &self.q_bias),
            ("k_bias", &self.k_bias),
            ("v_bias", &self.v_bias),
            ("out_bias", &self.out_bias),
        ] {
            if let Some(b) = bias {
                params.insert(name.to_string(), b);
            }
        }
        params
    }
    /// Named mutable parameters for state-dict loading.
    fn named_parameters_mut(&mut self) -> HashMap<String, &mut Tensor> {
        let mut params = HashMap::new();
        params.insert("q_proj".to_string(), &mut self.q_proj);
        params.insert("k_proj".to_string(), &mut self.k_proj);
        params.insert("v_proj".to_string(), &mut self.v_proj);
        params.insert("out_proj".to_string(), &mut self.out_proj);
        for (name, bias) in [
            ("q_bias", &mut self.q_bias),
            ("k_bias", &mut self.k_bias),
            ("v_bias", &mut self.v_bias),
            ("out_bias", &mut self.out_bias),
        ] {
            if let Some(b) = bias {
                params.insert(name.to_string(), b);
            }
        }
        params
    }

    /// Self-attention: queries, keys and values are all projections of `input`.
    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        self.forward_qkv(input, input, input, None, self.is_causal)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.q_proj, &self.k_proj, &self.v_proj, &self.out_proj];
        params.extend(
            [&self.q_bias, &self.k_bias, &self.v_bias, &self.out_bias]
                .into_iter()
                .flatten(),
        );
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = vec![
            &mut self.q_proj,
            &mut self.k_proj,
            &mut self.v_proj,
            &mut self.out_proj,
        ];
        params.extend(
            [
                &mut self.q_bias,
                &mut self.k_bias,
                &mut self.v_bias,
                &mut self.out_bias,
            ]
            .into_iter()
            .flatten(),
        );
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(batch: usize, seq: usize, embed: usize) -> Tensor {
        let n = batch * seq * embed;
        let mut t = Tensor::zeros(
            Shape::new(vec![batch, seq, embed]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        t.data_mut()
            .as_f32_slice_mut()
            .unwrap()
            .copy_from_slice(&data);
        t
    }

    #[test]
    fn self_attention_preserves_shape() {
        let mut mha =
            MultiheadAttention::new(8, 2, true, false, Device::cpu(), DataType::Float32).unwrap();
        let x = input(2, 5, 8);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape().dims(), &[2, 5, 8]);
    }

    #[test]
    fn cross_attention_takes_query_sequence_length() {
        let mha =
            MultiheadAttention::new(8, 4, false, false, Device::cpu(), DataType::Float32).unwrap();
        let q = input(2, 3, 8);
        let kv = input(2, 7, 8);
        let out = mha.forward_qkv(&q, &kv, &kv, None, false).unwrap();
        // Output follows the query's sequence length, not the key/value's.
        assert_eq!(out.shape().dims(), &[2, 3, 8]);
    }

    #[test]
    fn parameter_count_matches_projections() {
        let with_bias =
            MultiheadAttention::new(8, 2, true, false, Device::cpu(), DataType::Float32).unwrap();
        // 4 * (8*8) weights + 4 * 8 biases
        assert_eq!(with_bias.num_parameters(), 4 * 64 + 4 * 8);
        assert_eq!(with_bias.parameters().len(), 8);

        let no_bias =
            MultiheadAttention::new(8, 2, false, false, Device::cpu(), DataType::Float32).unwrap();
        assert_eq!(no_bias.num_parameters(), 4 * 64);
        assert_eq!(no_bias.parameters().len(), 4);
    }

    #[test]
    fn rejects_indivisible_embed_dim_and_bad_shapes() {
        assert!(
            MultiheadAttention::new(10, 4, true, false, Device::cpu(), DataType::Float32).is_err()
        );

        let mha =
            MultiheadAttention::new(8, 2, true, false, Device::cpu(), DataType::Float32).unwrap();
        // 2-D input is not (batch, seq, embed).
        let flat = Tensor::zeros(
            Shape::new(vec![4, 8]),
            DataType::Float32,
            Device::cpu(),
            false,
        );
        assert!(mha.forward_qkv(&flat, &flat, &flat, None, false).is_err());

        // Wrong embedding width.
        let wrong = input(2, 3, 6);
        assert!(
            mha.forward_qkv(&wrong, &wrong, &wrong, None, false)
                .is_err()
        );

        // Key and value sequence lengths must agree.
        let q = input(2, 3, 8);
        let k = input(2, 5, 8);
        let v = input(2, 4, 8);
        assert!(mha.forward_qkv(&q, &k, &v, None, false).is_err());
    }

    #[test]
    fn causal_forward_runs_and_backward_reaches_all_parameters() {
        let mut mha =
            MultiheadAttention::new(4, 2, true, true, Device::cpu(), DataType::Float32).unwrap();
        for p in mha.parameters_mut() {
            *p = p.detach().requires_grad_(true);
        }
        let x = input(1, 4, 4);
        let out = mha.forward(&x).unwrap();
        assert_eq!(out.shape().dims(), &[1, 4, 4]);

        let ones = Tensor::ones(out.shape().clone(), out.dtype(), out.device(), false);
        let grads = crate::autograd::backward_collect(&out, Some(ones)).unwrap();
        for p in mha.parameters() {
            assert!(grads.contains_key(&p.id()));
        }
    }
}

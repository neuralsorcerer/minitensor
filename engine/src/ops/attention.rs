// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::device::Device;
use crate::error::{MinitensorError, Result};
use crate::tensor::{DataType, Shape, Tensor, TensorData};
use std::sync::Arc;

fn scalar_tensor(value: f64, dtype: DataType, device: Device) -> Result<Tensor> {
    let mut data = TensorData::zeros_on_device(1, dtype, device);
    match dtype {
        DataType::Float32 => {
            let slice = data.as_f32_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error(
                    "Failed to get mutable f32 slice from scalar tensor",
                )
            })?;
            slice[0] = value as f32;
        }
        DataType::Float64 => {
            let slice = data.as_f64_slice_mut().ok_or_else(|| {
                MinitensorError::internal_error(
                    "Failed to get mutable f64 slice from scalar tensor",
                )
            })?;
            slice[0] = value;
        }
        _ => {
            return Err(MinitensorError::invalid_operation(
                "scaled_dot_product_attention only supports floating point tensors".to_string(),
            ));
        }
    }

    Ok(Tensor::new(
        Arc::new(data),
        Shape::new(vec![1]),
        dtype,
        device,
        false,
    ))
}

/// Scaled dot-product attention — the core primitive of Transformer models
/// (Vaswani et al., "Attention Is All You Need", 2017).
///
/// Computes `softmax(Q Kᵀ / sqrt(E) + bias) V` where the softmax is taken over
/// the key/value sequence axis. The whole computation is assembled from
/// autograd-tracked primitives (`matmul`, `softmax`, `where`, elementwise mul/add),
/// so gradients flow automatically to `query`, `key`, `value` and a float
/// `attn_mask`.
///
/// Shapes (all leading batch axes broadcast, so multi-head layouts such as
/// `(B, H, L, E)` work directly because [`matmul`] broadcasts batch dims):
/// * `query` — `(..., L, E)`
/// * `key`   — `(..., S, E)`
/// * `value` — `(..., S, Ev)`
/// * returns — `(..., L, Ev)`
///
/// `attn_mask` is broadcastable to the attention scores `(..., L, S)`:
/// * a **float** mask is added to the scores before softmax (use
///   `-inf` where attention is disallowed, or a relative-position bias);
/// * a **bool** mask keeps positions that are `true` and disables `false`
///   ones (PyTorch convention).
///
/// `is_causal` applies a causal (autoregressive) mask so query position `i`
/// attends only to key positions `j <= i`. When `L != S` the mask is aligned to
/// the bottom-right of the score matrix, matching PyTorch. Supplying both
/// `attn_mask` and `is_causal` is rejected.
///
/// `scale` overrides the default `1/sqrt(E)` scaling.
pub fn scaled_dot_product_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    attn_mask: Option<&Tensor>,
    is_causal: bool,
    scale: Option<f64>,
) -> Result<Tensor> {
    if !query.dtype().is_float() {
        return Err(MinitensorError::invalid_operation(
            "scaled_dot_product_attention requires floating point query/key/value tensors",
        ));
    }
    if query.dtype() != key.dtype() || query.dtype() != value.dtype() {
        return Err(MinitensorError::type_mismatch(
            format!("{:?}", query.dtype()),
            format!(
                "{:?}",
                if query.dtype() != key.dtype() {
                    key.dtype()
                } else {
                    value.dtype()
                }
            ),
        ));
    }
    if query.device() != key.device() || query.device() != value.device() {
        return Err(MinitensorError::device_mismatch(
            format!("{:?}", query.device()),
            format!(
                "{:?}",
                if query.device() != key.device() {
                    key.device()
                } else {
                    value.device()
                }
            ),
        ));
    }
    if query.ndim() < 2 || key.ndim() < 2 || value.ndim() < 2 {
        return Err(MinitensorError::invalid_operation(
            "scaled_dot_product_attention requires query, key and value to have at least 2 dimensions",
        ));
    }
    if is_causal && attn_mask.is_some() {
        return Err(MinitensorError::invalid_operation(
            "scaled_dot_product_attention does not accept an explicit attn_mask together with is_causal=true",
        ));
    }

    let q_dims = query.shape().dims();
    let k_dims = key.shape().dims();
    let v_dims = value.shape().dims();
    let l = q_dims[q_dims.len() - 2];
    let e = q_dims[q_dims.len() - 1];
    let e_k = k_dims[k_dims.len() - 1];
    let s = k_dims[k_dims.len() - 2];
    let s_v = v_dims[v_dims.len() - 2];
    if e != e_k {
        return Err(MinitensorError::shape_mismatch(
            q_dims.to_vec(),
            k_dims.to_vec(),
        ));
    }
    if s != s_v {
        return Err(MinitensorError::shape_mismatch(
            k_dims.to_vec(),
            v_dims.to_vec(),
        ));
    }

    let scale = scale.unwrap_or_else(|| 1.0 / (e as f64).sqrt());

    // scores = (Q @ Kᵀ) * scale — matmul broadcasts the leading batch axes.
    let kn = key.ndim() as isize;
    let key_t = key.transpose(kn - 2, kn - 1)?;
    let raw_scores = query.matmul(&key_t)?;
    let scale_t = scalar_tensor(scale, raw_scores.dtype(), raw_scores.device())?;
    let mut scores = crate::ops::arithmetic::mul(&raw_scores, &scale_t)?;

    if let Some(mask) = attn_mask {
        if mask.device() != scores.device() {
            return Err(MinitensorError::device_mismatch(
                format!("{:?}", scores.device()),
                format!("{:?}", mask.device()),
            ));
        }
        if mask.dtype().is_float() {
            // Additive bias (cast to the score dtype if needed).
            let mask = if mask.dtype() != scores.dtype() {
                mask.astype(scores.dtype())?
            } else {
                mask.clone()
            };
            scores = crate::ops::arithmetic::add(&scores, &mask)?;
        } else if mask.dtype() == DataType::Bool {
            // Boolean mask: keep `true`, disable `false`.
            let neg_inf = scalar_tensor(f64::NEG_INFINITY, scores.dtype(), scores.device())?;
            scores = crate::ops::selection::where_op(mask, &scores, &neg_inf)?;
        } else {
            return Err(MinitensorError::invalid_operation(
                "scaled_dot_product_attention attn_mask must be a float or bool tensor",
            ));
        }
    }

    if is_causal {
        // Positions to disable are strictly above the (bottom-right aligned)
        // diagonal: mask (i, j) where j - i > s - l, i.e. triu with the
        // diagonal offset `s - l + 1` (triu keeps col - row >= diagonal).
        let ones = Tensor::ones(
            Shape::new(vec![l, s]),
            DataType::Bool,
            scores.device(),
            false,
        );
        let diagonal = s as i64 - l as i64 + 1;
        let causal = ones.triu(diagonal)?;
        scores = crate::ops::selection::masked_fill_scalar(&scores, &causal, f64::NEG_INFINITY)?;
    }

    let last = scores.ndim() - 1;
    let attn = scores.softmax(Some(last))?;
    attn.matmul(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Shape, Tensor, TensorData};
    use std::sync::Arc;

    fn tensor_f32(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> Tensor {
        let mut td = TensorData::zeros_on_device(
            Shape::new(shape.clone()).numel(),
            DataType::Float32,
            Device::cpu(),
        );
        td.as_f32_slice_mut().unwrap().copy_from_slice(&data);
        Tensor::new(
            Arc::new(td),
            Shape::new(shape),
            DataType::Float32,
            Device::cpu(),
            requires_grad,
        )
    }

    #[test]
    fn identity_value_returns_weighted_rows() {
        // With q=k, an all-equal key set makes attention uniform, so the
        // output row is the mean of the value rows.
        let q = tensor_f32(vec![0.0, 0.0], vec![1, 2], false);
        let k = tensor_f32(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2], false);
        let v = tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
        let out = scaled_dot_product_attention(&q, &k, &v, None, false, None).unwrap();
        assert_eq!(out.shape().dims(), &[1, 2]);
        let got = out.data().as_f32_slice().unwrap().to_vec();
        assert!((got[0] - 2.0).abs() < 1e-5);
        assert!((got[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn causal_mask_blocks_future_positions() {
        // Query 0 can only attend to key 0, so its output equals value row 0
        // regardless of value row 1.
        let q = tensor_f32(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2], false);
        let k = tensor_f32(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2], false);
        let v = tensor_f32(vec![1.0, 5.0, 9.0, 13.0], vec![2, 2], false);
        let out = scaled_dot_product_attention(&q, &k, &v, None, true, None).unwrap();
        let got = out.data().as_f32_slice().unwrap().to_vec();
        // Row 0: only key 0 -> value row 0 = [1, 5].
        assert!((got[0] - 1.0).abs() < 1e-5);
        assert!((got[1] - 5.0).abs() < 1e-5);
        // Row 1: keys 0 and 1 equally -> mean of value rows = [5, 9].
        assert!((got[2] - 5.0).abs() < 1e-5);
        assert!((got[3] - 9.0).abs() < 1e-5);
    }

    #[test]
    fn rejects_mask_with_causal() {
        let q = tensor_f32(vec![0.0, 0.0], vec![1, 2], false);
        let k = tensor_f32(vec![0.0, 0.0], vec![1, 2], false);
        let v = tensor_f32(vec![1.0, 2.0], vec![1, 2], false);
        let mask = tensor_f32(vec![0.0], vec![1, 1], false);
        assert!(scaled_dot_product_attention(&q, &k, &v, Some(&mask), true, None).is_err());
    }

    #[test]
    fn backward_flows_to_all_inputs() {
        let q = tensor_f32(vec![0.1, 0.2, 0.3, 0.4], vec![2, 2], true);
        let k = tensor_f32(vec![0.5, 0.6, 0.7, 0.8], vec![2, 2], true);
        let v = tensor_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true);
        let out = scaled_dot_product_attention(&q, &k, &v, None, false, None).unwrap();
        let ones = Tensor::ones(out.shape().clone(), out.dtype(), out.device(), false);
        let grads = crate::autograd::backward_collect(&out, Some(ones)).unwrap();
        assert!(grads.contains_key(&q.id()));
        assert!(grads.contains_key(&k.id()));
        assert!(grads.contains_key(&v.id()));
    }
}

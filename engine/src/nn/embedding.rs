// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use super::{
    Layer,
    init::{InitMethod, init_parameter},
};
use crate::{
    device::Device,
    error::{MinitensorError, Result},
    ops::shape_ops::{index_select, reshape},
    tensor::{DataType, Shape, Tensor, TensorData},
};
use std::{collections::HashMap, sync::Arc};

/// Embedding lookup table — the input layer of essentially every language
/// model. Maps integer token ids to dense vectors by selecting rows of a
/// learned `[num_embeddings, embedding_dim]` weight matrix.
///
/// The lookup is `index_select`, whose backward scatter-adds into the weight
/// rows, so repeated tokens correctly accumulate gradient into the same row.
///
/// `padding_idx`, when set, marks a token whose embedding is fixed at zero: its
/// rows are zeroed in the forward pass by an autograd-tracked mask, which also
/// makes the gradient flowing back into that weight row exactly zero.
#[derive(Clone)]
pub struct Embedding {
    weight: Tensor,
    num_embeddings: usize,
    embedding_dim: usize,
    padding_idx: Option<usize>,
}

impl Embedding {
    /// Create a new embedding table.
    ///
    /// # Arguments
    /// * `num_embeddings` - Size of the vocabulary.
    /// * `embedding_dim` - Width of each embedding vector.
    /// * `padding_idx` - Optional token id whose embedding stays zero.
    /// * `device` - Device to place the weight on.
    /// * `dtype` - Floating point data type for the weight.
    pub fn new(
        num_embeddings: usize,
        embedding_dim: usize,
        padding_idx: Option<usize>,
        device: Device,
        dtype: DataType,
    ) -> Result<Self> {
        if num_embeddings == 0 || embedding_dim == 0 {
            return Err(MinitensorError::invalid_argument(
                "Embedding requires num_embeddings and embedding_dim to be non-zero",
            ));
        }
        if !dtype.is_float() {
            return Err(MinitensorError::invalid_argument(
                "Embedding weight must have a floating point dtype",
            ));
        }
        if let Some(idx) = padding_idx
            && idx >= num_embeddings
        {
            return Err(MinitensorError::invalid_argument(format!(
                "padding_idx {} is out of range for num_embeddings {}",
                idx, num_embeddings
            )));
        }

        // Token embeddings are conventionally drawn from N(0, 1); reuse the
        // shared initializer so dtype/device handling stays in one place.
        let weight = init_parameter(
            Shape::new(vec![num_embeddings, embedding_dim]),
            InitMethod::Normal {
                mean: 0.0,
                std: 1.0,
            },
            dtype,
            device,
        )?;

        Ok(Self {
            weight,
            num_embeddings,
            embedding_dim,
            padding_idx,
        })
    }

    /// Vocabulary size.
    pub fn num_embeddings(&self) -> usize {
        self.num_embeddings
    }

    /// Width of each embedding vector.
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Token id held at zero, if any.
    pub fn padding_idx(&self) -> Option<usize> {
        self.padding_idx
    }

    /// Immutable access to the embedding matrix.
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Read the input index tensor into host indices, validating the range.
    fn host_indices(&self, input: &Tensor) -> Result<Vec<usize>> {
        let input = input.contiguous()?;
        let raw: Vec<i64> = match input.dtype() {
            DataType::Int32 => input
                .data()
                .as_i32_slice()
                .ok_or_else(|| {
                    MinitensorError::internal_error("Failed to read i32 indices for Embedding")
                })?
                .iter()
                .map(|&v| v as i64)
                .collect(),
            DataType::Int64 => input
                .data()
                .as_i64_slice()
                .ok_or_else(|| {
                    MinitensorError::internal_error("Failed to read i64 indices for Embedding")
                })?
                .to_vec(),
            _ => {
                return Err(MinitensorError::invalid_operation(
                    "Embedding expects an integer index tensor (int32 or int64)",
                ));
            }
        };

        raw.into_iter()
            .map(|idx| {
                if idx < 0 || idx as usize >= self.num_embeddings {
                    Err(MinitensorError::index_error(
                        idx as isize,
                        0,
                        self.num_embeddings,
                    ))
                } else {
                    Ok(idx as usize)
                }
            })
            .collect()
    }
}

impl Layer for Embedding {
    /// Named parameters for serialization.
    fn named_parameters(&self) -> HashMap<String, &Tensor> {
        let mut params = HashMap::with_capacity(1);
        params.insert("weight".to_string(), &self.weight);
        params
    }
    /// Named mutable parameters for state-dict loading.
    fn named_parameters_mut(&mut self) -> HashMap<String, &mut Tensor> {
        let mut params = HashMap::with_capacity(1);
        params.insert("weight".to_string(), &mut self.weight);
        params
    }

    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        let indices = self.host_indices(input)?;

        // Gather the rows, then restore the input's shape with the embedding
        // dimension appended. Both ops are autograd-tracked.
        let gathered = index_select(&self.weight, 0, &indices)?;

        let mut out_dims = input.shape().dims().to_vec();
        out_dims.push(self.embedding_dim);
        let mut output = reshape(&gathered, Shape::new(out_dims))?;

        // Zero out padded positions (and, with them, their weight gradient).
        if let Some(pad) = self.padding_idx {
            let mask: Vec<f64> = indices
                .iter()
                .map(|&idx| if idx == pad { 0.0 } else { 1.0 })
                .collect();
            let mut mask_dims = input.shape().dims().to_vec();
            mask_dims.push(1);
            let mask_data = match self.weight.dtype() {
                DataType::Float32 => TensorData::from_vec_f32(
                    mask.iter().map(|&v| v as f32).collect(),
                    self.weight.device(),
                ),
                _ => TensorData::from_vec_f64(mask, self.weight.device()),
            };
            let mask_tensor = Tensor::new(
                Arc::new(mask_data),
                Shape::new(mask_dims),
                self.weight.dtype(),
                self.weight.device(),
                false,
            );
            output = crate::ops::arithmetic::mul(&output, &mask_tensor)?;
        }

        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weight]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weight]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn index_tensor(data: Vec<i64>, shape: Vec<usize>) -> Tensor {
        let mut t = Tensor::zeros(Shape::new(shape), DataType::Int64, Device::cpu(), false);
        t.data_mut()
            .as_i64_slice_mut()
            .unwrap()
            .copy_from_slice(&data);
        t
    }

    #[test]
    fn lookup_shape_and_rows() {
        let mut emb = Embedding::new(5, 3, None, Device::cpu(), DataType::Float32).unwrap();
        let ids = index_tensor(vec![0, 2, 4, 2], vec![2, 2]);
        let out = emb.forward(&ids).unwrap();
        assert_eq!(out.shape().dims(), &[2, 2, 3]);

        // Rows for repeated id 2 must be identical, and match the weight row.
        let got = out.data().as_f32_slice().unwrap().to_vec();
        let w = emb.weight().data().as_f32_slice().unwrap().to_vec();
        for k in 0..3 {
            assert_eq!(got[3 + k], w[2 * 3 + k]);
            assert_eq!(got[9 + k], w[2 * 3 + k]);
        }
    }

    #[test]
    fn padding_idx_returns_zero_rows() {
        let mut emb = Embedding::new(4, 2, Some(0), Device::cpu(), DataType::Float32).unwrap();
        let ids = index_tensor(vec![0, 1], vec![2]);
        let out = emb.forward(&ids).unwrap();
        let got = out.data().as_f32_slice().unwrap().to_vec();
        assert_eq!(got[0], 0.0);
        assert_eq!(got[1], 0.0);
    }

    #[test]
    fn rejects_out_of_range_and_float_input() {
        let mut emb = Embedding::new(3, 2, None, Device::cpu(), DataType::Float32).unwrap();
        assert!(emb.forward(&index_tensor(vec![3], vec![1])).is_err());
        assert!(emb.forward(&index_tensor(vec![-1], vec![1])).is_err());

        let floats = Tensor::zeros(Shape::new(vec![1]), DataType::Float32, Device::cpu(), false);
        assert!(emb.forward(&floats).is_err());
    }

    #[test]
    fn rejects_bad_padding_idx() {
        assert!(Embedding::new(3, 2, Some(3), Device::cpu(), DataType::Float32).is_err());
    }

    #[test]
    fn gradient_accumulates_into_repeated_rows() {
        let mut emb = Embedding::new(3, 2, None, Device::cpu(), DataType::Float32).unwrap();
        emb.weight = emb.weight.detach().requires_grad_(true);
        let ids = index_tensor(vec![1, 1], vec![2]);
        let out = emb.forward(&ids).unwrap();
        let ones = Tensor::ones(out.shape().clone(), out.dtype(), out.device(), false);
        let grads = crate::autograd::backward_collect(&out, Some(ones)).unwrap();
        let g = grads.get(&emb.weight.id()).unwrap();
        let gv = g.data().as_f32_slice().unwrap().to_vec();
        // Row 1 was selected twice, so its gradient is 2; other rows stay 0.
        assert_eq!(gv[0], 0.0);
        assert_eq!(gv[1], 0.0);
        assert_eq!(gv[2], 2.0);
        assert_eq!(gv[3], 2.0);
        assert_eq!(gv[4], 0.0);
        assert_eq!(gv[5], 0.0);
    }
}

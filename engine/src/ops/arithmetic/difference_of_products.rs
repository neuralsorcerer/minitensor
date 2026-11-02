// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! `a · b - c · d`, rounded once.
//!
//! Two products rounded on their own and then subtracted lose what those
//! roundings held, and where the products nearly cancel that is all of the
//! answer: each component of a cross product of nearly parallel vectors came
//! out as rounding noise. The difference is now the float nearest the exact
//! one; see `exact_sum::difference_of_products_f32` and `_f64`.

use std::sync::Arc;

use rustc_hash::FxHashMap;

use super::exact_sum::{difference_of_products_f32, difference_of_products_f64};
use crate::{
    autograd::{GradientFunction, TensorId, accumulate_grad, with_grad_fn},
    error::{MinitensorError, Result},
    ops::arithmetic,
    tensor::{DataType, Tensor, TensorData},
};

/// `a · b - c · d` of four float tensors of one dtype, broadcast together,
/// each element the float nearest the exact value.
pub fn difference_of_products(a: &Tensor, b: &Tensor, c: &Tensor, d: &Tensor) -> Result<Tensor> {
    let dtype = a.dtype();
    if [b, c, d].iter().any(|operand| operand.dtype() != dtype)
        || !matches!(dtype, DataType::Float32 | DataType::Float64)
    {
        return Err(MinitensorError::invalid_operation(format!(
            "difference_of_products takes four float32 or four float64 tensors, got {:?}, {:?}, {:?} and {:?}",
            a.dtype(),
            b.dtype(),
            c.dtype(),
            d.dtype()
        )));
    }
    let shape = a
        .shape()
        .broadcast_with(b.shape())?
        .broadcast_with(c.shape())?
        .broadcast_with(d.shape())?;
    let dims: Vec<isize> = shape.dims().iter().map(|&n| n as isize).collect();
    let spread = |operand: &Tensor| operand.detach().expand(dims.clone())?.contiguous();
    let (wa, wb, wc, wd) = (spread(a)?, spread(b)?, spread(c)?, spread(d)?);
    macro_rules! fill {
        ($accessor:ident, $from_vec:ident, $ty:ty, $kernel:ident) => {{
            let missing =
                || MinitensorError::internal_error("difference_of_products: an operand slice");
            let slices = (
                wa.data().$accessor().ok_or_else(missing)?,
                wb.data().$accessor().ok_or_else(missing)?,
                wc.data().$accessor().ok_or_else(missing)?,
                wd.data().$accessor().ok_or_else(missing)?,
            );
            let mut out = vec![<$ty>::default(); shape.numel()];
            crate::ops::map::par_out_chunks(&mut out, 1 << 14, &|start, block| {
                $kernel(block, start, slices)
            });
            TensorData::$from_vec(out, a.device())
        }};
    }
    let data = match dtype {
        DataType::Float32 => fill!(as_f32_slice, from_vec_f32, f32, fill_f32),
        _ => fill!(as_f64_slice, from_vec_f64, f64, fill_f64),
    };
    let requires_grad = [a, b, c, d].iter().any(|operand| operand.requires_grad())
        && crate::autograd::is_grad_enabled();
    let output = Tensor::new(Arc::new(data), shape, dtype, a.device(), requires_grad);
    if !requires_grad {
        return Ok(output);
    }
    let grad_fn = Arc::new(DifferenceOfProductsBackward {
        operands: [a.detach(), b.detach(), c.detach(), d.detach()],
        input_ids: [a.id(), b.id(), c.id(), d.id()],
        input_requires_grad: [
            a.requires_grad(),
            b.requires_grad(),
            c.requires_grad(),
            d.requires_grad(),
        ],
    });
    with_grad_fn(output, grad_fn)
}

type Operands<'a, T> = (&'a [T], &'a [T], &'a [T], &'a [T]);

fn fill_f32(block: &mut [f32], start: usize, (a, b, c, d): Operands<'_, f32>) {
    for (offset, slot) in block.iter_mut().enumerate() {
        let i = start + offset;
        *slot = difference_of_products_f32(a[i], b[i], c[i], d[i]);
    }
}

fn fill_f64(block: &mut [f64], start: usize, operands: Operands<'_, f64>) {
    #[inline(always)]
    fn body(block: &mut [f64], start: usize, (a, b, c, d): Operands<'_, f64>) {
        for (offset, slot) in block.iter_mut().enumerate() {
            let i = start + offset;
            *slot = difference_of_products_f64(a[i], b[i], c[i], d[i]);
        }
    }
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    fn body_fma(block: &mut [f64], start: usize, operands: Operands<'_, f64>) {
        body(block, start, operands)
    }
    #[cfg(target_arch = "x86_64")]
    if crate::ops::simd::simd_capabilities().avx2_fma {
        // SAFETY: avx2 and fma were detected on this CPU.
        return unsafe { body_fma(block, start, operands) };
    }
    body(block, start, operands)
}

/// The gradient of `a · b - c · d`: `b`, `a`, `-d` and `-c` times the
/// incoming gradient, each summed back over the axes its operand was
/// broadcast along.
struct DifferenceOfProductsBackward {
    operands: [Tensor; 4],
    input_ids: [TensorId; 4],
    input_requires_grad: [bool; 4],
}

impl GradientFunction for DifferenceOfProductsBackward {
    fn backward(&self, grad_output: &Tensor) -> Result<FxHashMap<TensorId, Tensor>> {
        let mut gradients = FxHashMap::default();
        let [a, b, c, d] = &self.operands;
        let partners = [(b, false), (a, false), (d, true), (c, true)];
        for (index, (partner, negated)) in partners.into_iter().enumerate() {
            if !self.input_requires_grad[index] {
                continue;
            }
            let term = arithmetic::mul(grad_output, partner)?;
            let term = if negated {
                arithmetic::neg(&term)?
            } else {
                term
            };
            let shape = self.operands[index].shape();
            let grad = crate::autograd::reduce_gradient_for_broadcasting(&term, shape)?;
            accumulate_grad(&mut gradients, self.input_ids[index], grad)?;
        }
        Ok(gradients)
    }

    fn input_ids(&self) -> &[TensorId] {
        &self.input_ids
    }
}

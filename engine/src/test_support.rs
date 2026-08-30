// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Tensor builders the unit tests share.
//!
//! "Allocate a buffer, copy a `Vec` into it, wrap it in a `Tensor`" had
//! accumulated eighteen hand-written copies across the crate, differing only in
//! the element width and in whether they set `requires_grad`. One generic
//! builder covers every one of them, because [`TensorElement`] already names
//! the dtype each width belongs to -- so the width is inferred from the data
//! rather than restated beside it, and a `Vec<f32>` cannot be labelled
//! `Float64` by a copy-paste.

use crate::{
    device::Device,
    tensor::{Shape, Tensor, TensorData, TensorElement},
};
use std::sync::Arc;

/// A CPU tensor holding `data`, shaped `shape`, at the dtype `T` names.
///
/// Panics on a length mismatch, which in a test is the right response: the
/// fixture is wrong, and continuing would test the wrong thing.
pub(crate) fn tensor_of<T: TensorElement>(
    data: Vec<T>,
    shape: Vec<usize>,
    requires_grad: bool,
) -> Tensor {
    tensor_on(data, shape, requires_grad, Device::cpu())
}

/// [`tensor_of`] on a device other than the CPU.
pub(crate) fn tensor_on<T: TensorElement>(
    data: Vec<T>,
    shape: Vec<usize>,
    requires_grad: bool,
    device: Device,
) -> Tensor {
    let shape = Shape::new(shape);
    assert_eq!(
        data.len(),
        shape.numel(),
        "the fixture's data and shape disagree"
    );
    Tensor::new(
        Arc::new(TensorData::from_vec::<T>(data, T::DTYPE, device)),
        shape,
        T::DTYPE,
        device,
        requires_grad,
    )
}

/// A 1-D CPU tensor holding `data`, which is the shape most unit tests want.
pub(crate) fn vector<T: TensorElement>(data: Vec<T>) -> Tensor {
    let len = data.len();
    tensor_of(data, vec![len], false)
}

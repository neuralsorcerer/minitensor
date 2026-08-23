// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Pooling layers.
//!
//! Neither holds parameters; they are thin, stateful wrappers that remember the
//! window geometry so a `Sequential` can carry them like any other layer.

use crate::{error::Result, nn::layer::Layer, tensor::Tensor};

/// Max pooling over a 4-D `[N, C, H, W]` input.
#[derive(Debug, Clone)]
pub struct MaxPool2d {
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
}

impl MaxPool2d {
    /// Create a max pooling layer.
    ///
    /// `stride` defaults to `kernel_size`, the convention every framework uses
    /// for pooling (unlike convolution, where it defaults to 1).
    pub fn new(
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: Option<(usize, usize)>,
    ) -> Self {
        Self {
            kernel_size,
            stride: stride.unwrap_or(kernel_size),
            padding: padding.unwrap_or((0, 0)),
        }
    }

    pub fn kernel_size(&self) -> (usize, usize) {
        self.kernel_size
    }

    pub fn stride(&self) -> (usize, usize) {
        self.stride
    }

    pub fn padding(&self) -> (usize, usize) {
        self.padding
    }
}

impl Layer for MaxPool2d {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        crate::ops::pooling::max_pool2d(input, self.kernel_size, self.stride, self.padding)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        Vec::new()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        Vec::new()
    }
}

/// Average pooling over a 4-D `[N, C, H, W]` input.
#[derive(Debug, Clone)]
pub struct AvgPool2d {
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    count_include_pad: bool,
}

impl AvgPool2d {
    /// Create an average pooling layer.
    ///
    /// `count_include_pad` defaults to true: padded cells are counted in the
    /// divisor unless asked otherwise.
    pub fn new(
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: Option<(usize, usize)>,
        count_include_pad: Option<bool>,
    ) -> Self {
        Self {
            kernel_size,
            stride: stride.unwrap_or(kernel_size),
            padding: padding.unwrap_or((0, 0)),
            count_include_pad: count_include_pad.unwrap_or(true),
        }
    }

    pub fn kernel_size(&self) -> (usize, usize) {
        self.kernel_size
    }

    pub fn stride(&self) -> (usize, usize) {
        self.stride
    }

    pub fn padding(&self) -> (usize, usize) {
        self.padding
    }

    pub fn count_include_pad(&self) -> bool {
        self.count_include_pad
    }
}

impl Layer for AvgPool2d {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        crate::ops::pooling::avg_pool2d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.count_include_pad,
        )
    }

    fn parameters(&self) -> Vec<&Tensor> {
        Vec::new()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        Vec::new()
    }
}

/// 1-D max pooling layer over `[N, C, L]`.
///
/// `stride` defaults to `kernel_size`, as for the 2-D layer.
#[derive(Debug, Clone)]
pub struct MaxPool1d {
    kernel_size: usize,
    stride: usize,
    padding: usize,
}

impl MaxPool1d {
    pub fn new(kernel_size: usize, stride: Option<usize>, padding: Option<usize>) -> Self {
        Self {
            kernel_size,
            stride: stride.unwrap_or(kernel_size),
            padding: padding.unwrap_or(0),
        }
    }

    pub fn kernel_size(&self) -> usize {
        self.kernel_size
    }

    pub fn stride(&self) -> usize {
        self.stride
    }

    pub fn padding(&self) -> usize {
        self.padding
    }
}

impl Layer for MaxPool1d {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        crate::ops::pooling::max_pool1d(input, self.kernel_size, self.stride, self.padding)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        Vec::new()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        Vec::new()
    }
}

/// 1-D average pooling layer over `[N, C, L]`.
#[derive(Debug, Clone)]
pub struct AvgPool1d {
    kernel_size: usize,
    stride: usize,
    padding: usize,
    count_include_pad: bool,
}

impl AvgPool1d {
    pub fn new(
        kernel_size: usize,
        stride: Option<usize>,
        padding: Option<usize>,
        count_include_pad: bool,
    ) -> Self {
        Self {
            kernel_size,
            stride: stride.unwrap_or(kernel_size),
            padding: padding.unwrap_or(0),
            count_include_pad,
        }
    }

    pub fn kernel_size(&self) -> usize {
        self.kernel_size
    }

    pub fn stride(&self) -> usize {
        self.stride
    }

    pub fn padding(&self) -> usize {
        self.padding
    }

    pub fn count_include_pad(&self) -> bool {
        self.count_include_pad
    }
}

impl Layer for AvgPool1d {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        crate::ops::pooling::avg_pool1d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.count_include_pad,
        )
    }

    fn parameters(&self) -> Vec<&Tensor> {
        Vec::new()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        Vec::new()
    }
}

/// Average pooling to a fixed output size over `[N, C, H, W]` signals.
///
/// The layer a classifier head is made of: a fixed-window pool needs a kernel
/// chosen for one input size, while this derives its windows from the ratio of
/// the extents and so hands the next layer the same shape whatever came in.
///
/// `output_size` of `(1, 1)` is the global average pool that ends most
/// convolutional networks.
#[derive(Clone)]
pub struct AdaptiveAvgPool2d {
    output_size: (usize, usize),
}

impl AdaptiveAvgPool2d {
    /// Create the layer. `output_size` defaults to `(1, 1)`.
    pub fn new(output_size: Option<(usize, usize)>) -> Self {
        Self {
            output_size: output_size.unwrap_or((1, 1)),
        }
    }

    pub fn output_size(&self) -> (usize, usize) {
        self.output_size
    }
}

impl Layer for AdaptiveAvgPool2d {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        crate::ops::pooling::adaptive_avg_pool2d(input, self.output_size)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        Vec::new()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        Vec::new()
    }
}

/// Average pooling to a fixed output size over `[N, C, L]` signals.
///
/// The layer a classifier head is made of: a fixed-window pool needs a kernel
/// chosen for one input size, while this derives its windows from the ratio of
/// the extents and so hands the next layer the same shape whatever came in.
#[derive(Clone)]
pub struct AdaptiveAvgPool1d {
    output_size: usize,
}

impl AdaptiveAvgPool1d {
    /// Create the layer. `output_size` defaults to `1`.
    pub fn new(output_size: Option<usize>) -> Self {
        Self {
            output_size: output_size.unwrap_or(1),
        }
    }

    pub fn output_size(&self) -> usize {
        self.output_size
    }
}

impl Layer for AdaptiveAvgPool1d {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        crate::ops::pooling::adaptive_avg_pool1d(input, self.output_size)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        Vec::new()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        Vec::new()
    }
}

/// Max pooling to a fixed output size over `[N, C, H, W]` signals.
///
/// The layer a classifier head is made of: a fixed-window pool needs a kernel
/// chosen for one input size, while this derives its windows from the ratio of
/// the extents and so hands the next layer the same shape whatever came in.
#[derive(Clone)]
pub struct AdaptiveMaxPool2d {
    output_size: (usize, usize),
}

impl AdaptiveMaxPool2d {
    /// Create the layer. `output_size` defaults to `(1, 1)`.
    pub fn new(output_size: Option<(usize, usize)>) -> Self {
        Self {
            output_size: output_size.unwrap_or((1, 1)),
        }
    }

    pub fn output_size(&self) -> (usize, usize) {
        self.output_size
    }
}

impl Layer for AdaptiveMaxPool2d {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        crate::ops::pooling::adaptive_max_pool2d(input, self.output_size)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        Vec::new()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        Vec::new()
    }
}

/// Max pooling to a fixed output size over `[N, C, L]` signals.
///
/// The layer a classifier head is made of: a fixed-window pool needs a kernel
/// chosen for one input size, while this derives its windows from the ratio of
/// the extents and so hands the next layer the same shape whatever came in.
#[derive(Clone)]
pub struct AdaptiveMaxPool1d {
    output_size: usize,
}

impl AdaptiveMaxPool1d {
    /// Create the layer. `output_size` defaults to `1`.
    pub fn new(output_size: Option<usize>) -> Self {
        Self {
            output_size: output_size.unwrap_or(1),
        }
    }

    pub fn output_size(&self) -> usize {
        self.output_size
    }
}

impl Layer for AdaptiveMaxPool1d {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        crate::ops::pooling::adaptive_max_pool1d(input, self.output_size)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        Vec::new()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        Vec::new()
    }
}

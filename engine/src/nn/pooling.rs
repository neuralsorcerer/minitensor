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
    /// `count_include_pad` defaults to true, matching PyTorch: padded cells are
    /// counted in the divisor unless asked otherwise.
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

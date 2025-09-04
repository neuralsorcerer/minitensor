// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

pub mod activation;
pub mod conv;
pub mod dense_layer;
pub mod dropout;
pub mod init;
pub mod layer;
pub mod loss;
pub mod normalization;
pub mod sequential;
pub mod utils;

// Re-export the main trait and common types
pub use activation::{LeakyReLU, ReLU, Sigmoid, Softmax, Tanh, ELU, GELU};
pub use dense_layer::DenseLayer;
pub use init::{init_bias, init_parameter, InitMethod};
pub use layer::{Layer, Module};
pub use loss::{BCELoss, CrossEntropyLoss, FocalLoss, HuberLoss, MAELoss, MSELoss};
pub use sequential::{Sequential, SequentialBuilder};

// Re-export from other modules when they exist
#[cfg(feature = "conv")]
pub use conv::*;

#[cfg(feature = "dropout")]
pub use dropout::*;

#[cfg(feature = "normalization")]
pub use normalization::*;

// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

pub mod layer;
pub mod dense_layer;
pub mod activation;
pub mod loss;
pub mod sequential;
pub mod init;
pub mod conv;
pub mod dropout;
pub mod normalization;
pub mod utils;

// Re-export the main trait and common types
pub use layer::{Layer, Module};
pub use dense_layer::DenseLayer;
pub use activation::{ReLU, Sigmoid, Tanh, Softmax, LeakyReLU, ELU, GELU};
pub use loss::{MSELoss, MAELoss, HuberLoss, CrossEntropyLoss, BCELoss, FocalLoss};
pub use sequential::{Sequential, SequentialBuilder};
pub use init::{InitMethod, init_parameter, init_bias};

// Re-export from other modules when they exist
#[cfg(feature = "conv")]
pub use conv::*;

#[cfg(feature = "dropout")]
pub use dropout::*;

#[cfg(feature = "normalization")]
pub use normalization::*;
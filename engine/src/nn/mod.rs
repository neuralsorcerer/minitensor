// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

pub mod activation;
pub mod attention;
pub mod conv;
pub mod dense_layer;
pub mod dropout;
pub mod embedding;
pub mod init;
pub mod layer;
pub mod loss;
pub mod normalization;
pub mod pooling;
pub mod recurrent;
pub mod sequential;
pub mod utils;

// Re-export the main trait and common types
pub use activation::{ELU, GELU, LeakyReLU, ReLU, Sigmoid, Softmax, Tanh};
pub use attention::MultiheadAttention;
pub use dense_layer::DenseLayer;
pub use embedding::Embedding;
pub use init::{InitMethod, init_bias, init_parameter};
pub use layer::{Layer, Module};
pub use loss::{
    BCELoss, BCEWithLogitsLoss, CrossEntropyLoss, FocalLoss, HuberLoss, LogCoshLoss, MAELoss,
    MSELoss, SmoothL1Loss,
};
pub use pooling::{
    AdaptiveAvgPool1d, AdaptiveAvgPool2d, AdaptiveMaxPool1d, AdaptiveMaxPool2d, AvgPool1d,
    AvgPool2d, MaxPool1d, MaxPool2d, Upsample,
};
pub use recurrent::{CellKind, GRU, LSTM, Recurrent};
pub use sequential::{Sequential, SequentialBuilder};

pub use conv::*;
pub use dropout::*;
pub use normalization::*;

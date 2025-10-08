# Copyright (c) 2025 Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Neural network module re-exporting Rust-backed components."""

from __future__ import annotations

from typing import List as _List

from .. import _core as _backend

_nn = _backend.nn

DenseLayer = _nn.DenseLayer
BatchNorm1d = _nn.BatchNorm1d
BatchNorm2d = _nn.BatchNorm2d
Conv2d = _nn.Conv2d
Dropout = _nn.Dropout
Dropout2d = _nn.Dropout2d
ELU = _nn.ELU
FocalLoss = _nn.FocalLoss
GELU = _nn.GELU
HuberLoss = _nn.HuberLoss
LeakyReLU = _nn.LeakyReLU
MAELoss = _nn.MAELoss
MSELoss = _nn.MSELoss
ReLU = _nn.ReLU
Sequential = _nn.Sequential
Sigmoid = _nn.Sigmoid
Softmax = _nn.Softmax
Tanh = _nn.Tanh
BCELoss = _nn.BCELoss
CrossEntropyLoss = _nn.CrossEntropyLoss
Module = _nn.Module

# Functional APIs exposed by the backend
batch_norm = _nn.batch_norm
conv2d = _nn.conv2d
cross_entropy = _nn.cross_entropy
dense_layer = _nn.dense_layer

__all__: _List[str] = [
    "Module",
    "Sequential",
    "DenseLayer",
    "BatchNorm1d",
    "BatchNorm2d",
    "Conv2d",
    "Dropout",
    "Dropout2d",
    "ELU",
    "FocalLoss",
    "GELU",
    "HuberLoss",
    "LeakyReLU",
    "MAELoss",
    "MSELoss",
    "ReLU",
    "Sigmoid",
    "Softmax",
    "Tanh",
    "BCELoss",
    "CrossEntropyLoss",
    "dense_layer",
    "conv2d",
    "batch_norm",
    "cross_entropy",
]

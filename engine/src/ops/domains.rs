// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! Fused, differentiable primitives for domains a tensor library leaves out.
//!
//! The same justification as `minitensor.kernels` on the Python side, one
//! level down. Each of these is an operation someone currently writes as a
//! chain of eight or twenty tensor calls, where every link allocates a full
//! intermediate and walks it once; written as a single kernel it allocates the
//! output and nothing else, and its derivative is a closed form rather than a
//! chain of recorded nodes.
//!
//! What is here rather than in Python is decided by whether the fusion needs
//! to be. `kernels.soft_assignment` is a Python `Function` because its
//! backward, written over the `[n, k]` responsibilities, is already the whole
//! saving and the forward is two ordinary calls. These two are not:
//!
//! * [`finance`] — Black-Scholes is about twenty elementwise operations whose
//!   intermediates are never wanted, and whose five partial derivatives all
//!   share `d1`, `d2` and the discount factor. Composed, it is twenty
//!   allocations forward and five separate gradient chains back, each
//!   recomputing `d1`.
//! * [`quantum`] — applying a gate to a state vector is a strided butterfly
//!   over pairs of amplitudes. There is no elementwise vocabulary for it: the
//!   composed form reshapes, slices, multiplies and concatenates, moving
//!   numbers between two halves of one buffer through several full copies of
//!   it.

#[path = "domains/finance.rs"]
pub mod finance;
#[path = "domains/quantum.rs"]
pub mod quantum;

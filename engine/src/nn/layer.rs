// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{error::Result, tensor::Tensor};
use std::collections::HashMap;

/// Trait for neural network layers
pub trait Layer: Send + Sync {
    /// Forward pass through the layer
    fn forward(&mut self, input: &Tensor) -> Result<Tensor>;

    /// Get layer parameters
    fn parameters(&self) -> Vec<&Tensor>;

    /// Get mutable layer parameters
    fn parameters_mut(&mut self) -> Vec<&mut Tensor>;

    /// Get persistent, non-trainable buffers (e.g. BatchNorm running stats).
    /// These are excluded from gradient updates but must be serialized so a
    /// reloaded model reproduces its inference behavior. Default: none.
    fn buffers(&self) -> Vec<&Tensor> {
        Vec::new()
    }

    /// Get mutable persistent buffers (for loading a state dict). Default: none.
    fn buffers_mut(&mut self) -> Vec<&mut Tensor> {
        Vec::new()
    }

    /// Set the layer to training mode
    fn train(&mut self) {
        // Default implementation - override in layers that need it
    }

    /// Set the layer to evaluation mode
    fn eval(&mut self) {
        // Default implementation - override in layers that need it
    }

    /// Get the number of parameters in this layer
    fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.numel()).sum()
    }

    /// Names for this layer's parameters, as they appear in a state dict.
    ///
    /// The default is empty, in which case [`Module::state_dict`] falls back to
    /// positional `param_{i}` keys. Overriding it is what makes a checkpoint
    /// readable and what lets a state dict be loaded into a layer whose
    /// parameter *order* differs.
    ///
    /// These hooks live on `Layer` rather than on `Module` because `Module` is
    /// supplied by a blanket `impl<T: Layer> Module for T`: an implementor
    /// cannot override a method on it without colliding with that impl, so
    /// naming would be unreachable from where layers are actually written.
    fn named_parameters(&self) -> HashMap<String, &Tensor> {
        HashMap::new()
    }

    /// Mutable counterpart of [`Self::named_parameters`], used when loading.
    fn named_parameters_mut(&mut self) -> HashMap<String, &mut Tensor> {
        HashMap::new()
    }

    /// Names for this layer's persistent buffers. See [`Self::named_parameters`].
    fn named_buffers(&self) -> HashMap<String, &Tensor> {
        HashMap::new()
    }

    /// Mutable counterpart of [`Self::named_buffers`], used when loading.
    fn named_buffers_mut(&mut self) -> HashMap<String, &mut Tensor> {
        HashMap::new()
    }
}

/// Base module trait that extends Layer with additional functionality
pub trait Module: Layer {
    /// Apply a function to all parameters.
    ///
    /// `Self: Sized` keeps this generic method out of the vtable so `Module`
    /// stays dyn-compatible; callers holding a `&mut dyn Module` can iterate
    /// `parameters_mut()` directly.
    fn apply<F>(&mut self, f: F) -> Result<()>
    where
        F: Fn(&mut Tensor) -> Result<()>,
        Self: Sized,
    {
        for param in self.parameters_mut() {
            f(param)?;
        }
        Ok(())
    }

    /// Get state dictionary for serialization
    fn state_dict(&self) -> crate::serialization::StateDict {
        let mut state_dict = crate::serialization::StateDict::new();

        // Add parameters (use named if provided, otherwise fall back to indexed names)
        let named_params = self.named_parameters();
        if named_params.is_empty() {
            for (i, tensor) in self.parameters().into_iter().enumerate() {
                let _ = state_dict.add_parameter(format!("param_{}", i), tensor);
            }
        } else {
            for (name, tensor) in named_params {
                let _ = state_dict.add_parameter(name, tensor);
            }
        }

        // Add buffers (named if provided, otherwise indexed like parameters).
        // The indexed fallback is what actually carries BatchNorm running stats
        // through the blanket `impl<T: Layer> Module for T`, which leaves
        // `named_buffers` at its empty default.
        let named_buffers = self.named_buffers();
        if named_buffers.is_empty() {
            for (i, tensor) in self.buffers().into_iter().enumerate() {
                let _ = state_dict.add_buffer(format!("buffer_{}", i), tensor);
            }
        } else {
            for (name, tensor) in named_buffers {
                let _ = state_dict.add_buffer(name, tensor);
            }
        }

        state_dict
    }

    /// Load state dictionary
    fn load_state_dict(
        &mut self,
        state_dict: &crate::serialization::StateDict,
        device: Option<crate::device::Device>,
    ) -> Result<()> {
        // Load parameters
        let mut named_params = self.named_parameters_mut();
        if named_params.is_empty() {
            // Fall back to indexed assignment
            let mut params = self.parameters_mut();
            for (i, param_ref) in params.iter_mut().enumerate() {
                if let Ok(loaded_tensor) =
                    state_dict.load_parameter(&format!("param_{}", i), device)
                {
                    **param_ref = loaded_tensor;
                }
            }
        } else {
            for (name, param_ref) in named_params.iter_mut() {
                if let Ok(loaded_tensor) = state_dict.load_parameter(name, device) {
                    // Replace parameter tensor in-place
                    **param_ref = loaded_tensor;
                }
            }
        }

        // Load buffers (named if provided, otherwise indexed to mirror
        // `state_dict`). The indexed path restores BatchNorm running stats.
        let mut named_buffers = self.named_buffers_mut();
        if named_buffers.is_empty() {
            let mut bufs = self.buffers_mut();
            for (i, buf_ref) in bufs.iter_mut().enumerate() {
                if let Ok(loaded_tensor) = state_dict.load_buffer(&format!("buffer_{}", i), device)
                {
                    **buf_ref = loaded_tensor;
                }
            }
        } else {
            for (name, buf_ref) in named_buffers.iter_mut() {
                if let Ok(loaded_tensor) = state_dict.load_buffer(name, device) {
                    **buf_ref = loaded_tensor;
                }
            }
        }

        Ok(())
    }
}

/// Automatic implementation of Module for all Layer implementations
impl<T: Layer> Module for T {}

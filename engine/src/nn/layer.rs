// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
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
}

/// Base module trait that extends Layer with additional functionality
pub trait Module: Layer {
    /// Get named parameters of the module
    fn named_parameters(&self) -> HashMap<String, &Tensor> {
        HashMap::new() // Default empty implementation
    }

    /// Get named mutable parameters of the module
    fn named_parameters_mut(&mut self) -> HashMap<String, &mut Tensor> {
        HashMap::new() // Default empty implementation
    }

    /// Get named buffers (non-trainable parameters) of the module
    fn named_buffers(&self) -> HashMap<String, &Tensor> {
        HashMap::new() // Default empty implementation
    }

    /// Get named mutable buffers of the module
    fn named_buffers_mut(&mut self) -> HashMap<String, &mut Tensor> {
        HashMap::new() // Default empty implementation
    }

    /// Apply a function to all parameters
    fn apply<F>(&mut self, f: F) -> Result<()>
    where
        F: Fn(&mut Tensor) -> Result<()>,
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

        // Add buffers (default none unless provided by implementation)
        for (name, tensor) in self.named_buffers() {
            let _ = state_dict.add_buffer(name, tensor);
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

        // Load buffers
        let mut named_buffers = self.named_buffers_mut();
        for (name, buf_ref) in named_buffers.iter_mut() {
            if let Ok(loaded_tensor) = state_dict.load_buffer(name, device) {
                // Replace buffer tensor in-place
                **buf_ref = loaded_tensor;
            }
        }

        Ok(())
    }
}

/// Automatic implementation of Module for all Layer implementations
impl<T: Layer> Module for T {}

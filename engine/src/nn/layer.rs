// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    error::{MinitensorError, Result},
    tensor::Tensor,
};
use std::collections::HashMap;

/// Which axis of the input a layer's width is read from.
pub(crate) enum FeatureAxis {
    /// The last axis, as `DenseLayer` and `MultiheadAttention` read it.
    Last,
    /// A fixed position, as the BatchNorm layers read their channel axis.
    At(usize),
}

/// Reject an input whose feature axis disagrees with the width the layer was
/// constructed for.
///
/// Built as `shape_mismatch(vec![expected], vec![got])`, this rendered as
/// "Shape mismatch: expected [10], got [7]" -- two one-element shapes, neither
/// of which the caller has. Their input is `[2, 7]` and their layer was built
/// for 10 features; nothing in the message says which constructor argument
/// `[10]` came from, which axis `7` was read off, or what the input's shape
/// was. The recurrent layers next door already report this as "LSTM expects
/// input feature size 8, got 3", so the library disagreed with itself about
/// how to say the same thing.
pub(crate) fn check_feature_dim(
    layer: &str,
    argument: &str,
    expected: usize,
    input: &Tensor,
    axis: FeatureAxis,
) -> Result<()> {
    let dims = input.shape().dims();
    // `subject` opens a clause ("the last dimension of the input is 7") and
    // `possessive` closes one ("an input whose last dimension is 10"), so the
    // two axis kinds need both forms rather than one shared noun phrase.
    let (index, subject, possessive) = match axis {
        FeatureAxis::Last => (
            dims.len().saturating_sub(1),
            "the last dimension".to_string(),
            "last dimension".to_string(),
        ),
        FeatureAxis::At(index) => (
            index,
            format!("dimension {index}"),
            format!("dimension {index}"),
        ),
    };
    let Some(&actual) = dims.get(index) else {
        return Ok(()); // rank is checked separately, with its own message
    };
    if actual == expected {
        return Ok(());
    }
    Err(MinitensorError::invalid_argument_with_suggestion(
        format!(
            "{layer} was built with {argument}={expected}, but {subject} of the \
             input is {actual} (input shape {dims:?})"
        ),
        format!(
            "Either construct the layer with {argument}={actual}, or give it an input \
             whose {possessive} is {expected}"
        ),
    ))
}

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
    ///
    /// Every entry the layer expects has to be present and shaped like the slot
    /// it lands in. Both checks used to be `if let Ok(..)`, which discarded the
    /// error, and each failure was silent in its own way:
    ///
    /// - a name the state dict did not carry -- a renamed parameter, a
    ///   truncated checkpoint, an empty state dict -- left that slot at whatever
    ///   it already held and reported success, so resuming from the checkpoint
    ///   quietly continued from the initialisation instead;
    /// - a name it did carry but at the wrong shape replaced the slot with that
    ///   tensor, leaving the layer structurally inconsistent. The load still
    ///   reported success and the first forward pass failed on a shape it never
    ///   mentions loading, pointing at the wrong place entirely.
    ///
    /// Both now collect and report, so one message names every problem rather
    /// than making the caller rediscover them one at a time.
    ///
    /// Checking happens before anything is written, so a load that fails leaves
    /// the layer exactly as it was. A caller that catches the error and falls
    /// back gets the model it had, not one with half a checkpoint in it.
    fn load_state_dict(
        &mut self,
        state_dict: &crate::serialization::StateDict,
        device: Option<crate::device::Device>,
    ) -> Result<()> {
        let mut missing: Vec<String> = Vec::new();
        let mut mismatched: Vec<String> = Vec::new();

        // Pass one: look every slot up and compare shapes, through the shared
        // accessors so nothing is modified. `named_*` and `named_*_mut` are
        // required to produce the same names, so what passes here is what the
        // assignment below will find.
        {
            let named = self.named_parameters();
            if named.is_empty() {
                for (i, param) in self.parameters().iter().enumerate() {
                    let name = format!("param_{}", i);
                    let loaded = state_dict.load_parameter(&name, device);
                    check_loadable(name, param, loaded, &mut missing, &mut mismatched);
                }
            } else {
                for (name, param) in named {
                    let loaded = state_dict.load_parameter(&name, device);
                    check_loadable(name, param, loaded, &mut missing, &mut mismatched);
                }
            }
        }
        {
            let named = self.named_buffers();
            if named.is_empty() {
                for (i, buffer) in self.buffers().iter().enumerate() {
                    let name = format!("buffer_{}", i);
                    let loaded = state_dict.load_buffer(&name, device);
                    check_loadable(name, buffer, loaded, &mut missing, &mut mismatched);
                }
            } else {
                for (name, buffer) in named {
                    let loaded = state_dict.load_buffer(&name, device);
                    check_loadable(name, buffer, loaded, &mut missing, &mut mismatched);
                }
            }
        }

        if !missing.is_empty() || !mismatched.is_empty() {
            missing.sort();
            mismatched.sort();
            let mut report = String::from("load_state_dict: ");
            if !missing.is_empty() {
                report.push_str(&format!(
                    "missing from the state dict: {}",
                    missing.join(", ")
                ));
            }
            if !mismatched.is_empty() {
                if !missing.is_empty() {
                    report.push_str("; ");
                }
                report.push_str(&format!("wrong shape: {}", mismatched.join(", ")));
            }
            return Err(MinitensorError::invalid_operation(report));
        }

        // Pass two: assign. Every lookup above succeeded at the right shape, so
        // a failure here would mean the two accessors disagree.
        let mut named_params = self.named_parameters_mut();
        if named_params.is_empty() {
            let mut params = self.parameters_mut();
            for (i, param_ref) in params.iter_mut().enumerate() {
                if let Ok(loaded) = state_dict.load_parameter(&format!("param_{}", i), device) {
                    **param_ref = loaded;
                }
            }
        } else {
            for (name, param_ref) in named_params.iter_mut() {
                if let Ok(loaded) = state_dict.load_parameter(name, device) {
                    **param_ref = loaded;
                }
            }
        }

        // Buffers (named if provided, otherwise indexed to mirror
        // `state_dict`). The indexed path restores BatchNorm running stats.
        let mut named_buffers = self.named_buffers_mut();
        if named_buffers.is_empty() {
            let mut bufs = self.buffers_mut();
            for (i, buf_ref) in bufs.iter_mut().enumerate() {
                if let Ok(loaded) = state_dict.load_buffer(&format!("buffer_{}", i), device) {
                    **buf_ref = loaded;
                }
            }
        } else {
            for (name, buf_ref) in named_buffers.iter_mut() {
                if let Ok(loaded) = state_dict.load_buffer(name, device) {
                    **buf_ref = loaded;
                }
            }
        }

        Ok(())
    }
}

/// Record why `loaded` cannot go into `slot`, if it cannot.
fn check_loadable(
    name: String,
    slot: &Tensor,
    loaded: Result<Tensor>,
    missing: &mut Vec<String>,
    mismatched: &mut Vec<String>,
) {
    match loaded {
        Ok(tensor) if tensor.shape().dims() == slot.shape().dims() => {}
        Ok(tensor) => mismatched.push(format!(
            "{name} (expected {:?}, got {:?})",
            slot.shape().dims(),
            tensor.shape().dims()
        )),
        Err(_) => missing.push(name),
    }
}

/// Automatic implementation of Module for all Layer implementations
impl<T: Layer> Module for T {}

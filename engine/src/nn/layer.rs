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
/// Emits the four `Layer` parameter accessors for a layer holding a `weight`
/// and an optional `bias` — `parameters`, `parameters_mut`,
/// `named_parameters` and `named_parameters_mut`.
///
/// The four are not merely repetitive, they have to agree: `state_dict` reads
/// the named views and an optimizer steps the unnamed ones, so a layer that
/// grows a parameter and updates only one pair gets trained on a set it does
/// not save, or saves one it does not train. Neither is a compile error and
/// neither shows up in a forward pass. Writing all four from one place is what
/// makes them one decision instead of four.
///
/// Only the layers whose parameters are exactly this shape use it — `Conv1d`,
/// `Conv2d` and `DenseLayer`. `Embedding` has no bias, `LayerNorm`'s is not
/// optional, and `BatchNorm`'s weight is optional too; each of those writes its
/// own, because pretending otherwise would need a macro with more cases than
/// callers.
#[macro_export]
macro_rules! weight_and_optional_bias_parameters {
    () => {
        fn parameters(&self) -> Vec<&$crate::tensor::Tensor> {
            let mut params = Vec::with_capacity(1 + self.bias.is_some() as usize);
            params.push(&self.weight);
            if let Some(ref bias) = self.bias {
                params.push(bias);
            }
            params
        }

        fn parameters_mut(&mut self) -> Vec<&mut $crate::tensor::Tensor> {
            let mut params = Vec::with_capacity(1 + self.bias.is_some() as usize);
            params.push(&mut self.weight);
            if let Some(ref mut bias) = self.bias {
                params.push(bias);
            }
            params
        }

        fn named_parameters(&self) -> std::collections::HashMap<String, &$crate::tensor::Tensor> {
            let mut params =
                std::collections::HashMap::with_capacity(1 + self.bias.is_some() as usize);
            params.insert("weight".to_string(), &self.weight);
            if let Some(ref bias) = self.bias {
                params.insert("bias".to_string(), bias);
            }
            params
        }

        fn named_parameters_mut(
            &mut self,
        ) -> std::collections::HashMap<String, &mut $crate::tensor::Tensor> {
            let mut params =
                std::collections::HashMap::with_capacity(1 + self.bias.is_some() as usize);
            params.insert("weight".to_string(), &mut self.weight);
            if let Some(ref mut bias) = self.bias {
                params.insert("bias".to_string(), bias);
            }
            params
        }
    };
}

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

#[cfg(test)]
mod parameter_view_tests {
    use crate::nn::{Conv1d, Conv2d, DenseLayer, Layer};

    /// A layer's four parameter accessors are four views of one set, and the
    /// two pairs are read by different callers: `state_dict` saves the named
    /// view, an optimizer steps the unnamed one. If they disagree, a parameter
    /// is trained but never saved, or saved but never trained — and neither
    /// shows up in a forward pass or in any gradient check.
    ///
    /// `weight_and_optional_bias_parameters!` is what keeps them one decision
    /// for the three layers that share this shape; this says what that decision
    /// has to produce, with and without a bias.
    #[test]
    fn the_named_and_unnamed_views_describe_the_same_parameters() {
        let dev = crate::device::Device::cpu();
        let dt = crate::tensor::DataType::Float32;
        let mut layers: Vec<(&str, Box<dyn Layer>)> = vec![
            (
                "DenseLayer+bias",
                Box::new(DenseLayer::new(4, 3, true, dev, dt).unwrap()),
            ),
            (
                "DenseLayer-bias",
                Box::new(DenseLayer::new(4, 3, false, dev, dt).unwrap()),
            ),
            (
                "Conv1d+bias",
                Box::new(Conv1d::new(2, 3, 3, None, None, None, None, true, dev, dt).unwrap()),
            ),
            (
                "Conv1d-bias",
                Box::new(Conv1d::new(2, 3, 3, None, None, None, None, false, dev, dt).unwrap()),
            ),
            (
                "Conv2d+bias",
                Box::new(Conv2d::new(2, 3, (3, 3), None, None, None, None, true, dev, dt).unwrap()),
            ),
            (
                "Conv2d-bias",
                Box::new(
                    Conv2d::new(2, 3, (3, 3), None, None, None, None, false, dev, dt).unwrap(),
                ),
            ),
        ];

        for (name, layer) in layers.iter_mut() {
            let named: Vec<String> = {
                let mut keys: Vec<String> = layer.named_parameters().keys().cloned().collect();
                keys.sort();
                keys
            };
            let expected: Vec<String> = if name.ends_with("+bias") {
                vec!["bias".to_string(), "weight".to_string()]
            } else {
                vec!["weight".to_string()]
            };
            assert_eq!(named, expected, "{name}: named parameters");

            assert_eq!(
                layer.parameters().len(),
                named.len(),
                "{name}: the unnamed view has a different number of parameters than the named one"
            );
            assert_eq!(
                layer.named_parameters_mut().len(),
                named.len(),
                "{name}: the mutable named view disagrees with the shared one"
            );
            assert_eq!(
                layer.parameters_mut().len(),
                named.len(),
                "{name}: the mutable unnamed view disagrees"
            );

            // The two immutable views must point at the same tensors, not
            // merely agree on how many there are.
            let by_id: std::collections::HashSet<_> =
                layer.parameters().iter().map(|t| t.id()).collect();
            let named_by_id: std::collections::HashSet<_> =
                layer.named_parameters().values().map(|t| t.id()).collect();
            assert_eq!(
                by_id, named_by_id,
                "{name}: the two views name different tensors"
            );
        }
    }
}

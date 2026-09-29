// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    error::{MinitensorError, Result},
    tensor::Tensor,
};
use std::collections::{HashMap, HashSet};

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

/// [`Layer::clone_layer`] for a layer that is `Clone`, which every built-in
/// layer is.
macro_rules! cloneable_layer {
    () => {
        fn clone_layer(&self) -> Option<Box<dyn $crate::nn::layer::Layer>> {
            Some(Box::new(self.clone()))
        }
    };
}
pub(crate) use cloneable_layer;

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

    /// A boxed copy of this layer, or `None` for one that cannot be copied.
    ///
    /// A container holds its children as `Box<dyn Layer>`, which cannot be
    /// cloned by type, so it copies itself by asking each child for this. The
    /// built-in layers all answer; the default is `None` so that a layer
    /// written outside the crate does not have to be `Clone`.
    ///
    /// The copy shares its tensors with the original, as `Clone` does: a
    /// caller that needs independent parameters replaces them afterwards.
    fn clone_layer(&self) -> Option<Box<dyn Layer>> {
        None
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

    /// The names [`Self::state_dict`] files this module's parameters under:
    /// its own names if it gives them, positional `param_{i}` keys if not.
    fn parameter_names(&self) -> Vec<String> {
        let named = self.named_parameters();
        if named.is_empty() {
            (0..self.parameters().len())
                .map(|i| format!("param_{i}"))
                .collect()
        } else {
            named.into_keys().collect()
        }
    }

    /// The names [`Self::state_dict`] files this module's buffers under; see
    /// [`Self::parameter_names`].
    fn buffer_names(&self) -> Vec<String> {
        let named = self.named_buffers();
        if named.is_empty() {
            (0..self.buffers().len())
                .map(|i| format!("buffer_{i}"))
                .collect()
        } else {
            named.into_keys().collect()
        }
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
    /// Every entry the layer expects has to be present, shaped like the slot
    /// it lands in and of its dtype. The first two checks used to be
    /// `if let Ok(..)`, which discarded the error, and each failure was silent
    /// in its own way:
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
    /// A tensor of the right shape and another dtype -- a float64 checkpoint
    /// loaded into a float32 layer -- changed the layer's dtype in place, and
    /// the first forward pass then refused its float32 input. Converting on the
    /// way in would decide the precision for the caller, so it is reported too.
    ///
    /// An entry the layer has no slot for is reported as well. Ignoring it let
    /// a checkpoint of a deeper model load into a shallower one: every slot the
    /// two shared was filled, the rest of the checkpoint was dropped, and the
    /// load reported success on weights that were never the ones trained.
    ///
    /// All of them collect and report, so one message names every problem
    /// rather than making the caller rediscover them one at a time.
    ///
    /// Checking happens before anything is written, so a load that fails leaves
    /// the layer exactly as it was. A caller that catches the error and falls
    /// back gets the model it had, not one with half a checkpoint in it.
    ///
    /// The values are written *into* the tensors the layer already holds, not
    /// swapped in as new ones. An optimizer keys a parameter by its identity
    /// and updates it through its own handle, so replacing the tensors left an
    /// optimizer built before the load stepping tensors the layer no longer
    /// had: every step succeeded and the model never changed. Writing in place
    /// keeps each slot's identity and whether it trains, and a parameter's
    /// storage too, trainable or frozen, so every handle to a parameter sees
    /// the loaded values. That write is refused, like any other, while a
    /// pending backward pass still needs the old ones.
    fn load_state_dict(
        &mut self,
        state_dict: &crate::serialization::StateDict,
        device: Option<crate::device::Device>,
    ) -> Result<()> {
        let mut problems = LoadProblems::default();

        // Pass one: look every slot up and compare shapes, through the shared
        // accessors so nothing is modified. `named_*` and `named_*_mut` are
        // required to produce the same names, so what passes here is what the
        // write below will find.
        let parameters: HashSet<String> = {
            let named = self.named_parameters();
            if named.is_empty() {
                for (i, param) in self.parameters().iter().enumerate() {
                    let name = format!("param_{}", i);
                    let loaded = state_dict.load_parameter(&name, device);
                    problems.check(name, param, loaded, true);
                }
            } else {
                for (name, param) in named {
                    let loaded = state_dict.load_parameter(&name, device);
                    problems.check(name, param, loaded, true);
                }
            }
            self.parameter_names().into_iter().collect()
        };
        let buffers: HashSet<String> = {
            let named = self.named_buffers();
            if named.is_empty() {
                for (i, buffer) in self.buffers().iter().enumerate() {
                    let name = format!("buffer_{}", i);
                    let loaded = state_dict.load_buffer(&name, device);
                    problems.check(name, buffer, loaded, false);
                }
            } else {
                for (name, buffer) in named {
                    let loaded = state_dict.load_buffer(&name, device);
                    problems.check(name, buffer, loaded, false);
                }
            }
            self.buffer_names().into_iter().collect()
        };
        problems.unexpected_entries(state_dict, &parameters, &buffers);
        problems.into_result()?;

        // Pass two: write. Every lookup above succeeded at the right shape and
        // dtype with nothing holding the old values, so a failure here would
        // mean the two accessors disagree.
        let mut named_params = self.named_parameters_mut();
        if named_params.is_empty() {
            let mut params = self.parameters_mut();
            for (i, param_ref) in params.iter_mut().enumerate() {
                if let Ok(loaded) = state_dict.load_parameter(&format!("param_{}", i), device) {
                    write_loaded(param_ref, loaded, true)?;
                }
            }
        } else {
            for (name, param_ref) in named_params.iter_mut() {
                if let Ok(loaded) = state_dict.load_parameter(name, device) {
                    write_loaded(param_ref, loaded, true)?;
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
                    write_loaded(buf_ref, loaded, false)?;
                }
            }
        } else {
            for (name, buf_ref) in named_buffers.iter_mut() {
                if let Ok(loaded) = state_dict.load_buffer(name, device) {
                    write_loaded(buf_ref, loaded, false)?;
                }
            }
        }

        Ok(())
    }
}

/// Put `loaded`'s values in `slot`.
///
/// Into the slot's own storage when both are on the CPU, which keeps its
/// identity. A parameter is written `through` its storage even while frozen:
/// an optimizer built before the load holds a handle to that storage, and a
/// frozen parameter given storage of its own would leave the optimizer
/// stepping the old one once the parameter was unfrozen. A buffer is stepped
/// by nothing, so it follows the ordinary copy-on-write rule.
///
/// A load onto another device moves the layer there, so that slot is
/// replaced -- keeping its own `requires_grad`, because a load replaces
/// values, not whether a slot trains: a state dict built from plain tensors
/// carries no flag, and taking the loaded tensor's froze every parameter it
/// reached.
fn write_loaded(slot: &mut Tensor, loaded: Tensor, through: bool) -> Result<()> {
    if slot.device().is_cpu() && loaded.device() == slot.device() {
        slot.write_values_from(&loaded, through)
    } else {
        *slot = loaded.requires_grad_(slot.requires_grad());
        Ok(())
    }
}

/// Why a state dict cannot be loaded, collected over every slot so that one
/// message names every problem.
#[derive(Default)]
struct LoadProblems {
    missing: Vec<String>,
    unexpected: Vec<String>,
    mismatched: Vec<String>,
    mistyped: Vec<String>,
    in_use: Vec<String>,
}

impl LoadProblems {
    /// Record why `loaded` cannot go into `slot`, if it cannot. A wrong shape
    /// is reported before a wrong dtype, since it is the one that cannot be
    /// fixed by a conversion.
    ///
    /// A slot a pending backward pass still reads cannot be written in place
    /// without changing the gradients that pass will produce, which is the
    /// rule every in-place write follows; it is checked here so the load is
    /// refused before anything has been written rather than halfway through.
    /// Written `through` (see [`write_loaded`]), a frozen parameter is written
    /// in place too, and a pass reads it as an operand all the same.
    fn check(&mut self, name: String, slot: &Tensor, loaded: Result<Tensor>, through: bool) {
        match loaded {
            Ok(tensor) if tensor.shape().dims() != slot.shape().dims() => {
                self.mismatched.push(format!(
                    "{name} (expected {:?}, got {:?})",
                    slot.shape().dims(),
                    tensor.shape().dims()
                ))
            }
            Ok(tensor) if tensor.dtype() != slot.dtype() => self.mistyped.push(format!(
                "{name} (expected {}, got {})",
                slot.dtype(),
                tensor.dtype()
            )),
            Ok(_) => {
                let in_place = through || slot.requires_grad();
                if in_place && slot.is_leaf() && crate::autograd::is_consumed_by_live_graph(slot) {
                    self.in_use.push(name);
                }
            }
            Err(_) => self.missing.push(name),
        }
    }

    /// Record every entry of `state_dict` the module has no slot for. A
    /// parameter and a buffer are separate namespaces, so an entry filed under
    /// the wrong one is unexpected there and its slot is missing; saying which
    /// namespace it belongs to explains both at once.
    fn unexpected_entries(
        &mut self,
        state_dict: &crate::serialization::StateDict,
        parameters: &HashSet<String>,
        buffers: &HashSet<String>,
    ) {
        for name in state_dict.parameters.keys() {
            if !parameters.contains(name) {
                self.unexpected.push(if buffers.contains(name) {
                    format!("{name} (given as a parameter; it is a buffer)")
                } else {
                    name.clone()
                });
            }
        }
        for name in state_dict.buffers.keys() {
            if !buffers.contains(name) {
                self.unexpected.push(if parameters.contains(name) {
                    format!("{name} (given as a buffer; it is a parameter)")
                } else {
                    format!("{name} (buffer)")
                });
            }
        }
    }

    fn into_result(self) -> Result<()> {
        let problems: Vec<String> = [
            ("missing from the state dict", self.missing),
            ("not in this module", self.unexpected),
            ("wrong shape", self.mismatched),
            ("wrong dtype", self.mistyped),
            (
                "still needed by a pending backward pass (call backward() or \
                 clear_autograd_graph() first)",
                self.in_use,
            ),
        ]
        .into_iter()
        .filter(|(_, names)| !names.is_empty())
        .map(|(kind, mut names)| {
            names.sort();
            format!("{kind}: {}", names.join(", "))
        })
        .collect();
        if problems.is_empty() {
            Ok(())
        } else {
            Err(MinitensorError::invalid_operation(format!(
                "load_state_dict: {}",
                problems.join("; ")
            )))
        }
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

    /// A load sets values, not whether a slot trains. A state dict built from
    /// tensors that do not require a gradient used to freeze every parameter
    /// it reached; the other direction -- a frozen layer loading a checkpoint
    /// saved from a trainable one -- has to keep the layer frozen.
    #[test]
    fn a_load_keeps_each_slot_trainable_or_frozen_as_it_was() {
        use super::Module;
        use crate::serialization::StateDict;
        use crate::tensor::{Shape, Tensor};

        let dev = crate::device::Device::cpu();
        let dt = crate::tensor::DataType::Float32;
        let plain = |dims: &[usize], requires_grad: bool| {
            Tensor::zeros(Shape::new(dims.to_vec()), dt, dev, false).requires_grad_(requires_grad)
        };
        let state_of = |requires_grad: bool| {
            let mut state = StateDict::new();
            state
                .add_parameter("weight".into(), &plain(&[3, 4], requires_grad))
                .unwrap();
            state
                .add_parameter("bias".into(), &plain(&[3], requires_grad))
                .unwrap();
            state
        };

        let mut trainable = DenseLayer::new(4, 3, true, dev, dt).unwrap();
        Module::load_state_dict(&mut trainable, &state_of(false), None).unwrap();
        assert!(trainable.parameters().iter().all(|p| p.requires_grad()));

        let mut frozen = DenseLayer::new(4, 3, true, dev, dt).unwrap();
        for p in frozen.parameters_mut() {
            *p = p.clone().requires_grad_(false);
        }
        Module::load_state_dict(&mut frozen, &state_of(true), None).unwrap();
        assert!(frozen.parameters().iter().all(|p| !p.requires_grad()));
    }

    /// A load writes into the tensors the layer holds. An optimizer steps
    /// parameters through handles taken before the load, keyed by identity;
    /// replacing the tensors left those handles pointing at storage the layer
    /// no longer read, so training silently stopped.
    #[test]
    fn a_load_keeps_identity_and_reaches_every_handle() {
        use super::Module;
        use crate::serialization::StateDict;
        use crate::tensor::{Shape, Tensor};

        let dev = crate::device::Device::cpu();
        let dt = crate::tensor::DataType::Float32;
        let mut layer = DenseLayer::new(4, 3, true, dev, dt).unwrap();
        let handles: Vec<Tensor> = layer.parameters().into_iter().cloned().collect();

        let mut state = StateDict::new();
        let ones = |dims: &[usize]| Tensor::ones(Shape::new(dims.to_vec()), dt, dev, false);
        state
            .add_parameter("weight".into(), &ones(&[3, 4]))
            .unwrap();
        state.add_parameter("bias".into(), &ones(&[3])).unwrap();
        Module::load_state_dict(&mut layer, &state, None).unwrap();

        for (handle, param) in handles.iter().zip(layer.parameters()) {
            assert_eq!(handle.id(), param.id());
            assert!(
                handle
                    .data()
                    .as_f32_slice()
                    .unwrap()
                    .iter()
                    .all(|&v| v == 1.0)
            );
        }
    }

    /// An entry the layer has no slot for is refused, and the layer keeps the
    /// values it had. Ignoring it loaded the shared part of a checkpoint from
    /// a larger model and reported success.
    #[test]
    fn an_entry_the_layer_has_no_slot_for_is_refused() {
        use super::Module;
        use crate::serialization::StateDict;
        use crate::tensor::{Shape, Tensor};

        let dev = crate::device::Device::cpu();
        let dt = crate::tensor::DataType::Float32;
        let ones = |dims: &[usize]| Tensor::ones(Shape::new(dims.to_vec()), dt, dev, false);
        let mut layer = DenseLayer::new(4, 3, true, dev, dt).unwrap();
        let before = layer.parameters()[0]
            .data()
            .as_f32_slice()
            .unwrap()
            .to_vec();

        let mut state = StateDict::new();
        state
            .add_parameter("weight".into(), &ones(&[3, 4]))
            .unwrap();
        state.add_parameter("bias".into(), &ones(&[3])).unwrap();
        state.add_parameter("scale".into(), &ones(&[3])).unwrap();
        state.add_buffer("bias".into(), &ones(&[3])).unwrap();

        let message = Module::load_state_dict(&mut layer, &state, None)
            .unwrap_err()
            .to_string();
        assert!(
            message
                .contains("not in this module: bias (given as a buffer; it is a parameter), scale"),
            "{message}"
        );
        assert_eq!(
            layer.parameters()[0].data().as_f32_slice().unwrap(),
            before.as_slice()
        );
    }
}

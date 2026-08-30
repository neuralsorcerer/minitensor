// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::error::_convert_error;
use crate::tensor::PyTensor;
use engine::autograd::NoGradGuard;
use engine::custom_ops::{
    BackwardContext, CustomOpBuilder, examples::register_example_ops, execute_custom_op,
    is_custom_op_registered, list_custom_ops, register_custom_op as register_op,
    unregister_custom_op,
};
use engine::error::MinitensorError;
use engine::{autograd::TensorId, tensor::Tensor};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use rustc_hash::FxHashMap;

/// Turn a Python exception into an engine error naming where it came from.
///
/// The message is the exception's own, because a user debugging their own
/// forward function needs to see their own traceback text -- not a wrapper
/// saying only that "the custom operation failed".
fn from_python(op: &str, stage: &str, error: PyErr) -> MinitensorError {
    Python::attach(|py| {
        MinitensorError::invalid_operation(format!(
            "custom op '{op}': its {stage} raised {}",
            error.value(py)
        ))
    })
}

/// The tensors of a slice, as Python objects a callable can be handed.
fn as_python_tensors<'py>(
    py: Python<'py>,
    tensors: impl IntoIterator<Item = Tensor>,
) -> PyResult<Bound<'py, PyTuple>> {
    let objects: PyResult<Vec<Py<PyAny>>> = tensors
        .into_iter()
        .map(|tensor| Ok(Py::new(py, PyTensor::from_tensor(tensor))?.into()))
        .collect();
    PyTuple::new(py, objects?)
}

/// A `Tensor` out of whatever the callable returned, or a message saying what
/// it returned instead.
fn tensor_from_result(
    op: &str,
    stage: &str,
    value: &Bound<PyAny>,
) -> Result<Tensor, MinitensorError> {
    if let Ok(tensor) = value.extract::<PyTensor>() {
        return Ok(tensor.tensor().clone());
    }
    // A layer wrapping a tensor is the other thing callers hand back.
    if let Ok(inner) = value.getattr("_tensor")
        && let Ok(tensor) = inner.extract::<PyTensor>()
    {
        return Ok(tensor.tensor().clone());
    }
    Err(MinitensorError::invalid_operation(format!(
        "custom op '{op}': its {stage} returned {}, not a Tensor",
        value
            .get_type()
            .name()
            .map(|name| name.to_string())
            .unwrap_or_else(|_| "an unknown type".to_string())
    )))
}

/// Register an operation whose forward and backward are Python callables.
///
/// This is the extension point: an operation the library does not have becomes
/// one it does, participating in autograd on the same terms as the built-in
/// ops, without a Rust toolchain or a rebuild.
///
/// `forward` is called with the input tensors as positional arguments and must
/// return a tensor. `backward`, if given, is called with the incoming gradient,
/// a tuple of the saved inputs, and the saved output, and must return one
/// gradient per input -- a single tensor when there is one input, otherwise a
/// sequence, in which `None` means no gradient flows to that input.
///
/// With a `backward`, the forward runs with gradient recording *off*: the
/// operations inside it are an implementation detail, and the gradient is
/// whatever `backward` says it is. Without one, the forward is recorded
/// normally and the operation is differentiable by composition, exactly as a
/// plain Python function would be. Those are the two things a caller can
/// sensibly mean, and which one they get is decided by whether they wrote a
/// backward.
#[pyfunction]
#[pyo3(signature = (name, forward, backward=None, num_inputs=1))]
fn register_custom_op(
    name: &str,
    forward: Py<PyAny>,
    backward: Option<Py<PyAny>>,
    num_inputs: usize,
) -> PyResult<()> {
    if num_inputs == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "a custom operation needs at least one input",
        ));
    }
    Python::attach(|py| {
        for (role, callable) in [("forward", Some(&forward)), ("backward", backward.as_ref())] {
            if let Some(callable) = callable
                && !callable.bind(py).is_callable()
            {
                return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "the {role} of a custom operation must be callable"
                )));
            }
        }
        Ok(())
    })?;

    let owned = name.to_string();
    let detached = backward.is_some();
    let forward_name = owned.clone();
    let mut builder =
        CustomOpBuilder::new(name, num_inputs).forward(move |inputs: &[&Tensor]| {
            Python::attach(|py| {
                let args = as_python_tensors(py, inputs.iter().map(|t| (*t).clone()))
                    .map_err(|e| from_python(&forward_name, "forward", e))?;
                // With a backward of its own, what the forward does internally is
                // an implementation detail: recording it would put a second path
                // to the same gradient in the graph, and the two would add.
                let returned = if detached {
                    let _guard = NoGradGuard::new();
                    forward.call1(py, &args)
                } else {
                    forward.call1(py, &args)
                }
                .map_err(|e| from_python(&forward_name, "forward", e))?;
                tensor_from_result(&forward_name, "forward", returned.bind(py))
            })
        });

    if let Some(backward) = backward {
        let backward_name = owned.clone();
        builder = builder
            .backward(move |ctx: &BackwardContext| python_backward(&backward_name, &backward, ctx));
    }

    register_op(builder.build().map_err(_convert_error)?).map_err(_convert_error)
}

/// Call a Python `backward` and turn what it returns into one gradient per
/// input.
fn python_backward(
    name: &str,
    backward: &Py<PyAny>,
    ctx: &BackwardContext,
) -> Result<FxHashMap<TensorId, Tensor>, MinitensorError> {
    Python::attach(|py| {
        let grad = Py::new(py, PyTensor::from_tensor(ctx.grad_output.clone()))
            .map_err(|e| from_python(name, "backward", e))?;
        let inputs = as_python_tensors(py, ctx.inputs.iter().cloned())
            .map_err(|e| from_python(name, "backward", e))?;
        let output = Py::new(py, PyTensor::from_tensor(ctx.output.clone()))
            .map_err(|e| from_python(name, "backward", e))?;

        let returned = backward
            .call1(py, (grad, inputs, output))
            .map_err(|e| from_python(name, "backward", e))?;
        let returned = returned.bind(py);

        // One input takes a bare tensor as well as a one-element sequence,
        // because writing `return grad * 2` is what anyone would write.
        let single = ctx.input_ids.len() == 1 && returned.extract::<PyTensor>().is_ok();
        let per_input: Vec<Option<Bound<PyAny>>> = if single {
            vec![Some(returned.clone())]
        } else {
            let items: Vec<Bound<PyAny>> = returned
                .try_iter()
                .and_then(|iter| iter.collect())
                .map_err(|e| from_python(name, "backward", e))?;
            if items.len() != ctx.input_ids.len() {
                return Err(MinitensorError::invalid_operation(format!(
                    "custom op '{name}': its backward returned {} gradients for {} inputs",
                    items.len(),
                    ctx.input_ids.len()
                )));
            }
            items
                .into_iter()
                .map(|item| if item.is_none() { None } else { Some(item) })
                .collect()
        };

        let mut gradients = FxHashMap::default();
        gradients.reserve(per_input.len());
        for ((value, id), input) in per_input.into_iter().zip(ctx.input_ids).zip(ctx.inputs) {
            // `None` means this input takes no gradient, which is a real
            // answer -- an index operand, say -- and not an omission.
            let Some(value) = value else { continue };
            let gradient = tensor_from_result(name, "backward", &value)?;
            if gradient.shape() != input.shape() {
                return Err(MinitensorError::invalid_operation(format!(
                    "custom op '{name}': its backward returned a gradient of shape {:?} \
                     for an input of shape {:?}",
                    gradient.shape().dims(),
                    input.shape().dims()
                )));
            }
            if gradient.dtype() != input.dtype() {
                return Err(MinitensorError::invalid_operation(format!(
                    "custom op '{name}': its backward returned a {} gradient for a {} input",
                    gradient.dtype(),
                    input.dtype()
                )));
            }
            gradients.insert(*id, gradient.detach());
        }
        Ok(gradients)
    })
}

/// Register example custom operations
#[pyfunction]
fn register_example_custom_ops() -> PyResult<()> {
    register_example_ops().map_err(_convert_error)?;
    Ok(())
}

/// Unregister a custom operation
#[pyfunction]
fn unregister_custom_op_py(name: &str) -> PyResult<()> {
    unregister_custom_op(name).map_err(_convert_error)?;
    Ok(())
}

/// Execute a custom operation
#[pyfunction]
fn execute_custom_op_py(name: &str, inputs: &Bound<PyList>) -> PyResult<PyTensor> {
    // Convert Python list to vector of tensor references
    let mut tensor_refs = Vec::new();
    let mut tensors = Vec::new();

    for item in inputs.iter() {
        let py_tensor: PyTensor = match item.extract::<PyTensor>() {
            Ok(t) => t,
            Err(_) => {
                let inner = item.getattr("_tensor")?;
                inner.extract::<PyTensor>()?
            }
        };
        tensors.push(py_tensor.tensor().clone());
    }

    // Create references
    for tensor in &tensors {
        tensor_refs.push(tensor);
    }

    // Execute the operation
    let result = execute_custom_op(name, &tensor_refs).map_err(_convert_error)?;

    Ok(PyTensor::from_tensor(result))
}

/// List all registered custom operations
#[pyfunction]
fn list_custom_ops_py() -> PyResult<Vec<String>> {
    list_custom_ops().map_err(_convert_error)
}

/// Check if a custom operation is registered
#[pyfunction]
fn is_custom_op_registered_py(name: &str) -> PyResult<bool> {
    is_custom_op_registered(name).map_err(_convert_error)
}

/// Initialize the custom operations module
pub fn init_custom_ops_module(_py: Python, parent_module: &Bound<PyModule>) -> PyResult<()> {
    // Add functions to parent module
    parent_module.add_function(wrap_pyfunction!(register_custom_op, parent_module)?)?;
    parent_module.add_function(wrap_pyfunction!(
        register_example_custom_ops,
        parent_module
    )?)?;
    parent_module.add_function(wrap_pyfunction!(unregister_custom_op_py, parent_module)?)?;
    parent_module.add_function(wrap_pyfunction!(execute_custom_op_py, parent_module)?)?;
    parent_module.add_function(wrap_pyfunction!(list_custom_ops_py, parent_module)?)?;
    parent_module.add_function(wrap_pyfunction!(is_custom_op_registered_py, parent_module)?)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::Python;

    #[test]
    fn test_custom_ops_bindings() {
        pyo3::Python::initialize();
        Python::attach(|_| {
            // Test that we can call the functions without panicking
            let result = register_example_custom_ops();
            assert!(result.is_ok());

            let ops = list_custom_ops_py();
            assert!(ops.is_ok());

            let is_registered = is_custom_op_registered_py("swish");
            assert!(is_registered.is_ok());
        });
    }
}

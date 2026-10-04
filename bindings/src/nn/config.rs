// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

//! The configuration getters the layers' constructors take and their classes
//! did not expose: every constructor keyword now reads back under its own
//! name, as the losses' already do.
//!
//! That is what lets a layer be rebuilt from what it reports -- `pickle` and
//! `copy.copy` do exactly that -- and it is what anyone inspecting a model
//! expects to be able to ask. A configuration that changes how the layer is
//! built reads back in the form the constructor accepts: `padding` is
//! `"same"` for a layer built that way, `approximate` is `"tanh"` or `"none"`.

use super::*;

/// `read(&layer)`, or the error for a module that is not that layer.
fn read<R>(
    module: &PyModule,
    class: &str,
    read: impl FnOnce(&ModuleType) -> Option<R>,
) -> PyResult<R> {
    read(module.inner.get()?).ok_or_else(|| PyTypeError::new_err(format!("Not a {class} layer")))
}

/// `"same"`, or the explicit padding.
fn padding_object(py: Python<'_>, same: bool, padding: (usize, usize)) -> PyResult<Py<PyAny>> {
    if same {
        return Ok(pyo3::types::PyString::new(py, "same").into_any().unbind());
    }
    Ok(padding.into_pyobject(py)?.into_any().unbind())
}

#[pymethods]
impl PyConv1d {
    /// Spacing between kernel taps
    #[getter]
    fn dilation(slf: PyRef<Self>) -> PyResult<usize> {
        read(slf.as_ref(), "Conv1d", |m| match m {
            ModuleType::Conv1d(l) => Some(l.dilation()),
            _ => None,
        })
    }

    /// Channel groups
    #[getter]
    fn groups(slf: PyRef<Self>) -> PyResult<usize> {
        read(slf.as_ref(), "Conv1d", |m| match m {
            ModuleType::Conv1d(l) => Some(l.groups()),
            _ => None,
        })
    }

    /// The bias tensor, or `None` for a layer built without one
    #[getter]
    fn bias(slf: PyRef<Self>) -> PyResult<Option<PyTensor>> {
        read(slf.as_ref(), "Conv1d", |m| match m {
            ModuleType::Conv1d(l) => Some(l.bias().map(|b| PyTensor::from_tensor(b.clone()))),
            _ => None,
        })
    }
}

#[pymethods]
impl PyConv2d {
    /// Step between output positions
    #[getter]
    fn stride(slf: PyRef<Self>) -> PyResult<(usize, usize)> {
        read(slf.as_ref(), "Conv2d", |m| match m {
            ModuleType::Conv2d(l) => Some(l.stride()),
            _ => None,
        })
    }

    /// The zeros on each side, or `"same"` for a layer built that way
    #[getter]
    fn padding(slf: PyRef<Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let (same, padding) = read(slf.as_ref(), "Conv2d", |m| match m {
            ModuleType::Conv2d(l) => Some((l.is_same_padding(), l.padding())),
            _ => None,
        })?;
        padding_object(py, same, padding)
    }

    /// Spacing between kernel taps
    #[getter]
    fn dilation(slf: PyRef<Self>) -> PyResult<(usize, usize)> {
        read(slf.as_ref(), "Conv2d", |m| match m {
            ModuleType::Conv2d(l) => Some(l.dilation()),
            _ => None,
        })
    }

    /// Channel groups
    #[getter]
    fn groups(slf: PyRef<Self>) -> PyResult<usize> {
        read(slf.as_ref(), "Conv2d", |m| match m {
            ModuleType::Conv2d(l) => Some(l.groups()),
            _ => None,
        })
    }

    /// The bias tensor, or `None` for a layer built without one
    #[getter]
    fn bias(slf: PyRef<Self>) -> PyResult<Option<PyTensor>> {
        read(slf.as_ref(), "Conv2d", |m| match m {
            ModuleType::Conv2d(l) => Some(l.bias().map(|b| PyTensor::from_tensor(b.clone()))),
            _ => None,
        })
    }
}

#[pymethods]
impl PyConvTranspose1d {
    /// Step between input positions in the output
    #[getter]
    fn stride(slf: PyRef<Self>) -> PyResult<usize> {
        read(slf.as_ref(), "ConvTranspose1d", |m| match m {
            ModuleType::ConvTranspose1d(l) => Some(l.stride()),
            _ => None,
        })
    }

    /// Padding taken off each side of the output
    #[getter]
    fn padding(slf: PyRef<Self>) -> PyResult<usize> {
        read(slf.as_ref(), "ConvTranspose1d", |m| match m {
            ModuleType::ConvTranspose1d(l) => Some(l.padding()),
            _ => None,
        })
    }

    /// Extra size added to one side of the output
    #[getter]
    fn output_padding(slf: PyRef<Self>) -> PyResult<usize> {
        read(slf.as_ref(), "ConvTranspose1d", |m| match m {
            ModuleType::ConvTranspose1d(l) => Some(l.output_padding()),
            _ => None,
        })
    }

    /// Spacing between kernel taps
    #[getter]
    fn dilation(slf: PyRef<Self>) -> PyResult<usize> {
        read(slf.as_ref(), "ConvTranspose1d", |m| match m {
            ModuleType::ConvTranspose1d(l) => Some(l.dilation()),
            _ => None,
        })
    }

    /// Channel groups
    #[getter]
    fn groups(slf: PyRef<Self>) -> PyResult<usize> {
        read(slf.as_ref(), "ConvTranspose1d", |m| match m {
            ModuleType::ConvTranspose1d(l) => Some(l.groups()),
            _ => None,
        })
    }

    /// The bias tensor, or `None` for a layer built without one
    #[getter]
    fn bias(slf: PyRef<Self>) -> PyResult<Option<PyTensor>> {
        read(slf.as_ref(), "ConvTranspose1d", |m| match m {
            ModuleType::ConvTranspose1d(l) => {
                Some(l.bias().map(|b| PyTensor::from_tensor(b.clone())))
            }
            _ => None,
        })
    }
}

#[pymethods]
impl PyConvTranspose2d {
    /// Step between input positions in the output
    #[getter]
    fn stride(slf: PyRef<Self>) -> PyResult<(usize, usize)> {
        read(slf.as_ref(), "ConvTranspose2d", |m| match m {
            ModuleType::ConvTranspose2d(l) => Some(l.stride()),
            _ => None,
        })
    }

    /// Padding taken off each side of the output
    #[getter]
    fn padding(slf: PyRef<Self>) -> PyResult<(usize, usize)> {
        read(slf.as_ref(), "ConvTranspose2d", |m| match m {
            ModuleType::ConvTranspose2d(l) => Some(l.padding()),
            _ => None,
        })
    }

    /// Spacing between kernel taps
    #[getter]
    fn dilation(slf: PyRef<Self>) -> PyResult<(usize, usize)> {
        read(slf.as_ref(), "ConvTranspose2d", |m| match m {
            ModuleType::ConvTranspose2d(l) => Some(l.dilation()),
            _ => None,
        })
    }

    /// Channel groups
    #[getter]
    fn groups(slf: PyRef<Self>) -> PyResult<usize> {
        read(slf.as_ref(), "ConvTranspose2d", |m| match m {
            ModuleType::ConvTranspose2d(l) => Some(l.groups()),
            _ => None,
        })
    }

    /// The bias tensor, or `None` for a layer built without one
    #[getter]
    fn bias(slf: PyRef<Self>) -> PyResult<Option<PyTensor>> {
        read(slf.as_ref(), "ConvTranspose2d", |m| match m {
            ModuleType::ConvTranspose2d(l) => {
                Some(l.bias().map(|b| PyTensor::from_tensor(b.clone())))
            }
            _ => None,
        })
    }
}

macro_rules! batch_norm_config {
    ($class:ident, $variant:ident, $name:literal) => {
        #[pymethods]
        impl $class {
            /// Added to the variance before its square root is taken
            #[getter]
            fn eps(slf: PyRef<Self>) -> PyResult<f64> {
                read(slf.as_ref(), $name, |m| match m {
                    ModuleType::$variant(l) => Some(l.eps()),
                    _ => None,
                })
            }

            /// Weight of each new batch in the running statistics
            #[getter]
            fn momentum(slf: PyRef<Self>) -> PyResult<f64> {
                read(slf.as_ref(), $name, |m| match m {
                    ModuleType::$variant(l) => Some(l.momentum()),
                    _ => None,
                })
            }

            /// Whether the layer learns a scale and a shift
            #[getter]
            fn affine(slf: PyRef<Self>) -> PyResult<bool> {
                read(slf.as_ref(), $name, |m| match m {
                    ModuleType::$variant(l) => Some(l.affine()),
                    _ => None,
                })
            }
        }
    };
}
batch_norm_config!(PyBatchNorm1d, BatchNorm1d, "BatchNorm1d");
batch_norm_config!(PyBatchNorm2d, BatchNorm2d, "BatchNorm2d");

#[pymethods]
impl PyGELU {
    /// `"tanh"` for the tanh approximation, `"none"` for the exact form
    #[getter]
    fn approximate(slf: PyRef<Self>) -> PyResult<&'static str> {
        read(slf.as_ref(), "GELU", |m| match m {
            ModuleType::Gelu(l) => Some(if l.is_approximate() { "tanh" } else { "none" }),
            _ => None,
        })
    }
}

#[pymethods]
impl PyLayerNorm {
    /// Whether the layer learns a scale and a shift
    #[getter]
    fn elementwise_affine(slf: PyRef<Self>) -> PyResult<bool> {
        read(slf.as_ref(), "LayerNorm", |m| match m {
            ModuleType::LayerNorm(l) => Some(l.elementwise_affine()),
            _ => None,
        })
    }
}

#[pymethods]
impl PyRMSNorm {
    /// Whether the layer learns a scale
    #[getter]
    fn elementwise_affine(slf: PyRef<Self>) -> PyResult<bool> {
        read(slf.as_ref(), "RMSNorm", |m| match m {
            ModuleType::RMSNorm(l) => Some(l.elementwise_affine()),
            _ => None,
        })
    }
}

#[pymethods]
impl PyMultiheadAttention {
    /// Whether the four projections carry additive biases
    #[getter]
    fn bias(slf: PyRef<Self>) -> PyResult<bool> {
        read(slf.as_ref(), "MultiheadAttention", |m| match m {
            ModuleType::MultiheadAttention(l) => Some(l.has_bias()),
            _ => None,
        })
    }
}

#[pymethods]
impl PyAvgPool2d {
    /// The zeros on each side
    #[getter]
    fn padding(slf: PyRef<Self>) -> PyResult<(usize, usize)> {
        read(slf.as_ref(), "AvgPool2d", |m| match m {
            ModuleType::AvgPool2d(l) => Some(l.padding()),
            _ => None,
        })
    }
}

#[pymethods]
impl PyUpsample {
    /// The output size, when it was given instead of a scale factor
    #[getter]
    fn size(slf: PyRef<Self>) -> PyResult<Option<Vec<usize>>> {
        read(slf.as_ref(), "Upsample", |m| match m {
            ModuleType::Upsample(l) => Some(l.size().map(<[usize]>::to_vec)),
            _ => None,
        })
    }

    /// The scale factor, when it was given instead of a size
    #[getter]
    fn scale_factor(slf: PyRef<Self>) -> PyResult<Option<Vec<f64>>> {
        read(slf.as_ref(), "Upsample", |m| match m {
            ModuleType::Upsample(l) => Some(l.scale_factor().map(<[f64]>::to_vec)),
            _ => None,
        })
    }

    /// `"nearest"`, or `"linear"` for the weighted average of neighbours
    #[getter]
    fn mode(slf: PyRef<Self>) -> PyResult<&'static str> {
        use engine::ops::interpolate::InterpolateMode;
        read(slf.as_ref(), "Upsample", |m| match m {
            ModuleType::Upsample(l) => Some(match l.mode() {
                InterpolateMode::Nearest => "nearest",
                InterpolateMode::Linear => "linear",
            }),
            _ => None,
        })
    }
}

#[pymethods]
impl PySequential {
    /// Add `module` at the end, unnamed: it is named by its position, as the
    /// modules a `Sequential` is built with are. `add_module` names one.
    fn append(slf: &Bound<'_, Self>, module: &Bound<'_, PyModule>) -> PyResult<()> {
        // Answered before either is borrowed, as `add_module` answers it.
        if module.is(slf) {
            return Err(PyValueError::new_err(
                "a Sequential cannot hold itself, or a module it is inside",
            ));
        }
        let mut this = slf.borrow_mut();
        let layer = PyModule::adopt_into(module, this.as_ref())?;
        match this.as_mut().inner.get_mut()? {
            ModuleType::Sequential(seq) => {
                seq.add_layer(layer);
                Ok(())
            }
            _ => Err(PyTypeError::new_err("Not a Sequential")),
        }
    }

    /// `(name, module)` for each module this holds, in order: the position
    /// as a string for a module added unnamed, the name `add_module` gave it
    /// otherwise -- the prefixes its `state_dict` keys carry.
    fn named_children(slf: PyRef<'_, Self>) -> PyResult<Vec<(String, Py<PyAny>)>> {
        let module = slf.as_ref();
        let names = module.child_names()?;
        let children = module.children(slf.py())?;
        Ok(names.into_iter().zip(children).collect())
    }
}

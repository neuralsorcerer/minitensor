// Copyright (c) Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::error::_convert_error;
use crate::tensor::PyTensor;
use engine::{
    serialization::{
        DeploymentModel, ModelMetadata, ModelSerializer, ModelVersion, SerializationFormat,
        SerializedModel, StateDict,
    },
    tensor::Shape,
};
use pyo3::prelude::*;
use std::collections::HashMap;

/// The format version a checkpoint was written with, and whether a given reader can load it.
#[pyclass(name = "ModelVersion", from_py_object)]
#[derive(Clone)]
pub struct PyModelVersion {
    inner: ModelVersion,
}

#[pymethods]
impl PyModelVersion {
    #[new]
    fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            inner: ModelVersion::new(major, minor, patch),
        }
    }

    #[staticmethod]
    fn current() -> Self {
        Self {
            inner: ModelVersion::current(),
        }
    }

    #[getter]
    fn major(&self) -> u32 {
        self.inner.major
    }

    #[getter]
    fn minor(&self) -> u32 {
        self.inner.minor
    }

    #[getter]
    fn patch(&self) -> u32 {
        self.inner.patch
    }

    #[getter]
    fn engine_version(&self) -> &str {
        &self.inner.engine_version
    }

    fn is_compatible(&self, other: &PyModelVersion) -> bool {
        self.inner.is_compatible(&other.inner)
    }

    fn is_newer(&self, other: &PyModelVersion) -> bool {
        self.inner.is_newer(&other.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "ModelVersion({}.{}.{})",
            self.inner.major, self.inner.minor, self.inner.patch
        )
    }

    fn __str__(&self) -> String {
        format!(
            "{}.{}.{}",
            self.inner.major, self.inner.minor, self.inner.patch
        )
    }
}

/// Name, description, architecture and shapes recorded alongside a saved model. Carried through save and load but not used by either.
#[pyclass(name = "ModelMetadata", from_py_object)]
#[derive(Clone)]
pub struct PyModelMetadata {
    inner: ModelMetadata,
}

#[pymethods]
impl PyModelMetadata {
    #[new]
    fn new(name: String, architecture: String) -> Self {
        Self {
            inner: ModelMetadata::new(name, architecture),
        }
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn description(&self) -> Option<&str> {
        self.inner.description.as_deref()
    }

    #[setter]
    fn set_description(&mut self, description: Option<String>) {
        self.inner.description = description;
    }

    #[getter]
    fn version(&self) -> PyModelVersion {
        PyModelVersion {
            inner: self.inner.version.clone(),
        }
    }

    #[getter]
    fn created_at(&self) -> &str {
        &self.inner.created_at
    }

    #[getter]
    fn platform(&self) -> &str {
        &self.inner.platform
    }

    #[getter]
    fn architecture(&self) -> &str {
        &self.inner.architecture
    }

    #[getter]
    fn input_shapes(&self) -> Vec<Vec<usize>> {
        self.inner
            .input_shapes
            .iter()
            .map(|s| s.dims().to_vec())
            .collect()
    }

    #[getter]
    fn output_shapes(&self) -> Vec<Vec<usize>> {
        self.inner
            .output_shapes
            .iter()
            .map(|s| s.dims().to_vec())
            .collect()
    }

    fn add_input_shape(&mut self, shape: Vec<usize>) {
        self.inner.input_shapes.push(Shape::new(shape));
    }

    fn add_output_shape(&mut self, shape: Vec<usize>) {
        self.inner.output_shapes.push(Shape::new(shape));
    }

    fn add_custom(&mut self, key: String, value: String) {
        self.inner.custom.insert(key, value);
    }

    fn get_custom(&self, key: &str) -> Option<&str> {
        self.inner.custom.get(key).map(|s| s.as_str())
    }

    fn __repr__(&self) -> String {
        format!(
            "ModelMetadata(name='{}', architecture='{}')",
            self.inner.name, self.inner.architecture
        )
    }
}

/// How a checkpoint is encoded: `json` (readable, large), `binary` (compact) or `msgpack` (compact, cross-language).
#[pyclass(name = "SerializationFormat", from_py_object)]
#[derive(Clone, Copy)]
pub struct PySerializationFormat {
    inner: SerializationFormat,
}

#[pymethods]
impl PySerializationFormat {
    #[new]
    fn new(format_str: &str) -> PyResult<Self> {
        let format = match format_str.to_lowercase().as_str() {
            "json" => SerializationFormat::Json,
            "binary" | "bin" => SerializationFormat::Binary,
            "messagepack" | "msgpack" => SerializationFormat::MessagePack,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown serialization format: {}",
                    format_str
                )));
            }
        };
        Ok(Self { inner: format })
    }

    #[staticmethod]
    fn json() -> Self {
        Self {
            inner: SerializationFormat::Json,
        }
    }

    #[staticmethod]
    fn binary() -> Self {
        Self {
            inner: SerializationFormat::Binary,
        }
    }

    #[staticmethod]
    fn messagepack() -> Self {
        Self {
            inner: SerializationFormat::MessagePack,
        }
    }

    fn extension(&self) -> &'static str {
        self.inner.extension()
    }

    fn __repr__(&self) -> String {
        match self.inner {
            SerializationFormat::Json => "SerializationFormat.JSON".to_string(),
            SerializationFormat::Binary => "SerializationFormat.BINARY".to_string(),
            SerializationFormat::MessagePack => "SerializationFormat.MESSAGEPACK".to_string(),
        }
    }
}

/// Reads and writes `SerializedModel` files in any of the `SerializationFormat`s.
#[pyclass(name = "ModelSerializer")]
pub struct PyModelSerializer;

#[pymethods]
impl PyModelSerializer {
    #[new]
    fn new() -> Self {
        Self
    }

    #[staticmethod]
    #[pyo3(signature = (model, path, format=None))]
    fn save(
        model: &PySerializedModel,
        path: &str,
        format: Option<&PySerializationFormat>,
    ) -> PyResult<()> {
        let format = format
            .map(|f| f.inner)
            .unwrap_or(SerializationFormat::Binary);
        ModelSerializer::save(&model.inner, path, format).map_err(_convert_error)?;
        Ok(())
    }

    #[staticmethod]
    #[pyo3(signature = (path, format=None))]
    fn load(path: &str, format: Option<&PySerializationFormat>) -> PyResult<PySerializedModel> {
        let format = format
            .map(|f| f.inner)
            .unwrap_or(SerializationFormat::Binary);
        let model = ModelSerializer::load(path, format).map_err(_convert_error)?;
        Ok(PySerializedModel { inner: model })
    }

    #[staticmethod]
    fn save_auto(model: &PySerializedModel, path: &str) -> PyResult<()> {
        ModelSerializer::save_auto(&model.inner, path).map_err(_convert_error)?;
        Ok(())
    }

    #[staticmethod]
    fn load_auto(path: &str) -> PyResult<PySerializedModel> {
        let model = ModelSerializer::load_auto(path).map_err(_convert_error)?;
        Ok(PySerializedModel { inner: model })
    }
}

/// A model's metadata and state dictionary together, as written to a file.
#[pyclass(name = "SerializedModel", from_py_object)]
#[derive(Clone)]
pub struct PySerializedModel {
    inner: SerializedModel,
}

#[pymethods]
impl PySerializedModel {
    #[new]
    fn new(metadata: &PyModelMetadata, state_dict: &PyStateDict) -> Self {
        Self {
            inner: SerializedModel::new(metadata.inner.clone(), state_dict.inner.clone()),
        }
    }

    #[getter]
    fn metadata(&self) -> PyModelMetadata {
        PyModelMetadata {
            inner: self.inner.metadata.clone(),
        }
    }

    #[getter]
    fn state_dict(&self) -> PyStateDict {
        PyStateDict {
            inner: self.inner.state_dict.clone(),
        }
    }

    #[getter]
    fn architecture_config(&self) -> Option<&str> {
        self.inner.architecture_config.as_deref()
    }

    #[setter]
    fn set_architecture_config(&mut self, config: Option<String>) {
        self.inner.architecture_config = config;
    }

    fn check_compatibility(&self) -> PyResult<()> {
        self.inner.check_compatibility().map_err(_convert_error)?;
        Ok(())
    }

    fn to_deployment_model(&self) -> PyDeploymentModel {
        PyDeploymentModel {
            inner: DeploymentModel::from_serialized_model(&self.inner),
        }
    }

    fn __repr__(&self) -> String {
        format!("SerializedModel(name='{}')", self.inner.metadata.name)
    }
}

/// A model's parameters and buffers by name, as returned by `module.state_dict()`. Buffers are the non-trainable tensors a layer needs to reproduce its behaviour, such as BatchNorm's running statistics.
#[pyclass(name = "StateDict", from_py_object)]
#[derive(Clone)]
pub struct PyStateDict {
    pub(crate) inner: StateDict,
}

#[pymethods]
impl PyStateDict {
    #[new]
    fn new() -> Self {
        Self {
            inner: StateDict::new(),
        }
    }

    fn parameter_names(&self) -> Vec<String> {
        self.inner.parameter_names().into_iter().cloned().collect()
    }

    fn buffer_names(&self) -> Vec<String> {
        self.inner.buffer_names().into_iter().cloned().collect()
    }

    /// The parameter stored under `name`.
    ///
    /// The engine has had `StateDict::load_parameter` since the start; without
    /// a binding for it a Python caller could list a checkpoint's contents and
    /// not read them, so `state_dict()` could only be handed straight back to
    /// `load_state_dict` and never inspected.
    fn get_parameter(&self, name: &str) -> PyResult<PyTensor> {
        let tensor = self
            .inner
            .load_parameter(name, None)
            .map_err(_convert_error)?;
        Ok(PyTensor::from_tensor(tensor))
    }

    /// The buffer stored under `name` -- a layer's non-trainable state, such as
    /// BatchNorm's running mean and variance.
    fn get_buffer(&self, name: &str) -> PyResult<PyTensor> {
        let tensor = self.inner.load_buffer(name, None).map_err(_convert_error)?;
        Ok(PyTensor::from_tensor(tensor))
    }

    /// Record a parameter under `name`, replacing any already there.
    fn add_parameter(&mut self, name: &str, tensor: &PyTensor) -> PyResult<()> {
        self.inner
            .add_parameter(name.to_string(), tensor.tensor())
            .map_err(_convert_error)
    }

    /// Record a buffer under `name`, replacing any already there.
    fn add_buffer(&mut self, name: &str, tensor: &PyTensor) -> PyResult<()> {
        self.inner
            .add_buffer(name.to_string(), tensor.tensor())
            .map_err(_convert_error)
    }

    /// Every parameter, by name.
    fn parameters(&self) -> PyResult<HashMap<String, PyTensor>> {
        self.inner
            .parameters
            .keys()
            .map(|name| Ok((name.clone(), self.get_parameter(name)?)))
            .collect()
    }

    /// Every buffer, by name.
    fn buffers(&self) -> PyResult<HashMap<String, PyTensor>> {
        self.inner
            .buffers
            .keys()
            .map(|name| Ok((name.clone(), self.get_buffer(name)?)))
            .collect()
    }

    /// `state["weight"]`, checking parameters before buffers.
    ///
    /// The two namespaces are separate in the file format but a caller reading
    /// a checkpoint rarely knows or cares which side a name is on; `__contains__`
    /// already spans both, so subscripting matches it.
    fn __getitem__(&self, name: &str) -> PyResult<PyTensor> {
        if self.inner.parameters.contains_key(name) {
            self.get_parameter(name)
        } else if self.inner.buffers.contains_key(name) {
            self.get_buffer(name)
        } else {
            Err(pyo3::exceptions::PyKeyError::new_err(format!(
                "no parameter or buffer named {name:?}; this state dict holds {:?}",
                self.inner
                    .parameter_names()
                    .into_iter()
                    .chain(self.inner.buffer_names())
                    .collect::<Vec<_>>()
            )))
        }
    }

    fn __len__(&self) -> usize {
        self.inner.parameters.len() + self.inner.buffers.len()
    }

    fn __contains__(&self, name: &str) -> bool {
        self.inner.parameters.contains_key(name) || self.inner.buffers.contains_key(name)
    }

    fn __repr__(&self) -> String {
        format!(
            "StateDict({} parameters, {} buffers)",
            self.inner.parameters.len(),
            self.inner.buffers.len()
        )
    }
}

// Internal helpers for other binding modules
impl PyStateDict {
    pub(crate) fn from_engine(inner: StateDict) -> Self {
        Self { inner }
    }
    pub(crate) fn inner_ref(s: &PyStateDict) -> &StateDict {
        &s.inner
    }
}

/// A saved model packaged with inference configuration, for loading without the training-time layer definitions.
#[pyclass(name = "DeploymentModel", from_py_object)]
#[derive(Clone)]
pub struct PyDeploymentModel {
    inner: DeploymentModel,
}

#[pymethods]
impl PyDeploymentModel {
    #[new]
    fn new(name: String, version: String) -> Self {
        Self {
            inner: DeploymentModel {
                name,
                version,
                input_shapes: Vec::new(),
                output_shapes: Vec::new(),
                state_dict: StateDict::new(),
                inference_config: HashMap::new(),
            },
        }
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn version(&self) -> &str {
        &self.inner.version
    }

    #[getter]
    fn input_shapes(&self) -> Vec<Vec<usize>> {
        self.inner
            .input_shapes
            .iter()
            .map(|s| s.dims().to_vec())
            .collect()
    }

    #[getter]
    fn output_shapes(&self) -> Vec<Vec<usize>> {
        self.inner
            .output_shapes
            .iter()
            .map(|s| s.dims().to_vec())
            .collect()
    }

    fn add_inference_config(&mut self, key: String, value: String) {
        self.inner.inference_config.insert(key, value);
    }

    fn get_inference_config(&self, key: &str) -> Option<&str> {
        self.inner.inference_config.get(key).map(|s| s.as_str())
    }

    fn save(&self, path: &str) -> PyResult<()> {
        self.inner.save(path).map_err(_convert_error)?;
        Ok(())
    }

    #[staticmethod]
    fn load(path: &str) -> PyResult<PyDeploymentModel> {
        let model = DeploymentModel::load(path).map_err(_convert_error)?;
        Ok(PyDeploymentModel { inner: model })
    }

    fn __repr__(&self) -> String {
        format!(
            "DeploymentModel(name='{}', version='{}')",
            self.inner.name, self.inner.version
        )
    }
}

/// Register serialization module with Python
#[pyfunction]
#[pyo3(signature = (model, path, format=None))]
fn save_model(model: &PySerializedModel, path: &str, format: Option<&str>) -> PyResult<()> {
    let format = if let Some(fmt_str) = format {
        Some(PySerializationFormat::new(fmt_str)?)
    } else {
        None
    };
    PyModelSerializer::save(model, path, format.as_ref())
}

/// Read a model saved by `save_model`, inferring the format from the file extension.
#[pyfunction]
#[pyo3(signature = (path, format=None))]
fn load_model(path: &str, format: Option<&str>) -> PyResult<PySerializedModel> {
    let format = if let Some(fmt_str) = format {
        Some(PySerializationFormat::new(fmt_str)?)
    } else {
        None
    };
    PyModelSerializer::load(path, format.as_ref())
}

pub fn register_serialization_module(py: Python, parent_module: &Bound<PyModule>) -> PyResult<()> {
    let serialization_module = PyModule::new(py, "serialization")?;
    serialization_module.setattr("__doc__", "Reading and writing model checkpoints.")?;

    // Add classes
    serialization_module.add_class::<PyModelVersion>()?;
    serialization_module.add_class::<PyModelMetadata>()?;
    serialization_module.add_class::<PySerializationFormat>()?;
    serialization_module.add_class::<PyModelSerializer>()?;
    serialization_module.add_class::<PySerializedModel>()?;
    serialization_module.add_class::<PyStateDict>()?;
    serialization_module.add_class::<PyDeploymentModel>()?;

    // Add convenience functions
    serialization_module.add_function(wrap_pyfunction!(save_model, &serialization_module)?)?;
    serialization_module.add_function(wrap_pyfunction!(load_model, &serialization_module)?)?;

    parent_module.add_submodule(&serialization_module)?;
    Ok(())
}

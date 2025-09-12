// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::error::_convert_error;
#[cfg(feature = "dynamic-loading")]
use engine::load_plugin as engine_load_plugin;
use engine::{
    PluginInfo, VersionInfo, get_plugin_info as engine_get_plugin_info,
    is_plugin_loaded as engine_is_plugin_loaded, list_plugins as engine_list_plugins,
    unload_plugin as engine_unload_plugin,
};
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::collections::HashMap;

/// Python wrapper for VersionInfo
#[pyclass(name = "VersionInfo")]
#[derive(Clone)]
pub struct PyVersionInfo {
    inner: VersionInfo,
}

#[pymethods]
impl PyVersionInfo {
    #[new]
    fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            inner: VersionInfo::new(major, minor, patch),
        }
    }

    #[staticmethod]
    fn parse(version_str: &str) -> PyResult<Self> {
        let version = VersionInfo::parse(version_str).map_err(_convert_error)?;
        Ok(Self { inner: version })
    }

    #[staticmethod]
    fn current() -> PyResult<Self> {
        let version = VersionInfo::current().map_err(_convert_error)?;
        Ok(Self { inner: version })
    }

    fn is_compatible_with(&self, other: &PyVersionInfo) -> bool {
        self.inner.is_compatible_with(&other.inner)
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

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "VersionInfo({}, {}, {})",
            self.inner.major, self.inner.minor, self.inner.patch
        )
    }
}

/// Python wrapper for PluginInfo
#[pyclass(name = "PluginInfo")]
#[derive(Clone)]
pub struct PyPluginInfo {
    inner: PluginInfo,
}

#[pymethods]
impl PyPluginInfo {
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn version(&self) -> PyVersionInfo {
        PyVersionInfo {
            inner: self.inner.version.clone(),
        }
    }

    #[getter]
    fn description(&self) -> &str {
        &self.inner.description
    }

    #[getter]
    fn author(&self) -> &str {
        &self.inner.author
    }

    #[getter]
    fn min_minitensor_version(&self) -> PyVersionInfo {
        PyVersionInfo {
            inner: self.inner.min_minitensor_version.clone(),
        }
    }

    #[getter]
    fn max_minitensor_version(&self) -> Option<PyVersionInfo> {
        self.inner
            .max_minitensor_version
            .as_ref()
            .map(|v| PyVersionInfo { inner: v.clone() })
    }

    fn __str__(&self) -> String {
        format!(
            "{} v{} by {}",
            self.inner.name, self.inner.version, self.inner.author
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "PluginInfo(name='{}', version='{}', author='{}')",
            self.inner.name, self.inner.version, self.inner.author
        )
    }
}

/// Python interface for creating custom plugins
#[pyclass(name = "CustomPlugin")]
pub struct PyCustomPlugin {
    info: PluginInfo,
    initialize_fn: Option<Py<PyAny>>,
    cleanup_fn: Option<Py<PyAny>>,
    custom_operations_fn: Option<Py<PyAny>>,
}

impl Clone for PyCustomPlugin {
    fn clone(&self) -> Self {
        Python::attach(|py| Self {
            info: self.info.clone(),
            initialize_fn: self.initialize_fn.as_ref().map(|f| f.clone_ref(py)),
            cleanup_fn: self.cleanup_fn.as_ref().map(|f| f.clone_ref(py)),
            custom_operations_fn: self.custom_operations_fn.as_ref().map(|f| f.clone_ref(py)),
        })
    }
}

#[pymethods]
impl PyCustomPlugin {
    #[new]
    #[pyo3(signature = (name, version, description, author, min_minitensor_version, max_minitensor_version=None))]
    fn new(
        name: String,
        version: &PyVersionInfo,
        description: String,
        author: String,
        min_minitensor_version: &PyVersionInfo,
        max_minitensor_version: Option<&PyVersionInfo>,
    ) -> Self {
        let info = PluginInfo {
            name,
            version: version.inner.clone(),
            description,
            author,
            min_minitensor_version: min_minitensor_version.inner.clone(),
            max_minitensor_version: max_minitensor_version.map(|v| v.inner.clone()),
        };

        Self {
            info,
            initialize_fn: None,
            cleanup_fn: None,
            custom_operations_fn: None,
        }
    }

    fn set_initialize_fn(&mut self, func: Py<PyAny>) {
        self.initialize_fn = Some(func);
    }

    fn set_cleanup_fn(&mut self, func: Py<PyAny>) {
        self.cleanup_fn = Some(func);
    }

    fn set_custom_operations_fn(&mut self, func: Py<PyAny>) {
        self.custom_operations_fn = Some(func);
    }

    #[getter]
    fn info(&self) -> PyPluginInfo {
        PyPluginInfo {
            inner: self.info.clone(),
        }
    }
}

// Note: We can't directly implement the Plugin trait for PyCustomPlugin
// because it requires Send + Sync, but PyObject is not Send + Sync.
// Instead, we'll create a wrapper that handles the Python calls safely.

/// Plugin system functions
#[pyfunction]
#[cfg_attr(not(feature = "dynamic-loading"), allow(unused_variables))]
fn load_plugin(path: &str) -> PyResult<()> {
    #[cfg(feature = "dynamic-loading")]
    {
        engine_load_plugin(path).map_err(_convert_error)?;
        Ok(())
    }
    #[cfg(not(feature = "dynamic-loading"))]
    {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Dynamic plugin loading is not available in this build",
        ))
    }
}

#[pyfunction]
fn unload_plugin(name: &str) -> PyResult<()> {
    engine_unload_plugin(name).map_err(_convert_error)?;
    Ok(())
}

#[pyfunction]
fn list_plugins() -> PyResult<Vec<PyPluginInfo>> {
    let plugins = engine_list_plugins().map_err(_convert_error)?;

    Ok(plugins
        .into_iter()
        .map(|info| PyPluginInfo { inner: info })
        .collect())
}

#[pyfunction]
fn get_plugin_info(name: &str) -> PyResult<PyPluginInfo> {
    let info = engine_get_plugin_info(name).map_err(_convert_error)?;

    Ok(PyPluginInfo { inner: info })
}

#[pyfunction]
fn is_plugin_loaded(name: &str) -> PyResult<bool> {
    engine_is_plugin_loaded(name).map_err(_convert_error)
}

/// Plugin registry for managing Python-based plugins
#[pyclass(name = "PluginRegistry")]
pub struct PyPluginRegistry {
    plugins: HashMap<String, PyCustomPlugin>,
}

#[pymethods]
impl PyPluginRegistry {
    #[new]
    fn new() -> Self {
        Self {
            plugins: HashMap::new(),
        }
    }

    fn register(&mut self, plugin: &PyCustomPlugin) -> PyResult<()> {
        let name = plugin.info.name.clone();

        // Check for duplicates
        if self.plugins.contains_key(&name) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Plugin '{}' is already registered",
                name
            )));
        }

        self.plugins.insert(name, plugin.clone());
        Ok(())
    }

    fn unregister(&mut self, name: &str) -> PyResult<()> {
        if self.plugins.remove(name).is_none() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Plugin '{}' is not registered",
                name
            )));
        }
        Ok(())
    }

    fn list_plugins(&self) -> Vec<PyPluginInfo> {
        self.plugins
            .values()
            .map(|plugin| PyPluginInfo {
                inner: plugin.info.clone(),
            })
            .collect()
    }

    fn get_plugin<'py>(&self, name: &str) -> PyResult<Py<PyCustomPlugin>> {
        Python::attach(|py| {
            if let Some(plugin) = self.plugins.get(name) {
                Py::new(
                    py,
                    PyCustomPlugin {
                        info: plugin.info.clone(),
                        initialize_fn: plugin.initialize_fn.as_ref().map(|f| f.clone_ref(py)),
                        cleanup_fn: plugin.cleanup_fn.as_ref().map(|f| f.clone_ref(py)),
                        custom_operations_fn: plugin
                            .custom_operations_fn
                            .as_ref()
                            .map(|f| f.clone_ref(py)),
                    },
                )
                .map_err(|e| e)
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                    "Plugin '{}' not found",
                    name
                )))
            }
        })
    }

    fn is_registered(&self, name: &str) -> bool {
        self.plugins.contains_key(name)
    }
}

/// Helper class for creating custom layers in Python
#[pyclass(name = "CustomLayer", subclass)]
pub struct PyCustomLayer {
    name: String,
    forward_fn: Option<Py<PyAny>>,
    parameters: HashMap<String, Py<PyAny>>,
}

#[pymethods]
impl PyCustomLayer {
    #[new]
    fn new(name: String) -> Self {
        Self {
            name,
            forward_fn: None,
            parameters: HashMap::new(),
        }
    }

    fn set_forward(&mut self, func: Py<PyAny>) {
        self.forward_fn = Some(func);
    }

    fn add_parameter(&mut self, name: String, tensor: Py<PyAny>) {
        self.parameters.insert(name, tensor);
    }

    fn get_parameter(&self, name: &str) -> PyResult<Py<PyAny>> {
        Python::attach(|py| {
            self.parameters
                .get(name)
                .map(|t| t.clone_ref(py))
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                        "Parameter '{}' not found",
                        name
                    ))
                })
        })
    }

    fn list_parameters(&self) -> Vec<String> {
        self.parameters.keys().cloned().collect()
    }

    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    fn forward(&self, py: Python, inputs: &Bound<PyList>) -> PyResult<Py<PyAny>> {
        if let Some(forward_fn) = &self.forward_fn {
            forward_fn.call1(py, (inputs,))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "Forward function not implemented",
            ))
        }
    }
}

/// Plugin development utilities
#[pyclass(name = "PluginBuilder")]
pub struct PyPluginBuilder {
    name: Option<String>,
    version: Option<VersionInfo>,
    description: Option<String>,
    author: Option<String>,
    min_minitensor_version: Option<VersionInfo>,
    max_minitensor_version: Option<VersionInfo>,
}

#[pymethods]
impl PyPluginBuilder {
    #[new]
    fn new() -> Self {
        Self {
            name: None,
            version: None,
            description: None,
            author: None,
            min_minitensor_version: None,
            max_minitensor_version: None,
        }
    }

    fn name(mut slf: PyRefMut<Self>, name: String) -> PyRefMut<Self> {
        slf.name = Some(name);
        slf
    }

    fn version<'a>(mut slf: PyRefMut<'a, Self>, version: &'a PyVersionInfo) -> PyRefMut<'a, Self> {
        slf.version = Some(version.inner.clone());
        slf
    }

    fn description(mut slf: PyRefMut<Self>, description: String) -> PyRefMut<Self> {
        slf.description = Some(description);
        slf
    }

    fn author(mut slf: PyRefMut<Self>, author: String) -> PyRefMut<Self> {
        slf.author = Some(author);
        slf
    }

    fn min_minitensor_version<'a>(
        mut slf: PyRefMut<'a, Self>,
        version: &'a PyVersionInfo,
    ) -> PyRefMut<'a, Self> {
        slf.min_minitensor_version = Some(version.inner.clone());
        slf
    }

    fn max_minitensor_version<'a>(
        mut slf: PyRefMut<'a, Self>,
        version: &'a PyVersionInfo,
    ) -> PyRefMut<'a, Self> {
        slf.max_minitensor_version = Some(version.inner.clone());
        slf
    }

    fn build(slf: PyRef<Self>) -> PyResult<PyCustomPlugin> {
        let name = slf
            .name
            .as_ref()
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("Plugin name is required")
            })?
            .clone();

        let version = slf
            .version
            .as_ref()
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("Plugin version is required")
            })?
            .clone();

        let description = slf
            .description
            .as_ref()
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("Plugin description is required")
            })?
            .clone();

        let author = slf
            .author
            .as_ref()
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("Plugin author is required")
            })?
            .clone();

        let min_minitensor_version = slf
            .min_minitensor_version
            .as_ref()
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Minimum minitensor version is required",
                )
            })?
            .clone();

        let info = PluginInfo {
            name,
            version,
            description,
            author,
            min_minitensor_version,
            max_minitensor_version: slf.max_minitensor_version.clone(),
        };

        Ok(PyCustomPlugin {
            info,
            initialize_fn: None,
            cleanup_fn: None,
            custom_operations_fn: None,
        })
    }
}

pub fn register_plugin_module(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyVersionInfo>()?;
    m.add_class::<PyPluginInfo>()?;
    m.add_class::<PyCustomPlugin>()?;
    m.add_class::<PyPluginRegistry>()?;
    m.add_class::<PyCustomLayer>()?;
    m.add_class::<PyPluginBuilder>()?;

    m.add_function(wrap_pyfunction!(load_plugin, m)?)?;
    m.add_function(wrap_pyfunction!(unload_plugin, m)?)?;
    m.add_function(wrap_pyfunction!(list_plugins, m)?)?;
    m.add_function(wrap_pyfunction!(get_plugin_info, m)?)?;
    m.add_function(wrap_pyfunction!(is_plugin_loaded, m)?)?;

    Ok(())
}

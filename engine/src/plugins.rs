// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    VERSION,
    custom_ops::{CustomOp, CustomOpRegistry},
    error::{MinitensorError, Result},
    nn::Layer,
};
use std::collections::HashMap;
#[cfg(feature = "dynamic-loading")]
use std::ffi::OsStr;
#[cfg(feature = "dynamic-loading")]
use std::path::Path;
use std::sync::{Arc, RwLock};

/// Version compatibility information for plugins
#[derive(Debug, Clone, PartialEq)]
pub struct VersionInfo {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl VersionInfo {
    /// Create a new version info
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Parse version from string (e.g., "1.2.3")
    pub fn parse(version_str: &str) -> Result<Self> {
        let parts: Vec<&str> = version_str.split('.').collect();
        if parts.len() != 3 {
            return Err(MinitensorError::invalid_argument(
                "Version must be in format 'major.minor.patch'",
            ));
        }

        let major = parts[0]
            .parse()
            .map_err(|_| MinitensorError::invalid_argument("Invalid major version number"))?;
        let minor = parts[1]
            .parse()
            .map_err(|_| MinitensorError::invalid_argument("Invalid minor version number"))?;
        let patch = parts[2]
            .parse()
            .map_err(|_| MinitensorError::invalid_argument("Invalid patch version number"))?;

        Ok(Self::new(major, minor, patch))
    }

    /// Check if this version is compatible with another version
    /// Compatible if major versions match and this version >= required version
    pub fn is_compatible_with(&self, required: &VersionInfo) -> bool {
        self.major == required.major
            && (self.minor > required.minor
                || (self.minor == required.minor && self.patch >= required.patch))
    }

    /// Get current minitensor version
    pub fn current() -> Result<Self> {
        Self::parse(VERSION)
    }
}

impl std::fmt::Display for VersionInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Plugin metadata and information
#[derive(Debug, Clone)]
pub struct PluginInfo {
    pub name: String,
    pub version: VersionInfo,
    pub description: String,
    pub author: String,
    pub min_minitensor_version: VersionInfo,
    pub max_minitensor_version: Option<VersionInfo>,
}

/// Trait that all plugins must implement
pub trait Plugin: Send + Sync {
    /// Get plugin information
    fn info(&self) -> &PluginInfo;

    /// Initialize the plugin (called when loaded)
    fn initialize(&self, registry: &CustomOpRegistry) -> Result<()>;

    /// Cleanup the plugin (called when unloaded)
    fn cleanup(&self, registry: &CustomOpRegistry) -> Result<()>;

    /// Get custom operations provided by this plugin
    fn custom_operations(&self) -> Vec<Arc<dyn CustomOp>>;

    /// Get custom layers provided by this plugin (optional)
    fn custom_layers(&self) -> Vec<Box<dyn Layer>> {
        Vec::new()
    }
}

/// Plugin loading and management system
pub struct PluginManager {
    loaded_plugins: RwLock<HashMap<String, Arc<dyn Plugin>>>,
    plugin_registry: Arc<CustomOpRegistry>,
}

impl PluginManager {
    /// Create a new plugin manager
    pub fn new(registry: Arc<CustomOpRegistry>) -> Self {
        Self {
            loaded_plugins: RwLock::new(HashMap::new()),
            plugin_registry: registry,
        }
    }

    /// Load a plugin from a shared library file
    #[cfg(feature = "dynamic-loading")]
    pub fn load_plugin<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        use libloading::{Library, Symbol};

        let path = path.as_ref();

        // Validate file extension
        if path.extension() != Some(OsStr::new("so"))
            && path.extension() != Some(OsStr::new("dll"))
            && path.extension() != Some(OsStr::new("dylib"))
        {
            return Err(MinitensorError::invalid_argument(
                "Plugin file must have .so, .dll, or .dylib extension",
            ));
        }

        // Load the library
        let lib = unsafe {
            Library::new(path).map_err(|e| {
                MinitensorError::plugin_error(format!("Failed to load plugin library: {}", e))
            })?
        };

        // Get the plugin creation function
        let create_plugin: Symbol<unsafe extern "C" fn() -> *mut dyn Plugin> = unsafe {
            lib.get(b"create_plugin").map_err(|e| {
                MinitensorError::plugin_error(format!(
                    "Plugin missing create_plugin function: {}",
                    e
                ))
            })?
        };

        // Create the plugin instance
        let plugin_ptr = unsafe { create_plugin() };
        if plugin_ptr.is_null() {
            return Err(MinitensorError::plugin_error(
                "Plugin creation returned null",
            ));
        }

        let plugin = unsafe { Arc::from_raw(plugin_ptr) };

        // Validate version compatibility
        let current_version = VersionInfo::current()?;
        let plugin_info = plugin.info();

        if !current_version.is_compatible_with(&plugin_info.min_minitensor_version) {
            return Err(MinitensorError::version_mismatch(format!(
                "Plugin '{}' requires minitensor >= {}, but current version is {}",
                plugin_info.name, plugin_info.min_minitensor_version, current_version
            )));
        }

        if let Some(max_version) = &plugin_info.max_minitensor_version {
            if !max_version.is_compatible_with(&current_version) {
                return Err(MinitensorError::version_mismatch(format!(
                    "Plugin '{}' requires minitensor <= {}, but current version is {}",
                    plugin_info.name, max_version, current_version
                )));
            }
        }

        // Check for name conflicts
        {
            let plugins = self.loaded_plugins.read().map_err(|_| {
                MinitensorError::internal_error("Failed to acquire plugins read lock")
            })?;

            if plugins.contains_key(&plugin_info.name) {
                return Err(MinitensorError::plugin_error(format!(
                    "Plugin '{}' is already loaded",
                    plugin_info.name
                )));
            }
        }

        // Initialize the plugin
        plugin.initialize(&self.plugin_registry)?;

        // Register custom operations
        for op in plugin.custom_operations() {
            self.plugin_registry.register(op)?;
        }

        // Store the plugin
        {
            let mut plugins = self.loaded_plugins.write().map_err(|_| {
                MinitensorError::internal_error("Failed to acquire plugins write lock")
            })?;

            plugins.insert(plugin_info.name.clone(), plugin);
        }

        Ok(())
    }

    /// Register a plugin directly (for statically linked plugins)
    pub fn register_plugin(&self, plugin: Arc<dyn Plugin>) -> Result<()> {
        let plugin_info = plugin.info();

        // Validate version compatibility
        let current_version = VersionInfo::current()?;

        if !current_version.is_compatible_with(&plugin_info.min_minitensor_version) {
            return Err(MinitensorError::version_mismatch(format!(
                "Plugin '{}' requires minitensor >= {}, but current version is {}",
                plugin_info.name, plugin_info.min_minitensor_version, current_version
            )));
        }

        if let Some(max_version) = &plugin_info.max_minitensor_version {
            if !max_version.is_compatible_with(&current_version) {
                return Err(MinitensorError::version_mismatch(format!(
                    "Plugin '{}' requires minitensor <= {}, but current version is {}",
                    plugin_info.name, max_version, current_version
                )));
            }
        }

        // Check for name conflicts
        {
            let plugins = self.loaded_plugins.read().map_err(|_| {
                MinitensorError::internal_error("Failed to acquire plugins read lock")
            })?;

            if plugins.contains_key(&plugin_info.name) {
                return Err(MinitensorError::plugin_error(format!(
                    "Plugin '{}' is already loaded",
                    plugin_info.name
                )));
            }
        }

        // Initialize the plugin
        plugin.initialize(&self.plugin_registry)?;

        // Register custom operations
        for op in plugin.custom_operations() {
            self.plugin_registry.register(op)?;
        }

        // Store the plugin
        {
            let mut plugins = self.loaded_plugins.write().map_err(|_| {
                MinitensorError::internal_error("Failed to acquire plugins write lock")
            })?;

            plugins.insert(plugin_info.name.clone(), plugin);
        }

        Ok(())
    }

    /// Unload a plugin by name
    pub fn unload_plugin(&self, name: &str) -> Result<()> {
        let plugin = {
            let mut plugins = self.loaded_plugins.write().map_err(|_| {
                MinitensorError::internal_error("Failed to acquire plugins write lock")
            })?;

            plugins.remove(name).ok_or_else(|| {
                MinitensorError::plugin_error(format!("Plugin '{}' is not loaded", name))
            })?
        };

        // Cleanup the plugin
        plugin.cleanup(&self.plugin_registry)?;

        // Unregister custom operations
        for op in plugin.custom_operations() {
            // Note: We don't fail if unregistration fails, as the operation might
            // have been unregistered manually
            let _ = self.plugin_registry.unregister(op.name());
        }

        Ok(())
    }

    /// List all loaded plugins
    pub fn list_plugins(&self) -> Result<Vec<PluginInfo>> {
        let plugins = self
            .loaded_plugins
            .read()
            .map_err(|_| MinitensorError::internal_error("Failed to acquire plugins read lock"))?;

        Ok(plugins.values().map(|p| p.info().clone()).collect())
    }

    /// Get information about a specific plugin
    pub fn get_plugin_info(&self, name: &str) -> Result<PluginInfo> {
        let plugins = self
            .loaded_plugins
            .read()
            .map_err(|_| MinitensorError::internal_error("Failed to acquire plugins read lock"))?;

        plugins.get(name).map(|p| p.info().clone()).ok_or_else(|| {
            MinitensorError::plugin_error(format!("Plugin '{}' is not loaded", name))
        })
    }

    /// Check if a plugin is loaded
    pub fn is_plugin_loaded(&self, name: &str) -> Result<bool> {
        let plugins = self
            .loaded_plugins
            .read()
            .map_err(|_| MinitensorError::internal_error("Failed to acquire plugins read lock"))?;

        Ok(plugins.contains_key(name))
    }
}

/// Global plugin manager instance
static GLOBAL_PLUGIN_MANAGER: std::sync::LazyLock<PluginManager> =
    std::sync::LazyLock::new(|| PluginManager::new(Arc::new(CustomOpRegistry::new())));

/// Load a plugin globally
#[cfg(feature = "dynamic-loading")]
pub fn load_plugin<P: AsRef<Path>>(path: P) -> Result<()> {
    GLOBAL_PLUGIN_MANAGER.load_plugin(path)
}

/// Register a plugin globally
pub fn register_plugin(plugin: Arc<dyn Plugin>) -> Result<()> {
    GLOBAL_PLUGIN_MANAGER.register_plugin(plugin)
}

/// Unload a plugin globally
pub fn unload_plugin(name: &str) -> Result<()> {
    GLOBAL_PLUGIN_MANAGER.unload_plugin(name)
}

/// List all loaded plugins globally
pub fn list_plugins() -> Result<Vec<PluginInfo>> {
    GLOBAL_PLUGIN_MANAGER.list_plugins()
}

/// Get plugin information globally
pub fn get_plugin_info(name: &str) -> Result<PluginInfo> {
    GLOBAL_PLUGIN_MANAGER.get_plugin_info(name)
}

/// Check if a plugin is loaded globally
pub fn is_plugin_loaded(name: &str) -> Result<bool> {
    GLOBAL_PLUGIN_MANAGER.is_plugin_loaded(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom_ops::CustomOpBuilder;

    // Mock plugin for testing
    struct TestPlugin {
        info: PluginInfo,
    }

    impl TestPlugin {
        fn new() -> Self {
            Self {
                info: PluginInfo {
                    name: "test_plugin".to_string(),
                    version: VersionInfo::new(1, 0, 0),
                    description: "A test plugin".to_string(),
                    author: "Test Author".to_string(),
                    min_minitensor_version: VersionInfo::new(0, 1, 0),
                    max_minitensor_version: None,
                },
            }
        }
    }

    impl Plugin for TestPlugin {
        fn info(&self) -> &PluginInfo {
            &self.info
        }

        fn initialize(&self, _registry: &CustomOpRegistry) -> Result<()> {
            Ok(())
        }

        fn cleanup(&self, _registry: &CustomOpRegistry) -> Result<()> {
            Ok(())
        }

        fn custom_operations(&self) -> Vec<Arc<dyn CustomOp>> {
            vec![
                CustomOpBuilder::new("test_plugin_op", 1)
                    .forward(|inputs| Ok(inputs[0].clone()))
                    .build()
                    .unwrap(),
            ]
        }
    }

    #[test]
    fn test_version_info() {
        let v1 = VersionInfo::new(1, 2, 3);
        let v2 = VersionInfo::parse("1.2.3").unwrap();
        assert_eq!(v1, v2);

        let v3 = VersionInfo::new(1, 2, 4);
        assert!(v3.is_compatible_with(&v1));
        assert!(!v1.is_compatible_with(&v3));

        let v4 = VersionInfo::new(2, 0, 0);
        assert!(!v4.is_compatible_with(&v1));
        assert!(!v1.is_compatible_with(&v4));
    }

    #[test]
    fn test_plugin_manager() {
        let registry = Arc::new(CustomOpRegistry::new());
        let manager = PluginManager::new(registry);

        let plugin = Arc::new(TestPlugin::new());
        manager.register_plugin(plugin).unwrap();

        assert!(manager.is_plugin_loaded("test_plugin").unwrap());

        let plugins = manager.list_plugins().unwrap();
        assert_eq!(plugins.len(), 1);
        assert_eq!(plugins[0].name, "test_plugin");

        let info = manager.get_plugin_info("test_plugin").unwrap();
        assert_eq!(info.name, "test_plugin");

        manager.unload_plugin("test_plugin").unwrap();
        assert!(!manager.is_plugin_loaded("test_plugin").unwrap());
    }

    #[test]
    fn test_version_compatibility() {
        let registry = Arc::new(CustomOpRegistry::new());
        let manager = PluginManager::new(registry);

        // Create a plugin that requires a future version
        let mut plugin = TestPlugin::new();
        plugin.info.min_minitensor_version = VersionInfo::new(999, 0, 0);

        let result = manager.register_plugin(Arc::new(plugin));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("version"));
    }
}

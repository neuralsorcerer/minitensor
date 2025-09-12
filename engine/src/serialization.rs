// Copyright (c) 2025 Soumyadip Sarkar.
// All rights reserved.
//
// This source code is licensed under the Apache-style license found in the
// LICENSE file in the root directory of this source tree.

use crate::{
    device::Device,
    error::{MinitensorError, Result},
    tensor::{DataType, Shape, Tensor},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
};

/// Version information for model compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelVersion {
    /// Major version (breaking changes)
    pub major: u32,
    /// Minor version (new features)
    pub minor: u32,
    /// Patch version (bug fixes)
    pub patch: u32,
    /// Engine version used to create the model
    pub engine_version: String,
}

impl ModelVersion {
    /// Create a new model version
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
            engine_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    /// Current model format version
    pub fn current() -> Self {
        Self::new(1, 0, 0)
    }

    /// Check if this version is compatible with another version
    pub fn is_compatible(&self, other: &ModelVersion) -> bool {
        // Major version must match for compatibility
        self.major == other.major
    }

    /// Check if this version is newer than another
    pub fn is_newer(&self, other: &ModelVersion) -> bool {
        if self.major != other.major {
            return self.major > other.major;
        }
        if self.minor != other.minor {
            return self.minor > other.minor;
        }
        self.patch > other.patch
    }
}

/// Metadata for serialized models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model name
    pub name: String,
    /// Model description
    pub description: Option<String>,
    /// Model version
    pub version: ModelVersion,
    /// Creation timestamp
    pub created_at: String,
    /// Platform information
    pub platform: String,
    /// Model architecture type
    pub architecture: String,
    /// Input shape information
    pub input_shapes: Vec<Shape>,
    /// Output shape information
    pub output_shapes: Vec<Shape>,
    /// Additional custom metadata
    pub custom: HashMap<String, String>,
}

impl ModelMetadata {
    /// Create new model metadata
    pub fn new(name: String, architecture: String) -> Self {
        Self {
            name,
            description: None,
            version: ModelVersion::current(),
            created_at: chrono::Utc::now().to_rfc3339(),
            platform: format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
            architecture,
            input_shapes: Vec::new(),
            output_shapes: Vec::new(),
            custom: HashMap::new(),
        }
    }

    /// Set description
    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }

    /// Add input shape
    pub fn add_input_shape(mut self, shape: Shape) -> Self {
        self.input_shapes.push(shape);
        self
    }

    /// Add output shape
    pub fn add_output_shape(mut self, shape: Shape) -> Self {
        self.output_shapes.push(shape);
        self
    }

    /// Add custom metadata
    pub fn add_custom(mut self, key: String, value: String) -> Self {
        self.custom.insert(key, value);
        self
    }
}

/// Serialized tensor data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedTensor {
    /// Tensor shape
    pub shape: Shape,
    /// Data type
    pub dtype: DataType,
    /// Device (for compatibility, actual device may differ on load)
    pub device: Device,
    /// Tensor data as bytes
    pub data: Vec<u8>,
    /// Whether tensor requires gradients
    pub requires_grad: bool,
}

impl SerializedTensor {
    /// Serialize a tensor
    pub fn from_tensor(tensor: &Tensor) -> Result<Self> {
        let data = match tensor.dtype() {
            DataType::Float32 => {
                let slice = tensor.data().as_f32_slice().ok_or_else(|| {
                    MinitensorError::serialization_error("Failed to get f32 data")
                })?;
                slice.iter().flat_map(|&x| x.to_le_bytes()).collect()
            }
            DataType::Float64 => {
                let slice = tensor.data().as_f64_slice().ok_or_else(|| {
                    MinitensorError::serialization_error("Failed to get f64 data")
                })?;
                slice.iter().flat_map(|&x| x.to_le_bytes()).collect()
            }
            DataType::Int32 => {
                let slice = tensor.data().as_i32_slice().ok_or_else(|| {
                    MinitensorError::serialization_error("Failed to get i32 data")
                })?;
                slice.iter().flat_map(|&x| x.to_le_bytes()).collect()
            }
            DataType::Int64 => {
                let slice = tensor.data().as_i64_slice().ok_or_else(|| {
                    MinitensorError::serialization_error("Failed to get i64 data")
                })?;
                slice.iter().flat_map(|&x| x.to_le_bytes()).collect()
            }
            DataType::Bool => {
                let slice = tensor.data().as_bool_slice().ok_or_else(|| {
                    MinitensorError::serialization_error("Failed to get bool data")
                })?;
                slice.iter().map(|&x| if x { 1u8 } else { 0u8 }).collect()
            }
        };

        Ok(Self {
            shape: tensor.shape().clone(),
            dtype: tensor.dtype(),
            device: tensor.device(),
            data,
            requires_grad: tensor.requires_grad(),
        })
    }

    /// Deserialize to tensor
    pub fn to_tensor(&self, target_device: Option<Device>) -> Result<Tensor> {
        let device = target_device.unwrap_or(self.device);
        let numel = self.shape.numel();

        // Create tensor data based on dtype
        let tensor_data = match self.dtype {
            DataType::Float32 => {
                if self.data.len() != numel * 4 {
                    return Err(MinitensorError::serialization_error(
                        "Invalid f32 data length",
                    ));
                }
                let mut values = Vec::with_capacity(numel);
                for chunk in self.data.chunks_exact(4) {
                    let bytes: [u8; 4] = chunk
                        .try_into()
                        .map_err(|_| MinitensorError::serialization_error("Invalid f32 bytes"))?;
                    values.push(f32::from_le_bytes(bytes));
                }
                crate::tensor::TensorData::from_vec_f32(values, device)
            }
            DataType::Float64 => {
                if self.data.len() != numel * 8 {
                    return Err(MinitensorError::serialization_error(
                        "Invalid f64 data length",
                    ));
                }
                let mut values = Vec::with_capacity(numel);
                for chunk in self.data.chunks_exact(8) {
                    let bytes: [u8; 8] = chunk
                        .try_into()
                        .map_err(|_| MinitensorError::serialization_error("Invalid f64 bytes"))?;
                    values.push(f64::from_le_bytes(bytes));
                }
                crate::tensor::TensorData::from_vec_f64(values, device)
            }
            DataType::Int32 => {
                if self.data.len() != numel * 4 {
                    return Err(MinitensorError::serialization_error(
                        "Invalid i32 data length",
                    ));
                }
                let mut values = Vec::with_capacity(numel);
                for chunk in self.data.chunks_exact(4) {
                    let bytes: [u8; 4] = chunk
                        .try_into()
                        .map_err(|_| MinitensorError::serialization_error("Invalid i32 bytes"))?;
                    values.push(i32::from_le_bytes(bytes));
                }
                crate::tensor::TensorData::from_vec_i32(values, device)
            }
            DataType::Int64 => {
                if self.data.len() != numel * 8 {
                    return Err(MinitensorError::serialization_error(
                        "Invalid i64 data length",
                    ));
                }
                let mut values = Vec::with_capacity(numel);
                for chunk in self.data.chunks_exact(8) {
                    let bytes: [u8; 8] = chunk
                        .try_into()
                        .map_err(|_| MinitensorError::serialization_error("Invalid i64 bytes"))?;
                    values.push(i64::from_le_bytes(bytes));
                }
                crate::tensor::TensorData::from_vec_i64(values, device)
            }
            DataType::Bool => {
                if self.data.len() != numel {
                    return Err(MinitensorError::serialization_error(
                        "Invalid bool data length",
                    ));
                }
                let values: Vec<bool> = self.data.iter().map(|&x| x != 0).collect();
                crate::tensor::TensorData::from_vec_bool(values, device)
            }
        };

        let tensor = Tensor::new(
            std::sync::Arc::new(tensor_data),
            self.shape.clone(),
            self.dtype,
            device,
            self.requires_grad,
        );

        Ok(tensor)
    }
}

/// Model state dictionary containing all parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateDict {
    /// Parameter tensors by name
    pub parameters: HashMap<String, SerializedTensor>,
    /// Buffer tensors by name (non-trainable parameters)
    pub buffers: HashMap<String, SerializedTensor>,
}

impl StateDict {
    /// Create empty state dict
    pub fn new() -> Self {
        Self {
            parameters: HashMap::new(),
            buffers: HashMap::new(),
        }
    }

    /// Add parameter tensor
    pub fn add_parameter(&mut self, name: String, tensor: &Tensor) -> Result<()> {
        let serialized = SerializedTensor::from_tensor(tensor)?;
        self.parameters.insert(name, serialized);
        Ok(())
    }

    /// Add buffer tensor
    pub fn add_buffer(&mut self, name: String, tensor: &Tensor) -> Result<()> {
        let serialized = SerializedTensor::from_tensor(tensor)?;
        self.buffers.insert(name, serialized);
        Ok(())
    }

    /// Get parameter names
    pub fn parameter_names(&self) -> Vec<&String> {
        self.parameters.keys().collect()
    }

    /// Get buffer names
    pub fn buffer_names(&self) -> Vec<&String> {
        self.buffers.keys().collect()
    }

    /// Load parameter tensor
    pub fn load_parameter(&self, name: &str, device: Option<Device>) -> Result<Tensor> {
        let serialized = self.parameters.get(name).ok_or_else(|| {
            MinitensorError::serialization_error(format!("Parameter '{}' not found", name))
        })?;
        serialized.to_tensor(device)
    }

    /// Load buffer tensor
    pub fn load_buffer(&self, name: &str, device: Option<Device>) -> Result<Tensor> {
        let serialized = self.buffers.get(name).ok_or_else(|| {
            MinitensorError::serialization_error(format!("Buffer '{}' not found", name))
        })?;
        serialized.to_tensor(device)
    }
}

impl Default for StateDict {
    fn default() -> Self {
        Self::new()
    }
}

/// Complete serialized model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedModel {
    /// Model metadata
    pub metadata: ModelMetadata,
    /// Model state dictionary
    pub state_dict: StateDict,
    /// Model architecture configuration (JSON string)
    pub architecture_config: Option<String>,
}

impl SerializedModel {
    /// Create new serialized model
    pub fn new(metadata: ModelMetadata, state_dict: StateDict) -> Self {
        Self {
            metadata,
            state_dict,
            architecture_config: None,
        }
    }

    /// Set architecture configuration
    pub fn with_architecture_config(mut self, config: String) -> Self {
        self.architecture_config = Some(config);
        self
    }

    /// Check version compatibility
    pub fn check_compatibility(&self) -> Result<()> {
        let current_version = ModelVersion::current();
        if !current_version.is_compatible(&self.metadata.version) {
            return Err(MinitensorError::serialization_error(format!(
                "Model version {}.{}.{} is not compatible with current version {}.{}.{}",
                self.metadata.version.major,
                self.metadata.version.minor,
                self.metadata.version.patch,
                current_version.major,
                current_version.minor,
                current_version.patch
            )));
        }
        Ok(())
    }
}

/// Model serialization format
#[derive(Debug, Clone, Copy)]
pub enum SerializationFormat {
    /// JSON format (human-readable, larger size)
    Json,
    /// Binary format (compact, faster)
    Binary,
    /// MessagePack format (compact, cross-language)
    MessagePack,
}

impl SerializationFormat {
    /// Get file extension for format
    pub fn extension(&self) -> &'static str {
        match self {
            SerializationFormat::Json => "json",
            SerializationFormat::Binary => "bin",
            SerializationFormat::MessagePack => "msgpack",
        }
    }
}

/// Model serializer for saving and loading models
pub struct ModelSerializer;

impl ModelSerializer {
    /// Save model to file
    pub fn save<P: AsRef<Path>>(
        model: &SerializedModel,
        path: P,
        format: SerializationFormat,
    ) -> Result<()> {
        let file = File::create(path).map_err(|e| {
            MinitensorError::serialization_error(format!("Failed to create file: {}", e))
        })?;
        let mut writer = BufWriter::new(file);

        match format {
            SerializationFormat::Json => {
                serde_json::to_writer_pretty(&mut writer, model).map_err(|e| {
                    MinitensorError::serialization_error(format!(
                        "JSON serialization failed: {}",
                        e
                    ))
                })?;
            }
            SerializationFormat::Binary => {
                bincode::serde::encode_into_std_write(
                    model,
                    &mut writer,
                    bincode::config::standard(),
                )
                .map_err(|e| {
                    MinitensorError::serialization_error(format!(
                        "Binary serialization failed: {}",
                        e
                    ))
                })?;
            }
            SerializationFormat::MessagePack => {
                rmp_serde::encode::write(&mut writer, model).map_err(|e| {
                    MinitensorError::serialization_error(format!(
                        "MessagePack serialization failed: {}",
                        e
                    ))
                })?;
            }
        }

        writer.flush().map_err(|e| {
            MinitensorError::serialization_error(format!("Failed to flush writer: {}", e))
        })?;
        Ok(())
    }

    /// Load model from file
    pub fn load<P: AsRef<Path>>(path: P, format: SerializationFormat) -> Result<SerializedModel> {
        let file = File::open(path).map_err(|e| {
            MinitensorError::serialization_error(format!("Failed to open file: {}", e))
        })?;
        let mut reader = BufReader::new(file);

        let model = match format {
            SerializationFormat::Json => serde_json::from_reader::<_, SerializedModel>(&mut reader)
                .map_err(|e| {
                    MinitensorError::serialization_error(format!(
                        "JSON deserialization failed: {}",
                        e
                    ))
                })?,
            SerializationFormat::Binary => {
                bincode::serde::decode_from_std_read(&mut reader, bincode::config::standard())
                    .map_err(|e| {
                        MinitensorError::serialization_error(format!(
                            "Binary deserialization failed: {}",
                            e
                        ))
                    })?
            }
            SerializationFormat::MessagePack => {
                rmp_serde::decode::from_read::<_, SerializedModel>(&mut reader).map_err(|e| {
                    MinitensorError::serialization_error(format!(
                        "MessagePack deserialization failed: {}",
                        e
                    ))
                })?
            }
        };

        Ok(model)
    }

    /// Save model with automatic format detection from extension
    pub fn save_auto<P: AsRef<Path>>(model: &SerializedModel, path: P) -> Result<()> {
        let path_ref = path.as_ref();
        let format = match path_ref.extension().and_then(|ext| ext.to_str()) {
            Some("json") => SerializationFormat::Json,
            Some("bin") => SerializationFormat::Binary,
            Some("msgpack") => SerializationFormat::MessagePack,
            _ => SerializationFormat::Binary, // Default to binary
        };

        Self::save(model, path, format)
    }

    /// Load model with automatic format detection from extension
    pub fn load_auto<P: AsRef<Path>>(path: P) -> Result<SerializedModel> {
        let path_ref = path.as_ref();
        let format = match path_ref.extension().and_then(|ext| ext.to_str()) {
            Some("json") => SerializationFormat::Json,
            Some("bin") => SerializationFormat::Binary,
            Some("msgpack") => SerializationFormat::MessagePack,
            _ => {
                // Try to detect format by reading first few bytes
                let mut file = File::open(path_ref).map_err(|e| {
                    MinitensorError::serialization_error(format!("Failed to open file: {}", e))
                })?;
                let mut buffer = [0u8; 4];
                file.read_exact(&mut buffer).map_err(|e| {
                    MinitensorError::serialization_error(format!(
                        "Failed to read file header: {}",
                        e
                    ))
                })?;

                if buffer[0] == b'{' {
                    SerializationFormat::Json
                } else if buffer[0] == 0x90 || buffer[0] == 0x80 {
                    SerializationFormat::MessagePack
                } else {
                    SerializationFormat::Binary
                }
            }
        };

        Self::load(path, format)
    }
}

/// Lightweight model format for production deployment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentModel {
    /// Minimal metadata
    pub name: String,
    pub version: String,
    pub input_shapes: Vec<Shape>,
    pub output_shapes: Vec<Shape>,
    /// Compressed state dictionary
    pub state_dict: StateDict,
    /// Inference configuration
    pub inference_config: HashMap<String, String>,
}

impl DeploymentModel {
    /// Create deployment model from full model
    pub fn from_serialized_model(model: &SerializedModel) -> Self {
        Self {
            name: model.metadata.name.clone(),
            version: format!(
                "{}.{}.{}",
                model.metadata.version.major,
                model.metadata.version.minor,
                model.metadata.version.patch
            ),
            input_shapes: model.metadata.input_shapes.clone(),
            output_shapes: model.metadata.output_shapes.clone(),
            state_dict: model.state_dict.clone(),
            inference_config: HashMap::new(),
        }
    }

    /// Add inference configuration
    pub fn add_inference_config(mut self, key: String, value: String) -> Self {
        self.inference_config.insert(key, value);
        self
    }

    /// Save deployment model (always uses binary format for efficiency)
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path).map_err(|e| {
            MinitensorError::serialization_error(format!("Failed to create file: {}", e))
        })?;
        let mut writer = BufWriter::new(file);

        bincode::serde::encode_into_std_write(self, &mut writer, bincode::config::standard())
            .map_err(|e| {
                MinitensorError::serialization_error(format!(
                    "Deployment model serialization failed: {}",
                    e
                ))
            })?;

        writer.flush().map_err(|e| {
            MinitensorError::serialization_error(format!("Failed to flush writer: {}", e))
        })?;
        Ok(())
    }

    /// Load deployment model
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path).map_err(|e| {
            MinitensorError::serialization_error(format!("Failed to open file: {}", e))
        })?;
        let mut reader = BufReader::new(file);

        bincode::serde::decode_from_std_read(&mut reader, bincode::config::standard()).map_err(
            |e| {
                MinitensorError::serialization_error(format!(
                    "Deployment model deserialization failed: {}",
                    e
                ))
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Shape;

    #[test]
    fn test_model_version() {
        let v1 = ModelVersion::new(1, 0, 0);
        let v2 = ModelVersion::new(1, 1, 0);
        let v3 = ModelVersion::new(2, 0, 0);

        assert!(v1.is_compatible(&v2));
        assert!(!v1.is_compatible(&v3));
        assert!(v2.is_newer(&v1));
        assert!(!v1.is_newer(&v2));
    }

    #[test]
    fn test_model_metadata() {
        let metadata = ModelMetadata::new("test_model".to_string(), "sequential".to_string())
            .with_description("Test model".to_string())
            .add_input_shape(Shape::new(vec![1, 28, 28]))
            .add_output_shape(Shape::new(vec![1, 10]))
            .add_custom("author".to_string(), "test".to_string());

        assert_eq!(metadata.name, "test_model");
        assert_eq!(metadata.description, Some("Test model".to_string()));
        assert_eq!(metadata.input_shapes.len(), 1);
        assert_eq!(metadata.output_shapes.len(), 1);
        assert_eq!(metadata.custom.get("author"), Some(&"test".to_string()));
    }

    #[test]
    fn test_serialization_format() {
        assert_eq!(SerializationFormat::Json.extension(), "json");
        assert_eq!(SerializationFormat::Binary.extension(), "bin");
        assert_eq!(SerializationFormat::MessagePack.extension(), "msgpack");
    }

    #[test]
    fn test_state_dict() {
        let mut state_dict = StateDict::new();

        // Create test tensor
        let shape = Shape::new(vec![2, 3]);
        let tensor = Tensor::zeros(shape, DataType::Float32, Device::cpu(), false);

        // Add parameter
        state_dict
            .add_parameter("weight".to_string(), &tensor)
            .unwrap();

        assert_eq!(state_dict.parameter_names().len(), 1);
        assert!(
            state_dict
                .parameter_names()
                .contains(&&"weight".to_string())
        );

        // Load parameter
        let loaded = state_dict.load_parameter("weight", None).unwrap();
        assert_eq!(loaded.shape(), tensor.shape());
        assert_eq!(loaded.dtype(), tensor.dtype());
    }
}

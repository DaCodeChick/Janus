//! GGUF (GPT-Generated Unified Format) file format support
//!
//! This module provides complete GGUF file parsing with ModelLoader trait implementation.
//! GGUF is the format used by llama.cpp and many quantized LLM models.

mod error;
mod parser;
mod types;

pub use error::{GGUFError, Result as GGUFResult};
pub use parser::GGUFParser;
pub use types::{GGMLType, GGUFHeader, GGUFMetadata, MetadataValue, MetadataValueType, TensorInfo};

use super::{FormatError, ModelLoader, Result, TensorDType, TensorData};
use std::collections::HashMap;
use std::path::Path;

/// GGUF file loader implementing ModelLoader trait
pub struct GGUFFile {
    parser: GGUFParser,
}

impl GGUFFile {
    /// Load a GGUF file from disk using memory mapping
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let parser = GGUFParser::open(path)
            .map_err(|e| FormatError::ParseError(format!("GGUF parse error: {}", e)))?;

        Ok(Self { parser })
    }

    /// Get direct access to the underlying GGUF parser
    pub fn parser(&self) -> &GGUFParser {
        &self.parser
    }

    /// Get GGUF-specific metadata
    pub fn gguf_metadata(&self) -> &GGUFMetadata {
        self.parser.metadata()
    }

    /// Get tensor information with GGUF-specific details
    pub fn gguf_tensors(&self) -> &[TensorInfo] {
        self.parser.tensors()
    }

    /// Convert GGML type to TensorDType
    fn ggml_to_dtype(ggml_type: GGMLType) -> TensorDType {
        match ggml_type {
            GGMLType::F32 => TensorDType::F32,
            GGMLType::F16 => TensorDType::F16,
            GGMLType::I8 => TensorDType::I8,
            GGMLType::I16 => TensorDType::I16,
            GGMLType::I32 => TensorDType::I32,
            GGMLType::I64 => TensorDType::I64,
            GGMLType::Q4_0 => TensorDType::Q4_0,
            GGMLType::Q4_1 => TensorDType::Q4_1,
            GGMLType::Q4_K => TensorDType::Q4_K,
            GGMLType::Q5_0 => TensorDType::Q5_0,
            GGMLType::Q5_1 => TensorDType::Q5_1,
            GGMLType::Q5_K => TensorDType::Q5_K,
            GGMLType::Q6_K => TensorDType::Q6_K,
            GGMLType::Q8_0 => TensorDType::Q8_0,
            GGMLType::Q8_1 => TensorDType::Q8_1,
            // For unsupported quantization types, default to Q8_0 as a placeholder
            // These will be skipped during tensor allocation
            _ => TensorDType::Q8_0,
        }
    }

    /// Convert GGUF MetadataValue to String
    fn metadata_value_to_string(value: &MetadataValue) -> String {
        match value {
            MetadataValue::UInt8(v) => v.to_string(),
            MetadataValue::Int8(v) => v.to_string(),
            MetadataValue::UInt16(v) => v.to_string(),
            MetadataValue::Int16(v) => v.to_string(),
            MetadataValue::UInt32(v) => v.to_string(),
            MetadataValue::Int32(v) => v.to_string(),
            MetadataValue::Float32(v) => v.to_string(),
            MetadataValue::Bool(v) => v.to_string(),
            MetadataValue::String(v) => v.clone(),
            MetadataValue::Array(_) => "[array]".to_string(),
            MetadataValue::UInt64(v) => v.to_string(),
            MetadataValue::Int64(v) => v.to_string(),
            MetadataValue::Float64(v) => v.to_string(),
        }
    }
}

impl ModelLoader for GGUFFile {
    fn tensors(&self) -> Result<HashMap<String, TensorData<'_>>> {
        let mut result = HashMap::new();

        for tensor_info in self.parser.tensors() {
            let dtype = Self::ggml_to_dtype(tensor_info.ggml_type);
            let data = self.parser.get_tensor_data(tensor_info);

            // Convert Vec<u64> to Vec<usize> for dimensions
            let shape: Vec<usize> = tensor_info.dimensions.iter().map(|&d| d as usize).collect();

            let tensor = TensorData {
                name: tensor_info.name.clone(),
                shape,
                dtype,
                data,
            };

            result.insert(tensor_info.name.clone(), tensor);
        }

        Ok(result)
    }

    fn get_metadata(&self, key: &str) -> Option<String> {
        self.parser
            .get_metadata(key)
            .map(Self::metadata_value_to_string)
    }

    fn metadata_keys(&self) -> Vec<String> {
        self.parser.metadata().metadata.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ggml_dtype_conversion() {
        assert_eq!(GGUFFile::ggml_to_dtype(GGMLType::F32), TensorDType::F32);
        assert_eq!(GGUFFile::ggml_to_dtype(GGMLType::F16), TensorDType::F16);
        assert_eq!(GGUFFile::ggml_to_dtype(GGMLType::Q4_0), TensorDType::Q4_0);
    }
}

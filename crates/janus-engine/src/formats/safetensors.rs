//! Safetensors file format parser
//!
//! Safetensors is a simple, safe format for storing tensors.
//! Format: [8-byte header length (u64 LE)][JSON header][tensor data]
//!
//! The JSON header contains metadata and data_offsets for each tensor.
//! Offsets are relative to the end of the JSON header (absolute = 8 + N + relative_offset).

use super::{FormatError, ModelLoader, Result, TensorDType, TensorData};
use byteorder::{LittleEndian, ReadBytesExt};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::Cursor;
use std::path::Path;

/// Safetensors tensor metadata from JSON header
#[derive(Debug, Deserialize, Serialize)]
struct TensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2], // [start, end] offsets relative to end of JSON
}

/// Safetensors JSON header structure
#[derive(Debug, Deserialize, Serialize)]
struct SafetensorsHeader {
    #[serde(flatten)]
    tensors: HashMap<String, TensorInfo>,
}

/// Safetensors file loader with zero-copy mmap
pub struct SafetensorsFile {
    mmap: Mmap,
    header_size: usize,
    tensors: HashMap<String, TensorInfo>,
}

impl SafetensorsFile {
    /// Load a safetensors file from disk using memory mapping
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Read the 8-byte header length
        if mmap.len() < 8 {
            return Err(FormatError::InvalidFormat(
                "File too small to contain header length".into(),
            ));
        }

        let mut cursor = Cursor::new(&mmap[0..8]);
        let header_len = cursor.read_u64::<LittleEndian>()? as usize;

        // Validate header size
        if mmap.len() < 8 + header_len {
            return Err(FormatError::InvalidFormat(
                "File too small to contain full header".into(),
            ));
        }

        // Parse JSON header
        let header_bytes = &mmap[8..8 + header_len];
        let header_str = std::str::from_utf8(header_bytes)
            .map_err(|e| FormatError::ParseError(format!("Invalid UTF-8 in header: {}", e)))?;

        let header: SafetensorsHeader = serde_json::from_str(header_str)
            .map_err(|e| FormatError::ParseError(format!("Invalid JSON header: {}", e)))?;

        tracing::debug!(
            "Loaded safetensors file with {} tensors, header size: {} bytes",
            header.tensors.len(),
            header_len
        );

        Ok(Self {
            mmap,
            header_size: 8 + header_len,
            tensors: header.tensors,
        })
    }

    /// Convert safetensors dtype string to TensorDType
    fn parse_dtype(dtype: &str) -> Result<TensorDType> {
        match dtype {
            "F32" => Ok(TensorDType::F32),
            "F16" => Ok(TensorDType::F16),
            "BF16" => Ok(TensorDType::BF16),
            "I8" => Ok(TensorDType::I8),
            "I16" => Ok(TensorDType::I16),
            "I32" => Ok(TensorDType::I32),
            "I64" => Ok(TensorDType::I64),
            "U8" => Ok(TensorDType::U8),
            "U16" => Ok(TensorDType::U16),
            "U32" => Ok(TensorDType::U32),
            "U64" => Ok(TensorDType::U64),
            _ => Err(FormatError::InvalidFormat(format!(
                "Unknown dtype: {}",
                dtype
            ))),
        }
    }

    /// Get the data slice for a tensor
    fn get_tensor_data(&self, info: &TensorInfo) -> Result<&[u8]> {
        let start = self.header_size + info.data_offsets[0];
        let end = self.header_size + info.data_offsets[1];

        if end > self.mmap.len() {
            return Err(FormatError::InvalidTensorData(
                "Tensor data offset exceeds file size".into(),
            ));
        }

        Ok(&self.mmap[start..end])
    }
}

impl ModelLoader for SafetensorsFile {
    fn tensors(&self) -> Result<HashMap<String, TensorData<'_>>> {
        let mut result = HashMap::new();

        for (name, info) in &self.tensors {
            let dtype = Self::parse_dtype(&info.dtype)?;
            let data = self.get_tensor_data(info)?;

            let tensor = TensorData {
                name: name.clone(),
                shape: info.shape.clone(),
                dtype,
                data,
            };

            // Validate data size
            let expected_size = tensor.expected_size();
            if data.len() != expected_size {
                tracing::warn!(
                    "Tensor '{}' data size mismatch: expected {} bytes, got {}",
                    name,
                    expected_size,
                    data.len()
                );
            }

            result.insert(name.clone(), tensor);
        }

        Ok(result)
    }

    fn get_metadata(&self, _key: &str) -> Option<String> {
        // Safetensors doesn't have a standard metadata system in the header
        // beyond tensor information
        None
    }

    fn metadata_keys(&self) -> Vec<String> {
        // No metadata keys in basic safetensors format
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_parsing() {
        assert!(matches!(
            SafetensorsFile::parse_dtype("F32"),
            Ok(TensorDType::F32)
        ));
        assert!(matches!(
            SafetensorsFile::parse_dtype("F16"),
            Ok(TensorDType::F16)
        ));
        assert!(matches!(
            SafetensorsFile::parse_dtype("I32"),
            Ok(TensorDType::I32)
        ));
        assert!(SafetensorsFile::parse_dtype("INVALID").is_err());
    }

    #[test]
    fn test_header_structure() {
        // Create a minimal safetensors header
        let mut header = HashMap::new();
        header.insert(
            "test_tensor".to_string(),
            TensorInfo {
                dtype: "F32".to_string(),
                shape: vec![2, 3],
                data_offsets: [0, 24], // 2*3*4 = 24 bytes
            },
        );

        let json = serde_json::to_string(&header).unwrap();
        assert!(json.contains("test_tensor"));
        assert!(json.contains("F32"));
    }
}

use super::error::{GGUFError, Result};
use super::types::*;
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use winnow::{
    binary::{le_f32, le_f64, le_u16, le_u32, le_u64, le_u8},
    error::ContextError,
    token::take,
    Parser,
};

/// Magic number for GGUF files: "GGUF" in ASCII
const GGUF_MAGIC: [u8; 4] = [0x47, 0x47, 0x55, 0x46]; // "GGUF"

/// Supported GGUF versions
const SUPPORTED_VERSION_MIN: u32 = 2;
const SUPPORTED_VERSION_MAX: u32 = 3;

/// Default alignment for tensor data
const DEFAULT_ALIGNMENT: u64 = 32;

/// GGUF file handle with memory-mapped data
pub struct GGUFFile {
    _mmap: Mmap,
    metadata: GGUFMetadata,
    #[allow(dead_code)]
    data_offset: usize,
}

impl GGUFFile {
    /// Open and parse a GGUF file
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        let (metadata, data_offset) = Self::parse_header(&mmap)?;

        Ok(Self {
            _mmap: mmap,
            metadata,
            data_offset,
        })
    }

    /// Get metadata
    pub fn metadata(&self) -> &GGUFMetadata {
        &self.metadata
    }

    /// Get tensor information
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.metadata.tensors
    }

    /// Get a specific metadata value by key
    pub fn get_metadata(&self, key: &str) -> Option<&MetadataValue> {
        self.metadata.metadata.get(key)
    }

    /// Parse GGUF header and metadata
    fn parse_header(data: &[u8]) -> Result<(GGUFMetadata, usize)> {
        let mut input = data;

        // Parse magic number
        let magic = parse_magic(&mut input)
            .map_err(|e| GGUFError::ParseError(format!("Failed to parse magic: {}", e)))?;

        if magic != GGUF_MAGIC {
            return Err(GGUFError::InvalidMagic(magic));
        }

        // Parse version
        let version: u32 =
            le_u32
                .parse_next(&mut input)
                .map_err(|e: winnow::error::ErrMode<ContextError>| {
                    GGUFError::ParseError(format!("Failed to parse version: {}", e))
                })?;

        if version < SUPPORTED_VERSION_MIN || version > SUPPORTED_VERSION_MAX {
            return Err(GGUFError::UnsupportedVersion(version));
        }

        // Parse tensor count
        let tensor_count =
            le_u64
                .parse_next(&mut input)
                .map_err(|e: winnow::error::ErrMode<ContextError>| {
                    GGUFError::ParseError(format!("Failed to parse tensor count: {}", e))
                })?;

        // Parse metadata KV count
        let metadata_kv_count =
            le_u64
                .parse_next(&mut input)
                .map_err(|e: winnow::error::ErrMode<ContextError>| {
                    GGUFError::ParseError(format!("Failed to parse metadata count: {}", e))
                })?;

        let header = GGUFHeader {
            version,
            tensor_count,
            metadata_kv_count,
        };

        // Parse metadata key-value pairs
        let metadata = parse_metadata_kvs(&mut input, metadata_kv_count as usize)
            .map_err(|e| GGUFError::ParseError(format!("Failed to parse metadata: {}", e)))?;

        // Get alignment from metadata (default to 32)
        let alignment =
            if let Some(MetadataValue::UInt32(align)) = metadata.get("general.alignment") {
                *align as u64
            } else {
                DEFAULT_ALIGNMENT
            };

        // Parse tensor info
        let tensors = parse_tensor_infos(&mut input, tensor_count as usize)
            .map_err(|e| GGUFError::ParseError(format!("Failed to parse tensor info: {}", e)))?;

        // Calculate data offset with alignment
        let header_size = data.len() - input.len();
        let data_offset = align_offset(header_size, alignment as usize);

        let gguf_metadata = GGUFMetadata {
            header,
            metadata,
            tensors,
            alignment,
        };

        Ok((gguf_metadata, data_offset))
    }
}

/// Parse magic number
fn parse_magic<'s>(
    input: &mut &'s [u8],
) -> std::result::Result<[u8; 4], winnow::error::ErrMode<ContextError>> {
    take(4usize).parse_next(input).map(|bytes: &[u8]| {
        let mut magic = [0u8; 4];
        magic.copy_from_slice(bytes);
        magic
    })
}

/// Parse a GGUF string (u64 length + UTF-8 bytes)
fn parse_string<'s>(
    input: &mut &'s [u8],
) -> std::result::Result<String, winnow::error::ErrMode<ContextError>> {
    let len = le_u64.parse_next(input)?;
    let bytes = take(len as usize).parse_next(input)?;
    Ok(String::from_utf8_lossy(bytes).into_owned())
}

/// Parse a metadata value
fn parse_metadata_value<'s>(
    input: &mut &'s [u8],
) -> std::result::Result<MetadataValue, winnow::error::ErrMode<ContextError>> {
    let value_type = le_u32.parse_next(input)?;

    let value_type = MetadataValueType::from_u32(value_type)
        .ok_or_else(|| winnow::error::ErrMode::Backtrack(ContextError::new()))?;

    match value_type {
        MetadataValueType::UInt8 => Ok(MetadataValue::UInt8(le_u8.parse_next(input)?)),
        MetadataValueType::Int8 => Ok(MetadataValue::Int8(le_u8.parse_next(input)? as i8)),
        MetadataValueType::UInt16 => Ok(MetadataValue::UInt16(le_u16.parse_next(input)?)),
        MetadataValueType::Int16 => Ok(MetadataValue::Int16(le_u16.parse_next(input)? as i16)),
        MetadataValueType::UInt32 => Ok(MetadataValue::UInt32(le_u32.parse_next(input)?)),
        MetadataValueType::Int32 => Ok(MetadataValue::Int32(le_u32.parse_next(input)? as i32)),
        MetadataValueType::Float32 => Ok(MetadataValue::Float32(le_f32.parse_next(input)?)),
        MetadataValueType::Bool => Ok(MetadataValue::Bool(le_u8.parse_next(input)? != 0)),
        MetadataValueType::String => Ok(MetadataValue::String(parse_string(input)?)),
        MetadataValueType::Array => {
            // Array: type + count + elements
            let elem_type = le_u32.parse_next(input)?;
            let count = le_u64.parse_next(input)?;

            let elem_type_enum = MetadataValueType::from_u32(elem_type)
                .ok_or_else(|| winnow::error::ErrMode::Backtrack(ContextError::new()))?;

            let mut array = Vec::with_capacity(count.min(1024) as usize);
            for _ in 0..count {
                array.push(parse_array_element(input, elem_type_enum)?);
            }

            Ok(MetadataValue::Array(array))
        }
        MetadataValueType::UInt64 => Ok(MetadataValue::UInt64(le_u64.parse_next(input)?)),
        MetadataValueType::Int64 => Ok(MetadataValue::Int64(le_u64.parse_next(input)? as i64)),
        MetadataValueType::Float64 => Ok(MetadataValue::Float64(le_f64.parse_next(input)?)),
    }
}

/// Parse array element (without type prefix)
fn parse_array_element<'s>(
    input: &mut &'s [u8],
    elem_type: MetadataValueType,
) -> std::result::Result<MetadataValue, winnow::error::ErrMode<ContextError>> {
    match elem_type {
        MetadataValueType::UInt8 => Ok(MetadataValue::UInt8(le_u8.parse_next(input)?)),
        MetadataValueType::Int8 => Ok(MetadataValue::Int8(le_u8.parse_next(input)? as i8)),
        MetadataValueType::UInt16 => Ok(MetadataValue::UInt16(le_u16.parse_next(input)?)),
        MetadataValueType::Int16 => Ok(MetadataValue::Int16(le_u16.parse_next(input)? as i16)),
        MetadataValueType::UInt32 => Ok(MetadataValue::UInt32(le_u32.parse_next(input)?)),
        MetadataValueType::Int32 => Ok(MetadataValue::Int32(le_u32.parse_next(input)? as i32)),
        MetadataValueType::Float32 => Ok(MetadataValue::Float32(le_f32.parse_next(input)?)),
        MetadataValueType::Bool => Ok(MetadataValue::Bool(le_u8.parse_next(input)? != 0)),
        MetadataValueType::String => Ok(MetadataValue::String(parse_string(input)?)),
        MetadataValueType::UInt64 => Ok(MetadataValue::UInt64(le_u64.parse_next(input)?)),
        MetadataValueType::Int64 => Ok(MetadataValue::Int64(le_u64.parse_next(input)? as i64)),
        MetadataValueType::Float64 => Ok(MetadataValue::Float64(le_f64.parse_next(input)?)),
        MetadataValueType::Array => Err(winnow::error::ErrMode::Backtrack(ContextError::new())),
    }
}

/// Parse metadata key-value pairs
fn parse_metadata_kvs<'s>(
    input: &mut &'s [u8],
    count: usize,
) -> std::result::Result<HashMap<String, MetadataValue>, winnow::error::ErrMode<ContextError>> {
    let mut metadata = HashMap::with_capacity(count);

    for _ in 0..count {
        let key = parse_string(input)?;
        let value = parse_metadata_value(input)?;
        metadata.insert(key, value);
    }

    Ok(metadata)
}

/// Parse tensor information entries
fn parse_tensor_infos<'s>(
    input: &mut &'s [u8],
    count: usize,
) -> std::result::Result<Vec<TensorInfo>, winnow::error::ErrMode<ContextError>> {
    let mut tensors = Vec::with_capacity(count);

    for _ in 0..count {
        let name = parse_string(input)?;
        let n_dimensions = le_u32.parse_next(input)?;

        // Parse dimensions
        let mut dimensions = Vec::with_capacity(n_dimensions as usize);
        for _ in 0..n_dimensions {
            dimensions.push(le_u64.parse_next(input)?);
        }

        let ggml_type_raw = le_u32.parse_next(input)?;
        let offset = le_u64.parse_next(input)?;

        let ggml_type = GGMLType::from_u32(ggml_type_raw)
            .ok_or_else(|| winnow::error::ErrMode::Backtrack(ContextError::new()))?;

        tensors.push(TensorInfo {
            name,
            n_dimensions,
            dimensions,
            ggml_type,
            offset,
        });
    }

    Ok(tensors)
}

/// Align offset to specified alignment
fn align_offset(offset: usize, alignment: usize) -> usize {
    ((offset + alignment - 1) / alignment) * alignment
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_magic_number() {
        assert_eq!(GGUF_MAGIC, [0x47, 0x47, 0x55, 0x46]);
    }

    #[test]
    fn test_parse_magic() {
        let data = b"GGUF\x02\x00\x00\x00";
        let mut input = &data[..];
        let magic = parse_magic(&mut input).unwrap();
        assert_eq!(magic, GGUF_MAGIC);
        assert_eq!(input.len(), 4);
    }

    #[test]
    fn test_align_offset() {
        assert_eq!(align_offset(0, 32), 0);
        assert_eq!(align_offset(1, 32), 32);
        assert_eq!(align_offset(32, 32), 32);
        assert_eq!(align_offset(33, 32), 64);
    }
}

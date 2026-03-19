use super::error::{GGUFError, Result};
use super::types::*;
use byteorder::{LittleEndian, ReadBytesExt};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Cursor, Read};
use std::path::Path;

/// Magic number for GGUF files: "GGUF" in ASCII
const GGUF_MAGIC: [u8; 4] = [0x47, 0x47, 0x55, 0x46]; // "GGUF"

/// Supported GGUF versions
const SUPPORTED_VERSION_MIN: u32 = 2;
const SUPPORTED_VERSION_MAX: u32 = 3;

/// Default alignment for tensor data
const DEFAULT_ALIGNMENT: u64 = 32;

/// GGUF file handle with memory-mapped data
pub struct GGUFFile {
    mmap: Mmap,
    metadata: GGUFMetadata,
    data_offset: usize,
}

impl GGUFFile {
    /// Open and parse a GGUF file
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        let (metadata, data_offset) = Self::parse_header(&mmap)?;

        Ok(Self {
            mmap,
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

    /// Get the data offset where tensor data starts
    pub fn data_offset(&self) -> usize {
        self.data_offset
    }

    /// Get a slice of tensor data from the mmap
    pub fn get_tensor_data(&self, tensor: &TensorInfo) -> &[u8] {
        let start = self.data_offset + tensor.offset as usize;
        let end = start + tensor.size_bytes() as usize;
        &self.mmap[start..end]
    }

    /// Get the entire mmap data (for advanced use)
    pub fn mmap_data(&self) -> &[u8] {
        &self.mmap
    }

    /// Parse GGUF header and metadata from byte slice
    fn parse_header(data: &[u8]) -> Result<(GGUFMetadata, usize)> {
        let mut reader = Cursor::new(data);

        // Parse and verify magic number
        let mut magic = [0u8; 4];
        reader
            .read_exact(&mut magic)
            .map_err(|_| GGUFError::IncompleteData("Failed to read magic number".into()))?;

        if magic != GGUF_MAGIC {
            return Err(GGUFError::InvalidMagic(magic));
        }

        // Parse version
        let version = reader
            .read_u32::<LittleEndian>()
            .map_err(|_| GGUFError::IncompleteData("Failed to read version".into()))?;

        if version < SUPPORTED_VERSION_MIN || version > SUPPORTED_VERSION_MAX {
            return Err(GGUFError::UnsupportedVersion(version));
        }

        // Parse tensor count
        let tensor_count = reader
            .read_u64::<LittleEndian>()
            .map_err(|_| GGUFError::IncompleteData("Failed to read tensor count".into()))?;

        // Parse metadata KV count
        let metadata_kv_count = reader
            .read_u64::<LittleEndian>()
            .map_err(|_| GGUFError::IncompleteData("Failed to read metadata count".into()))?;

        let header = GGUFHeader {
            version,
            tensor_count,
            metadata_kv_count,
        };

        // Parse metadata key-value pairs
        let metadata = parse_metadata_kvs(&mut reader, metadata_kv_count as usize)?;

        // Get alignment from metadata (default to 32)
        let alignment =
            if let Some(MetadataValue::UInt32(align)) = metadata.get("general.alignment") {
                *align as u64
            } else {
                DEFAULT_ALIGNMENT
            };

        // Parse tensor info
        let tensors = parse_tensor_infos(&mut reader, tensor_count as usize)?;

        // Calculate data offset with alignment
        let header_size = reader.position() as usize;
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

/// Parse a GGUF string (u64 length + UTF-8 bytes)
fn parse_string<R: Read>(reader: &mut R) -> Result<String> {
    let len = reader
        .read_u64::<LittleEndian>()
        .map_err(|_| GGUFError::IncompleteData("Failed to read string length".into()))?;

    let mut bytes = vec![0u8; len as usize];
    reader
        .read_exact(&mut bytes)
        .map_err(|_| GGUFError::IncompleteData("Failed to read string data".into()))?;

    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// Parse a metadata value
fn parse_metadata_value<R: Read>(reader: &mut R) -> Result<MetadataValue> {
    let value_type = reader
        .read_u32::<LittleEndian>()
        .map_err(|_| GGUFError::IncompleteData("Failed to read metadata type".into()))?;

    let value_type = MetadataValueType::try_from(value_type)?;

    match value_type {
        MetadataValueType::UInt8 => {
            Ok(MetadataValue::UInt8(reader.read_u8().map_err(|_| {
                GGUFError::IncompleteData("Failed to read UInt8".into())
            })?))
        }
        MetadataValueType::Int8 => {
            Ok(MetadataValue::Int8(reader.read_i8().map_err(|_| {
                GGUFError::IncompleteData("Failed to read Int8".into())
            })?))
        }
        MetadataValueType::UInt16 => Ok(MetadataValue::UInt16(
            reader
                .read_u16::<LittleEndian>()
                .map_err(|_| GGUFError::IncompleteData("Failed to read UInt16".into()))?,
        )),
        MetadataValueType::Int16 => Ok(MetadataValue::Int16(
            reader
                .read_i16::<LittleEndian>()
                .map_err(|_| GGUFError::IncompleteData("Failed to read Int16".into()))?,
        )),
        MetadataValueType::UInt32 => Ok(MetadataValue::UInt32(
            reader
                .read_u32::<LittleEndian>()
                .map_err(|_| GGUFError::IncompleteData("Failed to read UInt32".into()))?,
        )),
        MetadataValueType::Int32 => Ok(MetadataValue::Int32(
            reader
                .read_i32::<LittleEndian>()
                .map_err(|_| GGUFError::IncompleteData("Failed to read Int32".into()))?,
        )),
        MetadataValueType::Float32 => Ok(MetadataValue::Float32(
            reader
                .read_f32::<LittleEndian>()
                .map_err(|_| GGUFError::IncompleteData("Failed to read Float32".into()))?,
        )),
        MetadataValueType::Bool => Ok(MetadataValue::Bool(
            reader
                .read_u8()
                .map_err(|_| GGUFError::IncompleteData("Failed to read Bool".into()))?
                != 0,
        )),
        MetadataValueType::String => Ok(MetadataValue::String(parse_string(reader)?)),
        MetadataValueType::Array => {
            // Array: type + count + elements
            let elem_type = reader.read_u32::<LittleEndian>().map_err(|_| {
                GGUFError::IncompleteData("Failed to read array element type".into())
            })?;
            let count = reader
                .read_u64::<LittleEndian>()
                .map_err(|_| GGUFError::IncompleteData("Failed to read array count".into()))?;

            let elem_type_enum = MetadataValueType::try_from(elem_type)?;

            let mut array = Vec::with_capacity(count.min(1024) as usize);
            for _ in 0..count {
                array.push(parse_array_element(reader, elem_type_enum)?);
            }

            Ok(MetadataValue::Array(array))
        }
        MetadataValueType::UInt64 => Ok(MetadataValue::UInt64(
            reader
                .read_u64::<LittleEndian>()
                .map_err(|_| GGUFError::IncompleteData("Failed to read UInt64".into()))?,
        )),
        MetadataValueType::Int64 => Ok(MetadataValue::Int64(
            reader
                .read_i64::<LittleEndian>()
                .map_err(|_| GGUFError::IncompleteData("Failed to read Int64".into()))?,
        )),
        MetadataValueType::Float64 => Ok(MetadataValue::Float64(
            reader
                .read_f64::<LittleEndian>()
                .map_err(|_| GGUFError::IncompleteData("Failed to read Float64".into()))?,
        )),
    }
}

/// Parse array element (without type prefix)
fn parse_array_element<R: Read>(
    reader: &mut R,
    elem_type: MetadataValueType,
) -> Result<MetadataValue> {
    match elem_type {
        MetadataValueType::UInt8 => {
            Ok(MetadataValue::UInt8(reader.read_u8().map_err(|_| {
                GGUFError::IncompleteData("Failed to read array UInt8".into())
            })?))
        }
        MetadataValueType::Int8 => {
            Ok(MetadataValue::Int8(reader.read_i8().map_err(|_| {
                GGUFError::IncompleteData("Failed to read array Int8".into())
            })?))
        }
        MetadataValueType::UInt16 => Ok(MetadataValue::UInt16(
            reader
                .read_u16::<LittleEndian>()
                .map_err(|_| GGUFError::IncompleteData("Failed to read array UInt16".into()))?,
        )),
        MetadataValueType::Int16 => Ok(MetadataValue::Int16(
            reader
                .read_i16::<LittleEndian>()
                .map_err(|_| GGUFError::IncompleteData("Failed to read array Int16".into()))?,
        )),
        MetadataValueType::UInt32 => Ok(MetadataValue::UInt32(
            reader
                .read_u32::<LittleEndian>()
                .map_err(|_| GGUFError::IncompleteData("Failed to read array UInt32".into()))?,
        )),
        MetadataValueType::Int32 => Ok(MetadataValue::Int32(
            reader
                .read_i32::<LittleEndian>()
                .map_err(|_| GGUFError::IncompleteData("Failed to read array Int32".into()))?,
        )),
        MetadataValueType::Float32 => Ok(MetadataValue::Float32(
            reader
                .read_f32::<LittleEndian>()
                .map_err(|_| GGUFError::IncompleteData("Failed to read array Float32".into()))?,
        )),
        MetadataValueType::Bool => Ok(MetadataValue::Bool(
            reader
                .read_u8()
                .map_err(|_| GGUFError::IncompleteData("Failed to read array Bool".into()))?
                != 0,
        )),
        MetadataValueType::String => Ok(MetadataValue::String(parse_string(reader)?)),
        MetadataValueType::UInt64 => Ok(MetadataValue::UInt64(
            reader
                .read_u64::<LittleEndian>()
                .map_err(|_| GGUFError::IncompleteData("Failed to read array UInt64".into()))?,
        )),
        MetadataValueType::Int64 => Ok(MetadataValue::Int64(
            reader
                .read_i64::<LittleEndian>()
                .map_err(|_| GGUFError::IncompleteData("Failed to read array Int64".into()))?,
        )),
        MetadataValueType::Float64 => Ok(MetadataValue::Float64(
            reader
                .read_f64::<LittleEndian>()
                .map_err(|_| GGUFError::IncompleteData("Failed to read array Float64".into()))?,
        )),
        MetadataValueType::Array => {
            Err(GGUFError::ParseError("Nested arrays not supported".into()))
        }
    }
}

/// Parse metadata key-value pairs
fn parse_metadata_kvs<R: Read>(
    reader: &mut R,
    count: usize,
) -> Result<HashMap<String, MetadataValue>> {
    let mut metadata = HashMap::with_capacity(count);

    for _ in 0..count {
        let key = parse_string(reader)?;
        let value = parse_metadata_value(reader)?;
        metadata.insert(key, value);
    }

    Ok(metadata)
}

/// Parse tensor information entries
fn parse_tensor_infos<R: Read>(reader: &mut R, count: usize) -> Result<Vec<TensorInfo>> {
    let mut tensors = Vec::with_capacity(count);

    for _ in 0..count {
        let name = parse_string(reader)?;
        let n_dimensions = reader.read_u32::<LittleEndian>().map_err(|_| {
            GGUFError::IncompleteData("Failed to read tensor dimension count".into())
        })?;

        // Parse dimensions
        let mut dimensions = Vec::with_capacity(n_dimensions as usize);
        for _ in 0..n_dimensions {
            dimensions.push(reader.read_u64::<LittleEndian>().map_err(|_| {
                GGUFError::IncompleteData("Failed to read tensor dimension".into())
            })?);
        }

        let ggml_type_raw = reader
            .read_u32::<LittleEndian>()
            .map_err(|_| GGUFError::IncompleteData("Failed to read tensor type".into()))?;
        let offset = reader
            .read_u64::<LittleEndian>()
            .map_err(|_| GGUFError::IncompleteData("Failed to read tensor offset".into()))?;

        let ggml_type = GGMLType::try_from(ggml_type_raw)?;

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
    fn test_parse_magic_and_version() {
        // Create a minimal GGUF header: magic + version
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC); // Magic: "GGUF"
        data.extend_from_slice(&2u32.to_le_bytes()); // Version: 2
        data.extend_from_slice(&0u64.to_le_bytes()); // Tensor count: 0
        data.extend_from_slice(&0u64.to_le_bytes()); // Metadata KV count: 0

        let mut reader = Cursor::new(&data);

        // Verify magic
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic).unwrap();
        assert_eq!(magic, GGUF_MAGIC);

        // Verify version
        let version = reader.read_u32::<LittleEndian>().unwrap();
        assert_eq!(version, 2);
    }

    #[test]
    fn test_parse_minimal_gguf_header() {
        // Create a minimal valid GGUF file
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC); // Magic: "GGUF"
        data.extend_from_slice(&2u32.to_le_bytes()); // Version: 2
        data.extend_from_slice(&0u64.to_le_bytes()); // Tensor count: 0
        data.extend_from_slice(&0u64.to_le_bytes()); // Metadata KV count: 0

        let result = GGUFFile::parse_header(&data);
        assert!(result.is_ok());

        let (metadata, _offset) = result.unwrap();
        assert_eq!(metadata.header.version, 2);
        assert_eq!(metadata.header.tensor_count, 0);
        assert_eq!(metadata.header.metadata_kv_count, 0);
    }

    #[test]
    fn test_parse_string() {
        let mut data = Vec::new();
        let test_str = "test_string";
        data.extend_from_slice(&(test_str.len() as u64).to_le_bytes());
        data.extend_from_slice(test_str.as_bytes());

        let mut reader = Cursor::new(&data);
        let result = parse_string(&mut reader).unwrap();
        assert_eq!(result, test_str);
    }

    #[test]
    fn test_parse_metadata_uint32() {
        let mut data = Vec::new();
        data.extend_from_slice(&4u32.to_le_bytes()); // Type: UInt32
        data.extend_from_slice(&42u32.to_le_bytes()); // Value: 42

        let mut reader = Cursor::new(&data);
        let result = parse_metadata_value(&mut reader).unwrap();

        match result {
            MetadataValue::UInt32(val) => assert_eq!(val, 42),
            _ => panic!("Expected UInt32 metadata value"),
        }
    }

    #[test]
    fn test_parse_metadata_string() {
        let mut data = Vec::new();
        data.extend_from_slice(&8u32.to_le_bytes()); // Type: String
        let test_str = "hello";
        data.extend_from_slice(&(test_str.len() as u64).to_le_bytes());
        data.extend_from_slice(test_str.as_bytes());

        let mut reader = Cursor::new(&data);
        let result = parse_metadata_value(&mut reader).unwrap();

        match result {
            MetadataValue::String(val) => assert_eq!(val, test_str),
            _ => panic!("Expected String metadata value"),
        }
    }

    #[test]
    fn test_invalid_magic() {
        let mut data = Vec::new();
        data.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // Invalid magic
        data.extend_from_slice(&2u32.to_le_bytes()); // Version: 2

        let result = GGUFFile::parse_header(&data);
        assert!(matches!(result, Err(GGUFError::InvalidMagic(_))));
    }

    #[test]
    fn test_unsupported_version() {
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC); // Magic: "GGUF"
        data.extend_from_slice(&1u32.to_le_bytes()); // Version: 1 (unsupported)
        data.extend_from_slice(&0u64.to_le_bytes()); // Tensor count: 0
        data.extend_from_slice(&0u64.to_le_bytes()); // Metadata KV count: 0

        let result = GGUFFile::parse_header(&data);
        assert!(matches!(result, Err(GGUFError::UnsupportedVersion(1))));
    }

    #[test]
    fn test_align_offset() {
        assert_eq!(align_offset(0, 32), 0);
        assert_eq!(align_offset(1, 32), 32);
        assert_eq!(align_offset(32, 32), 32);
        assert_eq!(align_offset(33, 32), 64);
        assert_eq!(align_offset(100, 32), 128);
    }

    #[test]
    fn test_parse_header_with_metadata() {
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC); // Magic: "GGUF"
        data.extend_from_slice(&2u32.to_le_bytes()); // Version: 2
        data.extend_from_slice(&0u64.to_le_bytes()); // Tensor count: 0
        data.extend_from_slice(&1u64.to_le_bytes()); // Metadata KV count: 1

        // Add one metadata entry: key = "test.key", value = UInt32(123)
        let key = "test.key";
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key.as_bytes());
        data.extend_from_slice(&4u32.to_le_bytes()); // Type: UInt32
        data.extend_from_slice(&123u32.to_le_bytes()); // Value: 123

        let result = GGUFFile::parse_header(&data);
        assert!(result.is_ok());

        let (metadata, _offset) = result.unwrap();
        assert_eq!(metadata.header.metadata_kv_count, 1);
        assert!(metadata.metadata.contains_key("test.key"));

        match metadata.metadata.get("test.key").unwrap() {
            MetadataValue::UInt32(val) => assert_eq!(*val, 123),
            _ => panic!("Expected UInt32 metadata value"),
        }
    }
}

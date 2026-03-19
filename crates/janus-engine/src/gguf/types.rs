use std::collections::HashMap;

/// GGML tensor data types (quantization formats)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum GGMLType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    I8 = 16,
    I16 = 17,
    I32 = 18,
}

impl GGMLType {
    /// Parse from u32
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2_K),
            11 => Some(Self::Q3_K),
            12 => Some(Self::Q4_K),
            13 => Some(Self::Q5_K),
            14 => Some(Self::Q6_K),
            15 => Some(Self::Q8_K),
            16 => Some(Self::I8),
            17 => Some(Self::I16),
            18 => Some(Self::I32),
            _ => None,
        }
    }

    /// Get the size in bytes for a single element of this type
    pub fn element_size(&self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::I16 => 2,
            Self::I8 => 1,
            // Quantized types use block sizes
            Self::Q4_0 | Self::Q4_1 => 18, // 32 values in 18 bytes
            Self::Q5_0 | Self::Q5_1 => 22, // 32 values in 22 bytes
            Self::Q8_0 | Self::Q8_1 => 34, // 32 values in 34 bytes
            Self::Q2_K => 82,              // 256 values in 82 bytes
            Self::Q3_K => 110,             // 256 values in 110 bytes
            Self::Q4_K => 144,             // 256 values in 144 bytes
            Self::Q5_K => 176,             // 256 values in 176 bytes
            Self::Q6_K => 210,             // 256 values in 210 bytes
            Self::Q8_K => 292,             // 256 values in 292 bytes
        }
    }
}

/// Metadata value types
#[derive(Debug, Clone)]
pub enum MetadataValue {
    UInt8(u8),
    Int8(i8),
    UInt16(u16),
    Int16(i16),
    UInt32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
    UInt64(u64),
    Int64(i64),
    Float64(f64),
}

/// Tensor information
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor name
    pub name: String,
    /// Number of dimensions (max 4)
    pub n_dimensions: u32,
    /// Dimension sizes
    pub dimensions: Vec<u64>,
    /// Tensor data type
    pub ggml_type: GGMLType,
    /// Offset to tensor data in file
    pub offset: u64,
}

impl TensorInfo {
    /// Calculate total number of elements
    pub fn element_count(&self) -> u64 {
        self.dimensions.iter().product()
    }

    /// Calculate total size in bytes
    pub fn size_bytes(&self) -> u64 {
        let elem_count = self.element_count();
        let elem_size = self.ggml_type.element_size() as u64;

        // For quantized types, calculate based on block size
        match self.ggml_type {
            GGMLType::Q4_0
            | GGMLType::Q4_1
            | GGMLType::Q5_0
            | GGMLType::Q5_1
            | GGMLType::Q8_0
            | GGMLType::Q8_1 => {
                let block_size = 32;
                let num_blocks = (elem_count + block_size - 1) / block_size;
                num_blocks * elem_size
            }
            GGMLType::Q2_K
            | GGMLType::Q3_K
            | GGMLType::Q4_K
            | GGMLType::Q5_K
            | GGMLType::Q6_K
            | GGMLType::Q8_K => {
                let block_size = 256;
                let num_blocks = (elem_count + block_size - 1) / block_size;
                num_blocks * elem_size
            }
            _ => elem_count * elem_size,
        }
    }
}

/// Metadata type discriminator
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum MetadataValueType {
    UInt8 = 0,
    Int8 = 1,
    UInt16 = 2,
    Int16 = 3,
    UInt32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    UInt64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl MetadataValueType {
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::UInt8),
            1 => Some(Self::Int8),
            2 => Some(Self::UInt16),
            3 => Some(Self::Int16),
            4 => Some(Self::UInt32),
            5 => Some(Self::Int32),
            6 => Some(Self::Float32),
            7 => Some(Self::Bool),
            8 => Some(Self::String),
            9 => Some(Self::Array),
            10 => Some(Self::UInt64),
            11 => Some(Self::Int64),
            12 => Some(Self::Float64),
            _ => None,
        }
    }
}

/// GGUF file header
#[derive(Debug, Clone)]
pub struct GGUFHeader {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}

/// Complete GGUF metadata
#[derive(Debug, Clone)]
pub struct GGUFMetadata {
    pub header: GGUFHeader,
    pub metadata: HashMap<String, MetadataValue>,
    pub tensors: Vec<TensorInfo>,
    pub alignment: u64,
}

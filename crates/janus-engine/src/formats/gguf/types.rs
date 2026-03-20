use std::collections::HashMap;

use super::error::GGUFError;

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
    I64 = 19,
    F64 = 20,
    IQ2_XXS = 21,
    IQ2_XS = 22,
    IQ3_XXS = 23,
    IQ1_S = 24,
    IQ4_NL = 25,
    IQ3_S = 26,
    IQ2_S = 27,
    IQ4_XS = 28,
    IQ1_M = 29,
}

impl TryFrom<u32> for GGMLType {
    type Error = GGUFError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            9 => Ok(Self::Q8_1),
            10 => Ok(Self::Q2_K),
            11 => Ok(Self::Q3_K),
            12 => Ok(Self::Q4_K),
            13 => Ok(Self::Q5_K),
            14 => Ok(Self::Q6_K),
            15 => Ok(Self::Q8_K),
            16 => Ok(Self::I8),
            17 => Ok(Self::I16),
            18 => Ok(Self::I32),
            19 => Ok(Self::I64),
            20 => Ok(Self::F64),
            21 => Ok(Self::IQ2_XXS),
            22 => Ok(Self::IQ2_XS),
            23 => Ok(Self::IQ3_XXS),
            24 => Ok(Self::IQ1_S),
            25 => Ok(Self::IQ4_NL),
            26 => Ok(Self::IQ3_S),
            27 => Ok(Self::IQ2_S),
            28 => Ok(Self::IQ4_XS),
            29 => Ok(Self::IQ1_M),
            _ => Err(GGUFError::InvalidTensorType(value)),
        }
    }
}

impl GGMLType {
    /// Get the size in bytes for a single element of this type
    pub fn element_size(&self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::I16 => 2,
            Self::F64 | Self::I64 => 8,
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
            // IQ quantization types (approximate sizes)
            Self::IQ2_XXS => 66, // 256 values
            Self::IQ2_XS => 74,  // 256 values
            Self::IQ3_XXS => 98, // 256 values
            Self::IQ1_S => 50,   // 256 values
            Self::IQ4_NL => 136, // 256 values
            Self::IQ3_S => 110,  // 256 values
            Self::IQ2_S => 82,   // 256 values
            Self::IQ4_XS => 136, // 256 values
            Self::IQ1_M => 56,   // 256 values
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
            | GGMLType::Q8_K
            | GGMLType::IQ2_XXS
            | GGMLType::IQ2_XS
            | GGMLType::IQ3_XXS
            | GGMLType::IQ1_S
            | GGMLType::IQ4_NL
            | GGMLType::IQ3_S
            | GGMLType::IQ2_S
            | GGMLType::IQ4_XS
            | GGMLType::IQ1_M => {
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

impl TryFrom<u32> for MetadataValueType {
    type Error = GGUFError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::UInt8),
            1 => Ok(Self::Int8),
            2 => Ok(Self::UInt16),
            3 => Ok(Self::Int16),
            4 => Ok(Self::UInt32),
            5 => Ok(Self::Int32),
            6 => Ok(Self::Float32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::UInt64),
            11 => Ok(Self::Int64),
            12 => Ok(Self::Float64),
            _ => Err(GGUFError::InvalidMetadataType(value)),
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

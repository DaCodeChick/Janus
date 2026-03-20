//! Model file format loading abstraction layer
//!
//! This module provides a unified interface for loading model weights from different
//! file formats (GGUF, Safetensors, etc.) with zero-copy memory-mapped I/O.
//!
//! # Supported Formats
//!
//! - **GGUF**: GPT-Generated Unified Format - used by llama.cpp
//! - **Safetensors**: Safe, simple tensor storage format from Hugging Face
//!
//! # Usage
//!
//! ## Loading a GGUF file
//!
//! ```no_run
//! use janus_engine::{GGUFLoader, ModelLoader, ComputeEngine};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Load GGUF file with zero-copy mmap
//! let loader = GGUFLoader::from_file("model.gguf")?;
//!
//! // Access tensors via the unified ModelLoader trait
//! let tensors = loader.tensors()?;
//! println!("Loaded {} tensors", tensors.len());
//!
//! // Get specific tensor
//! let token_embd = loader.get_tensor("token_embd.weight")?;
//! println!("Token embedding shape: {:?}", token_embd.shape);
//!
//! // Allocate to GPU
//! let engine = ComputeEngine::new().await?;
//! let gpu_buffers = engine.allocate_tensors(&loader)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Loading a Safetensors file
//!
//! ```no_run
//! use janus_engine::{SafetensorsLoader, ModelLoader, ComputeEngine};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Load Safetensors file with zero-copy mmap
//! let loader = SafetensorsLoader::from_file("model.safetensors")?;
//!
//! // Same unified interface as GGUF!
//! let tensors = loader.tensors()?;
//! println!("Loaded {} tensors", tensors.len());
//!
//! // Allocate to GPU using the same code
//! let engine = ComputeEngine::new().await?;
//! let gpu_buffers = engine.allocate_tensors(&loader)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Generic code with ModelLoader trait
//!
//! ```no_run
//! use janus_engine::{ModelLoader, ComputeEngine};
//!
//! # async fn example<L: ModelLoader>(loader: &L) -> Result<(), Box<dyn std::error::Error>> {
//! // This function works with any format!
//! let tensors = loader.tensors()?;
//!
//! // Allocate to GPU - format-agnostic
//! let engine = ComputeEngine::new().await?;
//! let gpu_buffers = engine.allocate_tensors(loader)?;
//! # Ok(())
//! # }
//! ```
//!
//! # Zero-Copy Performance
//!
//! Both loaders use memory-mapped I/O (`mmap`) to avoid loading the entire model
//! into RAM. Tensor data is accessed as `&[u8]` slices pointing directly to the
//! memory-mapped file, enabling efficient zero-copy transfers to GPU.

pub mod gguf;
pub mod safetensors;

// Re-export GGUF types for convenience
pub use gguf::{GGMLType, GGUFError, GGUFFile, GGUFMetadata, MetadataValue, TensorInfo};

// Re-export Safetensors types
pub use safetensors::SafetensorsFile;

use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur during model loading
#[derive(Debug, Error)]
pub enum FormatError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid file format: {0}")]
    InvalidFormat(String),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Tensor not found: {0}")]
    TensorNotFound(String),

    #[error("Invalid tensor data: {0}")]
    InvalidTensorData(String),
}

pub type Result<T> = std::result::Result<T, FormatError>;

/// Data type of a tensor
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDType {
    F32,
    F16,
    BF16,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
}

impl TensorDType {
    /// Get the size in bytes of a single element of this data type
    pub fn element_size(&self) -> usize {
        match self {
            TensorDType::F32 | TensorDType::I32 | TensorDType::U32 => 4,
            TensorDType::F16 | TensorDType::BF16 | TensorDType::I16 | TensorDType::U16 => 2,
            TensorDType::I8 | TensorDType::U8 => 1,
            TensorDType::I64 | TensorDType::U64 => 8,
            // Quantized types have variable sizes - these are approximations
            TensorDType::Q4_0 | TensorDType::Q4_1 => 1, // ~0.5 bytes per element in blocks
            TensorDType::Q5_0 | TensorDType::Q5_1 => 1, // ~0.625 bytes per element in blocks
            TensorDType::Q8_0 | TensorDType::Q8_1 => 1,
        }
    }
}

/// Represents a tensor's metadata and data slice from a memory-mapped file
#[derive(Debug)]
pub struct TensorData<'a> {
    /// Name of the tensor
    pub name: String,

    /// Shape of the tensor (dimensions)
    pub shape: Vec<usize>,

    /// Data type of the tensor elements
    pub dtype: TensorDType,

    /// Raw byte slice pointing directly to the memory-mapped data
    /// This is a zero-copy reference to the underlying mmap
    pub data: &'a [u8],
}

impl<'a> TensorData<'a> {
    /// Calculate the total number of elements in the tensor
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Calculate the expected size in bytes based on shape and dtype
    pub fn expected_size(&self) -> usize {
        self.num_elements() * self.dtype.element_size()
    }
}

/// Trait for loading model weights from different file formats
pub trait ModelLoader {
    /// Get all tensors in the model as a HashMap
    /// Returns tensor name -> TensorData with zero-copy byte slices
    fn tensors(&self) -> Result<HashMap<String, TensorData<'_>>>;

    /// Get a specific tensor by name
    fn get_tensor(&self, name: &str) -> Result<TensorData<'_>> {
        self.tensors()?
            .remove(name)
            .ok_or_else(|| FormatError::TensorNotFound(name.to_string()))
    }

    /// Get metadata value by key (if supported by the format)
    fn get_metadata(&self, key: &str) -> Option<String>;

    /// Get all metadata keys available
    fn metadata_keys(&self) -> Vec<String>;
}

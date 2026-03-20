//! Janus Engine - High-Performance Modular LLM Inference Engine
//!
//! This crate provides:
//! - Model file format parsing (GGUF, Safetensors)
//! - GPU compute backend using WGPU
//! - Transformer model components
//! - Token sampling and decoding
//! - Plugin loading infrastructure

pub mod compute;
pub mod formats;
pub mod gguf;
pub mod loader;
pub mod model;

// Re-export commonly used types
pub use compute::{Buffer, BufferUsage, ComputeBackend, ComputeEngine, DeviceInfo, KVCache};
pub use formats::{
    gguf::GGUFFile as GGUFLoader,
    safetensors::SafetensorsFile as SafetensorsLoader,
    FormatError, ModelLoader, TensorData, TensorDType,
};
pub use gguf::{GGMLType, GGUFFile, MetadataValue, TensorInfo};
pub use model::{
    LMHead, LMHeadConfig, Model, ModelConfig, Sampler, SamplerConfig, Tokenizer,
    TokenizerError, TransformerBlock, TransformerBlockConfig,
};

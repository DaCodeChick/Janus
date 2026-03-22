//! Janus Engine - High-Performance Modular LLM Inference Engine
//!
//! This crate provides:
//! - Model file format parsing (GGUF, Safetensors)
//! - GPU compute backend using WGPU
//! - Transformer model components
//! - Token sampling and decoding
//! - Plugin loading infrastructure

// Enforce error handling best practices (per .opencode/AGENTS.md)
#![warn(clippy::unwrap_used)]
#![warn(clippy::expect_used)]
#![warn(clippy::panic)]
#![warn(clippy::todo)]
#![warn(clippy::unimplemented)]

pub mod compute;
pub mod formats;
pub mod loader;
pub mod model;

// Re-export commonly used types
pub use compute::{Buffer, BufferUsage, ComputeBackend, ComputeEngine, DeviceInfo, KVCache};
pub use formats::{
    GGMLType, GGUFError, GGUFFile, GGUFMetadata, MetadataValue, SafetensorsFile, TensorInfo,
    FormatError, ModelLoader, TensorData, TensorDType,
};
pub use model::{
    LMHead, LMHeadConfig, Model, ModelConfig, Sampler, SamplerConfig, Tokenizer,
    TokenizerError, TransformerBlock, TransformerBlockConfig, HuggingFaceConfig,
    ConfigError,
};

// Convenience aliases for model loaders
pub use formats::GGUFFile as GGUFLoader;
pub use formats::SafetensorsFile as SafetensorsLoader;

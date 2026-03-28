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
pub mod app;

// Re-export commonly used types
pub use compute::{Buffer, BufferUsage, ComputeBackend, ComputeEngine, DeviceInfo, GpuTensor, KVCache};
pub use app::{JanusApp, JanusPlugin};
pub use formats::{
    GGMLType, GgufError, GgufFile, GgufMetadata, MetadataValue, SafetensorsFile, TensorInfo,
    FormatError, ModelLoader, TensorData, TensorDType,
};

// Backward-compatible aliases
pub use formats::{GGUFError, GGUFFile, GGUFMetadata};
pub use model::{
    ChatFormatter, ChatMessage, ChatRole, ChatTemplateFormat,
    LMHead, LMHeadConfig, Model, ModelConfig, Sampler, SamplerConfig, Tokenizer,
    TokenizerError, TransformerBlock, TransformerBlockConfig, HuggingFaceConfig,
    ConfigError,
};

// Convenience aliases for model loaders
pub use formats::GgufFile as GgufLoader;
pub use formats::SafetensorsFile as SafetensorsLoader;

// Backward-compatible alias
pub use formats::GGUFFile as GGUFLoader;

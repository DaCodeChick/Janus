//! Janus Engine - High-Performance Modular LLM Inference Engine
//!
//! This crate provides:
//! - GGUF model file parsing
//! - GPU compute backend using WGPU
//! - Transformer model components
//! - Token sampling and decoding
//! - Plugin loading infrastructure

pub mod compute;
pub mod gguf;
pub mod loader;
pub mod model;

// Re-export commonly used types
pub use compute::{Buffer, BufferUsage, ComputeBackend, ComputeEngine, DeviceInfo, KVCache};
pub use gguf::{GGMLType, GGUFFile, MetadataValue, TensorInfo};
pub use model::{
    LMHead, LMHeadConfig, Sampler, SamplerConfig, Tokenizer, TokenizerError, TransformerBlock,
    TransformerBlockConfig,
};

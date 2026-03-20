//! Janus Engine - High-Performance Modular LLM Inference Engine
//!
//! This crate provides:
//! - GGUF model file parsing
//! - GPU compute backend using WGPU
//! - Transformer model components
//! - Plugin loading infrastructure

pub mod compute;
pub mod gguf;
pub mod loader;
pub mod model;

// Re-export commonly used types
pub use compute::{Buffer, BufferUsage, ComputeBackend, DeviceInfo};
pub use gguf::{GGMLType, GGUFFile, MetadataValue, TensorInfo};
pub use model::{TransformerBlock, TransformerBlockConfig};

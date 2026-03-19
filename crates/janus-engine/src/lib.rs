//! Janus Engine - High-Performance Modular LLM Inference Engine
//!
//! This crate provides:
//! - GGUF model file parsing
//! - GPU compute backend using WGPU
//! - Plugin loading infrastructure

pub mod gguf;
pub mod compute;
pub mod loader;

// Re-export commonly used types
pub use gguf::{GGUFFile, GGMLType, MetadataValue, TensorInfo};
pub use compute::{ComputeBackend, DeviceInfo, Buffer, BufferUsage};

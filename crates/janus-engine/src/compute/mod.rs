//! WGPU-based GPU compute backend for LLM inference

mod backend;
mod buffer;
pub mod cache;
mod engine;
mod error;
mod kernels;
pub mod ops;
pub mod pipeline_cache;

pub use backend::{ComputeBackend, DeviceInfo};
pub use buffer::{Buffer, BufferUsage};
pub use cache::KVCache;
pub use engine::ComputeEngine;
pub use error::{ComputeError, Result};
pub use kernels::Kernels;
pub use pipeline_cache::PipelineCache;

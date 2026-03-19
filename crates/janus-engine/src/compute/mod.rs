//! WGPU-based GPU compute backend for LLM inference

mod backend;
mod buffer;
mod error;
mod kernels;

pub use backend::{ComputeBackend, DeviceInfo};
pub use buffer::{Buffer, BufferUsage};
pub use error::{ComputeError, Result};
pub use kernels::Kernels;

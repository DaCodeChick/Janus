use crate::types::{PluginErrorCode, ResponseChunk};
use abi_stable::StableAbi;

/// Streaming callback function pointer (ABI-stable)
///
/// # Safety
/// - Must be thread-safe
/// - The `chunk` reference is only valid during callback invocation
/// - Must not panic across FFI boundary
#[repr(C)]
#[derive(StableAbi, Copy, Clone)]
pub struct StreamCallback {
    pub callback: extern "C" fn(chunk: &ResponseChunk) -> PluginErrorCode,
}

impl StreamCallback {
    pub const fn new(callback: extern "C" fn(&ResponseChunk) -> PluginErrorCode) -> Self {
        Self { callback }
    }

    pub fn call(&self, chunk: &ResponseChunk) -> PluginErrorCode {
        (self.callback)(chunk)
    }
}

/// Progress callback for long-running operations
///
/// # Arguments
/// - `progress`: Value from 0.0 to 1.0 indicating completion percentage
#[repr(C)]
#[derive(StableAbi, Copy, Clone)]
pub struct ProgressCallback {
    pub callback: extern "C" fn(progress: f32),
}

impl ProgressCallback {
    pub const fn new(callback: extern "C" fn(f32)) -> Self {
        Self { callback }
    }

    pub fn call(&self, progress: f32) {
        (self.callback)(progress);
    }
}

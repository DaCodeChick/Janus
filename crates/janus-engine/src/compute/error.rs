use thiserror::Error;

/// Compute backend errors
#[derive(Error, Debug)]
pub enum ComputeError {
    #[error("Failed to initialize WGPU adapter")]
    AdapterNotFound,

    #[error("Failed to request device: {0}")]
    DeviceRequestFailed(String),

    #[error("Buffer creation failed: {0}")]
    BufferCreationFailed(String),

    #[error("Shader compilation failed: {0}")]
    ShaderCompilationFailed(String),

    #[error("Kernel execution failed: {0}")]
    KernelExecutionFailed(String),

    #[error("Invalid buffer size: expected {expected}, got {actual}")]
    InvalidBufferSize { expected: usize, actual: usize },

    #[error("Buffer mapping failed")]
    BufferMappingFailed,

    #[error("Invalid tensor dimensions: {0}")]
    InvalidDimensions(String),

    #[error("Other error: {0}")]
    Other(String),
}

/// Result type for compute operations
pub type Result<T> = std::result::Result<T, ComputeError>;

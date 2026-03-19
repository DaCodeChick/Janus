use thiserror::Error;

/// GGUF parsing errors
#[derive(Error, Debug)]
pub enum GGUFError {
    #[error("Invalid magic number: expected GGUF, got {0:?}")]
    InvalidMagic([u8; 4]),

    #[error("Unsupported GGUF version: {0}")]
    UnsupportedVersion(u32),

    #[error("Invalid metadata value type: {0}")]
    InvalidMetadataType(u32),

    #[error("Invalid GGML tensor type: {0}")]
    InvalidTensorType(u32),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Incomplete data: {0}")]
    IncompleteData(String),
}

/// Result type for GGUF operations
pub type Result<T> = std::result::Result<T, GGUFError>;

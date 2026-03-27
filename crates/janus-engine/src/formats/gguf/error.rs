use thiserror::Error;

/// GGUF parsing errors with helpful suggestions
#[derive(Error, Debug)]
pub enum GgufError {
    #[error("Invalid GGUF file: magic number check failed\nExpected: GGUF (0x47475546)\nGot: {0:?}\n\nSuggestions:\n  - Verify this is a valid GGUF file (not a Safetensors or PyTorch file)\n  - Check if the file was corrupted during download\n  - Try re-downloading the model")]
    InvalidMagic([u8; 4]),

    #[error("Unsupported GGUF version: {0}\n\nSupported versions: 2, 3\n\nSuggestions:\n  - This file uses GGUF version {0} which is not yet supported\n  - Try converting the model to GGUF v2 or v3\n  - Check if a newer version of Janus supports this format")]
    UnsupportedVersion(u32),

    #[error("Invalid metadata value type: {0}\n\nSupported types: String (0), UInt8 (6), UInt32 (4), Float32 (7), Array (9)\n\nSuggestions:\n  - The GGUF file may be corrupted\n  - Try re-downloading or re-converting the model")]
    InvalidMetadataType(u32),

    #[error("Invalid GGML tensor type: {0}\n\nSupported types:\n  - F32 (0): 32-bit floating point\n  - Q4_K (12): 4-bit quantized (K-quant)\n  - Q5_K (13): 5-bit quantized (K-quant)\n  - Q8_0 (8): 8-bit quantized\n\nSuggestions:\n  - This tensor type is not supported yet\n  - Try using a different quantization format\n  - Check the Janus documentation for supported formats")]
    InvalidTensorType(u32),

    #[error("Parse error: {0}\n\nSuggestions:\n  - The GGUF file structure may be invalid\n  - Verify the file was completely downloaded\n  - Try re-downloading or re-converting the model")]
    ParseError(String),

    #[error("IO error while reading GGUF file: {0}\n\nSuggestions:\n  - Check file permissions\n  - Ensure sufficient disk space\n  - Verify the file path is correct")]
    IoError(#[from] std::io::Error),

    #[error("Incomplete data: {0}\n\nSuggestions:\n  - The file may have been truncated during download\n  - Verify file size matches expected size\n  - Try re-downloading the model")]
    IncompleteData(String),

    #[error("Missing required tensor: {tensor_name}\n\nExpected tensor patterns for {architecture}:\n{expected_tensors}\n\nSuggestions:\n  - Verify this is a complete model file (not a LoRA or partial checkpoint)\n  - Check if the model uses a different naming convention\n  - Try a different model file format (e.g., Safetensors)")]
    MissingTensor {
        tensor_name: String,
        architecture: String,
        expected_tensors: String,
    },

    #[error("Tensor shape mismatch: {tensor_name}\nExpected shape: {expected:?}\nActual shape: {actual:?}\n\nSuggestions:\n  - The model architecture may not match the config.json\n  - Verify you're using the correct config file for this model\n  - Check if the model is a variant with different dimensions")]
    TensorShapeMismatch {
        tensor_name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
}

/// Result type for GGUF operations
pub type Result<T> = std::result::Result<T, GgufError>;

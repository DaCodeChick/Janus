//! Model configuration parsing from HuggingFace config.json
//!
//! This module handles loading model hyperparameters from the standard
//! HuggingFace `config.json` format used by transformers.

use serde::Deserialize;
use std::fs;
use std::path::Path;
use thiserror::Error;

use crate::formats::{GgufMetadata, MetadataValue};

/// Errors that can occur when loading model configuration
#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("Failed to read config file: {0}\n\nSuggestions:\n  - Verify the file path exists\n  - Check file permissions\n  - If using a directory path, ensure config.json exists inside")]
    IoError(#[from] std::io::Error),

    #[error("Failed to parse config JSON: {0}\n\nSuggestions:\n  - Verify the file is valid JSON (use a JSON validator)\n  - Check for trailing commas or syntax errors\n  - Compare with a known-good config.json")]
    ParseError(#[from] serde_json::Error),

    #[error("Missing required field: {0}\n\nRequired fields:\n  - hidden_size\n  - num_hidden_layers\n  - num_attention_heads\n  - vocab_size\n\nSuggestions:\n  - Add the missing field to your config.json\n  - Use a complete config.json from HuggingFace model repo")]
    MissingField(String),

    #[error("Invalid configuration: {0}\n\nSuggestions:\n  - Verify all dimensions are positive integers\n  - Check that hidden_size is divisible by num_attention_heads\n  - Ensure num_key_value_heads divides num_attention_heads evenly (for GQA)")]
    InvalidConfig(String),

    #[error("Unsupported architecture: {architecture}\n\nSupported architectures:\n  - LlamaForCausalLM (LLaMA, Mistral, Vicuna)\n  - MistralForCausalLM (Mistral)\n  - GPTNeoXForCausalLM (TinyLlama, Pythia)\n  - PhiForCausalLM (Microsoft Phi)\n  - Phi3ForCausalLM (Microsoft Phi-3)\n  - GemmaForCausalLM (Google Gemma)\n  - Gemma2ForCausalLM (Google Gemma 2)\n  - QWenLMHeadModel (Alibaba Qwen)\n  - Qwen2ForCausalLM (Alibaba Qwen 2)\n\nGot architecture: {architecture}\n\nNote: Most modern decoder-only transformer architectures use compatible components\n(RoPE, RMSNorm, GQA) and should work with LLaMA-style inference.\n\nSuggestions:\n  - Check if this architecture is supported in the latest Janus version\n  - Try using a compatible config (many models are LLaMA-compatible)\n  - File an issue on GitHub if you need support for this architecture")]
    UnsupportedArchitecture { architecture: String },

    #[error("Missing required GGUF metadata key: {key}\n\nSuggestions:\n  - Verify this GGUF file contains full architecture metadata\n  - Ensure the model was exported with a recent GGUF version\n  - Check if this is a supported architecture")]
    MissingGgufMetadata { key: String },
}

/// Result type for config operations
pub type Result<T> = std::result::Result<T, ConfigError>;

/// Model configuration loaded from HuggingFace config.json
///
/// This struct deserializes the standard HuggingFace transformer config
/// and maps it to Janus's internal ModelConfig format.
///
/// # Example config.json
///
/// ```json
/// {
///   "architectures": ["LlamaForCausalLM"],
///   "hidden_size": 4096,
///   "intermediate_size": 11008,
///   "num_hidden_layers": 32,
///   "num_attention_heads": 32,
///   "num_key_value_heads": 32,
///   "vocab_size": 32000,
///   "max_position_embeddings": 2048,
///   "rms_norm_eps": 1e-5
/// }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct HuggingFaceConfig {
    /// Model hidden dimension (e.g., 4096 for LLaMA-7B)
    pub hidden_size: u32,

    /// Number of transformer layers (e.g., 32 for LLaMA-7B)
    pub num_hidden_layers: u32,

    /// Number of attention heads (e.g., 32 for LLaMA-7B)
    pub num_attention_heads: u32,

    /// Number of KV heads for grouped-query attention
    /// Falls back to num_attention_heads if not specified (multi-head attention)
    #[serde(default)]
    pub num_key_value_heads: Option<u32>,

    /// Vocabulary size (e.g., 32000 for LLaMA)
    pub vocab_size: u32,

    /// Feed-forward intermediate dimension (e.g., 11008 for LLaMA-7B)
    #[serde(default)]
    pub intermediate_size: Option<u32>,

    /// Maximum sequence length (e.g., 2048)
    #[serde(alias = "max_position_embeddings")]
    pub max_seq_length: Option<u32>,

    /// RMSNorm epsilon
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,

    /// Model architecture name (e.g., "LlamaForCausalLM")
    #[serde(default)]
    pub architectures: Option<Vec<String>>,
}

fn default_rms_norm_eps() -> f32 {
    1e-5
}

impl HuggingFaceConfig {
    /// Load configuration from a config.json file
    ///
    /// # Arguments
    /// * `path` - Path to config.json or directory containing it
    ///
    /// # Example
    /// ```no_run
    /// use janus_engine::model::HuggingFaceConfig;
    ///
    /// let config = HuggingFaceConfig::from_file("models/llama-7b/config.json")?;
    /// println!("Model has {} layers", config.num_hidden_layers);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();

        // If path is a directory, append "config.json"
        let config_path = if path.is_dir() {
            path.join("config.json")
        } else {
            path.to_path_buf()
        };

        let contents = fs::read_to_string(&config_path).map_err(|e| ConfigError::IoError(e))?;

        let config: HuggingFaceConfig = serde_json::from_str(&contents)?;

        // Validate required fields
        config.validate()?;

        Ok(config)
    }

    /// Validate that all required fields are present and valid
    pub fn validate(&self) -> Result<()> {
        // Check positive values
        if self.hidden_size == 0 {
            return Err(ConfigError::InvalidConfig(
                "hidden_size must be greater than 0".into(),
            ));
        }

        if self.num_hidden_layers == 0 {
            return Err(ConfigError::InvalidConfig(
                "num_hidden_layers must be greater than 0".into(),
            ));
        }

        if self.num_attention_heads == 0 {
            return Err(ConfigError::InvalidConfig(
                "num_attention_heads must be greater than 0".into(),
            ));
        }

        if self.vocab_size == 0 {
            return Err(ConfigError::InvalidConfig(
                "vocab_size must be greater than 0".into(),
            ));
        }

        // Check that hidden_size is divisible by num_attention_heads
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(ConfigError::InvalidConfig(format!(
                "hidden_size ({}) must be divisible by num_attention_heads ({})\nhead_dim would be {}.{} which is not an integer",
                self.hidden_size, 
                self.num_attention_heads,
                self.hidden_size / self.num_attention_heads,
                self.hidden_size % self.num_attention_heads
            )));
        }

        // Check that num_attention_heads is divisible by num_key_value_heads (for GQA)
        let num_kv_heads = self.num_kv_heads();
        if self.num_attention_heads % num_kv_heads != 0 {
            return Err(ConfigError::InvalidConfig(format!(
                "num_attention_heads ({}) must be divisible by num_key_value_heads ({})\nFor Grouped Query Attention, each KV head must serve an equal number of Q heads",
                self.num_attention_heads,
                num_kv_heads
            )));
        }

        // Validate architecture if specified
        if let Some(ref archs) = self.architectures {
            let supported_archs = [
                "LlamaForCausalLM",
                "MistralForCausalLM",
                "GPTNeoXForCausalLM",
                "PhiForCausalLM",
                "Phi3ForCausalLM",
                "GemmaForCausalLM",
                "Gemma2ForCausalLM",
                "QWenLMHeadModel",
                "Qwen2ForCausalLM",
            ];

            if !archs.is_empty() {
                let arch = &archs[0];
                if !supported_archs.contains(&arch.as_str()) {
                    return Err(ConfigError::UnsupportedArchitecture {
                        architecture: arch.clone(),
                    });
                }
            }
        }

        // Validate reasonable ranges (catch obviously wrong values)
        if self.hidden_size > 32768 {
            return Err(ConfigError::InvalidConfig(format!(
                "hidden_size ({}) seems unreasonably large (>32768)\nPlease verify this is correct",
                self.hidden_size
            )));
        }

        if self.num_hidden_layers > 256 {
            return Err(ConfigError::InvalidConfig(format!(
                "num_hidden_layers ({}) seems unreasonably large (>256)\nPlease verify this is correct",
                self.num_hidden_layers
            )));
        }

        if self.vocab_size > 1_000_000 {
            return Err(ConfigError::InvalidConfig(format!(
                "vocab_size ({}) seems unreasonably large (>1M)\nPlease verify this is correct",
                self.vocab_size
            )));
        }

        Ok(())
    }

    /// Get the number of key-value heads (for GQA)
    ///
    /// Returns num_key_value_heads if specified, otherwise falls back
    /// to num_attention_heads (standard multi-head attention)
    pub const fn num_kv_heads(&self) -> u32 {
        if let Some(kv_heads) = self.num_key_value_heads {
            kv_heads
        } else {
            self.num_attention_heads
        }
    }

    /// Get the head dimension
    ///
    /// Computed as hidden_size / num_attention_heads
    pub const fn head_dim(&self) -> u32 {
        self.hidden_size / self.num_attention_heads
    }

    /// Get the feed-forward dimension
    ///
    /// Returns intermediate_size if specified, otherwise computes
    /// the standard 4 * hidden_size (typical for transformers)
    pub const fn ffn_dim(&self) -> u32 {
        if let Some(intermediate) = self.intermediate_size {
            intermediate
        } else {
            4 * self.hidden_size
        }
    }

    /// Get the maximum sequence length
    ///
    /// Returns max_seq_length if specified, otherwise defaults to 2048
    pub const fn max_seq_len(&self) -> u32 {
        if let Some(max_len) = self.max_seq_length {
            max_len
        } else {
            2048
        }
    }
}

/// Convert HuggingFace config to Janus ModelConfig
///
/// This maps the HuggingFace config format to Janus's internal format
impl From<&HuggingFaceConfig> for crate::model::ModelConfig {
    fn from(config: &HuggingFaceConfig) -> Self {
        Self {
            hidden_dim: config.hidden_size,
            num_layers: config.num_hidden_layers,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_kv_heads(),
            head_dim: config.head_dim(),
            ffn_dim: config.ffn_dim(),
            vocab_size: config.vocab_size,
            max_seq_len: config.max_seq_len(),
            rms_norm_eps: config.rms_norm_eps,
            rope_freq_base: 10000.0,
            batch_size: 1, // Default to single-sequence inference
        }
    }
}

fn metadata_as_u32(metadata: &GgufMetadata, key: &str) -> Result<u32> {
    let value = metadata
        .metadata
        .get(key)
        .ok_or_else(|| ConfigError::MissingGgufMetadata {
            key: key.to_string(),
        })?;

    let as_u64 = match value {
        MetadataValue::UInt8(v) => *v as u64,
        MetadataValue::UInt16(v) => *v as u64,
        MetadataValue::UInt32(v) => *v as u64,
        MetadataValue::UInt64(v) => *v,
        MetadataValue::Int8(v) if *v >= 0 => *v as u64,
        MetadataValue::Int16(v) if *v >= 0 => *v as u64,
        MetadataValue::Int32(v) if *v >= 0 => *v as u64,
        MetadataValue::Int64(v) if *v >= 0 => *v as u64,
        _ => {
            return Err(ConfigError::InvalidConfig(format!(
                "GGUF metadata '{}' is not a positive integer",
                key
            )));
        }
    };

    u32::try_from(as_u64).map_err(|_| {
        ConfigError::InvalidConfig(format!("GGUF metadata '{}' does not fit in u32", key))
    })
}

fn metadata_as_f32(metadata: &GgufMetadata, key: &str) -> Result<f32> {
    let value = metadata
        .metadata
        .get(key)
        .ok_or_else(|| ConfigError::MissingGgufMetadata {
            key: key.to_string(),
        })?;

    match value {
        MetadataValue::Float32(v) => Ok(*v),
        MetadataValue::Float64(v) => Ok(*v as f32),
        MetadataValue::Int8(v) => Ok(*v as f32),
        MetadataValue::Int16(v) => Ok(*v as f32),
        MetadataValue::Int32(v) => Ok(*v as f32),
        MetadataValue::Int64(v) => Ok(*v as f32),
        MetadataValue::UInt8(v) => Ok(*v as f32),
        MetadataValue::UInt16(v) => Ok(*v as f32),
        MetadataValue::UInt32(v) => Ok(*v as f32),
        MetadataValue::UInt64(v) => Ok(*v as f32),
        _ => Err(ConfigError::InvalidConfig(format!(
            "GGUF metadata '{}' is not numeric",
            key
        ))),
    }
}

fn metadata_array_len_as_u32(metadata: &GgufMetadata, key: &str) -> Option<u32> {
    match metadata.metadata.get(key) {
        Some(MetadataValue::Array(values)) => u32::try_from(values.len()).ok(),
        _ => None,
    }
}

fn metadata_string(metadata: &GgufMetadata, key: &str) -> Option<String> {
    match metadata.metadata.get(key) {
        Some(MetadataValue::Bytes(v)) => String::from_utf8(v.clone()).ok(),
        _ => None,
    }
}

/// Build internal ModelConfig directly from GGUF metadata.
pub fn model_config_from_gguf_metadata(
    metadata: &GgufMetadata,
    tokenizer_vocab_size: u32,
) -> Result<crate::model::ModelConfig> {
    let architecture =
        metadata_string(metadata, "general.architecture").unwrap_or_else(|| "llama".to_string());

    let hidden_dim = metadata_as_u32(metadata, &format!("{}.embedding_length", architecture))?;
    let num_layers = metadata_as_u32(metadata, &format!("{}.block_count", architecture))?;
    let num_heads = metadata_as_u32(metadata, &format!("{}.attention.head_count", architecture))?;
    let num_kv_heads = match metadata_as_u32(
        metadata,
        &format!("{}.attention.head_count_kv", architecture),
    ) {
        Ok(v) => v,
        Err(ConfigError::MissingGgufMetadata { .. }) => num_heads,
        Err(e) => return Err(e),
    };
    let ffn_dim = metadata_as_u32(metadata, &format!("{}.feed_forward_length", architecture))?;
    let max_seq_len = metadata_as_u32(metadata, &format!("{}.context_length", architecture))?;
    let rms_norm_eps = metadata_as_f32(
        metadata,
        &format!("{}.attention.layer_norm_rms_epsilon", architecture),
    )?;
    let rope_freq_base =
        match metadata_as_f32(metadata, &format!("{}.rope.freq_base", architecture)) {
            Ok(v) => v,
            Err(ConfigError::MissingGgufMetadata { .. }) => 10000.0,
            Err(e) => return Err(e),
        };

    let vocab_size = match metadata_as_u32(metadata, &format!("{}.vocab_size", architecture)) {
        Ok(v) => v,
        Err(ConfigError::MissingGgufMetadata { .. }) => {
            metadata_array_len_as_u32(metadata, "tokenizer.ggml.tokens")
                .unwrap_or(tokenizer_vocab_size)
        }
        Err(e) => return Err(e),
    };

    Ok(crate::model::ModelConfig {
        hidden_dim,
        num_layers,
        num_heads,
        num_kv_heads,
        head_dim: hidden_dim / num_heads,
        ffn_dim,
        vocab_size,
        max_seq_len,
        rms_norm_eps,
        rope_freq_base,
        batch_size: 1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_parsing() {
        let json = r#"
        {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "vocab_size": 32000,
            "intermediate_size": 11008,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 0.00001
        }
        "#;

        let config: HuggingFaceConfig = serde_json::from_str(json).unwrap();

        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_hidden_layers, 32);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.head_dim(), 128);
        assert_eq!(config.ffn_dim(), 11008);
    }

    #[test]
    fn test_config_defaults() {
        let json = r#"
        {
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "num_attention_heads": 16,
            "vocab_size": 50000
        }
        "#;

        let config: HuggingFaceConfig = serde_json::from_str(json).unwrap();

        // Test defaults
        assert_eq!(config.num_kv_heads(), 16); // Falls back to num_attention_heads
        assert_eq!(config.ffn_dim(), 8192); // 4 * hidden_size
        assert_eq!(config.max_seq_len(), 2048); // Default
        assert_eq!(config.rms_norm_eps, 1e-5); // Default
    }

    #[test]
    fn test_config_validation() {
        let json = r#"
        {
            "hidden_size": 0,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000
        }
        "#;

        let config: HuggingFaceConfig = serde_json::from_str(json).unwrap();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_head_dim_calculation() {
        let json = r#"
        {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000
        }
        "#;

        let config: HuggingFaceConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.head_dim(), 128); // 4096 / 32
    }

    #[test]
    fn test_to_model_config() {
        let json = r#"
        {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "vocab_size": 32000,
            "intermediate_size": 11008,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 0.00001
        }
        "#;

        let hf_config: HuggingFaceConfig = serde_json::from_str(json).unwrap();
        let model_config: crate::model::ModelConfig = (&hf_config).into();

        assert_eq!(model_config.hidden_dim, 4096);
        assert_eq!(model_config.num_layers, 32);
        assert_eq!(model_config.num_heads, 32);
        assert_eq!(model_config.head_dim, 128);
        assert_eq!(model_config.ffn_dim, 11008);
        assert_eq!(model_config.vocab_size, 32000);
        assert_eq!(model_config.max_seq_len, 2048);
        assert_eq!(model_config.rms_norm_eps, 1e-5);
    }
}

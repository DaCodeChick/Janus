//! Model configuration parsing from HuggingFace config.json
//!
//! This module handles loading model hyperparameters from the standard
//! HuggingFace `config.json` format used by transformers.

use serde::Deserialize;
use std::fs;
use std::path::Path;
use thiserror::Error;

/// Errors that can occur when loading model configuration
#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("Failed to read config file: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Failed to parse config JSON: {0}")]
    ParseError(#[from] serde_json::Error),

    #[error("Missing required field: {0}")]
    MissingField(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
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
    fn validate(&self) -> Result<()> {
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
                "hidden_size ({}) must be divisible by num_attention_heads ({})",
                self.hidden_size, self.num_attention_heads
            )));
        }

        Ok(())
    }

    /// Get the number of key-value heads (for GQA)
    ///
    /// Returns num_key_value_heads if specified, otherwise falls back
    /// to num_attention_heads (standard multi-head attention)
    pub fn num_kv_heads(&self) -> u32 {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Get the head dimension
    ///
    /// Computed as hidden_size / num_attention_heads
    pub fn head_dim(&self) -> u32 {
        self.hidden_size / self.num_attention_heads
    }

    /// Get the feed-forward dimension
    ///
    /// Returns intermediate_size if specified, otherwise computes
    /// the standard 4 * hidden_size (typical for transformers)
    pub fn ffn_dim(&self) -> u32 {
        self.intermediate_size.unwrap_or(4 * self.hidden_size)
    }

    /// Get the maximum sequence length
    ///
    /// Returns max_seq_length if specified, otherwise defaults to 2048
    pub fn max_seq_len(&self) -> u32 {
        self.max_seq_length.unwrap_or(2048)
    }

    /// Convert to Janus ModelConfig
    ///
    /// This maps the HuggingFace config format to Janus's internal format
    pub fn to_model_config(&self) -> crate::model::ModelConfig {
        crate::model::ModelConfig {
            hidden_dim: self.hidden_size,
            num_layers: self.num_hidden_layers,
            num_heads: self.num_attention_heads,
            num_kv_heads: self.num_kv_heads(),
            head_dim: self.head_dim(),
            ffn_dim: self.ffn_dim(),
            vocab_size: self.vocab_size,
            max_seq_len: self.max_seq_len(),
            rms_norm_eps: self.rms_norm_eps,
        }
    }
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
        let model_config = hf_config.to_model_config();

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

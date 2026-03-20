//! Transformer model configuration
//!
//! Defines the architecture parameters for the full transformer model.

use crate::gguf::GGUFFile;
use std::collections::HashMap;

/// Configuration for the full transformer model
#[derive(Debug, Clone)]
pub struct TransformerConfig {
    /// Model vocabulary size
    pub vocab_size: usize,

    /// Hidden dimension size
    pub hidden_dim: usize,

    /// Number of transformer layers
    pub num_layers: usize,

    /// Number of attention heads
    pub num_heads: usize,

    /// Number of key-value heads (for GQA/MQA)
    pub num_kv_heads: usize,

    /// Feed-forward intermediate dimension
    pub intermediate_dim: usize,

    /// Maximum sequence length supported
    pub max_seq_len: usize,

    /// RMSNorm epsilon value
    pub norm_eps: f32,

    /// RoPE theta (base frequency)
    pub rope_theta: f32,
}

impl TransformerConfig {
    /// Create config from GGUF metadata
    pub fn from_gguf(gguf: &GGUFFile) -> Result<Self, String> {
        let metadata = &gguf.metadata().metadata;

        // Helper to extract required metadata
        let get_u64 = |key: &str| -> Result<u64, String> {
            metadata
                .get(key)
                .and_then(|v| {
                    if let crate::gguf::types::MetadataValue::UInt64(val) = v {
                        Some(*val)
                    } else if let crate::gguf::types::MetadataValue::UInt32(val) = v {
                        Some(*val as u64)
                    } else {
                        None
                    }
                })
                .ok_or_else(|| format!("Missing required metadata: {}", key))
        };

        let get_f32 = |key: &str| -> Result<f32, String> {
            metadata
                .get(key)
                .and_then(|v| {
                    if let crate::gguf::types::MetadataValue::Float32(val) = v {
                        Some(*val)
                    } else {
                        None
                    }
                })
                .ok_or_else(|| format!("Missing required metadata: {}", key))
        };

        // Common GGUF metadata keys for LLaMA/Mistral models
        let vocab_size = get_u64("llama.vocab_size").or_else(|_| get_u64("vocab_size"))? as usize;

        let hidden_dim =
            get_u64("llama.embedding_length").or_else(|_| get_u64("embedding_length"))? as usize;

        let num_layers = get_u64("llama.block_count").or_else(|_| get_u64("block_count"))? as usize;

        let num_heads = get_u64("llama.attention.head_count")
            .or_else(|_| get_u64("attention.head_count"))? as usize;

        let num_kv_heads = get_u64("llama.attention.head_count_kv")
            .or_else(|_| get_u64("attention.head_count_kv"))
            .unwrap_or(num_heads as u64) as usize; // Default to num_heads for MHA

        let intermediate_dim = get_u64("llama.feed_forward_length")
            .or_else(|_| get_u64("feed_forward_length"))? as usize;

        let max_seq_len = get_u64("llama.context_length")
            .or_else(|_| get_u64("context_length"))
            .unwrap_or(2048) as usize; // Default to 2048

        let norm_eps = get_f32("llama.attention.layer_norm_rms_epsilon")
            .or_else(|_| get_f32("attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5); // Default epsilon

        let rope_theta = get_f32("llama.rope.freq_base")
            .or_else(|_| get_f32("rope.freq_base"))
            .unwrap_or(10000.0); // Default RoPE base

        Ok(Self {
            vocab_size,
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads,
            intermediate_dim,
            max_seq_len,
            norm_eps,
            rope_theta,
        })
    }

    /// Get head dimension
    pub const fn head_dim(&self) -> usize {
        self.hidden_dim / self.num_heads
    }

    /// Check if using Grouped Query Attention
    pub const fn is_gqa(&self) -> bool {
        self.num_kv_heads < self.num_heads && self.num_kv_heads > 1
    }

    /// Check if using Multi-Query Attention
    pub const fn is_mqa(&self) -> bool {
        self.num_kv_heads == 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_head_dim_calculation() {
        let config = TransformerConfig {
            vocab_size: 32000,
            hidden_dim: 4096,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 32,
            intermediate_dim: 11008,
            max_seq_len: 2048,
            norm_eps: 1e-5,
            rope_theta: 10000.0,
        };

        assert_eq!(config.head_dim(), 128);
        assert!(!config.is_gqa());
        assert!(!config.is_mqa());
    }

    #[test]
    fn test_gqa_detection() {
        let config = TransformerConfig {
            vocab_size: 32000,
            hidden_dim: 4096,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8, // GQA
            intermediate_dim: 11008,
            max_seq_len: 2048,
            norm_eps: 1e-5,
            rope_theta: 10000.0,
        };

        assert!(config.is_gqa());
        assert!(!config.is_mqa());
    }

    #[test]
    fn test_mqa_detection() {
        let config = TransformerConfig {
            vocab_size: 32000,
            hidden_dim: 4096,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 1, // MQA
            intermediate_dim: 11008,
            max_seq_len: 2048,
            norm_eps: 1e-5,
            rope_theta: 10000.0,
        };

        assert!(!config.is_gqa());
        assert!(config.is_mqa());
    }
}

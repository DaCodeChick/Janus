//! Output layer for language model head
//!
//! This module implements the final projection from hidden states to vocabulary logits.

/// Language model head configuration
#[derive(Debug, Clone)]
pub struct LMHeadConfig {
    /// Hidden dimension (model dimension)
    pub hidden_dim: u32,
    /// Vocabulary size
    pub vocab_size: u32,
    /// RMSNorm epsilon for output normalization
    pub rms_norm_eps: f32,
}

/// Language model head (final output layer)
///
/// The LM head performs the final transformation from hidden states to vocabulary logits:
/// 1. Apply RMSNorm to the final hidden state
/// 2. Project to vocabulary size using a linear layer (matrix multiplication)
///
/// The output is a vector of logits (one per vocabulary token) representing
/// the model's predicted probability distribution for the next token.
pub struct LMHead {
    config: LMHeadConfig,
    /// Output normalization weight [hidden_dim]
    output_norm_weight: wgpu::Buffer,
    /// Output projection weight [hidden_dim × vocab_size]
    output_weight: wgpu::Buffer,
}

impl LMHead {
    /// Create a new language model head
    ///
    /// # Arguments
    /// * `config` - Configuration for the LM head
    /// * `output_norm_weight` - Weight buffer for final RMSNorm
    /// * `output_weight` - Weight buffer for vocabulary projection
    pub fn new(
        config: LMHeadConfig,
        output_norm_weight: wgpu::Buffer,
        output_weight: wgpu::Buffer,
    ) -> Self {
        Self {
            config,
            output_norm_weight,
            output_weight,
        }
    }

    /// Get the configuration
    pub const fn config(&self) -> &LMHeadConfig {
        &self.config
    }

    /// Get the vocabulary size
    pub const fn vocab_size(&self) -> u32 {
        self.config.vocab_size
    }
}

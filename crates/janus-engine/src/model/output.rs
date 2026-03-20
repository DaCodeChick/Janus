//! Output layer for language model head
//!
//! This module implements the final projection from hidden states to vocabulary logits.

use crate::compute::ops::{gemm, rmsnorm};
use crate::compute::{ComputeEngine, Result};

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
    #[allow(dead_code)] // TODO: RMSNorm weights will be used in future optimization
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

    /// Compute logits from final hidden state
    ///
    /// This performs the complete forward pass of the LM head:
    /// 1. RMSNorm(hidden_state) → normalized hidden state
    /// 2. GEMM(normalized, output_weight) → logits [vocab_size]
    ///
    /// # Arguments
    /// * `engine` - The compute engine for GPU operations
    /// * `hidden_state` - Final hidden state from the last transformer block [hidden_dim]
    ///
    /// # Returns
    /// GPU buffer containing logits [vocab_size] - unnormalized scores for each vocabulary token
    ///
    /// # Note
    /// The logits are NOT softmax-normalized. The sampler will handle token selection
    /// directly from these raw logits (using argmax for greedy, or applying temperature
    /// and softmax for sampling).
    pub async fn forward(
        &self,
        engine: &ComputeEngine,
        hidden_state: &wgpu::Buffer,
    ) -> Result<wgpu::Buffer> {
        // Step 1: Apply output normalization (RMSNorm)
        let normalized = rmsnorm(
            engine,
            hidden_state,
            self.config.hidden_dim,
            self.config.rms_norm_eps,
        )
        .await?;

        // Step 2: Project to vocabulary size
        // output_weight is [hidden_dim × vocab_size], so we do:
        // logits = normalized^T * output_weight
        // This is a matrix-vector multiplication producing [vocab_size] output
        let logits = gemm(
            engine,
            &normalized,
            &self.output_weight,
            1,
            self.config.hidden_dim,
            self.config.vocab_size,
        )
        .await?;

        Ok(logits)
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

//! Complete transformer model with autoregressive text generation
//!
//! This module implements the top-level `Model` struct that stitches together
//! all components for end-to-end LLM inference.
//!
//! # Example Usage
//!
//! ```no_run
//! use janus_engine::{ComputeEngine, Model, ModelConfig, Sampler, Tokenizer};
//! use janus_engine::model::TransformerBlock;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // 1. Initialize GPU compute engine
//!     let engine = ComputeEngine::new().await?;
//!     
//!     // 2. Load tokenizer
//!     let tokenizer = Tokenizer::from_file("tokenizer.json")?;
//!     
//!     // 3. Create sampler (greedy decoding)
//!     let sampler = Sampler::greedy(32000);
//!     
//!     // 4. Configure model architecture
//!     let config = ModelConfig {
//!         hidden_dim: 4096,
//!         num_layers: 32,
//!         num_heads: 32,
//!         num_kv_heads: 32, // For MHA; use fewer for GQA (e.g., 4 for TinyLlama)
//!         head_dim: 128,
//!         ffn_dim: 11008,
//!         vocab_size: 32000,
//!         max_seq_len: 2048,
//!         rms_norm_eps: 1e-5,
//!     };
//!     
//!     // 5. Load weights from GGUF file (pseudocode - actual loader TBD)
//!     // let token_embedding_table = load_tensor(&engine, "token_embd.weight");
//!     // let blocks = load_blocks(&engine, &config);
//!     // let output_norm_weight = load_tensor(&engine, "output_norm.weight");
//!     // let lm_head_weight = load_tensor(&engine, "lm_head.weight");
//!     
//!     // 6. Create model
//!     // let mut model = Model::new(
//!     //     config,
//!     //     engine,
//!     //     tokenizer,
//!     //     sampler,
//!     //     token_embedding_table,
//!     //     blocks,
//!     //     output_norm_weight,
//!     //     lm_head_weight,
//!     // )?;
//!     
//!     // 7. Generate text
//!     // let output = model.generate("Once upon a time", 100).await?;
//!     // println!("Generated: {}", output);
//!     
//!     Ok(())
//! }
//! ```

mod constructor;
mod embedding;
mod forward;
mod generation;
mod speculative_helpers;

#[cfg(test)]
mod tests;

use crate::compute::cache::KVCache;
use crate::compute::{ComputeEngine, PipelineCache, Result};
use crate::model::{block::TransformerBlock, sampler::Sampler, tokenizer::Tokenizer};
use wgpu::Buffer;

/// Configuration for the full transformer model
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Model hidden dimension (e.g., 4096 for LLaMA-7B)
    pub hidden_dim: u32,
    /// Number of transformer layers (e.g., 32 for LLaMA-7B)
    pub num_layers: u32,
    /// Number of attention heads (e.g., 32 for LLaMA-7B)
    pub num_heads: u32,
    /// Number of key-value heads for GQA (e.g., 4 for TinyLlama, 32 for MHA)
    pub num_kv_heads: u32,
    /// Dimension of each attention head
    pub head_dim: u32,
    /// Feed-forward intermediate dimension (e.g., 11008 for LLaMA-7B)
    pub ffn_dim: u32,
    /// Vocabulary size (e.g., 32000 for LLaMA)
    pub vocab_size: u32,
    /// Maximum sequence length (e.g., 2048)
    pub max_seq_len: u32,
    /// RMSNorm epsilon
    pub rms_norm_eps: f32,
    /// Batch size for parallel sequence processing (default: 1)
    /// 
    /// Number of sequences to process in parallel. Higher batch sizes improve
    /// throughput at the cost of increased VRAM usage.
    /// 
    /// Recommended values:
    /// - 1: Single-sequence inference (lowest latency, lowest VRAM)
    /// - 4-8: Good balance for multi-user serving
    /// - 16+: Maximum throughput (requires significant VRAM)
    pub batch_size: u32,
}

/// Complete transformer model for text generation
///
/// This struct holds all components needed for autoregressive LLM inference:
/// - Token embeddings
/// - Transformer blocks (layers)
/// - Output projection (LM head)
/// - KV cache for efficient generation
/// - Tokenizer and sampler
/// - Pre-allocated scratch buffers for static computation graph
pub struct Model {
    /// Configuration
    config: ModelConfig,

    /// Compute engine for GPU operations
    engine: ComputeEngine,

    /// Tokenizer for encoding/decoding text
    tokenizer: Tokenizer,

    /// Sampler for token selection
    sampler: Sampler,

    /// Token embedding table [vocab_size, hidden_dim]
    token_embedding_table: Buffer,

    /// Stack of transformer blocks (hidden layers)
    blocks: Vec<TransformerBlock>,

    /// Final RMSNorm weight [hidden_dim]
    output_norm_weight: Buffer,

    /// LM head weight [hidden_dim, vocab_size]
    lm_head_weight: Buffer,

    /// KV cache for efficient autoregressive generation
    cache: KVCache,

    // === Static Computation Graph: Pre-allocated Scratch Buffers ===
    /// Hidden state buffer (ping-pong buffer A) [hidden_dim]
    hidden_state: Buffer,

    /// Alternate hidden state buffer (ping-pong buffer B) [hidden_dim]
    hidden_state_alt: Buffer,

    /// Query projection buffer [num_heads * head_dim]
    q_buf: Buffer,

    /// Key projection buffer [num_kv_heads * head_dim]
    k_buf: Buffer,

    /// Value projection buffer [num_kv_heads * head_dim]
    v_buf: Buffer,

    /// Rotated query buffer (after RoPE) [num_heads * head_dim]
    q_rot_buf: Buffer,

    /// Rotated key buffer (after RoPE) [num_kv_heads * head_dim]
    k_rot_buf: Buffer,

    /// Attention output buffer [num_heads * head_dim]
    attn_out_buf: Buffer,

    /// Gate projection buffer [ffn_dim]
    gate_buf: Buffer,

    /// Up projection buffer [ffn_dim]
    up_buf: Buffer,

    /// Logits output buffer [vocab_size]
    logits_buf: Buffer,

    // === Additional scratch buffers for TransformerBlock ===
    /// Normalized input buffer for attention [hidden_dim]
    scratch_input_norm: Buffer,

    /// Projection output buffer [hidden_dim]
    scratch_proj_out: Buffer,

    /// First residual connection buffer [hidden_dim]
    scratch_hidden1: Buffer,

    /// FFN normalized input buffer [hidden_dim]
    scratch_ffn_norm: Buffer,

    /// SwiGLU output buffer [ffn_dim]
    scratch_swiglu: Buffer,

    /// FFN output buffer [hidden_dim]
    scratch_ffn_out: Buffer,

    // === Attention intermediate buffers ===
    /// Attention scores buffer (before softmax) [num_heads * max_seq_len]
    scores_buf: Buffer,

    /// Attention probabilities buffer (after softmax) [num_heads * max_seq_len]
    probs_buf: Buffer,

    // === RoPE Cache ===
    /// Pre-computed sin/cos values for RoPE [max_seq_len * head_dim]
    /// Layout: [position * head_dim + dim] contains (cos, sin) pair
    /// Eliminates expensive trigonometric computations during inference
    rope_cache: Buffer,

    // === Pipeline Cache: Pre-compiled Shaders and Pipelines ===
    /// Cached GPU pipelines for all operations (eliminates recompilation overhead)
    pipeline_cache: PipelineCache,
}

impl Model {
    /// Get model configuration
    pub const fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Get reference to compute engine
    pub const fn engine(&self) -> &ComputeEngine {
        &self.engine
    }

    /// Get reference to tokenizer
    pub const fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Get reference to KV cache
    pub const fn cache(&self) -> &KVCache {
        &self.cache
    }

    /// Reset the KV cache (for starting a new generation session)
    pub fn reset_cache(&mut self) {
        self.cache.reset();
    }

    /// Get a reference to the logits buffer
    ///
    /// This buffer contains the output logits from the most recent forward pass.
    /// The buffer has size [vocab_size] and contains f32 values.
    ///
    /// # Returns
    /// Reference to the logits buffer on GPU
    pub fn logits_buffer(&self) -> &wgpu::Buffer {
        &self.logits_buf
    }
    
    /// Validate model configuration for consistency
    fn validate_config(config: &ModelConfig) -> Result<()> {
        // Check basic constraints
        if config.hidden_dim == 0 {
            return Err(crate::compute::ComputeError::InvalidDimensions(
                "hidden_dim must be greater than 0".into()
            ));
        }
        
        if config.num_layers == 0 {
            return Err(crate::compute::ComputeError::InvalidDimensions(
                "num_layers must be greater than 0".into()
            ));
        }
        
        if config.num_heads == 0 {
            return Err(crate::compute::ComputeError::InvalidDimensions(
                "num_heads must be greater than 0".into()
            ));
        }
        
        if config.num_kv_heads == 0 {
            return Err(crate::compute::ComputeError::InvalidDimensions(
                "num_kv_heads must be greater than 0".into()
            ));
        }
        
        if config.vocab_size == 0 {
            return Err(crate::compute::ComputeError::InvalidDimensions(
                "vocab_size must be greater than 0".into()
            ));
        }
        
        // Validate head_dim consistency
        let computed_head_dim = config.hidden_dim / config.num_heads;
        if computed_head_dim != config.head_dim {
            return Err(crate::compute::ComputeError::InvalidDimensions(format!(
                "head_dim mismatch: config specifies {}, but hidden_dim / num_heads = {} / {} = {}\n\nSuggestions:\n  - Set head_dim to {} in your config\n  - Or adjust hidden_dim to be {} (num_heads × head_dim)",
                config.head_dim,
                config.hidden_dim,
                config.num_heads,
                computed_head_dim,
                computed_head_dim,
                config.num_heads * config.head_dim
            )));
        }
        
        // Validate hidden_dim is divisible by num_heads
        if config.hidden_dim % config.num_heads != 0 {
            return Err(crate::compute::ComputeError::InvalidDimensions(format!(
                "hidden_dim ({}) must be divisible by num_heads ({})\nhead_dim would be {}.{} which is not an integer",
                config.hidden_dim,
                config.num_heads,
                config.hidden_dim / config.num_heads,
                config.hidden_dim % config.num_heads
            )));
        }
        
        // Validate GQA configuration
        if config.num_heads % config.num_kv_heads != 0 {
            return Err(crate::compute::ComputeError::InvalidDimensions(format!(
                "num_heads ({}) must be divisible by num_kv_heads ({})\nFor Grouped Query Attention, each KV head must serve equal Q heads\nQ heads per KV head would be {}.{}",
                config.num_heads,
                config.num_kv_heads,
                config.num_heads / config.num_kv_heads,
                config.num_heads % config.num_kv_heads
            )));
        }
        
        // Validate reasonable ranges
        if config.max_seq_len == 0 {
            return Err(crate::compute::ComputeError::InvalidDimensions(
                "max_seq_len must be greater than 0".into()
            ));
        }
        
        if config.max_seq_len > 131072 {
            tracing::warn!(
                "max_seq_len ({}) is very large (>128K). This will use significant memory.",
                config.max_seq_len
            );
        }
        
        // Validate batch_size
        if config.batch_size == 0 {
            return Err(crate::compute::ComputeError::InvalidDimensions(
                "batch_size must be at least 1".into()
            ));
        }
        
        if config.batch_size > 64 {
            tracing::warn!(
                "batch_size ({}) is very large (>64). This will use significant VRAM. Consider reducing for better memory efficiency.",
                config.batch_size
            );
        }
        
        Ok(())
    }
}

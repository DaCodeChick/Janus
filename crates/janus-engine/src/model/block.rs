//! Transformer block implementation
//!
//! This module implements a single transformer block layer following the
//! LLaMA/Mistral architecture with:
//! - Multi-head self-attention with RoPE
//! - SwiGLU feed-forward network
//! - RMSNorm for layer normalization
//! - Residual connections

use crate::compute::cache::KVCache;
use crate::compute::{ComputeEngine, Result};
use crate::compute::ops::{
    add_tensors, compute_attention, elementwise_mul, gemm, rmsnorm, rope, silu,
};

/// Configuration for a transformer block
#[derive(Debug, Clone)]
pub struct TransformerBlockConfig {
    /// Model hidden dimension (e.g., 4096 for LLaMA-7B)
    pub hidden_dim: u32,
    /// Number of attention heads (e.g., 32 for LLaMA-7B)
    pub num_heads: u32,
    /// Number of key-value heads for GQA (e.g., 4 for TinyLlama)
    pub num_kv_heads: u32,
    /// Dimension of each attention head (typically hidden_dim / num_heads)
    pub head_dim: u32,
    /// Feed-forward intermediate dimension (typically 4 * hidden_dim)
    pub ffn_dim: u32,
    /// RMSNorm epsilon for numerical stability
    pub rms_norm_eps: f32,
}

/// A single transformer block layer
///
/// This struct holds GPU buffer references to all the weights for one layer
/// of a transformer model. Each weight tensor is stored in VRAM.
pub struct TransformerBlock {
    config: TransformerBlockConfig,

    // Attention weights
    /// Query projection weight [hidden_dim × hidden_dim]
    attn_q_weight: wgpu::Buffer,
    /// Key projection weight [hidden_dim × hidden_dim]
    attn_k_weight: wgpu::Buffer,
    /// Value projection weight [hidden_dim × hidden_dim]
    attn_v_weight: wgpu::Buffer,
    /// Output projection weight [hidden_dim × hidden_dim]
    attn_output_weight: wgpu::Buffer,

    // Feed-forward network weights
    /// Gate projection weight [hidden_dim × ffn_dim]
    ffn_gate_weight: wgpu::Buffer,
    /// Up projection weight [hidden_dim × ffn_dim]
    ffn_up_weight: wgpu::Buffer,
    /// Down projection weight [ffn_dim × hidden_dim]
    ffn_down_weight: wgpu::Buffer,

    // Normalization weights
    /// Attention input normalization weight [hidden_dim]
    #[allow(dead_code)] // TODO: RMSNorm weights will be used in future optimization
    attn_norm_weight: wgpu::Buffer,
    /// FFN input normalization weight [hidden_dim]
    #[allow(dead_code)] // TODO: RMSNorm weights will be used in future optimization
    ffn_norm_weight: wgpu::Buffer,
}

impl TransformerBlock {
    /// Create a new transformer block with the given weights
    ///
    /// # Arguments
    /// * `config` - Configuration parameters for this block
    /// * `attn_q_weight` - Query projection weight buffer
    /// * `attn_k_weight` - Key projection weight buffer
    /// * `attn_v_weight` - Value projection weight buffer
    /// * `attn_output_weight` - Attention output projection weight buffer
    /// * `ffn_gate_weight` - FFN gate projection weight buffer
    /// * `ffn_up_weight` - FFN up projection weight buffer
    /// * `ffn_down_weight` - FFN down projection weight buffer
    /// * `attn_norm_weight` - Attention normalization weight buffer
    /// * `ffn_norm_weight` - FFN normalization weight buffer
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: TransformerBlockConfig,
        attn_q_weight: wgpu::Buffer,
        attn_k_weight: wgpu::Buffer,
        attn_v_weight: wgpu::Buffer,
        attn_output_weight: wgpu::Buffer,
        ffn_gate_weight: wgpu::Buffer,
        ffn_up_weight: wgpu::Buffer,
        ffn_down_weight: wgpu::Buffer,
        attn_norm_weight: wgpu::Buffer,
        ffn_norm_weight: wgpu::Buffer,
    ) -> Self {
        Self {
            config,
            attn_q_weight,
            attn_k_weight,
            attn_v_weight,
            attn_output_weight,
            ffn_gate_weight,
            ffn_up_weight,
            ffn_down_weight,
            attn_norm_weight,
            ffn_norm_weight,
        }
    }

    /// Execute the forward pass of this transformer block
    ///
    /// This implements the full transformer block computation pipeline:
    /// 1. Attention block with residual:
    ///    - RMSNorm(input)
    ///    - Multi-head attention with RoPE
    ///    - Residual connection: input + attention_output
    /// 2. Feed-forward block with residual:
    ///    - RMSNorm(hidden_1)
    ///    - SwiGLU: SiLU(gate) * up
    ///    - Down projection
    ///    - Residual connection: hidden_1 + ffn_output
    ///
    /// # Arguments
    /// * `engine` - The compute engine for GPU operations
    /// * `input` - Input tensor [hidden_dim] for the current token
    /// * `cache` - KV cache for storing/retrieving attention keys and values
    /// * `seq_pos` - Position in the sequence (for RoPE)
    ///
    /// # Returns
    /// Output tensor [hidden_dim] after passing through this block
    pub async fn forward(
        &self,
        engine: &ComputeEngine,
        input: &wgpu::Buffer,
        cache: &mut KVCache,
        seq_pos: u32,
    ) -> Result<wgpu::Buffer> {
        // ===================================================================
        // ATTENTION BLOCK
        // ===================================================================

        // Step 1: Input normalization
        let input_norm = rmsnorm(
            engine,
            input,
            self.config.hidden_dim,
            self.config.rms_norm_eps,
        )
        .await?;

        // Step 2: Compute Q, K, V projections
        // For GQA: Q uses full hidden_dim, but K/V use num_kv_heads * head_dim
        let kv_dim = self.config.num_kv_heads * self.config.head_dim;
        
        let q = gemm(
            engine,
            &input_norm,
            &self.attn_q_weight,
            1,
            self.config.hidden_dim,
            self.config.hidden_dim,
        )
        .await?;

        let k = gemm(
            engine,
            &input_norm,
            &self.attn_k_weight,
            1,
            self.config.hidden_dim,
            kv_dim,
        )
        .await?;

        let v = gemm(
            engine,
            &input_norm,
            &self.attn_v_weight,
            1,
            self.config.hidden_dim,
            kv_dim,
        )
        .await?;

        // Step 3: Apply RoPE to Q and K
        let q_rot = rope(
            engine,
            &q,
            self.config.num_heads,
            self.config.head_dim,
            seq_pos,
            10000.0, // theta_base (standard value for LLaMA)
        )
        .await?;

        let k_rot = rope(
            engine,
            &k,
            self.config.num_kv_heads,
            self.config.head_dim,
            seq_pos,
            10000.0, // theta_base (standard value for LLaMA)
        )
        .await?;

        // Step 4: Update KV cache with new K and V
        cache.update(engine, &k_rot, &v, seq_pos).await?;

        // Step 5: Compute attention using cached K and V
        let (key_cache, value_cache) = cache.buffers();
        let current_seq_len = seq_pos + 1; // Sequence length including current token

        let attn_out = compute_attention(
            engine,
            &q_rot,
            key_cache,
            value_cache,
            current_seq_len,
            self.config.num_heads,
            self.config.num_kv_heads,
            self.config.head_dim,
        )
        .await?;

        // Step 6: Output projection
        let proj_out = gemm(
            engine,
            &attn_out,
            &self.attn_output_weight,
            1,
            self.config.hidden_dim,
            self.config.hidden_dim,
        )
        .await?;

        // Step 7: Residual connection 1
        let hidden_1 = add_tensors(engine, input, &proj_out, self.config.hidden_dim).await?;

        // ===================================================================
        // FEED-FORWARD NETWORK BLOCK
        // ===================================================================

        // Step 8: FFN input normalization
        let ffn_norm = rmsnorm(
            engine,
            &hidden_1,
            self.config.hidden_dim,
            self.config.rms_norm_eps,
        )
        .await?;

        // Step 9: Gate projection and activation (SiLU)
        let gate_proj = gemm(
            engine,
            &ffn_norm,
            &self.ffn_gate_weight,
            1,
            self.config.hidden_dim,
            self.config.ffn_dim,
        )
        .await?;

        let gate = silu(engine, &gate_proj, self.config.ffn_dim).await?;

        // Step 10: Up projection
        let up = gemm(
            engine,
            &ffn_norm,
            &self.ffn_up_weight,
            1,
            self.config.hidden_dim,
            self.config.ffn_dim,
        )
        .await?;

        // Step 11: Element-wise multiply gate and up (SwiGLU)
        let swiglu = elementwise_mul(engine, &gate, &up, self.config.ffn_dim).await?;

        // Step 12: Down projection
        let ffn_out = gemm(
            engine,
            &swiglu,
            &self.ffn_down_weight,
            1,
            self.config.ffn_dim,
            self.config.hidden_dim,
        )
        .await?;

        // Step 13: Residual connection 2
        let final_out = add_tensors(engine, &hidden_1, &ffn_out, self.config.hidden_dim).await?;

        Ok(final_out)
    }

    /// Get a reference to the block configuration
    pub const fn config(&self) -> &TransformerBlockConfig {
        &self.config
    }
}

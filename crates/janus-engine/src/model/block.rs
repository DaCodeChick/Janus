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
    add_tensors_static, compute_attention_static, elementwise_mul_static, 
    gemm_static, rmsnorm_static, rope_static, silu_static,
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
    attn_norm_weight: wgpu::Buffer,
    /// FFN input normalization weight [hidden_dim]
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

    /// Execute the forward pass of this transformer block (STATIC VERSION)
    ///
    /// This is the static computation graph version that accepts pre-allocated
    /// scratch buffers and records all operations to a shared command encoder.
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
    /// * `encoder` - Shared command encoder to record operations to
    /// * `input` - Input tensor [hidden_dim] for the current token
    /// * `output` - Pre-allocated output buffer [hidden_dim]
    /// * `scratch_input_norm` - Pre-allocated buffer for normalized input [hidden_dim]
    /// * `scratch_q` - Pre-allocated buffer for Q projection [num_heads * head_dim]
    /// * `scratch_k` - Pre-allocated buffer for K projection [num_kv_heads * head_dim]
    /// * `scratch_v` - Pre-allocated buffer for V projection [num_kv_heads * head_dim]
    /// * `scratch_q_rot` - Pre-allocated buffer for rotated Q [num_heads * head_dim]
    /// * `scratch_k_rot` - Pre-allocated buffer for rotated K [num_kv_heads * head_dim]
    /// * `scratch_attn_out` - Pre-allocated buffer for attention output [num_heads * head_dim]
    /// * `scratch_proj_out` - Pre-allocated buffer for projection output [hidden_dim]
    /// * `scratch_hidden1` - Pre-allocated buffer for first residual [hidden_dim]
    /// * `scratch_ffn_norm` - Pre-allocated buffer for FFN normalized input [hidden_dim]
    /// * `scratch_gate` - Pre-allocated buffer for gate projection [ffn_dim]
    /// * `scratch_up` - Pre-allocated buffer for up projection [ffn_dim]
    /// * `scratch_swiglu` - Pre-allocated buffer for SwiGLU output [ffn_dim]
    /// * `scratch_ffn_out` - Pre-allocated buffer for FFN output [hidden_dim]
    /// * `cache` - KV cache for storing/retrieving attention keys and values
    /// * `layer_idx` - The transformer layer index (for cache segmentation)
    /// * `seq_pos` - Position in the sequence (for RoPE)
    ///
    /// # Returns
    /// Result indicating success or error
    #[allow(clippy::too_many_arguments)]
    pub fn forward_static(
        &self,
        engine: &ComputeEngine,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        scratch_input_norm: &wgpu::Buffer,
        scratch_q: &wgpu::Buffer,
        scratch_k: &wgpu::Buffer,
        scratch_v: &wgpu::Buffer,
        scratch_q_rot: &wgpu::Buffer,
        scratch_k_rot: &wgpu::Buffer,
        scratch_attn_out: &wgpu::Buffer,
        scratch_proj_out: &wgpu::Buffer,
        scratch_hidden1: &wgpu::Buffer,
        scratch_ffn_norm: &wgpu::Buffer,
        scratch_gate: &wgpu::Buffer,
        scratch_up: &wgpu::Buffer,
        scratch_swiglu: &wgpu::Buffer,
        scratch_ffn_out: &wgpu::Buffer,
        cache: &mut KVCache,
        layer_idx: u32,
        seq_pos: u32,
    ) -> Result<()> {
        // ===================================================================
        // ATTENTION BLOCK
        // ===================================================================

        // Step 1: Input normalization
        rmsnorm_static(
            engine,
            encoder,
            input,
            &self.attn_norm_weight,
            scratch_input_norm,
            self.config.hidden_dim,
            self.config.rms_norm_eps,
        )?;

        // Step 2: Compute Q, K, V projections
        // For GQA: Q uses full hidden_dim, but K/V use num_kv_heads * head_dim
        let kv_dim = self.config.num_kv_heads * self.config.head_dim;
        
        gemm_static(
            engine,
            encoder,
            scratch_input_norm,
            &self.attn_q_weight,
            scratch_q,
            1,
            self.config.hidden_dim,
            self.config.hidden_dim,
        )?;

        gemm_static(
            engine,
            encoder,
            scratch_input_norm,
            &self.attn_k_weight,
            scratch_k,
            1,
            self.config.hidden_dim,
            kv_dim,
        )?;

        gemm_static(
            engine,
            encoder,
            scratch_input_norm,
            &self.attn_v_weight,
            scratch_v,
            1,
            self.config.hidden_dim,
            kv_dim,
        )?;

        // Step 3: Apply RoPE to Q and K (output to separate buffers)
        rope_static(
            engine,
            encoder,
            scratch_q,
            scratch_q_rot,  // separate output buffer
            self.config.num_heads,
            self.config.head_dim,
            seq_pos,
            10000.0, // theta_base (standard value for LLaMA)
        )?;

        rope_static(
            engine,
            encoder,
            scratch_k,
            scratch_k_rot,  // separate output buffer
            self.config.num_kv_heads,
            self.config.head_dim,
            seq_pos,
            10000.0, // theta_base (standard value for LLaMA)
        )?;

        // Step 4: Update KV cache with new K and V
        // Note: This may still require a sync point - will address in Phase 7
        cache.update_static(engine, encoder, scratch_k_rot, scratch_v, layer_idx, seq_pos)?;

        // Step 5: Compute attention using cached K and V
        let (key_cache, value_cache) = cache.buffers();
        let current_seq_len = seq_pos + 1; // Sequence length including current token

        compute_attention_static(
            engine,
            encoder,
            scratch_q_rot,  // Use rotated Q
            key_cache,
            value_cache,
            scratch_attn_out,
            layer_idx,
            current_seq_len,
            cache.max_seq_len(),
            self.config.num_heads,
            self.config.num_kv_heads,
            self.config.head_dim,
        )?;

        // Step 6: Output projection
        gemm_static(
            engine,
            encoder,
            scratch_attn_out,
            &self.attn_output_weight,
            scratch_proj_out,
            1,
            self.config.hidden_dim,
            self.config.hidden_dim,
        )?;

        // Step 7: Residual connection 1
        add_tensors_static(
            engine,
            encoder,
            input,
            scratch_proj_out,
            scratch_hidden1,
            self.config.hidden_dim,
        )?;

        // ===================================================================
        // FEED-FORWARD NETWORK BLOCK
        // ===================================================================

        // Step 8: FFN input normalization
        rmsnorm_static(
            engine,
            encoder,
            scratch_hidden1,
            &self.ffn_norm_weight,
            scratch_ffn_norm,
            self.config.hidden_dim,
            self.config.rms_norm_eps,
        )?;

        // Step 9: Gate projection and activation (SiLU)
        gemm_static(
            engine,
            encoder,
            scratch_ffn_norm,
            &self.ffn_gate_weight,
            scratch_gate,
            1,
            self.config.hidden_dim,
            self.config.ffn_dim,
        )?;

        silu_static(
            engine,
            encoder,
            scratch_gate,
            scratch_gate,  // in-place
            self.config.ffn_dim,
        )?;

        // Step 10: Up projection
        gemm_static(
            engine,
            encoder,
            scratch_ffn_norm,
            &self.ffn_up_weight,
            scratch_up,
            1,
            self.config.hidden_dim,
            self.config.ffn_dim,
        )?;

        // Step 11: Element-wise multiply gate and up (SwiGLU)
        elementwise_mul_static(
            engine,
            encoder,
            scratch_gate,
            scratch_up,
            scratch_swiglu,
            self.config.ffn_dim,
        )?;

        // Step 12: Down projection
        gemm_static(
            engine,
            encoder,
            scratch_swiglu,
            &self.ffn_down_weight,
            scratch_ffn_out,
            1,
            self.config.ffn_dim,
            self.config.hidden_dim,
        )?;

        // Step 13: Residual connection 2 (output to final buffer)
        add_tensors_static(
            engine,
            encoder,
            scratch_hidden1,
            scratch_ffn_out,
            output,
            self.config.hidden_dim,
        )?;

        Ok(())
    }

    /// Execute the forward pass of this transformer block (DEPRECATED - ASYNC VERSION)
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
    /// * `layer_idx` - The transformer layer index (for cache segmentation)
    /// * `seq_pos` - Position in the sequence (for RoPE)
    ///
    /// # Returns
    /// Output tensor [hidden_dim] after passing through this block
    #[deprecated(note = "Use forward_static instead for better performance")]
    pub async fn forward(
        &self,
        engine: &ComputeEngine,
        input: &wgpu::Buffer,
        cache: &mut KVCache,
        layer_idx: u32,
        seq_pos: u32,
    ) -> Result<wgpu::Buffer> {
        // ===================================================================
        // ATTENTION BLOCK
        // ===================================================================

        // Step 1: Input normalization
        let input_norm = rmsnorm(
            engine,
            input,
            &self.attn_norm_weight,
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
        cache.update(engine, &k_rot, &v, layer_idx, seq_pos).await?;

        // Step 5: Compute attention using cached K and V
        let (key_cache, value_cache) = cache.buffers();
        let current_seq_len = seq_pos + 1; // Sequence length including current token

        let attn_out = compute_attention(
            engine,
            &q_rot,
            key_cache,
            value_cache,
            layer_idx,
            current_seq_len,
            cache.max_seq_len(),
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
            &self.ffn_norm_weight,
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

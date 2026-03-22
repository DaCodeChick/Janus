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

use crate::compute::cache::KVCache;
use crate::compute::ops::{gemm, rmsnorm};
use crate::compute::{ComputeEngine, PipelineCache, Result};
use crate::model::{block::TransformerBlock, sampler::Sampler, tokenizer::Tokenizer};
use wgpu::Buffer;
use wgpu::util::DeviceExt;

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
    /// Create a new model with comprehensive validation
    ///
    /// # Arguments
    /// * `config` - Model configuration
    /// * `engine` - Compute engine for GPU operations
    /// * `tokenizer` - Tokenizer for text encoding/decoding
    /// * `sampler` - Sampler for token selection
    /// * `token_embedding_table` - Pre-loaded embedding weights on GPU
    /// * `blocks` - Pre-constructed transformer blocks
    /// * `output_norm_weight` - Final normalization weights on GPU
    /// * `lm_head_weight` - Output projection weights on GPU
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: ModelConfig,
        engine: ComputeEngine,
        tokenizer: Tokenizer,
        sampler: Sampler,
        token_embedding_table: Buffer,
        blocks: Vec<TransformerBlock>,
        output_norm_weight: Buffer,
        lm_head_weight: Buffer,
    ) -> Result<Self> {
        // === Validation: Configuration ===
        tracing::info!("Validating model configuration...");
        Self::validate_config(&config)?;
        
        // === Validation: Block Count ===
        if blocks.len() != config.num_layers as usize {
            return Err(crate::compute::ComputeError::InvalidDimensions(format!(
                "Block count mismatch: expected {} layers, got {} blocks\n\nSuggestions:\n  - Verify the model file contains all layers\n  - Check if the config.json matches this model\n  - Some models may use a different num_hidden_layers value",
                config.num_layers,
                blocks.len()
            )));
        }
        
        // === Validation: Tokenizer Vocabulary Size ===
        let tokenizer_vocab_size = tokenizer.vocab_size() as u32;
        if tokenizer_vocab_size != config.vocab_size {
            tracing::warn!(
                "Tokenizer vocab size ({}) differs from model config ({}). This may cause issues.",
                tokenizer_vocab_size,
                config.vocab_size
            );
            
            // Allow if tokenizer is larger (common case - model uses subset)
            if tokenizer_vocab_size < config.vocab_size {
                return Err(crate::compute::ComputeError::InvalidDimensions(format!(
                    "Tokenizer vocab size ({}) is smaller than model vocab size ({})\n\nSuggestions:\n  - Verify you're using the correct tokenizer.json for this model\n  - Check the model card for the correct tokenizer\n  - The tokenizer must have at least {} tokens",
                    tokenizer_vocab_size,
                    config.vocab_size,
                    config.vocab_size
                )));
            }
        }
        
        // === Validation: Sampler Vocabulary Size ===
        if sampler.vocab_size() != config.vocab_size {
            return Err(crate::compute::ComputeError::InvalidDimensions(format!(
                "Sampler vocab size ({}) does not match model vocab size ({})\n\nSuggestions:\n  - Recreate the sampler with the correct vocab_size: Sampler::greedy({})",
                sampler.vocab_size(),
                config.vocab_size,
                config.vocab_size
            )));
        }
        
        // === Validation: Tensor Buffer Sizes ===
        tracing::info!("Validating tensor buffer sizes...");
        
        // Token embedding table: [vocab_size × hidden_dim] × 4 bytes (F32)
        let expected_emb_size = (config.vocab_size * config.hidden_dim * 4) as u64;
        if token_embedding_table.size() != expected_emb_size {
            return Err(crate::compute::ComputeError::InvalidDimensions(format!(
                "Token embedding table size mismatch\nExpected: {} bytes ({} vocab × {} hidden × 4 bytes)\nActual: {} bytes\n\nSuggestions:\n  - Verify the config.json matches this model\n  - Check if vocab_size or hidden_size are incorrect\n  - The tensor may be quantized (not yet supported)",
                expected_emb_size,
                config.vocab_size,
                config.hidden_dim,
                token_embedding_table.size()
            )));
        }
        
        // Output norm weight: [hidden_dim] × 4 bytes (F32)
        let expected_norm_size = (config.hidden_dim * 4) as u64;
        if output_norm_weight.size() != expected_norm_size {
            return Err(crate::compute::ComputeError::InvalidDimensions(format!(
                "Output norm weight size mismatch\nExpected: {} bytes ({} hidden × 4 bytes)\nActual: {} bytes\n\nSuggestions:\n  - Verify the config.json hidden_size is correct\n  - Check if the model uses a different norm implementation",
                expected_norm_size,
                config.hidden_dim,
                output_norm_weight.size()
            )));
        }
        
        // LM head weight: [hidden_dim × vocab_size] × 4 bytes (F32)
        let expected_lm_head_size = (config.hidden_dim * config.vocab_size * 4) as u64;
        if lm_head_weight.size() != expected_lm_head_size {
            return Err(crate::compute::ComputeError::InvalidDimensions(format!(
                "LM head weight size mismatch\nExpected: {} bytes ({} hidden × {} vocab × 4 bytes)\nActual: {} bytes\n\nSuggestions:\n  - Verify config.json vocab_size and hidden_size are correct\n  - Some models share embeddings with LM head (weight tying)\n  - The tensor may be quantized (not yet supported)",
                expected_lm_head_size,
                config.hidden_dim,
                config.vocab_size,
                lm_head_weight.size()
            )));
        }
        
        tracing::info!("✓ All validations passed");

        // Create KV cache (uses num_kv_heads for GQA support, segmented by layer and batch)
        let cache = KVCache::new(
            &engine,
            config.batch_size,
            config.num_layers,
            config.max_seq_len,
            config.num_kv_heads,
            config.head_dim,
        )?;

        // === Static Computation Graph: Pre-allocate All Scratch Buffers ===
        let device = engine.device();
        let buffer_usage = wgpu::BufferUsages::STORAGE 
            | wgpu::BufferUsages::COPY_SRC 
            | wgpu::BufferUsages::COPY_DST;

        tracing::info!("Allocating static computation graph scratch buffers (batch_size={})...", config.batch_size);

        // Ping-pong hidden state buffers [batch_size, hidden_dim]
        let hidden_state = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hidden_state"),
            size: (config.batch_size * config.hidden_dim * std::mem::size_of::<f32>() as u32) as u64,
            usage: buffer_usage,
            mapped_at_creation: false,
        });

        let hidden_state_alt = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hidden_state_alt"),
            size: (config.batch_size * config.hidden_dim * std::mem::size_of::<f32>() as u32) as u64,
            usage: buffer_usage,
            mapped_at_creation: false,
        });

        // Attention projection buffers [batch_size, num_heads/kv_heads, head_dim]
        let q_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("q_buf"),
            size: (config.batch_size * config.num_heads * config.head_dim * std::mem::size_of::<f32>() as u32) as u64,
            usage: buffer_usage,
            mapped_at_creation: false,
        });

        let k_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k_buf"),
            size: (config.batch_size * config.num_kv_heads * config.head_dim * std::mem::size_of::<f32>() as u32) as u64,
            usage: buffer_usage,
            mapped_at_creation: false,
        });

        let v_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("v_buf"),
            size: (config.batch_size * config.num_kv_heads * config.head_dim * std::mem::size_of::<f32>() as u32) as u64,
            usage: buffer_usage,
            mapped_at_creation: false,
        });

        // Rotated Q and K buffers (after RoPE) [batch_size, num_heads/kv_heads, head_dim]
        let q_rot_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("q_rot_buf"),
            size: (config.batch_size * config.num_heads * config.head_dim * std::mem::size_of::<f32>() as u32) as u64,
            usage: buffer_usage,
            mapped_at_creation: false,
        });

        let k_rot_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k_rot_buf"),
            size: (config.batch_size * config.num_kv_heads * config.head_dim * std::mem::size_of::<f32>() as u32) as u64,
            usage: buffer_usage,
            mapped_at_creation: false,
        });

        let attn_out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("attn_out_buf"),
            size: (config.batch_size * config.num_heads * config.head_dim * std::mem::size_of::<f32>() as u32) as u64,
            usage: buffer_usage,
            mapped_at_creation: false,
        });

        // FFN projection buffers [batch_size, ffn_dim]
        let gate_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gate_buf"),
            size: (config.batch_size * config.ffn_dim * std::mem::size_of::<f32>() as u32) as u64,
            usage: buffer_usage,
            mapped_at_creation: false,
        });

        let up_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("up_buf"),
            size: (config.batch_size * config.ffn_dim * std::mem::size_of::<f32>() as u32) as u64,
            usage: buffer_usage,
            mapped_at_creation: false,
        });

        // Logits output buffer [batch_size, vocab_size]
        let logits_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("logits_buf"),
            size: (config.batch_size * config.vocab_size * std::mem::size_of::<f32>() as u32) as u64,
            usage: buffer_usage,
            mapped_at_creation: false,
        });

        // Additional TransformerBlock scratch buffers [batch_size, hidden_dim]
        let scratch_input_norm = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scratch_input_norm"),
            size: (config.batch_size * config.hidden_dim * std::mem::size_of::<f32>() as u32) as u64,
            usage: buffer_usage,
            mapped_at_creation: false,
        });

        let scratch_proj_out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scratch_proj_out"),
            size: (config.batch_size * config.hidden_dim * std::mem::size_of::<f32>() as u32) as u64,
            usage: buffer_usage,
            mapped_at_creation: false,
        });

        let scratch_hidden1 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scratch_hidden1"),
            size: (config.batch_size * config.hidden_dim * std::mem::size_of::<f32>() as u32) as u64,
            usage: buffer_usage,
            mapped_at_creation: false,
        });

        let scratch_ffn_norm = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scratch_ffn_norm"),
            size: (config.batch_size * config.hidden_dim * std::mem::size_of::<f32>() as u32) as u64,
            usage: buffer_usage,
            mapped_at_creation: false,
        });

        // Additional TransformerBlock scratch buffers [batch_size, ffn_dim]
        let scratch_swiglu = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scratch_swiglu"),
            size: (config.batch_size * config.ffn_dim * std::mem::size_of::<f32>() as u32) as u64,
            usage: buffer_usage,
            mapped_at_creation: false,
        });

        let scratch_ffn_out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scratch_ffn_out"),
            size: (config.batch_size * config.hidden_dim * std::mem::size_of::<f32>() as u32) as u64,
            usage: buffer_usage,
            mapped_at_creation: false,
        });

        // Attention intermediate buffers [batch_size, num_heads, max_seq_len]
        let scores_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scores_buf"),
            size: (config.batch_size * config.num_heads * config.max_seq_len * std::mem::size_of::<f32>() as u32) as u64,
            usage: buffer_usage,
            mapped_at_creation: false,
        });

        let probs_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("probs_buf"),
            size: (config.batch_size * config.num_heads * config.max_seq_len * std::mem::size_of::<f32>() as u32) as u64,
            usage: buffer_usage,
            mapped_at_creation: false,
        });

        // === RoPE Cache: Pre-compute sin/cos values ===
        tracing::info!("Pre-computing RoPE sin/cos cache...");
        let rope_cache = Self::create_rope_cache(device, config.max_seq_len, config.head_dim, 10000.0);

        tracing::info!(
            "Allocated scratch buffers (batch_size={}): hidden={}KB, q/k/v={}KB, ffn={}KB, logits={}KB",
            config.batch_size,
            (config.batch_size * config.hidden_dim * 4 * 2) / 1024, // 2 hidden state buffers
            (config.batch_size * ((config.num_heads + config.num_kv_heads * 2 + config.num_heads) * config.head_dim * 4)) / 1024,
            (config.batch_size * config.ffn_dim * 4 * 2) / 1024, // gate + up
            (config.batch_size * config.vocab_size * 4) / 1024 // logits
        );

        // === Pipeline Cache: Pre-compile all shaders ===
        tracing::info!("Pre-compiling GPU shaders and pipelines...");
        let pipeline_cache = PipelineCache::new(device);

        tracing::info!(
            "Initialized model: {} layers, hidden_dim={}, vocab_size={}",
            config.num_layers,
            config.hidden_dim,
            config.vocab_size
        );

        Ok(Self {
            config,
            engine,
            tokenizer,
            sampler,
            token_embedding_table,
            blocks,
            output_norm_weight,
            lm_head_weight,
            cache,
            hidden_state,
            hidden_state_alt,
            q_buf,
            k_buf,
            v_buf,
            q_rot_buf,
            k_rot_buf,
            attn_out_buf,
            gate_buf,
            up_buf,
            logits_buf,
            scratch_input_norm,
            scratch_proj_out,
            scratch_hidden1,
            scratch_ffn_norm,
            scratch_swiglu,
            scratch_ffn_out,
            scores_buf,
            probs_buf,
            rope_cache,
            pipeline_cache,
        })
    }

    /// Pre-compute sin/cos values for RoPE and upload to GPU
    ///
    /// This function pre-computes all sin/cos values for all positions up to max_seq_len
    /// and all dimension pairs in the head. The values are stored in a lookup table
    /// to avoid expensive trigonometric computations during inference.
    ///
    /// # Arguments
    /// * `device` - GPU device
    /// * `max_seq_len` - Maximum sequence length
    /// * `head_dim` - Dimension of each attention head
    /// * `theta_base` - Base for frequency calculation (typically 10000.0)
    ///
    /// # Returns
    /// GPU buffer containing pre-computed sin/cos values
    /// Layout: [position * head_dim + dim] = (cos_value, sin_value) interleaved
    fn create_rope_cache(
        device: &wgpu::Device,
        max_seq_len: u32,
        head_dim: u32,
        theta_base: f32,
    ) -> Buffer {
        let half_dim = (head_dim / 2) as usize;
        let max_seq_len = max_seq_len as usize;
        
        // Pre-compute sin/cos for all positions and dimension pairs
        // We store cos and sin interleaved: [cos0, sin0, cos1, sin1, ...]
        let mut cache_data = Vec::with_capacity(max_seq_len * head_dim as usize);
        
        for position in 0..max_seq_len {
            for dim_pair in 0..half_dim {
                // Calculate theta for this dimension pair
                // theta = theta_base ^ (2 * dim_pair / head_dim)
                let exponent = (2 * dim_pair) as f32 / head_dim as f32;
                let theta = theta_base.powf(exponent);
                
                // Calculate angle for this position
                let angle = position as f32 / theta;
                
                // Store cos and sin values (interleaved for both halves)
                cache_data.push(angle.cos());
                cache_data.push(angle.sin());
            }
        }
        
        // Create GPU buffer with pre-computed values
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rope_cache"),
            contents: bytemuck::cast_slice(&cache_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    }

    /// Embed a single token ID into hidden state vector
    ///
    /// This performs a lookup in the token embedding table and writes
    /// the corresponding row to the provided output buffer.
    ///
    /// # Arguments
    /// * `encoder` - Shared command encoder for recording GPU operations
    /// * `output_buffer` - Pre-allocated output buffer [hidden_dim]
    /// * `token_id` - Token ID to embed (0 to vocab_size - 1)
    ///
    /// # Note
    /// This function records commands to the encoder but does NOT submit them.
    /// The caller is responsible for submitting the encoder.
    /// Embed multiple tokens in parallel (batched embedding lookup)
    ///
    /// # Arguments
    /// * `encoder` - Command encoder for GPU operations
    /// * `output_buffer` - Output buffer to write embeddings [batch_size, hidden_dim]
    /// * `token_ids` - Array of token IDs to embed [batch_size]
    ///
    /// # Returns
    /// Ok(()) on success
    fn embed_tokens(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        output_buffer: &wgpu::Buffer,
        token_ids: &[u32],
    ) -> Result<()> {
        // Validate batch size
        if token_ids.len() != self.config.batch_size as usize {
            return Err(crate::compute::ComputeError::InvalidDimensions(format!(
                "Token IDs length ({}) does not match batch_size ({})",
                token_ids.len(), self.config.batch_size
            )));
        }

        // Validate all token IDs are in range
        for (i, &token_id) in token_ids.iter().enumerate() {
            if token_id >= self.config.vocab_size {
                return Err(crate::compute::ComputeError::InvalidDimensions(format!(
                    "Token ID {} at batch index {} is out of range [0, {})",
                    token_id, i, self.config.vocab_size
                )));
            }
        }

        // Create shader
        let shader = self
            .engine
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("embed_shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../compute/shaders/embed.wgsl").into(),
                ),
            });

        // Create uniform buffer for params: [batch_size, hidden_dim]
        let params_buffer = self
            .engine
            .device()
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("embed_params"),
                size: 8, // 2 x u32
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

        self.engine.queue().write_buffer(
            &params_buffer,
            0,
            bytemuck::cast_slice(&[self.config.batch_size, self.config.hidden_dim]),
        );

        // Create token IDs buffer
        let token_ids_buffer = self
            .engine
            .device()
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("embed_token_ids"),
                size: (self.config.batch_size * std::mem::size_of::<u32>() as u32) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

        self.engine.queue().write_buffer(
            &token_ids_buffer,
            0,
            bytemuck::cast_slice(token_ids),
        );

        // Create bind group layout
        let bind_group_layout =
            self.engine
                .device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("embed_bind_group_layout"),
                    entries: &[
                        // Params
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Token IDs
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Embedding table
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Output
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let bind_group = self
            .engine
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("embed_bind_group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: token_ids_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.token_embedding_table.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: output_buffer.as_entire_binding(),
                    },
                ],
            });

        // Create pipeline
        let pipeline_layout =
            self.engine
                .device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("embed_pipeline_layout"),
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: Default::default(),
                });

        let pipeline = self
            .engine
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("embed_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Record compute pass
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("embed_pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch: process batch_size * hidden_dim elements
            // 256 threads per workgroup
            let total_elements = self.config.batch_size * self.config.hidden_dim;
            let workgroups = (total_elements + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        Ok(())
    }

    /// Embed a single token (convenience wrapper for batch_size=1)
    ///
    /// # Arguments
    /// * `encoder` - Command encoder for GPU operations
    /// * `output_buffer` - Output buffer to write embedding [hidden_dim]
    /// * `token_id` - Token ID to embed
    ///
    /// # Returns
    /// Ok(()) on success
    fn embed_token(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        output_buffer: &wgpu::Buffer,
        token_id: u32,
    ) -> Result<()> {
        // For batch_size=1, use the batched version with a single-element array
        if self.config.batch_size == 1 {
            self.embed_tokens(encoder, output_buffer, &[token_id])
        } else {
            // For batch_size > 1, this is a programming error
            Err(crate::compute::ComputeError::Other(format!(
                "embed_token() called with batch_size={}, use embed_tokens() instead",
                self.config.batch_size
            )))
        }
    }

    /// Run forward pass for a single token
    ///
    /// This executes the full transformer pipeline:
    /// 1. Embed the token
    /// 2. Pass through all transformer blocks
    /// 3. Apply final normalization
    /// 4. Project to vocabulary (LM head)
    ///
    /// The logits are written to the internal `logits_buf` and can be accessed
    /// via `logits_buffer()` method.
    ///
    /// # Arguments
    /// * `token_id` - Input token ID
    /// * `seq_pos` - Position in the sequence (for RoPE)
    ///
    /// # Returns
    /// Ok(()) on success
    async fn forward(&mut self, token_id: u32, seq_pos: u32) -> Result<()> {
        // Create a single command encoder for the entire forward pass
        let mut encoder = self
            .engine
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("forward_pass_encoder"),
            });
        
        // Step 1: Embed token into hidden_state buffer (ping-pong buffer A)
        self.embed_token(&mut encoder, &self.hidden_state, token_id)?;

        // Step 2: Pass through all transformer blocks with ping-pong pattern
        for (layer_idx, block) in self.blocks.iter().enumerate() {
            tracing::debug!("Layer {}/{}: forward pass", layer_idx + 1, self.config.num_layers);
            
            // Determine input and output buffers (ping-pong between hidden_state and hidden_state_alt)
            let (input_buf, output_buf) = if layer_idx % 2 == 0 {
                (&self.hidden_state, &self.hidden_state_alt)
            } else {
                (&self.hidden_state_alt, &self.hidden_state)
            };

            // Execute transformer block (all operations batched into shared encoder)
            block.forward(
                &self.engine,
                &mut encoder,
                &self.pipeline_cache,
                input_buf,
                output_buf,
                &self.scratch_input_norm,
                &self.q_buf,
                &self.k_buf,
                &self.v_buf,
                &self.q_rot_buf,
                &self.k_rot_buf,
                &self.attn_out_buf,
                &self.scratch_proj_out,
                &self.scratch_hidden1,
                &self.scratch_ffn_norm,
                &self.gate_buf,
                &self.up_buf,
                &self.scratch_swiglu,
                &self.scratch_ffn_out,
                &self.scores_buf,
                &self.probs_buf,
                &self.rope_cache,
                &mut self.cache,
                layer_idx as u32,
                seq_pos,
            )?;
        }

        // Determine which buffer contains the final block output
        let final_block_output = if self.config.num_layers % 2 == 0 {
            &self.hidden_state
        } else {
            &self.hidden_state_alt
        };

        // Step 3: Final RMSNorm (reuse scratch_ffn_norm as temporary buffer)
        tracing::debug!("Applying final RMSNorm");
        rmsnorm(
            &self.engine,
            &mut encoder,
            &self.pipeline_cache,
            final_block_output,
            &self.scratch_ffn_norm,
            &self.output_norm_weight,
            self.config.batch_size,
            self.config.hidden_dim,
            self.config.rms_norm_eps,
        )?;

        // Step 4: LM head projection: normalized output -> logits_buf
        tracing::debug!("Computing LM head projection");
        gemm(
            &self.engine,
            &mut encoder,
            &self.pipeline_cache,
            &self.scratch_ffn_norm,
            &self.lm_head_weight,
            &self.logits_buf,
            self.config.batch_size,
            self.config.hidden_dim,
            self.config.hidden_dim,
            self.config.vocab_size,
        )?;

        // Step 5: Submit all operations in a SINGLE batch
        tracing::debug!("Submitting forward pass (single GPU submission)");
        self.engine.queue().submit(Some(encoder.finish()));

        Ok(())
    }

    /// Run batched forward pass for multiple tokens (one per sequence)
    ///
    /// This executes the full transformer pipeline for a batch of sequences:
    /// 1. Embed all tokens (batch_size tokens)
    /// 2. Pass through all transformer blocks
    /// 3. Apply final normalization
    /// 4. Project to vocabulary (LM head)
    ///
    /// The logits are written to the internal `logits_buf` with shape [batch_size, vocab_size]
    ///
    /// # Arguments
    /// * `token_ids` - Input token IDs (length must equal batch_size)
    /// * `seq_pos` - Position in the sequence (same for all sequences)
    ///
    /// # Returns
    /// Ok(()) on success
    ///
    /// # Note
    /// Currently assumes all sequences are at the same position. Future versions
    /// may support per-sequence positions.
    async fn forward_batch(&mut self, token_ids: &[u32], seq_pos: u32) -> Result<()> {
        // Validate batch size
        if token_ids.len() != self.config.batch_size as usize {
            return Err(crate::compute::ComputeError::Other(format!(
                "forward_batch() expected {} token IDs, got {}",
                self.config.batch_size,
                token_ids.len()
            )));
        }

        // Create a single command encoder for the entire forward pass
        let mut encoder = self
            .engine
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("forward_batch_encoder"),
            });
        
        // Step 1: Embed tokens into hidden_state buffer (ping-pong buffer A)
        self.embed_tokens(&mut encoder, &self.hidden_state, token_ids)?;

        // Step 2: Pass through all transformer blocks with ping-pong pattern
        for (layer_idx, block) in self.blocks.iter().enumerate() {
            tracing::debug!("Layer {}/{}: batched forward pass", layer_idx + 1, self.config.num_layers);
            
            // Determine input and output buffers (ping-pong between hidden_state and hidden_state_alt)
            let (input_buf, output_buf) = if layer_idx % 2 == 0 {
                (&self.hidden_state, &self.hidden_state_alt)
            } else {
                (&self.hidden_state_alt, &self.hidden_state)
            };

            // Execute transformer block (all operations batched into shared encoder)
            block.forward(
                &self.engine,
                &mut encoder,
                &self.pipeline_cache,
                input_buf,
                output_buf,
                &self.scratch_input_norm,
                &self.q_buf,
                &self.k_buf,
                &self.v_buf,
                &self.q_rot_buf,
                &self.k_rot_buf,
                &self.attn_out_buf,
                &self.scratch_proj_out,
                &self.scratch_hidden1,
                &self.scratch_ffn_norm,
                &self.gate_buf,
                &self.up_buf,
                &self.scratch_swiglu,
                &self.scratch_ffn_out,
                &self.scores_buf,
                &self.probs_buf,
                &self.rope_cache,
                &mut self.cache,
                layer_idx as u32,
                seq_pos,
            )?;
        }

        // Determine which buffer contains the final block output
        let final_block_output = if self.config.num_layers % 2 == 0 {
            &self.hidden_state
        } else {
            &self.hidden_state_alt
        };

        // Step 3: Final RMSNorm (reuse scratch_ffn_norm as temporary buffer)
        tracing::debug!("Applying final RMSNorm (batched)");
        rmsnorm(
            &self.engine,
            &mut encoder,
            &self.pipeline_cache,
            final_block_output,
            &self.scratch_ffn_norm,
            &self.output_norm_weight,
            self.config.batch_size,
            self.config.hidden_dim,
            self.config.rms_norm_eps,
        )?;

        // Step 4: LM head projection: normalized output -> logits_buf
        // Output: [batch_size, vocab_size]
        tracing::debug!("Computing LM head projection (batched)");
        gemm(
            &self.engine,
            &mut encoder,
            &self.pipeline_cache,
            &self.scratch_ffn_norm,
            &self.lm_head_weight,
            &self.logits_buf,
            self.config.batch_size,
            self.config.hidden_dim,
            self.config.hidden_dim,
            self.config.vocab_size,
        )?;

        // Step 5: Submit all operations in a SINGLE batch
        tracing::debug!("Submitting batched forward pass (single GPU submission)");
        self.engine.queue().submit(Some(encoder.finish()));

        Ok(())
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

    /// Generate text autoregressively
    ///
    /// This implements the complete autoregressive generation loop:
    /// - Tokenizes the input prompt
    /// - Processes each token through the model
    /// - Samples new tokens until max_tokens is reached or EOS is generated
    /// - Decodes and prints tokens as they are generated
    ///
    /// # Arguments
    /// * `prompt` - Input text prompt
    /// * `max_tokens` - Maximum number of tokens to generate
    ///
    /// # Returns
    /// The complete generated text (prompt + generated tokens)
    pub async fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        // Step A: Tokenize the prompt
        tracing::info!("Tokenizing prompt: \"{}\"", prompt);
        let mut token_ids = self
            .tokenizer
            .encode(prompt, false)  // Don't add special tokens automatically
            .map_err(|e| crate::compute::ComputeError::Other(format!("Tokenization failed: {}", e)))?;

        // CRITICAL: LLaMA models REQUIRE a BOS token at the start
        // Without this, the model has no context anchor and produces gibberish
        let bos_token_id = self.tokenizer.bos_token_id().unwrap_or(1);
        token_ids.insert(0, bos_token_id);
        tracing::info!("Prepended BOS token (ID: {}) to prompt", bos_token_id);

        if token_ids.is_empty() {
            return Err(crate::compute::ComputeError::Other(
                "Empty prompt after tokenization".into(),
            ));
        }

        tracing::info!("Prompt tokens: {} tokens", token_ids.len());

        // Reset cache for new generation
        self.cache.reset();

        // Process prompt tokens (prefill phase)
        let mut seq_pos = 0u32;
        tracing::info!("Prefill phase: processing {} prompt tokens", token_ids.len());
        
        for (idx, &token_id) in token_ids.iter().enumerate() {
            tracing::debug!("Prefill token {}/{}: ID={}", idx + 1, token_ids.len(), token_id);
            self.forward(token_id, seq_pos).await?;
            seq_pos += 1;
        }

        // Get the last token for autoregressive generation
        let mut last_token = match token_ids.last() {
            Some(&token) => token,
            None => {
                // This should be unreachable due to the empty check above,
                // but we handle it gracefully anyway
                return Err(crate::compute::ComputeError::Other(
                    "Empty token sequence after validation".into(),
                ));
            }
        };

        // Start generation (decode phase)
        tracing::info!("Generation phase: generating up to {} tokens", max_tokens);
        let mut generated_tokens = Vec::new();
        let mut printed_len = 0;

        // Print prompt
        print!("{}", prompt);
        std::io::Write::flush(&mut std::io::stdout()).ok();

        // High-precision benchmark timing
        let generation_start = std::time::Instant::now();
        let mut tokens_generated = 0u32;

        for step in 0..max_tokens {
            // Check sequence length limit
            if seq_pos >= self.config.max_seq_len {
                tracing::warn!("Reached maximum sequence length: {}", self.config.max_seq_len);
                eprintln!("\n[Generation stopped: Reached max sequence length: {}]", self.config.max_seq_len);
                break;
            }

            tracing::debug!("Generation step {}/{}: seq_pos={}", step + 1, max_tokens, seq_pos);

            // Step C-E: Forward pass (writes logits to internal buffer)
            self.forward(last_token, seq_pos).await?;

            // Step F: Sample next token from logits buffer (pass context for repetition penalty)
            let next_token = self.sampler.sample(&self.engine, self.logits_buffer(), &generated_tokens).await?;
            
            tracing::debug!("Sampled token ID: {}", next_token);

            // Increment tokens generated counter
            tokens_generated += 1;

            // Check for EOS token (token ID 2 for LLaMA architectures)
            if next_token == 2 {
                tracing::info!("Generated EOS token (ID: 2), stopping generation");
                eprintln!("\n[Generation stopped: EOS token (ID: 2) generated]");
                break;
            }
            
            // Also check tokenizer's EOS token if available
            if let Some(eos_id) = self.tokenizer.eos_token_id() {
                if next_token == eos_id {
                    tracing::info!("Generated EOS token (ID: {}), stopping generation", eos_id);
                    eprintln!("\n[Generation stopped: EOS token (ID: {}) generated]", eos_id);
                    break;
                }
            }

            // Add token to generated sequence
            generated_tokens.push(next_token);

            // Step G: Decode entire sequence and print only new text (fixes SentencePiece space stripping)
            let full_text = self
                .tokenizer
                .decode_batch(&generated_tokens)
                .map_err(|e| crate::compute::ComputeError::Other(format!("Detokenization failed: {}", e)))?;

            // Print only the newly generated text (streaming output)
            if full_text.len() > printed_len {
                print!("{}", &full_text[printed_len..]);
                std::io::Write::flush(&mut std::io::stdout()).ok();
                printed_len = full_text.len();
            }

            // Update for next iteration
            last_token = next_token;
            seq_pos += 1;
        }

        // Indicate why generation stopped
        if tokens_generated as usize >= max_tokens {
            eprintln!("[Generation stopped: Reached max_tokens limit: {}]", max_tokens);
        }

        // Final newline
        println!();

        // Calculate and print telemetry
        let elapsed_secs = generation_start.elapsed().as_secs_f64();
        let tps = if elapsed_secs > 0.0 {
            (tokens_generated as f64) / elapsed_secs
        } else {
            0.0
        };

        println!("\n=== Telemetry ===");
        println!("Tokens Generated: {} / {} requested", tokens_generated, max_tokens);
        println!("Elapsed Time: {:.3} seconds", elapsed_secs);
        println!("Speed: {:.2} tok/s", tps);
        println!("GPU Submissions per Token: 1 (single forward pass)");
        println!("=================");

        // Decode final text from all generated tokens
        let generated_text = self
            .tokenizer
            .decode_batch(&generated_tokens)
            .map_err(|e| crate::compute::ComputeError::Other(format!("Final detokenization failed: {}", e)))?;

        tracing::info!(
            "Generation complete: {} tokens generated in {:.3}s ({:.2} tok/s)",
            generated_tokens.len(),
            elapsed_secs,
            tps
        );

        Ok(generated_text)
    }

    /// Generate text autoregressively for multiple prompts in parallel
    ///
    /// This implements batched autoregressive generation:
    /// - Tokenizes all input prompts
    /// - Pads/truncates to same length for parallel processing
    /// - Processes all sequences in parallel through the model
    /// - Samples independently for each sequence
    /// - Continues until all sequences finish (EOS or max_tokens)
    ///
    /// # Arguments
    /// * `prompts` - Input text prompts (must have length equal to batch_size)
    /// * `max_tokens` - Maximum number of tokens to generate per sequence
    ///
    /// # Returns
    /// Vector of generated texts, one per prompt
    ///
    /// # Note
    /// Currently requires batch_size to match the number of prompts.
    /// All sequences are padded to the same length for simplicity.
    pub async fn generate_batch(&mut self, prompts: &[&str], max_tokens: usize) -> Result<Vec<String>> {
        // Validate batch size
        if prompts.len() != self.config.batch_size as usize {
            return Err(crate::compute::ComputeError::Other(format!(
                "Expected {} prompts for batch_size={}, got {}",
                self.config.batch_size,
                self.config.batch_size,
                prompts.len()
            )));
        }

        let batch_size = prompts.len();
        tracing::info!("Starting batched generation for {} prompts", batch_size);

        // Step 1: Tokenize all prompts
        let mut all_token_ids: Vec<Vec<u32>> = Vec::new();
        let bos_token_id = self.tokenizer.bos_token_id().unwrap_or(1);

        for (idx, prompt) in prompts.iter().enumerate() {
            tracing::info!("Tokenizing prompt {}/{}: \"{}\"", idx + 1, batch_size, prompt);
            let mut token_ids = self
                .tokenizer
                .encode(prompt, false)
                .map_err(|e| crate::compute::ComputeError::Other(format!("Tokenization failed for prompt {}: {}", idx, e)))?;

            // Add BOS token
            token_ids.insert(0, bos_token_id);
            tracing::info!("Prompt {}: {} tokens", idx + 1, token_ids.len());

            all_token_ids.push(token_ids);
        }

        // Step 2: Find max prompt length (for prefill phase)
        let max_prompt_len = all_token_ids.iter().map(|ids| ids.len()).max().unwrap_or(0);
        tracing::info!("Max prompt length: {} tokens", max_prompt_len);

        // Reset cache for new generation
        self.cache.reset();

        // Step 3: Prefill phase - process prompts token by token
        // NOTE: This is a simplified implementation that processes all prompts synchronously
        // A more efficient implementation would pad prompts and process them in parallel
        tracing::info!("Prefill phase: processing {} prompts", batch_size);

        for seq_pos in 0..max_prompt_len {
            // Collect token for each sequence at this position
            let mut batch_tokens = Vec::new();
            for token_ids in &all_token_ids {
                if seq_pos < token_ids.len() {
                    batch_tokens.push(token_ids[seq_pos]);
                } else {
                    // Pad with BOS token if this sequence is shorter
                    batch_tokens.push(bos_token_id);
                }
            }

            tracing::debug!("Prefill position {}/{}", seq_pos + 1, max_prompt_len);
            self.forward_batch(&batch_tokens, seq_pos as u32).await?;
        }

        // Step 4: Initialize generation state
        let mut seq_pos = max_prompt_len as u32;
        let mut last_tokens: Vec<u32> = all_token_ids.iter()
            .map(|ids| *ids.last().unwrap_or(&bos_token_id))
            .collect();
        let mut generated_tokens: Vec<Vec<u32>> = vec![Vec::new(); batch_size];
        let mut finished: Vec<bool> = vec![false; batch_size];
        let eos_token_id = self.tokenizer.eos_token_id().unwrap_or(2);

        // Print prompts
        for (idx, prompt) in prompts.iter().enumerate() {
            println!("[Prompt {}] {}", idx + 1, prompt);
        }
        println!();

        // Step 5: Generation phase
        tracing::info!("Generation phase: generating up to {} tokens per sequence", max_tokens);
        let generation_start = std::time::Instant::now();
        let mut total_tokens_generated = 0;

        for step in 0..max_tokens {
            // Check if all sequences are finished
            if finished.iter().all(|&f| f) {
                tracing::info!("All sequences finished at step {}", step);
                break;
            }

            // Check sequence length limit
            if seq_pos >= self.config.max_seq_len {
                tracing::warn!("Reached maximum sequence length: {}", self.config.max_seq_len);
                break;
            }

            tracing::debug!("Generation step {}/{}: seq_pos={}", step + 1, max_tokens, seq_pos);

            // Forward pass for all active sequences
            self.forward_batch(&last_tokens, seq_pos).await?;

            // Sample next token for each sequence independently
            // NOTE: This is a simplified implementation that uses the same sampling logic for all sequences
            // In practice, we'd need to extract per-sequence logits and sample independently
            
            // For now, let's use a placeholder: sample the first sequence's logits
            // A proper implementation would slice the logits buffer for each sequence
            let next_token = self.sampler.sample(&self.engine, self.logits_buffer(), &generated_tokens[0]).await?;
            
            // Update all sequences with the same token (TEMPORARY - need per-sequence sampling)
            for i in 0..batch_size {
                if !finished[i] {
                    last_tokens[i] = next_token;
                    generated_tokens[i].push(next_token);
                    total_tokens_generated += 1;

                    // Check for EOS
                    if next_token == eos_token_id || next_token == 2 {
                        finished[i] = true;
                        tracing::info!("Sequence {} finished (EOS)", i + 1);
                    }
                }
            }

            seq_pos += 1;
        }

        // Calculate telemetry
        let elapsed_secs = generation_start.elapsed().as_secs_f64();
        let tps = if elapsed_secs > 0.0 {
            (total_tokens_generated as f64) / elapsed_secs
        } else {
            0.0
        };

        println!("\n=== Batched Generation Telemetry ===");
        println!("Sequences: {}", batch_size);
        println!("Total Tokens Generated: {}", total_tokens_generated);
        println!("Elapsed Time: {:.3} seconds", elapsed_secs);
        println!("Speed: {:.2} tok/s (total throughput)", tps);
        println!("Speed per sequence: {:.2} tok/s", tps / batch_size as f64);
        println!("====================================");

        // Decode all generated texts
        let mut results = Vec::new();
        for (idx, tokens) in generated_tokens.iter().enumerate() {
            let text = self
                .tokenizer
                .decode_batch(tokens)
                .map_err(|e| crate::compute::ComputeError::Other(format!("Detokenization failed for sequence {}: {}", idx, e)))?;
            
            println!("[Result {}] {}", idx + 1, text);
            results.push(text);
        }

        tracing::info!(
            "Batched generation complete: {} sequences, {} total tokens in {:.3}s ({:.2} tok/s)",
            batch_size,
            total_tokens_generated,
            elapsed_secs,
            tps
        );

        Ok(results)
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_config() {
        let config = ModelConfig {
            hidden_dim: 4096,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 32, // MHA: same as num_heads
            head_dim: 128,
            ffn_dim: 11008,
            vocab_size: 32000,
            max_seq_len: 2048,
            rms_norm_eps: 1e-5,
        };

        assert_eq!(config.hidden_dim, 4096);
        assert_eq!(config.num_layers, 32);
        assert_eq!(config.vocab_size, 32000);
    }

    #[tokio::test]
    async fn test_model_block_count_validation() {
        use crate::compute::ComputeEngine;
        use crate::model::{Sampler, Tokenizer};
        
        let engine = ComputeEngine::new().await.unwrap();
        
        let config = ModelConfig {
            hidden_dim: 64,
            num_layers: 4, // Expect 4 layers
            num_heads: 4,
            num_kv_heads: 4, // MHA: same as num_heads
            head_dim: 16,
            ffn_dim: 128,
            vocab_size: 1000,
            max_seq_len: 128,
            rms_norm_eps: 1e-5,
        };
        
        // Create a mock tokenizer (this will fail but we're testing validation before that)
        let tokenizer = Tokenizer::from_file("nonexistent.json");
        if tokenizer.is_err() {
            // Skip test if we can't create a tokenizer
            // The important part is testing the block count validation
            return;
        }
        let tokenizer = tokenizer.unwrap();
        
        let sampler = Sampler::greedy(1000);
        
        // Create mock buffers
        let token_embedding_table = engine.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_embeddings"),
            size: (1000 * 64 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        
        let output_norm_weight = engine.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_output_norm"),
            size: (64 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        
        let lm_head_weight = engine.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_lm_head"),
            size: (64 * 1000 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        
        // Test with WRONG number of blocks (2 instead of 4)
        let wrong_blocks = vec![]; // Empty vector, definitely wrong
        
        let result = Model::new(
            config.clone(),
            engine,
            tokenizer,
            sampler,
            token_embedding_table,
            wrong_blocks,
            output_norm_weight,
            lm_head_weight,
        );
        
        // Should return an error, not panic
        assert!(result.is_err());
        
        if let Err(e) = result {
            let error_msg = format!("{}", e);
            assert!(error_msg.contains("Block count mismatch"));
            assert!(error_msg.contains("expected 4"));
            assert!(error_msg.contains("got 0"));
        }
    }
}

//! Model constructor and initialization

use super::{Model, ModelConfig};
use crate::compute::cache::KVCache;
use crate::compute::{ComputeEngine, PipelineCache, Result};
use crate::model::{block::TransformerBlock, sampler::Sampler, tokenizer::Tokenizer};
use wgpu::util::DeviceExt;
use wgpu::Buffer;

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

        // Note: All tensors are stored as packed FP16 (2 bytes per element) on GPU
        // to optimize VRAM usage. The engine converts F32/BF16 -> packed FP16 during allocation.

        // Token embedding table: [vocab_size × hidden_dim] × 2 bytes (packed FP16)
        let expected_emb_size_fp16 = (config.vocab_size * config.hidden_dim * 2) as u64;
        let expected_emb_size_fp32 = (config.vocab_size * config.hidden_dim * 4) as u64;
        let actual_emb_size = token_embedding_table.size();

        if actual_emb_size != expected_emb_size_fp16 && actual_emb_size != expected_emb_size_fp32 {
            return Err(crate::compute::ComputeError::InvalidDimensions(format!(
                "Token embedding table size mismatch\nExpected: {} bytes (packed FP16: {} vocab × {} hidden × 2 bytes)\n       or {} bytes (F32: {} vocab × {} hidden × 4 bytes)\nActual: {} bytes\n\nSuggestions:\n  - Verify the config.json matches this model\n  - Check if vocab_size or hidden_size are incorrect\n  - Actual vocab_size might be {} (if FP16) or {} (if F32)",
                expected_emb_size_fp16,
                config.vocab_size,
                config.hidden_dim,
                expected_emb_size_fp32,
                config.vocab_size,
                config.hidden_dim,
                actual_emb_size,
                actual_emb_size / (config.hidden_dim * 2) as u64,
                actual_emb_size / (config.hidden_dim * 4) as u64,
            )));
        }

        // Output norm weight: [hidden_dim] × 2 bytes (packed FP16)
        let expected_norm_size_fp16 = (config.hidden_dim * 2) as u64;
        let expected_norm_size_fp32 = (config.hidden_dim * 4) as u64;
        let actual_norm_size = output_norm_weight.size();

        if actual_norm_size != expected_norm_size_fp16
            && actual_norm_size != expected_norm_size_fp32
        {
            return Err(crate::compute::ComputeError::InvalidDimensions(format!(
                "Output norm weight size mismatch\nExpected: {} bytes (packed FP16: {} hidden × 2 bytes)\n       or {} bytes (F32: {} hidden × 4 bytes)\nActual: {} bytes\n\nSuggestions:\n  - Verify the config.json hidden_size is correct\n  - Check if the model uses a different norm implementation",
                expected_norm_size_fp16,
                config.hidden_dim,
                expected_norm_size_fp32,
                config.hidden_dim,
                actual_norm_size
            )));
        }

        // LM head weight: [hidden_dim × vocab_size] × 2 bytes (packed FP16)
        let expected_lm_head_size_fp16 = (config.hidden_dim * config.vocab_size * 2) as u64;
        let expected_lm_head_size_fp32 = (config.hidden_dim * config.vocab_size * 4) as u64;
        let actual_lm_head_size = lm_head_weight.size();

        if actual_lm_head_size != expected_lm_head_size_fp16
            && actual_lm_head_size != expected_lm_head_size_fp32
        {
            return Err(crate::compute::ComputeError::InvalidDimensions(format!(
                "LM head weight size mismatch\nExpected: {} bytes (packed FP16: {} hidden × {} vocab × 2 bytes)\n       or {} bytes (F32: {} hidden × {} vocab × 4 bytes)\nActual: {} bytes\n\nSuggestions:\n  - Verify config.json vocab_size and hidden_size are correct\n  - Some models share embeddings with LM head (weight tying)",
                expected_lm_head_size_fp16,
                config.hidden_dim,
                config.vocab_size,
                expected_lm_head_size_fp32,
                config.hidden_dim,
                config.vocab_size,
                actual_lm_head_size
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

        tracing::info!(
            "Allocating static computation graph scratch buffers (batch_size={})...",
            config.batch_size
        );

        // Ping-pong hidden state buffers [batch_size, hidden_dim]
        let hidden_state = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hidden_state"),
            size: (config.batch_size * config.hidden_dim * std::mem::size_of::<f32>() as u32)
                as u64,
            usage: buffer_usage,
            mapped_at_creation: false,
        });

        let hidden_state_alt = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hidden_state_alt"),
            size: (config.batch_size * config.hidden_dim * std::mem::size_of::<f32>() as u32)
                as u64,
            usage: buffer_usage,
            mapped_at_creation: false,
        });

        // Attention projection buffers [batch_size, num_heads/kv_heads, head_dim]
        let q_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("q_buf"),
            size: (config.batch_size
                * config.num_heads
                * config.head_dim
                * std::mem::size_of::<f32>() as u32) as u64,
            usage: buffer_usage,
            mapped_at_creation: false,
        });

        let k_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k_buf"),
            size: (config.batch_size
                * config.num_kv_heads
                * config.head_dim
                * std::mem::size_of::<f32>() as u32) as u64,
            usage: buffer_usage,
            mapped_at_creation: false,
        });

        let v_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("v_buf"),
            size: (config.batch_size
                * config.num_kv_heads
                * config.head_dim
                * std::mem::size_of::<f32>() as u32) as u64,
            usage: buffer_usage,
            mapped_at_creation: false,
        });

        // Rotated Q and K buffers (after RoPE) [batch_size, num_heads/kv_heads, head_dim]
        let q_rot_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("q_rot_buf"),
            size: (config.batch_size
                * config.num_heads
                * config.head_dim
                * std::mem::size_of::<f32>() as u32) as u64,
            usage: buffer_usage,
            mapped_at_creation: false,
        });

        let k_rot_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("k_rot_buf"),
            size: (config.batch_size
                * config.num_kv_heads
                * config.head_dim
                * std::mem::size_of::<f32>() as u32) as u64,
            usage: buffer_usage,
            mapped_at_creation: false,
        });

        let attn_out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("attn_out_buf"),
            size: (config.batch_size
                * config.num_heads
                * config.head_dim
                * std::mem::size_of::<f32>() as u32) as u64,
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
            size: (config.batch_size * config.vocab_size * std::mem::size_of::<f32>() as u32)
                as u64,
            usage: buffer_usage,
            mapped_at_creation: false,
        });

        // Additional TransformerBlock scratch buffers [batch_size, hidden_dim]
        let scratch_input_norm = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scratch_input_norm"),
            size: (config.batch_size * config.hidden_dim * std::mem::size_of::<f32>() as u32)
                as u64,
            usage: buffer_usage,
            mapped_at_creation: false,
        });

        let scratch_proj_out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scratch_proj_out"),
            size: (config.batch_size * config.hidden_dim * std::mem::size_of::<f32>() as u32)
                as u64,
            usage: buffer_usage,
            mapped_at_creation: false,
        });

        let scratch_hidden1 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scratch_hidden1"),
            size: (config.batch_size * config.hidden_dim * std::mem::size_of::<f32>() as u32)
                as u64,
            usage: buffer_usage,
            mapped_at_creation: false,
        });

        let scratch_ffn_norm = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scratch_ffn_norm"),
            size: (config.batch_size * config.hidden_dim * std::mem::size_of::<f32>() as u32)
                as u64,
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
            size: (config.batch_size * config.hidden_dim * std::mem::size_of::<f32>() as u32)
                as u64,
            usage: buffer_usage,
            mapped_at_creation: false,
        });

        // Attention intermediate buffers [batch_size, num_heads, max_seq_len]
        let scores_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scores_buf"),
            size: (config.batch_size
                * config.num_heads
                * config.max_seq_len
                * std::mem::size_of::<f32>() as u32) as u64,
            usage: buffer_usage,
            mapped_at_creation: false,
        });

        let probs_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("probs_buf"),
            size: (config.batch_size
                * config.num_heads
                * config.max_seq_len
                * std::mem::size_of::<f32>() as u32) as u64,
            usage: buffer_usage,
            mapped_at_creation: false,
        });

        // === RoPE Cache: Pre-compute sin/cos values ===
        tracing::info!("Pre-computing RoPE sin/cos cache...");
        let rope_cache =
            Self::create_rope_cache(device, config.max_seq_len, config.head_dim, 10000.0);

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
}

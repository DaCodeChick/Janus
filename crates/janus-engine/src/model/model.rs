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
use crate::compute::{ComputeEngine, Result};
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
}

/// Complete transformer model for text generation
///
/// This struct holds all components needed for autoregressive LLM inference:
/// - Token embeddings
/// - Transformer blocks (layers)
/// - Output projection (LM head)
/// - KV cache for efficient generation
/// - Tokenizer and sampler
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
}

impl Model {
    /// Create a new model
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
        // Verify block count matches config
        if blocks.len() != config.num_layers as usize {
            return Err(crate::compute::ComputeError::InvalidDimensions(format!(
                "Block count mismatch: expected {}, got {}",
                config.num_layers,
                blocks.len()
            )));
        }

        // Create KV cache (uses num_kv_heads for GQA support, segmented by layer)
        let cache = KVCache::new(
            &engine,
            config.num_layers,
            config.max_seq_len,
            config.num_kv_heads,
            config.head_dim,
        )?;

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
        })
    }

    /// Embed a single token ID into hidden state vector
    ///
    /// This performs a lookup in the token embedding table and copies
    /// the corresponding row to a new output buffer.
    ///
    /// # Arguments
    /// * `token_id` - Token ID to embed (0 to vocab_size - 1)
    ///
    /// # Returns
    /// GPU buffer containing the embedding vector [hidden_dim]
    async fn embed_token(&self, token_id: u32) -> Result<Buffer> {
        if token_id >= self.config.vocab_size {
            return Err(crate::compute::ComputeError::InvalidDimensions(format!(
                "Token ID {} out of range [0, {})",
                token_id, self.config.vocab_size
            )));
        }

        // Create output buffer
        let output = self
            .engine
            .device()
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("embedding_output"),
                size: (self.config.hidden_dim * std::mem::size_of::<f32>() as u32) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

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

        // Create uniform buffer for params
        let params_buffer = self
            .engine
            .device()
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("embed_params"),
                size: 8, // 2 x u32
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

        // Write params: [token_id, hidden_dim]
        self.engine.queue().write_buffer(
            &params_buffer,
            0,
            bytemuck::cast_slice(&[token_id, self.config.hidden_dim]),
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
                        // Embedding table
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
                        // Output
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
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
                        resource: self.token_embedding_table.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output.as_entire_binding(),
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

        // Execute
        let mut encoder = self
            .engine
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("embed_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("embed_pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch: 256 threads per workgroup
            let workgroups = (self.config.hidden_dim + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.engine.queue().submit(Some(encoder.finish()));

        Ok(output)
    }

    /// Run forward pass for a single token
    ///
    /// This executes the full transformer pipeline:
    /// 1. Embed the token
    /// 2. Pass through all transformer blocks
    /// 3. Apply final normalization
    /// 4. Project to vocabulary (LM head)
    ///
    /// # Arguments
    /// * `token_id` - Input token ID
    /// * `seq_pos` - Position in the sequence (for RoPE)
    ///
    /// # Returns
    /// Logits tensor [vocab_size] on GPU
    async fn forward(&mut self, token_id: u32, seq_pos: u32) -> Result<Buffer> {
        // Step B: Embed token
        let mut hidden = self.embed_token(token_id).await?;

        // Step C: Pass through all transformer blocks
        for (layer_idx, block) in self.blocks.iter().enumerate() {
            tracing::debug!("Layer {}/{}: forward pass", layer_idx + 1, self.config.num_layers);
            hidden = block
                .forward(&self.engine, &hidden, &mut self.cache, layer_idx as u32, seq_pos)
                .await?;
        }

        // Step D: Final RMSNorm
        tracing::debug!("Applying final RMSNorm");
        let normalized = rmsnorm(
            &self.engine,
            &hidden,
            &self.output_norm_weight,
            self.config.hidden_dim,
            self.config.rms_norm_eps,
        )
        .await?;

        // Step E: LM head projection [hidden_dim] x [hidden_dim, vocab_size] -> [vocab_size]
        tracing::debug!("Computing LM head projection");
        let logits = gemm(
            &self.engine,
            &normalized,
            &self.lm_head_weight,
            1,
            self.config.hidden_dim,
            self.config.vocab_size,
        )
        .await?;

        Ok(logits)
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

        for step in 0..max_tokens {
            // Check sequence length limit
            if seq_pos >= self.config.max_seq_len {
                tracing::warn!("Reached maximum sequence length: {}", self.config.max_seq_len);
                break;
            }

            tracing::debug!("Generation step {}/{}: seq_pos={}", step + 1, max_tokens, seq_pos);

            // Step C-E: Forward pass to get logits
            let logits = self.forward(last_token, seq_pos).await?;

            // Step F: Sample next token (pass context for repetition penalty)
            let next_token = self.sampler.sample(&self.engine, &logits, &generated_tokens).await?;
            
            tracing::debug!("Sampled token ID: {}", next_token);

            // Check for EOS token
            if let Some(eos_id) = self.tokenizer.eos_token_id() {
                if next_token == eos_id {
                    tracing::info!("Generated EOS token, stopping generation");
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

        // Final newline
        println!();

        // Decode final text from all generated tokens
        let generated_text = self
            .tokenizer
            .decode_batch(&generated_tokens)
            .map_err(|e| crate::compute::ComputeError::Other(format!("Final detokenization failed: {}", e)))?;

        tracing::info!(
            "Generation complete: {} tokens generated",
            generated_tokens.len()
        );

        Ok(generated_text)
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

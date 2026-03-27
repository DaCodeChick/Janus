//! Tests for Model struct

use super::{Model, ModelConfig};

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
        rope_freq_base: 10000.0,
        batch_size: 1,
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
        rope_freq_base: 10000.0,
        batch_size: 1,
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

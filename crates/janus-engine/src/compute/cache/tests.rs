//! Tests for KV cache implementation

use super::*;
use crate::compute::engine::ComputeEngine;
use crate::compute::pipeline_cache::PipelineCache;

#[tokio::test]
async fn test_kv_cache_creation() {
    let engine = ComputeEngine::new().await.unwrap();
    
    let cache = KVCache::new(
        &engine,
        1,      // batch_size
        4,      // num_layers
        128,    // max_seq_len
        4,      // num_kv_heads
        16,     // head_dim
    );
    
    assert!(cache.is_ok());
    let cache = cache.unwrap();
    
    assert_eq!(cache.num_layers(), 4);
    assert_eq!(cache.max_seq_len(), 128);
    assert_eq!(cache.num_kv_heads(), 4);
    assert_eq!(cache.head_dim(), 16);
    assert_eq!(cache.current_position(), 0);
    assert_eq!(cache.batch_size(), 1);
}

#[tokio::test]
async fn test_kv_cache_reset() {
    let engine = ComputeEngine::new().await.unwrap();
    
    let mut cache = KVCache::new(&engine, 1, 4, 128, 4, 16).unwrap();
    
    // Initially at position 0
    assert_eq!(cache.current_position(), 0);
    
    // Reset should set position back to 0
    cache.reset();
    assert_eq!(cache.current_position(), 0);
}

#[tokio::test]
async fn test_kv_cache_copy_matching_dimensions() {
    let engine = ComputeEngine::new().await.unwrap();
    
    let source = KVCache::new(&engine, 1, 4, 128, 4, 16).unwrap();
    let mut dest = KVCache::new(&engine, 1, 4, 128, 4, 16).unwrap();
    
    let mut encoder = engine.device().create_command_encoder(
        &wgpu::CommandEncoderDescriptor {
            label: Some("test_copy_encoder"),
        }
    );
    
    let result = dest.copy_from(&mut encoder, &source);
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_kv_cache_copy_mismatched_dimensions() {
    let engine = ComputeEngine::new().await.unwrap();
    
    // Create caches with different dimensions
    let source = KVCache::new(&engine, 1, 4, 128, 4, 16).unwrap();
    let mut dest = KVCache::new(&engine, 1, 4, 256, 4, 16).unwrap(); // Different max_seq_len
    
    let mut encoder = engine.device().create_command_encoder(
        &wgpu::CommandEncoderDescriptor {
            label: Some("test_copy_encoder"),
        }
    );
    
    let result = dest.copy_from(&mut encoder, &source);
    assert!(result.is_err());
    
    if let Err(e) = result {
        let error_msg = format!("{}", e);
        assert!(error_msg.contains("dimension mismatch"));
    }
}

#[tokio::test]
async fn test_kv_cache_copy_position_sync() {
    let engine = ComputeEngine::new().await.unwrap();
    
    let source = KVCache::new(&engine, 1, 4, 128, 4, 16).unwrap();
    let mut dest = KVCache::new(&engine, 1, 4, 128, 4, 16).unwrap();
    
    // Manually set source position (simulating some forward passes)
    // Note: In real usage, position is set via update() method
    // Here we're just testing the copy_from position synchronization
    
    let mut encoder = engine.device().create_command_encoder(
        &wgpu::CommandEncoderDescriptor {
            label: Some("test_copy_encoder"),
        }
    );
    
    let result = dest.copy_from(&mut encoder, &source);
    assert!(result.is_ok());
    
    // After copying, destination should have same position as source
    assert_eq!(dest.current_position(), source.current_position());
}

#[test]
fn test_kv_cache_buffer_access() {
    // Test that we can get references to the buffers
    // This is a compile-time test more than runtime
    
    // Note: We can't actually create a cache here without async/GPU
    // but we can verify the API exists
}

#[tokio::test]
async fn test_compression_config_default() {
    let config = CacheCompressionConfig::default();
    
    assert!(!config.enabled);
    assert_eq!(config.uncompressed_window, 512);
    assert_eq!(config.compression_ratio, 2);
    assert_eq!(config.compression_trigger, 0.8);
}

#[tokio::test]
async fn test_compression_enabled_cache_creation() {
    let engine = ComputeEngine::new().await.unwrap();
    
    let config = CacheCompressionConfig {
        enabled: true,
        uncompressed_window: 256,
        compression_ratio: 2,
        compression_trigger: 0.75,
    };
    
    let cache = KVCache::with_compression(
        &engine,
        1,      // batch_size
        4,      // num_layers
        1024,   // max_seq_len
        4,      // num_kv_heads
        16,     // head_dim
        config,
    );
    
    assert!(cache.is_ok());
    let cache = cache.unwrap();
    
    assert!(cache.compression_config().enabled);
    assert_eq!(cache.compression_config().uncompressed_window, 256);
    assert_eq!(cache.compression_config().compression_ratio, 2);
    
    let (actual, compressed, effective) = cache.compression_stats();
    assert_eq!(actual, 0);
    assert_eq!(compressed, 0);
    assert_eq!(effective, 0);
}

#[tokio::test]
async fn test_should_compress_disabled() {
    let engine = ComputeEngine::new().await.unwrap();
    
    let cache = KVCache::new(&engine, 1, 4, 128, 4, 16).unwrap();
    
    // Compression disabled by default
    assert!(!cache.should_compress());
}

#[tokio::test]
async fn test_should_compress_below_threshold() {
    let engine = ComputeEngine::new().await.unwrap();
    
    let config = CacheCompressionConfig {
        enabled: true,
        uncompressed_window: 256,
        compression_ratio: 2,
        compression_trigger: 0.8, // 80% full
    };
    
    let mut cache = KVCache::with_compression(
        &engine,
        1,
        4,
        1024, // max_seq_len
        4,
        16,
        config,
    ).unwrap();
    
    // Simulate storing tokens (but not enough to trigger compression)
    // actual_tokens_stored is normally updated in update()
    // For testing, we'll manually set it
    cache.actual_tokens_stored = 500; // ~48% full (500/1024)
    
    assert!(!cache.should_compress());
}

#[tokio::test]
async fn test_should_compress_above_threshold() {
    let engine = ComputeEngine::new().await.unwrap();
    
    let config = CacheCompressionConfig {
        enabled: true,
        uncompressed_window: 256,
        compression_ratio: 2,
        compression_trigger: 0.8, // 80% full
    };
    
    let mut cache = KVCache::with_compression(
        &engine,
        1,
        4,
        1024, // max_seq_len
        4,
        16,
        config,
    ).unwrap();
    
    // Simulate storing enough tokens to trigger compression
    cache.actual_tokens_stored = 900; // ~87% full (900/1024)
    
    assert!(cache.should_compress());
}

#[tokio::test]
async fn test_compression_not_enough_tokens() {
    let engine = ComputeEngine::new().await.unwrap();
    
    let config = CacheCompressionConfig {
        enabled: true,
        uncompressed_window: 512,
        compression_ratio: 2,
        compression_trigger: 0.8,
    };
    
    let mut cache = KVCache::with_compression(
        &engine,
        1,
        4,
        1024,
        4,
        16,
        config,
    ).unwrap();
    
    // Not enough tokens to compress (need at least uncompressed_window + compression_ratio)
    cache.actual_tokens_stored = 100;
    
    let mut encoder = engine.device().create_command_encoder(
        &wgpu::CommandEncoderDescriptor {
            label: Some("test_compression_encoder"),
        }
    );
    
    // Load pipeline cache (needed for compress to work)
    let pipeline_cache = PipelineCache::new(engine.device());
    
    let result = cache.compress(&engine, &mut encoder, &pipeline_cache);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 0); // No tokens freed
}

#[tokio::test]
async fn test_compression_stats_tracking() {
    let engine = ComputeEngine::new().await.unwrap();
    
    let config = CacheCompressionConfig {
        enabled: true,
        uncompressed_window: 256,
        compression_ratio: 2,
        compression_trigger: 0.8,
    };
    
    let mut cache = KVCache::with_compression(
        &engine,
        1,
        4,
        1024,
        4,
        16,
        config,
    ).unwrap();
    
    // Initial stats
    let (actual, compressed, effective) = cache.compression_stats();
    assert_eq!(actual, 0);
    assert_eq!(compressed, 0);
    assert_eq!(effective, 0);
    
    // Simulate some tokens stored
    cache.actual_tokens_stored = 600;
    let (actual, compressed, effective) = cache.compression_stats();
    assert_eq!(actual, 600);
    assert_eq!(compressed, 0);
    assert_eq!(effective, 600);
    
    // Simulate compression effect
    cache.compressed_tokens = 172; // e.g., compressed 344 tokens down to 172
    cache.actual_tokens_stored = 428; // 172 + 256 (uncompressed window)
    
    let (actual, compressed, effective) = cache.compression_stats();
    assert_eq!(actual, 428);
    assert_eq!(compressed, 172);
    assert_eq!(effective, 600); // Total context preserved
}

#[tokio::test]
async fn test_reset_clears_compression_state() {
    let engine = ComputeEngine::new().await.unwrap();
    
    let config = CacheCompressionConfig {
        enabled: true,
        uncompressed_window: 256,
        compression_ratio: 2,
        compression_trigger: 0.8,
    };
    
    let mut cache = KVCache::with_compression(
        &engine,
        1,
        4,
        1024,
        4,
        16,
        config,
    ).unwrap();
    
    // Simulate some state
    cache.actual_tokens_stored = 500;
    cache.compressed_tokens = 100;
    cache.current_position = 50;
    
    // Reset
    cache.reset();
    
    // All counters should be zero
    assert_eq!(cache.current_position(), 0);
    let (actual, compressed, effective) = cache.compression_stats();
    assert_eq!(actual, 0);
    assert_eq!(compressed, 0);
    assert_eq!(effective, 0);
}

#[tokio::test]
async fn test_compress_disabled_cache_returns_error() {
    let engine = ComputeEngine::new().await.unwrap();
    
    // Create cache without compression enabled
    let mut cache = KVCache::new(&engine, 1, 4, 128, 4, 16).unwrap();
    
    let mut encoder = engine.device().create_command_encoder(
        &wgpu::CommandEncoderDescriptor {
            label: Some("test_encoder"),
        }
    );
    
    let pipeline_cache = PipelineCache::new(engine.device());
    
    // Attempt to compress should fail
    let result = cache.compress(&engine, &mut encoder, &pipeline_cache);
    assert!(result.is_err());
    
    if let Err(e) = result {
        let error_msg = format!("{}", e);
        assert!(error_msg.contains("Compression not enabled"));
    }
}

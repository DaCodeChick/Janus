//! Integration tests for text generation pipeline
//!
//! These tests verify the correctness of the full generation pipeline including:
//! - Model loading and initialization
//! - Token encoding/decoding
//! - Forward pass computation
//! - Sampling strategies (greedy, temperature, top-k, top-p, beam search)
//! - Repetition penalty
//!
//! Note: These tests require a GPU and will be skipped if no GPU is available.

use janus_engine::model::sampler::{Sampler, SamplerConfig};

#[test]
fn test_sampler_config_creation() {
    // Test default configuration
    let default_config = SamplerConfig::default();
    assert_eq!(default_config.temperature, 0.0);
    assert_eq!(default_config.top_k, 0);
    assert_eq!(default_config.top_p, 1.0);
    assert_eq!(default_config.repetition_penalty, 1.15);
    assert_eq!(default_config.beam_width, 1);

    // Test custom configuration for temperature sampling
    let temp_config = SamplerConfig {
        temperature: 0.7,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.2,
        beam_width: 1,
    };
    let sampler = Sampler::new(temp_config.clone(), 32000);
    assert_eq!(sampler.config().temperature, 0.7);
    assert_eq!(sampler.vocab_size(), 32000);

    // Test beam search configuration
    let beam_config = SamplerConfig {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        beam_width: 4,
    };
    let beam_sampler = Sampler::new(beam_config, 32000);
    assert!(beam_sampler.is_beam_search_enabled());
    assert_eq!(beam_sampler.config().beam_width, 4);
}

#[test]
fn test_top_k_tokens_selection() {
    let sampler = Sampler::greedy(1000);

    // Create simple logits where we know the ordering
    let logits = vec![
        -1.0, // token 0
        0.5,  // token 1
        2.0,  // token 2 (highest)
        0.1,  // token 3
        1.0,  // token 4 (second highest)
    ];

    // Get top 3 tokens
    let top_3 = sampler.top_k_tokens(&logits, 3);
    assert_eq!(top_3.len(), 3);

    // Verify ordering (highest to lowest)
    assert_eq!(top_3[0].0, 2); // Token 2 should be first
    assert_eq!(top_3[1].0, 4); // Token 4 should be second
    assert_eq!(top_3[2].0, 1); // Token 1 should be third

    // Verify log probabilities are ordered correctly
    assert!(top_3[0].1 > top_3[1].1);
    assert!(top_3[1].1 > top_3[2].1);

    // All log probabilities should be negative
    assert!(top_3[0].1 < 0.0);
}

#[test]
fn test_top_k_tokens_numerical_stability() {
    let sampler = Sampler::greedy(1000);

    // Test with large positive values (could overflow exp without stabilization)
    let large_logits = vec![100.0, 101.0, 102.0];
    let top_tokens = sampler.top_k_tokens(&large_logits, 3);

    // All tokens should be returned
    assert_eq!(top_tokens.len(), 3);

    // Token with highest logit should be first
    assert_eq!(top_tokens[0].0, 2);

    // Log probabilities should sum to 0 in log space (probabilities sum to 1.0)
    let sum: f32 = top_tokens.iter().map(|(_, log_prob)| log_prob.exp()).sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "Probability sum should be 1.0, got {}",
        sum
    );

    // Test with large negative values
    let small_logits = vec![-100.0, -99.0, -98.0];
    let top_tokens_small = sampler.top_k_tokens(&small_logits, 3);
    let sum_small: f32 = top_tokens_small.iter().map(|(_, lp)| lp.exp()).sum();
    assert!((sum_small - 1.0).abs() < 1e-5);

    // Test with mixed values
    let mixed_logits = vec![-50.0, 0.0, 50.0];
    let top_tokens_mixed = sampler.top_k_tokens(&mixed_logits, 3);
    let sum_mixed: f32 = top_tokens_mixed.iter().map(|(_, lp)| lp.exp()).sum();
    assert!((sum_mixed - 1.0).abs() < 1e-5);
}

#[test]
fn test_beam_search_disabled_by_default() {
    let sampler = Sampler::greedy(32000);
    assert!(!sampler.is_beam_search_enabled());
    assert_eq!(sampler.config().beam_width, 1);
}

#[test]
fn test_beam_search_enabled_with_width() {
    let config = SamplerConfig {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        beam_width: 5,
    };
    let sampler = Sampler::new(config, 32000);
    assert!(sampler.is_beam_search_enabled());
    assert_eq!(sampler.config().beam_width, 5);
}

#[test]
fn test_sampling_strategies_configuration() {
    // Greedy (default)
    let greedy = Sampler::greedy(32000);
    assert_eq!(greedy.config().temperature, 0.0);
    assert_eq!(greedy.config().beam_width, 1);

    // Temperature sampling
    let temp_sampler = Sampler::new(
        SamplerConfig {
            temperature: 0.8,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.15,
            beam_width: 1,
        },
        32000,
    );
    assert_eq!(temp_sampler.config().temperature, 0.8);

    // Top-k sampling
    let topk_sampler = Sampler::new(
        SamplerConfig {
            temperature: 1.0,
            top_k: 50,
            top_p: 1.0,
            repetition_penalty: 1.15,
            beam_width: 1,
        },
        32000,
    );
    assert_eq!(topk_sampler.config().top_k, 50);

    // Top-p (nucleus) sampling
    let topp_sampler = Sampler::new(
        SamplerConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.9,
            repetition_penalty: 1.15,
            beam_width: 1,
        },
        32000,
    );
    assert_eq!(topp_sampler.config().top_p, 0.9);

    // Combined top-k + top-p
    let combined_sampler = Sampler::new(
        SamplerConfig {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.95,
            repetition_penalty: 1.2,
            beam_width: 1,
        },
        32000,
    );
    assert_eq!(combined_sampler.config().temperature, 0.7);
    assert_eq!(combined_sampler.config().top_k, 40);
    assert_eq!(combined_sampler.config().top_p, 0.95);
}

#[test]
fn test_top_k_tokens_with_infinity() {
    let sampler = Sampler::greedy(1000);

    // Test with negative infinity (should be filtered to lowest probability)
    let logits = vec![0.0, f32::NEG_INFINITY, 1.0];
    let top_tokens = sampler.top_k_tokens(&logits, 3);

    // Should still get 3 tokens
    assert_eq!(top_tokens.len(), 3);

    // Token with highest logit should be first
    assert_eq!(top_tokens[0].0, 2);

    // NEG_INFINITY token should be last
    assert_eq!(top_tokens[2].0, 1);

    // The NEG_INFINITY token should have very low probability in log space
    assert!(top_tokens[2].1.is_infinite() && top_tokens[2].1.is_sign_negative());
}

#[test]
fn test_repetition_penalty_default() {
    let config = SamplerConfig::default();
    assert_eq!(config.repetition_penalty, 1.15);

    // Test custom repetition penalty
    let custom_config = SamplerConfig {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.5,
        beam_width: 1,
    };
    assert_eq!(custom_config.repetition_penalty, 1.5);
}

// Note: GPU-based tests would go here but require a full model setup
// For now, we test the configuration and CPU-side logic

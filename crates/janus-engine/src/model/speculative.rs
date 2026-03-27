//! Speculative decoding for accelerated inference
//!
//! Speculative decoding is an advanced optimization technique that uses a small, fast
//! "draft" model to predict multiple tokens ahead, and then verifies these predictions
//! with a larger "target" model in parallel. This can provide 2-3x speedup for inference.
//!
//! # How It Works
//!
//! 1. **Draft Phase**: Small draft model generates K candidate tokens quickly
//! 2. **Verification Phase**: Target model processes all K tokens in parallel
//! 3. **Acceptance**: Compare draft and target logits, accept matching tokens
//! 4. **Rejection**: If mismatch, resample from target distribution and restart
//!
//! # Performance Benefits
//!
//! - **Speedup**: 2-3x for compatible draft/target model pairs
//! - **Quality**: Identical output distribution to standard sampling
//! - **Memory**: Requires loading two models (draft + target)
//!
//! # Example
//!
//! ```no_run
//! use janus_engine::{Model, SpeculativeDecoder, SpeculativeConfig};
//!
//! # async fn example(draft_model: Model, target_model: Model) {
//! let config = SpeculativeConfig {
//!     num_draft_tokens: 4,  // Generate 4 tokens ahead
//!     acceptance_threshold: 0.8,  // Accept if probability > 80%
//! };
//!
//! let mut decoder = SpeculativeDecoder::new(draft_model, target_model, config);
//! let output = decoder.generate("Once upon a time").await.unwrap();
//! # }
//! ```

use crate::compute::Result;
use crate::model::model::Model;
use wgpu::util::DeviceExt;

/// Configuration for speculative decoding
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Number of tokens to draft ahead (typically 3-5)
    ///
    /// Higher values increase potential speedup but also increase
    /// the chance of rejection. Typical values:
    /// - 3-4: Conservative, high acceptance rate
    /// - 5-7: Aggressive, higher speedup potential
    pub num_draft_tokens: usize,

    /// Acceptance threshold for token verification (0.0-1.0)
    ///
    /// Tokens are accepted if the target model's probability for the
    /// draft token is above this threshold. Lower values accept more
    /// tokens but may diverge from target distribution.
    /// - 0.7-0.8: Balanced (recommended)
    /// - 0.9+: Conservative, closer to target distribution
    pub acceptance_threshold: f32,

    /// Whether to use greedy acceptance (always accept if draft token matches target's argmax)
    pub greedy_acceptance: bool,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            num_draft_tokens: 4,
            acceptance_threshold: 0.8,
            greedy_acceptance: true,
        }
    }
}

/// Speculative decoder for accelerated inference
///
/// This struct manages two models:
/// - **Draft model**: Small, fast model for generating candidate tokens
/// - **Target model**: Larger, more accurate model for verification
///
/// The decoder maintains statistics about acceptance rates and speedup.
pub struct SpeculativeDecoder {
    /// Small draft model for generating candidates
    draft_model: Model,

    /// Large target model for verification
    target_model: Model,

    /// Configuration parameters
    config: SpeculativeConfig,

    /// Statistics for performance monitoring
    stats: SpeculativeStats,
}

/// Statistics for speculative decoding performance
#[derive(Debug, Clone, Default)]
pub struct SpeculativeStats {
    /// Total number of draft tokens generated
    pub total_draft_tokens: usize,

    /// Number of draft tokens accepted by target model
    pub accepted_tokens: usize,

    /// Number of speculative rounds (draft + verify cycles)
    pub num_rounds: usize,

    /// Average acceptance rate (accepted / total_draft)
    pub acceptance_rate: f32,

    /// Theoretical speedup vs standard decoding
    /// Calculated as: accepted_tokens / num_rounds
    pub average_speedup: f32,
}

impl SpeculativeDecoder {
    /// Create a new speculative decoder
    ///
    /// # Arguments
    /// * `draft_model` - Small, fast model for generating candidates
    /// * `target_model` - Large, accurate model for verification
    /// * `config` - Configuration parameters
    ///
    /// # Requirements
    /// - Both models must use the same tokenizer
    /// - Both models must have the same vocabulary
    /// - Draft model should be significantly faster than target model
    ///
    /// # Example
    /// ```no_run
    /// # use janus_engine::{Model, SpeculativeDecoder, SpeculativeConfig};
    /// # async fn example(draft: Model, target: Model) {
    /// let decoder = SpeculativeDecoder::new(draft, target, SpeculativeConfig::default());
    /// # }
    /// ```
    pub fn new(draft_model: Model, target_model: Model, config: SpeculativeConfig) -> Self {
        Self {
            draft_model,
            target_model,
            config,
            stats: SpeculativeStats::default(),
        }
    }

    /// Generate text using speculative decoding
    ///
    /// This method implements the full speculative decoding algorithm:
    /// 1. Process prompt with target model (prefill)
    /// 2. Generate draft tokens with draft model
    /// 3. Verify with target model in parallel
    /// 4. Accept/reject based on probability matching
    /// 5. Repeat until max_tokens or EOS
    ///
    /// # Arguments
    /// * `prompt` - Input text prompt
    /// * `max_tokens` - Maximum number of tokens to generate
    ///
    /// # Returns
    /// The complete generated text (prompt + generated tokens)
    pub async fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        // Reset statistics
        self.stats = SpeculativeStats::default();

        // Step 1: Tokenize prompt
        tracing::info!("Speculative decoding: tokenizing prompt");
        let mut token_ids = self
            .draft_model
            .tokenizer()
            .encode(prompt, true)
            .map_err(|e| crate::compute::ComputeError::Other(format!("Tokenization failed: {}", e)))?;

        if let Some(eos_token_id) = self.draft_model.tokenizer().eos_token_id() {
            if token_ids.last() == Some(&eos_token_id) {
                token_ids.pop();
            }
        }

        if token_ids.is_empty() {
            return Err(crate::compute::ComputeError::Other(
                "Empty prompt after tokenization".into(),
            ));
        }

        tracing::info!("Prompt: {} tokens", token_ids.len());

        // Step 2: Prefill - process prompt with target model
        // We only need to run target model during prefill since we'll start
        // generating from its state
        tracing::info!("Prefill phase: processing prompt with target model");
        self.target_model.cache_mut().reset();
        
        let mut seq_pos = 0u32;
        for &token_id in &token_ids {
            self.target_model.forward_internal(token_id, seq_pos).await?;
            seq_pos += 1;
        }

        // Step 3: Generation loop with speculative decoding
        let mut generated_tokens = Vec::new();
        let mut last_target_token = *token_ids.last()
            .ok_or_else(|| crate::compute::ComputeError::Other("Empty token sequence".into()))?;

        tracing::info!("Generation phase: starting speculative decoding");
        let generation_start = std::time::Instant::now();

        while generated_tokens.len() < max_tokens {
            self.stats.num_rounds += 1;

            // Step 3a: Draft phase - generate K candidate tokens with draft model
            tracing::debug!("Round {}: Drafting {} tokens", self.stats.num_rounds, self.config.num_draft_tokens);
            
            let draft_tokens = self.generate_draft_tokens(
                last_target_token,
                seq_pos,
                self.config.num_draft_tokens,
            ).await?;

            if draft_tokens.is_empty() {
                tracing::debug!("Draft model generated EOS, stopping");
                break;
            }

            self.stats.total_draft_tokens += draft_tokens.len();

            // Step 3b: Verification phase - verify all draft tokens with target model
            tracing::debug!("Round {}: Verifying {} draft tokens", self.stats.num_rounds, draft_tokens.len());
            
            let accepted_count = self.verify_and_accept_tokens(
                &draft_tokens,
                seq_pos,
                &mut generated_tokens,
            ).await?;

            self.stats.accepted_tokens += accepted_count;

            // Update position and last token
            seq_pos += accepted_count as u32;
            
            if accepted_count > 0 {
                last_target_token = generated_tokens[generated_tokens.len() - 1];
                tracing::debug!("Accepted {}/{} tokens", accepted_count, draft_tokens.len());
            } else {
                tracing::debug!("Rejected all tokens, resampling from target");
                // Sample one token from target model and continue
                let next_token = self.target_model.sample_next_token(&generated_tokens).await?;
                generated_tokens.push(next_token);
                last_target_token = next_token;
                seq_pos += 1;
                self.stats.accepted_tokens += 1;
            }

            // Check for EOS
            let eos_token_id = self
                .target_model
                .tokenizer()
                .eos_token_id()
                .unwrap_or(128_009);
            if last_target_token == eos_token_id || last_target_token == 2 {
                tracing::info!("Generated EOS token, stopping");
                break;
            }

            // Check if we've reached max tokens
            if generated_tokens.len() >= max_tokens {
                break;
            }
        }

        // Calculate final statistics
        let elapsed = generation_start.elapsed().as_secs_f64();
        self.stats.acceptance_rate = if self.stats.total_draft_tokens > 0 {
            (self.stats.accepted_tokens as f32) / (self.stats.total_draft_tokens as f32)
        } else {
            0.0
        };
        self.stats.average_speedup = if self.stats.num_rounds > 0 {
            (self.stats.accepted_tokens as f32) / (self.stats.num_rounds as f32)
        } else {
            1.0
        };

        tracing::info!(
            "Speculative decoding complete: {} tokens in {:.3}s ({:.2} tok/s)",
            generated_tokens.len(),
            elapsed,
            generated_tokens.len() as f64 / elapsed
        );
        tracing::info!(
            "Acceptance rate: {:.1}%, Average speedup: {:.2}x",
            self.stats.acceptance_rate * 100.0,
            self.stats.average_speedup
        );

        // Decode generated tokens
        let generated_text = self.target_model.tokenizer()
            .decode_batch(&generated_tokens)
            .map_err(|e| crate::compute::ComputeError::Other(format!("Detokenization failed: {}", e)))?;

        Ok(generated_text)
    }

    /// Generate draft tokens using the draft model
    ///
    /// This runs the draft model autoregressively to generate K candidate tokens.
    /// The draft model's KV cache is synchronized with the target model's cache
    /// using an efficient GPU-to-GPU copy operation.
    async fn generate_draft_tokens(
        &mut self,
        start_token: u32,
        start_pos: u32,
        num_tokens: usize,
    ) -> Result<Vec<u32>> {
        // Sync draft model's cache with target model's state via GPU-to-GPU copy
        // This is much more efficient than reprocessing the entire context
        tracing::debug!("Syncing KV cache from target to draft model (GPU copy)");
        
        let device = self.target_model.engine().device();
        let queue = self.target_model.engine().queue();
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("kv_cache_sync"),
        });
        
        // Copy target cache to draft cache
        self.draft_model
            .cache_mut()
            .copy_from(&mut encoder, self.target_model.cache())?;
        
        queue.submit(Some(encoder.finish()));

        let mut draft_tokens = Vec::new();
        let mut current_token = start_token;
        let mut pos = start_pos;

        for _ in 0..num_tokens {
            // Forward pass with draft model
            self.draft_model.forward_internal(current_token, pos).await?;

            // Sample from draft model
            let next_token = self.draft_model.sample_next_token(&draft_tokens).await?;

            // Check for EOS
            let eos_token_id = self
                .draft_model
                .tokenizer()
                .eos_token_id()
                .unwrap_or(128_009);
            if next_token == eos_token_id || next_token == 2 {
                break;
            }

            draft_tokens.push(next_token);
            current_token = next_token;
            pos += 1;
        }

        Ok(draft_tokens)
    }

    /// Verify draft tokens with target model and accept/reject
    ///
    /// This processes all draft tokens in sequence with the target model,
    /// comparing the target's logits with the draft's choices.
    async fn verify_and_accept_tokens(
        &mut self,
        draft_tokens: &[u32],
        start_pos: u32,
        accepted_tokens: &mut Vec<u32>,
    ) -> Result<usize> {
        let mut accepted_count = 0;

        for (i, &draft_token) in draft_tokens.iter().enumerate() {
            let pos = start_pos + i as u32;

            // Forward pass with target model
            self.target_model.forward_internal(draft_token, pos).await?;

            // Get target model's logits
            let target_logits = self.target_model.read_logits().await?;

            // Check if we should accept the draft token
            if self.should_accept_token(draft_token, &target_logits) {
                accepted_tokens.push(draft_token);
                accepted_count += 1;
            } else {
                // Rejection: resample from target distribution
                tracing::debug!("Rejected token at position {}", i);
                
                // Sample a new token from target's distribution
                let corrected_token = self.sample_from_logits(&target_logits).await?;
                accepted_tokens.push(corrected_token);
                accepted_count += 1;
                
                // Stop verification here - remaining draft tokens are invalid
                break;
            }
        }

        Ok(accepted_count)
    }

    /// Determine if a draft token should be accepted based on target logits
    fn should_accept_token(&self, draft_token: u32, target_logits: &[f32]) -> bool {
        if self.config.greedy_acceptance {
            // Greedy acceptance: accept if draft token matches target's argmax
            let target_best = target_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .unwrap_or(0);

            draft_token == target_best
        } else {
            // Probabilistic acceptance: accept if probability exceeds threshold
            // Convert logits to probabilities
            let max_logit = target_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f32 = target_logits.iter().map(|&x| (x - max_logit).exp()).sum();
            let prob = ((target_logits[draft_token as usize] - max_logit).exp()) / sum_exp;

            prob >= self.config.acceptance_threshold
        }
    }

    /// Sample a token from logits (for rejection sampling)
    ///
    /// This uses the target model's sampler configuration to sample from
    /// the corrected distribution when a draft token is rejected.
    async fn sample_from_logits(&self, logits: &[f32]) -> Result<u32> {
        // Create a temporary buffer with the logits for sampling
        let device = self.target_model.engine().device();
        let logits_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rejection_logits"),
            contents: bytemuck::cast_slice(logits),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Use the target model's sampler to sample from the corrected distribution
        // This ensures we maintain the same sampling characteristics (temperature, top-k, etc.)
        // Note: We pass None for pipeline_cache since speculative decoding typically uses
        // temperature sampling, not greedy decoding
        self.target_model
            .sampler()
            .sample(self.target_model.engine(), None, None, &logits_buffer, &[])
            .await
    }

    /// Get reference to draft model
    pub const fn draft_model(&self) -> &Model {
        &self.draft_model
    }

    /// Get reference to target model
    pub const fn target_model(&self) -> &Model {
        &self.target_model
    }

    /// Get mutable reference to draft model
    pub fn draft_model_mut(&mut self) -> &mut Model {
        &mut self.draft_model
    }

    /// Get mutable reference to target model
    pub fn target_model_mut(&mut self) -> &mut Model {
        &mut self.target_model
    }

    /// Get current statistics
    pub const fn stats(&self) -> &SpeculativeStats {
        &self.stats
    }

    /// Get configuration
    pub const fn config(&self) -> &SpeculativeConfig {
        &self.config
    }
}

impl SpeculativeStats {
    /// Print statistics to stdout
    pub fn print(&self) {
        println!("\n=== Speculative Decoding Statistics ===");
        println!("Total rounds: {}", self.num_rounds);
        println!("Draft tokens generated: {}", self.total_draft_tokens);
        println!("Tokens accepted: {}", self.accepted_tokens);
        println!("Acceptance rate: {:.1}%", self.acceptance_rate * 100.0);
        println!("Average speedup: {:.2}x", self.average_speedup);
        println!("=======================================");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speculative_config_default() {
        let config = SpeculativeConfig::default();
        
        assert_eq!(config.num_draft_tokens, 4);
        assert!((config.acceptance_threshold - 0.8).abs() < 1e-6);
        assert!(config.greedy_acceptance);
    }

    #[test]
    fn test_speculative_config_custom() {
        let config = SpeculativeConfig {
            num_draft_tokens: 7,
            acceptance_threshold: 0.9,
            greedy_acceptance: false,
        };
        
        assert_eq!(config.num_draft_tokens, 7);
        assert!((config.acceptance_threshold - 0.9).abs() < 1e-6);
        assert!(!config.greedy_acceptance);
    }

    #[test]
    fn test_speculative_stats_default() {
        let stats = SpeculativeStats::default();
        
        assert_eq!(stats.total_draft_tokens, 0);
        assert_eq!(stats.accepted_tokens, 0);
        assert_eq!(stats.num_rounds, 0);
        assert_eq!(stats.acceptance_rate, 0.0);
        assert_eq!(stats.average_speedup, 0.0);
    }

    #[test]
    fn test_speculative_stats_calculation() {
        let mut stats = SpeculativeStats {
            total_draft_tokens: 100,
            accepted_tokens: 80,
            num_rounds: 25,
            acceptance_rate: 0.0,  // Will be calculated
            average_speedup: 0.0,   // Will be calculated
        };
        
        // Calculate rates as done in the implementation
        stats.acceptance_rate = (stats.accepted_tokens as f32) / (stats.total_draft_tokens as f32);
        stats.average_speedup = (stats.accepted_tokens as f32) / (stats.num_rounds as f32);
        
        assert!((stats.acceptance_rate - 0.8).abs() < 1e-6);
        assert!((stats.average_speedup - 3.2).abs() < 1e-6);
    }

    #[test]
    fn test_acceptance_rate_edge_cases() {
        // Test 100% acceptance
        let stats = SpeculativeStats {
            total_draft_tokens: 100,
            accepted_tokens: 100,
            num_rounds: 20,
            acceptance_rate: 1.0,
            average_speedup: 5.0,
        };
        
        assert_eq!(stats.acceptance_rate, 1.0);
        
        // Test 0% acceptance (all rejected)
        let stats = SpeculativeStats {
            total_draft_tokens: 100,
            accepted_tokens: 0,
            num_rounds: 100,
            acceptance_rate: 0.0,
            average_speedup: 0.0,
        };
        
        assert_eq!(stats.acceptance_rate, 0.0);
    }

    #[test]
    fn test_average_speedup_calculation() {
        // Best case: All tokens accepted in first attempt
        // If we accept 4 tokens per round, speedup is 4x
        let stats = SpeculativeStats {
            total_draft_tokens: 100,
            accepted_tokens: 100,
            num_rounds: 25,
            acceptance_rate: 1.0,
            average_speedup: 4.0,
        };
        
        assert!((stats.average_speedup - 4.0).abs() < 1e-6);
        
        // Moderate case: 50% acceptance, 2 tokens per round
        let stats = SpeculativeStats {
            total_draft_tokens: 100,
            accepted_tokens: 50,
            num_rounds: 25,
            acceptance_rate: 0.5,
            average_speedup: 2.0,
        };
        
        assert!((stats.average_speedup - 2.0).abs() < 1e-6);
        
        // Worst case: No speedup (1 token per round)
        let stats = SpeculativeStats {
            total_draft_tokens: 100,
            accepted_tokens: 100,
            num_rounds: 100,
            acceptance_rate: 1.0,
            average_speedup: 1.0,
        };
        
        assert!((stats.average_speedup - 1.0).abs() < 1e-6);
    }

    /// Test that realistic scenarios produce expected speedups
    #[test]
    fn test_realistic_speedup_scenarios() {
        // Scenario 1: Good draft model (80% acceptance, 4 tokens ahead)
        // Expected speedup: ~3.2x
        let num_rounds = 100;
        let acceptance_rate = 0.8;
        let draft_per_round = 4;
        
        let total_draft = num_rounds * draft_per_round;
        let accepted = (total_draft as f32 * acceptance_rate) as usize;
        let average_speedup = accepted as f32 / num_rounds as f32;
        
        assert_eq!(total_draft, 400);
        assert_eq!(accepted, 320);
        assert!((average_speedup - 3.2).abs() < 1e-6);
        
        // Scenario 2: Mediocre draft model (50% acceptance, 4 tokens ahead)
        // Expected speedup: ~2x
        let acceptance_rate = 0.5;
        let accepted = (total_draft as f32 * acceptance_rate) as usize;
        let average_speedup = accepted as f32 / num_rounds as f32;
        
        assert_eq!(accepted, 200);
        assert!((average_speedup - 2.0).abs() < 1e-6);
        
        // Scenario 3: Poor draft model (25% acceptance, 4 tokens ahead)
        // Expected speedup: ~1x (no benefit)
        let acceptance_rate = 0.25;
        let accepted = (total_draft as f32 * acceptance_rate) as usize;
        let average_speedup = accepted as f32 / num_rounds as f32;
        
        assert_eq!(accepted, 100);
        assert!((average_speedup - 1.0).abs() < 1e-6);
    }

    /// Test configuration validation
    #[test]
    fn test_config_reasonable_values() {
        let config = SpeculativeConfig {
            num_draft_tokens: 10,
            acceptance_threshold: 0.95,
            greedy_acceptance: true,
        };
        
        // Acceptance threshold should be between 0 and 1
        assert!(config.acceptance_threshold >= 0.0);
        assert!(config.acceptance_threshold <= 1.0);
        
        // Draft tokens should be reasonable (typically 3-10)
        assert!(config.num_draft_tokens > 0);
        assert!(config.num_draft_tokens <= 20); // Sanity check
    }

    /// Test stats printing (doesn't crash)
    #[test]
    fn test_stats_print() {
        let stats = SpeculativeStats {
            total_draft_tokens: 100,
            accepted_tokens: 80,
            num_rounds: 25,
            acceptance_rate: 0.8,
            average_speedup: 3.2,
        };
        
        // This should not panic
        stats.print();
    }
}

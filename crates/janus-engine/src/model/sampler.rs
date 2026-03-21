//! Token sampling and decoding
//!
//! This module implements token sampling strategies for autoregressive text generation.
//! Supports:
//! - Greedy decoding (argmax selection)
//! - Temperature sampling
//! - Top-k sampling
//! - Top-p (nucleus) sampling
//! - Configurable repetition penalty
//!
//! Future enhancements:
//! - Beam search
//! - Mirostat sampling

use crate::compute::{ComputeEngine, Result};

/// Sampling strategy configuration
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    /// Sampling temperature (0.0 = greedy/argmax)
    pub temperature: f32,
    /// Top-k filtering (0 = disabled)
    pub top_k: u32,
    /// Top-p (nucleus) filtering (1.0 = disabled)
    pub top_p: f32,
    /// Repetition penalty (1.0 = no penalty, higher values penalize repetition more)
    pub repetition_penalty: f32,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0, // Greedy decoding by default
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.15,
        }
    }
}

/// Token sampler for selecting next tokens from logits
///
/// The sampler takes logits (vocabulary-sized probability distributions)
/// and selects the next token based on the configured sampling strategy.
pub struct Sampler {
    config: SamplerConfig,
    vocab_size: u32,
}

impl Sampler {
    /// Create a new sampler with the given configuration
    ///
    /// # Arguments
    /// * `config` - Sampling strategy configuration
    /// * `vocab_size` - Size of the model's vocabulary (e.g., 32000 for LLaMA)
    pub fn new(config: SamplerConfig, vocab_size: u32) -> Self {
        Self { config, vocab_size }
    }

    /// Create a sampler with greedy decoding (temperature = 0.0)
    ///
    /// # Arguments
    /// * `vocab_size` - Size of the model's vocabulary
    pub fn greedy(vocab_size: u32) -> Self {
        Self::new(SamplerConfig::default(), vocab_size)
    }

    /// Sample the next token from logits
    ///
    /// This function:
    /// 1. Reads the logits buffer from GPU to CPU
    /// 2. Applies repetition penalty based on context
    /// 3. Applies the sampling strategy (greedy, temperature, top-k, top-p)
    /// 4. Returns the selected token ID
    ///
    /// # Arguments
    /// * `engine` - The compute engine for GPU operations
    /// * `logits_buffer` - GPU buffer containing logits [vocab_size] floats
    /// * `context` - Previously generated tokens for repetition penalty
    ///
    /// # Returns
    /// The selected token ID (0 to vocab_size - 1)
    ///
    /// # Sampling Strategies
    /// - If temperature == 0.0: Greedy decoding (argmax)
    /// - If temperature > 0.0: Temperature sampling with optional top-k and top-p filtering
    pub async fn sample(
        &self,
        engine: &ComputeEngine,
        logits_buffer: &wgpu::Buffer,
        context: &[u32],
    ) -> Result<u32> {
        // Read logits from GPU to CPU
        let mut logits = self.read_logits_from_gpu(engine, logits_buffer).await?;

        // Apply repetition penalty to prevent infinite loops
        self.apply_repetition_penalty(&mut logits, context);

        // Apply sampling strategy
        let token_id = if self.config.temperature == 0.0 {
            // Greedy decoding: select token with highest logit
            self.argmax(&logits)
        } else {
            // Temperature sampling with optional top-k and top-p filtering
            self.sample_with_temperature(&mut logits)
        };

        Ok(token_id)
    }

    /// Read logits from GPU buffer to CPU memory
    ///
    /// # Arguments
    /// * `engine` - The compute engine
    /// * `logits_buffer` - GPU buffer containing logits
    ///
    /// # Returns
    /// Vector of logits as f32 values
    async fn read_logits_from_gpu(
        &self,
        engine: &ComputeEngine,
        logits_buffer: &wgpu::Buffer,
    ) -> Result<Vec<f32>> {
        let device = engine.device();
        let queue = engine.queue();

        // Create staging buffer for reading back to CPU
        let buffer_size = (self.vocab_size * std::mem::size_of::<f32>() as u32) as u64;
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("logits_staging_buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy from GPU buffer to staging buffer
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("logits_copy_encoder"),
        });
        encoder.copy_buffer_to_buffer(logits_buffer, 0, &staging_buffer, 0, buffer_size);
        queue.submit(Some(encoder.finish()));

        // Map the staging buffer for reading
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            // Ignore send errors - if receiver is dropped, we'll catch it below
            let _ = sender.send(result);
        });

        // Wait for the buffer to be mapped
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        
        // Handle both receiver and mapping errors properly
        let map_result = receiver
            .await
            .map_err(|_| crate::compute::ComputeError::BufferMappingFailed)?;
        
        map_result.map_err(|_| crate::compute::ComputeError::BufferMappingFailed)?;

        // Read the data
        let data = buffer_slice.get_mapped_range();
        let logits: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

        // Clean up
        drop(data);
        staging_buffer.unmap();

        Ok(logits)
    }

    /// Apply repetition penalty to logits to prevent infinite loops
    ///
    /// This penalizes tokens that have already been generated in the context,
    /// reducing the likelihood of repetitive output.
    ///
    /// # Arguments
    /// * `logits` - Mutable slice of logit values to modify
    /// * `context` - Previously generated token IDs
    ///
    /// # Implementation
    /// For each token in the context:
    /// - If its logit is positive, divide by penalty (reduces probability)
    /// - If its logit is negative, multiply by penalty (makes it more negative)
    fn apply_repetition_penalty(&self, logits: &mut [f32], context: &[u32]) {
        let rep_penalty = self.config.repetition_penalty;
        
        for &token_id in context {
            let idx = token_id as usize;
            if idx < logits.len() {
                let logit = &mut logits[idx];
                if *logit > 0.0 {
                    *logit /= rep_penalty;
                } else {
                    *logit *= rep_penalty;
                }
            }
        }
    }

    /// Sample using temperature with optional top-k and top-p filtering
    ///
    /// # Arguments
    /// * `logits` - Mutable slice of logit values
    ///
    /// # Returns
    /// Sampled token ID
    fn sample_with_temperature(&self, logits: &mut [f32]) -> u32 {
        // Apply top-k filtering if enabled
        if self.config.top_k > 0 {
            self.apply_top_k(logits, self.config.top_k);
        }

        // Apply temperature scaling
        let temperature = self.config.temperature;
        for logit in logits.iter_mut() {
            *logit /= temperature;
        }

        // Convert logits to probabilities using softmax
        let probs = self.softmax(logits);

        // Apply top-p (nucleus) filtering if enabled
        let probs = if self.config.top_p < 1.0 {
            self.apply_top_p(&probs, self.config.top_p)
        } else {
            probs
        };

        // Sample from the probability distribution
        self.sample_from_distribution(&probs)
    }

    /// Apply top-k filtering by setting logits of tokens outside top-k to -infinity
    ///
    /// # Arguments
    /// * `logits` - Mutable slice of logit values
    /// * `k` - Number of top tokens to keep
    fn apply_top_k(&self, logits: &mut [f32], k: u32) {
        if k == 0 || k >= logits.len() as u32 {
            return;
        }

        // Find the k-th largest value
        let mut indices: Vec<usize> = (0..logits.len()).collect();
        indices.sort_by(|&a, &b| {
            logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Set all values outside top-k to negative infinity
        let threshold_idx = indices[k as usize - 1];
        let threshold = logits[threshold_idx];
        
        for (i, logit) in logits.iter_mut().enumerate() {
            if *logit < threshold && i != threshold_idx {
                *logit = f32::NEG_INFINITY;
            }
        }
    }

    /// Apply top-p (nucleus) filtering
    ///
    /// # Arguments
    /// * `probs` - Probability distribution
    /// * `p` - Cumulative probability threshold
    ///
    /// # Returns
    /// Filtered probability distribution
    fn apply_top_p(&self, probs: &[f32], p: f32) -> Vec<f32> {
        // Create sorted indices by probability (descending)
        let mut indices: Vec<usize> = (0..probs.len()).collect();
        indices.sort_by(|&a, &b| {
            probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Find cumulative probability cutoff
        let mut cumsum = 0.0;
        let mut cutoff_idx = probs.len();
        for (i, &idx) in indices.iter().enumerate() {
            cumsum += probs[idx];
            if cumsum >= p {
                cutoff_idx = i + 1;
                break;
            }
        }

        // Create filtered distribution
        let mut filtered = vec![0.0; probs.len()];
        let mut sum = 0.0;
        for &idx in indices.iter().take(cutoff_idx) {
            filtered[idx] = probs[idx];
            sum += probs[idx];
        }

        // Renormalize
        if sum > 0.0 {
            for prob in filtered.iter_mut() {
                *prob /= sum;
            }
        }

        filtered
    }

    /// Compute softmax of logits to get probabilities
    ///
    /// # Arguments
    /// * `logits` - Slice of logit values
    ///
    /// # Returns
    /// Probability distribution
    fn softmax(&self, logits: &[f32]) -> Vec<f32> {
        // Find max for numerical stability
        let max_logit = logits.iter()
            .fold(f32::NEG_INFINITY, |max, &x| if x > max { x } else { max });

        // Compute exp(logit - max) and sum
        let mut exp_sum = 0.0;
        let exp_logits: Vec<f32> = logits.iter()
            .map(|&x| {
                if x.is_finite() {
                    let exp_val = (x - max_logit).exp();
                    exp_sum += exp_val;
                    exp_val
                } else {
                    0.0
                }
            })
            .collect();

        // Normalize to get probabilities
        exp_logits.iter().map(|&x| x / exp_sum).collect()
    }

    /// Sample a token from a probability distribution
    ///
    /// # Arguments
    /// * `probs` - Probability distribution
    ///
    /// # Returns
    /// Sampled token ID
    fn sample_from_distribution(&self, probs: &[f32]) -> u32 {
        use rand::Rng;
        
        let mut rng = rand::thread_rng();
        let random_value: f32 = rng.r#gen();
        
        let mut cumsum = 0.0;
        for (i, &prob) in probs.iter().enumerate() {
            cumsum += prob;
            if random_value < cumsum {
                return i as u32;
            }
        }
        
        // Fallback to last token (shouldn't happen with proper probabilities)
        (probs.len() - 1) as u32
    }

    /// Find the index of the maximum value (argmax)
    ///
    /// This implements greedy decoding by selecting the token with
    /// the highest logit value.
    ///
    /// # Arguments
    /// * `logits` - Slice of logit values
    ///
    /// # Returns
    /// Index of the maximum value (token ID)
    fn argmax(&self, logits: &[f32]) -> u32 {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                // Handle NaN by treating it as negative infinity
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less)
            })
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0)
    }

    /// Get the current sampler configuration
    pub const fn config(&self) -> &SamplerConfig {
        &self.config
    }

    /// Get the vocabulary size
    pub const fn vocab_size(&self) -> u32 {
        self.vocab_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argmax() {
        let sampler = Sampler::greedy(1000);

        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        assert_eq!(sampler.argmax(&logits), 3);

        let logits = vec![5.0, 1.0, 3.0];
        assert_eq!(sampler.argmax(&logits), 0);

        let logits = vec![-1.0, -5.0, -0.5];
        assert_eq!(sampler.argmax(&logits), 2);
    }

    #[test]
    fn test_argmax_with_nan() {
        let sampler = Sampler::greedy(1000);

        let logits = vec![0.1, f32::NAN, 0.3, 0.2];
        assert_eq!(sampler.argmax(&logits), 2);
    }

    #[test]
    fn test_sampler_config_default() {
        let config = SamplerConfig::default();
        assert_eq!(config.temperature, 0.0);
        assert_eq!(config.top_k, 0);
        assert_eq!(config.top_p, 1.0);
    }

    #[test]
    fn test_greedy_sampler_creation() {
        let sampler = Sampler::greedy(32000);
        assert_eq!(sampler.vocab_size(), 32000);
        assert_eq!(sampler.config().temperature, 0.0);
    }
}

//! Helper methods for speculative decoding

use super::Model;
use crate::compute::Result;

impl Model {
    /// Forward pass (internal, for speculative decoding)
    ///
    /// This is an internal method exposed for speculative decoding.
    /// It performs a forward pass without sampling.
    ///
    /// # Arguments
    /// * `token_id` - Input token ID
    /// * `seq_pos` - Position in the sequence
    pub(crate) async fn forward_internal(&mut self, token_id: u32, seq_pos: u32) -> Result<()> {
        self.forward(token_id, seq_pos).await
    }

    /// Read logits from GPU (internal, for speculative decoding)
    ///
    /// This reads the current logits buffer from GPU to CPU memory.
    /// Used by speculative decoding to compare draft and target distributions.
    pub(crate) async fn read_logits(&self) -> Result<Vec<f32>> {
        let device = self.engine.device();
        let queue = self.engine.queue();

        // Create staging buffer
        let buffer_size = (self.config.vocab_size * std::mem::size_of::<f32>() as u32) as u64;
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("speculative_logits_staging"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy from GPU to staging
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("speculative_logits_copy"),
        });
        encoder.copy_buffer_to_buffer(&self.logits_buf, 0, &staging_buffer, 0, buffer_size);
        queue.submit(Some(encoder.finish()));

        // Map and read
        println!("🔍 [Debug] About to map buffer. Ensuring queue is submitted...");
        let buffer_slice = staging_buffer.slice(..);
        let (sender, mut receiver) = tokio::sync::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        let mut spin_count: u64 = 0;
        let map_result = loop {
            let _ = device.poll(wgpu::PollType::Poll);
            match receiver.try_recv() {
                Ok(res) => break res,
                Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                    spin_count += 1;
                    if spin_count.is_multiple_of(10_000) {
                        println!(
                            "⚠️ [Debug] Polling loop has spun {} times. GPU callback is not firing.",
                            spin_count
                        );
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                }
                Err(tokio::sync::oneshot::error::TryRecvError::Closed) => {
                    panic!("FATAL: WebGPU aborted!");
                }
            }
        };
        
        map_result.map_err(|_| crate::compute::ComputeError::BufferMappingFailed)?;

        let data = buffer_slice.get_mapped_range();
        let logits: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        staging_buffer.unmap();

        Ok(logits)
    }

    /// Sample next token from current logits (internal, for speculative decoding)
    ///
    /// This samples a token from the current logits buffer using the model's sampler.
    ///
    /// # Arguments
    /// * `context` - Previously generated tokens for repetition penalty
    pub(crate) async fn sample_next_token(&self, context: &[u32]) -> Result<u32> {
        self
            .sampler
            .sample(
                &self.engine,
                Some(&self.pipeline_cache),
                Some((
                    &self.argmax_output_buf,
                    &self.argmax_staging_buf,
                    &self.argmax_bind_group,
                )),
                &self.logits_buf,
                context,
            )
            .await
    }

    /// Get mutable reference to KV cache (for cache synchronization)
    pub(crate) fn cache_mut(&mut self) -> &mut crate::compute::cache::KVCache {
        &mut self.cache
    }

    /// Get reference to sampler (for speculative decoding rejection sampling)
    pub(crate) const fn sampler(&self) -> &crate::model::sampler::Sampler {
        &self.sampler
    }
}

//! Forward pass operations

use super::Model;
use crate::compute::ops::{gemm, rmsnorm};
use crate::compute::Result;

impl Model {
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
    pub(super) async fn forward(&mut self, token_id: u32, seq_pos: u32) -> Result<()> {
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
            1, // M = 1 token (single-token generation)
            self.config.hidden_dim, // K = input dimension
            self.config.vocab_size, // N = output dimension (weight is [vocab_size, hidden_dim])
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
    pub(super) async fn forward_batch(&mut self, token_ids: &[u32], seq_pos: u32) -> Result<()> {
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
            1, // M = 1 token per sequence (batch dimension is separate)
            self.config.hidden_dim, // K = input dimension
            self.config.vocab_size, // N = output dimension (weight is [vocab_size, hidden_dim])
        )?;

        // Step 5: Submit all operations in a SINGLE batch
        tracing::debug!("Submitting batched forward pass (single GPU submission)");
        self.engine.queue().submit(Some(encoder.finish()));

        Ok(())
    }
}

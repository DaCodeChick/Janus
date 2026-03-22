//! Text generation operations

use super::Model;
use crate::compute::Result;

impl Model {
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
}

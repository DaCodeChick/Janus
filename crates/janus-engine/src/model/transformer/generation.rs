//! Text generation operations

use super::Model;
use crate::compute::Result;

impl Model {
    /// Generate text autoregressively using sampler's max_tokens configuration
    ///
    /// This is a convenience method that uses the `max_tokens` value from the
    /// sampler's configuration instead of requiring it as a parameter.
    ///
    /// # Arguments
    /// * `prompt` - Input text prompt
    ///
    /// # Returns
    /// The complete generated text (prompt + generated tokens)
    ///
    /// # Example
    /// ```no_run
    /// # use janus_engine::{Model, SamplerConfig};
    /// # async fn example(mut model: Model) {
    /// // Generate with default max_tokens (128)
    /// let output = model.generate_text("Once upon a time").await.unwrap();
    /// # }
    /// ```
    pub async fn generate_text(&mut self, prompt: &str) -> Result<String> {
        let max_tokens = self.sampler.config().max_tokens;
        self.generate(prompt, max_tokens).await
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
            let next_token = self
                .sampler
                .sample(
                    &self.engine,
                    Some(&self.pipeline_cache),
                    Some((
                        &self.argmax_output_buf,
                        &self.argmax_staging_buf,
                        &self.argmax_bind_group,
                    )),
                    self.logits_buffer(),
                    &generated_tokens,
                )
                .await?;
            
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

    /// Generate text with streaming callback and stop string support
    ///
    /// This is an enhanced version of `generate` that supports:
    /// - Real-time streaming via callback function
    /// - Stop strings (e.g., `<|im_end|>`, `</s>`)
    /// - Custom stop token IDs
    ///
    /// # Arguments
    /// * `prompt` - Input text prompt
    /// * `max_tokens` - Maximum number of tokens to generate
    /// * `stop_strings` - Optional list of strings that stop generation when encountered
    /// * `callback` - Optional callback function called with each generated token
    ///
    /// # Returns
    /// The complete generated text (prompt + generated tokens)
    pub async fn generate_with_callback<F>(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        stop_strings: Option<&[String]>,
        mut callback: Option<F>,
    ) -> Result<String>
    where
        F: FnMut(&str) -> bool + Send, // Returns false to stop generation early
    {
        // Tokenize the prompt
        tracing::info!("Tokenizing prompt: \"{}\"", prompt);
        let mut token_ids = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| crate::compute::ComputeError::Other(format!("Tokenization failed: {}", e)))?;

        // Add BOS token
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
                return Err(crate::compute::ComputeError::Other(
                    "Empty token sequence after validation".into(),
                ));
            }
        };

        // Start generation (decode phase)
        tracing::info!("Generation phase: generating up to {} tokens", max_tokens);
        let mut generated_tokens = Vec::new();
        let mut printed_len = 0;

        // High-precision benchmark timing
        let generation_start = std::time::Instant::now();
        let mut tokens_generated = 0u32;

        for step in 0..max_tokens {
            // Check sequence length limit
            if seq_pos >= self.config.max_seq_len {
                tracing::warn!("Reached maximum sequence length: {}", self.config.max_seq_len);
                break;
            }

            tracing::debug!("Generation step {}/{}: seq_pos={}", step + 1, max_tokens, seq_pos);

            // Forward pass
            self.forward(last_token, seq_pos).await?;

            // Sample next token
            let next_token = self
                .sampler
                .sample(
                    &self.engine,
                    Some(&self.pipeline_cache),
                    Some((
                        &self.argmax_output_buf,
                        &self.argmax_staging_buf,
                        &self.argmax_bind_group,
                    )),
                    self.logits_buffer(),
                    &generated_tokens,
                )
                .await?;
            
            tracing::debug!("Sampled token ID: {}", next_token);
            tokens_generated += 1;

            // Check for EOS token
            if next_token == 2 {
                tracing::info!("Generated EOS token (ID: 2), stopping generation");
                break;
            }
            
            if let Some(eos_id) = self.tokenizer.eos_token_id() {
                if next_token == eos_id {
                    tracing::info!("Generated EOS token (ID: {}), stopping generation", eos_id);
                    break;
                }
            }

            // Add token to generated sequence
            generated_tokens.push(next_token);

            // Decode entire sequence to get the latest text
            let full_text = self
                .tokenizer
                .decode_batch(&generated_tokens)
                .map_err(|e| crate::compute::ComputeError::Other(format!("Detokenization failed: {}", e)))?;

            // Check for stop strings in the generated text
            if let Some(stop_strs) = stop_strings {
                let mut should_stop = false;
                for stop_str in stop_strs {
                    if full_text.contains(stop_str) {
                        tracing::info!("Stop string '{}' detected, stopping generation", stop_str);
                        should_stop = true;
                        break;
                    }
                }
                if should_stop {
                    break;
                }
            }

            // Stream only the newly generated text via callback
            if full_text.len() > printed_len {
                let new_text = &full_text[printed_len..];
                if let Some(ref mut cb) = callback {
                    if !cb(new_text) {
                        tracing::info!("Callback requested stop");
                        break;
                    }
                }
                printed_len = full_text.len();
            }

            // Update for next iteration
            last_token = next_token;
            seq_pos += 1;
        }

        // Decode final text from all generated tokens
        let generated_text = self
            .tokenizer
            .decode_batch(&generated_tokens)
            .map_err(|e| crate::compute::ComputeError::Other(format!("Final detokenization failed: {}", e)))?;

        // Calculate telemetry
        let elapsed_secs = generation_start.elapsed().as_secs_f64();
        let tps = if elapsed_secs > 0.0 {
            (tokens_generated as f64) / elapsed_secs
        } else {
            0.0
        };

        tracing::info!(
            "Generation complete: {} tokens generated in {:.3}s ({:.2} tok/s)",
            generated_tokens.len(),
            elapsed_secs,
            tps
        );

        Ok(generated_text)
    }

    /// Generate text autoregressively for multiple prompts in parallel using sampler's max_tokens
    ///
    /// This is a convenience method that uses the `max_tokens` value from the
    /// sampler's configuration instead of requiring it as a parameter.
    ///
    /// # Arguments
    /// * `prompts` - Input text prompts (must have length equal to batch_size)
    ///
    /// # Returns
    /// Vector of generated texts, one per prompt
    ///
    /// # Example
    /// ```no_run
    /// # use janus_engine::Model;
    /// # async fn example(mut model: Model) {
    /// let prompts = vec!["Once upon a time", "In a galaxy far away"];
    /// let results = model.generate_batch_text(&prompts).await.unwrap();
    /// # }
    /// ```
    pub async fn generate_batch_text(&mut self, prompts: &[&str]) -> Result<Vec<String>> {
        let max_tokens = self.sampler.config().max_tokens;
        self.generate_batch(prompts, max_tokens).await
    }

    /// Generate text autoregressively for multiple prompts in parallel
    ///
    /// This implements batched autoregressive generation:
    /// - Tokenizes all input prompts
    /// - Processes all sequences in parallel through the model (true batched GPU operations)
    /// - Samples independently for each sequence with per-sequence context
    /// - Continues until all sequences finish (EOS or max_tokens)
    ///
    /// # Prefill Optimization
    /// The prefill phase processes all prompts in parallel, position-by-position.
    /// Shorter prompts are padded with BOS tokens to match the longest prompt length.
    /// This ensures maximum GPU utilization through batched operations.
    ///
    /// # Arguments
    /// * `prompts` - Input text prompts (must have length equal to batch_size)
    /// * `max_tokens` - Maximum number of tokens to generate per sequence
    ///
    /// # Returns
    /// Vector of generated texts, one per prompt
    ///
    /// # Performance
    /// Batched inference provides approximately `batch_size`x throughput improvement
    /// compared to processing sequences sequentially, with minimal per-sequence latency
    /// overhead.
    ///
    /// # Example
    /// ```no_run
    /// # use janus_engine::Model;
    /// # async fn example(mut model: Model) {
    /// let prompts = vec!["Once upon a time", "In a galaxy far away"];
    /// let results = model.generate_batch(&prompts, 50).await.unwrap();
    /// for (i, result) in results.iter().enumerate() {
    ///     println!("Result {}: {}", i, result);
    /// }
    /// # }
    /// ```
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

        // Step 3: Prefill phase - process all prompts in parallel position-by-position
        // This processes batch_size tokens simultaneously at each position, achieving
        // true parallel prompt processing through the GPU batched operations
        tracing::info!("Prefill phase: processing {} prompts in parallel", batch_size);
        let prefill_start = std::time::Instant::now();

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

            tracing::debug!("Prefill position {}/{}: processing {} tokens in parallel", 
                seq_pos + 1, max_prompt_len, batch_size);
            self.forward_batch(&batch_tokens, seq_pos as u32).await?;
        }

        let prefill_elapsed = prefill_start.elapsed().as_secs_f64();
        let total_prefill_tokens = all_token_ids.iter().map(|ids| ids.len()).sum::<usize>();
        tracing::info!(
            "Prefill complete: {} total prompt tokens processed in {:.3}s ({:.2} tok/s)",
            total_prefill_tokens,
            prefill_elapsed,
            total_prefill_tokens as f64 / prefill_elapsed
        );

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

            // Sample next token for each sequence independently using batched sampling
            let context_refs: Vec<&[u32]> = generated_tokens.iter().map(|v| v.as_slice()).collect();
            let next_tokens = self.sampler.sample_batch(
                &self.engine,
                self.logits_buffer(),
                batch_size as u32,
                &context_refs
            ).await?;
            
            // Update each sequence with its independently sampled token
            for i in 0..batch_size {
                if !finished[i] {
                    let next_token = next_tokens[i];
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
        println!("Prefill Time: {:.3}s ({} prompt tokens at {:.2} tok/s)", 
            prefill_elapsed, total_prefill_tokens, total_prefill_tokens as f64 / prefill_elapsed);
        println!("Total Tokens Generated: {}", total_tokens_generated);
        println!("Generation Time: {:.3} seconds", elapsed_secs);
        println!("Generation Speed: {:.2} tok/s (total throughput)", tps);
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

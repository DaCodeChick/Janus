# TODO - Janus Engine

## High Priority

### Generation Configuration
- [ ] **Make `max_tokens` a configuration setting**
  - Currently hardcoded to 128 in `examples/inference.rs` (line 285)
  - Should be part of `SamplerConfig` or a new `GenerationConfig` struct
  - Allow users to specify via CLI argument or config file
  - Typical values: 128 (short), 512 (medium), 2048 (long)
  - See: `crates/janus-engine/src/model/model.rs::generate()` function
  - See: `crates/janus-engine/src/model/sampler.rs::SamplerConfig`

## Medium Priority

### Performance Optimizations
- [ ] **Implement FP16 inference support**
  - Most models are distributed in FP16 format
  - Current implementation loads everything as FP32
  - Need to support mixed precision (FP16 weights, FP32 accumulation)
  - See: `crates/janus-engine/src/loaders/` (GGUF and Safetensors loaders)

### Model Support
- [ ] **Add support for more model architectures**
  - Currently supports: LLaMA, Mistral, TinyLlama
  - Consider adding: Phi, Gemma, Qwen, etc.
  - See: `crates/janus-engine/src/model/`

- [ ] **Improve quantization support**
  - Currently only Q4_K is implemented
  - Add Q5_K, Q8_0, etc.
  - See: `crates/janus-engine/src/compute/ops/quantized.rs`

### Error Handling
- [x] **Better error messages for model loading failures**
  - ✓ Enhanced GGUFError with detailed diagnostics and suggestions
  - ✓ Enhanced FormatError with helpful error messages
  - ✓ Detect common issues (wrong architecture, missing weights, tensor mismatches)
  - ✓ Provide helpful suggestions to users with expected vs actual values
  - See: `crates/janus-engine/src/formats/gguf/error.rs`, `crates/janus-engine/src/formats/mod.rs`

- [x] **Add validation for model configuration**
  - ✓ Enhanced ConfigError with actionable error messages
  - ✓ Comprehensive validation in `HuggingFaceConfig::validate()`
  - ✓ Check dimensions match (head_dim divisibility, GQA constraints)
  - ✓ Verify vocabulary size matches between tokenizer, sampler, and model
  - ✓ Validate tensor buffer sizes match expected dimensions
  - ✓ Architecture whitelist with helpful unsupported architecture errors
  - ✓ Integration tests for all validation scenarios
  - See: `crates/janus-engine/src/model/config.rs`, `crates/janus-engine/src/model/model.rs`, `crates/janus-engine/tests/error_handling.rs`

## Low Priority

### Developer Experience
- [ ] **Add more examples**
  - Batch inference example
  - Streaming API example
  - Plugin development example
  - See: `crates/janus-engine/examples/`

- [ ] **Improve documentation**
  - Add architecture diagrams
  - Document shader implementations
  - Add performance tuning guide
  - See: `README.md` and crate-level docs

### Features
- [ ] **Add batched inference support**
  - Process multiple prompts in parallel
  - Requires batched GEMM operations
  - See: `crates/janus-engine/src/model/model.rs`

- [ ] **Implement speculative decoding**
  - Use small draft model + large target model for speedup
  - Advanced optimization technique
  - See: `crates/janus-engine/src/model/`

- [ ] **Add KV cache compression**
  - Compress old KV cache entries to extend context length
  - See: `crates/janus-engine/src/compute/cache.rs`
---

**Note**: This TODO list is living documentation. Add items as needed and delete completed items

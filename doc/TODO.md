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

- [ ] **Optimize RoPE computation**
  - Consider pre-computing sin/cos values for common positions
  - Cache in lookup table instead of computing every time
  - See: `crates/janus-engine/src/compute/shaders/rope.wgsl`

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
- [ ] **Better error messages for model loading failures**
  - Detect common issues (wrong architecture, missing weights, etc.)
  - Provide helpful suggestions to users
  - See: `crates/janus-engine/src/loaders/`

- [ ] **Add validation for model configuration**
  - Check that dimensions match between config and weights
  - Verify vocabulary size matches tokenizer
  - See: `crates/janus-engine/src/model/model.rs::new()`

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

### Testing
- [ ] **Add integration tests**
  - Test full generation pipeline with small test model
  - Verify numerical correctness against reference implementations
  - See: `crates/janus-engine/tests/` (needs to be created)

- [ ] **Add benchmarking suite**
  - Systematic performance benchmarks for different model sizes
  - Compare against llama.cpp and other inference engines
  - See: `crates/janus-engine/benches/` (needs to be created)

### Features
- [ ] **Implement beam search**
  - Alternative to greedy/sampling for better quality
  - See: `crates/janus-engine/src/model/sampler.rs`

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

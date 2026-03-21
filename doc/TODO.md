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

- [x] **Improve quantization support**
  - ✓ Implemented Q5_K quantization (5-bit, 256-element superblocks)
  - ✓ Implemented Q8_0 quantization (8-bit, 32-element blocks)
  - ✓ Added WGSL shaders for on-the-fly dequantization (gemm_q5_k.wgsl, gemm_q8_0.wgsl)
  - ✓ Updated ComputeEngine to handle Q5_K and Q8_0 tensor allocation
  - ✓ Full matrix-vector multiplication support for all quantized formats
  - Supported formats: Q4_K, Q5_K, Q8_0
  - See: `crates/janus-engine/src/compute/ops/quantized.rs`, `crates/janus-engine/src/compute/shaders/`

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

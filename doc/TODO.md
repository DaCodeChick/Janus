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
- [x] **Implement FP16 mixed-precision inference support**
  - ✓ Implemented packed FP16 format (2 f16s per u32) for 50% VRAM reduction
  - ✓ F32, F16, and BF16 tensors are converted to packed FP16 on CPU
  - ✓ Shaders unpack FP16 on-the-fly using `unpack2x16float()` WebGPU builtin
  - ✓ All computation happens in FP32 precision (mixed-precision)
  - ✓ Updated GEMM, matmul, and embed shaders
  - ✓ Comprehensive integration tests
  - See: `crates/janus-engine/src/compute/engine.rs` (packing functions)
  - See: `crates/janus-engine/src/compute/shaders/` (unpacking in shaders)

### Model Support
- [x] **Add support for more model architectures**
  - ✓ Added support for Microsoft Phi (Phi, Phi-3)
  - ✓ Added support for Google Gemma (Gemma, Gemma 2)
  - ✓ Added support for Alibaba Qwen (Qwen, Qwen 2)
  - ✓ All architectures use compatible LLaMA-style components (RoPE, RMSNorm, GQA)
  - ✓ Comprehensive validation and error messages
  - ✓ Integration tests for all new architectures
  - Supported: LLaMA, Mistral, TinyLlama, Phi, Phi-3, Gemma, Gemma 2, Qwen, Qwen2, GPT-NeoX
  - See: `crates/janus-engine/src/model/config.rs`, `doc/SUPPORTED_MODELS.md`

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

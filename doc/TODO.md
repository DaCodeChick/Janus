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

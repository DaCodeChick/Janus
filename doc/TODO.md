# TODO - Janus Engine

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
- [x] **Implement speculative decoding** ✅
  - Use small draft model + large target model for speedup
  - Advanced optimization technique with GPU-based KV cache copying
  - Proper sampling using target model's sampler configuration
  - Comprehensive unit tests for statistics and configuration
  - See: `crates/janus-engine/src/model/speculative.rs`

- [x] **Add KV cache compression** ✅
  - Compress old KV cache entries to extend context length (2-4x effective context)
  - Sliding window approach with configurable uncompressed window
  - Automatic compression triggering based on cache fill ratio
  - GPU-accelerated compression via WGSL shader
  - Comprehensive unit tests for compression configuration and statistics
  - See: `crates/janus-engine/src/compute/cache.rs`
---

**Note**: This TODO list is living documentation. Add items as needed and delete completed items

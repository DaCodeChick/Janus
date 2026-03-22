# Batched Inference Implementation Plan

## Overview

Implement batched inference support to process multiple prompts in parallel, significantly improving throughput for multi-user scenarios.

## Current Architecture (Single Token)

### Buffer Shapes
- `hidden_state`: `[hidden_dim]` - single token's hidden state
- `q_buf`, `k_buf`, `v_buf`: `[num_heads/kv_heads, head_dim]` - single token projections
- `logits_buf`: `[vocab_size]` - single token's output logits
- KV cache: `[num_layers, max_seq_len, num_kv_heads, head_dim]` - all positions, all layers

### Forward Pass
1. `embed_token()`: token_id → `hidden_state[hidden_dim]`
2. For each layer:
   - Input norm: `[hidden_dim]` → `[hidden_dim]`
   - QKV projection: `[hidden_dim]` → `[num_heads, head_dim]` each
   - RoPE: rotate Q and K
   - Attention: query vs all cached K/V → `[num_heads, head_dim]`
   - Output projection: `[num_heads*head_dim]` → `[hidden_dim]`
   - FFN: `[hidden_dim]` → `[ffn_dim]` → `[hidden_dim]`
3. Final norm + LM head: `[hidden_dim]` → `[vocab_size]`

## Target Architecture (Batched)

### Buffer Shapes
- `hidden_state`: `[batch_size, hidden_dim]` - batch of token hidden states
- `q_buf`, `k_buf`, `v_buf`: `[batch_size, num_heads/kv_heads, head_dim]` - batch projections
- `logits_buf`: `[batch_size, vocab_size]` - batch output logits
- KV cache: `[batch_size, num_layers, max_seq_len, num_kv_heads, head_dim]` - per-sequence caches

### Forward Pass
1. `embed_tokens()`: `token_ids[batch_size]` → `hidden_state[batch_size, hidden_dim]`
2. For each layer:
   - Input norm: `[batch_size, hidden_dim]` → `[batch_size, hidden_dim]`
   - QKV projection (batched GEMM): `[batch_size, hidden_dim]` × `[hidden_dim, num_heads*head_dim]` → `[batch_size, num_heads*head_dim]`
   - RoPE: rotate Q and K (parallel for batch)
   - Attention (parallel): each sequence attends to its own cache
   - Output projection (batched GEMM): `[batch_size, hidden_dim]` × `[hidden_dim, hidden_dim]`
   - FFN (batched GEMM): same pattern
3. Final norm + LM head: `[batch_size, hidden_dim]` → `[batch_size, vocab_size]`

## Implementation Strategy

### Phase 1: Core Infrastructure (Foundation)
1. **Add batch_size parameter to ModelConfig**
   - Add `batch_size: u32` to `ModelConfig`
   - Default to 1 for backward compatibility
   - Update validation logic

2. **Extend buffer allocations**
   - Multiply buffer sizes by `batch_size`
   - `hidden_state`: `batch_size * hidden_dim * 4 bytes`
   - `q_buf`: `batch_size * num_heads * head_dim * 4 bytes`
   - `logits_buf`: `batch_size * vocab_size * 4 bytes`
   - etc.

3. **Update KV cache structure**
   - Add `batch_size` dimension to cache buffers
   - Current: `[num_layers * max_seq_len * num_kv_heads * head_dim]`
   - New: `[batch_size * num_layers * max_seq_len * num_kv_heads * head_dim]`
   - Update `update()` to handle batch index

### Phase 2: Shader Updates (GPU Operations)
1. **Embedding shader** (`embed.wgsl`)
   - Current: single token_id → `[hidden_dim]`
   - New: `token_ids[batch_size]` → `[batch_size, hidden_dim]`
   - Dispatch: `(hidden_dim * batch_size + 255) / 256` workgroups

2. **GEMM operations** (`gemm.wgsl`)
   - Already supports matrix-matrix multiply
   - Update uniforms to include batch_size
   - Dispatch across batch dimension
   - Example: `[batch_size, M]` × `[M, N]` → `[batch_size, N]`

3. **Attention shader** (`attention.wgsl`)
   - Add batch_idx parameter
   - Each batch item reads from its own cache segment
   - Cache offset: `batch_idx * (num_layers * max_seq_len * num_kv_heads * head_dim)`
   - Layer offset within batch: `layer_idx * (max_seq_len * num_kv_heads * head_dim)`

4. **RoPE shader** (`rope.wgsl`)
   - Process all batch items in parallel
   - Each thread handles one element across all sequences
   - Or dispatch batch_size separate passes

5. **RMSNorm, activation shaders**
   - Add batch dimension to processing
   - Normalize per-sequence, not across batch

### Phase 3: API Changes (User-Facing)
1. **Batched forward pass**
   ```rust
   async fn forward_batch(&mut self, token_ids: &[u32], seq_positions: &[u32]) -> Result<()>
   ```
   - `token_ids.len() == batch_size`
   - `seq_positions.len() == batch_size` (each sequence can be at different position)
   - Update all internal calls to use batch indices

2. **Batched generation**
   ```rust
   pub async fn generate_batch(&mut self, prompts: &[&str], max_tokens: usize) -> Result<Vec<String>>
   ```
   - Tokenize all prompts
   - Pad to same length OR use dynamic batching (advanced)
   - Run forward passes in batch
   - Sample from `logits[batch_idx]` for each sequence
   - Continue until all sequences hit EOS or max_tokens

3. **Sequence state management**
   - Track per-sequence state:
     - Current position
     - EOS reached?
     - Generated tokens
   - Early stopping: remove finished sequences from batch (advanced)

### Phase 4: Optimizations (Performance)
1. **Dynamic batching**
   - Allow different sequence lengths in same batch
   - Mask attention for padding tokens
   - More complex but better utilization

2. **Continuous batching**
   - Add new sequences when old ones finish
   - Maximize GPU utilization
   - Requires careful cache management

3. **Batch GEMM optimization**
   - Use optimized batch GEMM kernels
   - Consider cuBLAS-style batched operations
   - Profile and tune workgroup sizes

## Implementation Order

### Milestone 1: Single Batch Support (Week 1)
- [ ] Add `batch_size` to ModelConfig (default=1)
- [ ] Update buffer allocations with batch dimension
- [ ] Update KV cache with batch dimension
- [ ] Add `forward_batch()` method (calls forward() in loop initially)
- [ ] Update tests to verify batch_size=1 still works

### Milestone 2: Batched Embeddings (Week 1-2)
- [ ] Update `embed.wgsl` shader for batch
- [ ] Update `embed_token()` to `embed_tokens()` for batch
- [ ] Test embedding batch on GPU

### Milestone 3: Batched Attention (Week 2-3)
- [ ] Update attention shader for batch indexing
- [ ] Update KV cache update for batch
- [ ] Ensure each sequence attends to its own history
- [ ] Test attention with batch_size > 1

### Milestone 4: Batched GEMM & FFN (Week 3-4)
- [ ] Update GEMM operations for batched input
- [ ] Update all matmul calls to handle batch
- [ ] Update RoPE for batch processing
- [ ] Update RMSNorm for batch processing
- [ ] Integration test: full forward pass with batch_size > 1

### Milestone 5: Batched Generation API (Week 4)
- [ ] Implement `generate_batch()` method
- [ ] Add sequence state tracking
- [ ] Handle per-sequence EOS detection
- [ ] Comprehensive integration tests
- [ ] Benchmark: throughput comparison (batch=1 vs batch=8)

### Milestone 6: Optimizations (Week 5+)
- [ ] Profile batch operations
- [ ] Optimize workgroup dispatch sizes
- [ ] Consider dynamic batching (optional)
- [ ] Documentation and examples

## Expected Performance Impact

### Throughput Improvement
- **Batch size 1**: Baseline (100%)
- **Batch size 4**: ~300-350% throughput (3-3.5x)
- **Batch size 8**: ~500-600% throughput (5-6x)
- **Batch size 16**: ~700-900% throughput (7-9x)

Diminishing returns due to:
- VRAM constraints (larger batches = more memory)
- GPU compute saturation
- Memory bandwidth limits

### Latency Impact
- **Per-token latency**: Similar or slightly higher (amortized overhead)
- **Time-to-first-token**: Similar for small batches
- **Overall generation time per sequence**: Depends on batch size and concurrency

### VRAM Usage
- **Batch size 1**: Baseline
- **Batch size 8**: ~1.5-2x VRAM (mostly KV cache growth)
- **Batch size 16**: ~2-3x VRAM

Most memory goes to:
1. KV cache (grows linearly with batch_size)
2. Intermediate activations (grows linearly with batch_size)
3. Model weights (constant, shared across batch)

## Challenges & Considerations

### 1. Variable Sequence Lengths
- **Problem**: Different prompts have different lengths
- **Solution**: Pad to longest or use dynamic masking
- **Trade-off**: Padding wastes compute, masking adds complexity

### 2. EOS Token Handling
- **Problem**: Sequences finish at different times
- **Solution**: Continue until all finish OR remove from batch dynamically
- **Trade-off**: Simple (wait for all) vs complex (dynamic removal)

### 3. Memory Management
- **Problem**: Larger batches need more VRAM
- **Solution**: Allow configurable batch_size, OOM detection
- **Trade-off**: Throughput vs VRAM constraints

### 4. Shader Complexity
- **Problem**: Batched shaders are more complex
- **Solution**: Careful indexing, thorough testing
- **Trade-off**: Code complexity vs performance

## Testing Strategy

### Unit Tests
- [ ] Test batched embedding lookup
- [ ] Test batched GEMM operations
- [ ] Test batched attention with different batch sizes
- [ ] Test KV cache indexing for batches

### Integration Tests
- [ ] Forward pass with batch_size=1 (regression)
- [ ] Forward pass with batch_size=4
- [ ] Forward pass with batch_size=8
- [ ] Generation with multiple prompts
- [ ] Edge cases: empty batch, single prompt, max batch

### Performance Tests
- [ ] Benchmark throughput: tokens/sec for various batch sizes
- [ ] Measure latency: time per token
- [ ] Profile VRAM usage
- [ ] Compare to single-sequence baseline

## Success Metrics

1. **Correctness**: Batched inference produces same results as sequential
2. **Throughput**: 4-6x improvement with batch_size=8
3. **VRAM**: Memory usage stays within reasonable bounds (< 3x baseline)
4. **API**: Clean, intuitive batched generation API
5. **Tests**: Comprehensive test coverage (>80%)

## References

- Current forward pass: `src/model/model.rs:750`
- GEMM operations: `src/compute/ops/matmul.rs`
- Attention: `src/compute/ops/attention.rs`
- KV Cache: `src/compute/cache.rs`
- Embedding shader: `src/compute/shaders/embed.wgsl`

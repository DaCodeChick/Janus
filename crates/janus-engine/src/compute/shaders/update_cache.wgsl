// KV Cache Update Shader - Batched
//
// This shader copies newly calculated Key or Value matrices for a batch of tokens
// into the correct positions within the batched KV cache buffer.
//
// The KV cache is a ring buffer that stores the Key/Value projections for all
// previous tokens in each sequence, allowing us to avoid recomputing them during
// autoregressive generation.
//
// For GQA (Grouped Query Attention), the cache stores num_kv_heads (not num_query_heads).
// For example, TinyLlama has 32 query heads but only 4 KV heads.
//
// Batched Layout: [batch_size][num_layers][max_seq_len][num_kv_heads][head_dim]
// Each sequence in the batch has its own independent cache segment.

struct UpdateCacheUniforms {
    batch_size: u32,      // Number of sequences in batch
    cache_position: u32,  // Position in the cache to write to
    token_dim: u32,       // Dimension of each token's K or V vector (head_dim)
    num_heads: u32,       // Number of KV attention heads (NOT query heads!)
    layer_idx: u32,       // Transformer layer index
    max_seq_len: u32,     // Maximum sequence length
    num_layers: u32,      // Total number of transformer layers
    _pad: u32,            // Padding for alignment (8 u32s = 32 bytes)
}

@group(0) @binding(0) var<storage, read> new_kv: array<f32>;          // New K or V for batch (batch_size * num_kv_heads * head_dim)
@group(0) @binding(1) var<storage, read_write> cache: array<f32>;     // KV cache buffer (batch_size * num_layers * max_seq_len * num_kv_heads * head_dim)
@group(0) @binding(2) var<uniform> uniforms: UpdateCacheUniforms;

// Each thread copies one element from new_kv to the cache
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let total_idx = global_id.x;
    
    // Total elements to copy (batch_size * num_kv_heads * head_dim)
    let elements_per_batch = uniforms.num_heads * uniforms.token_dim;
    let total_elements = uniforms.batch_size * elements_per_batch;
    
    if (total_idx >= total_elements) {
        return;
    }
    
    // Decode which batch item and which element within that batch
    let batch_idx = total_idx / elements_per_batch;
    let element_idx = total_idx % elements_per_batch;
    
    // Calculate batch offset in cache
    // Each batch has its own [num_layers][max_seq_len][num_kv_heads][head_dim] segment
    let batch_cache_size = uniforms.num_layers * uniforms.max_seq_len * uniforms.num_heads * uniforms.token_dim;
    let batch_offset = batch_idx * batch_cache_size;
    
    // Calculate layer offset within this batch's cache
    let layer_size = uniforms.max_seq_len * uniforms.num_heads * uniforms.token_dim;
    let layer_offset = uniforms.layer_idx * layer_size;
    
    // Calculate the cache offset for this position within the layer
    // Cache layout: [batch_size][num_layers][max_seq_len][num_kv_heads][head_dim]
    let position_offset = uniforms.cache_position * (uniforms.num_heads * uniforms.token_dim);
    let cache_idx = batch_offset + layer_offset + position_offset + element_idx;
    
    // Copy from new_kv to cache
    // new_kv layout: [batch_idx][element_idx]
    cache[cache_idx] = new_kv[total_idx];
}

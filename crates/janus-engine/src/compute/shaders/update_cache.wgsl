// KV Cache Update Shader
//
// This shader copies a newly calculated Key or Value matrix for the current token
// into the correct position within the massive KV cache buffer.
//
// The KV cache is a ring buffer that stores the Key/Value projections for all
// previous tokens in the sequence, allowing us to avoid recomputing them during
// autoregressive generation.
//
// For GQA (Grouped Query Attention), the cache stores num_kv_heads (not num_query_heads).
// For example, TinyLlama has 32 query heads but only 4 KV heads.
//
// Layout: [num_layers][max_seq_len][num_kv_heads][head_dim]

struct UpdateCacheUniforms {
    cache_position: u32,  // Position in the cache to write to
    token_dim: u32,       // Dimension of each token's K or V vector (head_dim)
    num_heads: u32,       // Number of KV attention heads (NOT query heads!)
    layer_idx: u32,       // Transformer layer index
    max_seq_len: u32,     // Maximum sequence length
    _pad: vec3<u32>,      // Padding for alignment
}

@group(0) @binding(0) var<storage, read> new_kv: array<f32>;          // New K or V for current token (num_kv_heads * head_dim)
@group(0) @binding(1) var<storage, read_write> cache: array<f32>;     // KV cache buffer (num_layers * max_seq_len * num_kv_heads * head_dim)
@group(0) @binding(2) var<uniform> uniforms: UpdateCacheUniforms;

// Each thread copies one element from new_kv to the cache
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    // Total elements to copy (num_kv_heads * head_dim for one token)
    let total_elements = uniforms.num_heads * uniforms.token_dim;
    
    if (idx >= total_elements) {
        return;
    }
    
    // Calculate layer offset to segment cache by layer
    let layer_size = uniforms.max_seq_len * uniforms.num_heads * uniforms.token_dim;
    let layer_offset = uniforms.layer_idx * layer_size;
    
    // Calculate the cache offset for this position within the layer
    // Cache layout: [num_layers][max_seq_len][num_kv_heads][head_dim]
    let position_offset = uniforms.cache_position * (uniforms.num_heads * uniforms.token_dim);
    let cache_idx = layer_offset + position_offset + idx;
    
    // Copy from new_kv to cache
    cache[cache_idx] = new_kv[idx];
}

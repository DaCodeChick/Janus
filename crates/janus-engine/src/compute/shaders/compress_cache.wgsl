//! KV Cache Compression Shader
//!
//! This shader compresses KV cache entries by averaging pairs of consecutive tokens.
//! Used to extend effective context length beyond physical cache size.
//!
//! Compression strategy:
//! - Input: Full precision KV cache entries
//! - Output: Compressed entries (N:1 compression via averaging)
//! - Preserves semantic information while reducing memory usage

struct CompressionUniforms {
    batch_size: u32,
    num_layers: u32,
    max_seq_len: u32,
    num_kv_heads: u32,
    head_dim: u32,
    compression_start: u32,  // Starting position to compress
    compression_end: u32,    // Ending position to compress
    compression_ratio: u32,  // Compression ratio (2 = 2:1, 4 = 4:1)
}

@group(0) @binding(0) var<storage, read> input_cache: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_cache: array<f32>;
@group(0) @binding(2) var<uniform> uniforms: CompressionUniforms;

// Calculate index into cache buffer
// Layout: [batch][layer][position][head][dim]
fn cache_index(batch: u32, layer: u32, pos: u32, head: u32, dim: u32) -> u32 {
    let batch_offset = batch * uniforms.num_layers * uniforms.max_seq_len * uniforms.num_kv_heads * uniforms.head_dim;
    let layer_offset = layer * uniforms.max_seq_len * uniforms.num_kv_heads * uniforms.head_dim;
    let pos_offset = pos * uniforms.num_kv_heads * uniforms.head_dim;
    let head_offset = head * uniforms.head_dim;
    return batch_offset + layer_offset + pos_offset + head_offset + dim;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_id = global_id.x;
    
    // Calculate total elements to process (compressed positions * heads * dims)
    let compressed_positions = (uniforms.compression_end - uniforms.compression_start) / uniforms.compression_ratio;
    let total_elements = uniforms.batch_size * uniforms.num_layers * compressed_positions * uniforms.num_kv_heads * uniforms.head_dim;
    
    if (thread_id >= total_elements) {
        return;
    }
    
    // Decompose thread ID into (batch, layer, compressed_pos, head, dim)
    var remaining = thread_id;
    let dim = remaining % uniforms.head_dim;
    remaining = remaining / uniforms.head_dim;
    let head = remaining % uniforms.num_kv_heads;
    remaining = remaining / uniforms.num_kv_heads;
    let compressed_pos = remaining % compressed_positions;
    remaining = remaining / compressed_positions;
    let layer = remaining % uniforms.num_layers;
    let batch = remaining / uniforms.num_layers;
    
    // Calculate source position range (multiple tokens to average)
    let source_start_pos = uniforms.compression_start + (compressed_pos * uniforms.compression_ratio);
    
    // Average multiple tokens into one compressed token
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < uniforms.compression_ratio; i = i + 1u) {
        let source_pos = source_start_pos + i;
        if (source_pos < uniforms.compression_end) {
            let source_idx = cache_index(batch, layer, source_pos, head, dim);
            sum = sum + input_cache[source_idx];
        }
    }
    let compressed_value = sum / f32(uniforms.compression_ratio);
    
    // Write compressed value to output at the compressed position
    let output_pos = compressed_pos;
    let output_idx = cache_index(batch, layer, output_pos, head, dim);
    output_cache[output_idx] = compressed_value;
}

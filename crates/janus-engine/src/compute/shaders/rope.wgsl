// Rotary Positional Embeddings (RoPE) - LLaMA half-split variant (OPTIMIZED, BATCHED)
// 
// RoPE applies rotation to pairs of dimensions in the embedding space
// to encode positional information. This allows the model to understand
// token ordering without explicit position embeddings.
//
// LLaMA uses a "half-split" approach where the head dimension is split
// in half, and each dimension in the first half is paired with the
// corresponding dimension in the second half.
//
// For dimension i in [0, head_dim/2):
//   angle = position / (10000 ^ (2i / head_dim))
//   x'[i] = x[i] * cos(angle) - x[i + head_dim/2] * sin(angle)
//   x'[i + head_dim/2] = x[i] * sin(angle) + x[i + head_dim/2] * cos(angle)
//
// OPTIMIZATION: This version uses pre-computed sin/cos values from a lookup table
// instead of computing them on-the-fly, eliminating expensive trigonometric operations.
//
// BATCHED: Processes [batch_size, num_heads, head_dim] tensors in parallel

struct RopeUniforms {
    batch_size: u32,     // Number of sequences in batch
    num_heads: u32,      // Number of attention heads
    head_dim: u32,       // Dimension of each attention head
    position: u32,       // Current position in sequence
}

@group(0) @binding(0) var<storage, read> input: array<f32>;           // Input tensor (batch_size * num_heads * head_dim)
@group(0) @binding(1) var<storage, read_write> output: array<f32>;    // Output tensor (batch_size * num_heads * head_dim)
@group(0) @binding(2) var<uniform> uniforms: RopeUniforms;
@group(0) @binding(3) var<storage, read> rope_cache: array<f32>;      // Pre-computed sin/cos values

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    // Each thread processes one dimension pair (half-split) for one batch item and one head
    let pair_idx = idx;
    let half_dim = uniforms.head_dim / 2u;
    let total_pairs = uniforms.batch_size * uniforms.num_heads * half_dim;
    
    if (pair_idx >= total_pairs) {
        return;
    }
    
    // Calculate batch index, head index, and dimension pair
    let batch_idx = pair_idx / (uniforms.num_heads * half_dim);
    let remainder = pair_idx % (uniforms.num_heads * half_dim);
    let head_idx = remainder / half_dim;
    let dim_pair = remainder % half_dim;
    
    // Calculate indices for the two elements to rotate
    // Layout: [batch_size, num_heads, head_dim]
    // First half: [0, head_dim/2), Second half: [head_dim/2, head_dim)
    let base_idx = (batch_idx * uniforms.num_heads + head_idx) * uniforms.head_dim;
    let idx_1 = base_idx + dim_pair;
    let idx_2 = idx_1 + half_dim;
    
    // Get input values
    let x0 = input[idx_1];
    let x1 = input[idx_2];
    
    // Lookup pre-computed sin/cos values from cache
    // Cache layout: [position * head_dim/2 + dim_pair] stores (cos, sin) interleaved
    // NOTE: Same position used for all batch items (same sequence position for entire batch)
    let cache_idx = uniforms.position * half_dim + dim_pair;
    let cos_angle = rope_cache[cache_idx * 2u];
    let sin_angle = rope_cache[cache_idx * 2u + 1u];
    
    // Apply rotation (half-split variant)
    let y0 = x0 * cos_angle - x1 * sin_angle;
    let y1 = x0 * sin_angle + x1 * cos_angle;
    
    // Write output
    output[idx_1] = y0;
    output[idx_2] = y1;
}

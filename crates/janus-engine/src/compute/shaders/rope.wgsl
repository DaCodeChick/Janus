// Rotary Positional Embeddings (RoPE) - LLaMA half-split variant
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

struct RopeUniforms {
    seq_len: u32,        // Sequence length (total number of tokens)
    head_dim: u32,       // Dimension of each attention head
    position: u32,       // Current position in sequence
    theta_base: f32,     // Base for frequency calculation (typically 10000.0)
}

@group(0) @binding(0) var<storage, read> input: array<f32>;           // Input tensor (seq_len * head_dim)
@group(0) @binding(1) var<storage, read_write> output: array<f32>;    // Output tensor (seq_len * head_dim)
@group(0) @binding(2) var<uniform> uniforms: RopeUniforms;

// Constants
const PI: f32 = 3.14159265359;

// Calculate the frequency for a given dimension pair
fn get_theta(dim_idx: u32) -> f32 {
    let exponent = f32(dim_idx) / f32(uniforms.head_dim);
    return pow(uniforms.theta_base, exponent);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    // Each thread processes one dimension pair (half-split)
    let pair_idx = idx;
    let half_dim = uniforms.head_dim / 2u;
    let total_pairs = uniforms.seq_len * half_dim;
    
    if (pair_idx >= total_pairs) {
        return;
    }
    
    // Calculate which token and which dimension pair within that token
    let token_idx = pair_idx / half_dim;
    let dim_pair = pair_idx % half_dim;
    
    // Calculate indices for the two elements to rotate
    // First half: [0, head_dim/2), Second half: [head_dim/2, head_dim)
    let idx_1 = token_idx * uniforms.head_dim + dim_pair;
    let idx_2 = idx_1 + half_dim;
    
    // Get input values
    let x0 = input[idx_1];
    let x1 = input[idx_2];
    
    // Calculate position
    // CRITICAL: All heads for a single token MUST use the same position
    // token_idx evaluates to head_idx when processing one token at a time
    let position = uniforms.position;
    
    // Calculate rotation angle
    // theta = theta_base ^ (2 * dim_pair / head_dim)
    let theta = get_theta(dim_pair * 2u);
    let angle = f32(position) / theta;
    
    // Calculate sin and cos
    let cos_angle = cos(angle);
    let sin_angle = sin(angle);
    
    // Apply rotation (half-split variant)
    let y0 = x0 * cos_angle - x1 * sin_angle;
    let y1 = x0 * sin_angle + x1 * cos_angle;
    
    // Write output
    output[idx_1] = y0;
    output[idx_2] = y1;
}

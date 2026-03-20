// Rotary Positional Embeddings (RoPE)
// 
// RoPE applies rotation to pairs of dimensions in the embedding space
// to encode positional information. This allows the model to understand
// token ordering without explicit position embeddings.
//
// For each pair of dimensions (2i, 2i+1), we rotate by an angle that
// depends on the position and the dimension:
//   angle = position / (10000 ^ (2i / d_model))
//
// The rotation is applied as:
//   x'[2i]   = x[2i] * cos(angle) - x[2i+1] * sin(angle)
//   x'[2i+1] = x[2i] * sin(angle) + x[2i+1] * cos(angle)

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
    
    // Each thread processes one pair of dimensions
    let pair_idx = idx;
    let total_pairs = (uniforms.seq_len * uniforms.head_dim) / 2u;
    
    if (pair_idx >= total_pairs) {
        return;
    }
    
    // Calculate which token and which dimension pair within that token
    let token_idx = pair_idx / (uniforms.head_dim / 2u);
    let dim_pair = pair_idx % (uniforms.head_dim / 2u);
    
    // Calculate the base index for this pair
    let base_idx = token_idx * uniforms.head_dim + dim_pair * 2u;
    
    // Get input values
    let x0 = input[base_idx];
    let x1 = input[base_idx + 1u];
    
    // Calculate position (this can be the token's position in the sequence)
    // For now, we use the token_idx relative to the start, but in practice
    // you might want to pass absolute positions
    let position = token_idx + uniforms.position;
    
    // Calculate rotation angle
    let theta = get_theta(dim_pair * 2u);
    let angle = f32(position) / theta;
    
    // Calculate sin and cos
    let cos_angle = cos(angle);
    let sin_angle = sin(angle);
    
    // Apply rotation
    let y0 = x0 * cos_angle - x1 * sin_angle;
    let y1 = x0 * sin_angle + x1 * cos_angle;
    
    // Write output
    output[base_idx] = y0;
    output[base_idx + 1u] = y1;
}

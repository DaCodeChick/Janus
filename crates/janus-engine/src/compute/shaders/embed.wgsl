// Token embedding lookup shader (batched)
// Copies multiple rows from the embedding table to the output buffer
// Supports batched inference: processes batch_size tokens in parallel
// Embedding table is stored in packed f16 format (2 f16s per u32)

struct EmbedParams {
    batch_size: u32,
    hidden_dim: u32,
}

@group(0) @binding(0) var<uniform> params: EmbedParams;
@group(0) @binding(1) var<storage, read> token_ids: array<u32>; // [batch_size] token IDs
@group(0) @binding(2) var<storage, read> embedding_table: array<u32>; // [vocab_size, hidden_dim / 2] packed f16
@group(0) @binding(3) var<storage, read_write> output: array<f32>; // [batch_size, hidden_dim] f32

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let total_idx = global_id.x;
    let batch_size = params.batch_size;
    let hidden_dim = params.hidden_dim;
    
    // Calculate which batch item and which hidden dimension element
    let batch_idx = total_idx / hidden_dim;
    let hidden_idx = total_idx % hidden_dim;
    
    // Bounds check
    if (batch_idx >= batch_size || hidden_idx >= hidden_dim) {
        return;
    }
    
    // Get token ID for this batch item
    let token_id = token_ids[batch_idx];
    
    // Calculate global index in the embedding table
    let embedding_idx = token_id * hidden_dim + hidden_idx;
    
    // Read packed u32 (contains 2 f16 values)
    let packed = embedding_table[embedding_idx / 2u];
    
    // Unpack using WebGPU builtin: returns vec2<f32> with converted values
    let unpacked = unpack2x16float(packed);
    
    // Select the correct f16 value (low 16 bits = even index, high 16 bits = odd index)
    let is_odd = (embedding_idx % 2u) != 0u;
    let value = select(unpacked.x, unpacked.y, is_odd);
    
    // Write to output at [batch_idx, hidden_idx]
    output[batch_idx * hidden_dim + hidden_idx] = value;
}

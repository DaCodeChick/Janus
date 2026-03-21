// Token embedding lookup shader
// Copies a single row from the embedding table to the output buffer
// Embedding table is stored in packed f16 format (2 f16s per u32)

@group(0) @binding(0) var<uniform> params: vec2<u32>; // [token_id, hidden_dim]
@group(0) @binding(1) var<storage, read> embedding_table: array<u32>; // [vocab_size, hidden_dim / 2] packed f16
@group(0) @binding(2) var<storage, read_write> output: array<f32>; // [hidden_dim] f32

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let token_id = params.x;
    let hidden_dim = params.y;
    let idx = global_id.x;
    
    if (idx >= hidden_dim) {
        return;
    }
    
    // Calculate global index in the embedding table
    let embedding_idx = token_id * hidden_dim + idx;
    
    // Read packed u32 (contains 2 f16 values)
    let packed = embedding_table[embedding_idx / 2u];
    
    // Unpack using WebGPU builtin: returns vec2<f32> with converted values
    let unpacked = unpack2x16float(packed);
    
    // Select the correct f16 value (low 16 bits = even index, high 16 bits = odd index)
    let is_odd = (embedding_idx % 2u) != 0u;
    let value = select(unpacked.x, unpacked.y, is_odd);
    
    // Copy to output
    output[idx] = value;
}

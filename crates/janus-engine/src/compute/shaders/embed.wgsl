// Token embedding lookup shader
// Copies a single row from the embedding table to the output buffer

@group(0) @binding(0) var<uniform> params: vec2<u32>; // [token_id, hidden_dim]
@group(0) @binding(1) var<storage, read> embedding_table: array<f32>; // [vocab_size, hidden_dim]
@group(0) @binding(2) var<storage, read_write> output: array<f32>; // [hidden_dim]

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let token_id = params.x;
    let hidden_dim = params.y;
    let idx = global_id.x;
    
    if (idx >= hidden_dim) {
        return;
    }
    
    // Copy embedding_table[token_id][idx] to output[idx]
    let embedding_idx = token_id * hidden_dim + idx;
    output[idx] = embedding_table[embedding_idx];
}

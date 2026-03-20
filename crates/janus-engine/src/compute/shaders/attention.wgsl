// Scaled Dot-Product Attention with Grouped Query Attention (GQA)
//
// Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
//
// This shader implements multi-head attention with GQA support:
// - Query has num_heads (e.g., 32 for TinyLlama)
// - Key/Value has num_kv_heads (e.g., 4 for TinyLlama)
// - Each KV head is shared across num_heads/num_kv_heads query heads
//
// Steps:
// 1. Compute Q * K^T for one query position across all key positions
// 2. Scale by 1/sqrt(head_dim) 
// 3. Apply softmax (done in separate shader pass)
// 4. Multiply attention weights by V to get output
//
// Layout:
// - Query: [num_heads][head_dim] - single token's query
// - Keys: [num_layers][seq_len][num_kv_heads][head_dim] - all cached keys
// - Values: [num_layers][seq_len][num_kv_heads][head_dim] - all cached values
// - Output: [num_heads][head_dim] - attention output for this token

struct Params {
    seq_len: u32,       // Current sequence length (how many tokens in cache)
    num_heads: u32,     // Number of query attention heads
    num_kv_heads: u32,  // Number of key-value attention heads (for GQA)
    head_dim: u32,      // Dimension of each head
    scale: f32,         // 1/sqrt(head_dim) for scaled dot-product
    layer_idx: u32,     // Transformer layer index
    max_seq_len: u32,   // Maximum sequence length
    _pad: u32,          // Padding for alignment
}

// Step 1: Compute QK^T scores
// Each workgroup handles one head, each thread computes score for one key position
// For GQA: query head maps to kv_head = query_head / (num_heads / num_kv_heads)
@group(0) @binding(0) var<storage, read> query: array<f32>;      // [num_heads * head_dim]
@group(0) @binding(1) var<storage, read> keys: array<f32>;       // [num_layers * seq_len * num_kv_heads * head_dim]
@group(0) @binding(2) var<storage, read_write> scores: array<f32>; // [num_heads * seq_len]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn compute_qk_scores(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let head_idx = workgroup_id.x;
    let key_pos = global_id.x - (workgroup_id.x * 256u);
    
    // Early exit if beyond sequence length
    if (key_pos >= params.seq_len || head_idx >= params.num_heads) {
        return;
    }
    
    let head_dim = params.head_dim;
    
    // Calculate layer offset to segment cache by layer
    let layer_size = params.max_seq_len * params.num_kv_heads * head_dim;
    let layer_offset = params.layer_idx * layer_size;
    
    // For GQA: map query head to corresponding KV head
    // kv_head_idx = query_head_idx / (num_heads / num_kv_heads)
    let kv_head_idx = (head_idx * params.num_kv_heads) / params.num_heads;
    
    // Compute dot product between query[head_idx] and key[key_pos][kv_head_idx]
    var dot_product: f32 = 0.0;
    
    for (var d = 0u; d < head_dim; d++) {
        let q_idx = head_idx * head_dim + d;
        let k_idx = layer_offset + key_pos * params.num_kv_heads * head_dim + kv_head_idx * head_dim + d;
        dot_product += query[q_idx] * keys[k_idx];
    }
    
    // Scale by 1/sqrt(head_dim) and store
    let score_idx = head_idx * params.seq_len + key_pos;
    scores[score_idx] = dot_product * params.scale;
}

// Step 2: Multiply attention probabilities by values
// Input: attention_probs from softmax [num_heads * seq_len]
// This computes the weighted sum of values
// For GQA: each query head maps to its corresponding KV head
@group(0) @binding(0) var<storage, read> attention_probs: array<f32>; // [num_heads * seq_len]
@group(0) @binding(1) var<storage, read> values: array<f32>;          // [num_layers * seq_len * num_kv_heads * head_dim]
@group(0) @binding(2) var<storage, read_write> output: array<f32>;    // [num_heads * head_dim]
@group(0) @binding(3) var<uniform> params_v: Params;

@compute @workgroup_size(256)
fn apply_attention(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let head_idx = global_id.x / params_v.head_dim;
    let dim_idx = global_id.x % params_v.head_dim;
    
    // Early exit if beyond bounds
    if (head_idx >= params_v.num_heads || dim_idx >= params_v.head_dim) {
        return;
    }
    
    let seq_len = params_v.seq_len;
    let head_dim = params_v.head_dim;
    let num_kv_heads = params_v.num_kv_heads;
    
    // Calculate layer offset to segment cache by layer
    let layer_size = params_v.max_seq_len * num_kv_heads * head_dim;
    let layer_offset = params_v.layer_idx * layer_size;
    
    // For GQA: map query head to corresponding KV head
    let kv_head_idx = (head_idx * num_kv_heads) / params_v.num_heads;
    
    // Compute weighted sum: sum over all positions (attention_prob[pos] * value[pos][dim])
    var weighted_sum: f32 = 0.0;
    
    for (var pos = 0u; pos < seq_len; pos++) {
        let prob_idx = head_idx * seq_len + pos;
        let value_idx = layer_offset + pos * num_kv_heads * head_dim + kv_head_idx * head_dim + dim_idx;
        weighted_sum += attention_probs[prob_idx] * values[value_idx];
    }
    
    // Store output
    let out_idx = head_idx * head_dim + dim_idx;
    output[out_idx] = weighted_sum;
}

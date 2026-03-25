// Scaled Dot-Product Attention with Grouped Query Attention (GQA) - Batched
//
// Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
//
// This shader implements batched multi-head attention with GQA support:
// - Query has num_heads (e.g., 32 for TinyLlama)
// - Key/Value has num_kv_heads (e.g., 4 for TinyLlama)
// - Each KV head is shared across num_heads/num_kv_heads query heads
// - Processes batch_size sequences in parallel
//
// Steps:
// 1. Compute Q * K^T for one query position across all key positions
// 2. Scale by 1/sqrt(head_dim) 
// 3. Apply softmax (done in separate shader pass)
// 4. Multiply attention weights by V to get output
//
// Batched Layout:
// - Query: [batch_size][num_heads][head_dim] - batch of queries
// - Keys: [batch_size][num_layers][max_seq_len][num_kv_heads][head_dim] - per-sequence cached keys
// - Values: [batch_size][num_layers][max_seq_len][num_kv_heads][head_dim] - per-sequence cached values
// - Output: [batch_size][num_heads][head_dim] - attention output for batch
//
// Each sequence in the batch:
// - Has its own KV cache segment
// - Attends only to its own history (no cross-sequence attention)
// - Can be at a different position in its sequence

struct Params {
    batch_size: u32,    // Number of sequences in batch
    seq_len: u32,       // Current sequence length (how many tokens in cache)
    num_heads: u32,     // Number of query attention heads
    num_kv_heads: u32,  // Number of key-value attention heads (for GQA)
    head_dim: u32,      // Dimension of each head
    scale: f32,         // 1/sqrt(head_dim) for scaled dot-product
    layer_idx: u32,     // Transformer layer index
    max_seq_len: u32,   // Maximum sequence length
    num_layers: u32,    // Total number of transformer layers (for cache indexing)
    _pad0: u32,         // Padding to 64 bytes (16*4) for WGSL uniform alignment
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
    _pad4: u32,
    _pad5: u32,
    _pad6: u32,
}

// Step 1: Compute QK^T scores (batched)
// Each workgroup handles one (batch_item, head) pair
// Each thread computes score for one key position
// For GQA: query head maps to kv_head = query_head / (num_heads / num_kv_heads)
//
// Input layout:
// - query: [batch_size * num_heads * head_dim]
// - keys: [batch_size * num_layers * max_seq_len * num_kv_heads * head_dim]
// Output layout:
// - scores: [batch_size * num_heads * max_seq_len]
@group(0) @binding(0) var<storage, read> query: array<f32>;      // [batch_size][num_heads][head_dim]
@group(0) @binding(1) var<storage, read> keys: array<f32>;       // [batch_size][num_layers][max_seq_len][num_kv_heads][head_dim]
@group(0) @binding(2) var<storage, read_write> scores: array<f32>; // [batch_size][num_heads][max_seq_len]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn compute_qk_scores(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    // Y workgroup dimension encodes: batch_idx * num_heads + head_idx
    let combined_idx = workgroup_id.y;
    let batch_idx = combined_idx / params.num_heads;
    let head_idx = combined_idx % params.num_heads;

    // X workgroup dimension tiles sequence positions by 256 threads per workgroup
    let key_pos = workgroup_id.x * 256u + local_id.x;
    
    // Early exit if beyond bounds
    if (batch_idx >= params.batch_size || head_idx >= params.num_heads || key_pos >= params.seq_len) {
        return;
    }
    
    let head_dim = params.head_dim;
    let num_kv_heads = params.num_kv_heads;
    let max_seq_len = params.max_seq_len;
    
    // Calculate batch offset in KV cache
    // Each batch item has its own [num_layers][max_seq_len][num_kv_heads][head_dim] segment
    let batch_cache_size = params.num_layers * max_seq_len * num_kv_heads * head_dim;
    let batch_offset = batch_idx * batch_cache_size;
    
    // Calculate layer offset within this batch's cache
    let layer_size = max_seq_len * num_kv_heads * head_dim;
    let layer_offset = params.layer_idx * layer_size;
    
    // For GQA: map query head to corresponding KV head
    // kv_head_idx = query_head_idx / (num_heads / num_kv_heads)
    let kv_head_idx = (head_idx * num_kv_heads) / params.num_heads;
    
    // Compute dot product between query[batch_idx][head_idx] and key[batch_idx][layer][key_pos][kv_head_idx]
    var dot_product: f32 = 0.0;
    
    for (var d = 0u; d < head_dim; d++) {
        // Query index: [batch_idx][head_idx][d]
        let q_idx = batch_idx * params.num_heads * head_dim + head_idx * head_dim + d;
        
        // Key index: [batch_idx][layer_idx][key_pos][kv_head_idx][d]
        let k_idx = batch_offset + layer_offset + key_pos * num_kv_heads * head_dim + kv_head_idx * head_dim + d;
        
        dot_product += query[q_idx] * keys[k_idx];
    }
    
    // Scale by 1/sqrt(head_dim) and store
    // Score index: [batch_idx][head_idx][key_pos]
    // CRITICAL: Use max_seq_len for stride, not seq_len, since buffer is sized [batch, heads, max_seq_len]
    let score_idx = batch_idx * params.num_heads * max_seq_len + head_idx * max_seq_len + key_pos;
    scores[score_idx] = dot_product * params.scale;
}

// Step 2: Multiply attention probabilities by values (batched)
// Input: attention_probs from softmax [batch_size][num_heads][max_seq_len]
// This computes the weighted sum of values for each batch item
// For GQA: each query head maps to its corresponding KV head
//
// Input layout:
// - attention_probs: [batch_size * num_heads * max_seq_len]
// - values: [batch_size * num_layers * max_seq_len * num_kv_heads * head_dim]
// Output layout:
// - output: [batch_size * num_heads * head_dim]
@group(0) @binding(0) var<storage, read> attention_probs: array<f32>; // [batch_size][num_heads][max_seq_len]
@group(0) @binding(1) var<storage, read> values: array<f32>;          // [batch_size][num_layers][max_seq_len][num_kv_heads][head_dim]
@group(0) @binding(2) var<storage, read_write> output: array<f32>;    // [batch_size][num_heads][head_dim]
@group(0) @binding(3) var<uniform> params_v: Params;

@compute @workgroup_size(256)
fn apply_attention(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let total_idx = global_id.x;
    let batch_size = params_v.batch_size;
    let num_heads = params_v.num_heads;
    let head_dim = params_v.head_dim;
    
    // Decode: [batch_idx][head_idx][dim_idx]
    let elements_per_batch = num_heads * head_dim;
    let batch_idx = total_idx / elements_per_batch;
    let remainder = total_idx % elements_per_batch;
    let head_idx = remainder / head_dim;
    let dim_idx = remainder % head_dim;
    
    // Early exit if beyond bounds
    if (batch_idx >= batch_size || head_idx >= num_heads || dim_idx >= head_dim) {
        return;
    }
    
    let seq_len = params_v.seq_len;
    let num_kv_heads = params_v.num_kv_heads;
    let max_seq_len = params_v.max_seq_len;
    
    // Calculate batch offset in KV cache
    let batch_cache_size = params_v.num_layers * max_seq_len * num_kv_heads * head_dim;
    let batch_offset = batch_idx * batch_cache_size;
    
    // Calculate layer offset within this batch's cache
    let layer_size = max_seq_len * num_kv_heads * head_dim;
    let layer_offset = params_v.layer_idx * layer_size;
    
    // For GQA: map query head to corresponding KV head
    let kv_head_idx = (head_idx * num_kv_heads) / num_heads;
    
    // Compute weighted sum: sum over all positions (attention_prob[pos] * value[pos][dim])
    var weighted_sum: f32 = 0.0;
    
    for (var pos = 0u; pos < seq_len; pos++) {
        // Probability index: [batch_idx][head_idx][pos]
        // CRITICAL: Use max_seq_len for stride, not seq_len, since buffer is sized [batch, heads, max_seq_len]
        let prob_idx = batch_idx * num_heads * max_seq_len + head_idx * max_seq_len + pos;
        
        // Value index: [batch_idx][layer_idx][pos][kv_head_idx][dim_idx]
        let value_idx = batch_offset + layer_offset + pos * num_kv_heads * head_dim + kv_head_idx * head_dim + dim_idx;
        
        weighted_sum += attention_probs[prob_idx] * values[value_idx];
    }
    
    // Store output at [batch_idx][head_idx][dim_idx]
    let out_idx = batch_idx * num_heads * head_dim + head_idx * head_dim + dim_idx;
    output[out_idx] = weighted_sum;
}

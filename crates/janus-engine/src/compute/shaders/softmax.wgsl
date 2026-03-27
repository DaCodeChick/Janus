// Numerically stable softmax computation
//
// Softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
//
// This shader implements a two-pass algorithm:
// Pass 1: Find max value in the sequence (for numerical stability)
// Pass 2: Compute exp(x - max) and sum
// Pass 3: Normalize by dividing by sum
//
// Layout: Each workgroup processes one row (one query position's attention scores)

struct Params {
    seq_len: u32,        // Length of the sequence to apply softmax over (current sequence length)
    num_heads: u32,      // Number of attention heads
    batch_size: u32,     // Number of rows to process
    max_seq_len: u32,    // Maximum sequence length (buffer stride)
}

@group(0) @binding(0) var<storage, read> input: array<f32>;      // Input scores [batch_size][seq_len]
@group(0) @binding(1) var<storage, read_write> output: array<f32>; // Output probabilities [batch_size][seq_len]
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let row_idx = workgroup_id.x;  // Which row (query position) we're processing
    let thread_idx = local_id.x;
    let seq_len = params.seq_len;
    
    let is_active = row_idx < params.batch_size;
    
    // Pass 1: Find maximum value in this row for numerical stability
    var local_max: f32 = -1e10;
    
    // Each thread processes multiple elements if seq_len > 256
    if (is_active) {
        for (var i = thread_idx; i < seq_len; i += 256u) {
            // CRITICAL: Use max_seq_len for stride since buffer is [batch, heads, max_seq_len]
            let idx = row_idx * params.max_seq_len + i;
            local_max = max(local_max, input[idx]);
        }
    }
    
    // Store local max in shared memory
    shared_data[thread_idx] = local_max;
    workgroupBarrier();
    
    // Parallel reduction to find global max (binary tree reduction)
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (thread_idx < stride) {
            shared_data[thread_idx] = max(shared_data[thread_idx], shared_data[thread_idx + stride]);
        }
        workgroupBarrier();
    }
    
    let max_val = shared_data[0];
    
    // Pass 2: Compute exp(x - max) and accumulate sum
    var local_sum: f32 = 0.0;
    
    if (is_active) {
        for (var i = thread_idx; i < seq_len; i += 256u) {
            // CRITICAL: Use max_seq_len for stride since buffer is [batch, heads, max_seq_len]
            let idx = row_idx * params.max_seq_len + i;
            let exp_val = exp(input[idx] - max_val);
            output[idx] = exp_val;  // Store intermediate exp values
            local_sum += exp_val;
        }
    }
    
    // Store local sum in shared memory
    shared_data[thread_idx] = local_sum;
    workgroupBarrier();
    
    // Parallel reduction to find total sum (binary tree reduction)
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (thread_idx < stride) {
            shared_data[thread_idx] += shared_data[thread_idx + stride];
        }
        workgroupBarrier();
    }
    
    let sum_val = shared_data[0];
    
    // Pass 3: Normalize by dividing by sum
    if (is_active) {
        for (var i = thread_idx; i < seq_len; i += 256u) {
            // CRITICAL: Use max_seq_len for stride since buffer is [batch, heads, max_seq_len]
            let idx = row_idx * params.max_seq_len + i;
            output[idx] = output[idx] / sum_val;
        }
    }
}

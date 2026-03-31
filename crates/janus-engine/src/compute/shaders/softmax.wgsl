// Numerically stable softmax computation
//
// Softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
//
// This implementation uses one invocation per row for strict numerical stability.

struct Params {
    seq_len: u32,
    num_heads: u32,
    batch_size: u32,
    max_seq_len: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;       // [rows][max_seq_len]
@group(0) @binding(1) var<storage, read_write> output: array<f32>; // [rows][max_seq_len]
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row_idx = global_id.x;
    if (row_idx >= params.batch_size) {
        return;
    }

    let base_idx = row_idx * params.max_seq_len;

    // 1. Find the maximum score in the sequence to prevent exp() overflow
    var max_val: f32 = -999999.0;
    for (var i = 0u; i < params.seq_len; i++) {
        let val = input[base_idx + i];
        if (val > max_val) {
            max_val = val;
        }
    }

    // 2. Subtract the max score before calculating exp()
    var sum_exp: f32 = 0.0;
    for (var i = 0u; i < params.seq_len; i++) {
        let e = exp(input[base_idx + i] - max_val);
        output[base_idx + i] = e;
        sum_exp = sum_exp + e;
    }

    let safe_sum = max(sum_exp, 1e-20);

    // 3. Normalize the probabilities
    for (var i = 0u; i < params.seq_len; i++) {
        output[base_idx + i] = output[base_idx + i] / safe_sum;
    }
}
